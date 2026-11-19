#!/usr/bin/env python
"""Benchmark distributed mefikit against medcoupling InterpKernelDEC on TET4.

The layout is the one used by InterpKernelDEC::

    n source ranks + p target ranks, with n = p
    size = n + p

Source and target meshes are independent, shifted, tetrahedralized structured
hexahedral boxes. Every rank owns one local piece: source ranks own source
cells, target ranks own target cells.

Run with (the process count must be even)::

    mpirun -n 4 uv run python tests/bench_mpi_tet_vs_medcoupling.py
    mpirun -n 6 uv run python tests/bench_mpi_tet_vs_medcoupling.py \
        --n-src 16 --n-tgt 12 --iters 10

Preparation time is the ``DistributedConservativeP0`` constructor for mefikit
and ``InterpKernelDEC.synchronize()`` for medcoupling. Application time is the
MPI transfer of one intensive field. A build of medcoupling without MPI does
not expose ``InterpKernelDEC``; in that case only the mefikit side is run and
the script reports that the comparison is unavailable.
"""

from __future__ import annotations

import argparse
import time

import medcoupling as mc
import numpy as np
from mpi4py import MPI

import mefikit as mf

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()


def build_tet_box(n: int, lo: float, hi: float) -> mf.UMesh:
    """Structured hexahedral box, tetrahedralized into TET4 cells."""
    mesh = mf.build_cmesh(*(3 * [np.linspace(lo, hi, n + 1)])).simplexize()
    assert mesh.block_types() == ["TET4"], mesh.block_types()
    return mesh


def partition(mesh: mf.UMesh, lo: int, hi: int) -> mf.UMesh | None:
    """Extract a contiguous local piece of TET4 cells."""
    if hi <= lo:
        return None
    ids = np.arange(lo, hi, dtype=np.uintp)
    return mesh.select({"TET4": ids}).to_mesh(False).remove_unused_coords()


def contiguous_range(n: int, r: int, n_ranks: int) -> tuple[int, int]:
    return r * n // n_ranks, (r + 1) * n // n_ranks


def mc_mesh(mesh: mf.UMesh) -> mc.MEDCouplingUMesh:
    result = mesh.to_mc()
    result.setMeshDimension(3)
    return result


def mc_field(
    mesh: mc.MEDCouplingUMesh, values: np.ndarray
) -> mc.MEDCouplingFieldDouble:
    field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field.setMesh(mesh)
    field.setNature(mc.IntensiveConservation)
    field.setArray(mc.DataArrayDouble(np.ascontiguousarray(values.reshape(-1, 1))))
    return field


def collect_target_values(values: np.ndarray, is_target: bool) -> np.ndarray | None:
    """Gather target-rank arrays on rank 0, in target-rank order."""
    gathered = COMM.gather(values if is_target else None, root=0)
    if RANK != 0:
        return None
    chunks = [chunk for chunk in gathered if chunk is not None and chunk.size]
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-src", type=int, default=12)
    parser.add_argument("--n-tgt", type=int, default=10)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if SIZE < 2 or SIZE % 2:
        if RANK == 0:
            raise SystemExit(
                "this benchmark needs an even number of ranks: n = p = size / 2"
            )
        return

    n_src_ranks = SIZE // 2
    n_tgt_ranks = SIZE // 2
    is_source = RANK < n_src_ranks
    is_target = not is_source
    target_rank = RANK - n_src_ranks

    src_local = None
    src_values = None
    if is_source:
        src_full = build_tet_box(args.n_src, 0.0, 1.0)
        src_values = (np.arange(src_full.num_elements(), dtype=float) + 1.0).reshape(
            -1, 1
        )
        src_full.set_field("T", {"TET4": src_values})
        s_lo, s_hi = contiguous_range(src_full.num_elements(), RANK, n_src_ranks)
        src_local = partition(src_full, s_lo, s_hi)
        if src_local is not None:
            src_local.set_field(
                "T", {"TET4": np.ascontiguousarray(src_values[s_lo:s_hi])}
            )

    tgt_local = None
    if is_target:
        tgt_full = build_tet_box(args.n_tgt, 0.03, 0.97)
        t_lo, t_hi = contiguous_range(tgt_full.num_elements(), target_rank, n_tgt_ranks)
        tgt_local = partition(tgt_full, t_lo, t_hi)

    n_src_total = COMM.allreduce(
        0 if src_local is None else src_local.num_elements(), op=MPI.SUM
    )
    n_tgt_total = COMM.allreduce(
        0 if tgt_local is None else tgt_local.num_elements(), op=MPI.SUM
    )

    # --- mefikit distributed transfer -------------------------------------
    COMM.Barrier()
    t0 = time.perf_counter()
    op = mf.transfer.DistributedConservativeP0(src_local, tgt_local, comm=COMM)
    COMM.Barrier()
    t1 = time.perf_counter()
    mf_prepare = t1 - t0

    mf_apply_times = []
    mf_local: dict[str, np.ndarray] = {}
    for _ in range(args.iters):
        COMM.Barrier()
        a0 = time.perf_counter()
        mf_local = op.eval("T")
        COMM.Barrier()
        a1 = time.perf_counter()
        if RANK == 0:
            mf_apply_times.append(a1 - a0)
    mf_apply = float(np.median(mf_apply_times)) if RANK == 0 else 0.0
    mf_values = (
        np.asarray(mf_local["TET4"]).reshape(-1, 1)
        if "TET4" in mf_local
        else np.empty((0, 1))
    )
    mf_gathered = collect_target_values(mf_values, is_target)

    # --- medcoupling InterpKernelDEC --------------------------------------
    mc_available = (
        hasattr(mc, "InterpKernelDEC")
        and hasattr(mc, "MPIProcessorGroup")
        and hasattr(mc, "CommInterface")
    )
    mc_prepare = None
    mc_apply = None
    mc_gathered = None
    if mc_available:
        if src_local is None or tgt_local is None:
            raise RuntimeError(
                "the benchmark needs at least one source cell and one target "
                "cell per participating rank; reduce the number of ranks or "
                "increase the mesh size"
            )

        interface = mc.CommInterface()
        source_group = mc.MPIProcessorGroup(interface, list(range(n_src_ranks)))
        target_group = mc.MPIProcessorGroup(interface, list(range(n_src_ranks, SIZE)))
        dec = mc.InterpKernelDEC(source_group, target_group)

        if is_source:
            source_mc = mc_mesh(src_local)
            local_values = np.asarray(src_local.fields["T"].numpy()).reshape(-1, 1)
            source_field = mc_field(source_mc, local_values)
            dec.attachLocalField(source_field)
        else:
            target_mc = mc_mesh(tgt_local)
            local_n = tgt_local.num_elements()
            target_field = mc_field(target_mc, np.zeros((local_n, 1)))
            dec.attachLocalField(target_field)

        COMM.Barrier()
        c0 = time.perf_counter()
        dec.synchronize()
        COMM.Barrier()
        c1 = time.perf_counter()
        mc_prepare = c1 - c0

        mc_apply_times = []
        for _ in range(args.iters):
            COMM.Barrier()
            b0 = time.perf_counter()
            if is_source:
                dec.sendData()
            else:
                dec.recvData()
            COMM.Barrier()
            b1 = time.perf_counter()
            if RANK == 0:
                mc_apply_times.append(b1 - b0)
        mc_apply = float(np.median(mc_apply_times)) if RANK == 0 else 0.0

        if is_target and tgt_local is not None:
            mc_values = target_field.getArray().toNumPyArray().reshape(-1, 1)
        else:
            mc_values = np.empty((0, 1))
        mc_gathered = collect_target_values(mc_values, is_target)

        dec.release()
        source_group.release()
        target_group.release()

    if RANK != 0:
        return

    print(f"source: {args.n_src}^3 HEX8 -> {n_src_total} TET4 on {n_src_ranks} ranks")
    print(f"target: {args.n_tgt}^3 HEX8 -> {n_tgt_total} TET4 on {n_tgt_ranks} ranks")
    print(f"n = p = {n_src_ranks}; size = n + p = {SIZE}")
    print()
    print(f"{'case':32s} {'prepare [s]':>12s} {'apply [s]':>12s}")
    print(f"{'mefikit distributed':32s} {mf_prepare:12.6f} {mf_apply:12.6f}")
    if mc_available:
        print(
            f"{'medcoupling InterpKernelDEC':32s} {mc_prepare:12.6f} {mc_apply:12.6f}"
        )
    else:
        print(
            "medcoupling InterpKernelDEC unavailable in this build; "
            "distributed comparison skipped."
        )
    print()

    if mc_available and mc_gathered is not None and mf_gathered is not None:
        np.testing.assert_allclose(mf_gathered, mc_gathered, rtol=1e-10, atol=1e-12)
        print(
            "max |mefikit - medcoupling| = "
            f"{np.max(np.abs(mf_gathered - mc_gathered)):.3e}"
        )


if __name__ == "__main__":
    main()
