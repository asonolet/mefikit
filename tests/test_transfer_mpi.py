from __future__ import annotations

import numpy as np
import pytest

import mefikit as mf

mpi = pytest.importorskip("mpi4py.MPI")
comm = mpi.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def partition(mesh: mf.UMesh, et: str, lo: int, hi: int) -> mf.UMesh | None:
    """Extract cells ``[lo, hi)`` from ``mesh`` as a compact local piece."""
    if hi <= lo:
        return None
    ids = np.arange(lo, hi, dtype=np.uintp)
    return mesh.select({et: ids}).to_mesh(with_fields=False).remove_unused_coords()


def contiguous_range(n: int, r: int, n_ranks: int) -> tuple[int, int]:
    return r * n // n_ranks, (r + 1) * n // n_ranks


def reference_from_rank0(
    reference: np.ndarray,
    lo: int,
    hi: int,
    tag: int,
) -> np.ndarray:
    """Slice a rank-0 reference field and broadcast the local slice."""
    if rank == 0:
        local = reference[lo:hi]
        for dest in range(1, size):
            d_lo, d_hi = contiguous_range(reference.shape[0], dest, size)
            comm.send(reference[d_lo:d_hi], dest=dest, tag=tag)
        return local
    return comm.recv(source=0, tag=tag)


def local_slice(field: dict[str, np.ndarray], et: str) -> np.ndarray:
    if et not in field:
        return np.empty((0, 1), dtype=np.float64)
    return np.asarray(field[et]).reshape(field[et].shape[0], -1)


def test_distributed_matches_sequential_2d() -> None:
    """Distributed P0 transfer matches the sequential operator on every rank."""
    xs = np.linspace(0.0, 1.0, 8)
    src_full = mf.build_cmesh(xs, xs)
    tgt_full = mf.build_cmesh(xs + 0.013, xs[1:] - 0.007)
    values = np.stack(
        [
            np.arange(1, src_full.num_elements() + 1, dtype=float),
            np.arange(1, src_full.num_elements() + 1, dtype=float) * 3.0,
        ],
        axis=1,
    )
    src_full.set_field("T", {"QUAD4": values})

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "QUAD4", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"QUAD4": np.ascontiguousarray(values[s_lo:s_hi])})

    op = mf.transfer.DistributedConservativeP0(src, tgt, def_val=np.nan)
    got = local_slice(op.eval("T"), "QUAD4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full, def_val=np.nan).eval("T")[
                "QUAD4"
            ]
        ).reshape(tgt_full.num_elements(), -1)
    expected_local = reference_from_rank0(reference, t_lo, t_hi, tag=100)

    np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_extensive_and_reuse() -> None:
    """The prepared operator is reusable and honours the extensive flag."""
    xs = np.linspace(0.0, 1.0, 6)
    src_full = mf.build_cmesh(xs, xs)
    tgt_full = mf.build_cmesh(xs + 0.02, xs[1:] - 0.01)
    f1 = np.arange(1, src_full.num_elements() + 1, dtype=float).reshape(-1, 1)
    f2 = (np.arange(1, src_full.num_elements() + 1, dtype=float) * 0.5).reshape(-1, 1)
    src_full.set_field("F1", {"QUAD4": f1})
    src_full.set_field("F2", {"QUAD4": f2})

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "QUAD4", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("F1", {"QUAD4": np.ascontiguousarray(f1[s_lo:s_hi])})
        src.set_field("F2", {"QUAD4": np.ascontiguousarray(f2[s_lo:s_hi])})

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got_ext = local_slice(op.eval("F1", extensive=True), "QUAD4")
    op.apply_update(src, "F2", tgt, "R", extensive=True)
    got_reuse = (
        np.asarray(tgt.fields["R"].numpy()).reshape(tgt.num_elements(), -1)
        if tgt is not None
        else np.empty((0, 1))
    )

    reference_ext = None
    reference_reuse = None
    if rank == 0:
        seq = mf.transfer.ConservativeP0(src_full, tgt_full)
        reference_ext = np.asarray(seq.eval("F1", extensive=True)["QUAD4"]).reshape(
            tgt_full.num_elements(), -1
        )
        reference_reuse = np.asarray(seq.eval("F2", extensive=True)["QUAD4"]).reshape(
            tgt_full.num_elements(), -1
        )
    expected_ext = reference_from_rank0(reference_ext, t_lo, t_hi, tag=101)
    expected_reuse = reference_from_rank0(reference_reuse, t_lo, t_hi, tag=102)

    np.testing.assert_allclose(got_ext, expected_ext, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got_reuse, expected_reuse, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_poly_source() -> None:
    """Variable-size (polygon) source cells are exchanged and mapped correctly."""
    xs = np.linspace(0.0, 1.0, 5)
    src_full = mf.build_cmesh(xs, xs).polyze()
    values = (np.arange(1, src_full.num_elements() + 1, dtype=float)).reshape(-1, 1)
    src_full.set_field("T", {"PGON": values})
    tgt_full = mf.build_cmesh(xs + 0.01, xs[1:] - 0.02)

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "PGON", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"PGON": np.ascontiguousarray(values[s_lo:s_hi])})

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got = local_slice(op.eval("T"), "QUAD4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["QUAD4"]
        ).reshape(tgt_full.num_elements(), -1)
    expected_local = reference_from_rank0(reference, t_lo, t_hi, tag=106)

    np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_downcast_3d_to_2d() -> None:
    """A distributed 3D -> 2D downcast reproduces the sequential result."""
    xs = np.linspace(0.0, 1.0, 5)
    src_full = mf.build_cmesh(xs, xs, xs)
    values = np.arange(1, src_full.num_elements() + 1, dtype=float).reshape(-1, 1)
    src_full.set_field("T", {"HEX8": values})

    tgt_full = surface_grid_3d(
        np.linspace(0.03, 0.97, 5),
        np.linspace(0.05, 0.95, 4),
        0.4,
    )
    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "HEX8", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"HEX8": np.ascontiguousarray(values[s_lo:s_hi])})

    op = mf.transfer.DistributedConservativeP0(src, tgt, def_val=-1.0)
    got = local_slice(op.eval("T"), "QUAD4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full, def_val=-1.0).eval("T")[
                "QUAD4"
            ]
        ).reshape(tgt_full.num_elements(), -1)
    expected_local = reference_from_rank0(reference, t_lo, t_hi, tag=103)

    np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_empty_local_pieces() -> None:
    """Ranks with no local source or target cell still participate."""
    src_full = mf.build_cmesh(np.array([0.0, 1.0]), np.array([0.0, 1.0]))
    src_full.set_field("T", {"QUAD4": np.array([[3.0]])})
    tgt_full = mf.build_cmesh(np.array([0.0, 0.5, 1.0]), np.array([0.0, 1.0]))

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "QUAD4", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"QUAD4": np.array([[3.0]])})

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got = local_slice(op.eval("T"), "QUAD4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["QUAD4"]
        ).reshape(tgt_full.num_elements(), -1)
    expected_local = reference_from_rank0(reference, t_lo, t_hi, tag=104)

    np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


@pytest.mark.skipif(size < 2, reason="needs at least one source and one target rank")
def test_distributed_disjoint_source_and_target_ranks() -> None:
    """Source and target ranks may be disjoint, as in InterpKernelDEC."""
    n_src = max(1, size // 2)
    is_src = rank < n_src
    is_tgt = not is_src

    xs = np.linspace(0.0, 1.0, 6)
    src_full = mf.build_cmesh(xs, xs)
    tgt_full = mf.build_cmesh(xs + 0.013, xs[1:] - 0.01)
    values = np.arange(1, src_full.num_elements() + 1, dtype=float).reshape(-1, 1)
    src_full.set_field("T", {"QUAD4": values})

    src = None
    s_lo = s_hi = 0
    if is_src:
        s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, n_src)
        src = partition(src_full, "QUAD4", s_lo, s_hi)
        assert src is not None
        src.set_field("T", {"QUAD4": np.ascontiguousarray(values[s_lo:s_hi])})

    tgt = None
    t_lo = t_hi = 0
    n_tgt = size - n_src
    if is_tgt:
        t = rank - n_src
        t_lo, t_hi = contiguous_range(tgt_full.num_elements(), t, n_tgt)
        tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got = local_slice(op.eval("T"), "QUAD4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["QUAD4"]
        ).reshape(tgt_full.num_elements(), -1)

    if is_tgt:
        expected_local = comm.recv(source=0, tag=105)
        np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    elif rank == 0:
        for t in range(n_tgt):
            dest = n_src + t
            lo, hi = contiguous_range(tgt_full.num_elements(), t, n_tgt)
            comm.send(reference[lo:hi], dest=dest, tag=105)
    comm.Barrier()


def test_distributed_tetra_matches_sequential() -> None:
    """Shifted tetrahedralized hexahedra match the sequential P0 transfer."""
    src_full = mf.build_cmesh(
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
    ).simplexize()
    tgt_full = mf.build_cmesh(
        np.linspace(0.03, 0.97, 5),
        np.linspace(0.03, 0.97, 5),
        np.linspace(0.03, 0.97, 5),
    ).simplexize()
    assert src_full.block_types() == ["TET4"]
    assert tgt_full.block_types() == ["TET4"]

    values = (np.arange(1, src_full.num_elements() + 1, dtype=float)).reshape(-1, 1)
    src_full.set_field("T", {"TET4": values})

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "TET4", s_lo, s_hi)
    tgt = partition(tgt_full, "TET4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"TET4": np.ascontiguousarray(values[s_lo:s_hi])})

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got = local_slice(op.eval("T"), "TET4")

    reference = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["TET4"]
        ).reshape(tgt_full.num_elements(), -1)
    expected_local = reference_from_rank0(reference, t_lo, t_hi, tag=107)

    np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


@pytest.mark.skipif(
    size < 2 or size % 2,
    reason="needs n = p source/target ranks with size = n + p",
)
def test_distributed_tetra_disjoint_ranks() -> None:
    """TET4 transfer with n = p source/target ranks, size = n + p."""
    n_src = size // 2
    n_tgt = size // 2
    is_src = rank < n_src
    is_tgt = not is_src
    target_rank = rank - n_src

    src_full = mf.build_cmesh(
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
        np.linspace(0.0, 1.0, 6),
    ).simplexize()
    tgt_full = mf.build_cmesh(
        np.linspace(0.03, 0.97, 5),
        np.linspace(0.03, 0.97, 5),
        np.linspace(0.03, 0.97, 5),
    ).simplexize()
    values = (np.arange(1, src_full.num_elements() + 1, dtype=float)).reshape(-1, 1)
    src_full.set_field("T", {"TET4": values})

    src = None
    if is_src:
        s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, n_src)
        src = partition(src_full, "TET4", s_lo, s_hi)
        assert src is not None
        src.set_field("T", {"TET4": np.ascontiguousarray(values[s_lo:s_hi])})

    tgt = None
    t_lo = t_hi = 0
    if is_tgt:
        t_lo, t_hi = contiguous_range(tgt_full.num_elements(), target_rank, n_tgt)
        tgt = partition(tgt_full, "TET4", t_lo, t_hi)

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    got = local_slice(op.eval("T"), "TET4")

    expected_local = None
    if rank == 0:
        reference = np.asarray(
            mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["TET4"]
        ).reshape(tgt_full.num_elements(), -1)
        for t in range(n_tgt):
            lo, hi = contiguous_range(tgt_full.num_elements(), t, n_tgt)
            comm.send(reference[lo:hi], dest=n_src + t, tag=108)
    elif is_tgt:
        expected_local = comm.recv(source=0, tag=108)

    if is_tgt:
        assert expected_local is not None
        np.testing.assert_allclose(got, expected_local, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_one_shot() -> None:
    """The one-shot ``mf.transfer.distributed`` helper matches the class API."""
    xs = np.linspace(0.0, 1.0, 7)
    src_full = mf.build_cmesh(xs, xs)
    tgt_full = mf.build_cmesh(xs + 0.01, xs[1:] - 0.02)
    values = (np.arange(1, src_full.num_elements() + 1, dtype=float)).reshape(-1, 1)
    src_full.set_field("T", {"QUAD4": values})

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "QUAD4", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"QUAD4": np.ascontiguousarray(values[s_lo:s_hi])})

    got = local_slice(
        mf.transfer.distributed(src, "T", tgt, comm=comm, def_val=-2.0), "QUAD4"
    )
    op = mf.transfer.DistributedConservativeP0(src, tgt, comm=comm, def_val=-2.0)
    expected = local_slice(op.eval("T"), "QUAD4")

    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_extensive_mass_conservation() -> None:
    """The distributed extensive transfer conserves the global integral."""
    src_full = mf.build_cmesh(
        np.linspace(0.0, 1.0, 5),
        np.linspace(0.0, 1.0, 5),
    )
    tgt_full = mf.build_cmesh(
        np.linspace(0.0, 1.0, 7),
        np.linspace(0.0, 1.0, 4),
    )
    src_full.set_field(
        "T", {"QUAD4": np.ones((src_full.num_elements(), 1), dtype=float)}
    )

    s_lo, s_hi = contiguous_range(src_full.num_elements(), rank, size)
    t_lo, t_hi = contiguous_range(tgt_full.num_elements(), rank, size)
    src = partition(src_full, "QUAD4", s_lo, s_hi)
    tgt = partition(tgt_full, "QUAD4", t_lo, t_hi)
    if src is not None:
        src.set_field("T", {"QUAD4": np.ones((s_hi - s_lo, 1), dtype=float)})

    op = mf.transfer.DistributedConservativeP0(src, tgt)
    local = local_slice(op.eval("T", extensive=True), "QUAD4")
    total = comm.allreduce(float(local.sum()), op=mpi.SUM)

    np.testing.assert_allclose(total, 1.0, rtol=1e-12, atol=1e-12)
    comm.Barrier()


def test_distributed_rejects_higher_order_cells() -> None:
    """Quadratic/higher-order cells are best effort and not supported distributed."""
    mesh = mf.UMesh(
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.0],
                [1.0, 0.5],
                [0.5, 1.0],
                [0.0, 0.5],
                [0.5, 0.5],
            ]
        )
    )
    mesh.add_regular_block("QUAD9", np.arange(9, dtype=np.uintp).reshape(1, 9))

    with pytest.raises(ValueError, match="first-order"):
        mf.transfer.DistributedConservativeP0(mesh, None)


def surface_grid_3d(xs, ys, z) -> mf.UMesh:
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    coords = np.ascontiguousarray(
        np.c_[x.ravel(), y.ravel(), np.full(x.size, float(z))]
    )
    conn = []
    ny = len(ys)
    for i in range(len(xs) - 1):
        for j in range(ny - 1):
            n0 = i * ny + j
            n1 = (i + 1) * ny + j
            n2 = (i + 1) * ny + j + 1
            n3 = i * ny + j + 1
            conn.append([n0, n1, n2, n3])
    mesh = mf.UMesh(coords)
    mesh.add_regular_block(
        "QUAD4", np.ascontiguousarray(np.array(conn), dtype=np.uintp)
    )
    return mesh
