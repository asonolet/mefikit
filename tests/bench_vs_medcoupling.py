#!/usr/bin/env python
"""Benchmark mefikit against medcoupling on equivalent mesh operations.

Run with:  uv run python tests/bench_vs_medcoupling.py

Covered operations
------------------
* remap-2d    : P0 conservative transfer on QUAD4 meshes (build/prepare + transfer)
* remap-3d    : same on HEX8 meshes
* remap-3d-poly: same with the target converted to polyhedra (polyze / convertAllToPoly)
* merge-nodes : collapse coincident nodes on a duplicated-interface structured stack
* descend     : build the (dim-1) descending mesh (faces) of a hexa grid
* overlay     : 2D overlay/imprint of a fine grid with an embedded coarse grid

Each step is timed with the median of ``N_ITER`` runs; medcoupling meshes are built
from the very same mefikit meshes via ``UMesh.to_mc()`` so both sides operate on the
identical geometry. A lightweight correctness cross-check is printed at the end.

medcoupling nature enum used for the remapping:
    IntensiveConservation = 37, ExtensiveConservation = 35
"""

from __future__ import annotations

import time

import medcoupling as mc
import numpy as np

import mefikit as mf

N_ITER = 10

MC_INTENSIVE = 37
MC_EXTENSIVE = 35

# --- workloads -------------------------------------------------------------
N2D = 96  # 96x96 = 9216 QUAD4 cells
N3D = 16  # 16^3  = 4096 HEX8 cells
NPOLY = 16  # poly remap target, 16^3 source
MERGE_N = 24  # 2 stacked 24x24 HEX8 layers with a duplicated interface
DESCEND_N = 24  # 24^3 hexa grid -> faces
OVERLAY_N = 32  # 32x32 grid overlayed by an embedded 8x8 block

RTOL = 1e-9
ATOL = 1e-9


def median_time(fn, n=N_ITER) -> float:
    """Median wall time of ``fn`` over ``n`` runs, in milliseconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def mc_mesh(mesh: mf.UMesh, dim: int) -> mc.MEDCouplingUMesh:
    """The medcoupling twin of a mefikit mesh (to_mc, mesh dimension set)."""
    m = mesh.to_mc()
    m.setMeshDimension(dim)
    return m


def mc_field(mmesh: mc.MEDCouplingUMesh, vals: np.ndarray, nature: int, name="T"):
    f = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    a = mc.DataArrayDouble(np.ascontiguousarray(vals.ravel()))
    a.setName(name)
    f.setArray(a)
    f.setMesh(mmesh)
    f.setNature(nature)
    return f


def used_nodes(mesh: mf.UMesh) -> int:
    ids = np.concatenate([np.asarray(b) for b in mesh.blocks().values()])
    return int(np.unique(ids).size)


def area_2d(mesh: mf.UMesh) -> float:
    """Total 2D area from mefikit's own per-cell measures."""
    return float(sum(np.asarray(v).sum() for v in mesh.measure().values()))


def field_2d(n: int) -> np.ndarray:
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    xc, yc = (i + 0.5) / n, (j + 0.5) / n
    return (1.0 + 0.5 * np.sin(2 * np.pi * xc) * np.cos(np.pi * yc)).reshape(-1, 1)


def field_3d(n: int) -> np.ndarray:
    i, j, k = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    xc, yc, zc = (i + 0.5) / n, (j + 0.5) / n, (k + 0.5) / n
    return (
        1.0 + 0.5 * np.sin(2 * np.pi * xc) * np.cos(np.pi * yc) * np.cos(np.pi * zc)
    ).reshape(-1, 1)


def dump_merged_mesh(n: int) -> mf.UMesh:
    """Two stacked HEX8 layers; the shared interface is duplicated (2 node sets)."""
    gx = np.linspace(0.0, 1.0, n + 1)
    px, py = np.meshgrid(gx, gx, indexing="ij")
    z0 = np.c_[px.ravel(), py.ravel(), np.zeros((n + 1) ** 2)]
    z1 = np.c_[px.ravel(), py.ravel(), np.ones((n + 1) ** 2)]
    coords = np.ascontiguousarray(np.vstack([z0, z1, z1, z0 + 2.0]), np.float64)

    nid = lambda i, j, l: l * (n + 1) ** 2 + i * (n + 1) + j
    conn = []
    for i in range(n):
        for j in range(n):
            conn.append(
                [
                    nid(i, j, 0),
                    nid(i + 1, j, 0),
                    nid(i + 1, j + 1, 0),
                    nid(i, j + 1, 0),
                    nid(i, j, 1),
                    nid(i + 1, j, 1),
                    nid(i + 1, j + 1, 1),
                    nid(i, j + 1, 1),
                ]
            )
    for i in range(n):
        for j in range(n):
            conn.append(
                [
                    nid(i, j, 2),
                    nid(i + 1, j, 2),
                    nid(i + 1, j + 1, 2),
                    nid(i, j + 1, 2),
                    nid(i, j, 3),
                    nid(i + 1, j, 3),
                    nid(i + 1, j + 1, 3),
                    nid(i, j + 1, 3),
                ]
            )
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("HEX8", np.ascontiguousarray(np.array(conn), np.uintp))
    return mesh


# --- remap timing + validation ---------------------------------------------
def bench_remap(dim: int, n: int, build_iter, poly: bool = False):
    """Returns (mf_build, mc_prepare, mf_apply, mc_transfer, checks)."""
    x = np.linspace(0.0, 1.0, n + 1)
    axes = [x] * dim
    shift = [0.5 / n] + [0.0] * (dim - 1)
    et = "QUAD4" if dim == 2 else "HEX8"
    vals = field_2d(n) if dim == 2 else field_3d(n)

    src = mf.build_cmesh(*axes)
    tgt = mf.build_cmesh(*[a + s for a, s in zip(axes, shift)])
    if poly:
        tgt = tgt.polyze()

    sm = mc_mesh(src, dim)
    tm = mc_mesh(mf.build_cmesh(*[a + s for a, s in zip(axes, shift)]), dim)
    if poly:
        tm.convertAllToPoly()

    # --- timing: build / prepare ---
    def mf_build():
        mf.ConservativeP0(src, tgt)

    vt = mc.MEDCouplingRemapper()

    def mc_prepare():
        vt.prepare(sm, tm, "P0P0")

    t_mf_build = median_time(mf_build, build_iter)
    t_mc_prepare = median_time(mc_prepare, build_iter)

    # --- timing: transfer ---
    src.set_field("T", {et: np.ascontiguousarray(vals)})
    op = mf.ConservativeP0(src, tgt)

    def mf_apply():
        op.apply_update(src, "T", tgt, "T", def_val=0.0)

    field = mc_field(sm, vals, MC_INTENSIVE)
    vt.prepare(sm, tm, "P0P0")

    def mc_transfer():
        vt.transferField(field, 0.0)

    t_mf_apply = median_time(mf_apply)
    t_mc_transfer = median_time(mc_transfer)

    # --- checks ---
    checks = {}

    def read_mf():
        parts = [np.asarray(v).ravel() for v in tgt.fields["T"].values().values()]
        return np.concatenate(parts)

    op.apply_update(src, "T", tgt, "T", def_val=0.0)
    out_mf = read_mf()
    out_mc = np.asarray(vt.transferField(field, 0.0).getArray().getValues())

    if not poly:
        diff = np.max(np.abs(out_mf - out_mc))
        checks["intensive match (mefi==mc)"] = (
            np.allclose(out_mf, out_mc, rtol=RTOL, atol=ATOL),
            diff,
        )

    # Extensive semantics differ between the two libraries:
    #   mefikit extensive=True  -> target cell integrals (mass): out_i = int_tgt_i f
    #   medcoupling EXTENSIVE   -> conservative weights that preserve the input sum
    # To compare both on the same footing, run on a *coincident* grid (full coverage)
    # and feed medcoupling the per-cell source integrals f*vol so that both totals are
    # the total mass  sum(f)*vol. Both must equal the analytic total mass.
    vol = 1.0 / n**dim
    mass_analytic = float(vals.sum() * vol)
    cs = mf.build_cmesh(*axes)
    ct = cs.polyze() if poly else mf.build_cmesh(*axes)
    cs.set_field("T", {et: np.ascontiguousarray(vals)})
    opc = mf.ConservativeP0(cs, ct)
    opc.apply_update(cs, "T", ct, "T", def_val=0.0, extensive=True)
    parts = [np.asarray(v).ravel() for v in ct.fields["T"].values().values()]
    mass_mf = float(np.concatenate(parts).sum())
    csm = mc_mesh(cs, dim)
    ctm = mc_mesh(mf.build_cmesh(*axes), dim)
    if poly:
        ctm.convertAllToPoly()
    mass_mc = float(
        np.asarray(
            vt.transferField(mc_field(csm, vals * vol, MC_EXTENSIVE), 0.0)
            .getArray()
            .getValues()
        ).sum()
    )
    tol = max(1e-9, mass_analytic * 1e-9)
    checks["mass (mefi == analytic)"] = (
        abs(mass_mf - mass_analytic) <= tol,
        (round(mass_analytic, 10), round(mass_mf, 10)),
    )
    checks["mass (mc == mefi)"] = (
        abs(mass_mc - mass_mf) <= tol,
        (round(mass_mf, 10), round(mass_mc, 10)),
    )

    return t_mf_build, t_mc_prepare, t_mf_apply, t_mc_transfer, checks


def bench_remap_2d():
    return bench_remap(2, N2D, build_iter=10, poly=False)


def bench_remap_3d():
    return bench_remap(3, N3D, build_iter=10, poly=False)


def bench_remap_3d_poly():
    return bench_remap(3, NPOLY, build_iter=5, poly=True)


def bench_merge():
    mesh = dump_merged_mesh(MERGE_N)
    mm = mesh.to_mc()
    t_mf = median_time(lambda: mesh.merge_nodes(1e-12))
    t_mc = median_time(lambda: mm.mergeNodes(1e-12))
    used_after = used_nodes(mesh.merge_nodes(1e-12))
    mc_ref = mc_mesh(mesh, 3)
    mc_ref.mergeNodes(1e-12)
    nodes_after = mc_ref.getNumberOfNodes()
    return (
        t_mf,
        t_mc,
        {
            "used nodes == 1875": (used_after == 1875, used_after),
            "mc nodes == mefi": (nodes_after == used_after, (used_after, nodes_after)),
        },
    )


def bench_descend():
    n = DESCEND_N
    axes = [np.linspace(0.0, 1.0, n + 1)] * 3
    mesh = mf.build_cmesh(*axes)
    mm = mc_mesh(mesh, 3)
    t_mf = median_time(lambda: mesh.descend())
    t_mc = median_time(lambda: mm.buildDescendingConnectivity())
    f_mf = int(mesh.descend().blocks()["QUAD4"].shape[0])
    f_mc = int(mm.buildDescendingConnectivity()[0].getNumberOfCells())
    expected = 3 * n * n * (n + 1)  # faces of a structured n^3 hexa grid
    return (
        t_mf,
        t_mc,
        {
            "faces == 3 n^2 (n+1)": (
                (f_mf == expected) and (f_mc == expected),
                (expected, f_mf, f_mc),
            )
        },
    )


def bench_overlay():
    n = OVERLAY_N
    m1 = mf.build_cmesh(np.linspace(0.0, 1.0, n + 1), np.linspace(0.0, 1.0, n + 1))
    m2 = mf.build_cmesh(np.linspace(0.2, 0.7, 9), np.linspace(0.2, 0.7, 9))
    m1m = mc_mesh(m1, 2)
    m2m = mc_mesh(m2, 2)
    t_mf = median_time(lambda: m1.overlay(m2), 5)
    t_mc = median_time(
        lambda: mc.MEDCouplingUMesh.Intersect2DMeshes(m1m, m2m, 1e-12), 5
    )
    a_mf = area_2d(m1.overlay(m2))
    a_mc = float(
        sum(
            mc.MEDCouplingUMesh.Intersect2DMeshes(m1m, m2m, 1e-12)[0]
            .getMeasureField(True)
            .getArray()
            .getValues()
        )
    )
    return (
        t_mf,
        t_mc,
        {
            "area == 1 (both)": (
                abs(a_mf - 1.0) < ATOL and abs(a_mc - 1.0) < ATOL,
                (a_mf, a_mc),
            )
        },
    )


# --- main -------------------------------------------------------------------
def row(case, step, mf_t, mc_t):
    ratio = mc_t / mf_t if mf_t > 0 else float("inf")
    print(f"  {case:18s} {step:18s} {mf_t:12.3f} {mc_t:12.3f} {ratio:9.1f}x")


def show_checks(tag, checks):
    for label, (ok, info) in checks.items():
        status = "OK " if ok else "FAIL"
        print(f"    [{status}] {tag}: {label}  {info}")


def main():
    print("=" * 70)
    print("mefikit vs medcoupling benchmark")
    print("=" * 70)
    print(f"workloads: remap-2d {N2D}^2, remap-3d {N3D}^3, remap-3d-poly {NPOLY}^3,")
    print(f"           merge-nodes {MERGE_N}x{MERGE_N}x2 (duplicated interface),")
    print(f"           descend {DESCEND_N}^3, overlay {OVERLAY_N}^2 (+ embedded 8x8)")
    print(f"iterations: {N_ITER} per step (medians)")
    print()
    print(
        f"  {'case':<18s} {'step':<18s} {'mefikit ms':>12s} {'medcoup ms':>12s} {'mc/mf':>9s}"
    )
    print("  " + "-" * 66)

    checks_all = {}

    for name, fn in [
        ("remap-2d", bench_remap_2d),
        ("remap-3d", bench_remap_3d),
        ("remap-3d-poly", bench_remap_3d_poly),
    ]:
        b, p, a, tr, checks = fn()
        row(name, "build/prepare", b, p)
        row(name, "transfer", a, tr)
        checks_all[name] = checks

    b, p, checks = bench_merge()
    row("merge-nodes", "merge", b, p)
    checks_all["merge-nodes"] = checks

    for name, fn in [("descend", bench_descend), ("overlay", bench_overlay)]:
        a, b, checks = fn()
        row(name, "run", a, b)
        checks_all[name] = checks

    print("  " + "-" * 66)
    print()
    print("correctness cross-checks (same geometry on both sides):")
    for tag, checks in checks_all.items():
        show_checks(tag, checks)

    print()
    print("done.")


if __name__ == "__main__":
    main()
