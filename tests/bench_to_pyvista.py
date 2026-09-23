#!/usr/bin/env python
"""Benchmark ``UMesh.to_pyvista()`` on polyhedral meshes.

Run with:  uv run python tests/bench_to_pyvista.py

Covered workloads
-----------------
* med files      : tests/data/mesh_27.med and mesh_36.med (2000 polyhedra each)
                   and cell80_mesh_27.med (single polyhedron)
* synthetic      : ``polyze()`` of structured N^3 HEX8 grids (N = 4..24) to
                   show scaling on larger polyhedral meshes
* polygons       : ``polyze()`` of structured NxN quad grids (N = 64..256)
                   producing PGON blocks

For each mesh the median wall time of ``to_pyvista()`` is reported (in ms)
together with the throughput in cells/s. A lightweight correctness check is
printed for every mesh (cell count and cell types match the source mesh).
"""

from __future__ import annotations

import time

import numpy as np

import mefikit as mf

N_ITER = 10

MED_FILES = [
    "tests/data/mesh_27.med",
    "tests/data/mesh_36.med",
    "tests/data/cell80_mesh_27.med",
]

SYNTH_GRIDS = [4, 8, 16, 24]
PGON_GRIDS = [64, 128, 256]


def median_time(fn, n=N_ITER) -> float:
    """Median wall time of ``fn`` over ``n`` runs, in milliseconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times) * 1e3)


def bench(mesh: mf.UMesh) -> tuple[float, bool]:
    mesh.to_pyvista()  # warm up pyvista imports
    t_total = median_time(mesh.to_pyvista, n=N_ITER)
    ok = mesh.to_pyvista().n_cells == mesh.num_elements()
    return t_total, ok


def main() -> None:
    print("=" * 70)
    print("UMesh.to_pyvista() polyhedral conversion benchmark")
    print("=" * 70)
    print(f"iterations: {N_ITER} per mesh (medians)")
    print()
    print(f"  {'workload':<30s} {'cells':>7s} {'ms/conv':>9s} {'cells/s':>12s}")
    print("  " + "-" * 66)

    rows = []
    for path in MED_FILES:
        mesh = mf.UMesh.read(path)
        t_total, ok = bench(mesh)
        rows.append((path, mesh.num_elements(), t_total, ok))

    for n in SYNTH_GRIDS:
        axes = [np.linspace(0.0, 1.0, n + 1)] * 3
        mesh = mf.build_cmesh(*axes).polyze()
        t_total, ok = bench(mesh)
        rows.append((f"polyze {n}^3 hexa", mesh.num_elements(), t_total, ok))

    for n in PGON_GRIDS:
        gx = np.linspace(0.0, 1.0, n + 1)
        mesh = mf.build_cmesh(gx, gx).polyze()
        t_total, ok = bench(mesh)
        rows.append((f"polyze {n}x{n} quad", mesh.num_elements(), t_total, ok))

    all_ok = True
    for workload, ncells, t_total, ok in rows:
        all_ok &= ok
        rate = ncells / (t_total * 1e-3) if t_total > 0 else float("inf")
        print(
            f"  {workload:<30s} {ncells:>7d} {t_total:>9.3f} "
            f"{rate:>12.0f}   {'OK' if ok else 'FAIL'}",
            flush=True,
        )

    print("  " + "-" * 66)
    print(f"correctness: {'all meshes OK' if all_ok else 'FAILURES PRESENT'}")
    print()
    print("done.")


if __name__ == "__main__":
    main()
