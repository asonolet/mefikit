"""Correctness comparison of mefikit.conformize against medcoupling conformize3D.

MEDCoupling's ``conformize3D`` handles only partition-like non-conformities: one side of an
interface must tile the other one exactly (grids of 3 vs 6 cells work, 2 vs 3 do not), and
anything else is rejected with an exception. mefikit's ``conformize`` also imprints interfaces
whose faces genuinely cross. Every case below therefore asserts mefikit's correctness on its
own, and compares the two libraries only where medcoupling can run at all.

Each case is a unit-box mesh built from axis-aligned boxes, welded by coordinate so that
shared interface corners are shared node ids and no two nodes coincide. Correctness is
checked with three complementary tools:

- a face census (every geometric cell face must be covered by at most two cells, and every
  face covered by exactly one cell must lie on the box surface): this is what catches
  non-conformal interfaces, which ``mf.is_conform`` deliberately does not;
- the measured volume (1.0, every cell positive): this catches corrupted geometry and the
  flipped-face orientation medcoupling's output is known to need its
  ``orientCorrectlyPolyhedrons`` healing step for;
- cell and node counts: conformize must neither split cells apart nor create nodes, except
  at genuine interface crossings.
"""

from __future__ import annotations

from typing import NamedTuple

import medcoupling as mc
import numpy as np
import pytest

import mefikit as mf

TOL = 1e-9
_SEP = np.iinfo(np.uint64).max
_HEX_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


# --- mesh builders ----------------------------------------------------------
def box_mesh(boxes):
    """HEX8 mesh from axis-aligned boxes `[(x0, x1, y0, y1, z0, z1), ...]`, welded by coordinate."""
    ids, coords, cells = {}, [], []
    for x0, x1, y0, y1, z0, z1 in boxes:
        ring = []
        for z in (z0, z1):
            for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
                key = (round(x, 12), round(y, 12), round(z, 12))
                i = ids.get(key)
                if i is None:
                    i = len(coords)
                    ids[key] = i
                    coords.append(key)
                ring.append(i)
        cells.append(ring)
    mesh = mf.UMesh(np.array(coords, np.float64))
    mesh.add_regular_block("HEX8", np.array(cells, np.uintp))
    return mesh


def _grid(n, z0, z1):
    """n x n unit-square cells between z0 and z1."""
    return [
        (i / n, (i + 1) / n, j / n, (j + 1) / n, z0, z1)
        for i in range(n)
        for j in range(n)
    ]


def _strips(n, z0, z1):
    """n strips along x over the unit square."""
    return [(i / n, (i + 1) / n, 0.0, 1.0, z0, z1) for i in range(n)]


def _refine(box, n):
    """Subdivides one box into n x n x n sub-boxes."""
    x0, x1, y0, y1, z0, z1 = box
    return [
        (
            x0 + i * (x1 - x0) / n,
            x0 + (i + 1) * (x1 - x0) / n,
            y0 + j * (y1 - y0) / n,
            y0 + (j + 1) * (y1 - y0) / n,
            z0 + k * (z1 - z0) / n,
            z0 + (k + 1) * (z1 - z0) / n,
        )
        for i in range(n)
        for j in range(n)
        for k in range(n)
    ]


def _full_grid(n):
    """n x n x n unit-box cells."""
    return [
        (i / n, (i + 1) / n, j / n, (j + 1) / n, k / n, (k + 1) / n)
        for i in range(n)
        for j in range(n)
        for k in range(n)
    ]


def _without(boxes, *drop):
    return [b for b in boxes if not all(a == c for a, c in zip(b, drop))]


# --- cases ------------------------------------------------------------------
class Case(NamedTuple):
    mesh: mf.UMesh
    cells_out: int  # cells after conformizing: inputs are never split by these cases
    nodes_out: int  # node ids used after conformizing (only crossings add some)
    problems_in: int  # interior-once faces of the input: its non-conformities
    mc_ok: bool  # whether medcoupling conformize3D can run at all


def _strips_3_vs_6():
    # The medcoupling-friendly example: one side of the interface is 3 strips, the other 6.
    return box_mesh(_strips(6, 0.0, 0.5) + _strips(3, 0.5, 1.0))


def _one_vs_grid():
    return box_mesh(_grid(3, 0.0, 0.5) + [(0.0, 1.0, 0.0, 1.0, 0.5, 1.0)])


def _corner_three_directions():
    # A 2x2x2 grid whose (0, 0, 0) cell is refined 2x2x2: one-against-many interfaces
    # glued in +x, +y and +z at the same time.
    cell = (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
    return box_mesh(_without(_full_grid(2), *cell) + _refine(cell, 2))


def _center_six_directions():
    # A 3x3x3 grid whose center cell is refined 2x2x2: interfaces in all six directions.
    center = (1 / 3, 2 / 3, 1 / 3, 2 / 3, 1 / 3, 2 / 3)
    return box_mesh(_without(_full_grid(3), *center) + _refine(center, 2))


def _adjacent_pair():
    # Two side-by-side refined cells: perpendicular interfaces meeting along shared edges.
    r1 = (0.0, 0.5, 0.0, 0.5, 0.0, 0.5)
    r2 = (0.5, 1.0, 0.0, 0.5, 0.0, 0.5)
    base = [b for b in _full_grid(2) if b != r1 and b != r2]
    return box_mesh(base + _refine(r1, 2) + _refine(r2, 2))


def _strips_2_vs_3():
    # The medcoupling-hostile example: 2 strips against 3, so their edges cross.
    return box_mesh(_strips(3, 0.0, 0.5) + _strips(2, 0.5, 1.0))


def _grids_3_vs_4():
    # Crossing 3x3 against 4x4 tilings: twelve genuine intersection points on the interface.
    return box_mesh(_grid(3, 0.0, 0.5) + _grid(4, 0.5, 1.0))


CASES = {
    "strips_3_vs_6": Case(_strips_3_vs_6(), 9, 36, 9, True),
    "one_vs_grid_3": Case(_one_vs_grid(), 10, 36, 10, True),
    "corner_three_directions": Case(_corner_three_directions(), 15, 46, 15, True),
    "center_six_directions": Case(_center_six_directions(), 34, 83, 30, True),
    "adjacent_pair": Case(_adjacent_pair(), 22, 60, 20, True),
    "strips_2_vs_3": Case(_strips_2_vs_3(), 5, 24, 5, False),
    "grids_3_vs_4": Case(_grids_3_vs_4(), 25, 90, 25, False),
}


# --- face census ------------------------------------------------------------
def _blocks(mesh):
    for et, conn in mesh.blocks().items():
        if isinstance(conn, tuple):
            data, offsets = conn
            prev = 0
            for off in offsets:
                yield et, np.asarray(data[prev:off])
                prev = off
        else:
            for cell in np.asarray(conn):
                yield et, cell


def _cell_faces(et, conn):
    if et == "PHED":
        faces, cur = [], []
        for n in conn:
            if n == _SEP:
                faces.append([int(g) for g in cur])
                cur = []
            else:
                cur.append(int(n))
        if cur:
            faces.append([int(g) for g in cur])
        return faces
    if et == "HEX8":
        return [[int(conn[i]) for i in f] for f in _HEX_FACES]
    raise AssertionError(f"unhandled cell type {et!r}")


def _face_key(pts, tol=1e-9):
    """The outline of a planar face, as a frozenset of rounded corner coordinates.

    Collinear ring vertices are dropped first, so a face ring carrying the collinear node
    an imprint introduced compares equal to its plain partner ring.
    """
    pts = [tuple(p) for p in pts]
    dedup = []
    for p in pts:
        if not dedup or max(abs(a - b) for a, b in zip(p, dedup[-1])) > tol:
            dedup.append(p)
    if len(dedup) > 1 and max(abs(a - b) for a, b in zip(dedup[0], dedup[-1])) <= tol:
        dedup.pop()

    def collinear(a, b, c):
        ab, bc = np.subtract(b, a), np.subtract(c, b)
        return float(np.linalg.norm(np.cross(ab, bc))) <= tol * max(
            float(np.linalg.norm(ab)), tol
        )

    changed = True
    while changed and len(dedup) > 3:
        changed = False
        for i in range(len(dedup)):
            if collinear(dedup[i - 1], dedup[i], dedup[(i + 1) % len(dedup)]):
                dedup.pop(i)
                changed = True
                break
    return frozenset((round(x, 9), round(y, 9), round(z, 9)) for x, y, z in dedup)


def census(mesh):
    """Maps every geometric cell face outline to the number of cells covering it."""
    coords = np.asarray(mesh.coords())
    counts = {}
    for et, conn in _blocks(mesh):
        for ring in _cell_faces(et, conn):
            key = _face_key(coords[ring])
            counts[key] = counts.get(key, 0) + 1
    return counts


def _on_box_surface(key, tol=1e-9):
    """True iff the whole face lies in one of the six unit-box boundary planes."""
    return any(
        all(abs(p[axis] - value) <= tol for p in key)
        for axis in range(3)
        for value in (0.0, 1.0)
    )


def problems(mesh):
    """Non-conformities a face census can see: interior faces covered once or over twice."""
    found = []
    for key, count in census(mesh).items():
        if count > 2:
            found.append(("shared by more than two cells", count, key))
        elif count == 1 and not _on_box_surface(key):
            found.append(("interior face covered by a single cell", count, key))
    return found


def used_nodes(mesh):
    return len(
        np.unique(np.concatenate([cell[cell != _SEP] for _, cell in _blocks(mesh)]))
    )


def used_bbox(mesh):
    used = np.unique(np.concatenate([cell[cell != _SEP] for _, cell in _blocks(mesh)]))
    coords = np.asarray(mesh.coords())[used]
    return coords.min(0), coords.max(0)


def cell_volumes(mesh):
    return np.concatenate([np.asarray(v) for v in mesh.measure().values()])


def total_volume(mesh):
    return float(sum(np.asarray(v).sum() for v in mesh.measure().values()))


def assert_conformal_mesh(mesh, case):
    """The shared correctness contract for every conformized mesh."""
    assert mesh.num_elements() == case.cells_out
    assert used_nodes(mesh) == case.nodes_out
    lo, hi = used_bbox(mesh)
    np.testing.assert_allclose(lo, 0.0, atol=1e-12)
    np.testing.assert_allclose(hi, 1.0, atol=1e-12)
    assert abs(total_volume(mesh) - 1.0) < 1e-9
    assert (cell_volumes(mesh) > 0.0).all()
    assert problems(mesh) == []
    assert mf.is_conform(mesh, TOL).is_conform


# --- medcoupling runner -----------------------------------------------------
def mc_conformize(mesh):
    """medcoupling's documented workflow for conformize3D."""
    mm = mesh.to_mc()
    mm.setMeshDimension(3)
    mm.convertAllToPoly()
    mm.conformize3D(TOL)
    mm.orientCorrectlyPolyhedrons()
    return mm


# --- tests ------------------------------------------------------------------
def test_inputs_are_non_conformal():
    for name, case in CASES.items():
        found = problems(case.mesh)
        assert len(found) == case.problems_in, name
        assert case.problems_in > 0, name


@pytest.mark.parametrize("name", CASES)
def test_mefikit_conformize(name):
    case = CASES[name]
    assert_conformal_mesh(mf.conformize(case.mesh, TOL), case)


@pytest.mark.parametrize("name", CASES)
def test_medcoupling_conformize3D(name):
    case = CASES[name]
    if not case.mc_ok:
        with pytest.raises(mc.InterpKernelException, match="partition-like"):
            mc_conformize(case.mesh)
        return

    mm = mc_conformize(case.mesh)
    assert mm.getNumberOfCells() == case.cells_out
    assert mm.getNumberOfNodes() == case.nodes_out
    vols = np.asarray(mm.getMeasureField(True).getArray().getValues())
    assert (vols > 0.0).all()
    assert abs(float(vols.sum()) - 1.0) < 1e-9

    out = mf.UMesh.from_mc(mm)
    assert_conformal_mesh(out, case)

    # Same geometry as mefikit: no cell was split differently by either library.
    mf_vols = np.sort(cell_volumes(mf.conformize(case.mesh, TOL)))
    np.testing.assert_allclose(np.sort(vols), mf_vols, atol=1e-9)
