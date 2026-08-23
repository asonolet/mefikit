import numpy as np
import pytest

import mefikit as mf


def _cells(mesh):
    for et, conn in mesh.blocks().items():
        if isinstance(conn, tuple):
            data, offsets = conn
            cells = []
            prev = 0
            for off in offsets:
                cells.append(np.asarray(data[prev:off]))
                prev = off
        else:
            cells = list(np.asarray(conn))
        for cell in cells:
            yield et, cell


def _mesh_area(mesh):
    total = 0.0
    for _, cell in _cells(mesh):
        pts = np.array([mesh.coords()[i] for i in cell])
        x, y = pts[:, 0], pts[:, 1]
        total += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return total


@pytest.fixture
def overlay_pair():
    mesh1 = mf.build_cmesh([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    mesh2 = mf.build_cmesh([0.25, 0.75], [0.25, 0.75])
    return mesh1, mesh2


def test_overlay_enum_variants():
    for name in (
        "IMPRINT",
        "UNION",
        "INTERSECTION",
        "DIFFERENCE",
        "SYMMETRIC_DIFFERENCE",
    ):
        assert hasattr(mf.OverlayOperation, name)


def test_overlay_imprint(overlay_pair):
    mesh1, mesh2 = overlay_pair
    out = mesh1.overlay(mesh2)
    assert _mesh_area(out) == pytest.approx(1.0)


def test_overlay_union(overlay_pair):
    mesh1, mesh2 = overlay_pair
    out = mesh1.overlay(mesh2, mf.OverlayOperation.UNION)
    assert _mesh_area(out) == pytest.approx(1.0)


def test_overlay_intersection(overlay_pair):
    mesh1, mesh2 = overlay_pair
    out = mesh1.overlay(mesh2, mf.OverlayOperation.INTERSECTION)
    assert _mesh_area(out) == pytest.approx(0.25)


def test_overlay_difference(overlay_pair):
    mesh1, mesh2 = overlay_pair
    out = mesh1.overlay(mesh2, mf.OverlayOperation.DIFFERENCE)
    assert _mesh_area(out) == pytest.approx(0.75)


def test_overlay_symmetric_difference(overlay_pair):
    mesh1, mesh2 = overlay_pair
    out = mesh1.overlay(mesh2, mf.OverlayOperation.SYMMETRIC_DIFFERENCE)
    assert _mesh_area(out) == pytest.approx(0.75)


def test_overlay_disjoint_union():
    mesh1 = mf.build_cmesh([0.0, 1.0], [0.0, 1.0])
    mesh2 = mf.build_cmesh([2.0, 3.0], [2.0, 3.0])
    out = mesh1.overlay(mesh2, mf.OverlayOperation.UNION)
    assert list(_cells(out)) and _mesh_area(out) == pytest.approx(2.0)


def _grid_surface(n, tri=False, transform=None):
    coords = np.zeros(((n + 1) * (n + 1), 3))
    for j in range(n + 1):
        for i in range(n + 1):
            coords[j * (n + 1) + i] = (i / n, j / n, 0.0)
    if transform is not None:
        coords = transform(coords)

    def nid(i, j):
        return j * (n + 1) + i

    cells = []
    for j in range(n):
        for i in range(n):
            quad = [nid(i, j), nid(i + 1, j), nid(i + 1, j + 1), nid(i, j + 1)]
            if tri:
                cells.append(quad[:3])
                cells.append([quad[0], quad[2], quad[3]])
            else:
                cells.append(quad)
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("TRI3" if tri else "QUAD4", np.array(cells, dtype=np.uint))
    return mesh


def _area3d(mesh):
    total = 0.0
    for _, cell in _cells(mesh):
        pts = np.array([mesh.coords()[i] for i in cell])
        normal = np.cross(pts, np.roll(pts, -1, axis=0)).sum(axis=0)
        total += 0.5 * np.linalg.norm(normal)
    return total


def _num_cells(mesh):
    return sum(1 for _ in _cells(mesh))


def test_overlay_surfaces_identical_grids():
    out = _grid_surface(4).overlay_surfaces(_grid_surface(4))
    assert _num_cells(out.refined1) == 16
    assert _num_cells(out.refined2) == 16
    assert _area3d(out.refined1) == pytest.approx(1.0)
    assert _area3d(out.refined2) == pytest.approx(1.0)


def test_overlay_surfaces_tri_vs_quad():
    out = _grid_surface(4).overlay_surfaces(_grid_surface(4, tri=True))
    assert _num_cells(out.refined1) == 32
    assert _num_cells(out.refined2) == 32
    assert _area3d(out.refined1) == pytest.approx(1.0)
    assert _area3d(out.refined2) == pytest.approx(1.0)


def _tilt(c):
    out = c.copy()
    out[:, 2] += c[:, 1]
    return out


def _shift_x10(c):
    out = c.copy()
    out[:, 0] += 10.0
    return out


def test_overlay_surfaces_tilted():
    # tilt about the x-axis so mesh2 only touches mesh1 along the y = 0 edge
    tilted = _grid_surface(4, transform=_tilt)
    out = _grid_surface(4).overlay_surfaces(tilted)
    assert _area3d(out.refined1) == pytest.approx(1.0)
    assert _area3d(out.refined2) == pytest.approx(np.sqrt(2.0))


def test_overlay_surfaces_parents():
    out = _grid_surface(2).overlay_surfaces(_grid_surface(4))
    parents1 = dict(out.parents1)
    parents2 = dict(out.parents2)
    assert set(parents1) == {("QUAD4", i) for i in range(4)}
    assert set(parents2) == {("QUAD4", i) for i in range(16)}
    # every refined face appears as the product of exactly one input face
    pieces1 = [piece for pieces in parents1.values() for piece in pieces]
    pieces2 = [piece for pieces in parents2.values() for piece in pieces]
    assert len(pieces1) == len(set(pieces1)) == _num_cells(out.refined1)
    assert len(pieces2) == len(set(pieces2)) == _num_cells(out.refined2)


def test_overlay_surfaces_partial_overlap_raises():
    with pytest.raises(ValueError):
        _grid_surface(4).overlay_surfaces(_grid_surface(8))


def test_overlay_surfaces_disjoint_verbatim():
    far = _grid_surface(2, transform=_shift_x10)
    out = _grid_surface(2).overlay_surfaces(far)
    assert _num_cells(out.refined1) == 4
    assert _num_cells(out.refined2) == 4
    assert min(np.array(out.refined2.coords())[:, 0]) == pytest.approx(10.0)
