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
