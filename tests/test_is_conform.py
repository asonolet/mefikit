"""Regression tests for `mefikit.is_conform`.

`is_conform` decides whether a mesh is conformal: cells do not intersect one another, and two
neighbour cells that share any node share a whole common face. A recurring false negative is a
cell whose boundary is spelled with coincident duplicate nodes, which this module pins down.
"""

from pathlib import Path

import numpy as np
import pytest

import mefikit as mf

_TOL = 1e-9
_SEP = np.iinfo(np.uint64).max

_FIXTURES = Path(__file__).parent / "conformize"


def _phed(coords, cells):
    """Builds a `PHED` mesh: `cells` is a list of cells, each a list of face rings."""
    data, offsets = [], []
    for i, cell in enumerate(cells):
        if i:
            data.append(_SEP)
        for face in cell:
            data.extend(face)
            data.append(_SEP)
        offsets.append(len(data))
    data.pop()
    offsets[-1] = len(data)
    mesh = mf.UMesh(np.asarray(coords, dtype=float))
    mesh.add_poly_block(
        "PHED", np.asarray(data, dtype=np.uintp), np.asarray(offsets, dtype=np.uintp)
    )
    return mesh


def _unit_cube_faces():
    """The six faces of the unit cube with nodes `n(i, j, k) = i + 2 j + 4 k`."""
    return [
        [0, 2, 3, 1],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [1, 3, 7, 5],
        [3, 2, 6, 7],
        [2, 0, 4, 6],
    ]


def _unit_cube_coords():
    return [
        [float(i), float(j), float(k)]
        for k in range(2)
        for j in range(2)
        for i in range(2)
    ]


def _kinds(report):
    return [issue["kind"] for issue in report.issues]


def test_accepts_a_single_hex():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    assert mf.is_conform(mesh, _TOL).is_conform


def test_accepts_two_stacked_hexes():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0, 2.0])
    assert mf.is_conform(mesh, _TOL).is_conform


def test_rejects_cell_folded_by_coincident_duplicates():
    """The reported false negative: a cell spells part of its boundary with coincident duplicates.

    The unit cube's `x = 1` face references fresh nodes at the locations of nodes 5 and 7, which
    the top face still uses. The cell folds onto itself, so the mesh is not conformal.
    """
    coords = _unit_cube_coords()
    coords.append(coords[5])
    coords.append(coords[7])
    faces = _unit_cube_faces()
    faces[3] = [1, 3, 9, 8]
    mesh = _phed(coords, [faces])

    report = mf.is_conform(mesh, _TOL)
    assert not report.is_conform
    assert report.n_issues >= 2
    assert set(_kinds(report)) == {"merged_nodes"}


def test_rejects_partially_shared_interface():
    """Two cells whose interface shares some node ids but not all is not conformal."""
    coords = _unit_cube_coords()
    a_faces = _unit_cube_faces()

    base = len(coords)
    for k in range(2):
        for j in range(2):
            for i in range(2):
                coords.append([1.0 + i, float(j), float(k)])
    b_faces = [[base + n for n in face] for face in _unit_cube_faces()]
    # B's x = 1 face keeps A's ids 1 and 5 but spells the other two with its own nodes.
    b_faces[2] = [1, 5, base + 7, base + 4]

    mesh = _phed(coords, [a_faces, b_faces])
    report = mf.is_conform(mesh, _TOL)
    assert not report.is_conform
    assert "merged_nodes" in _kinds(report)


def test_rejects_volume_overlap():
    """Two cells overlapping in volume are reported as intersecting."""
    coords = _unit_cube_coords()
    a_faces = _unit_cube_faces()
    base = len(coords)
    for x, y, z in _unit_cube_coords():
        coords.append([x + 0.5, y, z])
    b_faces = [[base + n for n in face] for face in _unit_cube_faces()]

    mesh = _phed(coords, [a_faces, b_faces])
    report = mf.is_conform(mesh, _TOL)
    assert not report.is_conform
    assert "intersecting_cells" in _kinds(report)


def test_reports_messages_and_positions():
    """Every issue carries a message and a centre, so a report can drive a selection."""
    coords = _unit_cube_coords()
    coords.append(coords[5])
    coords.append(coords[7])
    faces = _unit_cube_faces()
    faces[3] = [1, 3, 9, 8]
    report = mf.is_conform(_phed(coords, [faces]), _TOL)
    for issue in report.issues:
        assert isinstance(issue["message"], str) and issue["message"]
        assert len(issue["center"]) == 3


def test_invalid_tolerance_raises():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        mf.is_conform(mesh, -1.0)


def _read(name):
    return mf.UMesh.read(str(_FIXTURES / name))


def _phedron_only(mesh):
    return mesh.select(mf.sel.types(["PHED"])).to_mesh()


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "Compound_Mesh_to_colinearize_simplier_41.med",
            {"non_conforming_faces", "intersecting_cells"},
        ),
        (
            "Mesh_poly_non_conforme_complexe.med",
            {"non_conforming_faces", "intersecting_cells"},
        ),
    ],
)
def test_rejects_polyhedral_fixtures(name, expected):
    """The shipped non-conforming polyhedral fixtures are detected, with the expected kinds.

    Their defects are interfaces that are only partially shared (mismatched face
    tessellations) plus a genuine volume overlap; both are found without any tolerance slack.
    """
    report = mf.is_conform(_phedron_only(_read(name)), _TOL)
    assert not report.is_conform
    assert set(_kinds(report)) == expected


def test_rejects_poly_fixture():
    """`Mesh_poly_non_conforme.med` carries interfaces that are only partially shared."""
    report = mf.is_conform(_phedron_only(_read("Mesh_poly_non_conforme.med")), _TOL)
    assert not report.is_conform
    assert set(_kinds(report)) == {"non_conforming_faces"}
    assert report.n_issues == 3


def test_rejects_simplify_fixture():
    """`Mesh_non_conforme_simplify.med` folds its cells through near-coincident duplicate nodes.

    The pair of coincident nodes of each fold is about `8e-3` apart, so the default `1e-9`
    tolerance does not see it; with a tolerance matching that spacing the folds are reported.
    """
    mesh = _phedron_only(_read("Mesh_non_conforme_simplify.med"))
    assert mf.is_conform(mesh, _TOL).is_conform
    report = mf.is_conform(mesh, 1e-2)
    assert not report.is_conform
    assert set(_kinds(report)) == {"merged_nodes"}
    assert report.n_issues == 4


def test_fixtures_reject_non_poly_cells():
    """A whole fixture mixing polyhedral, polygonal and simplex cells is reported as unsupported.

    `conformize` only handles volume cells, so a mesh holding polygonal or simplex cells is
    rejected up front rather than silently diagnosed.
    """
    report = mf.is_conform(_read("Mesh_poly_non_conforme.med"), _TOL)
    assert not report.is_conform
    assert set(_kinds(report)) == {"unsupported_cells"}
