from __future__ import annotations

import medcoupling as mc
import numpy as np
import pytest

import mefikit as mf

# ---------------------------------------------------------------------------
# Mesh / field helpers
# ---------------------------------------------------------------------------


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


def line_mesh(points) -> mf.UMesh:
    coords = np.ascontiguousarray(np.asarray(points, dtype=float))
    conn = np.array([[i, i + 1] for i in range(len(coords) - 1)], dtype=np.uintp)
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("SEG2", conn)
    return mesh


def point_mesh(points) -> mf.UMesh:
    coords = np.ascontiguousarray(np.asarray(points, dtype=float))
    conn = np.arange(len(coords), dtype=np.uintp).reshape(-1, 1)
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("VERTEX", conn)
    return mesh


def set_field(mesh: mf.UMesh, et: str, values) -> None:
    values = np.asarray(values, dtype=float)
    mesh.set_field("T", {et: np.ascontiguousarray(values.reshape(values.shape[0], -1))})


def mf_remap(
    src: mf.UMesh,
    tgt: mf.UMesh,
    src_et: str,
    tgt_et: str,
    values,
    *,
    extensive: bool = False,
    def_val: float = 0.0,
) -> np.ndarray:
    set_field(src, src_et, values)
    operator = mf.transfer.ConservativeP0(src, tgt, def_val=def_val)
    return np.asarray(operator.eval("T", extensive=extensive)[tgt_et]).ravel()


def mc_mesh(mesh: mf.UMesh, dim: int) -> mc.MEDCouplingUMesh:
    result = mesh.to_mc()
    result.setMeshDimension(dim)
    return result


def mc_remap(
    src: mc.MEDCouplingUMesh,
    tgt: mc.MEDCouplingUMesh,
    values,
    *,
    def_val: float = 0.0,
    intersection_type: int | None = None,
    nature: int = mc.IntensiveConservation,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
    field.setMesh(src)
    field.setArray(
        mc.DataArrayDouble(np.ascontiguousarray(values.reshape(values.shape[0], -1)))
    )
    field.setNature(nature)
    remapper = mc.MEDCouplingRemapper()
    if intersection_type is not None:
        remapper.setIntersectionType(intersection_type)
    remapper.prepare(src, tgt, "P0P0")
    return np.asarray(
        remapper.transferField(field, def_val).getArray().toNumPyArray()
    ).ravel()


def target_measure(mesh: mf.UMesh, et: str) -> np.ndarray:
    return np.asarray(mesh.measure()[et]).ravel()


# ---------------------------------------------------------------------------
# 3D -> 2D
# ---------------------------------------------------------------------------


def test_3d_to_2d_matches_medcoupling():
    xs = np.linspace(0.0, 1.0, 5)
    src = mf.build_cmesh(xs, xs, xs)
    values = np.arange(1, src.num_elements() + 1, dtype=float)

    # Shifted surface grid at z = 0.4 (not a source-cell plane), non-conforming in x and y.
    tgt = surface_grid_3d(
        np.linspace(0.03, 0.97, 6),
        np.linspace(0.05, 0.95, 5),
        0.4,
    )

    expected = mc_remap(mc_mesh(src, 3), mc_mesh(tgt, 2), values, def_val=-1.0)
    got = mf_remap(src, tgt, "HEX8", "QUAD4", values, def_val=-1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    # Mefikit extensive = integral over the target; MEDCoupling intensive * target measure.
    expected_extensive = expected * target_measure(tgt, "QUAD4")
    got_extensive = mf_remap(src, tgt, "HEX8", "QUAD4", values, extensive=True)
    np.testing.assert_allclose(
        got_extensive, expected_extensive, rtol=1e-10, atol=1e-12
    )


def test_3d_to_2d_partial_coverage_and_default_matches_medcoupling():
    xs = np.linspace(0.0, 1.0, 5)
    src = mf.build_cmesh(xs, xs, xs)
    values = np.linspace(1.0, 2.0, src.num_elements())

    tgt = surface_grid_3d(
        np.linspace(-0.5, 1.5, 9),
        np.linspace(-0.5, 1.5, 9),
        0.4,
    )

    expected = mc_remap(mc_mesh(src, 3), mc_mesh(tgt, 2), values, def_val=np.nan)
    got = mf_remap(src, tgt, "HEX8", "QUAD4", values, def_val=np.nan)
    covered = np.isfinite(expected)
    assert np.any(~covered)  # at least one fully uncovered target cell
    np.testing.assert_array_equal(np.isfinite(got), covered)
    np.testing.assert_allclose(got[covered], expected[covered], rtol=1e-10, atol=1e-12)


def test_3d_to_2d_shared_face_matches_medcoupling():
    src = mf.build_cmesh(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    values = np.array([10.0, 20.0])
    tgt = mf.UMesh(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        )
    )
    tgt.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uintp))

    expected = mc_remap(mc_mesh(src, 3), mc_mesh(tgt, 2), values, def_val=0.0)
    got = mf_remap(src, tgt, "HEX8", "QUAD4", values, def_val=0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(got, [30.0], rtol=1e-12, atol=1e-12)


def test_3d_to_2d_oblique_plane_matches_medcoupling():
    src = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
    )
    values = np.arange(1, src.num_elements() + 1, dtype=float)
    tgt = mf.UMesh(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
    )
    tgt.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uintp))

    expected = mc_remap(mc_mesh(src, 3), mc_mesh(tgt, 2), values, def_val=-1.0)
    got = mf_remap(src, tgt, "HEX8", "QUAD4", values, def_val=-1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_3d_to_2d_vector_field_matches_medcoupling():
    xs = np.linspace(0.0, 1.0, 3)
    src = mf.build_cmesh(xs, xs, xs)
    values = np.stack(
        [
            np.arange(1, src.num_elements() + 1, dtype=float),
            np.arange(1, src.num_elements() + 1, dtype=float) * 2.0,
        ],
        axis=1,
    )
    tgt = surface_grid_3d(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4), 0.4)

    expected = mc_remap(mc_mesh(src, 3), mc_mesh(tgt, 2), values, def_val=0.0)
    got = mf_remap(src, tgt, "HEX8", "QUAD4", values, def_val=0.0)
    np.testing.assert_allclose(got, expected.ravel(), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 2D -> 1D
# ---------------------------------------------------------------------------


def test_2d_to_1d_matches_medcoupling():
    xs = np.linspace(0.0, 1.0, 5)
    src = mf.build_cmesh(xs, xs)
    values = np.arange(1, src.num_elements() + 1, dtype=float)
    tgt = line_mesh([[x, 0.4] for x in np.linspace(-0.2, 1.2, 9)])

    expected = mc_remap(
        mc_mesh(src, 2),
        mc_mesh(tgt, 1),
        values,
        def_val=-1.0,
        intersection_type=mc.Geometric2D,
    )
    got = mf_remap(src, tgt, "QUAD4", "SEG2", values, def_val=-1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)

    expected_extensive = mc_remap(
        mc_mesh(src, 2),
        mc_mesh(tgt, 1),
        values,
        def_val=0.0,
        intersection_type=mc.Geometric2D,
    ) * target_measure(tgt, "SEG2")
    got_extensive = mf_remap(
        src, tgt, "QUAD4", "SEG2", values, extensive=True, def_val=0.0
    )
    np.testing.assert_allclose(
        got_extensive, expected_extensive, rtol=1e-10, atol=1e-12
    )


def test_2d_surface_to_1d_in_3d_matches_medcoupling_2d():
    xs = np.linspace(0.0, 1.0, 5)
    src_2d = mf.build_cmesh(xs, xs)
    tgt_2d = line_mesh([[x, 0.4] for x in np.linspace(-0.2, 1.2, 9)])
    values = np.arange(1, src_2d.num_elements() + 1, dtype=float)

    # Same surface and line embedded in the plane z = 0.
    src_3d = mf.UMesh(
        np.ascontiguousarray(
            np.c_[np.asarray(src_2d.coords()), np.zeros(src_2d.coords().shape[0])]
        )
    )
    src_3d.add_regular_block(
        "QUAD4", np.asarray(src_2d.blocks()["QUAD4"], dtype=np.uintp)
    )
    tgt_3d = line_mesh([[x, 0.4, 0.0] for x in np.linspace(-0.2, 1.2, 9)])

    expected = mc_remap(
        mc_mesh(src_2d, 2),
        mc_mesh(tgt_2d, 1),
        values,
        def_val=-1.0,
        intersection_type=mc.Geometric2D,
    )
    got = mf_remap(src_3d, tgt_3d, "QUAD4", "SEG2", values, def_val=-1.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


def test_2d_surface_to_1d_in_3d_off_plane_is_default():
    src = mf.UMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
    )
    src.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uintp))
    tgt = line_mesh([[0.25, 0.5, 0.1], [0.75, 0.5, 0.1]])
    got = mf_remap(src, tgt, "QUAD4", "SEG2", [7.0], def_val=99.0)
    assert got[0] == pytest.approx(99.0)


# ---------------------------------------------------------------------------
# 3D -> 1D and 0D (no measure-based MEDCoupling P0P0 for these pairs)
# ---------------------------------------------------------------------------


def test_3d_to_1d_analytic():
    src = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
    )
    values = np.arange(1, src.num_elements() + 1, dtype=float)
    tgt = line_mesh([[0.1, 0.4, 0.4], [0.9, 0.4, 0.4]])

    got = mf_remap(src, tgt, "HEX8", "SEG2", values)
    got_extensive = mf_remap(src, tgt, "HEX8", "SEG2", values, extensive=True)
    assert got.shape == (1,)
    assert got_extensive[0] == pytest.approx(got[0] * 0.8)

    # The segment crosses two source cells; equal lengths average their values.
    assert got[0] == pytest.approx(1.5)


def test_3d_to_0d_and_2d_to_0d():
    src3 = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
    )
    values3 = np.arange(1, src3.num_elements() + 1, dtype=float)
    pts3 = point_mesh([[0.25, 0.25, 0.25], [2.0, 0.5, 0.5]])
    got3 = mf_remap(src3, pts3, "HEX8", "VERTEX", values3, def_val=-1.0)
    assert got3[0] == pytest.approx(1.0)  # first cell containing (0.25, 0.25, 0.25)
    assert got3[1] == pytest.approx(-1.0)

    # MEDCoupling only defines the extensive point transfer (its intensive mode needs a
    # positive target measure).
    sm = mc_mesh(src3, 3)
    tm = mc_mesh(pts3, 0)
    expected = mc_remap(
        sm,
        tm,
        values3,
        def_val=-1.0,
        intersection_type=mc.PointLocator,
        nature=mc.ExtensiveConservation,
    )
    got_ext = mf_remap(
        src3, pts3, "HEX8", "VERTEX", values3, extensive=True, def_val=-1.0
    )
    np.testing.assert_allclose(got_ext, expected, rtol=1e-10, atol=1e-12)

    src2 = mf.build_cmesh(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    values2 = np.arange(1, src2.num_elements() + 1, dtype=float)
    pts2 = point_mesh([[0.25, 0.25], [2.0, 0.25]])
    got2 = mf_remap(src2, pts2, "QUAD4", "VERTEX", values2, def_val=-1.0)
    assert got2[0] == pytest.approx(1.0)
    assert got2[1] == pytest.approx(-1.0)


def test_0d_point_on_interface_has_single_owner():
    src = mf.build_cmesh(
        np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    values = np.array([10.0, 20.0])
    tgt = point_mesh([[1.0, 0.5, 0.5], [1.5, 0.5, 0.5]])
    got = mf_remap(src, tgt, "HEX8", "VERTEX", values, def_val=-1.0)
    # On the shared face the lowest source index owns the point, never both cells.
    assert got[0] == pytest.approx(10.0)
    assert got[1] == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# API consistency on a downcast operator
# ---------------------------------------------------------------------------


def test_downcast_eval_matches_apply_update_and_lazy_assignment():
    src = mf.build_cmesh(
        np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)
    )
    values = np.arange(1, src.num_elements() + 1, dtype=float).reshape(-1, 1)
    src.set_field("T", {"HEX8": values})
    tgt = surface_grid_3d(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4), 0.4)

    operator = mf.transfer.ConservativeP0(src, tgt)
    eager = np.asarray(operator.eval("T")["QUAD4"]).ravel()

    ref = surface_grid_3d(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4), 0.4)
    operator.apply_update(src, "T", ref, "T", def_val=0.0)
    np.testing.assert_allclose(eager, np.asarray(ref.fields["T"].numpy()).ravel())

    lazy = surface_grid_3d(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4), 0.4)
    lazy.fields["T"] = operator(mf.Field("T"))
    np.testing.assert_allclose(np.asarray(lazy.fields["T"].numpy()).ravel(), eager)


def test_downcast_upcast_raises():
    src = surface_grid_3d(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), 0.0)
    tgt = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
    )
    with pytest.raises(BaseException, match="cannot transfer from lower-dimensional"):
        mf.transfer.ConservativeP0(src, tgt)
