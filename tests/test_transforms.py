import numpy as np

import mefikit as mf


def _quad_mesh():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = mf.UMesh(coords)
    m.add_regular_block("QUAD4", np.array([[0, 1, 3, 2]], dtype=np.uint))
    return m


def test_translate():
    m = _quad_mesh()
    t = m.translate([10.0, 20.0])
    assert np.isclose(t.coords()[1, 0], 11.0)
    assert np.isclose(t.coords()[1, 1], 20.0)
    # the input mesh is unchanged
    assert np.isclose(m.coords()[1, 0], 1.0)


def test_rotate():
    m = _quad_mesh()
    r = m.rotate([0.0, 0.0, 1.0], np.pi / 2)  # radians
    assert np.allclose(r.coords()[1], [0.0, 1.0], atol=1e-9)


def test_2d_rejects_out_of_plane():
    m = _quad_mesh()
    with np.testing.assert_raises(ValueError):
        m.rotate([1.0, 0.0, 0.0], np.pi / 2)  # moves out of the plane
    with np.testing.assert_raises(ValueError):
        m.translate([0.0, 0.0, 0.5])


def test_scale_and_mirror():
    m = _quad_mesh()
    s = m.scale([2.0, 2.0])
    assert np.isclose(s.coords()[3, 0], 2.0)
    g = m.mirror([1.0, 0.0, 0.0])
    assert np.isclose(g.coords()[1, 0], -1.0)


def test_scale_uniform():
    m = _quad_mesh()
    s = m.scale_uniform(3.0)
    assert np.isclose(s.coords()[3, 1], 3.0)


def test_transform_value():
    m = _quad_mesh()
    tr = mf.Transform.translation([1.0, 2.0]) @ mf.Transform.rotation(
        [0.0, 0.0, 1.0], np.pi / 2
    )
    # matrix-product order: `tr2` is applied first, so node (1, 0) -> (0, 1) -> (1, 3)
    out = m.transform(tr)
    assert np.allclose(out.coords()[1], [1.0, 3.0], atol=1e-9)
    # `@` matches the numpy matrix product
    a = mf.Transform.translation([1.0, 2.0, 3.0])
    b = mf.Transform.scaling([2.0])
    assert np.allclose((a @ b).matrix(), a.matrix() @ b.matrix())
    # and the result is a hand-computable homogeneous matrix
    aa = a.matrix().copy()
    aa[0, 3], aa[1, 3], aa[2, 3] = 1.0, 2.0, 3.0
    assert np.allclose(aa, a.matrix())


def test_transform_from_matrix():
    m = _quad_mesh()
    mat = np.eye(4)
    mat[0, 3] = 7.0
    out = m.transform(mf.Transform.from_matrix(mat))
    assert np.isclose(out.coords()[0, 0], 7.0)
    # raw 4x4 arrays are also accepted
    out = m.transform(mat)
    assert np.isclose(out.coords()[0, 0], 7.0)


def test_transform_then_and_inverse():
    tr = mf.Transform.translation([1.0, 0.0, 0.0]).then(
        mf.Transform.rotation([0.0, 0.0, 1.0], np.pi / 2)
    )
    m = _quad_mesh()
    out = m.transform(tr)
    # node (1, 0) translated to (2, 0) then rotated to (0, 2)
    assert np.allclose(out.coords()[1], [0.0, 2.0], atol=1e-9)
    back = tr.inverse()
    restored = out.transform(back)
    assert np.allclose(restored.coords(), m.coords(), atol=1e-9)


def test_duplicate():
    m = _quad_mesh()
    dup = m.duplicate(mf.Transform.translation([0.0, 3.0, 0.0]), 3)
    assert dup.coords().shape[0] == 12
    assert dup.block_types() == ["QUAD4"]
    # copies sit at y = 0, 3, 6
    assert np.allclose(
        np.sort(np.unique(dup.coords()[:, 1])), [0.0, 1.0, 3.0, 4.0, 6.0, 7.0]
    )


def test_duplicate_zero_rejected():
    m = _quad_mesh()
    with np.testing.assert_raises(ValueError):
        m.duplicate(mf.Transform.identity(), 0)


def test_aggregate_and_concat():
    m = _quad_mesh()
    other = m.translate([5.0, 0.0])
    joined = mf.concat(m, other)
    assert joined.coords().shape[0] == 8
    blk = joined.blocks()["QUAD4"]
    assert np.array_equal(blk[1], [4, 5, 7, 6])  # shifted node indices

    agg = mf.aggregate([m, other, m.translate([10.0, 0.0])])
    assert agg.coords().shape[0] == 12


def test_aggregate_mismatched_dimension():
    m = _quad_mesh()
    seg = mf.UMesh(np.array([[0.0], [1.0]]))
    with np.testing.assert_raises(ValueError):
        mf.concat(m, seg)


def test_set_coords():
    m = _quad_mesh()
    m.set_coords(np.full((4, 2), 1.0))
    assert np.allclose(m.coords(), 1.0)
    with np.testing.assert_raises(ValueError):
        m.set_coords(np.zeros((3, 2)))


def test_transform_coords_function():
    m = _quad_mesh()
    m.transform_coords(lambda c: np.concatenate((c[:, :1] ** 2, c[:, 1:]), axis=1))
    assert np.isclose(m.coords()[1, 0], 1.0)
    with np.testing.assert_raises(ValueError):
        m.transform_coords(lambda c: c[:, :1])  # wrong shape


def test_transform_preserves_fields_and_groups():
    m = _quad_mesh()
    m.set_field("temperature", {"QUAD4": np.array([21.5])})
    m.groups["heated"] = mf.sel.ids({"QUAD4": [0]})
    t = m.translate([3.0, 0.0])
    assert np.allclose(t.fields["temperature"].values()["QUAD4"], [21.5])
    assert len(t.groups["heated"]) == 1  # group survives the transform


def test_angles_are_radians():
    m = _quad_mesh()
    # 180 degrees is NOT a valid "radian" reading: a real 90° turn is pi/2.
    r = m.rotate([0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(r.coords()[1], [0.0, 1.0], atol=1e-9)


def test_rotate_about():
    m = _quad_mesh()
    # rotate the quad by pi about its center: p -> 2*c - p
    r = m.rotate_about([0.5, 0.5], [0.0, 0.0, 1.0], np.pi)
    assert np.allclose(r.coords(), 1.0 - m.coords(), atol=1e-9)
    # 90° about the origin pivots (1, 0) onto the y-axis
    r = m.rotate_about([0.0, 0.0], [0.0, 0.0, 1.0], np.pi / 2)
    assert np.allclose(r.coords()[1], [0.0, 1.0], atol=1e-9)
    # Transform value equivalent
    tr = mf.Transform.rotation_about([0.5, 0.5], [0.0, 0.0, 1.0], np.pi)
    assert np.allclose(m.transform(tr).coords(), 1.0 - m.coords(), atol=1e-9)


def test_mirror_about():
    m = _quad_mesh()
    g = m.mirror_about([0.5, 0.5], [1.0, 0.0, 0.0])
    assert np.allclose(g.coords()[:, 0], 1.0 - m.coords()[:, 0])
    tr = mf.Transform.reflection_about([0.5, 0.5], [1.0, 0.0, 0.0])
    assert np.allclose(m.transform(tr).coords(), g.coords())


def test_2d_about_variant_rejects_out_of_plane():
    m = _quad_mesh()
    with np.testing.assert_raises(ValueError):
        m.rotate_about([0.5, 0.5], [1.0, 0.0, 0.0], np.pi / 2)


def test_constructor_errors():
    with np.testing.assert_raises(ValueError):
        mf.Transform.from_matrix(np.eye(3))
    bad = np.eye(4).copy()
    bad[3, 3] = 2.0
    with np.testing.assert_raises(ValueError):
        mf.Transform.from_matrix(bad)
    with np.testing.assert_raises(ValueError):
        mf.Transform.translation([1.0, 2.0, 3.0, 4.0])
    with np.testing.assert_raises(ValueError):
        mf.Transform.rotation([0.0, 0.0, 0.0], 1.0)
    with np.testing.assert_raises(ValueError):
        mf.Transform.reflection([0.0, 0.0, 0.0])
    with np.testing.assert_raises(ValueError):
        mf.Transform.scaling([2.0, 0.0, 1.0]).inverse()


def test_identity_and_apply():
    m = _quad_mesh()
    ident = mf.Transform.identity()
    assert np.allclose(ident.matrix(), np.eye(4))
    assert np.allclose(m.transform(ident).coords(), m.coords())
    tr = mf.Transform.translation([3.0, 0.0])
    assert np.allclose(tr.apply(m).coords(), m.transform(tr).coords())


def test_transform_rejects_wrong_type():
    m = _quad_mesh()
    with np.testing.assert_raises(TypeError):
        m.transform("not a transform")
    # a 3x3 numpy array is a matrix-typed argument that fails the 4x4 contract
    with np.testing.assert_raises(ValueError):
        m.transform(np.eye(3))


def test_1d_translate_and_rotate():
    # a segment mesh is a 1D line; it must stay on the x-axis
    m = mf.UMesh(np.array([[0.0], [1.0], [2.0]]))
    m.add_regular_block("SEG2", np.array([[0, 1], [1, 2]], dtype=np.uint))
    t = m.translate([1.0])
    assert np.isclose(t.coords()[1, 0], 2.0)
    with np.testing.assert_raises(ValueError):
        m.translate([0.0, 1.0])
    # rotation about the line direction is a no-op, rotation about z lifts the line
    r = m.rotate([1.0, 0.0, 0.0], np.pi / 2)
    assert np.allclose(r.coords(), m.coords(), atol=1e-9)
    with np.testing.assert_raises(ValueError):
        m.rotate([0.0, 0.0, 1.0], np.pi / 2)


def test_3d_rotate_about_arbitrary_axis():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    m = mf.UMesh(pts)
    m.add_regular_block("TET4", np.array([[0, 1, 2, 3]], dtype=np.uint))
    r = m.rotate([1.0, 1.0, 0.0], np.pi / 2)
    # (1, 0, 0) -> (0.5, 0.5, -1/sqrt(2)), (0, 1, 0) -> (0.5, 0.5, 1/sqrt(2))
    a = 1.0 / np.sqrt(2.0)
    assert np.allclose(r.coords()[1], [0.5, 0.5, -a], atol=1e-9)
    assert np.allclose(r.coords()[2], [0.5, 0.5, a], atol=1e-9)
    # rigid motion preserves the (positive) volume
    assert np.isclose(r.measure()["TET4"].sum(), 1.0 / 6.0)


def test_mirror_flips_signed_volume():
    m = mf.build_cmesh(range(2), range(2), range(2))
    vol = m.measure()["HEX8"].sum()
    assert vol > 0.0
    mirrored = m.mirror([1.0, 0.0, 0.0])
    assert np.isclose(mirrored.measure()["HEX8"].sum(), -vol)
