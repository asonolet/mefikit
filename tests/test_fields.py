import numpy as np
import pytest

import mefikit as mf

COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
)
BLOCK = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.uint)


@pytest.fixture()
def quad2():
    mesh = mf.UMesh(COORDS.copy())
    mesh.add_regular_block("QUAD4", BLOCK.copy())
    mesh.set_field("T", {"QUAD4": np.array([[10.0], [20.0]])})
    return mesh


@pytest.fixture()
def quad3():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("QUAD4", np.array([[0, 1, 3, 2]], dtype=np.uint))
    return mesh


# --- creation ---


def test_create_field_from_float(quad2):
    quad2.fields["Z"] = 0.5
    vals = quad2.fields["Z"].values()["QUAD4"]
    assert vals.shape == (2,)
    assert np.allclose(vals, 0.5)


def test_create_field_from_array_1d(quad2):
    quad2.fields["Z"] = np.array([1.0, 2.0])
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [1.0, 2.0])


def test_create_field_from_array_2d(quad2):
    quad2.fields["Z"] = np.array([[1.0], [2.0]])
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [[1.0], [2.0]])


def test_create_field_from_per_block_dict(quad2):
    quad2.fields["Z"] = {"QUAD4": np.array([[3.0], [4.0]])}
    assert np.allclose(quad2.fields["Z"].values()["QUAD4"], [[3.0], [4.0]])


def test_create_field_from_field_expr(quad2):
    quad2.fields["T2"] = mf.Field("T") * 2
    assert np.allclose(quad2.fields["T2"].values()["QUAD4"].ravel(), [20.0, 40.0])


def test_create_field_from_field_name(quad2):
    quad2.fields["copy"] = "T"
    assert np.allclose(quad2.fields["copy"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_create_field_from_unknown_name_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields["X"] = "nope"


# --- mapping protocol ---


def test_fields_mapping_protocol(quad2):
    assert quad2.fields.keys() == ["T"]
    assert [name for name, _ in quad2.fields.items()] == ["T"]
    assert list(iter(quad2.fields)) == ["T"]
    assert len(quad2.fields) == 1
    assert "T" in quad2.fields
    assert "nope" not in quad2.fields


def test_fields_to_dict_matches_values(quad2):
    exported = quad2.fields.to_dict()
    assert set(exported) == {"T"}
    ref = quad2.fields["T"]
    for etype, arr in ref.values().items():
        assert np.array_equal(exported["T"][etype], arr)


def test_rename_missing_raises(quad2):
    with pytest.raises(KeyError):
        quad2.fields.rename("nope", "X")


def test_rename_collision_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields.rename("T", "T")


def test_delete_missing_raises(quad2):
    with pytest.raises(KeyError):
        del quad2.fields["nope"]


# --- field handle ---


def test_ref_metadata(quad2):
    ref = quad2.fields["T"]
    assert tuple(ref.shape) == (1,)
    assert ref.dimension() == 2
    assert len(ref) == 2


def test_ref_values_and_numpy_agree(quad2):
    ref = quad2.fields["T"]
    assert np.array_equal(np.asarray(ref.numpy()), ref.values()["QUAD4"])


def test_getitem_gathers_rows(quad2):
    got = quad2.fields["T"][{"QUAD4": [1]}]
    assert np.allclose(got["QUAD4"], [[20.0]])


def test_setitem_wildcards(quad2):
    quad2.fields["T"][...] = 7.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"], [[7.0], [7.0]])
    quad2.fields["T"][:] = 9.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"], [[9.0], [9.0]])


def test_setitem_with_selection(quad2):
    quad2.fields["T"][mf.sel.ids({"QUAD4": [0]})] = 42.0
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [42.0, 20.0])


def test_setitem_with_field_expr_rhs(quad2):
    quad2.fields["T"][...] = mf.Field("T") + 1
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [11.0, 21.0])


def test_setitem_with_field_name_rhs(quad2):
    quad2.fields["T"][...] = "T"
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])
    quad2.fields["T"][{"QUAD4": [0]}] = "T"
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_setitem_unknown_name_raises(quad2):
    with pytest.raises(ValueError):
        quad2.fields["T"][...] = "nope"


def test_setitem_row_order_regression(quad2):
    quad2.fields["T"][{"QUAD4": [1, 0]}] = np.array([[30.0], [40.0]])
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [40.0, 30.0])


def test_setitem_per_block_rhs(quad2):
    quad2.fields["T"][{"QUAD4": [0]}] = {"QUAD4": np.array([[5.0]])}
    assert np.allclose(quad2.fields["T"].values()["QUAD4"].ravel(), [5.0, 20.0])


def test_ref_reductions(quad2):
    ref = quad2.fields["T"]
    v = np.array([10.0, 20.0])
    assert np.allclose(ref.min(), v.min())
    assert np.allclose(ref.max(), v.max())
    assert np.allclose(ref.sum(), v.sum())
    assert np.allclose(ref.mean(), v.mean())
    assert np.allclose(ref.var(), v.var())
    assert np.allclose(ref.var(ddof=1), v.var(ddof=1))
    assert np.allclose(ref.std(ddof=1), v.std(ddof=1))


def test_ref_integral(quad2):
    # both quads have measure 1.0 -> integral(T) = 10 * 1 + 20 * 1
    assert np.allclose(quad2.fields["T"].integral(), 30.0)


# --- lazy selections ---


def test_select_ids_len_repr(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    ids = dict(result.ids())
    assert ids["QUAD4"].tolist() == [0, 1]
    assert len(result) == 2
    assert repr(result)


def test_select_reductions(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    assert np.allclose(result.min("T"), [10.0])
    assert np.allclose(result.mean(mf.Field("T") * 2), [30.0])
    assert np.allclose(result.sum("T"), [30.0])
    assert np.allclose(result.var(mf.Field("T"), 1), [50.0])
    assert np.allclose(result.std(mf.Field("T"), 1), [np.sqrt(50.0)])


def test_select_unknown_field_raises(quad2):
    result = quad2.select(mf.sel.all())
    with pytest.raises(ValueError):
        result.mean("nope")
    with pytest.raises(ValueError):
        result.integral("nope")


def test_select_integral(quad2):
    result = quad2.select(mf.sel.ids({"QUAD4": [1]}))
    assert np.allclose(result.integral("T"), 20.0)


def test_select_to_mesh_fields(quad2):
    result = quad2.select(mf.sel.rect([0.5, -1.0], [3.0, 2.0]))
    bare = result.to_mesh(with_fields=False)
    assert bare.num_elements() == 2
    assert bare.fields.keys() == []
    carried = result.to_mesh()
    assert carried.fields.keys() == ["T"]
    assert np.allclose(carried.fields["T"].values()["QUAD4"].ravel(), [10.0, 20.0])


def test_sel_all(quad2):
    assert len(quad2.select(mf.sel.all())) == 2


# --- expressions: pow / index ---


def test_field_pow_eval(quad2):
    quad2.fields["P"] = mf.Field("T") ** 2
    assert np.allclose(quad2.fields["P"].values()["QUAD4"].ravel(), [100.0, 400.0])


def test_field_getitem_selects_component(quad2):
    # T is a [n, 1] field; selecting component 0 yields a [n] field.
    quad2.set_field("V", {"QUAD4": np.array([[1.0, 2.0], [3.0, 4.0]])})
    quad2.fields["v0"] = mf.Field("V")[0]
    assert np.allclose(quad2.fields["v0"].values()["QUAD4"].ravel(), [1.0, 3.0])


# --- expressions: matmul / dot ---


def test_field_matmul_vector_dot(quad2):
    # [n, 3] @ [3] -> [n, 1]
    quad2.set_field("W", {"QUAD4": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])})
    quad2.fields["d"] = mf.Field("W").dot([1.0, 2.0, 3.0])
    assert np.allclose(quad2.fields["d"].values()["QUAD4"].ravel(), [14.0, 32.0])


def test_field_matmul_matrix_times_matrix(quad2):
    ident = np.eye(3)
    quad2.set_field("M", {"QUAD4": np.tile(ident, (2, 1, 1))})
    quad2.fields["prod"] = mf.Field("M") @ mf.Field("M")
    out = quad2.fields["prod"].values()["QUAD4"]
    assert out.shape == (2, 3, 3)
    assert np.allclose(out, np.tile(ident, (2, 1, 1)))


# --- surface normals ---


def test_normal_module_fields(quad3):
    quad3.fields["n"] = mf.Normal
    n = quad3.fields["n"].values()["QUAD4"]
    assert n.shape == (1, 3)
    assert np.allclose(n, [[0.0, 0.0, 1.0]])

    quad3.fields["nz"] = mf.Nz
    assert np.allclose(quad3.fields["nz"].values()["QUAD4"].ravel(), [1.0])


def test_normal_component_expression(quad3):
    quad3.fields["nx"] = mf.Nx
    quad3.fields["ny"] = mf.Ny
    assert np.allclose(quad3.fields["nx"].values()["QUAD4"].ravel(), [0.0])
    assert np.allclose(quad3.fields["ny"].values()["QUAD4"].ravel(), [0.0])


def test_normal_on_boundary_not_volume():
    # A mesh holding both a HEX8 volume and a QUAD4 boundary face must yield normals
    # on the QUAD4 (space_dim - 1) block, never on the HEX8 volume.
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint))
    mesh.add_regular_block("QUAD4", np.array([[4, 5, 6, 7]], dtype=np.uint))

    mesh.fields["n"] = mf.Normal
    values = mesh.fields["n"].values()
    assert "HEX8" not in values
    assert np.allclose(values["QUAD4"], [[0.0, 0.0, 1.0]])

    mesh.fields["nz"] = mf.Nz
    assert "HEX8" not in mesh.fields["nz"].values()
    assert np.allclose(mesh.fields["nz"].values()["QUAD4"].ravel(), [1.0])


def hex_quad_mesh():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    mesh = mf.UMesh(coords)
    mesh.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint))
    mesh.add_regular_block("QUAD4", np.array([[4, 5, 6, 7]], dtype=np.uint))
    return mesh


def test_normal_times_boundary_field_infers_dim():
    # Normal combined with a field stored on the QUAD4 boundary must infer the
    # hypersurface dimension (space_dim - 1) and evaluate both on QUAD4.
    mesh = hex_quad_mesh()
    mesh.fields["quad_val"] = {"QUAD4": np.array([2.0])}
    mesh.fields["flux"] = mf.Normal * mf.Field("quad_val")
    values = mesh.fields["flux"].values()
    assert "HEX8" not in values
    assert np.allclose(values["QUAD4"], [[0.0, 0.0, 2.0]])


def test_normal_times_volume_field_mixed_dim_errors():
    # Normal (hypersurface) combined with a volume field cannot be evaluated at a single
    # dimension, so assigning must raise.
    mesh = hex_quad_mesh()
    mesh.fields["vol_val"] = {"HEX8": np.array([3.0])}
    with pytest.raises(BaseException):
        mesh.fields["flux"] = mf.Normal * mf.Field("vol_val")


def test_eval_explicit_dim_is_strict():
    # An explicit dim=2 targets the QUAD4 block strictly.
    mesh = hex_quad_mesh()
    mesh.fields["quad_val"] = {"QUAD4": np.array([2.0])}
    res = mesh.eval(mf.Nz * mf.Field("quad_val"), dim=2)
    assert "QUAD4" in res and "HEX8" not in res
    assert np.allclose(res["QUAD4"].ravel(), [2.0])
