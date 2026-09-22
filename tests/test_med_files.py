import numpy as np
import pytest

import mefikit as mf

mc = pytest.importorskip("medcoupling")

_CODE_TO_TYPE = {
    0: "VERTEX",
    1: "SEG2",
    2: "SEG3",
    3: "TRI3",
    6: "TRI6",
    4: "QUAD4",
    8: "QUAD8",
    14: "TET4",
    20: "TET10",
    18: "HEX8",
    5: "PGON",
    31: "PHED",
}


def _make_mixed_mesh():
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
            [0.5, 0.5, 0.5],
        ]
    )
    m = mf.UMesh(coords)
    m.add_regular_block(
        "HEX8",
        np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp),
        fields={"T": np.array([30.0])},
    )
    m.add_regular_block(
        "TET4",
        np.array(
            [[0, 1, 2, 8], [1, 2, 6, 8], [2, 3, 6, 8], [3, 0, 6, 8]], dtype=np.uintp
        ),
        fields={"T": np.array([10.0, 20.0, 40.0, 50.0])},
    )
    m.add_regular_block(
        "QUAD4",
        np.array([[0, 1, 2, 3]], dtype=np.uintp),
        fields={"V": np.array([[1.0, 2.0, 3.0]])},
    )
    m.add_regular_block(
        "TRI3",
        np.array([[4, 5, 6]], dtype=np.uintp),
        fields={"V": np.array([[4.0, 5.0, 6.0]])},
    )
    m.add_regular_block(
        "SEG2", np.array([[0, 1]], dtype=np.uintp), fields={"s1": np.array([7.0])}
    )
    m.add_regular_block(
        "VERTEX", np.array([[8]], dtype=np.uintp), fields={"p0": np.array([9.0])}
    )
    return m


def _medcoupling_field_values(path):
    """Read a mefikit MED file with medcoupling.

    Returns ``{field_name: {element_type: np.ndarray}}`` where per-type values
    are re-extracted from the field tuples ordering medcoupling uses on read.
    """
    fum = mc.MEDFileUMesh(path)
    ffields = mc.MEDFileFields(path)
    fields = {}
    for name in ffields.getFieldsNames():
        mts = ffields.getFieldWithName(name)
        f = mts.field(1, 1, fum)
        vals = f.getArray().toNumPyArray()
        n_comp = f.getNumberOfComponents()
        per_type = {}
        for code, ranges in mts.getFieldSplitedByType(1, 1):
            chunks = [
                np.asarray(vals[b:e]).reshape(-1, n_comp)
                for (_ent, (b, e), *_r) in ranges
            ]
            per_type[_CODE_TO_TYPE[int(code)]] = np.vstack(chunks)
        fields[name] = per_type
    return fields


def test_med_fields_readable_by_medcoupling(tmp_path):
    path = str(tmp_path / "fields.med")
    m = _make_mixed_mesh()
    m.write(path)

    fields = _medcoupling_field_values(path)
    assert set(fields) == {"T", "V", "s1", "p0"}

    for name in ("T", "V", "s1", "p0"):
        for et, values in m.fields[name].values().items():
            got = np.asarray(fields[name][et]).reshape(-1)
            want = np.asarray(values).reshape(-1)
            assert np.allclose(got, want), f"{name}[{et}]"


def test_med_field_nom_attr_is_fixed_string(tmp_path):
    import h5py

    path = str(tmp_path / "nom.med")
    m = mf.UMesh(np.arange(30.0).reshape(10, 3))
    m.add_regular_block(
        "TET4",
        np.array([[0, 1, 2, 6], [1, 2, 3, 7]], dtype=np.uintp),
        fields={"T": np.array([1.0, 2.0])},
    )
    m.add_regular_block(
        "QUAD4",
        np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.uintp),
        fields={"V": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])},
    )
    m.write(path)

    with h5py.File(path) as h:
        cha = h["CHA"]
        t = cha["T"]
        v = cha["V"]
        # One fixed-length ASCII string of 16 * n_components bytes per field.
        assert t.attrs["NOM"].dtype == h5py.string_dtype("ascii", 16)
        assert t.attrs["NOM"].shape == ()
        assert v.attrs["NOM"].dtype == h5py.string_dtype("ascii", 48)
        assert v.attrs["NOM"].shape == ()
        assert t.attrs["NOM"].decode() == "T" + " " * 15
        assert (
            v.attrs["NOM"].decode()
            == "V1" + " " * 14 + "V2" + " " * 14 + "V3" + " " * 14
        )


def test_med_fields_roundtrip_via_mefikit(tmp_path):
    path = str(tmp_path / "roundtrip.med")
    m = _make_mixed_mesh()
    m.write(path)
    m2 = mf.UMesh.read(path)
    for name in ("T", "V", "s1", "p0"):
        for et, values in m.fields[name].values().items():
            assert np.array_equal(
                np.asarray(m2.fields[name].values()[et]), np.asarray(values)
            )


def test_med_groups_readable_by_medcoupling(tmp_path):
    """Element groups must be readable by the MED library (medcoupling).

    MED stores the group names of each family in FAS/mesh/ELEME/<FAM>/GRO/NOM
    as an H5T_ARRAY of 80 signed bytes per name; medcoupling rejects other
    layouts with an HDF5 conversion error.
    """
    path = str(tmp_path / "groups.med")
    m = mf.UMesh(np.arange(30.0).reshape(10, 3))
    m.add_regular_block(
        "QUAD4",
        np.array(
            [[0, 1, 2, 3], [1, 2, 3, 4], [4, 5, 6, 7], [5, 6, 7, 8]], dtype=np.uintp
        ),
        fields={"T": np.array([1.0, 2.0, 4.0, 8.0])},
    )
    m.groups["mygroup"] = {"QUAD4": np.array([1, 3])}
    m.write(path)

    fum = mc.MEDFileUMesh(path)
    levels = list(fum.getNonEmptyLevelsExt())
    matched = False
    for lev in levels:
        gs = list(fum.getGroupsOnSpecifiedLev(lev))
        if "mygroup" in gs:
            arr = fum.getGroupArr(lev, "mygroup", False).toNumPyArray().ravel()
            assert sorted(arr.tolist()) == [1, 3]
            matched = True
    assert matched, f"group 'mygroup' missing on levels {levels}"


def test_med_groups_roundtrip_via_mefikit(tmp_path):
    path = str(tmp_path / "groups_rt.med")
    m = mf.UMesh(np.arange(30.0).reshape(10, 3))
    m.add_regular_block(
        "QUAD4",
        np.array(
            [[0, 1, 2, 3], [1, 2, 3, 4], [4, 5, 6, 7], [5, 6, 7, 8]], dtype=np.uintp
        ),
    )
    m.groups["a"] = {"QUAD4": np.array([0, 1])}
    m.groups["b"] = {"QUAD4": np.array([1, 2, 3])}
    m.write(path)
    m2 = mf.UMesh.read(path)
    assert set(m2.groups) == {"a", "b"}
    assert sorted(m2.groups["a"].ids()["QUAD4"]) == [0, 1]
    assert sorted(m2.groups["b"].ids()["QUAD4"]) == [1, 2, 3]


def test_med_groups_nom_is_int8_array_dataset(tmp_path):
    """On-disk GRO/NOM must be the exact layout MED uses (1-D dataset whose
    element type is a fixed array of 80 int8), otherwise the MED reader
    cannot convert it while reading."""
    import h5py

    path = str(tmp_path / "groups_nom.med")
    m = mf.UMesh(np.arange(30.0).reshape(10, 3))
    m.add_regular_block(
        "QUAD4",
        np.array(
            [[0, 1, 2, 3], [1, 2, 3, 4], [4, 5, 6, 7], [5, 6, 7, 8]], dtype=np.uintp
        ),
    )
    m.groups["mygroup"] = {"QUAD4": np.array([1, 3])}
    m.write(path)

    with h5py.File(path) as h:
        d = h["/FAS/mesh/ELEME/FAM_1_/GRO/NOM"]
        assert d.dtype == np.dtype(("i1", (80,))), d.dtype
        assert d.shape == (1,)
        assert bytes(d[0]) == b"mygroup" + b" " * 73
