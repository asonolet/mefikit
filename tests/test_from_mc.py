import os

import numpy as np
import pytest

import mefikit as mf

mc = pytest.importorskip("medcoupling")


def _mc_mesh(coords, cells, dim, name="mesh"):
    conn = []
    for c in cells:
        conn.extend(c)
    arr = mc.DataArrayInt(conn)
    arrI = mc.DataArrayInt(np.cumsum([0] + [len(c) for c in cells]).tolist())
    m = mc.MEDCouplingUMesh(name, dim)
    m.setCoords(mc.DataArrayDouble(coords.tolist()))
    m.setConnectivity(arr, arrI)
    return m


def _mc_field(name, mesh, values):
    f = mc.MEDCouplingFieldDouble(mc.ON_CELLS)
    f.setName(name)
    f.setArray(mc.DataArrayDouble(values))
    f.setMesh(mesh)
    return f


@pytest.mark.parametrize(
    "et,nn,dim",
    [
        ("VERTEX", 1, 1),
        ("SEG2", 2, 1),
        ("SEG3", 3, 1),
        ("TRI3", 3, 2),
        ("TRI6", 6, 2),
        ("QUAD4", 4, 2),
        ("QUAD8", 8, 2),
        ("TET4", 4, 3),
        ("TET10", 10, 3),
        ("HEX8", 8, 3),
    ],
)
def test_from_mc_roundtrip_static_types(et, nn, dim):
    coords = np.arange(nn * dim, dtype=float).reshape(nn, dim)
    m = mf.UMesh(coords)
    m.add_regular_block(et, np.arange(nn).reshape(1, -1).astype(np.uintp))
    mc_mesh = m.to_mc()
    res = mf.UMesh.from_mc(mc_mesh)
    assert res.block_types() == [et]
    assert np.array_equal(res.blocks()[et], m.blocks()[et])
    assert np.array_equal(res.coords(), coords)


def test_from_mc_hex8_permutation():
    coords = np.arange(24.0).reshape(8, 3)
    m = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    res = mf.UMesh.from_mc(m)
    assert np.array_equal(res.blocks()["HEX8"], [[0, 1, 2, 3, 4, 5, 6, 7]])


def test_from_mc_tet4_permutation():
    coords = np.arange(12.0).reshape(4, 3)
    m = _mc_mesh(coords, [[14, 0, 1, 3, 2]], 3)
    res = mf.UMesh.from_mc(m)
    assert np.array_equal(res.blocks()["TET4"], [[0, 1, 2, 3]])


def test_from_mc_polygon():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    m = _mc_mesh(coords, [[5, 0, 1, 2], [5, 1, 3, 2]], 2)
    res = mf.UMesh.from_mc(m)
    data, offsets = res.blocks()["PGON"]
    assert np.array_equal(data, [0, 1, 2, 1, 3, 2])
    assert np.array_equal(offsets, [3, 6])


def test_from_mc_polyhed():
    coords = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    m = _mc_mesh(
        coords,
        [
            [
                31,
                0,
                2,
                1,
                -1,
                1,
                3,
                0,
                -1,
                2,
                0,
                3,
                -1,
                3,
                1,
                2,
            ]
        ],
        3,
    )
    res = mf.UMesh.from_mc(m)
    data, offsets = res.blocks()["PHED"]
    last = np.iinfo(np.uintp).max
    assert np.array_equal(
        data,
        [0, 2, 1, last, 1, 3, 0, last, 2, 0, 3, last, 3, 1, 2, last],
    )
    assert np.array_equal(offsets, [16])


def test_from_mc_medfile_levels_and_groups():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    lev2 = _mc_mesh(coords, [[4, 0, 1, 2, 3]], 2)
    lev2.tryToShareSameCoords(lev3, 1e-12)
    fum = mc.MEDFileUMesh()
    fum.setMeshAtLevel(0, lev3)
    fum.setMeshAtLevel(-1, lev2)
    g3 = mc.DataArrayInt([0])
    g3.setName("g3")
    g2 = mc.DataArrayInt([0])
    g2.setName("g2")
    fum.addGroup(0, g3)
    fum.addGroup(-1, g2)
    res = mf.UMesh.from_mc(fum)
    assert res.block_types() == ["QUAD4", "HEX8"]
    assert np.array_equal(res.blocks()["HEX8"], [[0, 1, 2, 3, 4, 5, 6, 7]])
    assert np.array_equal(res.blocks()["QUAD4"], [[0, 1, 2, 3]])
    for grp in ("g3", "g2"):
        assert grp in res.groups
    assert set(res.groups["g3"].ids()) == {"HEX8"}
    assert set(res.groups["g2"].ids()) == {"QUAD4"}
    assert np.array_equal(res.groups["g3"].ids()["HEX8"], [0])
    assert np.array_equal(res.groups["g2"].ids()["QUAD4"], [0])


def test_from_mc_group_spanning_levels():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    lev2 = _mc_mesh(coords, [[4, 0, 1, 2, 3]], 2)
    lev2.tryToShareSameCoords(lev3, 1e-12)
    fum = mc.MEDFileUMesh()
    fum.setMeshAtLevel(0, lev3)
    fum.setMeshAtLevel(-1, lev2)
    g3 = mc.DataArrayInt([0])
    g3.setName("shared")
    g2 = mc.DataArrayInt([0])
    g2.setName("shared")
    fum.addGroup(0, g3)
    fum.addGroup(-1, g2)
    res = mf.UMesh.from_mc(fum)
    ids = res.groups["shared"].ids()
    assert set(ids) == {"HEX8", "QUAD4"}
    assert np.array_equal(ids["HEX8"], [0])
    assert np.array_equal(ids["QUAD4"], [0])


def test_from_mc_fields_mixed_types_same_level():
    coords = np.arange(24.0).reshape(8, 3)
    m = mf.UMesh(coords)
    m.add_regular_block("TET4", np.array([[0, 1, 2, 3]], dtype=np.uintp))
    m.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp))
    mc_mesh = m.to_mc()
    f = _mc_field("T", mc_mesh, [10.0, 20.0])
    res = mf.UMesh.from_mc(mc_mesh, fields=[f])
    assert set(res.fields["T"].values()) == {"TET4", "HEX8"}
    assert np.array_equal(res.fields["T"].values()["TET4"], [[10.0]])
    assert np.array_equal(res.fields["T"].values()["HEX8"], [[20.0]])


def test_from_mc_fields_on_medfile_level():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    fum = mc.MEDFileUMesh()
    fum.setMeshAtLevel(0, lev3)
    level0 = fum.getMeshAtLevel(0, False)
    f = _mc_field("F", level0, [42.0])
    res = mf.UMesh.from_mc(fum, fields=[f])
    assert np.array_equal(res.fields["F"].values()["HEX8"], [[42.0]])


def test_from_mc_field_on_foreign_mesh_raises():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    other = _mc_mesh(coords, [[14, 0, 1, 2]], 3, name="other")
    f = _mc_field("G", other, [1.0])
    with pytest.raises(ValueError, match="not attached"):
        mf.UMesh.from_mc(lev3, fields=[f])


def test_from_mc_on_nodes_field_raises():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    f = mc.MEDCouplingFieldDouble(mc.ON_NODES)
    f.setName("nodal")
    f.setArray(mc.DataArrayDouble([1.0] * 8))
    f.setMesh(lev3)
    with pytest.raises(ValueError, match="ON_CELLS"):
        mf.UMesh.from_mc(lev3, fields=[f])


def test_from_mc_node_group_skipped_with_warning():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    fum = mc.MEDFileUMesh()
    fum.setMeshAtLevel(0, lev3)
    ng = mc.DataArrayInt([0, 1, 2])
    ng.setName("nodes")
    fum.addNodeGroup(ng)
    with pytest.warns(UserWarning, match="Node group 'nodes'"):
        res = mf.UMesh.from_mc(fum)
    assert "nodes" not in res.groups


def test_from_mc_unsupported_cell_type_skipped_with_warning():
    coords = np.arange(24.0).reshape(8, 3)
    m = _mc_mesh(coords, [[15, 0, 1, 2, 3, 4]], 3)
    with pytest.warns(UserWarning, match=r"code\(s\) 15"):
        res = mf.UMesh.from_mc(m)
    assert res.block_types() == []


def test_from_mc_single_field_input():
    coords = np.arange(24.0).reshape(8, 3)
    lev3 = _mc_mesh(coords, [[18, 4, 5, 6, 7, 0, 1, 2, 3]], 3)
    f = _mc_field("F", lev3, [42.0])
    res = mf.UMesh.from_mc(lev3, fields=f)
    assert "F" in res.fields


def test_from_mc_type_error():
    with pytest.raises(TypeError):
        mf.UMesh.from_mc(3)


def test_from_mc_build_cmesh_roundtrip():
    m = mf.build_cmesh(*(np.linspace(0.0, 1.0, 21),) * 2)
    mc_mesh = m.to_mc()
    res = mf.UMesh.from_mc(mc_mesh)
    assert res.block_types() == m.block_types()
    for et in m.block_types():
        assert np.array_equal(res.blocks()[et], m.blocks()[et])
    assert np.array_equal(res.coords(), m.coords())


def test_from_mc_read_med_file(tmp_path):
    coords = np.arange(30.0).reshape(10, 3)
    m = mf.UMesh(coords)
    m.add_regular_block("TET4", np.array([[0, 1, 2, 6], [1, 2, 3, 7]], dtype=np.uintp))
    mc_mesh = m.to_mc()
    path = os.path.join(tmp_path, "m.med")
    fum = mc.MEDFileUMesh()
    fum.setMeshAtLevel(0, mc_mesh)
    fum.write(path, 2)
    fum2 = mc.MEDFileUMesh(path)
    res = mf.UMesh.from_mc(fum2)
    assert res.block_types() == ["TET4"]
    assert np.array_equal(res.blocks()["TET4"], m.blocks()["TET4"])
    assert np.array_equal(res.coords(), coords)
