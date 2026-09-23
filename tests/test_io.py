import numpy as np
import pyvista as pv

import mefikit as mf


def phed_mesh():
    mesh = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), [0.0, 1.0]
    )
    return mesh.polyze()


def test_to_pv_phed():
    ph = phed_mesh()
    ug = ph.to_pyvista()
    assert ug.n_cells == ph.num_elements()
    assert np.all(ug.celltypes == int(pv.CellType.POLYHEDRON))
    for i in range(ug.n_cells):
        cell = ug.get_cell(i)
        assert cell.type == pv.CellType.POLYHEDRON
        assert cell.n_faces == 6
        assert cell.n_points == 8


def test_to_pv_phed_roundtrip():
    ph = phed_mesh()
    data, offsets = ph.blocks()["PHED"]
    ug = ph.to_pyvista()

    max_id = np.iinfo(np.uintp).max
    rebuilt = []
    rebuilt_offsets = []
    for i in range(ug.n_cells):
        cell = ug.get_cell(i)
        for j, face in enumerate(cell.faces):
            rebuilt.extend(face.point_ids)
            if j < cell.n_faces - 1:
                rebuilt.append(max_id)
        rebuilt_offsets.append(len(rebuilt))

    assert np.array_equal(np.array(rebuilt_offsets, dtype=np.uintp), offsets)
    assert np.array_equal(np.array(rebuilt, dtype=np.uintp), data)


def test_to_pv_phed_fields():
    ph = phed_mesh()
    ph.set_field("Heat", {"PHED": np.linspace(0.0, 1.0, ph.num_elements())})
    ug = ph.to_pyvista()
    assert np.allclose(ug.cell_data["Heat"], np.linspace(0.0, 1.0, ph.num_elements()))


def test_to_pv_pgon():
    m = mf.build_cmesh(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)).polyze()
    ug = m.to_pyvista()
    assert ug.n_cells == m.num_elements()
    assert np.all(ug.celltypes == int(pv.CellType.POLYGON))
    for i in range(ug.n_cells):
        cell = ug.get_cell(i)
        assert cell.n_points == 4


def test_to_pv_pgon_roundtrip():
    m = mf.build_cmesh(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4)).polyze()
    data, offsets = m.blocks()["PGON"]
    ug = m.to_pyvista()

    pos = 0
    rebuilt = []
    rebuilt_offsets = []
    while pos < ug.cells.size:
        n = int(ug.cells[pos])
        rebuilt.extend(ug.cells[pos + 1 : pos + 1 + n])
        rebuilt_offsets.append(len(rebuilt))
        pos += 1 + n

    assert np.array_equal(np.array(rebuilt, dtype=np.uintp), data)
    assert np.array_equal(np.array(rebuilt_offsets, dtype=np.uintp), offsets)


def test_to_mc_umesh3(umesh3):
    assert umesh3.to_mc()


def test_to_mc_umesh2(umesh2):
    assert umesh2.to_mc()


def test_to_pv_umesh3(umesh3):
    assert umesh3.to_pyvista()


def test_to_pv_umesh2(umesh2):
    assert umesh2.to_pyvista()


def test_to_mc_keeps_every_cell():
    m2 = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    mm2 = m2.to_mc()
    assert mm2.getNumberOfCells() == 16
    assert mm2.getNumberOfNodes() == m2.coords().shape[0]

    m3 = mf.build_cmesh(*(np.linspace(0.0, 1.0, 5),) * 3)
    mm3 = m3.to_mc()
    assert mm3.getNumberOfCells() == 64
    assert mm3.getNumberOfNodes() == m3.coords().shape[0]
