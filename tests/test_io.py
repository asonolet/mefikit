import numpy as np

import mefikit as mf


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
