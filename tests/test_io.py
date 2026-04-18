def test_to_mc_umesh3(umesh3):
    assert umesh3.to_mc()


def test_to_mc_umesh2(umesh2):
    assert umesh2.to_mc()


def test_to_pv_umesh3(umesh3):
    assert umesh3.to_pyvista()


def test_to_pv_umesh2(umesh2):
    assert umesh2.to_pyvista()
