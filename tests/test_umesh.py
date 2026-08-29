import mefikit as mf


def test_instance(umesh3):
    assert isinstance(umesh3, mf.UMesh)


def test_print(umesh3):
    print(umesh3)
    assert str(umesh3).startswith("""UMeshBase {\n    coords:""")
