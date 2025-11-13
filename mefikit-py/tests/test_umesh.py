import mefikit as mf
import numpy as np
import pytest


@pytest.fixture
def mesh():
    umesh = mf.UMesh(np.arange(30, dtype=float).reshape((10, 3)))
    umesh.add_regular_block("TRI3", np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.uint))
    return umesh


def test_instance(mesh):
    assert isinstance(mesh, mf.UMesh)


def test_print(mesh):
    print(mesh)
    assert mesh.__str__().startswith("""UMeshBase {\n    coords:""")
