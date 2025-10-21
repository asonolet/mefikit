import mefikit as mf
import numpy as np
import pytest


@pytest.fixture
def mesh():
    return mf.UMesh(np.arange(30, dtype=float).reshape((10, 3)))


def test_print(mesh):
    a = np.arange(30).reshape((10, 3))
    a_str = np.array2string(
        a, separator=", ", formatter={"float_kind": lambda x: f"{x:.17g}"}
    ).replace(" ", "")
    assert mesh.__str__().replace(" ", "") == f"UMesh:\n======\ncoords\n{a_str}"
