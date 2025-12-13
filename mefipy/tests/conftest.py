import pytest

import mefikit as mf
import numpy as np


@pytest.fixture
def umesh3():
    umesh = mf.UMesh(np.arange(30, dtype=float).reshape((10, 3)))
    umesh.add_regular_block(
        "TRI3", np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.uint)
    )
    umesh.add_regular_block(
        "QUAD4", np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.uint)
    )
    umesh.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint))
    return umesh


@pytest.fixture
def umesh2():
    umesh = mf.build_cmesh(range(5), np.linspace(0.0, 1.0))
    return umesh
