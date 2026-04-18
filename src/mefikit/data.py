import mefikit as mf
import numpy as np


def umesh3():
    umesh = mf.UMesh(np.arange(30, dtype=float).reshape((10, 3)))
    umesh.add_regular_block(
        "TRI3", np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=np.uint)
    )
    umesh.add_regular_block(
        "QUAD4", np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint)
    )
    umesh.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint))
    return umesh


def cmesh3():
    umesh = mf.build_cmesh(
        np.linspace(0.0, 0.5, 10), np.linspace(0.0, 1.0, 20), range(3)
    )
    smesh2 = umesh.descend()
    for et, block in smesh2.blocks().items():
        umesh.add_regular_block(et, block)
    smesh1 = smesh2.descend()
    for et, block in smesh1.blocks().items():
        umesh.add_regular_block(et, block)
    return umesh


def umesh2():
    umesh = mf.build_cmesh(range(5), np.linspace(0.0, 1.0))
    return umesh
