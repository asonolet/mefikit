import numpy as np

import mefikit as mf


def _quad8_coords():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]
    )


def test_polyze_quad8_roundtrip():
    m = mf.UMesh(_quad8_coords())
    m.add_regular_block("QUAD8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp))

    poly = m.polyze()
    assert poly.block_types() == ["QPOLY"]
    data, offsets = poly.blocks()["QPOLY"]
    assert offsets.tolist() == [8]
    assert data.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]

    back = poly.unpolyze()
    assert back.block_types() == ["QUAD8"]
    conn = back.blocks()["QUAD8"]
    assert conn.tolist() == [[0, 1, 2, 3, 4, 5, 6, 7]]


def test_polyze_tri7_drops_centre():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
            [0.3, 0.3],
        ]
    )
    m = mf.UMesh(coords)
    m.add_regular_block("TRI7", np.array([[0, 1, 2, 3, 4, 5, 6]], dtype=np.uintp))

    poly = m.polyze()
    data, offsets = poly.blocks()["QPOLY"]
    assert offsets.tolist() == [6]
    assert data.tolist() == [0, 1, 2, 3, 4, 5]

    assert poly.unpolyze().block_types() == ["TRI6"]


def test_qpoly_curved_edge_area():
    # Straight mid-side nodes: the QPOLY area equals the flat quad area.
    m = mf.UMesh(_quad8_coords())
    m.add_regular_block("QUAD8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp))
    straight = m.polyze().measure()["QPOLY"]
    assert np.isclose(straight[0], 1.0)

    # Bulge the bottom edge outward: the curved shape has a larger area.
    coords = _quad8_coords()
    coords[4] = [0.5, -0.4]
    m2 = mf.UMesh(coords)
    m2.add_regular_block("QUAD8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp))
    bulged = m2.polyze().measure()["QPOLY"]
    assert bulged[0] > 1.0
    assert np.isclose(
        bulged[0], 1.0 + 0.5 * (0.5125**2) * (2.69880 - 0.42847), rtol=1e-4
    )
