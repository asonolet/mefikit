import numpy as np

import mefikit as mf


def test_instance(umesh3):
    assert isinstance(umesh3, mf.UMesh)


def test_print(umesh3):
    print(umesh3)
    assert str(umesh3).startswith("""UMeshBase {\n    coords:""")


def test_reorient_hexa_fixes_reversed_axis():
    m = mf.build_cmesh(range(2), range(2), range(2))
    assert np.isclose(m.measure()["HEX8"].sum(), 1.0)

    # Reversing one axis mirrors every cell (negative signed volume).
    mirrored = mf.build_cmesh(range(2), list(reversed(range(2))), range(2))
    assert mirrored.measure()["HEX8"].sum() < 0.0

    fixed = mirrored.reorient()
    assert np.isclose(fixed.measure()["HEX8"].sum(), 1.0)

    # A canonical mesh is unchanged by reorienting.
    fixed_again = m.reorient()
    assert np.isclose(fixed_again.measure()["HEX8"].sum(), 1.0)


def test_reorient_tet4_reversed_winding_is_fixed():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    m = mf.UMesh(pts)
    m.add_regular_block("TET4", np.array([[0, 2, 1, 3]], dtype=np.uint))
    assert m.measure()["TET4"].sum() < 0.0

    fixed = m.reorient()
    assert np.isclose(fixed.measure()["TET4"].sum(), 1.0 / 6.0)


def test_reorient_2d_smoke():
    # Signed 2D measures are not exposed to Python, so this checks that reorienting a
    # counter-clockwise quad mesh runs, preserves the area measure and is idempotent.
    # (Signed 2D winding is verified in the Rust unit tests.)
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    m = mf.UMesh(coords)
    m.add_regular_block("QUAD4", np.array([[0, 3, 2, 1]], dtype=np.uint))
    fixed = m.reorient()
    assert np.isclose(fixed.measure()["QUAD4"].sum(), 1.0)
    once = m.reorient()
    assert once.measure()["QUAD4"].sum() == once.reorient().measure()["QUAD4"].sum()


def test_reorient_preserves_groups_and_fields():
    m = mf.build_cmesh(range(2), list(reversed(range(2))), range(2))
    m.set_field("temperature", {"HEX8": np.array([21.5])})
    m.groups["heated"] = mf.sel.ids({"HEX8": [0]})
    assert m.measure()["HEX8"].sum() < 0.0

    fixed = m.reorient()
    assert fixed.measure()["HEX8"].sum() > 0.0
    assert np.isclose(fixed.measure()["HEX8"].sum(), 1.0)
    ids = dict(fixed.groups["heated"].ids())
    assert "HEX8" in ids and ids["HEX8"][0] == 0
    assert np.allclose(fixed.fields["temperature"].values()["HEX8"], [21.5])
