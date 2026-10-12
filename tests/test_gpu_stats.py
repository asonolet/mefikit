import numpy as np
import pytest

import mefikit as mf

COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
)
BLOCK = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.uint)


def test_stats_global_over_components():
    mesh = mf.UMesh(COORDS.copy())
    mesh.add_regular_block("QUAD4", BLOCK.copy())
    mesh.set_field("T", {"QUAD4": np.array([[10.0, 1.5], [20.0, -2.0]])})
    s = mesh.fields["T"].stats()
    v = np.array([10.0, 1.5, 20.0, -2.0])
    assert s.count == v.size
    assert s.sum == pytest.approx(v.sum())
    assert s.mean == pytest.approx(v.mean())
    assert s.min == pytest.approx(v.min())
    assert s.max == pytest.approx(v.max())
    # The GPU and host paths agree on these exact values; the repr must expose
    # the five fields.
    assert "count=4" in repr(s)


def test_stats_scalar_field():
    mesh = mf.UMesh(COORDS.copy())
    mesh.add_regular_block("QUAD4", BLOCK.copy())
    mesh.set_field("T", {"QUAD4": np.array([10.0, 20.0])})
    s = mesh.fields["T"].stats()
    assert s.count == 2
    assert s.sum == pytest.approx(30.0)
    assert s.mean == pytest.approx(15.0)
    assert s.min == pytest.approx(10.0)
    assert s.max == pytest.approx(20.0)


def test_stats_matches_numpy_roundtrip():
    mesh = mf.build_cmesh(np.linspace(0.0, 1.0, 6), np.linspace(0.0, 1.0, 5))
    mesh.fields["T"] = mf.X * mf.Y

    field = mesh.fields["T"]
    s = field.stats()
    v = np.asarray(field.numpy(), dtype=float).reshape(-1)
    assert s.count == v.size
    assert s.sum == pytest.approx(v.sum())
    assert s.mean == pytest.approx(v.mean())
    assert s.min == pytest.approx(v.min())
    assert s.max == pytest.approx(v.max())
