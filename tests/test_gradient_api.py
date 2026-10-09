from __future__ import annotations

import numpy as np
import pytest

import mefikit as mf


def affine_2d_source():
    """4x4 QUAD4 grid carrying the scalar field f = 1 + 2x - 3y."""
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    src.fields["T"] = 1.0 + 2.0 * mf.X - 3.0 * mf.Y
    return src


EXPECTED = np.array([2.0, -3.0])


@pytest.mark.parametrize(
    "weighting",
    [
        mf.DistanceWeighting.Constant(),
        mf.DistanceWeighting.InverseDistance(2.0),
        mf.DistanceWeighting.CompactSupport(2.0),
        mf.DistanceWeighting.Gaussian(),
    ],
)
def test_on_source_gradient_of_affine_field(weighting):
    src = affine_2d_source()
    grad = mf.Gradient(src, k=9, weighting=weighting)
    out = grad.eval("T")
    assert set(out) == {"QUAD4"}
    vals = np.asarray(out["QUAD4"])
    assert vals.shape == (16, 2)
    np.testing.assert_allclose(vals, np.broadcast_to(EXPECTED, vals.shape), atol=1e-7)


def test_gradient_on_target_mesh():
    src = affine_2d_source()
    tgt = mf.build_cmesh(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))
    grad = mf.Gradient(src, tgt, k=9)
    vals = np.asarray(grad.eval("T")["QUAD4"])
    assert vals.shape == (tgt.num_elements(), 2)
    np.testing.assert_allclose(vals, np.broadcast_to(EXPECTED, vals.shape), atol=1e-7)


def test_lazy_components_of_gradient_field():
    src = affine_2d_source()
    tgt = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    grad = mf.Gradient(src, k=9)
    tgt.fields["gx"] = grad("T")[0]
    tgt.fields["gy"] = grad("T")[1]
    np.testing.assert_allclose(tgt.fields["gx"].numpy(), 2.0, atol=1e-7)
    np.testing.assert_allclose(tgt.fields["gy"].numpy(), -3.0, atol=1e-7)


def test_apply_update_stores_vector_field():
    src = affine_2d_source()
    tgt = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    grad = mf.Gradient(src, k=9)
    grad.apply_update(src, "T", tgt, "grad")
    vals = tgt.fields["grad"].numpy()
    assert vals.shape == (16, 2)
    np.testing.assert_allclose(vals, np.broadcast_to(EXPECTED, vals.shape), atol=1e-7)


def test_at_points_gradient_of_affine_field():
    src = affine_2d_source()
    points = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])
    grad = mf.Gradient.at_points(src, points, k=9)
    vals = grad.eval_points("T")
    assert vals.shape == (3, 2)
    np.testing.assert_allclose(vals, np.broadcast_to(EXPECTED, vals.shape), atol=1e-7)


def test_at_points_rejects_cell_evaluation():
    src = affine_2d_source()
    grad = mf.Gradient.at_points(src, np.array([[0.5, 0.5]]), k=9)
    with pytest.raises(ValueError):
        grad.eval("T")
    with pytest.raises(ValueError):
        grad("T")


def test_cell_gradient_rejects_point_evaluation():
    src = affine_2d_source()
    grad = mf.Gradient(src, k=9)
    with pytest.raises(ValueError):
        grad.eval_points("T")


def test_gradient_of_constant_field_is_zero():
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    src.fields["T"] = 7.0
    grad = mf.Gradient(src, k=9)
    vals = np.asarray(grad.eval("T")["QUAD4"])
    np.testing.assert_allclose(vals, 0.0, atol=1e-9)


def test_gradient_3d_hex():
    src = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3)
    )
    src.fields["T"] = 1.0 + 2.0 * mf.X - 3.0 * mf.Y + 4.0 * mf.Z
    grad = mf.Gradient(src, k=8)
    vals = np.asarray(grad.eval("T")["HEX8"])
    assert vals.shape == (src.num_elements(), 3)
    expected = np.array([2.0, -3.0, 4.0])
    np.testing.assert_allclose(vals, np.broadcast_to(expected, vals.shape), atol=1e-7)


def test_gradient_mixed_element_type_source():
    src = mf.UMesh(
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0], [1.0, 2.0]]
        )
    )
    src.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uint))
    src.add_regular_block("TRI3", np.array([[2, 3, 4], [4, 5, 2]], dtype=np.uint))
    src.fields["T"] = 1.0 + 2.0 * mf.X - 3.0 * mf.Y
    grad = mf.Gradient(src, k=3)
    out = grad.eval("T")
    assert set(out) == {"QUAD4", "TRI3"}
    for arr in out.values():
        vals = np.asarray(arr)
        assert vals.shape[1] == 2
        np.testing.assert_allclose(
            vals, np.broadcast_to(EXPECTED, vals.shape), atol=1e-7
        )
