from __future__ import annotations

import numpy as np
import pytest

import mefikit as mf

EXPECTED = np.array([2.0, -3.0])


def affine_2d_source():
    """5x5 QUAD4 grid carrying the scalar field f = 1 + 2x - 3y."""
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    src.fields["T"] = 1.0 + 2.0 * mf.X - 3.0 * mf.Y
    return src


def offset_target():
    """4x4 QUAD4 grid inset inside the unit square."""
    return mf.build_cmesh(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))


@pytest.fixture(scope="session")
def wgpu():
    """Skips the module when no ``wgpu`` adapter is available."""
    try:
        mf.Gradient(affine_2d_source(), k=4, device="wgpu")
    except ValueError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no wgpu device: {exc}")
    return "wgpu"


def test_on_source_gradient_matches_cpu(wgpu):
    src = affine_2d_source()
    gpu = np.asarray(mf.Gradient(src, k=9, device=wgpu).eval("T")["QUAD4"])
    cpu = np.asarray(mf.Gradient(src, k=9).eval("T")["QUAD4"])
    np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(gpu, np.broadcast_to(EXPECTED, cpu.shape), atol=1e-4)


def test_on_target_gradient_matches_cpu(wgpu):
    src = affine_2d_source()
    tgt = offset_target()
    gpu = np.asarray(mf.Gradient(src, tgt, k=9, device=wgpu).eval("T")["QUAD4"])
    cpu = np.asarray(mf.Gradient(src, tgt, k=9).eval("T")["QUAD4"])
    assert gpu.shape == (tgt.num_elements(), 2)
    np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)


def test_at_points_gradient_matches_cpu(wgpu):
    src = affine_2d_source()
    points = np.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]])
    gpu = np.asarray(
        mf.Gradient.at_points(src, points, k=9, device=wgpu).eval_points("T")
    )
    cpu = np.asarray(mf.Gradient.at_points(src, points, k=9).eval_points("T"))
    np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)


def test_auto_device_matches_cpu():
    # `auto` either picks a device or falls back to the host; both must agree.
    src = affine_2d_source()
    tgt = offset_target()
    auto = np.asarray(mf.Gradient(src, tgt, k=9, device="auto").eval("T")["QUAD4"])
    cpu = np.asarray(mf.Gradient(src, tgt, k=9).eval("T")["QUAD4"])
    np.testing.assert_allclose(auto, cpu, atol=1e-4, rtol=1e-4)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        mf.Gradient(affine_2d_source(), k=4, device="cuda")
