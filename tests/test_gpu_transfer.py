from __future__ import annotations

import numpy as np
import pytest

import mefikit as mf


def affine_source():
    """5x5 QUAD4 grid carrying the scalar field f = 1 + 2x - 3y."""
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 1.0, 5))
    src.fields["T"] = 1.0 + 2.0 * mf.X - 3.0 * mf.Y
    return src


def offset_target():
    return mf.build_cmesh(np.linspace(0.1, 0.9, 4), np.linspace(0.1, 0.9, 4))


@pytest.fixture(scope="session")
def wgpu():
    """Skips the module when no ``wgpu`` adapter is available."""
    try:
        mf.InverseDistance(affine_source(), offset_target(), k=4, device="wgpu")
    except ValueError as exc:  # pragma: no cover - depends on the machine
        pytest.skip(f"no wgpu device: {exc}")
    return "wgpu"


def test_inverse_distance_matches_cpu(wgpu):
    src = affine_source()
    tgt = offset_target()
    gpu = np.asarray(mf.InverseDistance(src, tgt, k=8, device=wgpu).eval("T")["QUAD4"])
    cpu = np.asarray(mf.InverseDistance(src, tgt, k=8).eval("T")["QUAD4"])
    np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "weighting",
    [
        mf.DistanceWeighting.Constant(),
        mf.DistanceWeighting.InverseDistance(2.0),
        mf.DistanceWeighting.CompactSupport(2.0),
        mf.DistanceWeighting.Gaussian(),
    ],
)
def test_moving_least_squares_matches_cpu(wgpu, weighting):
    src = affine_source()
    tgt = offset_target()
    gpu = np.asarray(
        mf.MovingLeastSquares(src, tgt, k=12, weighting=weighting, device=wgpu).eval(
            "T"
        )["QUAD4"]
    )
    cpu = np.asarray(
        mf.MovingLeastSquares(src, tgt, k=12, weighting=weighting).eval("T")["QUAD4"]
    )
    np.testing.assert_allclose(gpu, cpu, atol=1e-4, rtol=1e-4)


def test_auto_device_matches_cpu():
    src = affine_source()
    tgt = offset_target()
    auto = np.asarray(
        mf.InverseDistance(src, tgt, k=8, device="auto").eval("T")["QUAD4"]
    )
    cpu = np.asarray(mf.InverseDistance(src, tgt, k=8).eval("T")["QUAD4"])
    np.testing.assert_allclose(auto, cpu, atol=1e-4, rtol=1e-4)


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError):
        mf.MovingLeastSquares(affine_source(), offset_target(), k=4, device="cuda")
