"""Tests for the slicing tool."""

import numpy as np
import pytest

import mefikit as mf


def test_slice_3d_hex_midplane():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    sliced = mesh.slice([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

    assert sliced.block_types() == ["PGON"]
    assert sliced.num_elements() == 1
    assert sliced.coords().shape == (4, 3)
    assert np.allclose(sliced.coords()[:, 2], 0.5)
    assert sum(sliced.measure()["PGON"]) == pytest.approx(1.0)


def test_slice_3d_on_internal_interface_deduplicates_cells():
    mesh = mf.build_cmesh([0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    sliced = mesh.slice([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

    assert sliced.num_elements() == 4
    assert sliced.coords().shape == (9, 3)
    assert sum(sliced.measure()["PGON"]) == pytest.approx(1.0)


def test_slice_line_3d():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    sliced = mesh.slice_line([0.5, 0.5, 0.0], [0.0, 0.0, 1.0])

    assert sliced.block_types() == ["SEG2"]
    assert sliced.num_elements() == 1
    assert sliced.coords().shape == (2, 3)
    assert np.allclose(sliced.coords()[:, :2], 0.5)
    assert sum(sliced.measure()["SEG2"]) == pytest.approx(1.0)


def test_slice_line_3d_oblique():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    sliced = mesh.slice_line([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    assert sliced.num_elements() == 1
    assert sum(sliced.measure()["SEG2"]) == pytest.approx(np.sqrt(3.0))


def test_slice_line_2d():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0])
    sliced = mesh.slice_line([0.0, 1.0], [1.0, -1.0])

    assert sliced.block_types() == ["SEG2"]
    assert sliced.num_elements() == 1
    assert sliced.coords().shape == (2, 2)
    assert sum(sliced.measure()["SEG2"]) == pytest.approx(np.sqrt(2.0))


def test_slice_2d_line():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0])
    sliced = mesh.slice([0.5, 0.0], [1.0, 0.0])

    assert sliced.block_types() == ["SEG2"]
    assert sliced.num_elements() == 1
    assert sliced.coords().shape == (2, 2)
    assert np.allclose(sliced.coords()[:, 0], 0.5)
    assert sum(sliced.measure()["SEG2"]) == pytest.approx(1.0)


def test_slice_2d_on_internal_interface_deduplicates_cells():
    mesh = mf.build_cmesh([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])
    sliced = mesh.slice([0.5, 0.0], [1.0, 0.0])

    assert sliced.num_elements() == 2
    assert sliced.coords().shape == (3, 2)
    assert sum(sliced.measure()["SEG2"]) == pytest.approx(1.0)


def test_slice_then_transfer_field():
    mesh = mf.build_cmesh(
        np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 3),
        np.linspace(0.0, 1.0, 3),
    )
    mesh.fields["T"] = 7.0
    sliced = mesh.slice([0.0, 0.0, 0.5], [0.0, 0.0, 1.0])

    mf.transfer.ConstantPiecewise(mesh, sliced).apply_update(mesh, "T", sliced, "T")

    values = sliced.fields["T"].values()["PGON"]
    assert values.shape == (4,)
    assert np.allclose(values, 7.0)


def test_slice_line_then_transfer_field():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    mesh.fields["T"] = 7.0
    sliced = mesh.slice_line([0.5, 0.5, 0.0], [0.0, 0.0, 1.0])

    mf.transfer.ConstantPiecewise(mesh, sliced).apply_update(mesh, "T", sliced, "T")

    assert np.allclose(sliced.fields["T"].values()["SEG2"], 7.0)


def test_slice_rejects_bad_normal():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        mesh.slice([0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        mesh.slice([0.0, 0.0, 0.0], [0.0, 0.0, 1.0])
    with pytest.raises(ValueError):
        mesh.slice_line([0.0, 0.0], [0.0, 0.0])
