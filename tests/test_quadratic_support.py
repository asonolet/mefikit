"""Regression tests for quadratic-cell support outside the 2D overlay."""

import numpy as np
import pytest

import mefikit as mf


def _quad8(coords):
    mesh = mf.UMesh(coords)
    mesh.add_regular_block(
        "QUAD8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uintp)
    )
    return mesh


def test_conservative_p0_rejects_quadratic_cells():
    coords = np.array(
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
    mesh = _quad8(coords)
    mesh.set_field("f", {"QUAD8": np.ones((1, 1))})
    with pytest.raises(ValueError, match="does not support quadratic cells"):
        mf.transfer.ConservativeP0(mesh, mesh)


def test_quadratic_surface_in_3d_has_a_centroid():
    # A QUAD8 surface embedded in 3D used to panic on centroid3, which is what
    # centroid-based selections use.
    xy = np.array(
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
    mesh = _quad8(np.column_stack([xy, np.zeros(8)]))
    ids = mesh.select(mf.sel.bbox([-1.0, -1.0, -1.0], [2.0, 2.0, 2.0])).ids()
    assert list(ids["QUAD8"]) == [0]
    # A box that excludes the cell's corner centroid must not select it.
    far = mesh.select(mf.sel.bbox([3.0, 3.0, 3.0], [4.0, 4.0, 4.0])).ids()
    assert len(far.get("QUAD8", [])) == 0
