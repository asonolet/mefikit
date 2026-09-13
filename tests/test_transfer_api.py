from __future__ import annotations

import numpy as np
import pytest

import mefikit as mf


def mixed_quad_tri_source():
    """2D source with both QUAD4 and TRI3 cells at the topological dimension."""
    src = mf.UMesh(
        np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 2.0], [1.0, 2.0]]
        )
    )
    src.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uint))
    src.add_regular_block("TRI3", np.array([[2, 3, 4], [4, 5, 2]], dtype=np.uint))
    src.set_field("T", {"QUAD4": np.array([[1.0]]), "TRI3": np.array([[2.0], [3.0]])})
    return src


def hex_cube_source():
    """Single HEX8 cell in the unit cube."""
    src = mf.UMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        )
    )
    src.add_regular_block("HEX8", np.array([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=np.uint))
    src.set_field("T", {"HEX8": np.array([[5.0]])})
    return src


def quad_surface_target():
    """Single QUAD4 surface in (x, y, z=0)."""
    tgt = mf.UMesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    )
    tgt.add_regular_block("QUAD4", np.array([[0, 1, 2, 3]], dtype=np.uint))
    return tgt


def mls(k):
    return lambda s, t: mf.transfer.MovingLeastSquares(s, t, k=k)


def invdist(k):
    return lambda s, t: mf.transfer.InverseDistance(s, t, k=k)


# --- new API (eval / __call__ / lazy field assignment) vs apply_update ---


@pytest.mark.parametrize(
    "make_trsf",
    [
        mf.transfer.ConstantPiecewise,
        mls(5),
        invdist(4),
        mf.transfer.ConservativeP0,
    ],
)
def test_eval_matches_apply_update_2d(make_trsf):
    xs = np.linspace(0.0, 1.0, 5)
    src = mf.build_cmesh(xs, xs)
    tgt = mf.build_cmesh(xs + 0.03, xs[1:] - 0.02)
    src.set_field("T", {"QUAD4": (np.arange(16, dtype=float) + 1.0).reshape(-1, 1)})

    trsf = make_trsf(src, tgt)
    eager = trsf.eval("T")
    eager_vals = np.concatenate([np.asarray(a).ravel() for a in eager.values()])

    ref = mf.build_cmesh(xs + 0.03, xs[1:] - 0.02)
    trsf.apply_update(src, "T", ref, "T", def_val=0.0)
    ref_vals = ref.fields["T"].numpy().ravel()

    assert eager_vals.shape == ref_vals.shape
    assert np.allclose(eager_vals, ref_vals, equal_nan=True)


@pytest.mark.parametrize(
    "make_trsf",
    [
        mf.transfer.ConstantPiecewise,
        mls(10),
        invdist(4),
        mf.transfer.ConservativeP0,
    ],
)
def test_call_and_lazy_assignment_match_apply_update(make_trsf):
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 4), np.linspace(0.0, 1.0, 4))
    tgt = mf.build_cmesh(np.linspace(0.25, 0.75, 3), np.linspace(0.25, 0.75, 3))
    src.set_field("T", {"QUAD4": (np.arange(9, dtype=float) + 1.0).reshape(-1, 1)})

    trsf = make_trsf(src, tgt)
    tgt_lazy = mf.build_cmesh(np.linspace(0.25, 0.75, 3), np.linspace(0.25, 0.75, 3))
    tgt_lazy.fields["T"] = trsf(mf.Field("T"))

    tgt_call = mf.build_cmesh(np.linspace(0.25, 0.75, 3), np.linspace(0.25, 0.75, 3))
    tgt_call.fields["T"] = trsf("T") * 1.0

    ref = mf.build_cmesh(np.linspace(0.25, 0.75, 3), np.linspace(0.25, 0.75, 3))
    trsf.apply_update(src, "T", ref, "T")

    for field in (
        tgt_lazy.fields["T"].numpy().ravel(),
        tgt_call.fields["T"].numpy().ravel(),
    ):
        np.testing.assert_allclose(field, ref.fields["T"].numpy().ravel())


# --- mixed element type source (regression: MLS/InverseDistance n_src) ---


@pytest.mark.parametrize(
    "make_trsf",
    [
        mf.transfer.ConstantPiecewise,
        mls(4),
        invdist(4),
        mf.transfer.ConservativeP0,
    ],
)
def test_mixed_element_type_source(make_trsf):
    src = mixed_quad_tri_source()
    tgt = mf.build_cmesh(np.linspace(0.1, 0.9, 3), np.linspace(0.1, 1.9, 3))
    trsf = make_trsf(src, tgt)
    out = trsf.eval("T")
    vals = np.concatenate([np.asarray(a).ravel() for a in out.values()])
    assert vals.size == tgt.num_elements()
    # every target cell samples the source; none may be zeros (default) here
    assert not np.any(np.isclose(vals, 0.0))


# --- downcast: 3D volume source -> 2D surface target ---


@pytest.mark.parametrize(
    "make_trsf",
    [
        mf.transfer.ConstantPiecewise,
        mls(4),
        invdist(4),
    ],
)
def test_downcast_3d_to_2d(make_trsf):
    src = hex_cube_source()
    tgt = quad_surface_target()
    trsf = make_trsf(src, tgt)
    out = trsf.eval("T")
    vals = np.concatenate([np.asarray(a).ravel() for a in out.values()])
    assert vals.size == tgt.num_elements()
    assert vals[0] == pytest.approx(5.0)


# --- def_val semantics: uncovered target cells get the sentinel ---


def test_uncovered_cells_default_zero():
    src = mf.build_cmesh(np.array([0.0, 0.25, 0.5]), np.array([0.0, 0.25, 0.5]))
    tgt = mf.build_cmesh(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]))
    src.set_field("T", {"QUAD4": np.ones((4, 1))})

    trsf = mf.transfer.ConservativeP0(src, tgt)
    vals = trsf.eval("T").get("QUAD4", None)
    assert vals is not None
    vals = np.asarray(vals).ravel()
    # 2x2 target grid: one cell inside [0, 0.5]^2, three outside (zeros)
    assert np.any(np.isclose(vals, 0.0))
    assert np.any(np.isclose(vals, 1.0))

    trsf_nan = mf.transfer.ConservativeP0(src, tgt, def_val=np.nan)
    vals_nan = np.asarray(trsf_nan.eval("T")["QUAD4"]).ravel()
    covered = np.isfinite(vals_nan)
    assert np.any(~covered)
    assert np.all(np.isclose(vals_nan[covered], 1.0))


# --- extensive flag parity ---


@pytest.mark.parametrize(
    "name",
    ["ConstantPiecewise", "MovingLeastSquares", "InverseDistance", "ConservativeP0"],
)
def test_apply_update_accepts_extensive(name):
    src = mf.build_cmesh(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
    tgt = mf.build_cmesh(np.linspace(0.1, 0.9, 2), np.linspace(0.1, 0.9, 2))
    src.set_field("T", {"QUAD4": np.ones((4, 1))})
    make = {
        "ConstantPiecewise": mf.transfer.ConstantPiecewise,
        "MovingLeastSquares": mls(4),
        "InverseDistance": invdist(4),
        "ConservativeP0": mf.transfer.ConservativeP0,
    }
    trsf = make[name](src, tgt)
    trsf.apply_update(src, "T", tgt, "T", def_val=0.0, extensive=True)
    vals = tgt.fields["T"].numpy().ravel()
    assert np.all(np.isfinite(vals))
