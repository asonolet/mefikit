import numpy as np
import pytest

import mefikit as mf

COORDS = np.array(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
)
BLOCK = np.array([[0, 1, 2, 3], [1, 4, 5, 2]], dtype=np.uint)


@pytest.fixture()
def quad2():
    mesh = mf.UMesh(COORDS.copy())
    mesh.add_regular_block("QUAD4", BLOCK.copy())
    return mesh


@pytest.fixture()
def ts(quad2):
    for t in (0.0, 0.5, 1.0):
        quad2.time_series["T"].add(10.0 + t, time=t)
    return quad2.time_series["T"]


# --- mapping / series creation ---


def test_mapping_empty(quad2):
    assert len(quad2.time_series) == 0
    assert quad2.time_series.keys() == []


def test_add_returns_fieldref(quad2):
    ref = quad2.time_series["T"].add(1.0, time=0.0)
    assert isinstance(ref, mf.FieldRef)
    assert np.allclose(ref.numpy(), 1.0)


def test_step_names_are_encoded(quad2):
    quad2.time_series["T"].add(1.0, time=0.0)
    assert list(quad2.fields.keys()) == ["T_step_0_time_0"]


def test_steps_are_sequential(quad2):
    for i, t in enumerate((0.0, 0.25, 0.5)):
        quad2.time_series["T"].add(i, time=t)
    assert quad2.time_series["T"].steps() == [0, 1, 2]


def test_mapping_iteration(quad2):
    quad2.time_series["T"].add(1.0, time=0.0)
    quad2.time_series["U"].add(2.0, time=0.0)
    ts = quad2.time_series
    assert ts.keys() == ["T", "U"]
    assert set(ts) == {"T", "U"}
    assert "T" in ts and "Z" not in ts
    assert len(ts) == 2
    assert [s.base for s in ts.values()] == ["T", "U"]
    assert [pair[0] for pair in ts.items()] == ["T", "U"]
    assert isinstance(ts["T"], mf.TimeSeries)


def test_missing_series_is_factory(quad2):
    s = quad2.time_series["T"]
    assert isinstance(s, mf.TimeSeries)
    assert len(s) == 0
    assert s.steps() == []
    assert "T" not in quad2.time_series
    s.add(1.0, time=0.0)
    assert "T" in quad2.time_series
    assert np.allclose(s.at(0).numpy().ravel(), 1.0)


def test_static_fields_coexist(quad2):
    quad2.fields["steady"] = 7.0
    quad2.time_series["T"].add(1.0, time=0.0)
    quad2.fields["other"] = 3.0
    assert set(quad2.fields.keys()) == {"steady", "other", "T_step_0_time_0"}
    assert len(quad2.time_series) == 1


# --- accessors ---


def test_base_and_dim(ts):
    assert ts.base == "T"
    assert ts.dim() == 2


def test_times_roundtrip(quad2):
    for t in (0.5, 1.0 / 3, -1e-5):
        quad2.time_series["T"].add(1.0, time=t)
    assert quad2.time_series["T"].times() == [0.5, 1.0 / 3, -1e-5]


def test_times_need_not_be_monotonic(quad2):
    quad2.time_series["T"].add(0.0, time=1.0)
    quad2.time_series["T"].add(0.0, time=0.0)
    s = quad2.time_series["T"]
    assert s.times() == [1.0, 0.0]
    assert np.allclose(s.latest().numpy().ravel(), [0.0, 0.0])


def test_at(ts):
    assert np.allclose(ts.at(0).numpy().ravel(), 10.0)
    assert np.allclose(ts.at(2).numpy().ravel(), 11.0)


def test_at_missing_step_raises(ts):
    with pytest.raises(KeyError):
        ts.at(5)


def test_at_time(ts):
    assert np.allclose(ts.at_time(0.5).numpy().ravel(), 10.5)


def test_at_time_duplicate_returns_lowest_step(quad2):
    quad2.time_series["T"].add(1.0, time=0.5)
    quad2.time_series["T"].add(2.0, time=0.5)
    s = quad2.time_series["T"]
    assert np.allclose(s.at_time(0.5).numpy().ravel(), [1.0, 1.0])


def test_at_time_missing_raises(ts):
    with pytest.raises(KeyError):
        ts.at_time(3.25)


def test_nearest(ts):
    assert np.allclose(ts.nearest(0.4).numpy().ravel(), 10.5)
    assert np.allclose(ts.nearest(-0.2).numpy().ravel(), 10.0)
    assert np.allclose(ts.nearest(3.0).numpy().ravel(), 11.0)


def test_nearest_tie_prefers_lower_step(quad2):
    quad2.time_series["T"].add(1.0, time=0.0)
    quad2.time_series["T"].add(2.0, time=0.4)
    quad2.time_series["T"].add(3.0, time=0.5)
    assert np.allclose(quad2.time_series["T"].nearest(0.45).numpy().ravel(), [2.0, 2.0])


def test_latest(ts):
    assert np.allclose(ts.latest().numpy().ravel(), 11.0)


def test_getitem_int(ts):
    assert np.allclose(ts[1].numpy().ravel(), 10.5)


def test_getitem_float_raises(ts):
    with pytest.raises(TypeError):
        ts[0.8]


def test_getitem_other_type_raises(ts):
    with pytest.raises(TypeError):
        ts["1"]


def test_len_and_contains(ts):
    assert len(ts) == 3
    assert 1 in ts
    assert 7 not in ts


def test_items(ts):
    items = ts.items()
    assert len(items) == 3
    assert items[0][:2] == (0, 0.0)
    assert np.allclose(items[0][2].numpy().ravel(), 10.0)
    assert items[1][:2] == (1, 0.5)
    assert items[2][:2] == (2, 1.0)


def test_iteration(ts):
    pairs = list(ts)
    assert len(pairs) == 3
    assert pairs[0][0] == 0.0  # noqa: RUF069
    assert np.allclose(pairs[0][1].numpy().ravel(), 10.0)


# --- mutation ---


def test_add_appends_new_step(ts):
    ts.add(12.0, time=1.5)
    assert ts.steps() == [0, 1, 2, 3]
    assert ts.times() == [0.0, 0.5, 1.0, 1.5]
    assert np.allclose(ts.latest().numpy().ravel(), 12.0)


def test_add_from_expression(quad2):
    quad2.fields["x"] = 1.0
    quad2.time_series["T"].add(mf.Field("x") * 2, time=0.0)
    assert np.allclose(quad2.time_series["T"].at(0).numpy().ravel(), 2.0)


def test_set_time(ts):
    ts.set_time(1, 0.75)
    assert ts.times() == [0.0, 0.75, 1.0]
    assert np.allclose(ts.at(1).numpy().ravel(), [10.5, 10.5])


def test_set_time_to_existing_collision_raises(quad2):
    quad2.time_series["T"].add(1.0, time=0.5)
    quad2.fields["T_step_0_time_1"] = 5.0
    with pytest.raises(ValueError):
        quad2.time_series["T"].set_time(0, 1.0)


def test_set_time_on_missing_step_raises(ts):
    with pytest.raises(KeyError):
        ts.set_time(9, 0.0)


def test_remove(ts):
    ts.remove(1)
    assert ts.steps() == [0, 2]
    assert ts.times() == [0.0, 1.0]


def test_delitem(ts):
    del ts[0]
    assert ts.steps() == [1, 2]


def test_remove_missing_step_raises(ts):
    with pytest.raises(KeyError):
        ts.remove(7)


def test_rename(ts, quad2):
    ts.rename("Q")
    assert ts.base == "Q"
    assert np.allclose(ts.at(0).numpy().ravel(), 10.0)
    assert sorted(quad2.fields.keys()) == [
        "Q_step_0_time_0",
        "Q_step_1_time_0.5",
        "Q_step_2_time_1",
    ]
    assert "T" not in quad2.fields


def test_rename_updates_mapping(ts, quad2):
    ts.rename("Q")
    assert "Q" in quad2.time_series
    assert "T" not in quad2.time_series
    assert quad2.time_series["Q"].base == ts.base == "Q"


def test_rename_to_existing_series_raises(quad2):
    quad2.time_series["T"].add(1.0, time=0.0)
    quad2.time_series["U"].add(9.0, time=0.0)
    with pytest.raises(ValueError):
        quad2.time_series["T"].rename("U")


def test_rename_to_existing_field_raises(quad2):
    quad2.time_series["T"].add(1.0, time=0.0)
    quad2.fields["U_step_0_time_1"] = 5.0
    with pytest.raises(ValueError):
        quad2.time_series["T"].rename("U")


# --- dimensions and errors ---


def test_dim_conflict_raises(quad2):
    quad2.add_regular_block("SEG2", np.array([[0, 1]], dtype=np.uint))
    quad2.time_series["T"].add(1.0, time=0.0, dim=1)
    with pytest.raises(ValueError):
        quad2.time_series["T"].add(1.0, time=1.0, dim=2)


def test_lower_dim_series(quad2):
    quad2.add_regular_block("SEG2", np.array([[0, 1]], dtype=np.uint))
    quad2.time_series["edge"].add(5.0, time=0.0, dim=1)
    s = quad2.time_series["edge"]
    assert s.dim() == 1
    assert np.allclose(s.at(0).numpy().ravel(), [5.0])
    quad2.time_series["edge"].add(6.0, time=1.0, dim=1)
    assert s.times() == [0.0, 1.0]
    # a full-dimension series can coexist
    quad2.time_series["T"].add(1.0, time=0.0)
    assert quad2.time_series["T"].dim() == 2


def test_dim_on_empty_dimension_raises(quad2):
    with pytest.raises(ValueError):
        quad2.time_series["T"].add(1.0, time=0.0, dim=1)


def test_unknown_expr_field_raises(quad2):
    with pytest.raises(ValueError):
        quad2.time_series["T"].add(mf.Field("nope"), time=0.0)
