import numpy as np
import pytest

import mefikit as mf


def _make_fluid():
    mesh = mf.build_cmesh([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
    mesh.fields["pressure"] = np.array([3.0])
    mesh.fields["velocity"] = np.array([[1.0, 2.0, 3.0]])
    for value, time in [(1.0, 0.0), (2.0, 0.25), (4.0, 1.0)]:
        mesh.time_series["T"].add(np.array([value]), time=time)
    mesh.groups["wall"] = {"HEX8": [0]}
    return mesh


def _make_solid():
    mesh = mf.UMesh(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    mesh.add_regular_block("TRI3", np.array([[0, 1, 2]], dtype=np.uintp))
    mesh.fields["density"] = np.array([5.0])
    mesh.groups["boundary"] = {"TRI3": [0]}
    return mesh


def _make_container():
    container = mf.MedFile()
    container["fluid"] = _make_fluid()
    container["solid"] = _make_solid()
    return container


def _write_container(tmp_path, name="container.med"):
    path = tmp_path / name
    _make_container().write(str(path))
    return path


def test_inspect_reports_structure_without_arrays(tmp_path):
    path = _write_container(tmp_path)
    structure = mf.MedFile.inspect(str(path))

    assert len(structure) == 2
    assert set(structure.meshes) == {"fluid", "solid"}
    assert set(structure.fields) == {"T", "pressure", "velocity", "density"}

    fluid = structure.meshes["fluid"]
    assert fluid.name == "fluid"
    assert fluid.space_dimension == 3
    assert fluid.num_nodes == 8
    assert fluid.num_cells == 1
    assert fluid.element_types == {"HE8": 1}
    assert fluid.unsupported_element_types == []
    assert fluid.groups == ["wall"]

    solid = structure.meshes["solid"]
    assert solid.space_dimension == 2
    assert solid.num_nodes == 3
    assert solid.element_types == {"TR3": 1}
    assert solid.groups == ["boundary"]

    temperature = structure.fields["T"]
    assert temperature.mesh_name == "fluid"
    assert temperature.dimension == 3
    assert temperature.components == 1
    assert [step.index for step in temperature.steps] == [0, 1, 2]
    assert [step.time for step in temperature.steps] == [0.0, 0.25, 1.0]

    velocity = structure.fields["velocity"]
    assert velocity.components == 3
    assert len(velocity.steps) == 1

    density = structure.fields["density"]
    assert density.mesh_name == "solid"
    assert density.dimension == 2


def test_umesh_read_rejects_multi_mesh_file(tmp_path):
    path = _write_container(tmp_path)
    with pytest.raises(ValueError, match=r"use MedFile\.read"):
        mf.UMesh.read(str(path))


def test_roundtrip_multi_mesh_container(tmp_path):
    path = _write_container(tmp_path)
    container = mf.MedFile.read(str(path))

    assert container.keys() == ["fluid", "solid"]
    assert len(container) == 2
    assert "fluid" in container
    assert set(container) == {"fluid", "solid"}
    assert [pair[0] for pair in container.items()] == ["fluid", "solid"]

    fluid = container["fluid"]
    np.testing.assert_array_equal(
        fluid.fields["pressure"].values()["HEX8"], np.array([3.0])
    )
    np.testing.assert_array_equal(
        fluid.fields["velocity"].values()["HEX8"], np.array([[1.0, 2.0, 3.0]])
    )
    assert fluid.time_series["T"].times() == [0.0, 0.25, 1.0]
    assert set(fluid.groups) == {"wall"}
    assert sorted(fluid.groups["wall"].ids()["HEX8"]) == [0]

    solid = container["solid"]
    np.testing.assert_array_equal(
        solid.fields["density"].values()["TRI3"], np.array([5.0])
    )
    assert set(solid.groups) == {"boundary"}


@pytest.mark.parametrize(
    ("step_selection", "expected_times"),
    [
        (None, [0.0, 0.25, 1.0]),
        (1, [0.25]),
        (-1, [1.0]),
        (slice(0, 2), [0.0, 0.25]),
        (slice(2, 0, -1), [0.25, 1.0]),
        (range(0, 3, 2), [0.0, 1.0]),
        ([2, 0], [0.0, 1.0]),
    ],
)
def test_selection_loads_only_selected_steps(tmp_path, step_selection, expected_times):
    path = _write_container(tmp_path)
    container = mf.MedFile.read(str(path), {"fluid": {"T": step_selection}})

    assert container.keys() == ["fluid"]
    fluid = container["fluid"]
    assert set(fluid.time_series) == {"T"}
    assert fluid.time_series["T"].times() == expected_times
    assert "pressure" not in fluid.fields
    assert "velocity" not in fluid.fields


def test_empty_step_sequence_loads_no_field(tmp_path):
    path = _write_container(tmp_path)
    container = mf.MedFile.read(str(path), {"fluid": {"T": []}})

    fluid = container["fluid"]
    assert len(fluid.fields) == 0
    assert "T" not in fluid.time_series


def test_empty_selections_load_nothing_or_geometry_only(tmp_path):
    path = _write_container(tmp_path)

    empty = mf.MedFile.read(str(path), {})
    assert len(empty) == 0

    mesh_only = mf.MedFile.read(str(path), {"solid": {}})
    assert mesh_only.keys() == ["solid"]
    solid = mesh_only["solid"]
    assert solid.num_elements() == 1
    assert len(solid.fields) == 0
    assert set(solid.groups) == {"boundary"}


def test_read_mesh_returns_one_loaded_mesh(tmp_path):
    path = _write_container(tmp_path)

    fluid = mf.MedFile.read_mesh(str(path), "fluid")
    assert fluid.num_elements() == 1
    assert "pressure" in fluid.fields
    assert fluid.time_series["T"].times() == [0.0, 0.25, 1.0]
    assert set(fluid.groups) == {"wall"}

    partial = mf.MedFile.read_mesh(
        str(path), "fluid", {"T": slice(1, None), "pressure": None}
    )
    assert partial.time_series["T"].times() == [0.25, 1.0]
    assert "pressure" in partial.fields
    assert "velocity" not in partial.fields

    geometry = mf.MedFile.read_mesh(str(path), "fluid", {})
    assert geometry.num_elements() == 1
    assert len(geometry.fields) == 0
    assert set(geometry.groups) == {"wall"}


def test_partial_update_keeps_unloaded_entries(tmp_path):
    path = _write_container(tmp_path)

    partial = mf.MedFile.read(str(path), {"solid": {"density": None}})
    partial["solid"].fields["density"] = np.array([9.0])
    partial.write(str(path), mode="update")

    full = mf.MedFile.read(str(path))
    np.testing.assert_array_equal(
        full["solid"].fields["density"].values()["TRI3"], np.array([9.0])
    )
    fluid = full["fluid"]
    np.testing.assert_array_equal(
        fluid.fields["pressure"].values()["HEX8"], np.array([3.0])
    )
    assert fluid.time_series["T"].times() == [0.0, 0.25, 1.0]


def test_write_modes(tmp_path):
    path = tmp_path / "modes.med"
    container = mf.MedFile()
    container["mesh"] = _make_solid()

    container.write(str(path))
    assert path.exists()
    with pytest.raises(OSError):
        container.write(str(path), mode="create")

    container.write(str(path), mode="overwrite")
    assert mf.MedFile.read(str(path)).keys() == ["mesh"]

    with pytest.raises(ValueError, match="unknown MED write mode"):
        container.write(str(path), mode="bogus")


def test_duplicate_field_names_are_rejected(tmp_path):
    fluid = _make_fluid()
    solid = _make_solid()
    solid.fields["pressure"] = np.array([7.0])

    container = mf.MedFile()
    container["fluid"] = fluid
    container["solid"] = solid

    with pytest.raises(ValueError, match="file-global"):
        container.write(str(tmp_path / "duplicate.med"))


def test_static_field_and_time_series_cannot_share_a_name(tmp_path):
    fluid = _make_fluid()
    fluid.fields["T"] = np.array([7.0])

    container = mf.MedFile()
    container["fluid"] = fluid

    with pytest.raises(ValueError, match="both a static field and a time series"):
        container.write(str(tmp_path / "conflict.med"))


def test_invalid_selections_raise(tmp_path):
    path = _write_container(tmp_path)

    with pytest.raises(KeyError):
        mf.MedFile.read(str(path), {"missing": None})
    with pytest.raises(KeyError):
        mf.MedFile.read(str(path), {"fluid": {"missing": None}})
    with pytest.raises(ValueError, match="belongs to mesh"):
        mf.MedFile.read(str(path), {"fluid": {"density": None}})
    with pytest.raises(ValueError, match="out of range"):
        mf.MedFile.read(str(path), {"fluid": {"T": 3}})
    with pytest.raises(TypeError):
        mf.MedFile.read(str(path), {"fluid": 3})
    with pytest.raises(TypeError):
        mf.MedFile.read(str(path), {"fluid": {"T": True}})
    with pytest.raises(TypeError):
        mf.MedFile.read(str(path), {"fluid": {"T": b"\x00"}})
    with pytest.raises(TypeError):
        mf.MedFile.read(str(path), {"fluid": {"T": [True]}})
    with pytest.raises(KeyError):
        mf.MedFile.read_mesh(str(path), "missing")


def test_unsupported_element_types_are_reported_and_skipped(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = _write_container(tmp_path)

    with h5py.File(path, "r+") as handle:
        med_mesh = handle["ENS_MAA/fluid"]
        step_name = next(iter(med_mesh))
        mai = med_mesh[step_name]["MAI"]
        pyramid = mai.create_group("PY5")
        pyramid.attrs["CGT"] = 1
        pyramid.attrs["CGS"] = 1
        pyramid.attrs["GEO"] = 205
        pyramid.attrs["PFL"] = np.bytes_(b"MED_NO_PROFILE_INTERNAL")
        nod = pyramid.create_dataset(
            "NOD", data=np.array([[1, 2, 3, 4, 5]], dtype=np.int64).ravel()
        )
        nod.attrs["CGT"] = 1
        nod.attrs["NBR"] = 1

    structure = mf.MedFile.inspect(str(path))
    fluid = structure.meshes["fluid"]
    assert fluid.unsupported_element_types == ["PY5"]
    assert fluid.num_cells == 2

    loaded = mf.MedFile.read(str(path))
    assert loaded["fluid"].block_types() == ["HEX8"]
    assert loaded["fluid"].num_elements() == 1


def test_medcoupling_reads_multi_mesh_container(tmp_path):
    mc = pytest.importorskip("medcoupling")
    path = _write_container(tmp_path)

    meshes = mc.MEDFileMeshes(str(path))
    assert set(meshes.getMeshesNames()) == {"fluid", "solid"}

    fields = mc.MEDFileFields(str(path))
    assert set(fields.getFieldsNames()) == {"T", "pressure", "velocity", "density"}

    fluid_mesh = mc.MEDFileUMesh(str(path), "fluid")
    pressure = fields.getFieldWithName("pressure").field(1, 1, fluid_mesh)
    np.testing.assert_array_equal(pressure.getArray().toNumPyArray(), [3.0])
    velocity = fields.getFieldWithName("velocity").field(1, 1, fluid_mesh)
    np.testing.assert_array_equal(
        velocity.getArray().toNumPyArray().ravel(), [1.0, 2.0, 3.0]
    )

    solid_mesh = mc.MEDFileUMesh(str(path), "solid")
    density = fields.getFieldWithName("density").field(1, 1, solid_mesh)
    np.testing.assert_array_equal(density.getArray().toNumPyArray(), [5.0])

    temperature = fields.getFieldWithName("T")
    assert temperature.getTimeSteps() == [(0, 0, 0.0), (1, 1, 0.25), (2, 2, 1.0)]
    values = [
        temperature.field(iteration, order, fluid_mesh).getArray().toNumPyArray()[0]
        for iteration, order, _time in temperature.getTimeSteps()
    ]
    np.testing.assert_array_equal(values, [1.0, 2.0, 4.0])
