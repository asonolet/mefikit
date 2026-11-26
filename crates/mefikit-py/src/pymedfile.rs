use std::collections::BTreeMap;

use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{
    PyAny, PyBool, PyBytes, PyDict, PyEllipsis, PyIterator, PyList, PySlice, PySliceMethods,
    PyString,
};

use mefikit::io::{
    MedFieldInfo, MedFile, MedFileStructure, MedMeshInfo, MedSelection, MedStepInfo, MedWriteMode,
    MeshSelection, StepSelection,
};
use mefikit::mesh::UMesh;
use mefikit::prelude as mf;

use super::pyumesh::PyUMesh;

fn map_io_err(err: mf::MefikitIOError) -> PyErr {
    match err {
        mf::MefikitIOError::Io(err) => pyo3::exceptions::PyOSError::new_err(err.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

/// One time step of a MED field.
#[pyclass(frozen, module = "mefikit.mefipy", skip_from_py_object)]
#[pyo3(name = "MedStepInfo")]
#[derive(Clone)]
pub struct PyMedStepInfo {
    /// Zero-based index in the field's step order.
    #[pyo3(get)]
    pub index: usize,
    /// MED iteration number.
    #[pyo3(get)]
    pub iteration: i64,
    /// MED order number.
    #[pyo3(get)]
    pub order: i64,
    /// Physical time.
    #[pyo3(get)]
    pub time: f64,
}

#[pymethods]
impl PyMedStepInfo {
    fn __repr__(&self) -> String {
        format!(
            "MedStepInfo(index={}, iteration={}, order={}, time={})",
            self.index, self.iteration, self.order, self.time
        )
    }
}

impl From<MedStepInfo> for PyMedStepInfo {
    fn from(info: MedStepInfo) -> Self {
        Self {
            index: info.index,
            iteration: info.iteration,
            order: info.order,
            time: info.time,
        }
    }
}

/// Metadata for one mesh in a MED file.
#[pyclass(frozen, module = "mefikit.mefipy", skip_from_py_object)]
#[pyo3(name = "MedMeshInfo")]
#[derive(Clone)]
pub struct PyMedMeshInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub space_dimension: usize,
    #[pyo3(get)]
    pub num_nodes: usize,
    #[pyo3(get)]
    pub num_cells: usize,
    /// Cell count per supported MED element type name.
    #[pyo3(get)]
    pub element_types: BTreeMap<String, usize>,
    /// MED element type names that mefikit cannot read.
    #[pyo3(get)]
    pub unsupported_element_types: Vec<String>,
    #[pyo3(get)]
    pub groups: Vec<String>,
}

#[pymethods]
impl PyMedMeshInfo {
    fn __repr__(&self) -> String {
        format!(
            "MedMeshInfo(name={:?}, space_dimension={}, num_nodes={}, num_cells={}, \
             element_types={:?}, unsupported_element_types={:?}, groups={:?})",
            self.name,
            self.space_dimension,
            self.num_nodes,
            self.num_cells,
            self.element_types,
            self.unsupported_element_types,
            self.groups
        )
    }
}

impl From<MedMeshInfo> for PyMedMeshInfo {
    fn from(info: MedMeshInfo) -> Self {
        Self {
            name: info.name,
            space_dimension: info.space_dimension,
            num_nodes: info.num_nodes,
            num_cells: info.num_cells,
            element_types: info.element_types,
            unsupported_element_types: info.unsupported_element_types,
            groups: info.groups,
        }
    }
}

/// Metadata for one field in a MED file.
#[pyclass(frozen, module = "mefikit.mefipy", skip_from_py_object)]
#[pyo3(name = "MedFieldInfo")]
#[derive(Clone)]
pub struct PyMedFieldInfo {
    #[pyo3(get)]
    pub name: String,
    #[pyo3(get)]
    pub mesh_name: String,
    /// Topological dimension, or `None` when the field only uses unsupported types.
    #[pyo3(get)]
    pub dimension: Option<usize>,
    #[pyo3(get)]
    pub components: usize,
    #[pyo3(get)]
    pub steps: Vec<PyMedStepInfo>,
}

#[pymethods]
impl PyMedFieldInfo {
    fn __repr__(&self) -> String {
        format!(
            "MedFieldInfo(name={:?}, mesh_name={:?}, dimension={:?}, components={}, steps={:?})",
            self.name,
            self.mesh_name,
            self.dimension,
            self.components,
            self.steps.iter().map(|s| s.index).collect::<Vec<_>>()
        )
    }
}

impl From<MedFieldInfo> for PyMedFieldInfo {
    fn from(info: MedFieldInfo) -> Self {
        Self {
            name: info.name,
            mesh_name: info.mesh_name,
            dimension: info.dimension.map(u8::from).map(usize::from),
            components: info.components,
            steps: info.steps.into_iter().map(PyMedStepInfo::from).collect(),
        }
    }
}

/// Metadata-only description of a MED file.
#[pyclass(frozen, module = "mefikit.mefipy", skip_from_py_object)]
#[pyo3(name = "MedFileStructure")]
#[derive(Clone)]
pub struct PyMedFileStructure {
    pub inner: MedFileStructure,
}

#[pymethods]
impl PyMedFileStructure {
    /// Meshes by name.
    #[getter]
    fn meshes(&self) -> BTreeMap<String, PyMedMeshInfo> {
        self.inner
            .meshes
            .iter()
            .map(|(name, info)| (name.clone(), PyMedMeshInfo::from(info.clone())))
            .collect()
    }

    /// Fields by name.
    #[getter]
    fn fields(&self) -> BTreeMap<String, PyMedFieldInfo> {
        self.inner
            .fields
            .iter()
            .map(|(name, info)| (name.clone(), PyMedFieldInfo::from(info.clone())))
            .collect()
    }

    fn __len__(&self) -> usize {
        self.inner.meshes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "MedFileStructure(meshes={:?}, fields={:?})",
            self.inner.meshes.keys().collect::<Vec<_>>(),
            self.inner.fields.keys().collect::<Vec<_>>()
        )
    }
}

/// A multi-mesh MED file with named meshes and their fields.
#[pyclass(module = "mefikit.mefipy")]
#[pyo3(name = "MedFile")]
pub struct PyMedFile {
    pub meshes: BTreeMap<String, Py<PyUMesh>>,
}

impl PyMedFile {
    fn from_inner(py: Python<'_>, inner: MedFile) -> PyResult<Self> {
        let mut meshes = BTreeMap::new();
        for (name, mesh) in inner.meshes {
            meshes.insert(name, Py::new(py, PyUMesh::from(mesh))?);
        }
        Ok(Self { meshes })
    }

    fn get_mesh(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyUMesh>> {
        self.meshes
            .get(name)
            .map(|mesh| mesh.clone_ref(py))
            .ok_or_else(|| PyKeyError::new_err(name.to_string()))
    }
}

#[pymethods]
impl PyMedFile {
    #[new]
    fn new() -> Self {
        Self {
            meshes: BTreeMap::new(),
        }
    }

    /// Reads a MED file, optionally selecting meshes, fields and time steps.
    ///
    /// ``selection=None`` loads every mesh, field and step. An empty dict loads
    /// no mesh. Otherwise the dict maps mesh names to either ``None`` (all
    /// fields and steps) or a field dict. A field maps to ``None`` (all steps),
    /// an integer step index (negative counts from the end), a slice, or a
    /// range/sequence of step indices. An empty field dict loads only the mesh
    /// geometry and groups.
    #[staticmethod]
    #[pyo3(signature = (path, selection=None))]
    fn read(py: Python<'_>, path: &str, selection: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let structure = match selection {
            Some(_) => Some(MedFile::inspect(path).map_err(map_io_err)?),
            None => None,
        };
        let parsed = match (selection, &structure) {
            (Some(value), Some(structure)) => parse_med_selection(value, structure)?,
            _ => None,
        };
        let inner = MedFile::read(path, parsed.as_ref()).map_err(map_io_err)?;
        Self::from_inner(py, inner)
    }

    /// Reads one named mesh with an optional field/step selection.
    ///
    /// ``selection=None`` loads every field of the mesh. An empty dict loads
    /// only the mesh geometry and groups. Otherwise it maps field names to a
    /// step selection (``None``, an integer, a slice, or a range/sequence).
    #[staticmethod]
    #[pyo3(signature = (path, mesh, selection=None))]
    fn read_mesh(
        path: &str,
        mesh: &str,
        selection: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyUMesh> {
        let structure = MedFile::inspect(path).map_err(map_io_err)?;
        if !structure.meshes.contains_key(mesh) {
            return Err(PyKeyError::new_err(format!(
                "MED file has no mesh named {mesh:?}"
            )));
        }
        let parsed = match selection {
            Some(value) => Some(parse_mesh_selection(value, mesh, &structure)?),
            None => None,
        };
        let mesh = MedFile::read_mesh(path, mesh, parsed.as_ref()).map_err(map_io_err)?;
        Ok(mesh.into())
    }

    /// Reads structure and field metadata without loading any array.
    #[staticmethod]
    fn inspect(path: &str) -> PyResult<PyMedFileStructure> {
        let inner = MedFile::inspect(path).map_err(map_io_err)?;
        Ok(PyMedFileStructure { inner })
    }

    /// Writes the container.
    ///
    /// ``mode`` is ``"update"`` (default), ``"create"`` or ``"overwrite"``.
    /// Update mode opens an existing file and replaces only the meshes present
    /// in this container; a missing file is created.
    #[pyo3(signature = (path, mode="update"))]
    fn write(&self, py: Python<'_>, path: &str, mode: &str) -> PyResult<()> {
        let mode = parse_write_mode(mode)?;
        let guards: Vec<_> = self
            .meshes
            .iter()
            .map(|(name, mesh)| (name.as_str(), mesh.bind(py).borrow()))
            .collect();
        let meshes: Vec<(&str, &UMesh)> = guards
            .iter()
            .map(|(name, mesh)| (*name, &mesh.inner))
            .collect();
        mefikit::io::write_named_meshes(path, mode, &meshes).map_err(map_io_err)
    }

    /// Mesh names, in sorted order.
    pub fn keys(&self) -> Vec<String> {
        self.meshes.keys().cloned().collect()
    }

    /// Mesh objects, in sorted name order.
    pub fn values(&self) -> Vec<Py<PyUMesh>> {
        Python::attach(|py| {
            self.meshes
                .values()
                .map(|mesh| mesh.clone_ref(py))
                .collect()
        })
    }

    /// ``(name, mesh)`` pairs, in sorted name order.
    pub fn items(&self) -> Vec<(String, Py<PyUMesh>)> {
        Python::attach(|py| {
            self.meshes
                .iter()
                .map(|(name, mesh)| (name.clone(), mesh.clone_ref(py)))
                .collect()
        })
    }

    pub fn __len__(&self) -> usize {
        self.meshes.len()
    }

    pub fn __contains__(&self, name: &str) -> bool {
        self.meshes.contains_key(name)
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        let list = PyList::new(slf.py(), slf.keys())?;
        list.as_any().try_iter()
    }

    pub fn __getitem__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyUMesh>> {
        self.get_mesh(py, name)
    }

    pub fn __setitem__(&mut self, name: &str, mesh: Py<PyUMesh>) {
        self.meshes.insert(name.to_string(), mesh);
    }

    pub fn __delitem__(&mut self, name: &str) -> PyResult<()> {
        if self.meshes.remove(name).is_none() {
            return Err(PyKeyError::new_err(name.to_string()));
        }
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "MedFile(meshes={:?})",
            self.meshes.keys().collect::<Vec<_>>()
        )
    }
}

fn parse_write_mode(mode: &str) -> PyResult<MedWriteMode> {
    match mode.to_ascii_lowercase().as_str() {
        "create" => Ok(MedWriteMode::Create),
        "overwrite" => Ok(MedWriteMode::Overwrite),
        "update" => Ok(MedWriteMode::Update),
        other => Err(PyValueError::new_err(format!(
            "unknown MED write mode {other:?}; expected 'create', 'overwrite' or 'update'"
        ))),
    }
}

fn parse_med_selection(
    selection: &Bound<'_, PyAny>,
    structure: &MedFileStructure,
) -> PyResult<Option<MedSelection>> {
    if selection.is_none() || selection.is_instance_of::<PyEllipsis>() {
        return Ok(None);
    }
    let dict = selection
        .cast::<PyDict>()
        .map_err(|_| PyTypeError::new_err("selection must be None or a dict mapping mesh names"))?;

    let mut meshes = BTreeMap::new();
    for (key, value) in dict {
        let mesh_name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("mesh names in a MED selection must be strings"))?;
        if !structure.meshes.contains_key(&mesh_name) {
            return Err(PyKeyError::new_err(format!(
                "MED file has no mesh named {mesh_name:?}"
            )));
        }
        let mesh_selection = if value.is_none() || value.is_instance_of::<PyEllipsis>() {
            MeshSelection::All
        } else if value.cast::<PyDict>().is_ok() {
            parse_field_selection(&value, &mesh_name, structure)?
        } else {
            return Err(PyTypeError::new_err(format!(
                "selection for mesh {mesh_name:?} must be None or a dict of fields"
            )));
        };
        meshes.insert(mesh_name, mesh_selection);
    }
    Ok(Some(MedSelection { meshes }))
}

fn parse_mesh_selection(
    selection: &Bound<'_, PyAny>,
    mesh_name: &str,
    structure: &MedFileStructure,
) -> PyResult<MeshSelection> {
    if selection.is_none() || selection.is_instance_of::<PyEllipsis>() {
        return Ok(MeshSelection::All);
    }
    if !structure.meshes.contains_key(mesh_name) {
        return Err(PyKeyError::new_err(format!(
            "MED file has no mesh named {mesh_name:?}"
        )));
    }
    if selection.cast::<PyDict>().is_err() {
        return Err(PyTypeError::new_err(
            "mesh selection must be None or a dict mapping field names to steps",
        ));
    }
    parse_field_selection(selection, mesh_name, structure)
}

fn parse_field_selection(
    selection: &Bound<'_, PyAny>,
    mesh_name: &str,
    structure: &MedFileStructure,
) -> PyResult<MeshSelection> {
    let dict = selection.cast::<PyDict>().map_err(|_| {
        PyTypeError::new_err("field selection must be a dict mapping field names to steps")
    })?;

    let mut fields = BTreeMap::new();
    for (key, value) in dict {
        let field_name: String = key
            .extract()
            .map_err(|_| PyTypeError::new_err("field names in a MED selection must be strings"))?;
        let info = structure.fields.get(&field_name).ok_or_else(|| {
            PyKeyError::new_err(format!("MED file has no field named {field_name:?}"))
        })?;
        if info.mesh_name != mesh_name {
            return Err(PyValueError::new_err(format!(
                "field {field_name:?} belongs to mesh {:?}, not {mesh_name:?}",
                info.mesh_name
            )));
        }
        fields.insert(field_name, parse_step_selection(&value, info.steps.len())?);
    }
    Ok(MeshSelection::Fields(fields))
}

fn parse_step_selection(value: &Bound<'_, PyAny>, total: usize) -> PyResult<StepSelection> {
    if value.is_none() || value.is_instance_of::<PyEllipsis>() {
        return Ok(StepSelection::All);
    }
    if value.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "a MED step selection must be None, an integer, a slice or a sequence of integers",
        ));
    }
    if let Ok(index) = value.extract::<i64>() {
        return Ok(StepSelection::Index(index));
    }
    if let Ok(slice) = value.cast::<PySlice>() {
        let indices = slice.indices(total as isize)?;
        let mut result = Vec::new();
        let mut index = indices.start;
        if indices.step > 0 {
            while index < indices.stop {
                result.push(index as i64);
                index += indices.step;
            }
        } else {
            while index > indices.stop {
                result.push(index as i64);
                index += indices.step;
            }
        }
        return Ok(StepSelection::Indices(result));
    }
    if value.is_instance_of::<PyString>()
        || value.is_instance_of::<PyBytes>()
        || value.is_instance_of::<PyDict>()
    {
        return Err(PyTypeError::new_err(
            "a MED step selection must be None, an integer, a slice or a sequence of integers",
        ));
    }

    let mut indices = Vec::new();
    for item in value.try_iter().map_err(|_| {
        PyTypeError::new_err(
            "a MED step selection must be None, an integer, a slice or a sequence of integers",
        )
    })? {
        let item = item?;
        if item.is_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("MED step indices must be integers"));
        }
        indices.push(
            item.extract::<i64>()
                .map_err(|_| PyTypeError::new_err("MED step indices must be integers"))?,
        );
    }
    Ok(StepSelection::Indices(indices))
}
