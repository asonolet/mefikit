use pyo3::prelude::*;
use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
};

use mefikit::{
    mesh::{ElementIds, ElementType, FieldArcD},
    prelude as mf,
    tools::{
        Descendable, Measurable, MeshSelect, NodeDuplicates, Overlayable,
        fieldexpr::{MeshEvalUpdatable, MeshEvaluable},
    },
};

use std::path::Path;

use numpy::ndarray as nd;
use numpy::{self as np, PyReadonlyArray2};

use super::element::{etype_to_str, str_to_etype};
use crate::element_ids::PyElementIds;
use crate::{pyfield::PyField, select::PySelection};

#[pyclass(str)]
#[pyo3(name = "UMesh")]
#[derive(PartialEq)]
pub struct PyUMesh {
    inner: mf::UMesh,
}

#[derive(IntoPyObject)]
enum PyConnectivity<'py> {
    Regular(Bound<'py, np::PyArray2<usize>>),
    Poly(
        Bound<'py, np::PyArray1<usize>>,
        Bound<'py, np::PyArray1<usize>>,
    ),
}

#[pymethods]
impl PyUMesh {
    #[new]
    fn new(coords: np::PyReadonlyArray2<'_, f64>) -> Self {
        mf::UMesh::new(coords.as_array().to_shared()).into()
    }

    /// Returns a copy owned by python of the array coordinates
    fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, np::PyArray2<f64>> {
        np::PyArray2::from_array(py, &self.inner.coords())
    }

    fn block_types(&self) -> Vec<String> {
        self.inner
            .element_types()
            .map(|&et| etype_to_str(et))
            .collect()
    }

    fn blocks<'py>(&self, py: Python<'py>) -> BTreeMap<String, PyConnectivity<'py>> {
        self.inner
            .blocks()
            .map(|(&et, block)| {
                let et = etype_to_str(et);
                let conn = match &block.connectivity {
                    mf::Connectivity::Regular(c) => {
                        PyConnectivity::Regular(np::PyArray2::from_array(py, c))
                    }
                    mf::Connectivity::Poly(conn) => PyConnectivity::Poly(
                        np::PyArray1::from_array(py, &conn.data),
                        np::PyArray1::from_array(py, &conn.offsets),
                    ),
                };
                (et, conn)
            })
            .collect()
    }

    fn fields<'py>(
        &self,
        py: Python<'py>,
    ) -> BTreeMap<String, BTreeMap<String, Bound<'py, np::PyArray<f64, nd::IxDyn>>>> {
        self.inner
            .fields()
            .map(|(field_name, field)| {
                (
                    field_name,
                    field
                        .0
                        .iter()
                        .map(|(&et, block)| {
                            let et = etype_to_str(et);
                            let arr = np::PyArray::from_array(py, block);
                            (et, arr)
                        })
                        .collect(),
                )
            })
            .collect()
    }

    fn to_json(&self) -> String {
        serde_json::to_string(&self.inner).unwrap()
    }

    fn to_json_pretty(&self) -> String {
        serde_json::to_string_pretty(&self.inner).unwrap()
    }

    /// Add a regular block of elements to the mesh.
    #[pyo3(signature = (et, block, fields=None))]
    fn add_regular_block(
        &mut self,
        et: &str,
        block: np::PyReadonlyArray2<'_, usize>,
        fields: Option<BTreeMap<String, np::PyReadonlyArray<'_, f64, nd::IxDyn>>>,
    ) {
        let fields = fields.map(|f| {
            f.iter()
                .map(|(n, f)| (n.to_owned(), f.as_array().to_shared()))
                .collect()
        });
        self.inner
            .add_regular_block(str_to_etype(et), block.as_array().to_shared(), fields);
    }

    /// Add a field to the mesh.
    fn set_field(
        &mut self,
        name: &str,
        field: BTreeMap<String, np::PyReadonlyArray<'_, f64, nd::IxDyn>>,
    ) {
        let field: BTreeMap<ElementType, _> = field
            .iter()
            .map(|(et, f)| (str_to_etype(et), f.as_array().to_shared()))
            .collect();
        self.inner.update_field(name, FieldArcD::new(field));
    }

    #[staticmethod]
    fn read(path: &str) -> Self {
        let path = Path::new(path);
        mf::read(path).unwrap().into()
    }

    fn write(&self, path: &str) {
        let path = Path::new(path);
        let mesh = self.inner.view();
        let _ = mf::write(path, mesh);
    }

    #[pyo3(signature = (src_dim=None, target_dim=None))]
    fn descend(&self, src_dim: Option<usize>, target_dim: Option<usize>) -> Self {
        let src_dim = src_dim.map(|i| i.try_into().unwrap());
        let target_dim = target_dim.map(|i| i.try_into().unwrap());
        self.inner.descend(src_dim, target_dim).into()
    }

    #[pyo3(signature = (src_dim=None, target_dim=None))]
    fn descend_update(
        &mut self,
        src_dim: Option<usize>,
        target_dim: Option<usize>,
    ) -> Option<Self> {
        let src_dim = src_dim.map(|i| i.try_into().unwrap());
        let target_dim = target_dim.map(|i| i.try_into().unwrap());
        self.inner
            .descend_update(src_dim, target_dim)
            .map(|m| m.into())
    }

    #[pyo3(signature = (src_dim=None, target_dim=None))]
    fn boundaries(&self, src_dim: Option<usize>, target_dim: Option<usize>) -> Self {
        let src_dim = src_dim.map(|i| i.try_into().unwrap());
        let target_dim = target_dim.map(|i| i.try_into().unwrap());
        self.inner.boundaries(src_dim, target_dim).into()
    }

    #[pyo3(signature = (src_dim=None, target_dim=None))]
    fn boundaries_update(
        &mut self,
        src_dim: Option<usize>,
        target_dim: Option<usize>,
    ) -> Option<Self> {
        let src_dim = src_dim.map(|i| i.try_into().unwrap());
        let target_dim = target_dim.map(|i| i.try_into().unwrap());
        self.inner
            .boundaries_update(src_dim, target_dim)
            .map(|m| m.into())
    }

    #[pyo3(signature = (src_dim=None, link_dim=None, with_fields=true))]
    fn connected_components(
        &self,
        src_dim: Option<usize>,
        link_dim: Option<usize>,
        with_fields: bool,
    ) -> Vec<Self> {
        let src_dim = src_dim.map(|i| i.try_into().unwrap());
        let link_dim = link_dim.map(|i| i.try_into().unwrap());
        mf::compute_connected_components(&self.inner, src_dim, link_dim, with_fields)
            .into_iter()
            .map(|m| m.into())
            .collect()
    }

    fn measure<'py>(&self, py: Python<'py>) -> BTreeMap<String, Bound<'py, np::PyArray1<f64>>> {
        mf::measure(&self.inner.view(), None)
            .iter()
            .map(|(&et, arr)| (etype_to_str(et), np::PyArray1::from_array(py, arr)))
            .collect()
    }

    fn measure_update(&mut self) {
        self.inner.measure_update("Measure", None);
    }

    // Returns a copy owned by python of the array coordinates
    // fn fields<'py>(&self, py: Python<'py>) -> BTreeMap<String, np::PyField<f64>> {
    //     self.inner
    //         .fields()
    //         .map(|(n, f)| {
    //             let pyf = np::PyArray::from_array(py, f.into());
    //             (n, pyf)
    //         })
    //         .collect()
    // }

    fn crack(&self, cut_mesh: &PyUMesh) -> Self {
        mf::crack(self.inner.clone(), &cut_mesh.inner.view()).into()
    }

    #[pyo3(signature = (reference, eps=1e-12))]
    fn snap(&self, reference: &PyUMesh, eps: f64) -> Self {
        let mut snapped = self.inner.clone();
        snapped.snap_on(&reference.inner.view(), eps);
        snapped.into()
    }

    #[pyo3(signature = (eps=1e-12))]
    fn merge_nodes(&self, eps: f64) -> Self {
        let mut merged = self.inner.clone();
        merged.merge_nodes(eps);
        merged.into()
    }

    fn extrude(&self, along: &Bound<'_, PyAny>) -> PyResult<Self> {
        let along: Vec<f64> = along.extract()?;
        let new_mesh = mf::extrude(&self.inner.view(), &along);
        Ok(new_mesh.into())
    }

    fn extrude_parallel(&self, along: PyReadonlyArray2<'_, f64>) -> Self {
        let new_mesh = mf::extrude_parallel(&self.inner.view(), along.as_array());
        new_mesh.into()
    }

    fn extrude_curv(&self, along: PyReadonlyArray2<'_, f64>) -> Self {
        let new_mesh = mf::extrude_curv(&self.inner.view(), along.as_array());
        new_mesh.into()
    }

    #[pyo3(signature = (expr, with_fields=true))]
    fn select(&self, expr: PySelection, with_fields: bool) -> Self {
        let (_, submesh) = self.inner.select(expr.into(), with_fields);
        submesh.into()
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        expr: PyField,
    ) -> BTreeMap<String, Bound<'py, np::PyArray<f64, nd::IxDyn>>> {
        // TODO: manage level
        let f = self.inner.eval_field(None, expr.into());
        f.0.into_iter()
            .map(|(et, v)| (etype_to_str(et), np::PyArray::from_owned_array(py, v)))
            .collect()
    }

    fn eval_update(&mut self, name: &str, expr: PyField) {
        self.inner.eval_update_field(name, None, expr.into());
    }

    fn split(&self) -> Self {
        let new_mesh = mf::split(&self.inner.view());
        new_mesh.into()
    }

    fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    // ==================== Group Operations ====================

    /// Create a group from a selection expression.
    ///
    /// Evaluates the selection on this mesh and adds all matching elements
    /// to a named group. This is the primary way to create groups.
    fn select_to_group(&mut self, name: &str, expr: PySelection) {
        let eids = self.inner.select_ids(expr.into());
        self.inner.add_to_group(name, &eids);
    }

    /// Add elements to a group.
    ///
    /// Accepts either a Selection expression (elements matching the expression
    /// are added) or a dict of element IDs (e.g. {"QUAD4": [0, 1, 2]}).
    #[pyo3(signature = (name, source))]
    fn add_to_group(&mut self, name: &str, source: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(expr) = source.extract::<PySelection>() {
            let eids = self.inner.select_ids(expr.into());
            self.inner.add_to_group(name, &eids);
        } else if let Ok(dict) = source.cast::<pyo3::types::PyDict>() {
            let eids = PyElementIds::from_dict(dict);
            self.inner.add_to_group(name, &eids.into());
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "source must be a Selection or a dict of element IDs",
            ));
        }
        Ok(())
    }

    /// Remove elements from a group.
    ///
    /// Accepts either a Selection expression (elements matching the expression
    /// are removed) or a dict of element IDs (e.g. {"QUAD4": [0, 1, 2]}).
    #[pyo3(signature = (name, source))]
    fn remove_from_group(&mut self, name: &str, source: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(expr) = source.extract::<PySelection>() {
            let eids = self.inner.select_ids(expr.into());
            self.inner.remove_from_group(name, &eids);
        } else if let Ok(dict) = source.cast::<pyo3::types::PyDict>() {
            let eids = PyElementIds::from_dict(dict);
            self.inner.remove_from_group(name, &eids.into());
        } else {
            return Err(pyo3::exceptions::PyTypeError::new_err(
                "source must be a Selection or a dict of element IDs",
            ));
        }
        Ok(())
    }

    /// Delete a group entirely.
    fn delete_group(&mut self, name: &str) {
        self.inner.delete_group(name);
    }

    /// Rename a group.
    fn rename_group(&mut self, old_name: &str, new_name: &str) {
        self.inner.rename_group(old_name, new_name);
    }

    /// Replace all groups at once from a dict.
    ///
    /// Each key is a group name, each value is a dict mapping element type
    /// strings to lists of element indices.
    fn set_groups(&mut self, groups: BTreeMap<String, BTreeMap<String, Vec<usize>>>) {
        let mut rust_groups: BTreeMap<String, ElementIds> = BTreeMap::new();
        for (name, type_map) in groups {
            let mut eids = ElementIds::new();
            for (et_str, indices) in type_map {
                eids.add_block(str_to_etype(&et_str), indices);
            }
            rust_groups.insert(name, eids);
        }
        self.inner.set_groups(rust_groups);
    }

    /// List all group names.
    fn group_names(&self) -> Vec<String> {
        self.inner.group_names()
    }

    /// Check if a group exists.
    fn has_group(&self, name: &str) -> bool {
        self.inner.has_group(name)
    }

    /// Computes the boolean overlay of this mesh (as mesh1) with `mesh2`.
    ///
    /// The operation defaults to `OverlayOperation.IMPRINT` which refines the domain of
    /// `self` with the edges of `mesh2`.
    #[pyo3(signature = (mesh2, operation=None))]
    fn overlay(&self, mesh2: &PyUMesh, operation: Option<PyOverlayOperation>) -> PyResult<PyUMesh> {
        let operation = operation.unwrap_or(PyOverlayOperation::Imprint).into();
        let result = self.inner.overlay(mesh2.inner.clone(), operation);
        Ok(result.into())
    }

    /// Imprints this surface mesh with `mesh2` wherever the two coincide in 3D space.
    ///
    /// Both meshes must be 2D meshes embedded in 3D space (TRI3, QUAD4 or PGON faces).
    /// The two refined surfaces returned in the `SurfaceOverlay` result share the same
    /// coordinates array, so intersection nodes exist once and both sides become mutually
    /// conformal on the coincident areas. Areas not covered by the other surface are
    /// copied verbatim. Raises `ValueError` when coplanar patches only partially overlap.
    #[pyo3(signature = (mesh2, tol=1e-9))]
    fn overlay_surfaces(&self, mesh2: &PyUMesh, tol: f64) -> PyResult<PySurfaceOverlay> {
        let out = self
            .inner
            .overlay_surfaces(&mesh2.inner.view(), tol)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(out.into())
    }
}

pub fn into_view(mesh: &PyUMesh) -> mf::UMeshView<'_> {
    mesh.inner.view()
}

pub fn into_mut(mesh: &mut PyUMesh) -> &mut mf::UMesh {
    &mut mesh.inner
}

impl Display for PyUMesh {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::UMesh> for PyUMesh {
    fn from(umesh: mf::UMesh) -> Self {
        PyUMesh { inner: umesh }
    }
}

impl From<PyUMesh> for mf::UMesh {
    fn from(pyumesh: PyUMesh) -> Self {
        pyumesh.inner
    }
}

impl<'a> From<&'a PyUMesh> for &'a mf::UMesh {
    fn from(pyumesh: &'a PyUMesh) -> Self {
        &pyumesh.inner
    }
}

/// Boolean-like operation to perform on two 2D meshes.
#[pyclass(eq, eq_int, from_py_object, name = "OverlayOperation")]
#[derive(Clone, Copy, PartialEq)]
pub enum PyOverlayOperation {
    /// Refine `mesh1` with the edges of `mesh2` while keeping `mesh1`'s domain.
    #[pyo3(name = "IMPRINT")]
    Imprint,
    /// Keep the domain covered by at least one of the two meshes.
    #[pyo3(name = "UNION")]
    Union,
    /// Keep the domain covered by both meshes.
    #[pyo3(name = "INTERSECTION")]
    Intersection,
    /// Keep the domain of `mesh1` not covered by `mesh2`.
    #[pyo3(name = "DIFFERENCE")]
    Difference,
    /// Keep the domain covered by exactly one of the two meshes.
    #[pyo3(name = "SYMMETRIC_DIFFERENCE")]
    SymmetricDifference,
}

impl From<PyOverlayOperation> for mf::OverlayOperation {
    fn from(op: PyOverlayOperation) -> Self {
        match op {
            PyOverlayOperation::Imprint => mf::OverlayOperation::Imprint,
            PyOverlayOperation::Union => mf::OverlayOperation::Union,
            PyOverlayOperation::Intersection => mf::OverlayOperation::Intersection,
            PyOverlayOperation::Difference => mf::OverlayOperation::Difference,
            PyOverlayOperation::SymmetricDifference => mf::OverlayOperation::SymmetricDifference,
        }
    }
}

/// Result of `UMesh.overlay_surfaces`.
///
/// `refined1` and `refined2` hold the imprinted faces of the first and second input
/// surface respectively; they share the same coordinates array. The parents maps relate
/// each input face `(type, index)` to the elements it produced in the refined mesh.
#[pyclass]
#[pyo3(name = "SurfaceOverlay")]
pub struct PySurfaceOverlay {
    inner: mf::SurfaceOverlay,
}

/// Parent map key/value type: element ids as `(type name, index)` pairs.
type PyParents = BTreeMap<(String, usize), Vec<(String, usize)>>;

fn parents_to_py<'a>(
    parents: impl IntoIterator<Item = (&'a mf::ElementId, &'a Vec<mf::ElementId>)>,
) -> PyParents {
    parents
        .into_iter()
        .map(|(face, pieces)| {
            let key = (etype_to_str(face.element_type()), face.index());
            let pieces = pieces
                .iter()
                .map(|id| (etype_to_str(id.element_type()), id.index()))
                .collect();
            (key, pieces)
        })
        .collect()
}

#[pymethods]
impl PySurfaceOverlay {
    /// Refined faces of the first input surface.
    #[getter]
    fn refined1(&self) -> PyUMesh {
        self.inner.refined1.clone().into()
    }

    /// Refined faces of the second input surface.
    #[getter]
    fn refined2(&self) -> PyUMesh {
        self.inner.refined2.clone().into()
    }

    /// Input face id -> produced elements, for the first surface.
    #[getter]
    fn parents1(&self) -> PyParents {
        parents_to_py(&self.inner.parents1)
    }

    /// Input face id -> produced elements, for the second surface.
    #[getter]
    fn parents2(&self) -> PyParents {
        parents_to_py(&self.inner.parents2)
    }
}

impl From<mf::SurfaceOverlay> for PySurfaceOverlay {
    fn from(inner: mf::SurfaceOverlay) -> Self {
        PySurfaceOverlay { inner }
    }
}
