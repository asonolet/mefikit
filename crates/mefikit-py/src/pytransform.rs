//! Python bindings for geometry transforms (`mefikit.Transform`).
//!
//! A `Transform` is a 4x4 homogeneous matrix value; compose with `@` and apply
//! it to meshes with the mesh methods (`mesh.transform(tr)`, `mesh.translate(...)`,
//! ...) or with the module-level `aggregate` / `concat` helpers.
//!
//! Angles are in **radians**.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use mefikit::prelude as mf;
use numpy::{self as np, PyReadonlyArray2};

use super::element_ids::ids_to_pydict;
use super::pyumesh::PyUMesh;

/// An affine transform encoded as a homogeneous 4x4 matrix.
///
/// Coordinates are column vectors: `p' = M * [p; 1]`. Simple operations are
/// provided as classmethods (`Transform.translation`, `Transform.rotation`, ...);
/// any combination is the matrix product `tr1 @ tr2`. Angles are in radians.
#[pyclass]
#[pyo3(name = "Transform")]
pub struct PyTransform {
    pub(crate) inner: mf::Transform,
}

fn map_err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

impl From<mf::Transform> for PyTransform {
    fn from(inner: mf::Transform) -> Self {
        PyTransform { inner }
    }
}

#[pymethods]
impl PyTransform {
    /// Builds a transform from an explicit 4x4 homogeneous matrix.
    #[new]
    fn new(mat: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::from_matrix(mat.as_array()).map_err(map_err)?,
        })
    }

    /// The identity transform.
    #[staticmethod]
    fn identity() -> Self {
        PyTransform {
            inner: mf::Transform::identity(),
        }
    }

    /// Translation by a 1-, 2- or 3-component vector.
    #[staticmethod]
    fn translation(v: Vec<f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::translation(&v).map_err(map_err)?,
        })
    }

    /// Non-uniform scaling by 1-, 2- or 3-component factors.
    #[staticmethod]
    fn scaling(factors: Vec<f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::scaling(&factors).map_err(map_err)?,
        })
    }

    /// Rotation by `angle` radians around an axis through the origin.
    #[staticmethod]
    fn rotation(axis: Vec<f64>, angle: f64) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::rotation(&axis, angle).map_err(map_err)?,
        })
    }

    /// Rotation by `angle` radians around an axis through `center`.
    #[staticmethod]
    fn rotation_about(center: Vec<f64>, axis: Vec<f64>, angle: f64) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::rotation_about(&center, &axis, angle).map_err(map_err)?,
        })
    }

    /// Reflection through a plane through the origin with the given normal.
    #[staticmethod]
    fn reflection(normal: Vec<f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::reflection(&normal).map_err(map_err)?,
        })
    }

    /// Reflection through a plane through `point` with the given normal.
    #[staticmethod]
    fn reflection_about(point: Vec<f64>, normal: Vec<f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::reflection_about(&point, &normal).map_err(map_err)?,
        })
    }

    /// Builds a transform from an explicit 4x4 homogeneous matrix.
    #[staticmethod]
    fn from_matrix(mat: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        Ok(PyTransform {
            inner: mf::Transform::from_matrix(mat.as_array()).map_err(map_err)?,
        })
    }

    /// The 4x4 homogeneous matrix backing this transform.
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, np::PyArray2<f64>> {
        np::PyArray2::from_array(py, &self.inner.matrix())
    }

    /// Composes as "apply `self` first, then `after`" (left-to-right reading).
    fn then(&self, after: &PyTransform) -> PyTransform {
        PyTransform {
            inner: self.inner.then(&after.inner),
        }
    }

    /// The inverse affine transform.
    fn inverse(&self) -> PyResult<PyTransform> {
        Ok(PyTransform {
            inner: self.inner.inverse().map_err(map_err)?,
        })
    }

    /// Applies this transform to a mesh.
    fn apply(&self, mesh: &PyUMesh) -> PyResult<PyUMesh> {
        mf::transform(&mesh.inner.view(), &self.inner)
            .map_err(map_err)
            .map(Into::into)
    }

    /// Matrix-product composition: `tr1 @ tr2` applies `tr2` first.
    fn __matmul__(&self, rhs: &PyTransform) -> PyTransform {
        PyTransform {
            inner: &self.inner * &rhs.inner,
        }
    }

    fn __repr__(&self) -> String {
        format!("Transform({:?})", self.inner.matrix().as_slice().unwrap())
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.inner.matrix())
    }
}

/// Joins several non-intersecting meshes sharing the same space dimension.
///
/// Coordinates are concatenated, node indices are offset, and equal element
/// types become one block (fields and groups are preserved).
#[pyfunction]
pub fn aggregate(meshes: Vec<PyRef<'_, PyUMesh>>) -> PyResult<PyUMesh> {
    let views: Vec<mf::UMeshView> = meshes.iter().map(|m| m.inner.view()).collect();
    mf::aggregate(&views).map_err(map_err).map(Into::into)
}

/// Joins exactly two non-intersecting meshes (see `aggregate`).
#[pyfunction]
pub fn concat<'py>(a: PyRef<'py, PyUMesh>, b: PyRef<'py, PyUMesh>) -> PyResult<PyUMesh> {
    mf::concat(&a.inner.view(), &b.inner.view())
        .map_err(map_err)
        .map(Into::into)
}

/// Stitches two or more volume meshes into a single conforming polyhedral mesh.
///
/// The meshes must be 3D volume meshes (`TET4`, `HEX8` or `PHED` cells) lying in
/// the same coordinate space. Boundary faces that two or more meshes have in
/// common (within `tol`) are refined so that they become mutually conformal, and
/// the resulting interface nodes are shared. The result is a single `PHED` mesh
/// with a shared coordinates array: families are relabeled per input mesh, while
/// fields and groups are dropped.
///
/// Raises `ValueError` if fewer than two meshes are given, if a mesh is not a
/// 3D volume mesh, if a coincident region is not planar within `tol`, or if a
/// quadratic cell is encountered.
#[pyfunction]
#[pyo3(signature = (meshes, tol=1e-9))]
pub fn stitch(meshes: Vec<PyRef<'_, PyUMesh>>, tol: f64) -> PyResult<PyUMesh> {
    let views: Vec<mf::UMeshView> = meshes.iter().map(|m| m.inner.view()).collect();
    mf::stitch(&views, tol)
        .map_err(|e| map_err(e.to_string()))
        .map(Into::into)
}

/// Conformizes a single 3D volume mesh: `stitch` applied to a mesh with itself.
///
/// The mesh is split into its *parts* (groups of cells linked by a chain of conforming faces,
/// so two blocks merely touching through their own copies of the interface nodes are two parts),
/// and those parts are made conformal to each other exactly as `stitch` would do for separate
/// meshes. A mesh that is already conformal comes back unchanged, apart from the conversion of
/// its cells to `PHED`.
///
/// The result is a single `PHED` mesh whose families are relabeled (per part), with fields and
/// groups dropped. Overlapping volumes are not detected and faces shared by more than two cells
/// are not repaired; see `is_conform` for the diagnostic.
///
/// Raises `ValueError` if the mesh is not a 3D volume mesh, if a coincident region is not
/// planar within `tol`, or if a quadratic cell is encountered.
#[pyfunction]
#[pyo3(signature = (mesh, tol=1e-9))]
pub fn conformize(mesh: &PyUMesh, tol: f64) -> PyResult<PyUMesh> {
    mf::conformize(&mesh.inner.view(), tol)
        .map_err(|e| map_err(e.to_string()))
        .map(Into::into)
}

/// Checks whether a mesh is internally conformal and, when it is not, reports why and where.
///
/// Reported defects: a face shared by more than two cells, two nodes within `tol` that cells
/// meeting only through those nodes should have shared (the case `conformize` repairs), and
/// input `conformize` would reject (not embedded in 3D, no volume cell, unsupported cell type).
/// Interfaces that overlap without sharing any coincident node are not detected: recognizing
/// them costs the geometric imprinting `conformize` performs.
///
/// `max_issues` caps the length of `report.issues` (10 by default, `None` keeps everything);
/// `report.n_issues` always holds the true total. Raises `ValueError` on an invalid tolerance.
#[pyfunction]
#[pyo3(signature = (mesh, tol=1e-9, max_issues=10))]
pub fn is_conform(
    mesh: &PyUMesh,
    tol: f64,
    max_issues: Option<usize>,
) -> PyResult<PyConformanceReport> {
    let report =
        mf::is_conform(&mesh.inner.view(), tol, max_issues).map_err(|e| map_err(e.to_string()))?;
    Ok(PyConformanceReport { inner: report })
}

/// The outcome of `is_conform`: whether the mesh is conformal, and if not, why and where.
#[pyclass]
#[pyo3(name = "ConformanceReport")]
pub struct PyConformanceReport {
    inner: mf::ConformanceReport,
}

/// The cells of an issue, as the `{element type: indices}` mapping used everywhere else.
fn cells_to_dict<'py>(py: Python<'py>, cells: &[mf::ElementId]) -> Bound<'py, PyDict> {
    let mut eids = mf::ElementIds::new();
    for id in cells {
        eids.add(id.element_type(), id.index());
    }
    ids_to_pydict(py, &eids)
}

impl PyConformanceReport {
    /// Turns one issue into a dict: `kind`, the variant fields (`center`, `cells`, ...), and a
    /// ready-made human-readable `message`.
    fn issue_dict<'py>(
        py: Python<'py>,
        issue: &mf::ConformanceIssue,
    ) -> PyResult<Bound<'py, PyDict>> {
        use mf::ConformanceIssue;

        let dict = PyDict::new(py);
        match issue {
            ConformanceIssue::MergedNodes {
                center,
                nodes,
                cells_a,
                cells_b,
            } => {
                dict.set_item("kind", "merged_nodes")?;
                dict.set_item("center", center.to_vec())?;
                dict.set_item("nodes", nodes.to_vec())?;
                dict.set_item("cells_a", cells_to_dict(py, cells_a))?;
                dict.set_item("cells_b", cells_to_dict(py, cells_b))?;
            }
            ConformanceIssue::OverlappingFaces { center, cells } => {
                dict.set_item("kind", "overlapping_faces")?;
                dict.set_item("center", center.to_vec())?;
                dict.set_item("cells", cells_to_dict(py, cells))?;
            }
            ConformanceIssue::InvalidSpaceDimension { found } => {
                dict.set_item("kind", "invalid_space_dimension")?;
                dict.set_item("found", found)?;
            }
            ConformanceIssue::UnsupportedCells { cells } => {
                dict.set_item("kind", "unsupported_cells")?;
                dict.set_item("cells", cells_to_dict(py, cells))?;
            }
            ConformanceIssue::NoVolumeCells => {
                dict.set_item("kind", "no_volume_cells")?;
            }
        }
        dict.set_item("message", issue.to_string())?;
        Ok(dict)
    }
}

#[pymethods]
impl PyConformanceReport {
    /// Whether the mesh is conformal.
    #[getter]
    fn is_conform(&self) -> bool {
        self.inner.is_conform()
    }

    /// The total number of problems found, which may be larger than `len(issues)`.
    #[getter]
    fn n_issues(&self) -> usize {
        self.inner.n_issues
    }

    /// Whether `issues` was capped by `max_issues`, in which case it does not list everything.
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated
    }

    /// The problems found, ordered from the most structural to the most numerical. Each issue is
    /// a dict: `kind` (`merged_nodes`, `overlapping_faces`, `invalid_space_dimension`,
    /// `unsupported_cells`, `no_volume_cells`), the variant fields (`center` as `xyz` coordinates,
    /// `nodes`, `cells` / `cells_a` / `cells_b` as `{element type: indices}` mappings), and
    /// `message`.
    #[getter]
    fn issues<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let list = PyList::empty(py);
        for issue in &self.inner.issues {
            list.append(Self::issue_dict(py, issue)?)?;
        }
        Ok(list)
    }

    fn __bool__(&self) -> bool {
        self.inner.is_conform()
    }

    fn __repr__(&self) -> String {
        format!(
            "ConformanceReport(n_issues={}, truncated={})",
            self.inner.n_issues, self.inner.truncated
        )
    }
}

/// Helper used by `PyUMesh.transform`: accepts either a `Transform` or a raw
/// 4x4 homogeneous numpy matrix.
pub fn extract_transform(arg: &Bound<'_, PyAny>) -> PyResult<mf::Transform> {
    if let Ok(tr) = arg.cast::<PyTransform>() {
        return Ok(tr.borrow().inner.clone());
    }
    if let Ok(mat) = arg.extract::<PyReadonlyArray2<'_, f64>>() {
        return mf::Transform::from_matrix(mat.as_array()).map_err(map_err);
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected a Transform or a 4x4 homogeneous matrix.",
    ))
}
