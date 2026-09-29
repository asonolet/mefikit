//! Python bindings for geometry transforms (`mefikit.Transform`).
//!
//! A `Transform` is a 4x4 homogeneous matrix value; compose with `@` and apply
//! it to meshes with the mesh methods (`mesh.transform(tr)`, `mesh.translate(...)`,
//! ...) or with the module-level `aggregate` / `concat` helpers.
//!
//! Angles are in **radians**.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use mefikit::prelude as mf;
use numpy::{self as np, PyReadonlyArray2};

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
