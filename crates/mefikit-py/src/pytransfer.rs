use pyo3::prelude::*;
use std::fmt::{Display, Formatter};

use mefikit::{prelude as mf, tools::Transfer};

use crate::pyumesh::{PyUMesh, into_mut, into_view};

#[pyclass(from_py_object)]
#[pyo3(name = "DistanceWeighting")]
#[derive(Clone)]
pub enum PyDistanceWeighting {
    None(),
    InverseDistance { exponent: f64 },
    Gaussian(),
}

impl From<PyDistanceWeighting> for mf::DistanceWeighting {
    fn from(weighting: PyDistanceWeighting) -> Self {
        match weighting {
            PyDistanceWeighting::None() => mf::DistanceWeighting::None,
            PyDistanceWeighting::InverseDistance { exponent } => {
                mf::DistanceWeighting::InverseDistance { exponent }
            }
            PyDistanceWeighting::Gaussian() => mf::DistanceWeighting::Gaussian,
        }
    }
}

#[pyclass(str, from_py_object)]
#[pyo3(name = "ConstantPiecewise")]
#[derive(Clone)]
pub struct PyConstantPiecewise {
    inner: mf::ConstantPiecewiseTransfer,
}

impl Display for PyConstantPiecewise {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::ConstantPiecewiseTransfer> for PyConstantPiecewise {
    fn from(transfer: mf::ConstantPiecewiseTransfer) -> Self {
        PyConstantPiecewise { inner: transfer }
    }
}

impl From<PyConstantPiecewise> for mf::ConstantPiecewiseTransfer {
    fn from(pytransfer: PyConstantPiecewise) -> Self {
        pytransfer.inner
    }
}

#[pymethods]
impl PyConstantPiecewise {
    #[new]
    fn new(src_mesh: &PyUMesh, tgt_mesh: &PyUMesh) -> Self {
        mf::ConstantPiecewiseTransfer::new(
            &into_view(src_mesh),
            &into_view(tgt_mesh),
            mf::PointLocation::Centroid,
        )
        .into()
    }

    #[pyo3(signature = (src_mesh, field_name, tgt_mesh, tgt_field_name=None, def_val=0.0))]
    fn apply_update(
        &self,
        src_mesh: &PyUMesh,
        field_name: &str,
        tgt_mesh: &mut PyUMesh,
        tgt_field_name: Option<&str>,
        def_val: f64,
    ) {
        let name = tgt_field_name.unwrap_or(field_name);
        let src_view = into_view(src_mesh);
        let field = src_view.field(field_name, None).unwrap();
        let field_nature = mf::FieldNature::Intensive;
        self.inner
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }
}

#[pyclass(str, from_py_object)]
#[pyo3(name = "MovingLeastSquares")]
#[derive(Clone)]
pub struct PyMovingLeastSquares {
    inner: mf::MovingLeastSquaresTransfer,
}

impl Display for PyMovingLeastSquares {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::MovingLeastSquaresTransfer> for PyMovingLeastSquares {
    fn from(transfer: mf::MovingLeastSquaresTransfer) -> Self {
        PyMovingLeastSquares { inner: transfer }
    }
}

impl From<PyMovingLeastSquares> for mf::MovingLeastSquaresTransfer {
    fn from(pytransfer: PyMovingLeastSquares) -> Self {
        pytransfer.inner
    }
}

#[pymethods]
impl PyMovingLeastSquares {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, k=10, weighting=PyDistanceWeighting::None()))]
    fn new(
        src_mesh: &PyUMesh,
        tgt_mesh: &PyUMesh,
        k: usize,
        weighting: PyDistanceWeighting,
    ) -> Self {
        mf::MovingLeastSquaresTransfer::new(
            &into_view(src_mesh),
            &into_view(tgt_mesh),
            k,
            weighting.into(),
        )
        .into()
    }

    #[pyo3(signature = (src_mesh, field_name, tgt_mesh, tgt_field_name=None, def_val=0.0))]
    fn apply_update(
        &self,
        src_mesh: &PyUMesh,
        field_name: &str,
        tgt_mesh: &mut PyUMesh,
        tgt_field_name: Option<&str>,
        def_val: f64,
    ) {
        let name = tgt_field_name.unwrap_or(field_name);
        let src_view = into_view(src_mesh);
        let field = src_view.field(field_name, None).unwrap();
        let field_nature = mf::FieldNature::Intensive;
        self.inner
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }
}

#[pyclass(str, from_py_object)]
#[pyo3(name = "InverseDistance")]
#[derive(Clone)]
pub struct PyInverseDistance {
    inner: mf::InverseDistanceTransfer,
}

impl Display for PyInverseDistance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::InverseDistanceTransfer> for PyInverseDistance {
    fn from(transfer: mf::InverseDistanceTransfer) -> Self {
        PyInverseDistance { inner: transfer }
    }
}

impl From<PyInverseDistance> for mf::InverseDistanceTransfer {
    fn from(pytransfer: PyInverseDistance) -> Self {
        pytransfer.inner
    }
}

#[pymethods]
impl PyInverseDistance {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, k=4, exponent=2.0))]
    fn new(src_mesh: &PyUMesh, tgt_mesh: &PyUMesh, k: usize, exponent: f64) -> Self {
        mf::InverseDistanceTransfer::new(&into_view(src_mesh), &into_view(tgt_mesh), k, exponent)
            .into()
    }

    #[pyo3(signature = (src_mesh, field_name, tgt_mesh, tgt_field_name=None, def_val=0.0))]
    fn apply_update(
        &self,
        src_mesh: &PyUMesh,
        field_name: &str,
        tgt_mesh: &mut PyUMesh,
        tgt_field_name: Option<&str>,
        def_val: f64,
    ) {
        let name = tgt_field_name.unwrap_or(field_name);
        let src_view = into_view(src_mesh);
        let field = src_view.field(field_name, None).unwrap();
        let field_nature = mf::FieldNature::Intensive;
        self.inner
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }
}
