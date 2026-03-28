use std::fmt::{Display, Formatter};

use mefikit::tools::sel::Comparable;
use numpy as np;
use numpy::ndarray as nd;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use mefikit::prelude as mf;
use mefikit::prelude::fieldexpr::FieldExpr;

use crate::select::PySelection;

#[pyclass(str)]
#[pyo3(name = "Field")]
#[derive(Clone)]
pub struct PyField {
    inner: FieldExpr,
}

impl Display for PyField {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<FieldExpr> for PyField {
    fn from(fieldexpr: FieldExpr) -> Self {
        PyField { inner: fieldexpr }
    }
}

impl From<PyField> for FieldExpr {
    fn from(pyfield: PyField) -> Self {
        pyfield.inner
    }
}

impl<'py> TryFrom<&Bound<'py, PyAny>> for PyField {
    type Error = PyErr;

    fn try_from(value: &Bound<'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(v) = value.extract::<PyField>() {
            return Ok(v);
        }
        if let Ok(v) = value.extract::<String>() {
            return Ok(mf::fieldexpr::field(&v).into());
        }
        if let Ok(v) = value.extract::<np::PyReadonlyArray0<f64>>() {
            return Ok(mf::fieldexpr::arr(v.as_array().into_owned()).into());
        }
        if let Ok(v) = value.extract::<f64>() {
            return Ok(mf::fieldexpr::arr(nd::arr0(v)).into());
        }
        if let Ok(v) = value.extract::<np::PyReadonlyArray1<f64>>() {
            return Ok(mf::fieldexpr::arr(v.as_array().into_owned()).into());
        }
        if let Ok(v) = value.extract::<Vec<f64>>() {
            return Ok(mf::fieldexpr::arr(nd::arr1(&v)).into());
        }
        Err(PyTypeError::new_err("Could not convert to PyField"))
    }
}

#[pymethods]
impl PyField {
    #[new]
    pub fn new<'py>(any: &Bound<'py, PyAny>) -> PyResult<PyField> {
        any.try_into()
    }
    pub fn __add__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((self.inner.clone() + pyf.inner).into())
    }
    pub fn __sub__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((self.inner.clone() - pyf.inner).into())
    }
    pub fn __mul__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((self.inner.clone() * pyf.inner).into())
    }
    pub fn __div__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((self.inner.clone() / pyf.inner).into())
    }
    pub fn __radd__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((pyf.inner + self.inner.clone()).into())
    }
    pub fn __rsub__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((pyf.inner - self.inner.clone()).into())
    }
    pub fn __rmul__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((pyf.inner * self.inner.clone()).into())
    }
    pub fn __rdiv__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PyField> {
        let pyf: PyField = other.try_into()?;
        Ok((pyf.inner / self.inner.clone()).into())
    }
    pub fn abs(&self) -> PyField {
        self.inner.clone().abs().into()
    }
    pub fn tan(&self) -> PyField {
        self.inner.clone().tan().into()
    }
    pub fn exp(&self) -> PyField {
        self.inner.clone().exp().into()
    }
    pub fn cos(&self) -> PyField {
        self.inner.clone().cos().into()
    }
    pub fn sin(&self) -> PyField {
        self.inner.clone().sin().into()
    }
    pub fn log10(&self) -> PyField {
        self.inner.clone().log10().into()
    }
    pub fn ln(&self) -> PyField {
        self.inner.clone().ln().into()
    }
    pub fn square(&self) -> PyField {
        self.inner.clone().square().into()
    }
    pub fn sqrt(&self) -> PyField {
        self.inner.clone().sqrt().into()
    }
    pub fn __gt__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PySelection> {
        let pyf: PyField = other.try_into()?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().gt(pyf.inner)).into())
    }
    pub fn __ge__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PySelection> {
        let pyf: PyField = other.try_into()?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().geq(pyf.inner)).into())
    }
    pub fn __lt__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PySelection> {
        let pyf: PyField = other.try_into()?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().lt(pyf.inner)).into())
    }
    pub fn __le__<'py>(&'py self, other: &Bound<'py, PyAny>) -> PyResult<PySelection> {
        let pyf: PyField = other.try_into()?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().leq(pyf.inner)).into())
    }
}
