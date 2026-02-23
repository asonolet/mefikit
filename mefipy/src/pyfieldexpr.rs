use numpy::{self as np, PyArrayMethods};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use std::fmt::{Display, Formatter};

use mefikit::{prelude as mf, tools::sel::Comparable};

use super::select::PySelection;

#[pyclass(str)]
#[pyo3(name = "Field")]
#[derive(Clone)]
pub struct PyFieldExpr {
    inner: mf::FieldExpr,
}

impl Display for PyFieldExpr {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::FieldExpr> for PyFieldExpr {
    fn from(expr: mf::FieldExpr) -> Self {
        PyFieldExpr { inner: expr }
    }
}

impl From<PyFieldExpr> for mf::FieldExpr {
    fn from(pyexpr: PyFieldExpr) -> Self {
        pyexpr.inner
    }
}

fn any_to_fieldexp<'py>(val: Bound<'py, PyAny>) -> PyResult<mf::FieldExpr> {
    if let Ok(name) = val.extract::<&str>() {
        Ok(mf::field(name))
    } else if let Ok(arr) = val.extract::<Bound<'py, np::PyArrayDyn<f64>>>() {
        let arr = arr.readonly().as_array().to_owned();
        Ok(mf::arr(arr))
    } else {
        Err(PyTypeError::new_err(
            "Cannot convert this type to fieldexpr (either sting or array).",
        ))
    }
}

#[pymethods]
impl PyFieldExpr {
    #[new]
    pub fn new(val: Bound<'_, PyAny>) -> PyResult<PyFieldExpr> {
        let fieldexp = any_to_fieldexp(val)?;
        Ok(PyFieldExpr { inner: fieldexp })
    }
    pub fn sin(&self) -> Self {
        self.inner.clone().sin().into()
    }
    pub fn cos(&self) -> Self {
        self.inner.clone().cos().into()
    }
    pub fn sqrt(&self) -> Self {
        self.inner.clone().sqrt().into()
    }
    pub fn square(&self) -> Self {
        self.inner.clone().square().into()
    }
    pub fn exp(&self) -> Self {
        self.inner.clone().exp().into()
    }
    pub fn ln(&self) -> Self {
        self.inner.clone().ln().into()
    }
    pub fn log10(&self) -> Self {
        self.inner.clone().log10().into()
    }
    pub fn tan(&self) -> Self {
        self.inner.clone().tan().into()
    }
    pub fn abs(&self) -> Self {
        self.inner.clone().abs().into()
    }
    pub fn __add__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() + rhs).into())
    }
    pub fn __radd__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() + rhs).into())
    }
    pub fn __sub__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() - rhs).into())
    }
    pub fn __rsub__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((rhs - self.inner.clone()).into())
    }
    pub fn __mul__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() * rhs).into())
    }
    pub fn __rmul__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() * rhs).into())
    }
    pub fn __div__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((self.inner.clone() / rhs).into())
    }
    pub fn __rdiv__(&self, rhs: Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok((rhs / self.inner.clone()).into())
    }
    pub fn __ge__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().geq(rhs)).into())
    }
    pub fn __gt__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().gt(rhs)).into())
    }
    pub fn __le__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().leq(rhs)).into())
    }
    pub fn __lt__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().lt(rhs)).into())
    }
    pub fn __eq__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().eq(rhs)).into())
    }
    pub fn __ne__(&self, rhs: Bound<'_, PyAny>) -> PyResult<PySelection> {
        let rhs = any_to_fieldexp(rhs)?;
        Ok(mf::Selection::FieldSelection(self.inner.clone().neq(rhs)).into())
    }
}
