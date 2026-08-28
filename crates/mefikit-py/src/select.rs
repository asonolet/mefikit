use numpy::ndarray as nd;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::{Display, Formatter};

use mefikit::mesh::Dimension;
use mefikit::prelude as mf;
use mefikit::tools::{
    MeshSelect,
    fieldexpr::{FieldExpr, MeshEvaluable},
};

use super::element::str_to_etype;
use super::element_ids::{PyElementIds, ids_to_pydict};
use super::pyfield::PyField;
use super::pyumesh::PyUMesh;
use crate::pyfields::{
    ReduceOp, measure_weighted, reduce_blocks, reduce_to_py, validate_expr_fields,
};

#[pyclass(str, from_py_object)]
#[pyo3(name = "Selection")]
#[derive(Clone)]
pub struct PySelection {
    inner: mf::Selection,
}

impl Display for PySelection {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.inner)
    }
}

impl From<mf::Selection> for PySelection {
    fn from(sel: mf::Selection) -> Self {
        PySelection { inner: sel }
    }
}

impl From<PySelection> for mf::Selection {
    fn from(pysel: PySelection) -> Self {
        pysel.inner
    }
}

#[pyfunction]
pub fn nbbox(min: [f64; 3], max: [f64; 3], all: bool) -> PySelection {
    mf::sel::nbbox(min, max, all).into()
}
#[pyfunction]
pub fn nrect(min: [f64; 2], max: [f64; 2], all: bool) -> PySelection {
    mf::sel::nrect(min, max, all).into()
}
/// This method filters upon nodes position.
#[pyfunction]
pub fn nsphere(center: [f64; 3], r2: f64, all: bool) -> PySelection {
    mf::sel::nsphere(center, r2, all).into()
}
#[pyfunction]
pub fn ncircle(center: [f64; 2], r2: f64, all: bool) -> PySelection {
    mf::sel::ncircle(center, r2, all).into()
}
#[pyfunction]
pub fn nids(ids: Vec<usize>, all: bool) -> PySelection {
    mf::sel::nids(ids, all).into()
}
#[pyfunction]
pub fn bbox(min: [f64; 3], max: [f64; 3]) -> PySelection {
    mf::sel::bbox(min, max).into()
}
#[pyfunction]
pub fn rect(min: [f64; 2], max: [f64; 2]) -> PySelection {
    mf::sel::rect(min, max).into()
}
#[pyfunction]
pub fn sphere(center: [f64; 3], r2: f64) -> PySelection {
    mf::sel::sphere(center, r2).into()
}
#[pyfunction]
pub fn circle(center: [f64; 2], r2: f64) -> PySelection {
    mf::sel::circle(center, r2).into()
}
// TODO: Enable ElementType and Dimension exposure to Python
// #[pyfunction]
// pub fn types(elems: Vec<ElementType>) -> PySelection {
//     mf::sel::types(elems).into()
// }
// #[pyfunction]
// pub fn dimensions(dims: Vec<Dimension>) -> PySelection {
//     mf::sel::dimensions(dims).into()
// }
#[pyfunction]
pub fn all() -> PySelection {
    mf::sel::all().into()
}
#[pyfunction]
pub fn ids<'py>(eids: Bound<'py, PyDict>) -> PySelection {
    let eids = PyElementIds::from_dict(&eids);
    mf::sel::ids(eids.into()).into()
}
#[pyfunction]
pub fn group(name: &str) -> PySelection {
    mf::sel::group(name).into()
}
#[pyfunction]
pub fn exclude_group(name: &str) -> PySelection {
    mf::sel::exclude_group(name).into()
}
#[pyfunction]
pub fn types(types_str: Vec<String>) -> PySelection {
    let elems: Vec<mf::ElementType> = types_str.iter().map(|s| str_to_etype(s)).collect();
    mf::sel::types(elems).into()
}

#[pymethods]
impl PySelection {
    pub fn __and__(&self, other: &PySelection) -> PySelection {
        (self.inner.clone() & other.inner.clone()).into()
    }
    pub fn __or__(&self, other: &PySelection) -> PySelection {
        (self.inner.clone() | other.inner.clone()).into()
    }
    pub fn __xor__(&self, other: &PySelection) -> PySelection {
        (self.inner.clone() ^ other.inner.clone()).into()
    }
    pub fn __sub__(&self, other: &PySelection) -> PySelection {
        (self.inner.clone() - other.inner.clone()).into()
    }
    pub fn __invert__(&self) -> PySelection {
        (!self.inner.clone()).into()
    }
    pub fn nbbox(&self, min: [f64; 3], max: [f64; 3], all: bool) -> PySelection {
        self.inner.clone().nbbox(min, max, all).into()
    }
    pub fn nrect(&self, min: [f64; 2], max: [f64; 2], all: bool) -> PySelection {
        self.inner.clone().nrect(min, max, all).into()
    }
    /// This method filters upon nodes position.
    pub fn nsphere(&self, center: [f64; 3], r2: f64, all: bool) -> PySelection {
        self.inner.clone().nsphere(center, r2, all).into()
    }
    pub fn ncircle(&self, center: [f64; 2], r2: f64, all: bool) -> PySelection {
        self.inner.clone().ncircle(center, r2, all).into()
    }
    pub fn nids(&self, ids: Vec<usize>, all: bool) -> PySelection {
        self.inner.clone().nids(ids, all).into()
    }
    pub fn bbox(&self, min: [f64; 3], max: [f64; 3]) -> PySelection {
        self.inner.clone().bbox(min, max).into()
    }
    pub fn rect(&self, min: [f64; 2], max: [f64; 2]) -> PySelection {
        self.inner.clone().rect(min, max).into()
    }
    pub fn sphere(&self, center: [f64; 3], r2: f64) -> PySelection {
        self.inner.clone().sphere(center, r2).into()
    }
    pub fn circle(&self, center: [f64; 2], r2: f64) -> PySelection {
        self.inner.clone().circle(center, r2).into()
    }
    pub fn group(&self, name: &str) -> PySelection {
        self.inner.clone().group(name).into()
    }
    pub fn exclude_group(&self, name: &str) -> PySelection {
        self.inner.clone().exclude_group(name).into()
    }
}

pub(crate) enum Selector {
    Expr(mf::Selection),
    Ids(mf::ElementIds),
}

pub(crate) fn extract_selector(source: &Bound<'_, PyAny>) -> PyResult<Selector> {
    if let Ok(expr) = source.extract::<PySelection>() {
        return Ok(Selector::Expr(expr.into()));
    }
    if let Ok(dict) = source.cast::<PyDict>() {
        return Ok(Selector::Ids(PyElementIds::from_dict(dict).into()));
    }
    if is_wildcard(source) {
        return Ok(Selector::Expr(mf::sel::all()));
    }
    Err(PyTypeError::new_err(
        "expected a Selection expression or a dict of element ids {\"ETYPE\": [indices]}",
    ))
}

fn is_wildcard(obj: &Bound<'_, PyAny>) -> bool {
    if obj.is_none() || obj.is_instance_of::<pyo3::types::PyEllipsis>() {
        return true;
    }
    if let Ok(s) = obj.cast::<pyo3::types::PySlice>() {
        let start = s.getattr("start").ok().is_some_and(|v| v.is_none());
        let stop = s.getattr("stop").ok().is_some_and(|v| v.is_none());
        return start && stop;
    }
    if let Ok(t) = obj.cast::<pyo3::types::PyTuple>() {
        return !t.is_empty() && t.iter().all(|item| is_wildcard(&item));
    }
    false
}

pub(crate) fn resolve_selector(mesh: &mf::UMesh, selector: Selector) -> mf::ElementIds {
    match selector {
        Selector::Expr(expr) => mesh.select_ids(expr, None),
        Selector::Ids(ids) => ids,
    }
}

#[pyclass]
#[pyo3(name = "SelectionResult")]
pub struct PySelectionResult {
    pub(crate) mesh: Py<PyUMesh>,
    pub(crate) expr: mf::Selection,
    pub(crate) dim: Option<Dimension>,
}

impl PySelectionResult {
    fn field_of(expr: &Bound<'_, PyAny>) -> PyResult<PyField> {
        PyField::try_from(expr)
    }

    fn reduce_expr(&self, fexpr: PyField, op: ReduceOp, ddof: usize) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let eids = inner.select_ids(self.expr.clone(), self.dim);
            if eids.is_empty() {
                return Err(PyValueError::new_err("the selection is empty"));
            }
            let dim = inner.topological_dimension().ok_or_else(|| {
                PyValueError::new_err("cannot reduce over a mesh without elements")
            })?;
            let expr_ref: FieldExpr = fexpr.clone().into();
            validate_expr_fields(inner, &expr_ref, dim)?;
            let evaluated = inner.eval_field(Some(dim), fexpr.into());
            let mut blocks: Vec<nd::ArrayD<f64>> = Vec::new();
            for (et, arr) in evaluated.0.iter() {
                let Some(idxs) = eids.get(et) else { continue };
                if idxs.is_empty() {
                    continue;
                }
                blocks.push(arr.select(nd::Axis(0), idxs));
            }
            if blocks.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "the selection contains no elements carrying fields on dimension {} elements",
                    u8::from(dim)
                )));
            }
            let views: Vec<nd::ArrayViewD<'_, f64>> = blocks.iter().map(|b| b.view()).collect();
            let reduced = reduce_blocks(&views, op, ddof)?;
            Ok(reduce_to_py(py, reduced))
        })
    }

    fn integral_expr(&self, fexpr: PyField) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let eids = inner.select_ids(self.expr.clone(), self.dim);
            if eids.is_empty() {
                return Err(PyValueError::new_err("the selection is empty"));
            }
            let dim = inner.topological_dimension().ok_or_else(|| {
                PyValueError::new_err("cannot integrate over a mesh without elements")
            })?;
            let expr_ref: FieldExpr = fexpr.clone().into();
            validate_expr_fields(inner, &expr_ref, dim)?;
            let evaluated = inner.eval_field(Some(dim), fexpr.into());
            let measures = mf::measure(&inner.view(), Some(dim));
            let mut blocks: Vec<nd::ArrayD<f64>> = Vec::new();
            for (et, arr) in evaluated.0.iter() {
                let Some(idxs) = eids.get(et) else { continue };
                if idxs.is_empty() {
                    continue;
                }
                let meas = measures.get(et).ok_or_else(|| {
                    PyValueError::new_err(
                        "missing measure for an element type present in the selection",
                    )
                })?;
                let gathered = arr.select(nd::Axis(0), idxs);
                let gathered_meas = meas.select(nd::Axis(0), idxs);
                blocks.push(measure_weighted(&gathered_meas, gathered.view())?);
            }
            if blocks.is_empty() {
                return Err(PyValueError::new_err(format!(
                    "the selection contains no elements carrying fields on dimension {} elements",
                    u8::from(dim)
                )));
            }
            let views: Vec<nd::ArrayViewD<'_, f64>> = blocks.iter().map(|b| b.view()).collect();
            let reduced = reduce_blocks(&views, ReduceOp::Sum, 0)?;
            Ok(reduce_to_py(py, reduced))
        })
    }
}

#[pymethods]
impl PySelectionResult {
    pub fn ids<'py>(&self, py: Python<'py>) -> Bound<'py, PyDict> {
        let eids = Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            inner.select_ids(self.expr.clone(), self.dim)
        });
        ids_to_pydict(py, &eids)
    }

    #[pyo3(signature = (with_fields=true))]
    pub fn to_mesh(&self, with_fields: bool) -> PyUMesh {
        let submesh = Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            inner.select(self.expr.clone(), with_fields, self.dim).1
        });
        submesh.into()
    }

    pub fn __len__(&self) -> usize {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            inner.select_ids(self.expr.clone(), self.dim).len()
        })
    }

    pub fn min(&self, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Min, 0)
    }

    pub fn max(&self, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Max, 0)
    }

    pub fn sum(&self, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Sum, 0)
    }

    pub fn mean(&self, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Mean, 0)
    }

    #[pyo3(signature = (expr, ddof=0))]
    pub fn var(&self, expr: &Bound<'_, PyAny>, ddof: usize) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Var, ddof)
    }

    #[pyo3(signature = (expr, ddof=0))]
    pub fn std(&self, expr: &Bound<'_, PyAny>, ddof: usize) -> PyResult<Py<PyAny>> {
        self.reduce_expr(Self::field_of(expr)?, ReduceOp::Std, ddof)
    }

    pub fn integral(&self, expr: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.integral_expr(Self::field_of(expr)?)
    }

    pub fn __repr__(&self) -> String {
        format!("SelectionResult(n_elements={})", self.__len__())
    }
}
