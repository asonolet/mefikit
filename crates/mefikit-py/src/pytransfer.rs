use numpy as np;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use mefikit::prelude as mf;
use mefikit::prelude::fieldexpr::{Evaluable, FieldExpr, TransferOp};
use mefikit::tools::Transfer;

use crate::element::etype_to_str;
use crate::pyfield::PyField;
use crate::pyumesh::{PyUMesh, into_mut, into_view};

/// Evaluates `expr` on the source mesh, materialising it as an owned field.
fn eval_source<'py>(
    py: Python<'py>,
    src_mesh: &Py<PyUMesh>,
    expr: &Bound<'py, PyAny>,
) -> PyResult<mf::FieldOwnedD> {
    let pyf: PyField = expr.try_into()?;
    let src_mesh = src_mesh.bind(py);
    let guard = src_mesh.borrow();
    let src_view = into_view(&guard);
    Ok(pyf.inner.evaluate(&src_view, None).to_owned())
}

/// Materialises `expr` on the source mesh as a lazy transfer expression onto the target cells.
fn transfer_expr<'py>(
    py: Python<'py>,
    src_mesh: &Py<PyUMesh>,
    op: &Arc<TransferOp>,
    def_val: f64,
    expr: &Bound<'py, PyAny>,
) -> PyResult<PyField> {
    let source = eval_source(py, src_mesh, expr)?;
    Ok(FieldExpr::Transfer {
        source_values: source,
        op: op.clone(),
        tgt_dim: op.tgt_dim(),
        default: def_val,
    }
    .into())
}

/// Materialises `expr` on the source mesh and transfers it onto the target cells eagerly.
fn transfer_eval<'py>(
    py: Python<'py>,
    src_mesh: &Py<PyUMesh>,
    op: &Arc<TransferOp>,
    def_val: f64,
    expr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let source = eval_source(py, src_mesh, expr)?;
    let src_view = source.view();
    let field = op.apply(&src_view, mf::FieldNature::Intensive, def_val);
    let dict = PyDict::new(py);
    for (et, arr) in field.0.iter() {
        dict.set_item(etype_to_str(*et), np::PyArray::from_array(py, arr))?;
    }
    Ok(dict)
}

// ---------------------------------------------------------------------------
// DistanceWeighting
// ---------------------------------------------------------------------------

#[pyclass(from_py_object)]
#[pyo3(name = "DistanceWeighting")]
#[derive(Clone)]
pub enum PyDistanceWeighting {
    Constant(),
    InverseDistance { exponent: f64 },
    Gaussian(),
}

impl From<PyDistanceWeighting> for mf::DistanceWeighting {
    fn from(weighting: PyDistanceWeighting) -> Self {
        match weighting {
            PyDistanceWeighting::Constant() => mf::DistanceWeighting::Constant,
            PyDistanceWeighting::InverseDistance { exponent } => {
                mf::DistanceWeighting::InverseDistance { exponent }
            }
            PyDistanceWeighting::Gaussian() => mf::DistanceWeighting::Gaussian,
        }
    }
}

// ---------------------------------------------------------------------------
// ConstantPiecewise
// ---------------------------------------------------------------------------

#[pyclass(str)]
#[pyo3(name = "ConstantPiecewise")]
pub struct PyConstantPiecewise {
    op: Arc<TransferOp>,
    src_mesh: Py<PyUMesh>,
    def_val: f64,
}

impl Display for PyConstantPiecewise {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.op)
    }
}

#[pymethods]
impl PyConstantPiecewise {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, def_val=0.0))]
    fn new(src_mesh: &Bound<'_, PyUMesh>, tgt_mesh: &Bound<'_, PyUMesh>, def_val: f64) -> Self {
        let transfer = mf::ConstantPiecewiseTransfer::new(
            &into_view(&src_mesh.borrow()),
            &into_view(&tgt_mesh.borrow()),
            mf::PointLocation::Centroid,
        );
        PyConstantPiecewise {
            op: Arc::new(TransferOp::ConstantPiecewise(transfer)),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
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
        self.op
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }

    fn __call__<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<PyField> {
        transfer_expr(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    fn eval<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        transfer_eval(py, &self.src_mesh, &self.op, self.def_val, expr)
    }
}

// ---------------------------------------------------------------------------
// MovingLeastSquares
// ---------------------------------------------------------------------------

#[pyclass(str)]
#[pyo3(name = "MovingLeastSquares")]
pub struct PyMovingLeastSquares {
    op: Arc<TransferOp>,
    src_mesh: Py<PyUMesh>,
    def_val: f64,
}

impl Display for PyMovingLeastSquares {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.op)
    }
}

#[pymethods]
impl PyMovingLeastSquares {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, k=10, weighting=PyDistanceWeighting::Constant(), def_val=0.0))]
    fn new(
        src_mesh: &Bound<'_, PyUMesh>,
        tgt_mesh: &Bound<'_, PyUMesh>,
        k: usize,
        weighting: PyDistanceWeighting,
        def_val: f64,
    ) -> Self {
        let transfer = mf::MovingLeastSquaresTransfer::new(
            &into_view(&src_mesh.borrow()),
            &into_view(&tgt_mesh.borrow()),
            k,
            weighting.into(),
        );
        PyMovingLeastSquares {
            op: Arc::new(TransferOp::MovingLeastSquares(transfer)),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
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
        self.op
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }

    fn __call__<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<PyField> {
        transfer_expr(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    fn eval<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        transfer_eval(py, &self.src_mesh, &self.op, self.def_val, expr)
    }
}

// ---------------------------------------------------------------------------
// ConservativeP0
// ---------------------------------------------------------------------------

#[pyclass(str)]
#[pyo3(name = "ConservativeP0")]
pub struct PyConservativeP0 {
    op: Arc<TransferOp>,
    src_mesh: Py<PyUMesh>,
    def_val: f64,
}

impl Display for PyConservativeP0 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.op)
    }
}

#[pymethods]
impl PyConservativeP0 {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, def_val=0.0))]
    fn new(src_mesh: &Bound<'_, PyUMesh>, tgt_mesh: &Bound<'_, PyUMesh>, def_val: f64) -> Self {
        let transfer = mf::ConservativeP0Transfer::new(
            &into_view(&src_mesh.borrow()),
            &into_view(&tgt_mesh.borrow()),
        );
        PyConservativeP0 {
            op: Arc::new(TransferOp::ConservativeP0(transfer)),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
    }

    /// Transfers a field defined on 2D source cells to the 2D target cells, weighted by the cell
    /// intersection measures. By default the transferred values are intensive (integral over the
    /// target cell divided by its measure); set `extensive=True` to keep the raw summed values.
    #[pyo3(signature = (src_mesh, field_name, tgt_mesh, tgt_field_name=None, def_val=0.0, extensive=false))]
    fn apply_update(
        &self,
        src_mesh: &PyUMesh,
        field_name: &str,
        tgt_mesh: &mut PyUMesh,
        tgt_field_name: Option<&str>,
        def_val: f64,
        extensive: bool,
    ) {
        let name = tgt_field_name.unwrap_or(field_name);
        let src_view = into_view(src_mesh);
        let field = src_view.field(field_name, None).unwrap();
        let field_nature = if extensive {
            mf::FieldNature::Extensive
        } else {
            mf::FieldNature::Intensive
        };
        self.op
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }

    fn __call__<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<PyField> {
        transfer_expr(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    fn eval<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        transfer_eval(py, &self.src_mesh, &self.op, self.def_val, expr)
    }
}

// ---------------------------------------------------------------------------
// InverseDistance
// ---------------------------------------------------------------------------

#[pyclass(str)]
#[pyo3(name = "InverseDistance")]
pub struct PyInverseDistance {
    op: Arc<TransferOp>,
    src_mesh: Py<PyUMesh>,
    def_val: f64,
}

impl Display for PyInverseDistance {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.op)
    }
}

#[pymethods]
impl PyInverseDistance {
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh, k=4, exponent=2.0, def_val=0.0))]
    fn new(
        src_mesh: &Bound<'_, PyUMesh>,
        tgt_mesh: &Bound<'_, PyUMesh>,
        k: usize,
        exponent: f64,
        def_val: f64,
    ) -> Self {
        let transfer = mf::InverseDistanceTransfer::new(
            &into_view(&src_mesh.borrow()),
            &into_view(&tgt_mesh.borrow()),
            k,
            exponent,
        );
        PyInverseDistance {
            op: Arc::new(TransferOp::InverseDistance(transfer)),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
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
        self.op
            .apply_update(into_mut(tgt_mesh), name, &field, field_nature, def_val);
    }

    fn __call__<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<PyField> {
        transfer_expr(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    fn eval<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        transfer_eval(py, &self.src_mesh, &self.op, self.def_val, expr)
    }
}
