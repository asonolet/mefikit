use numpy as np;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use mefikit::prelude as mf;
use mefikit::tools::Gradient;

use crate::element::etype_to_str;
use crate::pyfield::PyField;
use crate::pytransfer::PyDistanceWeighting;
use crate::pyumesh::{PyUMesh, into_mut, into_view};

/// Builds a gradient expression for `expr`, to be evaluated later on the target cells.
fn gradient_expr<'py>(
    py: Python<'py>,
    src_mesh: &Py<PyUMesh>,
    op: &Arc<mf::GradientOperator>,
    def_val: f64,
    expr: &Bound<'py, PyAny>,
) -> PyResult<PyField> {
    if !op.has_cell_target() {
        return Err(PyValueError::new_err(
            "This Gradient was built at arbitrary points; use eval_points instead",
        ));
    }
    let pyf: PyField = expr.try_into()?;
    let guard = src_mesh.bind(py).borrow();
    Ok(Arc::clone(op).expr(&guard.inner, pyf.inner, def_val).into())
}

/// Evaluates `expr` on the source mesh and computes its gradient on the target cells eagerly.
fn gradient_eval<'py>(
    py: Python<'py>,
    src_mesh: &Py<PyUMesh>,
    op: &Arc<mf::GradientOperator>,
    def_val: f64,
    expr: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if !op.has_cell_target() {
        return Err(PyValueError::new_err(
            "This Gradient was built at arbitrary points; use eval_points instead",
        ));
    }
    let pyf: PyField = expr.try_into()?;
    let guard = src_mesh.bind(py).borrow();
    let field = op.eval(&guard.inner, pyf.inner, def_val);
    let dict = PyDict::new(py);
    for (et, arr) in field.0.iter() {
        dict.set_item(etype_to_str(*et), np::PyArray::from_array(py, arr))?;
    }
    Ok(dict)
}

#[pyclass(str)]
#[pyo3(name = "Gradient")]
pub struct PyGradient {
    op: Arc<mf::GradientOperator>,
    src_mesh: Py<PyUMesh>,
    def_val: f64,
}

impl Display for PyGradient {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:#?}", self.op)
    }
}

#[pymethods]
impl PyGradient {
    /// Builds a moving least-squares gradient operator.
    ///
    /// With no `tgt_mesh` the gradient is evaluated at the source mesh's own cell centroids;
    /// otherwise it is evaluated at the target mesh's cell centroids.
    #[new]
    #[pyo3(signature = (src_mesh, tgt_mesh=None, k=10, weighting=PyDistanceWeighting::Constant(), def_val=0.0))]
    fn new(
        src_mesh: &Bound<'_, PyUMesh>,
        tgt_mesh: Option<&Bound<'_, PyUMesh>>,
        k: usize,
        weighting: PyDistanceWeighting,
        def_val: f64,
    ) -> Self {
        let method = mf::GradientMethod::MovingLeastSquares {
            k,
            weighting: weighting.into(),
        };
        let src_guard = src_mesh.borrow();
        let src = into_view(&src_guard);
        let op = match tgt_mesh {
            Some(tgt) => {
                let tgt_guard = tgt.borrow();
                mf::GradientOperator::new(&src, &into_view(&tgt_guard), method)
            }
            None => mf::GradientOperator::on_source(&src, method),
        };
        PyGradient {
            op: Arc::new(op),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
    }

    /// Builds a moving least-squares gradient operator evaluated at arbitrary `points`.
    ///
    /// `points` is an `(n_points, d)` array of coordinates in the source mesh's space.
    #[staticmethod]
    #[pyo3(signature = (src_mesh, points, k=10, weighting=PyDistanceWeighting::Constant(), def_val=0.0))]
    fn at_points(
        src_mesh: &Bound<'_, PyUMesh>,
        points: np::PyReadonlyArray2<'_, f64>,
        k: usize,
        weighting: PyDistanceWeighting,
        def_val: f64,
    ) -> Self {
        let method = mf::GradientMethod::MovingLeastSquares {
            k,
            weighting: weighting.into(),
        };
        let src_guard = src_mesh.borrow();
        let src = into_view(&src_guard);
        let points = points.as_array();
        let op = mf::GradientOperator::at_points(&src, &points, method);
        PyGradient {
            op: Arc::new(op),
            src_mesh: src_mesh.clone().unbind(),
            def_val,
        }
    }

    /// Computes the gradient of `field_name` on `src_mesh` and stores it in `tgt_mesh`.
    #[pyo3(signature = (src_mesh, field_name, tgt_mesh, tgt_field_name=None, def_val=0.0))]
    fn apply_update(
        &self,
        src_mesh: &PyUMesh,
        field_name: &str,
        tgt_mesh: &mut PyUMesh,
        tgt_field_name: Option<&str>,
        def_val: f64,
    ) -> PyResult<()> {
        if !self.op.has_cell_target() {
            return Err(PyValueError::new_err(
                "This Gradient was built at arbitrary points and has no target mesh",
            ));
        }
        let name = tgt_field_name.unwrap_or(field_name);
        let src_view = into_view(src_mesh);
        let field = src_view.field(field_name, None).unwrap();
        self.op
            .apply_update(into_mut(tgt_mesh), name, &field, def_val);
        Ok(())
    }

    /// Wraps the gradient of `expr` as a lazy field on the target mesh.
    fn __call__<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<PyField> {
        gradient_expr(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    /// Evaluates `expr` on the source mesh and computes its gradient on the target cells eagerly.
    fn eval<'py>(&self, py: Python<'py>, expr: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyDict>> {
        gradient_eval(py, &self.src_mesh, &self.op, self.def_val, expr)
    }

    /// Evaluates the gradient of `expr` at the operator's evaluation points, as an `(n, d)` array.
    fn eval_points<'py>(
        &self,
        py: Python<'py>,
        expr: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, np::PyArray2<f64>>> {
        if self.op.has_cell_target() {
            return Err(PyValueError::new_err(
                "This Gradient was built on mesh cells; build it with Gradient.at_points instead",
            ));
        }
        let pyf: PyField = expr.try_into()?;
        let guard = self.src_mesh.bind(py).borrow();
        let grad = self.op.eval_points(&guard.inner, pyf.inner, self.def_val);
        Ok(np::PyArray2::from_array(py, &grad))
    }
}
