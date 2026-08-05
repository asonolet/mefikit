use pyo3::prelude::*;
use std::fmt::{Display, Formatter};

use mefikit::{prelude as mf, tools::Transfer};

use crate::pyumesh::{PyUMesh, into_mut, into_view};

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
