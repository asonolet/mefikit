use std::collections::BTreeMap;

use ndarray as nd;
use pyo3::prelude::*;

use mefikit::prelude as mf;

#[pyclass]
#[pyo3(name = "Field")]
pub struct PyField {
    inner: mf::FieldOwnedD,
}

impl From<BTreeMap<mf::ElementType, nd::ArrayD<f64>>> for PyField {
    fn from(value: BTreeMap<mf::ElementType, nd::ArrayD<f64>>) -> Self {
        PyField {
            inner: mf::FieldOwnedD::new(value),
        }
    }
}

impl From<PyField> for BTreeMap<mf::ElementType, nd::ArrayD<f64>> {
    fn from(value: PyField) -> Self {
        value.inner.0
    }
}
