use pyo3::prelude::*;

use mefikit::prelude as mf;

#[pyclass]
#[pyo3(name = "Field")]
pub struct PyField {
    inner: mf::FieldOwnedD,
}
