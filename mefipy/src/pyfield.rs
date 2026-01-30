use pyo3::prelude::*;
use std::collections::BTreeMap;

use mefikit::prelude as mf;

use numpy as np;

#[pyclass]
#[pyo3(name = "Field")]
pub struct PyField {
    inner: mf::FieldOwnedD,
}
