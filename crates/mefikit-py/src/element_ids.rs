use numpy as np;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyDictMethods;

use mefikit::mesh::ElementIds;

use super::element::{etype_to_str, str_to_etype};

#[pyclass]
#[pyo3(name = "ElementIds")]
pub struct PyElementIds {
    inner: ElementIds,
}

impl From<PyElementIds> for Py<PyDict> {
    fn from(eids: PyElementIds) -> Self {
        Python::attach(|py| {
            let dict = PyDict::new(py);
            for (et, ids) in eids.inner.iter_blocks() {
                let py_ids = np::PyArray1::from_vec(py, ids.clone());
                dict.set_item(etype_to_str(*et), py_ids).unwrap();
            }
            dict.into()
        })
    }
}

impl From<PyElementIds> for ElementIds {
    fn from(pyeids: PyElementIds) -> Self {
        pyeids.inner
    }
}

impl From<ElementIds> for PyElementIds {
    fn from(eids: ElementIds) -> Self {
        PyElementIds { inner: eids }
    }
}

impl PyElementIds {
    pub fn from_dict<'py>(dict: &Bound<'py, PyDict>) -> Self {
        let mut eids = ElementIds::new();
        for (key, value) in dict.iter() {
            let et_str: &str = key.extract().unwrap();
            let et = str_to_etype(et_str);
            let ids_array: np::PyReadonlyArray1<usize> = value.extract().unwrap();
            let ids = ids_array.as_array().to_vec();
            eids.add_block(et, ids);
        }
        PyElementIds { inner: eids }
    }
}
