use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::fmt::{Display, Formatter};

use mefikit::prelude as mf;

use super::element_ids::PyElementIds;

#[pyclass(str)]
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
pub fn ids<'py>(eids: Bound<'py, PyDict>) -> PySelection {
    let eids = PyElementIds::from_dict(&eids);
    mf::sel::ids(eids.into()).into()
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
}
