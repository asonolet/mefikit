use pyo3::exceptions::{PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyIterator, PyList};

use mefikit::prelude as mf;

use super::element_ids::ids_to_pydict;
use super::pyumesh::PyUMesh;
use super::select::{extract_selector, resolve_selector};

#[pyclass]
#[pyo3(name = "GroupsMapping")]
pub struct PyGroupsMapping {
    pub(crate) mesh: Py<PyUMesh>,
}

impl PyGroupsMapping {
    fn with_inner<R>(&self, f: impl FnOnce(&mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&self.mesh.bind(py).borrow().inner))
    }

    fn with_inner_mut<R>(&self, f: impl FnOnce(&mut mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&mut self.mesh.bind(py).borrow_mut().inner))
    }

    fn sorted_names(&self) -> Vec<String> {
        let mut names = self.with_inner(|m| m.group_names());
        names.sort_unstable();
        names
    }

    fn make_ref(&self, name: String) -> PyGroupRef {
        PyGroupRef {
            mesh: Python::attach(|py| self.mesh.clone_ref(py)),
            name,
        }
    }
}

#[pymethods]
impl PyGroupsMapping {
    pub fn keys(&self) -> Vec<String> {
        self.sorted_names()
    }

    pub fn values(&self) -> Vec<PyGroupRef> {
        self.sorted_names()
            .into_iter()
            .map(|name| self.make_ref(name))
            .collect()
    }

    #[pyo3(name = "items")]
    pub fn items_pairs(&self) -> Vec<(String, PyGroupRef)> {
        self.sorted_names()
            .into_iter()
            .map(|name| {
                let r = self.make_ref(name.clone());
                (name, r)
            })
            .collect()
    }

    pub fn __len__(&self) -> usize {
        self.sorted_names().len()
    }

    pub fn __contains__(&self, name: &str) -> bool {
        self.with_inner(|m| m.has_group(name))
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        let list = PyList::new(slf.py(), slf.keys())?;
        list.as_any().try_iter()
    }

    pub fn __getitem__(&self, name: &str) -> PyResult<PyGroupRef> {
        let exists = self.with_inner(|m| m.has_group(name));
        if !exists {
            return Err(PyKeyError::new_err(name.to_string()));
        }
        Ok(self.make_ref(name.to_string()))
    }

    pub fn __setitem__(&self, name: &str, source: &Bound<'_, PyAny>) -> PyResult<()> {
        let selector = extract_selector(source)?;
        let eids =
            Python::attach(|py| resolve_selector(&self.mesh.bind(py).borrow().inner, selector));
        self.with_inner_mut(|m| {
            if m.has_group(name) {
                m.delete_group(name);
            }
            m.add_to_group(name, &eids);
        });
        Ok(())
    }

    pub fn __delitem__(&self, name: &str) -> PyResult<()> {
        self.with_inner_mut(|m| {
            if !m.has_group(name) {
                return Err(PyKeyError::new_err(name.to_string()));
            }
            m.delete_group(name);
            Ok(())
        })
    }

    pub fn rename(&self, old_name: &str, new_name: &str) -> PyResult<()> {
        self.with_inner_mut(|m| {
            if !m.has_group(old_name) {
                return Err(PyKeyError::new_err(old_name.to_string()));
            }
            if m.has_group(new_name) {
                return Err(PyValueError::new_err(format!(
                    "group '{new_name}' already exists"
                )));
            }
            m.rename_group(old_name, new_name);
            Ok(())
        })
    }

    pub fn __repr__(&self) -> String {
        format!("GroupsMapping({:?})", self.sorted_names())
    }
}

#[pyclass]
#[pyo3(name = "GroupRef")]
pub struct PyGroupRef {
    pub(crate) mesh: Py<PyUMesh>,
    pub(crate) name: String,
}

impl PyGroupRef {
    fn with_inner<R>(&self, f: impl FnOnce(&mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&self.mesh.bind(py).borrow().inner))
    }

    fn with_inner_mut<R>(&self, f: impl FnOnce(&mut mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&mut self.mesh.bind(py).borrow_mut().inner))
    }

    fn ensure_exists(&self) -> PyResult<()> {
        let exists = self.with_inner(|m| m.has_group(&self.name));
        if exists {
            Ok(())
        } else {
            Err(PyKeyError::new_err(format!(
                "no group named '{}'",
                self.name
            )))
        }
    }
}

#[pymethods]
impl PyGroupRef {
    pub fn add(&self, source: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_exists()?;
        let selector = extract_selector(source)?;
        let eids =
            Python::attach(|py| resolve_selector(&self.mesh.bind(py).borrow().inner, selector));
        let name = self.name.clone();
        self.with_inner_mut(move |m| m.add_to_group(&name, &eids));
        Ok(())
    }

    pub fn remove(&self, source: &Bound<'_, PyAny>) -> PyResult<()> {
        self.ensure_exists()?;
        let selector = extract_selector(source)?;
        let eids =
            Python::attach(|py| resolve_selector(&self.mesh.bind(py).borrow().inner, selector));
        let name = self.name.clone();
        self.with_inner_mut(move |m| m.remove_from_group(&name, &eids));
        Ok(())
    }

    pub fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.ensure_exists()?;
        let eids = self.with_inner(|m| m.group_elements(&self.name));
        Ok(ids_to_pydict(py, &eids))
    }

    #[pyo3(signature = (with_fields=true))]
    pub fn to_mesh(&self, with_fields: bool) -> PyResult<PyUMesh> {
        self.ensure_exists()?;
        let name = self.name.clone();
        Ok(self.with_inner(|m| m.extract(&m.group_elements(&name), with_fields).into()))
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.ensure_exists()?;
        Ok(self.with_inner(|m| m.group_elements(&self.name).len()))
    }

    pub fn __repr__(&self) -> String {
        let size = self.with_inner(|m| {
            if m.has_group(&self.name) {
                Some(m.group_elements(&self.name).len())
            } else {
                None
            }
        });
        match size {
            Some(n) => format!("GroupRef({:?}, n_elements={n})", self.name),
            None => format!("GroupRef({:?}, <deleted>)", self.name),
        }
    }
}
