use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyIterator, PyList};

use mefikit::mesh::Dimension;
use mefikit::prelude as mf;
use mefikit::tools::timeseries::{FieldKey, TimeStep, nearest_step};

use super::pyfields::{PyFieldRef, assign_field_at, extract_field_value};
use super::pyumesh::PyUMesh;

fn resolve_series_dim(
    inner: &mf::UMesh,
    base: &str,
    dim: Option<Dimension>,
) -> PyResult<Dimension> {
    if let Some((existing, _)) = inner.resolve_time_series(base, None) {
        if let Some(d) = dim
            && d != existing
        {
            return Err(PyValueError::new_err(format!(
                "the time series '{base}' already exists on dimension {}: cannot add a step on dimension {}",
                u8::from(existing),
                u8::from(d)
            )));
        }
        return Ok(existing);
    }
    if let Some(d) = dim {
        return Ok(d);
    }
    inner.topological_dimension().ok_or_else(|| {
        PyValueError::new_err(format!(
            "cannot infer a dimension for the new time series '{base}': the mesh has no elements"
        ))
    })
}

#[pyclass]
#[pyo3(name = "TimeSeriesMapping")]
pub struct PyTimeSeriesMapping {
    pub(crate) mesh: Py<PyUMesh>,
}

impl PyTimeSeriesMapping {
    fn with_inner<R>(&self, f: impl FnOnce(&mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&self.mesh.bind(py).borrow().inner))
    }

    fn sorted_bases(&self) -> Vec<String> {
        self.with_inner(|m| m.time_series_bases().into_iter().map(|(b, _)| b).collect())
    }

    fn make_series(&self, base: String) -> PyTimeSeries {
        Python::attach(|py| PyTimeSeries {
            mesh: self.mesh.clone_ref(py),
            base,
        })
    }
}

#[pymethods]
impl PyTimeSeriesMapping {
    pub fn keys(&self) -> Vec<String> {
        self.sorted_bases()
    }

    pub fn values(&self) -> Vec<PyTimeSeries> {
        self.sorted_bases()
            .into_iter()
            .map(|base| self.make_series(base))
            .collect()
    }

    pub fn items(&self) -> Vec<(String, PyTimeSeries)> {
        self.sorted_bases()
            .into_iter()
            .map(|base| {
                let series = self.make_series(base.clone());
                (base, series)
            })
            .collect()
    }

    pub fn __len__(&self) -> usize {
        self.sorted_bases().len()
    }

    pub fn __contains__(&self, base: &str) -> bool {
        self.with_inner(|m| m.resolve_time_series(base, None).is_some())
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        let list = PyList::new(slf.py(), slf.keys())?;
        list.as_any().try_iter()
    }

    /// Returns a handle on the series, creating it on first use: it materializes
    /// once the first step is added.
    pub fn __getitem__(&self, base: &str) -> PyTimeSeries {
        self.make_series(base.to_owned())
    }

    pub fn __repr__(&self) -> String {
        format!("TimeSeriesMapping({:?})", self.sorted_bases())
    }
}

#[pyclass]
#[pyo3(name = "TimeSeries")]
pub struct PyTimeSeries {
    pub(crate) mesh: Py<PyUMesh>,
    pub(crate) base: String,
}

impl PyTimeSeries {
    fn steps_of(&self, inner: &mf::UMesh) -> Vec<TimeStep> {
        inner
            .resolve_time_series(&self.base, None)
            .map(|(_, steps)| steps)
            .unwrap_or_default()
    }

    fn make_ref(&self, py: Python<'_>, name: String) -> PyFieldRef {
        PyFieldRef {
            mesh: self.mesh.clone_ref(py),
            name,
        }
    }

    fn with_inner_steps<R>(&self, f: impl FnOnce(&[TimeStep]) -> R) -> R {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            f(&steps)
        })
    }
}

#[pymethods]
impl PyTimeSeries {
    #[getter]
    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn dim(&self) -> PyResult<u8> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let dim = inner
                .resolve_time_series(&self.base, None)
                .map(|(dim, _)| dim)
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("no time series named '{}'", self.base))
                })?;
            Ok(u8::from(dim))
        })
    }

    pub fn steps(&self) -> Vec<u64> {
        self.with_inner_steps(|steps| steps.iter().map(|s| s.step).collect())
    }

    pub fn times(&self) -> Vec<f64> {
        self.with_inner_steps(|steps| steps.iter().map(|s| s.time).collect())
    }

    pub fn at(&self, step: u64) -> PyResult<PyFieldRef> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            let ts = steps.iter().find(|s| s.step == step).ok_or_else(|| {
                PyKeyError::new_err(format!("no step {step} in time series '{}'", self.base))
            })?;
            Ok(self.make_ref(py, ts.name.clone()))
        })
    }

    pub fn at_time(&self, time: f64) -> PyResult<PyFieldRef> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            let ts = steps.iter().find(|s| s.time == time).ok_or_else(|| {
                PyKeyError::new_err(format!(
                    "no step of time series '{}' has time {time}",
                    self.base
                ))
            })?;
            Ok(self.make_ref(py, ts.name.clone()))
        })
    }

    pub fn nearest(&self, time: f64) -> PyResult<PyFieldRef> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            let ts = nearest_step(&steps, time).ok_or_else(|| {
                PyKeyError::new_err(format!("no step in time series '{}'", self.base))
            })?;
            Ok(self.make_ref(py, ts.name.clone()))
        })
    }

    pub fn latest(&self) -> PyResult<PyFieldRef> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            let ts = steps.last().ok_or_else(|| {
                PyKeyError::new_err(format!("no step in time series '{}'", self.base))
            })?;
            Ok(self.make_ref(py, ts.name.clone()))
        })
    }

    #[pyo3(signature = (value, time, dim=None))]
    pub fn add(
        &self,
        value: &Bound<'_, PyAny>,
        time: f64,
        dim: Option<usize>,
    ) -> PyResult<PyFieldRef> {
        let value = extract_field_value(value)?;
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;
            let dim = match dim {
                Some(d) => {
                    let d = Dimension::try_from(d).map_err(PyValueError::new_err)?;
                    resolve_series_dim(inner, &self.base, Some(d))?
                }
                None => resolve_series_dim(inner, &self.base, None)?,
            };
            let step = inner.next_step(&self.base, Some(dim));
            let name = FieldKey::new(self.base.as_str(), step, time).name();
            assign_field_at(inner, &name, value, Some(dim))?;
            Ok(self.make_ref(py, name))
        })
    }

    pub fn set_time(&self, step: u64, time: f64) -> PyResult<()> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;
            let (dim, steps) = inner.resolve_time_series(&self.base, None).ok_or_else(|| {
                PyKeyError::new_err(format!("no time series named '{}'", self.base))
            })?;
            let old = steps.iter().find(|s| s.step == step).ok_or_else(|| {
                PyKeyError::new_err(format!("no step {step} in time series '{}'", self.base))
            })?;
            let new_name = FieldKey::new(self.base.as_str(), step, time).name();
            if new_name != old.name {
                if inner.field(&new_name, Some(dim)).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "a field named '{new_name}' already exists"
                    )));
                }
                let field = inner
                    .remove_field(&old.name, Some(dim))
                    .ok_or_else(|| PyKeyError::new_err(old.name.clone()))?;
                inner.update_field(&new_name, field);
            }
            Ok(())
        })
    }

    pub fn remove(&self, step: u64) -> PyResult<()> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;
            let (dim, steps) = inner.resolve_time_series(&self.base, None).ok_or_else(|| {
                PyKeyError::new_err(format!("no time series named '{}'", self.base))
            })?;
            let name = steps
                .iter()
                .find(|s| s.step == step)
                .ok_or_else(|| {
                    PyKeyError::new_err(format!("no step {step} in time series '{}'", self.base))
                })?
                .name
                .clone();
            inner
                .remove_field(&name, Some(dim))
                .ok_or_else(|| PyKeyError::new_err(name))?;
            Ok(())
        })
    }

    pub fn rename(mut slf: PyRefMut<'_, Self>, new_base: &str) -> PyResult<()> {
        Python::attach(|py| {
            let mesh = slf.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;
            let (dim, steps) = inner.resolve_time_series(&slf.base, None).ok_or_else(|| {
                PyKeyError::new_err(format!("no time series named '{}'", slf.base))
            })?;
            if inner.resolve_time_series(new_base, None).is_some() {
                return Err(PyValueError::new_err(format!(
                    "a time series named '{new_base}' already exists"
                )));
            }
            let targets: Vec<(String, String)> = steps
                .iter()
                .map(|s| {
                    (
                        s.name.clone(),
                        FieldKey::new(new_base, s.step, s.time).name(),
                    )
                })
                .collect();
            for (_, new_name) in &targets {
                if inner.field(new_name, Some(dim)).is_some() {
                    return Err(PyValueError::new_err(format!(
                        "a field named '{new_name}' already exists"
                    )));
                }
            }
            for (old_name, new_name) in targets {
                let field = inner
                    .remove_field(&old_name, Some(dim))
                    .ok_or_else(|| PyKeyError::new_err(old_name))?;
                inner.update_field(&new_name, field);
            }
            Ok(())
        })?;
        slf.base = new_base.to_owned();
        Ok(())
    }

    pub fn items(&self) -> Vec<(u64, f64, PyFieldRef)> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            self.steps_of(inner)
                .into_iter()
                .map(|s| (s.step, s.time, self.make_ref(py, s.name)))
                .collect()
        })
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        let list = PyList::empty(slf.py());
        for (_, time, field) in slf.items() {
            list.append((time, field))?;
        }
        list.as_any().try_iter()
    }

    pub fn __len__(&self) -> usize {
        self.with_inner_steps(|steps| steps.len())
    }

    pub fn __contains__(&self, step: u64) -> bool {
        self.with_inner_steps(|steps| steps.iter().any(|s| s.step == step))
    }

    pub fn __getitem__(&self, index: &Bound<'_, PyAny>) -> PyResult<PyFieldRef> {
        let step = index.extract::<u64>().map_err(|_| {
            PyTypeError::new_err(
                "expected an integer step, use `at`/`at_time`/`nearest` for time lookups",
            )
        })?;
        self.at(step)
    }

    pub fn __delitem__(&self, step: u64) -> PyResult<()> {
        self.remove(step)
    }

    pub fn __repr__(&self) -> PyResult<String> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let steps = self.steps_of(inner);
            if steps.is_empty() {
                return Ok(format!("TimeSeries('{}')", self.base));
            }
            let dim = inner
                .resolve_time_series(&self.base, None)
                .map(|(dim, _)| u8::from(dim))
                .unwrap_or(0);
            let pairs: Vec<String> = steps
                .iter()
                .map(|s| format!("{}:{}", s.step, s.time))
                .collect();
            Ok(format!(
                "TimeSeries('{}', dim={}) {{{}}}",
                self.base,
                dim,
                pairs.join(", ")
            ))
        })
    }
}
