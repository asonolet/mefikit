//! Global field statistics: count / sum / mean / min / max.
//!
//! Computed on the CubeCL device when the `gpu` feature is linked, with a
//! transparent host fallback whenever no device can run the kernel (and when
//! the feature is off), so this API never fails on machines without a device.

use pyo3::prelude::*;

#[cfg(feature = "gpu")]
use mefikit::prelude as mf;

/// Basic statistics over all values of a field.
#[pyclass(skip_from_py_object, name = "Stats")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyStats {
    /// Number of values.
    #[pyo3(get)]
    pub count: usize,
    /// Sum of all values.
    #[pyo3(get)]
    pub sum: f64,
    /// Arithmetic mean of all values.
    #[pyo3(get)]
    pub mean: f64,
    /// Minimum value.
    #[pyo3(get)]
    pub min: f64,
    /// Maximum value.
    #[pyo3(get)]
    pub max: f64,
}

#[pymethods]
impl PyStats {
    fn __repr__(&self) -> String {
        format!(
            "Stats(count={}, sum={}, mean={}, min={}, max={})",
            self.count, self.sum, self.mean, self.min, self.max
        )
    }
}

/// Global statistics of `values`, running on the CubeCL device when available.
pub(crate) fn stats(values: &[f64]) -> PyStats {
    #[cfg(feature = "gpu")]
    {
        let device = cubecl::Device::default();
        if let Ok(got) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            mf::field_stats(&device, values)
        })) {
            return PyStats {
                count: got.count,
                sum: got.sum,
                mean: got.mean,
                min: got.min,
                max: got.max,
            };
        }
        // No usable device (e.g. wgpu linked without an adapter): fall back.
    }
    host_stats(values)
}

fn host_stats(values: &[f64]) -> PyStats {
    let count = values.len();
    if count == 0 {
        return PyStats {
            count: 0,
            sum: 0.0,
            mean: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }
    let sum: f64 = values.iter().sum();
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    PyStats {
        count,
        sum,
        mean: sum / count as f64,
        min,
        max,
    }
}
