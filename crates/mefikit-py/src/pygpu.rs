//! Device selection shared by the GPU-backed Python gradient and transfer tools.
//!
//! The meshless fit of the [`Gradient`](crate::pygradient::PyGradient) and the moving
//! least-squares / inverse-distance transfers can run on a CubeCL device. The Python API takes a
//! `device` string (`"cpu"`, `"auto"` or `"wgpu"`); this module turns it into the optional device
//! the operator constructors expect. `"cpu"` and `"auto"` fall back to the f64 host implementation.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Resolves a `device` specification to a CubeCL device.
///
/// Returns `Ok(None)` when the fit should run on the host CPU: the caller asked for `"cpu"` (the
/// default), or `"auto"` found no usable `wgpu` adapter. `"wgpu"` is strict and raises when no
/// adapter is available; an unknown string is rejected.
#[cfg(feature = "gpu")]
pub(crate) fn resolve_device(spec: &str) -> PyResult<Option<cubecl::Device>> {
    use cubecl::device::{Device, WgpuDeviceKind};

    match spec {
        "cpu" => Ok(None),
        "auto" => {
            let found = std::panic::catch_unwind(|| Device::wgpu(WgpuDeviceKind::DefaultDevice))
                .ok()
                .and_then(Result::ok);
            Ok(found)
        }
        "wgpu" => {
            let found = std::panic::catch_unwind(|| Device::wgpu(WgpuDeviceKind::DefaultDevice))
                .map_err(|_| {
                    PyValueError::new_err("device='wgpu' panicked while opening an adapter")
                })?;
            found.map(Some).map_err(|err| {
                PyValueError::new_err(format!("device='wgpu' is unavailable: {err}"))
            })
        }
        other => Err(PyValueError::new_err(format!(
            "device must be 'cpu', 'auto' or 'wgpu', got '{other}'"
        ))),
    }
}

/// Without the `gpu` feature there is no device to select; only the CPU path is available.
#[cfg(not(feature = "gpu"))]
pub(crate) fn resolve_device(spec: &str) -> PyResult<Option<()>> {
    match spec {
        "cpu" | "auto" => Ok(None),
        "wgpu" => Err(PyValueError::new_err(
            "device='wgpu' requires a build with the `gpu` feature",
        )),
        other => Err(PyValueError::new_err(format!(
            "device must be 'cpu', 'auto' or 'wgpu', got '{other}'"
        ))),
    }
}
