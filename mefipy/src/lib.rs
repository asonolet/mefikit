use pyo3::prelude::*;

mod pyumesh;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod mefipy {
    use pyo3::{prelude::*, types::PyTuple};

    use mefikit::prelude as mf;

    #[pymodule_export]
    use super::pyumesh::PyUMesh;

    #[pyfunction]
    #[pyo3(signature = (*args))]
    pub fn build_cmesh(py: Python, args: &Bound<'_, PyTuple>) -> PyResult<PyUMesh> {
        let mut builder = mf::RegularUMeshBuilder::new();
        for arg in args {
            builder = builder.add_axis(arg.unbind().extract(py)?)
        }
        Ok(builder.build().into())
    }
}
