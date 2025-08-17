use pyo3::prelude::*;

#[pyclass]
struct UMesh {}

#[pymethods]
impl UMesh {
    #[new]
    fn new() -> Self {
        UMesh {}
    }

    fn hello(&self) -> &'static str {
        "Hello from Rust!"
    }
}

#[pymodule]
fn umesh_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<UMesh>()?;
    Ok(())
}
