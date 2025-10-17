use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "mefikit")]
mod mefikitpy {
    use pyo3::prelude::*;

    use mefikit as mf;
    use ndarray as nd;
    use numpy::borrow::PyReadonlyArray2;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    #[pyclass]
    #[pyo3(name = "UMesh")]
    struct PyUMesh {
        // Add fields here
        inner: mf::UMesh,
    }

    #[pymethods]
    impl PyUMesh {
        #[new]
        fn new() -> Self {
            PyUMesh {
                inner: mf::UMesh::new(
                    nd::ArcArray2::<f64>::from_shape_vec((1, 1), vec![0.0]).unwrap(),
                ),
            }
        }
    }
}
