use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "mefikit")]
mod mefikitpy {
    use pyo3::prelude::*;

    use mefikit as mf;
    use numpy::borrow::PyReadonlyArray2;

    #[pyclass]
    #[pyo3(name = "UMesh")]
    struct PyUMesh {
        // Add fields here
        inner: mf::UMesh,
    }

    #[pymethods]
    impl PyUMesh {
        #[new]
        fn new(coords: PyReadonlyArray2<'_, f64>) -> Self {
            let coords = coords.as_array();

            PyUMesh {
                inner: mf::UMesh::new(coords.to_shared()),
            }
        }

        fn __str__(&self) -> String {
            format!("UMesh:\n======\ncoords\n{}", self.inner.coords())
        }
    }
}
