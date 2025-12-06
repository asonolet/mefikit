use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "mefikit")]
mod mefikitpy {
    use pyo3::prelude::*;
    use std::fmt::{Display, Formatter};

    use mefikit::prelude as mf;

    use std::path::Path;

    use numpy as np;
    use serde_json;

    #[pyclass(str)]
    #[pyo3(name = "UMesh")]
    #[derive(PartialEq)]
    struct PyUMesh {
        inner: mf::UMesh,
    }

    #[pymethods]
    impl PyUMesh {
        #[new]
        fn new(coords: np::PyReadonlyArray2<'_, f64>) -> Self {
            PyUMesh {
                inner: mf::UMesh::new(coords.as_array().to_shared()),
            }
        }

        /// Returns a copy owned by python of the array coordinates
        fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, np::PyArray2<f64>> {
            np::PyArray2::from_array(py, &self.inner.coords())
        }

        fn to_json(&self) -> String {
            serde_json::to_string(&self.inner).unwrap()
        }

        fn to_json_pretty(&self) -> String {
            serde_json::to_string_pretty(&self.inner).unwrap()
        }

        /// Add a regular block of elements to the mesh.
        fn add_regular_block(&mut self, et: &str, block: np::PyReadonlyArray2<'_, usize>) {
            let et = match et {
                "VERTEX" => mf::ElementType::VERTEX,
                "TET4" => mf::ElementType::TET4,
                "QUAD4" => mf::ElementType::QUAD4,
                "TRI3" => mf::ElementType::TRI3,
                "HEX8" => mf::ElementType::HEX8,
                _ => panic!("Unsupported element type: {}", et),
            };
            self.inner
                .add_regular_block(et, block.as_array().to_shared());
        }

        // fn get_regular_connectivity(&self, et: &str) -> Py<PyReadwriteArray2<'_, usize>> {
        //     let et = match { et } {
        //         "VERTEX" => mf::ElementType::VERTEX,
        //         "TET4" => mf::ElementType::TET4,
        //         "QUAD4" => mf::ElementType::QUAD4,
        //         "TRI3" => mf::ElementType::TRI3,
        //         "HEX8" => mf::ElementType::HEX8,
        //         _ => panic!("Unsupported element type: {}", et),
        //     };
        //     let conn = self.inner.element_blocks.get(&et).unwrap().connectivity;
        //     PyReadwriteArray2::from_array(py, &conn).into()
        // }

        #[staticmethod]
        fn read(path: &str) -> Self {
            let path = Path::new(path);
            mf::read(path).unwrap().into()
        }

        fn write(&self, path: &str) {
            let path = Path::new(path);
            let mesh = self.inner.view();
            let _ = mf::write(path, mesh);
        }
    }

    impl Display for PyUMesh {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:#?}", self.inner)
        }
    }

    impl From<mf::UMesh> for PyUMesh {
        fn from(umesh: mf::UMesh) -> Self {
            PyUMesh { inner: umesh }
        }
    }

    impl From<PyUMesh> for mf::UMesh {
        fn from(pyumesh: PyUMesh) -> Self {
            pyumesh.inner
        }
    }
}
