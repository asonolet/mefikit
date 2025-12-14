use pyo3::prelude::*;

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
#[pyo3(name = "mefikit")]
mod mefipy {
    use pyo3::{prelude::*, types::PyTuple};
    use std::{
        collections::BTreeMap,
        fmt::{Display, Formatter},
    };

    use mefikit::prelude as mf;

    use std::path::Path;

    use numpy as np;
    use serde_json;

    #[pyclass(str)]
    #[pyo3(name = "UMesh")]
    #[derive(PartialEq)]
    pub struct PyUMesh {
        inner: mf::UMesh,
    }

    fn etype_to_str(et: mf::ElementType) -> String {
        use mf::ElementType::*;
        match et {
            VERTEX => "VERTEX",
            SEG2 => "SEG2",
            SEG3 => "SEG3",
            SEG4 => "SEG4",
            SPLINE => "SPLINE",
            TRI3 => "TRI3",
            TRI6 => "TRI6",
            TRI7 => "TRI7",
            QUAD4 => "QUAD4",
            QUAD8 => "QUAD8",
            QUAD9 => "QUAD9",
            PGON => "PGON",
            TET4 => "TET4",
            TET10 => "TET10",
            HEX8 => "HEX8",
            HEX21 => "HEX21",
            PHED => "PHED",
        }
        .to_string()
    }

    fn str_to_etype(et: &str) -> mf::ElementType {
        use mf::ElementType::*;
        match et {
            "VERTEX" => VERTEX,
            "SEG2" => SEG2,
            "SEG3" => SEG3,
            "SEG4" => SEG4,
            "TRI3" => TRI3,
            "TRI6" => TRI6,
            "TRI7" => TRI7,
            "QUAD4" => QUAD4,
            "QUAD8" => QUAD8,
            "QUAD9" => QUAD9,
            "TET4" => TET4,
            "TET10" => TET10,
            "HEX8" => HEX8,
            "HEX21" => HEX21,
            _ => panic!("Unsupported element type: {}", et),
        }
    }

    #[derive(IntoPyObject)]
    enum PyConnectivity<'py> {
        Regular(Bound<'py, np::PyArray2<usize>>),
        Poly(
            Bound<'py, np::PyArray1<usize>>,
            Bound<'py, np::PyArray1<usize>>,
        ),
    }

    #[pymethods]
    impl PyUMesh {
        #[new]
        fn new(coords: np::PyReadonlyArray2<'_, f64>) -> Self {
            mf::UMesh::new(coords.as_array().to_shared()).into()
        }

        /// Returns a copy owned by python of the array coordinates
        fn coords<'py>(&self, py: Python<'py>) -> Bound<'py, np::PyArray2<f64>> {
            np::PyArray2::from_array(py, &self.inner.coords())
        }

        fn block_types(&self) -> Vec<String> {
            self.inner
                .blocks()
                .map(|(&et, _)| etype_to_str(et))
                .collect()
        }

        fn blocks<'py>(&self, py: Python<'py>) -> BTreeMap<String, PyConnectivity<'py>> {
            self.inner
                .blocks()
                .map(|(&et, block)| {
                    let et = etype_to_str(et);
                    let conn = match &block.connectivity {
                        mf::Connectivity::Regular(c) => {
                            PyConnectivity::Regular(np::PyArray2::from_array(py, c))
                        }
                        mf::Connectivity::Poly { data, offsets } => PyConnectivity::Poly(
                            np::PyArray1::from_array(py, data),
                            np::PyArray1::from_array(py, offsets),
                        ),
                    };
                    (et, conn)
                })
                .collect()
        }

        fn to_json(&self) -> String {
            serde_json::to_string(&self.inner).unwrap()
        }

        fn to_json_pretty(&self) -> String {
            serde_json::to_string_pretty(&self.inner).unwrap()
        }

        /// Add a regular block of elements to the mesh.
        fn add_regular_block(&mut self, et: &str, block: np::PyReadonlyArray2<'_, usize>) {
            self.inner
                .add_regular_block(str_to_etype(et), block.as_array().to_shared());
        }

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

        #[pyo3(signature = (src_dim=None, target_dim=None))]
        fn submesh(&self, src_dim: Option<usize>, target_dim: Option<usize>) -> Self {
            let src_dim = src_dim.map(|i| i.try_into().unwrap());
            let target_dim = target_dim.map(|i| i.try_into().unwrap());
            mf::compute_submesh(&self.inner, src_dim, target_dim).into()
        }

        #[pyo3(signature = (src_dim=None))]
        fn boundaries(&self, src_dim: Option<usize>) -> Self {
            let src_dim = src_dim.map(|i| i.try_into().unwrap());
            mf::compute_boundaries(&self.inner, src_dim).into()
        }

        fn measure<'py>(&self, py: Python<'py>) -> BTreeMap<String, Bound<'py, np::PyArray1<f64>>> {
            mf::measure(self.inner.view())
                .iter()
                .map(|(&et, arr)| (etype_to_str(et), np::PyArray1::from_array(py, arr)))
                .collect()
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
