use pyo3::prelude::*;

mod element;
mod element_ids;
mod pyfield;
mod pytransfer;
mod pyumesh;
mod select;

#[pymodule]
mod sel {
    #[pymodule_export]
    use super::select::{
        bbox, circle, exclude_group, group, ids, nbbox, ncircle, nids, nrect, nsphere, rect,
        sphere, types,
    };
}

#[pymodule]
mod transfer {
    #[pymodule_export]
    use super::pytransfer::{
        PyConservativeP0, PyConstantPiecewise, PyDistanceWeighting, PyInverseDistance,
        PyMovingLeastSquares,
    };
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
mod mefipy {
    use pyo3::{prelude::*, types::PyTuple};

    use mefikit::prelude as mf;

    #[pymodule_export]
    use super::sel;

    #[pymodule_export]
    use super::transfer;

    #[pymodule_export]
    use super::pyumesh::{PyOverlayOperation, PySurfaceOverlay, PyUMesh};

    #[pymodule_export]
    use super::pyfield::PyField;

    #[pyfunction]
    #[pyo3(signature = (*args))]
    pub fn build_cmesh(args: &Bound<'_, PyTuple>) -> PyResult<PyUMesh> {
        let mut builder = mf::RegularUMeshBuilder::new();
        for arg in args {
            builder = builder.add_axis(arg.extract()?)
        }
        Ok(builder.build().into())
    }
}
