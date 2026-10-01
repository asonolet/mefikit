//! C++ bindings for [mefikit](https://github.com/asonolet/mefikit), built with
//! [`cxx`](https://cxx.rs).
//!
//! The C++ API is declared in [`ffi::bridge`] and implemented across
//! [`mesh`] (the `UMesh` handle) and [`transfer`] (the `TransferOperator`
//! handle). See `crates/mefikit-ffi/README.md` for how to build and link it.
//!
//! ```ignore
//! // C++: every array is a plain const std::vector reference.
//! auto mesh = mefikit::UMesh::from_coords(coords, n_nodes, space_dim);
//! mesh->add_regular_block(mefikit::ElementType::QUAD4, conn, n_elements);
//! mesh->set_field_uniform("T", mefikit::ElementType::QUAD4, 1, values);
//! ```

pub mod error;
pub mod ffi;
pub mod mesh;
pub mod transfer;
pub mod types;

pub use error::Error;
pub use ffi::{TransferOperator, UMesh, bridge, set_field, transfer_field};
