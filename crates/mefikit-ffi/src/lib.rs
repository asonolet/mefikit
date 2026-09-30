//! C++ bindings for [mefikit](https://github.com/asonolet/mefikit), built with
//! [`cxx`](https://cxx.rs).
//!
//! The C++ API is declared in [`ffi::bridge`] and implemented across
//! [`mesh`] (the `UMesh` handle) and [`transfer`] (the `TransferOperator`
//! handle). See `crates/mefikit-ffi/README.md` for how to build and link it.
//!
//! ```no_run
//! use mefikit_ffi::ffi::{UMesh, bridge::ElementType};
//!
//! let coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
//! let mut mesh = UMesh::from_coords(&coords, 4, 2).unwrap();
//! mesh.add_regular_block(ElementType::QUAD4.into(), &[0, 1, 2, 3], 1).unwrap();
//! assert_eq!(mesh.n_nodes(), 4);
//! ```

pub mod error;
pub mod ffi;
pub mod mesh;
pub mod transfer;
pub mod types;

pub use error::Error;
pub use ffi::{TransferOperator, UMesh, bridge, transfer_field};
