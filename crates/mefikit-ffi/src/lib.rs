//! Minimal C++ bindings for mefikit, built with [`cxx`](https://cxx.rs).
//!
//! The C++ API is deliberately limited to:
//!
//! - building a [`UMesh`] from coordinates and element blocks;
//! - reading and writing one element type's field values;
//! - preparing and applying any of mefikit's four transfer methods:
//!   conservative P0, piecewise-constant, inverse-distance and moving
//!   least-squares.
//!
//! Everything C++ can see is declared in [`ffi::bridge`]. The implementation
//! lives in [`mesh`] and [`transfer`].

mod error;
pub mod ffi;
mod mesh;
mod transfer;
mod types;

pub use error::Error;
pub use ffi::{Transfer, UMesh};
