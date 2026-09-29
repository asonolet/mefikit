//! Re-exports of [`ffi::Error`] so downstream Rust code can name it without
//! reaching into the bridge module.

pub use crate::ffi::Error;
