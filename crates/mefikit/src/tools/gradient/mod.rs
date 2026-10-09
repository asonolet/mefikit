//! Gradient of a scalar field on meshes and point clouds.
//!
//! This tool reuses the same meshless machinery as the transfer tool (see `tools::meshless`) to
//! estimate the gradient of a per-cell scalar field at the cells of a mesh or at an arbitrary
//! point cloud. The result is a vector field with one trailing component per space direction.

mod moving_least_squares;
mod new;
mod operator;

pub use operator::{Gradient, GradientMethod, GradientOperator, GradientTarget};
