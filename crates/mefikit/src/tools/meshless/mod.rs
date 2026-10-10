//! Shared meshless reconstruction machinery.
//!
//! This module holds the pieces common to every nearest-neighbour-based reconstruction operator
//! (the moving least-squares / inverse-distance transfers and the gradient tool, and any future
//! operator such as divergence or laplacian):
//!
//! - [`DistanceWeighting`], the distance kernel of the local weighted fit.
//! - the point-level solvers (k-nearest-neighbours search, kernel evaluation, affine fit), in the
//!   [`solver`] submodule.
//! - the mesh-to-point plumbing (source/target centroids, dimension validation), in the [`points`]
//!   submodule.

pub mod kernel;
mod points;
mod solver;

pub use kernel::DistanceWeighting;
pub(crate) use points::{source_centroids, target_centroids, validated_dims};
pub(crate) use solver::{NeighbourScheme, solve_gradient_neighbours, solve_neighbours};
