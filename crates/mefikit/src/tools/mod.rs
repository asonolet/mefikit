//! Mesh manipulation tools and algorithms.
//!
//! This module provides various utilities for mesh operations including:
//! - Connected component analysis
//! - Mesh cracking (splitting shared nodes/faces)
//! - Mesh extrusion (raising dimension)
//! - Field expressions and evaluation
//! - Structured grid generation
//! - Mesh overlay operations
//! - Geometric measurements
//! - Neighbor computation
//! - Element selection
//! - Node snapping
//! - Field transfer between meshes

/// Centroids of meshes.
pub mod centroids;
/// Connected component analysis for meshes.
pub mod connected_components;
/// Crack along shared faces/nodes to separate mesh regions.
pub mod crack;
/// Mesh extrusion to build a higher-dimensional mesh.
pub mod extrude;
/// Field expression evaluation and manipulation.
pub mod fieldexpr;
/// Structured grid generation utilities.
pub mod grid;
/// Geometric measurement utilities for meshes.
pub mod measure;
/// Neighbor computation for mesh elements.
pub mod neighbours;
/// Boolean-like overlay operations on 2D meshes.
pub mod overlay;
/// Element and node selection utilities.
pub mod selector;
/// Node snapping to merge nearby nodes.
pub mod snap;
pub mod spatial_index;
/// Cell splitting to create finer meshes.
pub mod split_cells;
/// Reusable field transfer operators between meshes.
pub mod transfer;

pub use centroids::*;
pub use connected_components::*;
pub use crack::*;
pub use extrude::*;
pub use grid::*;
pub use measure::*;
pub use neighbours::*;
pub use overlay::*;
pub use selector::*;
pub use snap::*;
pub use split_cells::*;
pub use transfer::*;
