//! Element trait definitions for geometric and topological operations.
//!
//! This module provides traits that extend elements with geometric queries
//! (coordinates, measures, centroids) and topological operations
//! (subentities, simplex decomposition).

pub mod cut;
mod element_geo;
mod element_topo;
pub mod is_in;
pub mod measures;
mod seg_intersect;
mod symmetry;
mod utils;

pub use cut::Cutable;
pub use element_geo::ElementGeo;
pub use element_topo::ElementTopo;
pub use seg_intersect::{Intersection, Intersections, PointId, intersect_seg_seg};
pub use utils::SortedVecKey;
