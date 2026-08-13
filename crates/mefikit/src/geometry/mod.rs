//! Internal geometry: owned geometric primitives and point-in-region tests.
//!
//! This module provides owned [`Segment`], [`Polygon`] and [`Polyhedron`] values together with
//! robust point-in-region tests and convexity tracking. Element traits and mesh tools delegate
//! their geometric queries to these types.

mod convexity;
mod polygon;
mod polyhedron;
mod region;
mod segment;

pub use convexity::Convexity;
pub use polygon::{Polygon, in_bezier_polygon, in_quadratic_polygon};
pub use polyhedron::{Polyhedron, point_in_phed, point_in_phed2};
pub use region::{in_aa_bbox, in_aa_rectangle, in_circle, in_sphere};
pub use segment::{Intersection, Intersections, PointId, Segment, intersect_seg_seg};

pub(crate) use polygon::{
    area_polygon3, area_quad2, area_tri2, convex_polygon_contains2, vertex_centroid,
};
pub(crate) use polyhedron::{hex_volume, tet_volume};
