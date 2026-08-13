//! Internal geometry: owned geometric primitives and point-in-region tests.
//!
//! # Layering contract
//!
//! This module is organized in two layers:
//!
//! 1. **Owned types**: [`Segment`](crate::geometry::Segment),
//!    [`Polygon`](crate::geometry::Polygon) and [`Polyhedron`](crate::geometry::Polyhedron) — the
//!    canonical API, carrying their own data and lazily tracked convexity. Everything here is
//!    `pub` and is the supported surface for element traits and mesh tools.
//! 2. **`pub(crate)` helpers**: `area_tri2`, `area_quad2`, `area_polygon3`, `signed_area2`,
//!    `into_ccw2`, `vertex_centroid`, `bounds_iter`, `cross2`, `convex_polygon_contains2`,
//!    `tet_volume`, `hex_volume`, `tet_contains`, `hex_contains`. These are allocation-free,
//!    bit-exact shortcuts for the common regular element types (TRI3, QUAD4, SEG2, TET4, HEX8).
//!
//! Rules:
//!
//! - **Fast paths must never allocate or build owned types.** The hot paths (`measure*`,
//!   `centroid*`, `bounds*`, `is_point_inside` in `element_geo.rs`, the conservative transfer in
//!   `conservative_p0.rs`) route regular elements through the `pub(crate)` helpers. Building a
//!   `Polygon`/`Polyhedron` per element was the cause of a single-threaded regression and is a
//!   misuse of this module. The `*_matches_*` tests pin the helpers to be bit-exact with the
//!   owned types.
//! - **Naming convention**: the 2D `pub(crate)` helpers carry a `2`/`3` suffix (`area_tri2`,
//!   `area_polygon3`), while shared primitives that appear in several places (`cross2`,
//!   `bounds_iter`, `signed_area2`, `into_ccw2`, `vertex_centroid`) have no suffix and are
//!   re-exported once from `mod.rs`. Prefer the shared primitive over a local copy when the
//!   computation is identical.
//! - **Exact vs robust predicates**: `orient2d2` (Shewchuk via the `robust` crate) is the exact
//!   orientation test driving `Polygon::contains` and convexity. The naive `cross2` is the shared
//!   helper for ear clipping and Sutherland–Hodgman clipping, where vertices come from an already
//!   constructed face and small scale error is acceptable. Do not silently swap one for the
//!   other. (`point_in_phed2` is a backward-compatibility alias of `point_in_phed`.)

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
    area_polygon3, area_quad2, area_tri2, bounds_iter, convex_polygon_contains2, cross2, into_ccw2,
    signed_area2, vertex_centroid,
};
pub(crate) use polyhedron::{hex_contains, hex_volume, tet_contains, tet_volume};
