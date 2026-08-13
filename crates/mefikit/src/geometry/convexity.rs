//! Convexity status of geometric regions.

/// Known convexity of a region.
///
/// Element types with a fixed number of nodes (TRI3, QUAD4, TET4, ...) are always convex, so
/// their convexity is known at construction time. Arbitrary polygons and polyhedra (PGON, PHED)
/// can be concave and require an on-demand test.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Convexity {
    /// The region is known to be convex.
    Convex,
    /// The region is known to be concave.
    Concave,
    /// The convexity is not known and must be computed on demand.
    Unknown,
}
