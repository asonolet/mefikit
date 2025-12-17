pub mod connected_components;
pub mod grid;
/// Module for intersecting meshes.
///
/// In this context, intersections operations can be separated in the following cases:
/// - 1d + 1d
///   - cut: Intersection of 1D elements with other 1D elements producing 0D elements.
///   - cut_add: Intersecting 1D elements with 1D elements producing 1D mesh conformized to both
///     meshes.
/// - 2d + 1d
///   - cut_edges: Intersecting 2D elements with 1D elements and producing 2D mesh which contains
///     intersections nodes in the connectivity. This new 2D mesh has the same number of elements as
///     the original 2D mesh.
///   - cute_faces: Intersecting 2D elements with 1D elements and producing 2D mesh by cutting the
///     2D mesh with the 1D mesh. The new 2D mesh may contains more elements, and conformizes with
///     the 1D mesh.
/// - 2d + 2d
///   - cut_union: Intersecting both meshes fully.
///   - cut_intersect: Only keeping the domain covered by both meshes and intersecting them.
///   - cut_xor: Only keeping the domain covered by one of the meshes and intersecting them.
///
/// The input meshes do not need to be clean (i.e. they can have unmerged nodes). They need to be
/// conformed (i.e. no overlapping elements).
/// In all cases, the operation gives a "conformized without merging nodes" mesh. The user can
/// choose to merge nodes after the operation if needed.
///
/// Note: The implementation of these operations is not trivial. The main difficulty is to
/// manage non conformities and numerical precision issues. The implementation should be robust
/// and handle these issues gracefully.
pub mod intersect;
pub mod measure;
pub mod neighbours;
pub mod selector;
pub mod snap;

pub use connected_components::*;
pub use grid::*;
pub use intersect::*;
pub use measure::*;
pub use neighbours::*;
pub use selector::*;
pub use snap::*;
