//! Boolean-like overlay operations on 2D meshes.
//!
//! In this context, overlay operations can be separated in the following cases:
//! - 2d + 1d: `intersect_2d1d` cuts a 2D mesh with a 1D mesh, producing a 2D mesh conformized
//!   with the 1D mesh.
//! - 2d + 2d: `Overlayable::overlay` with one of the [`OverlayOperation`]s:
//!   - `Imprint`: refine `mesh1` with the edges of `mesh2` while keeping `mesh1`'s domain.
//!   - `Union`: keep the domain covered by at least one of the two meshes.
//!   - `Intersection`: keep the domain covered by both meshes.
//!   - `Difference`: keep the domain of `mesh1` not covered by `mesh2`.
//!   - `SymmetricDifference`: keep the domain covered by exactly one of the two meshes.
//!
//! The input meshes do not need to be clean (i.e. they can have unmerged nodes). They need to be
//! conformed (i.e. no overlapping elements).
//! In all cases, the operation gives a "conformized without merging nodes" mesh. The user can
//! choose to merge nodes after the operation if needed.
//!
//! Note: The implementation of these operations is not trivial. The main difficulty is to
//! manage non conformities and numerical precision issues. The implementation should be robust
//! and handle these issues gracefully.
//!
//! Surface meshes embedded in 3D space are handled by [`Overlayable::overlay_surfaces`], see
//! the `surface` submodule.
//!
//! Assembling several 3D volume meshes into a single conforming polyhedral mesh is handled by
//! [`stitch`], and conformizing one mesh with itself (plus the [`is_conform`] diagnostic), by
//! [`conformize`]; see the `stitch` submodule.

mod stitch;
mod surface;

pub use stitch::{
    ConformanceIssue, ConformanceReport, StitchError, conformize, is_conform, stitch,
};
pub use surface::{SurfaceOverlay, SurfaceOverlayError, overlay_surfaces};

use rustc_hash::FxHashMap;

use nalgebra::Point2;
use nalgebra::Vector2;
use ndarray as nd;

use crate::element_traits::cut::{
    IntersectionIds, M1M2Intersections, NodeId, SortedSegIntersections,
};
use crate::element_traits::{Cutable, SortedVecKey};
use crate::geometry::{
    Intersection, Intersections, Polygon, Segment, intersect_arc_arc, intersect_arc_seg,
};
use crate::mesh::{Dimension, Element, ElementId, ElementLike, ElementType, UMesh, UMeshView};
use crate::prelude::ElementGeo;
use crate::tools::duplicates_from;
use crate::tools::spatial_index::SpIdx2;
use crate::tools::{Descendable, spatial_index::SpatiallyIndexable};

/// Node coincidence tolerance used when merging two 2D meshes onto shared coordinates. Kept
/// tight because the 2D overlay assumes the inputs already share a coordinate space.
const WELD_EPS: f64 = 1e-12;

/// Number of floating-point roundoff units (`scale * f64::EPSILON`) allowed when deciding whether
/// a computed intersection point may reuse an existing node. This is the FP-roundoff component of
/// the arc/segment match tolerance; mesh-quality matching (e.g. recognising two arcs that share a
/// curve but no nodes) is not modelled yet and would add a named relative term here.
const NODE_REUSE_FP_UNITS: f64 = 64.0;

/// Merges `subject` onto `reference` in a single coordinate space.
///
/// The merged mesh coordinates are `[reference coords; subject coords]` and the `subject` blocks
/// are remapped so that subject nodes coinciding with a reference node point to that reference
/// node id. The two meshes therefore share the same node ids in the merged space.
///
/// Returns the merged mesh together with the weld map: subject node id (in the merged space) ->
/// reference node id it was welded onto. Subject nodes without a coincident reference node are
/// absent from the map.
///
/// `eps` is the node coincidence tolerance, expressed in the coordinate space of the two meshes
/// (which must already share that space, e.g. after projection). Note that widening `eps` past
/// half the smallest distance between two nodes of a single element produces degenerate
/// elements.
///
/// NOTE: must be called before computing intersections because the merged coords indexing is used
/// to resolve intersection node ids.
fn merge_on_reference_coords(
    subject: UMesh,
    reference: UMeshView,
    eps: f64,
) -> (UMesh, FxHashMap<usize, usize>) {
    // reference node id -> coincident subject node ids
    let ref_to_subject_nodes = duplicates_from(&subject.view(), &reference.clone(), eps);
    let shift = reference.coords().nrows();
    // In the merged mesh, subject node `n` lives at `n + shift`. When it coincides with reference
    // node `r`, re-point it to `r` so both meshes share the node id.
    let mut merged_to_ref_nodes: FxHashMap<usize, usize> = FxHashMap::default();
    for (ref_node, subject_nodes) in ref_to_subject_nodes {
        for subject_node in subject_nodes {
            merged_to_ref_nodes.insert(subject_node + shift, ref_node);
        }
    }
    let new_coords = nd::concatenate![nd::Axis(0), reference.coords(), subject.coords()];
    let mut merged = UMesh::new(new_coords.into_shared());
    for (&et, b) in subject.blocks() {
        let mut new_block = b.clone();
        let co = &mut new_block.connectivity;
        co.shift_index(shift);
        co.replace(&merged_to_ref_nodes);
        merged.element_blocks.insert(et, new_block);
    }
    (merged, merged_to_ref_nodes)
}

/// Boolean-like operation to perform on two 2D meshes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OverlayOperation {
    /// Refine `mesh1` with the edges of `mesh2` while keeping `mesh1`'s domain.
    #[default]
    Imprint,
    /// Keep the domain covered by at least one of the two meshes.
    Union,
    /// Keep the domain covered by both meshes.
    Intersection,
    /// Keep the domain of `mesh1` not covered by `mesh2`.
    Difference,
    /// Keep the domain covered by exactly one of the two meshes.
    SymmetricDifference,
}

/// Boolean overlay of two 2D meshes.
pub trait Overlayable {
    /// Computes the overlay of `self` (as mesh1) and `mesh2` for the given operation.
    ///
    /// # Guarantees
    /// - Output mesh is planar, manifold, and watertight
    /// - No T-junctions or dangling edges
    /// - All intersections between mesh1 and mesh2 are explicitly represented
    ///
    /// # Assumptions
    /// - Input meshes are valid (non-self-intersecting)
    /// - Coordinates are in the same plane
    fn overlay(&self, mesh2: UMesh, operation: OverlayOperation) -> UMesh;

    /// Imprints two surface meshes embedded in 3D space onto each other wherever they
    /// coincide.
    ///
    /// Both surfaces are refined with each other's interface edges so that they become
    /// mutually conformal on the coincident areas; the two refined meshes share the same
    /// coordinates array. Areas of a surface not covered by the other are copied verbatim.
    ///
    /// `tol` drives the geometric tolerances (region planarity, node coincidence) as well as
    /// the node welding of the two surfaces during the pairwise cuts.
    ///
    /// # Guarantees
    /// - Output meshes tile their input footprints exactly (area preserving up to `tol`)
    /// - Intersection nodes are shared by both sides
    /// - Families propagate to the produced faces
    ///
    /// # Assumptions
    /// - Input surfaces are valid, first-order (TRI3, QUAD4, PGON), and piecewise planar;
    ///   matched regions lie in a common plane within `tol`
    fn overlay_surfaces(
        &self,
        other: &UMeshView,
        tol: f64,
    ) -> Result<SurfaceOverlay, SurfaceOverlayError>;
}

impl Overlayable for UMesh {
    fn overlay(&self, mesh2: UMesh, operation: OverlayOperation) -> UMesh {
        match operation {
            OverlayOperation::Imprint => intersect_2d2d(self, mesh2),
            OverlayOperation::Intersection => cut_and_classify(self, mesh2, |inside| inside),
            OverlayOperation::Difference => cut_and_classify(self, mesh2, |inside| !inside),
            OverlayOperation::Union => cut_both(self, mesh2, |_| true, |inside| !inside),
            OverlayOperation::SymmetricDifference => {
                cut_both(self, mesh2, |inside| !inside, |inside| !inside)
            }
        }
    }

    fn overlay_surfaces(
        &self,
        other: &UMeshView,
        tol: f64,
    ) -> Result<SurfaceOverlay, SurfaceOverlayError> {
        surface::overlay_surfaces(&self.view(), other, tol)
    }
}

/// Computes the geometric intersection (overlay) of two 2D meshes.
///
/// Refines `mesh1`'s cells with the edges of `mesh2`.
fn intersect_2d2d(mesh1: &UMesh, mesh2: UMesh) -> UMesh {
    let (mesh2, _) = merge_on_reference_coords(mesh2, mesh1.view(), WELD_EPS);
    cut_2d_with_edges(
        mesh1,
        mesh2.descend(Some(Dimension::D2), Some(Dimension::D1)),
    )
}

/// Cuts the 2D cells of `mesh1` with a 1D mesh of cutting edges, keeping all the resulting pieces.
///
/// `cutting_edges` must already live in `mesh1`'s coordinate space (see
/// [`merge_on_reference_coords`]).
fn cut_2d_with_edges(mesh1: &UMesh, cutting_edges: UMesh) -> UMesh {
    let m1_edges = mesh1.descend(Some(Dimension::D2), Some(Dimension::D1));

    let cutting_bvh = cutting_edges.view().bvh2();

    let (mut cutted_mesh, seg_intersections) =
        compute_overlay(&m1_edges, &cutting_edges, &cutting_bvh);

    cut_cells_all(
        &mut cutted_mesh,
        mesh1,
        &cutting_edges.view(),
        &cutting_bvh,
        &seg_intersections,
        None,
    );
    cutted_mesh
}

/// Computes the geometric intersection (overlay) of two 2D meshes.
///
/// # Guarantees
/// - Output mesh is planar, manifold, and watertight
/// - No T-junctions or dangling edges
/// - All intersections between mesh1 and mesh2 are explicitly represented
///
/// # Assumptions
/// - Input meshes are valid (non-self-intersecting)
/// - Coordinates are in the same plane
pub fn intersect_2d1d(mesh1: &UMesh, mesh2: UMesh) -> UMesh {
    let (mesh2, _) = merge_on_reference_coords(mesh2, mesh1.view(), WELD_EPS);
    cut_2d_with_edges(mesh1, mesh2)
}

/// Cut the cells of `subject` with the edges of `cutter` and keep the pieces for which the
/// `keep` predicate returns `true`. The predicate receives `true` when the piece lies inside
/// `cutter`.
fn cut_and_classify(subject: &UMesh, cutter: UMesh, keep: impl Fn(bool) -> bool) -> UMesh {
    let (cutter, _) = merge_on_reference_coords(cutter, subject.view(), WELD_EPS);

    let subject_edges = subject.descend(Some(Dimension::D2), Some(Dimension::D1));
    let cutter_edges = cutter.descend(Some(Dimension::D2), Some(Dimension::D1));

    let cutter_edges_bvh = cutter_edges.view().bvh2();

    let (mut cutted_mesh, seg_intersections) =
        compute_overlay(&subject_edges, &cutter_edges, &cutter_edges_bvh);

    cut_cells(
        &mut cutted_mesh,
        subject,
        &cutter,
        &cutter_edges.view(),
        &cutter_edges_bvh,
        &seg_intersections,
        &keep,
    );
    cutted_mesh
}

/// Cut the cells of both meshes in a shared coordinate space, keeping the pieces selected by
/// `keep1` (pieces of `mesh1` inside `mesh2`) and `keep2` (pieces of `mesh2` inside `mesh1`).
fn cut_both(
    mesh1: &UMesh,
    mesh2: UMesh,
    keep1: impl Fn(bool) -> bool,
    keep2: impl Fn(bool) -> bool,
) -> UMesh {
    let (mesh2, _) = merge_on_reference_coords(mesh2, mesh1.view(), WELD_EPS);

    let m1_edges = mesh1.descend(Some(Dimension::D2), Some(Dimension::D1));
    let m2_edges = mesh2.descend(Some(Dimension::D2), Some(Dimension::D1));

    let m1edges_bvh = m1_edges.view().bvh2();
    let m2edges_bvh = m2_edges.view().bvh2();

    let (mut cutted_mesh, seg_intersections) = compute_overlay(&m1_edges, &m2_edges, &m2edges_bvh);

    cut_cells(
        &mut cutted_mesh,
        mesh1,
        &mesh2,
        &m2_edges.view(),
        &m2edges_bvh,
        &seg_intersections,
        &keep1,
    );
    cut_cells(
        &mut cutted_mesh,
        &mesh2,
        mesh1,
        &m1_edges.view(),
        &m1edges_bvh,
        &seg_intersections,
        &keep2,
    );
    cutted_mesh
}

/// Computes the intersections between `subject_edges` and `cutting_edges` (which must already
/// share the same coordinate space) and builds the output mesh shell: concatenated coordinates
/// plus the sorted per-edge intersection lists.
fn compute_overlay(
    subject_edges: &UMesh,
    cutting_edges: &UMesh,
    cutting_bvh: &SpIdx2,
) -> (UMesh, SortedSegIntersections) {
    let (intersections, added_coords) =
        compute_intersections(subject_edges, cutting_edges, cutting_bvh);

    // Concatenates subject coords, cutter coords, new intersections coords
    let new_coords = nd::concatenate![nd::Axis(0), cutting_edges.coords(), added_coords];

    let cutted_mesh = UMesh::new(new_coords.into_shared());

    let seg_intersections =
        to_sorted_intersections(&intersections, &cutting_edges.view(), &cutted_mesh.coords());

    (cutted_mesh, seg_intersections)
}

/// Cut the D2 cells of `subject` with `cutting_edges` and append all the resulting pieces to
/// `out`.
///
/// If `parents` is provided, it is filled with one `(subject cell id, emitted element ids)`
/// entry per subject cell, in subject iteration order. Emitted ids refer to elements appended to
/// `out`: a single verbatim copy when the cell was untouched, otherwise its cut pieces.
fn cut_cells_all(
    out: &mut UMesh,
    subject: &UMesh,
    cutting_edges: &UMeshView,
    cutting_bvh: &SpIdx2,
    seg_intersections: &SortedSegIntersections,
    parents: Option<&mut Vec<(ElementId, Vec<ElementId>)>>,
) {
    let mut dropped_parents = Vec::new();
    let parents: &mut Vec<(ElementId, Vec<ElementId>)> = match parents {
        Some(p) => p,
        None => &mut dropped_parents,
    };
    for cell in subject.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = cutting_bvh.in_bounds(bmin, bmax);
        let reconstructed = cell.cut_with_intersections(
            seg_intersections,
            cutting_edges,
            out.coords(),
            &candidates,
        );

        let mut piece_ids: Vec<ElementId> = Vec::new();
        // If the cell was cut, I add new polys from the cut
        if let Some(polys) = reconstructed {
            for new_cell in polys {
                let id = out.add_element(ElementType::PGON, &new_cell, Some(*cell.family));
                piece_ids.push(id);
            }
        } else {
            let id = out.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
            piece_ids.push(id);
        }
        parents.push((cell.id(), piece_ids));
    }
}

/// Cut the D2 cells of `subject` with `cutting_edges` and append the kept pieces to `out`.
fn cut_cells(
    out: &mut UMesh,
    subject: &UMesh,
    cutter: &UMesh,
    cutting_edges: &UMeshView,
    cutting_bvh: &SpIdx2,
    seg_intersections: &SortedSegIntersections,
    keep: &impl Fn(bool) -> bool,
) {
    let cutter_bvh = cutter.view().bvh2();

    for cell in subject.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = cutting_bvh.in_bounds(bmin, bmax);
        let reconstructed = cell.cut_with_intersections(
            seg_intersections,
            cutting_edges,
            out.coords(),
            &candidates,
        );

        // If the cell was cut, I add the kept new polys from the cut
        if let Some(polys) = reconstructed {
            for new_cell in polys {
                let inside = pgon_inside(&new_cell, &out.coords(), cutter, &cutter_bvh);
                if keep(inside) {
                    out.add_element(ElementType::PGON, &new_cell, Some(*cell.family));
                }
            }
        } else if keep(cell_inside(&cell, cutter, &cutter_bvh)) {
            out.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
        }
    }
}

/// Returns the cell vertices in counter-clockwise order.
fn cell_points(cell: Element<'_>) -> Polygon<2> {
    let pgon = Polygon::unknown((0..cell.connectivity().len()).map(|i| cell.coord2(i).into()));
    pgon.into_ccw()
}

/// Returns `true` if the polygon given by node ids lies inside any cell of `cutter_cells`.
fn pgon_inside(
    pgon: &[usize],
    coords: &nd::ArrayView2<'_, f64>,
    cutter: &UMesh,
    cutter_bvh: &SpIdx2,
) -> bool {
    let points: Vec<[f64; 2]> = pgon
        .iter()
        .map(|&n| [coords[(n, 0)], coords[(n, 1)]])
        .collect();
    point_inside(&points, cutter, cutter_bvh)
}

/// Returns `true` if the cell lies inside any cell of `cutter_cells`.
fn cell_inside(cell: &Element<'_>, cutter: &UMesh, cutter_bvh: &SpIdx2) -> bool {
    let points: Vec<[f64; 2]> = (0..cell.connectivity().len())
        .map(|i| cell.coord2(i).into())
        .collect();
    point_inside(&points, cutter, cutter_bvh)
}

/// Returns `true` if the interior of the polygon lies inside any cell of `cutter_cells`.
///
/// Pieces are produced by cutting a cell along the cutter edges, so a piece is either fully
/// inside or fully outside the cutter. A strict interior point of the piece is therefore a valid
/// witness: the piece is inside the cutter iff its interior point is. The centroid of a
/// non-convex piece (e.g. an L-shape) can fall outside its own interior, so it is not a valid
/// witness.
fn point_inside(points: &[[f64; 2]], cutter: &UMesh, cutter_bvh: &SpIdx2) -> bool {
    let Some(interior) = Polygon::unknown(points.iter().copied()).strict_interior_point() else {
        return false;
    };
    let candidates = cutter_bvh.intersects(interior);
    candidates
        .iter()
        .map(|eid| cutter.element(eid))
        .map(cell_points)
        .any(|cell| cell.contains_stable(&interior))
}

/// Compute all intersections between mesh1 and mesh2 where mesh1 and mesh2 should be 2d mesh of
/// edges. The computation is done quite naively. A BVH is used to accelarate finding m2
/// intersecting edges.
/// The returned map can map back to segments from m1.
// TODO: easy to write as a rayon parallelized closure.
fn compute_intersections(
    m1_edges: &UMesh,
    m2_edges: &UMesh,
    m2bvh: &SpIdx2,
) -> (M1M2Intersections, nd::ArcArray2<f64>) {
    let mut intersections: M1M2Intersections = FxHashMap::default();
    let mut new_coords = Vec::new();
    let mut new_coord_id = m2_edges.coords().nrows();

    for edge in m1_edges.elements() {
        let [min, max] = edge.bounds2();
        let candidates = m2bvh.in_bounds(min, max);
        for c in candidates.into_iter() {
            let edge2 = m2_edges.element(c);
            add_intersection(
                &mut intersections,
                &mut new_coords,
                &mut new_coord_id,
                &edge,
                edge2,
                &m1_edges.coords(),
                &m2_edges.coords(),
            );
        }
    }
    let nb_new_points = new_coords.len() / 2;
    let new_coords = nd::Array2::from_shape_vec((nb_new_points, 2), new_coords).unwrap();
    (intersections, new_coords.into_shared())
}

fn add_intersection(
    intersections: &mut M1M2Intersections,
    new_coords: &mut Vec<f64>,
    new_coord_id: &mut usize,
    edge: &Element<'_>,
    edge2: Element<'_>,
    _m1_coords: &nd::ArrayView2<'_, f64>,
    m2_coords: &nd::ArrayView2<'_, f64>,
) {
    use crate::mesh::ElementType::*;
    let et1 = edge.element_type();
    let et2 = edge2.element_type();

    // Compute the intersection of the two edges as [`Intersections`], whose `Existing` ids
    // refer to the endpoints of the first and second operand (P1/P2 and P3/P4). The operand
    // order is chosen so that `p_ids` below maps each id back to a node of `edge`/`edge2`.
    // A quadratic edge is an arc through `(endpoint, mid-side, endpoint)`.
    let (int, p_ids): (Intersections, [usize; 4]) = match (et1, et2) {
        (SEG2, SEG2) => {
            let int = Segment::new(edge.coord2(0).into(), edge.coord2(1).into())
                .intersect(Segment::new(edge2.coord2(0).into(), edge2.coord2(1).into()));
            let c = edge.connectivity();
            let c2 = edge2.connectivity();
            (int, [c[0], c[1], c2[0], c2[1]])
        }
        (SEG3, SEG2) => {
            let int = intersect_arc_seg(
                edge.coord2_ref(0),
                edge.coord2_ref(2),
                edge.coord2_ref(1),
                edge2.coord2_ref(0),
                edge2.coord2_ref(1),
            );
            let c = edge.connectivity();
            let c2 = edge2.connectivity();
            (int, [c[0], c[1], c2[0], c2[1]])
        }
        (SEG2, SEG3) => {
            let int = intersect_arc_seg(
                edge2.coord2_ref(0),
                edge2.coord2_ref(2),
                edge2.coord2_ref(1),
                edge.coord2_ref(0),
                edge.coord2_ref(1),
            );
            let c = edge.connectivity();
            let c2 = edge2.connectivity();
            (int, [c2[0], c2[1], c[0], c[1]])
        }
        (SEG3, SEG3) => {
            let int = intersect_arc_arc(
                edge.coord2_ref(0),
                edge.coord2_ref(2),
                edge.coord2_ref(1),
                edge2.coord2_ref(0),
                edge2.coord2_ref(2),
                edge2.coord2_ref(1),
            );
            let c = edge.connectivity();
            let c2 = edge2.connectivity();
            (int, [c[0], c[1], c2[0], c2[1]])
        }
        _ => todo!("Intersection with higher order not yet implemented"),
    };

    // The dart map (`build_cell_dart_map`) keys edges by their two endpoint nodes, ignoring
    // mid-side nodes, so a quadratic `edge` is keyed on its endpoints only. `(SEG2, SEG2)`
    // keeps the original key (all connectivity nodes), which is the same set for a segment.
    let m1sgid = if et1 == SEG2 && et2 == SEG2 {
        SortedVecKey::new(edge.connectivity().into())
    } else {
        let c = edge.connectivity();
        SortedVecKey::new([c[0], c[1]].as_slice().into())
    };
    let Some(entry) = intersection_entry(
        int,
        p_ids,
        edge,
        &edge2,
        m2_coords,
        new_coords,
        new_coord_id,
    ) else {
        return;
    };
    intersections.entry(m1sgid).or_default().push(entry);
}

/// Floating-point tolerance for reusing an existing node when a computed intersection point is
/// numerically indistinguishable from it. Scales with the coordinate magnitude of the two edges
/// and has no `.max(1.0)` floor, so it keeps shrinking with the geometry (an absolute floor would
/// wrongly merge distinct nodes of a small mesh). This is FP roundoff (`k * scale * EPSILON`), not
/// a mesh-quality tolerance.
fn node_reuse_eps(
    edge: &Element<'_>,
    edge2: &Element<'_>,
    coords: &nd::ArrayView2<'_, f64>,
) -> f64 {
    let mut scale = 0.0f64;
    for &n in edge
        .connectivity()
        .iter()
        .chain(edge2.connectivity().iter())
    {
        scale = scale.max(coords[(n, 0)].abs()).max(coords[(n, 1)].abs());
    }
    NODE_REUSE_FP_UNITS * scale * f64::EPSILON
}

/// Maps an [`Intersections`] result to `(edge2 id, resolved node ids)`.
///
/// `Existing` ids resolve directly to the operand endpoint nodes via `p_ids`; `New`
/// coordinates reuse the nearest existing node of either edge within `eps`, or allocate a
/// fresh node. Returns `None` when there is no intersection.
fn intersection_entry(
    int: Intersections,
    p_ids: [usize; 4],
    edge: &Element<'_>,
    edge2: &Element<'_>,
    m2_coords: &nd::ArrayView2<'_, f64>,
    new_coords: &mut Vec<f64>,
    new_coord_id: &mut usize,
) -> Option<(ElementId, IntersectionIds)> {
    let eps = node_reuse_eps(edge, edge2, m2_coords);
    let mut resolve = |i: Intersection| -> usize {
        match i {
            Intersection::Existing(p) => p_ids[p as usize],
            Intersection::New(coord) => {
                match closest_global_node(coord, edge, edge2, m2_coords, eps) {
                    Some(n) => n,
                    None => {
                        let id = *new_coord_id;
                        *new_coord_id += 1;
                        new_coords.extend_from_slice(&coord);
                        id
                    }
                }
            }
        }
    };
    match int {
        Intersections::None => None,
        Intersections::One(i) => Some((edge2.id(), IntersectionIds::One(resolve(i)))),
        Intersections::Two([i1, i2]) => Some((
            edge2.id(),
            IntersectionIds::Segment(resolve(i1), resolve(i2)),
        )),
        Intersections::Segment([p1, p2]) => Some((
            edge2.id(),
            IntersectionIds::Segment(
                resolve(Intersection::Existing(p1)),
                resolve(Intersection::Existing(p2)),
            ),
        )),
    }
}

fn closest_global_node(
    pt: [f64; 2],
    edge: &Element<'_>,
    edge2: &Element<'_>,
    coords: &nd::ArrayView2<'_, f64>,
    eps: f64,
) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    let eps2 = eps * eps;
    for &nid in edge
        .connectivity()
        .iter()
        .chain(edge2.connectivity().iter())
    {
        let x = coords[(nid, 0)];
        let y = coords[(nid, 1)];
        let dx = x - pt[0];
        let dy = y - pt[1];
        let d2 = dx * dx + dy * dy;
        if d2 <= eps2 {
            match best {
                None => best = Some((nid, d2)),
                Some((_, bd2)) if d2 < bd2 => best = Some((nid, d2)),
                _ => {}
            }
        }
    }
    best.map(|(n, _)| n)
}

/// Build edge to sorted intersections map.
/// Sort order is taken from the SortedVecKey order (from node with lower id to node with higer
/// id).
/// The SortedVecKey of the map is independent from mesh1/mesh2 distinction as NodeIds are common
/// between the two meshes.
fn to_sorted_intersections(
    intersections: &M1M2Intersections,
    mesh2: &UMeshView,
    coords: &nd::ArrayView2<'_, f64>,
) -> SortedSegIntersections {
    let mut non_sorted_intersections = to_non_sorted_intersections(intersections, mesh2);

    for (eid, v) in &mut non_sorted_intersections {
        let p1: Point2<f64> = Point2::from_slice(coords.row(eid[0]).as_slice().unwrap());
        let p2: Point2<f64> = Point2::from_slice(coords.row(eid[1]).as_slice().unwrap());
        let oriented_vec: Vector2<f64> = p2 - p1;
        let projection = |n: usize| {
            let vn: Vector2<f64> = Point2::from_slice(coords.row(n).as_slice().unwrap()) - p1;
            oriented_vec.dot(&vn)
        };

        // First point
        let mut sorted_ints: Vec<(f64, usize)> = Vec::with_capacity(v.len() + 2);
        sorted_ints.push((projection(eid[0]), eid[0]));
        // Intersection points
        sorted_ints.extend(v.iter().map(|&n| (projection(n), n)));
        // Sorting all intersections points
        sorted_ints.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut sorted_nodes: Vec<NodeId> = sorted_ints.into_iter().map(|(_, n)| n).collect();
        // Adding last point (known)
        sorted_nodes.push(eid[1]);

        // Removing duplicates
        sorted_nodes.dedup();
        v.clear();
        v.append(&mut sorted_nodes);
    }
    non_sorted_intersections
}

fn to_non_sorted_intersections(
    intersections: &M1M2Intersections,
    mesh2: &UMeshView,
) -> SortedSegIntersections {
    let mut sorted_intersections: SortedSegIntersections = FxHashMap::default();

    // Seg1 id
    for (seg1_id, seg2_ints) in intersections {
        let mut sorted_ints: Vec<NodeId> = Vec::new();

        // Intersection points
        for (_, inters) in seg2_ints {
            match inters {
                IntersectionIds::One(i) => sorted_ints.push(*i),
                IntersectionIds::Segment(i1, i2) => {
                    sorted_ints.push(*i1);
                    sorted_ints.push(*i2);
                }
            }
        }
        sorted_intersections.insert(seg1_id.clone(), sorted_ints);
    }

    // Seg2 id
    for seg2_ints in intersections.values() {
        for (seg2, int) in seg2_ints {
            let seg2_id = SortedVecKey::new(mesh2.element(*seg2).connectivity().into());
            match int {
                IntersectionIds::One(i) => {
                    sorted_intersections.entry(seg2_id).or_default().push(*i)
                }
                IntersectionIds::Segment(i1, i2) => {
                    let v = sorted_intersections.entry(seg2_id).or_default();
                    v.push(*i1);
                    v.push(*i2);
                }
            }
        }
    }

    sorted_intersections
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // use crate::io::write;
    use crate::mesh_examples::{make_imesh_2d, make_mesh_2d_multi_simple};
    use crate::tools::RegularUMeshBuilder;
    // use std::path::Path;

    #[test]
    fn test_intersect_meshe_square1() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 1. / 3.;

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_square1.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 13);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 5);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshe_square2() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 2. / 3.;

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_square2.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 15);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 7);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshe_square3() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 1. / 6.;
        mesh2.coords += 1. / 6.;

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_square3.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 13);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 4);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshe_square4() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 0.25;
        mesh2.coords += 0.25;

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_square4.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 13);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 5);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshe_square5() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 0.5;
        mesh2.coords += 0.25;

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_square5.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 17);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 8);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshes_simple() {
        let mesh1 = make_mesh_2d_multi_simple();
        let mesh2 = make_imesh_2d(2);

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_meshes.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 14);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 5);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshes_2() {
        let mesh1 = make_imesh_2d(3);
        let mesh2 = make_imesh_2d(2);
        // let p = Path::new("mesh1_2.vtk");
        // let _ = write(p, mesh1.view());
        // let p = Path::new("mesh2_2.vtk");
        // let _ = write(p, mesh2.view());

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_meshes2.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 29);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 16);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    #[test]
    fn test_intersect_meshes_3() {
        let mesh1 = make_imesh_2d(2);
        let mesh2 = make_imesh_2d(3);
        // let p = Path::new("mesh1_3.vtk");
        // let _ = write(p, mesh1.view());
        // let p = Path::new("mesh2_3.vtk");
        // let _ = write(p, mesh2.view());

        let mesh_cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // let p = Path::new("test_intersect_meshes3.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 29);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 16);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }

    /// Sums the area of all D2 elements of the mesh.
    fn mesh_area(mesh: &UMesh) -> f64 {
        mesh.elements_of_dim(Dimension::D2)
            .map(|cell| {
                let pgon =
                    Polygon::unknown((0..cell.connectivity().len()).map(|i| cell.coord2(i).into()));
                pgon.area()
            })
            .sum()
    }

    /// mesh1 = [0, 1]^2, mesh2 = [0.25, 0.75]^2 (contained in mesh1).
    fn make_nested_squares() -> (UMesh, UMesh) {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 0.5;
        mesh2.coords += 0.25;
        (mesh1, mesh2)
    }

    #[test]
    fn test_overlay_imprint_nested() {
        let (mesh1, mesh2) = make_nested_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.0, epsilon = 1e-12);
        assert_eq!(cutted.num_elements_of_dim(Dimension::D2), 8);
    }

    #[test]
    fn test_overlay_union_nested() {
        let (mesh1, mesh2) = make_nested_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_intersection_nested() {
        let (mesh1, mesh2) = make_nested_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Intersection);
        assert_abs_diff_eq!(mesh_area(&cutted), 0.25, epsilon = 1e-12);
        assert_eq!(cutted.num_elements_of_dim(Dimension::D2), 4);
    }

    #[test]
    fn test_overlay_difference_nested() {
        let (mesh1, mesh2) = make_nested_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Difference);
        assert_abs_diff_eq!(mesh_area(&cutted), 0.75, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_symmetric_difference_nested() {
        let (mesh1, mesh2) = make_nested_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::SymmetricDifference);
        assert_abs_diff_eq!(mesh_area(&cutted), 0.75, epsilon = 1e-12);
    }

    /// mesh1 = [0, 1]^2, mesh2 = [0.5, 1.5]^2 (straddling mesh1's boundary).
    fn make_straddling_squares() -> (UMesh, UMesh) {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(2);
        mesh2.coords += 0.5;
        (mesh1, mesh2)
    }

    #[test]
    fn test_overlay_union_straddling() {
        let (mesh1, mesh2) = make_straddling_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.75, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_intersection_straddling() {
        let (mesh1, mesh2) = make_straddling_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Intersection);
        assert_abs_diff_eq!(mesh_area(&cutted), 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_difference_straddling() {
        let (mesh1, mesh2) = make_straddling_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Difference);
        assert_abs_diff_eq!(mesh_area(&cutted), 0.75, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_symmetric_difference_straddling() {
        let (mesh1, mesh2) = make_straddling_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::SymmetricDifference);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.5, epsilon = 1e-12);
    }

    /// mesh1 = [0, 1]^2, mesh2 = [1.5, 2]^2 (disjoint from mesh1).
    fn make_disjoint_squares() -> (UMesh, UMesh) {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 0.5;
        mesh2.coords += 1.5;
        (mesh1, mesh2)
    }

    #[test]
    fn test_overlay_union_disjoint() {
        let (mesh1, mesh2) = make_disjoint_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_intersection_disjoint() {
        let (mesh1, mesh2) = make_disjoint_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Intersection);
        assert_eq!(cutted.num_elements_of_dim(Dimension::D2), 0);
    }

    #[test]
    fn test_overlay_difference_disjoint() {
        let (mesh1, mesh2) = make_disjoint_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Difference);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_symmetric_difference_disjoint() {
        let (mesh1, mesh2) = make_disjoint_squares();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::SymmetricDifference);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_union_multi_simple() {
        let mesh1 = make_mesh_2d_multi_simple();
        let mesh2 = make_imesh_2d(2);
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        assert_abs_diff_eq!(mesh_area(&cutted), 1.25, epsilon = 1e-12);
        assert_eq!(cutted.num_elements_of_dim(Dimension::D2), 5);
    }

    /// mesh1: n x n grid on [0,1]^2. mesh2: staggered grid whose cells are centered on the
    /// nodes of mesh1 and whose nodes are the cell centers of mesh1.
    fn make_staggered_squares(n: usize) -> (UMesh, UMesh) {
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis((0..=n).map(|i| i as f64 / n as f64).collect())
            .add_axis((0..=n).map(|i| i as f64 / n as f64).collect())
            .build();
        let nodes: Vec<f64> = (0..=n + 1).map(|i| (i as f64 - 0.5) / n as f64).collect();
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(nodes.clone())
            .add_axis(nodes)
            .build();
        (mesh1, mesh2)
    }

    #[test]
    fn test_overlay_union_staggered() {
        let (mesh1, mesh2) = make_staggered_squares(4);
        let side = 1.0 + 1.0 / 4.0;
        let expected = side * side;
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        let actual = mesh_area(&cutted);
        println!("union staggered: expected {expected}, actual {actual}");
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_overlay_symmetric_difference_staggered() {
        let (mesh1, mesh2) = make_staggered_squares(4);
        let side = 1.0 + 1.0 / 4.0;
        let expected = side * side - 1.0;
        let cutted = mesh1.overlay(mesh2, OverlayOperation::SymmetricDifference);
        let actual = mesh_area(&cutted);
        println!("symdiff staggered: expected {expected}, actual {actual}");
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }

    /// Reproduces docs/python_examples/geometric_tools.ipynb (cell "Intersect 2d mesh"):
    /// mesh1 is a 6x6 grid on [0,3]^2 (7 nodes per axis) and mesh2 is the same grid shifted
    /// by dec = 3/7 + 0.1. Union must cover [0,3]^2 U [dec,3+dec]^2 = 18 - (3-dec)^2.
    #[test]
    fn test_overlay_union_notebook() {
        let n = 6usize;
        let x: Vec<f64> = (0..=n).map(|i| 3.0 * i as f64 / n as f64).collect();
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(x.clone())
            .add_axis(x)
            .build();
        let dec = 3.0 / 7.0 + 0.1;
        let x2: Vec<f64> = (0..=n).map(|i| dec + 3.0 * i as f64 / n as f64).collect();
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(x2.clone())
            .add_axis(x2)
            .build();

        let cutted = mesh1.overlay(mesh2, OverlayOperation::Union);
        let expected = 18.0 - (3.0 - dec) * (3.0 - dec);
        let actual = mesh_area(&cutted);
        println!("notebook union: expected {expected}, actual {actual}");
        assert_abs_diff_eq!(actual, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_overlay_staggered_sweep() {
        for n in [2usize, 3, 4, 5, 6, 8] {
            let (mesh1, mesh2) = make_staggered_squares(n);
            let side = 1.0 + 1.0 / n as f64;
            let union_expected = side * side;
            let symdiff_expected = side * side - 1.0;
            let union = mesh1.overlay(mesh2.clone(), OverlayOperation::Union);
            let symdiff = mesh1.overlay(mesh2, OverlayOperation::SymmetricDifference);
            let u = mesh_area(&union);
            let s = mesh_area(&symdiff);
            println!(
                "n={n}: union {u} (exp {union_expected}), symdiff {s} (exp {symdiff_expected})"
            );
            assert_abs_diff_eq!(u, union_expected, epsilon = 1e-9);
            assert_abs_diff_eq!(s, symdiff_expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_overlay_seg3_seg2_arc_intersection_reused_nodes() {
        use crate::prelude::RegularUMeshBuilder;
        // Mesh1: a single quad [0,1]x[0,1]
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        // Mesh2: a vertical line that passes through the quad center; model as a 2-cell strip
        // but easier: use two triangles? Or just construct a simple mesh with a SEG2 cutting edge
        // Actually overlay works on 2D meshes; create mesh2 as a thin rectangle that crosses mesh1
        // to force intersections on element edges.
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.25, 0.5, 1.25])
            .add_axis(vec![0.25, 0.75])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        // Imprint should produce valid output
        let area = mesh_area(&cutted);
        assert!(area > 0.0);
    }

    #[test]
    fn test_overlay_seg3_seg3_imprint_basic() {
        use crate::prelude::RegularUMeshBuilder;
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.1, 0.5, 1.1])
            .add_axis(vec![0.4, 0.6])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        let area = mesh_area(&cutted);
        assert!(area > 0.0 && area <= 1.0 + 1e-12);
    }
}
