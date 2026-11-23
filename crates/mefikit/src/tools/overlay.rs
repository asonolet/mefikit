//! Boolean-like overlay operations on 2D meshes.
//!
//! In this context, overlay operations can be separated in the following cases:
//! - 2d + 1d: [`Overlayable::intersect_1d`] cuts a 2D mesh with a 1D mesh of `SEG2` or
//!   `PLINE` edges, producing a 2D mesh conformized with the 1D mesh.
//! - 2d + 2d: `Overlayable::overlay` with one of the [`OverlayOperation`]s:
//!   - `Imprint`: refine `mesh1` with the edges of `mesh2` while keeping `mesh1`'s domain.
//!     Quadratic cells are supported in both meshes (arc/arc intersections).
//!   - `Union`: keep the domain covered by at least one of the two meshes.
//!   - `Intersection`: keep the domain covered by both meshes.
//!   - `Difference`: keep the domain of `mesh1` not covered by `mesh2`.
//!   - `SymmetricDifference`: keep the domain covered by exactly one of the two meshes.
//!
//! The four boolean operations are only implemented for first-order meshes: their
//! inside/outside classification still assumes straight-edged cut pieces. Calling them
//! with a quadratic mesh is rejected rather than returning a wrong result.
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
//! [`stitch`], and conformizing one mesh with itself by [`conformize`]; see the `stitch`
//! submodule. Mesh conformance diagnostics live in [`crate::tools::conformance`].

pub(crate) mod stitch;
pub(crate) mod surface;

pub use stitch::{StitchError, conformize, stitch};
pub use surface::{SurfaceOverlay, SurfaceOverlayError, overlay_surfaces};

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use ndarray as nd;

use crate::element_traits::cut::{
    CutEdgeSource, CutPiece, IntersectionIds, M1M2Intersections, NodeId, SortedSegIntersections,
};
use crate::element_traits::{Cutable, ElementTopo, SortedVecKey};
use crate::geometry::{
    ArcEdge, Intersection, Intersections, PointId, Polygon, Segment, intersect_arc_arc,
    intersect_arc_seg,
};
use crate::mesh::{Dimension, Element, ElementId, ElementLike, ElementType, UMesh, UMeshView};
use crate::prelude::ElementGeo;
use crate::tools::duplicates_from;
use crate::tools::spatial_index::SpIdx2;
use crate::tools::{Descendable, spatial_index::SpatiallyIndexable};

/// Node coincidence tolerance used when merging two 2D meshes onto shared coordinates. Kept
/// tight because the 2D overlay assumes the inputs already share a coordinate space.
const WELD_EPS: f64 = 1e-12;

/// Number of floating-point roundoff units (`scale * f64::EPSILON`) allowed when deciding
/// whether a computed intersection point may reuse an existing node.
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
#[non_exhaustive]
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

/// Boolean overlay of two 2D meshes, and intersection of a 2D mesh with a 1D edge mesh.
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
    /// - Quadratic cells (`TRI6`, `TRI7`, `QUAD8`, `QUAD9`, `QPOLY`) are supported in both
    ///   meshes for [`OverlayOperation::Imprint`]; their edges are circular arcs through
    ///   their mid-side node and cut pieces are emitted as `QPOLY`. The four other
    ///   operations are only implemented for first-order meshes.
    fn overlay(&self, mesh2: UMesh, operation: OverlayOperation) -> UMesh;

    /// Intersects this 2D mesh with a 1D mesh of edges (`SEG2` or `PLINE`).
    ///
    /// Refines the cells of `self` with the edges of `mesh2` and keeps every piece: this is
    /// the [`OverlayOperation::Imprint`] operation for meshes of different dimensions.
    /// `PLINE` cells are split into their consecutive `SEG2` edges before the cut, so both
    /// input forms are handled by the same 2d/1d machinery.
    ///
    /// # Assumptions
    /// - `mesh2` contains only `SEG2` or `PLINE` elements, and those elements do not
    ///   intersect one another.
    /// - Both meshes are valid, planar and share the same coordinate space.
    /// - Quadratic cells are only supported in `self`, as for
    ///   [`OverlayOperation::Imprint`].
    ///
    /// # Panics
    /// Panics if `mesh2` contains an element type other than `SEG2` or `PLINE`.
    fn intersect_1d(&self, mesh2: UMesh) -> UMesh;

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
        assert!(
            operation == OverlayOperation::Imprint
                || (!mesh_has_quadratic_2d(self) && !mesh_has_quadratic_2d(&mesh2)),
            "boolean overlay operations (UNION, INTERSECTION, DIFFERENCE, \
             SYMMETRIC_DIFFERENCE) are not supported for quadratic meshes; only IMPRINT is"
        );
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

    fn intersect_1d(&self, mesh2: UMesh) -> UMesh {
        intersect_2d1d(self, as_cutting_edges(mesh2))
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

/// Converts a 1D cutter mesh into a `SEG2`-only edge mesh.
///
/// `PLINE` cells are split into their consecutive `SEG2` edges so the cutter reaches
/// `compute_overlay` in the only 1D form it understands. The coordinates are copied
/// unchanged, so node ids stay valid.
fn as_cutting_edges(mesh: UMesh) -> UMesh {
    let mut edges: Vec<[usize; 2]> = Vec::new();
    // Coincident edges would each create their own intersection nodes, splitting a
    // cell twice along the same line and losing area. The 1D elements are assumed not
    // to cross, so an exact duplicate can only be an input artefact: keep one copy.
    let mut seen: FxHashSet<(usize, usize)> = FxHashSet::default();
    let mut push_edge = |a: usize, b: usize| {
        if seen.insert((a.min(b), a.max(b))) {
            edges.push([a, b]);
        }
    };
    for cell in mesh.elements() {
        match cell.element_type() {
            ElementType::SEG2 => {
                let co = cell.connectivity();
                push_edge(co[0], co[1]);
            }
            ElementType::PLINE => {
                for w in cell.connectivity().windows(2) {
                    push_edge(w[0], w[1]);
                }
            }
            et => panic!("intersect_1d expects a 1D mesh of SEG2 or PLINE elements, found {et:?}"),
        }
    }
    let n = edges.len();
    let mut out = UMesh::new(mesh.coords().to_shared());
    if n > 0 {
        let flat: Vec<usize> = edges.into_iter().flatten().collect();
        let conn = nd::Array2::from_shape_vec((n, 2), flat).expect("two nodes per SEG2 edge");
        out.add_regular_block(ElementType::SEG2, conn.into_shared(), None);
    }
    out
}

/// Refines the 2D cells of `mesh1` with the `SEG2` edges of `mesh2`, keeping all pieces.
///
/// `mesh2` must already be a `SEG2`-only mesh (see [`as_cutting_edges`]); this is the
/// shared core of [`Overlayable::intersect_1d`] and [`intersect_2d2d`] after its cutter
/// has been descended from 2D to 1D.
fn intersect_2d1d(mesh1: &UMesh, mesh2: UMesh) -> UMesh {
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

    let seg_intersections = to_sorted_intersections(
        &intersections,
        subject_edges,
        cutting_edges,
        &cutted_mesh.coords(),
    );

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
    let mut mid_store = MidStore::default();
    for cell in subject.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = cutting_bvh.in_bounds(bmin, bmax);
        let boundary_edges = cell.boundary_edges();
        let reconstructed = cell.cut_with_intersections(
            seg_intersections,
            cutting_edges,
            out.coords(),
            &candidates,
        );

        let mut piece_ids: Vec<ElementId> = Vec::new();
        // If the cell was cut, I add new polys from the cut
        if let Some(pieces) = reconstructed {
            for piece in pieces {
                let id = emit_piece(out, &cell, &boundary_edges, &piece, &mut mid_store);
                piece_ids.push(id);
            }
        } else {
            let id = out.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
            piece_ids.push(id);
        }
        parents.push((cell.id(), piece_ids));
    }
}

#[inline]
fn point2(coords: &nd::ArrayView2<'_, f64>, node: usize) -> [f64; 2] {
    [coords[(node, 0)], coords[(node, 1)]]
}

#[inline]
fn edge_key(conn: &[usize]) -> SortedVecKey {
    SortedVecKey::new(conn.iter().copied().collect())
}

#[inline]
fn chord_midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

fn point_on_segment2(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let len = ab[0].hypot(ab[1]);
    if len == 0.0 {
        return ap[0].hypot(ap[1]) == 0.0;
    }
    let scale = len.max(ap[0].hypot(ap[1]));
    let eps = 64.0 * scale * f64::EPSILON;
    let cross = ab[0] * ap[1] - ab[1] * ap[0];
    if cross.abs() > eps * len {
        return false;
    }
    let dot = ab[0] * ap[0] + ab[1] * ap[1];
    dot >= -eps * len && dot <= len * len + eps * len
}

/// Source curve of a synthesized mid-side node. Keeping the source in the key prevents two
/// distinct arcs with the same endpoints (e.g. a lens) from sharing a mid by mistake.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum MidSource {
    Boundary(SortedVecKey),
    Cutter(ElementId),
}

#[derive(Default)]
struct MidStore {
    by_edge: FxHashMap<(MidSource, usize, usize), usize>,
}

impl MidStore {
    fn get_or_create(
        &mut self,
        out: &mut UMesh,
        source: MidSource,
        a: usize,
        b: usize,
        coord: [f64; 2],
    ) -> usize {
        let key = (source, a.min(b), a.max(b));
        if let Some(&id) = self.by_edge.get(&key) {
            return id;
        }
        let id = out.coords().nrows();
        out.append_coord(nd::arr1(&coord).view())
            .expect("appending a single 2D coordinate");
        self.by_edge.insert(key, id);
        id
    }
}

fn boundary_arc(out: &UMesh, edge: &[usize]) -> Option<ArcEdge> {
    if edge.len() != 3 {
        return None;
    }
    let coords = out.coords();
    let arc = ArcEdge::new(
        point2(&coords, edge[0]),
        point2(&coords, edge[2]),
        point2(&coords, edge[1]),
    );
    arc.is_curved().then_some(arc)
}

/// True when the cut piece is exactly the original cell boundary, in the same order, using
/// only boundary edges. Such a piece must be copied verbatim so the original element type is
/// preserved.
fn is_verbatim(
    cell: &Element<'_>,
    boundary_edges: &[SmallVec<[usize; 3]>],
    piece: &CutPiece,
) -> bool {
    let n = cell.num_vertices();
    if piece.ring.len() != n {
        return false;
    }
    let mut parent: Vec<usize> = cell.connectivity()[..n].to_vec();
    let mut ring = piece.ring.clone();
    parent.sort_unstable();
    ring.sort_unstable();
    if parent != ring {
        return false;
    }
    for (i, source) in piece.edges.iter().enumerate() {
        let a = piece.ring[i];
        let b = piece.ring[(i + 1) % piece.ring.len()];
        let CutEdgeSource::Boundary(j) = *source else {
            return false;
        };
        let Some(edge) = boundary_edges.get(j) else {
            return false;
        };
        if !((edge[0] == a && edge[1] == b) || (edge[0] == b && edge[1] == a)) {
            return false;
        }
    }
    true
}

/// Appends one cut piece to `out`, restoring the curvature of the original boundary edges.
///
/// A touched piece of a quadratic cell is always emitted as [`ElementType::QPOLY`]; pieces of
/// linear cells stay [`ElementType::PGON`]. Untouched pieces are copied verbatim.
fn emit_piece(
    out: &mut UMesh,
    cell: &Element<'_>,
    boundary_edges: &[SmallVec<[usize; 3]>],
    piece: &CutPiece,
    mid_store: &mut MidStore,
) -> ElementId {
    if is_verbatim(cell, boundary_edges, piece) {
        return out.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
    }

    let quadratic_cell = boundary_edges.iter().any(|edge| edge.len() == 3);
    if !quadratic_cell {
        return out.add_element(ElementType::PGON, &piece.ring, Some(*cell.family));
    }

    let n = piece.ring.len();
    let mut mids: Vec<usize> = Vec::with_capacity(n);
    for i in 0..n {
        let a = piece.ring[i];
        let b = piece.ring[(i + 1) % n];
        let mid = match piece.edges[i] {
            CutEdgeSource::Boundary(j) => {
                let edge = &boundary_edges[j];
                let coords = out.coords();
                let pa = point2(&coords, a);
                let pb = point2(&coords, b);
                if let Some(arc) = boundary_arc(out, edge) {
                    let sa = arc.param(&pa);
                    let sb = arc.param(&pb);
                    let sm = arc.mid_param();
                    if let (Some(sa), Some(sb), Some(sm)) = (sa, sb, sm) {
                        let (lo, hi) = if sa <= sb { (sa, sb) } else { (sb, sa) };
                        let margin = 1e-12;
                        if sm > lo + margin && sm < hi - margin {
                            edge[2]
                        } else {
                            let mid = arc
                                .sub_midpoint(&pa, &pb)
                                .expect("non-degenerate arc has a sub-arc midpoint");
                            mid_store.get_or_create(
                                out,
                                MidSource::Boundary(edge_key(edge)),
                                a,
                                b,
                                mid,
                            )
                        }
                    } else {
                        mid_store.get_or_create(
                            out,
                            MidSource::Boundary(edge_key(edge)),
                            a,
                            b,
                            chord_midpoint(pa, pb),
                        )
                    }
                } else if edge.len() == 3 && point_on_segment2(pa, pb, point2(&coords, edge[2])) {
                    edge[2]
                } else {
                    mid_store.get_or_create(
                        out,
                        MidSource::Boundary(edge_key(edge)),
                        a,
                        b,
                        chord_midpoint(pa, pb),
                    )
                }
            }
            CutEdgeSource::Cutter(eid) => {
                let coords = out.coords();
                let mid = chord_midpoint(point2(&coords, a), point2(&coords, b));
                mid_store.get_or_create(out, MidSource::Cutter(eid), a, b, mid)
            }
        };
        mids.push(mid);
    }

    let mut conn = piece.ring.clone();
    conn.extend(mids);
    out.add_element(ElementType::QPOLY, &conn, Some(*cell.family))
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
    let mut mid_store = MidStore::default();

    for cell in subject.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = cutting_bvh.in_bounds(bmin, bmax);
        let boundary_edges = cell.boundary_edges();
        let reconstructed = cell.cut_with_intersections(
            seg_intersections,
            cutting_edges,
            out.coords(),
            &candidates,
        );

        // If the cell was cut, I add the kept new polys from the cut
        if let Some(pieces) = reconstructed {
            for piece in pieces {
                let inside = pgon_inside(&piece.ring, &out.coords(), cutter, &cutter_bvh);
                if keep(inside) {
                    emit_piece(out, &cell, &boundary_edges, &piece, &mut mid_store);
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
    // Untouched quadratic cells still carry their mid-side nodes in the connectivity;
    // only the corners bound the cell.
    let n = if is_quadratic_2d(cell.element_type()) {
        cell.num_vertices()
    } else {
        cell.connectivity().len()
    };
    let points: Vec<[f64; 2]> = (0..n).map(|i| *cell.coord2_ref(i)).collect();
    point_inside(&points, cutter, cutter_bvh)
}

/// Returns `true` if the interior of the polygon lies inside any cell of `cutter_cells`.
///
/// Pieces are produced by cutting a cell along the cutter edges, so a piece is either fully
/// inside or fully outside the cutter. A strict interior point of the piece is therefore a valid
/// witness: the piece is inside the cutter iff its interior point is. The centroid of a
/// non-convex piece (e.g. an L-shape) can fall outside its own interior, so it is not a valid
/// witness.
/// Whether a mesh contains any 2D cell with curved (circular-arc) edges.
fn mesh_has_quadratic_2d(mesh: &UMesh) -> bool {
    mesh.blocks().any(|(et, _)| is_quadratic_2d(*et))
}

/// Whether a 2D cell type has curved (circular-arc) edges.
fn is_quadratic_2d(et: ElementType) -> bool {
    matches!(
        et,
        ElementType::TRI6
            | ElementType::TRI7
            | ElementType::QUAD8
            | ElementType::QUAD9
            | ElementType::QPOLY
    )
}

fn point_inside(points: &[[f64; 2]], cutter: &UMesh, cutter_bvh: &SpIdx2) -> bool {
    let Some(interior) = Polygon::unknown(points.iter().copied()).strict_interior_point() else {
        return false;
    };
    let candidates = cutter_bvh.intersects(interior);
    candidates
        .iter()
        .map(|eid| cutter.element(eid))
        .any(|cell| {
            if is_quadratic_2d(cell.element_type()) {
                // A quadratic cell stores its mid-side nodes after the corners: treating
                // them as polygon vertices would give a bogus "inside" answer.
                cell.is_point_inside(&[interior[0], interior[1]])
            } else {
                cell_points(cell).contains_stable(&interior)
            }
        })
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
    coords: &nd::ArrayView2<'_, f64>,
) {
    use crate::mesh::ElementType::*;
    let c1 = edge.connectivity();
    let c2 = edge2.connectivity();
    let (int, p_ids): (Intersections, [usize; 4]) =
        match (edge.element_type(), edge2.element_type()) {
            (SEG2, SEG2) => (
                Segment::new(edge.coord2(0).into(), edge.coord2(1).into())
                    .intersect(Segment::new(edge2.coord2(0).into(), edge2.coord2(1).into())),
                [c1[0], c1[1], c2[0], c2[1]],
            ),
            (SEG3, SEG2) => (
                intersect_arc_seg(
                    edge.coord2_ref(0),
                    edge.coord2_ref(2),
                    edge.coord2_ref(1),
                    edge2.coord2_ref(0),
                    edge2.coord2_ref(1),
                ),
                [c1[0], c1[1], c2[0], c2[1]],
            ),
            (SEG2, SEG3) => {
                let p0 = edge.coord2_ref(0);
                let p1 = edge.coord2_ref(1);
                let mid = [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])];
                (
                    intersect_arc_arc(
                        p0,
                        &mid,
                        p1,
                        edge2.coord2_ref(0),
                        edge2.coord2_ref(2),
                        edge2.coord2_ref(1),
                    ),
                    [c1[0], c1[1], c2[0], c2[1]],
                )
            }
            (SEG3, SEG3) => (
                intersect_arc_arc(
                    edge.coord2_ref(0),
                    edge.coord2_ref(2),
                    edge.coord2_ref(1),
                    edge2.coord2_ref(0),
                    edge2.coord2_ref(2),
                    edge2.coord2_ref(1),
                ),
                [c1[0], c1[1], c2[0], c2[1]],
            ),
            (a, b) => panic!("unsupported edge intersection between {a:?} and {b:?}"),
        };

    let Some((eid2, ids)) =
        intersection_entry(int, p_ids, edge, &edge2, coords, new_coords, new_coord_id)
    else {
        return;
    };
    let m1sgid = SortedVecKey::new(edge.connectivity().into());
    intersections.entry(m1sgid).or_default().push((eid2, ids));
}

#[inline]
fn p_node(id: PointId, nodes: [usize; 4]) -> usize {
    match id {
        PointId::P1 => nodes[0],
        PointId::P2 => nodes[1],
        PointId::P3 => nodes[2],
        PointId::P4 => nodes[3],
    }
}

fn node_reuse_eps(
    edge: &Element<'_>,
    edge2: &Element<'_>,
    coords: &nd::ArrayView2<'_, f64>,
) -> f64 {
    let mut nodes = edge
        .connectivity()
        .iter()
        .chain(edge2.connectivity().iter());
    let Some(&first) = nodes.next() else {
        return 0.0;
    };
    let origin = point2(coords, first);
    let mut scale = 0.0f64;
    for &n in edge
        .connectivity()
        .iter()
        .chain(edge2.connectivity().iter())
    {
        let p = point2(coords, n);
        scale = scale.max((p[0] - origin[0]).hypot(p[1] - origin[1]));
    }
    NODE_REUSE_FP_UNITS * scale * f64::EPSILON
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
        let p = point2(coords, nid);
        let dx = p[0] - pt[0];
        let dy = p[1] - pt[1];
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

/// Maps an [`Intersections`] result to `(edge2 id, resolved node ids)`.
fn intersection_entry(
    int: Intersections,
    p_ids: [usize; 4],
    edge: &Element<'_>,
    edge2: &Element<'_>,
    coords: &nd::ArrayView2<'_, f64>,
    new_coords: &mut Vec<f64>,
    new_coord_id: &mut usize,
) -> Option<(ElementId, IntersectionIds)> {
    let eps = node_reuse_eps(edge, edge2, coords);
    let mut resolve = |i: Intersection| -> usize {
        match i {
            Intersection::Existing(p) => p_node(p, p_ids),
            Intersection::New(coord) => {
                match closest_global_node(coord, edge, edge2, coords, eps) {
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
        Intersections::One(i) => {
            let n = resolve(i);
            Some((edge2.id(), IntersectionIds::One(n)))
        }
        Intersections::Two([i1, i2]) => {
            let n1 = resolve(i1);
            let n2 = resolve(i2);
            Some((
                edge2.id(),
                if n1 == n2 {
                    IntersectionIds::One(n1)
                } else {
                    IntersectionIds::Segment(n1, n2)
                },
            ))
        }
        Intersections::Segment([p1, p2]) => {
            let n1 = resolve(Intersection::Existing(p1));
            let n2 = resolve(Intersection::Existing(p2));
            Some((
                edge2.id(),
                if n1 == n2 {
                    IntersectionIds::One(n1)
                } else {
                    IntersectionIds::Segment(n1, n2)
                },
            ))
        }
    }
}

#[derive(Clone, Copy)]
struct EdgeInfo {
    p0: usize,
    p1: usize,
    mid: Option<usize>,
}

fn collect_edge_info(meshes: &[&UMesh]) -> FxHashMap<SortedVecKey, EdgeInfo> {
    let mut infos = FxHashMap::default();
    for mesh in meshes {
        for edge in mesh.elements() {
            let conn = edge.connectivity();
            let info = match edge.element_type() {
                ElementType::SEG2 => EdgeInfo {
                    p0: conn[0],
                    p1: conn[1],
                    mid: None,
                },
                ElementType::SEG3 => EdgeInfo {
                    p0: conn[0],
                    p1: conn[1],
                    mid: Some(conn[2]),
                },
                _ => continue,
            };
            infos.entry(edge_key(conn)).or_insert(info);
        }
    }
    infos
}

/// Build edge to sorted intersections map.
///
/// Intersections are ordered along the edge itself: by circular-arc parameter for `SEG3` and by
/// chord projection for straight edges.
fn to_sorted_intersections(
    intersections: &M1M2Intersections,
    subject_edges: &UMesh,
    cutting_edges: &UMesh,
    coords: &nd::ArrayView2<'_, f64>,
) -> SortedSegIntersections {
    let mut non_sorted_intersections =
        to_non_sorted_intersections(intersections, &cutting_edges.view());
    let edge_infos = collect_edge_info(&[subject_edges, cutting_edges]);

    for (eid, v) in &mut non_sorted_intersections {
        let info = edge_infos.get(eid).copied().unwrap_or(EdgeInfo {
            p0: eid[0],
            p1: eid[1],
            mid: None,
        });
        let p0 = point2(coords, info.p0);
        let p1 = point2(coords, info.p1);
        let arc = info.mid.map(|mid| {
            let mid = point2(coords, mid);
            ArcEdge::new(p0, mid, p1)
        });
        let arc = arc.filter(|arc| arc.is_curved());
        let chord = [p1[0] - p0[0], p1[1] - p0[1]];

        let param = |n: usize| {
            let p = point2(coords, n);
            if let Some(arc) = arc {
                arc.param(&p).unwrap_or(0.0).clamp(0.0, 1.0)
            } else {
                (p[0] - p0[0]) * chord[0] + (p[1] - p0[1]) * chord[1]
            }
        };

        let mut sorted_ints: Vec<(f64, usize)> = Vec::with_capacity(v.len() + 2);
        sorted_ints.push((param(info.p0), info.p0));
        sorted_ints.push((param(info.p1), info.p1));
        sorted_ints.extend(v.iter().map(|&n| (param(n), n)));
        sorted_ints.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        let mut sorted_nodes: Vec<NodeId> = sorted_ints.into_iter().map(|(_, n)| n).collect();

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

    fn exact_cell_area(cell: Element<'_>) -> f64 {
        use ElementType::*;
        match cell.element_type() {
            TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY => {
                crate::geometry::quadratic_polygon_area2(&cell.to_quadratic_polygon2()).abs()
            }
            _ => Polygon::unknown(cell.vertices2()).area(),
        }
    }

    fn exact_mesh_area(mesh: &UMesh) -> f64 {
        mesh.elements_of_dim(Dimension::D2)
            .map(exact_cell_area)
            .sum()
    }

    /// A QUAD8 cell over `[0, 1]^2` whose four edges bulge outward by `bulge`.
    fn quad8_square(bulge: f64) -> UMesh {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -bulge],
            [1.0 + bulge, 0.5],
            [0.5, 1.0 + bulge],
            [-bulge, 0.5],
        ]);
        let mut mesh = UMesh::new(coords.into_shared());
        mesh.add_regular_block(
            ElementType::QUAD8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh
    }

    #[test]
    fn test_overlay_quad8_imprint_preserves_curved_area() {
        let mesh1 = quad8_square(0.15);
        let before = exact_mesh_area(&mesh1);
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.25, 0.5, 1.25])
            .add_axis(vec![0.25, 0.75])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert!(
            cutted
                .elements_of_dim(Dimension::D2)
                .any(|c| c.element_type() == ElementType::QPOLY),
            "cut quadratic pieces must be QPOLY"
        );
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
    }

    #[test]
    fn test_overlay_two_intersections_on_one_arc() {
        // A single QUAD8 whose bottom edge bulges downwards; every other edge is straight.
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.2],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::QUAD8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);

        // A single horizontal segment crossing the bulged arc twice.
        let coords2 = nd::arr2(&[[-0.5, -0.05], [1.5, -0.05]]);
        let mut mesh2 = UMesh::new(coords2.into_shared());
        mesh2.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        let cutted = mesh1.intersect_1d(mesh2);

        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        let mut lens = false;
        for cell in cutted.elements_of_dim(Dimension::D2) {
            if cell.element_type() == ElementType::QPOLY && cell.connectivity().len() == 4 {
                let n = cell.num_vertices();
                assert_eq!(n, 2, "a two-edge lens has two vertices and two mids");
                lens = true;
            }
        }
        assert!(lens, "expected a two-vertex QPOLY lens piece");
    }

    #[test]
    fn test_overlay_qpoly_input() {
        let mesh1 = quad8_square(0.15);
        let before = exact_mesh_area(&mesh1);
        let qpoly = crate::tools::polyze(&mesh1.view());
        assert!(
            qpoly
                .elements()
                .any(|c| c.element_type() == ElementType::QPOLY)
        );
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.25, 0.5, 1.25])
            .add_axis(vec![0.25, 0.75])
            .build();
        let cutted = qpoly.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        assert!(
            cutted
                .elements_of_dim(Dimension::D2)
                .any(|c| c.element_type() == ElementType::QPOLY)
        );
    }

    #[test]
    fn test_overlay_curved_tri6_imprint() {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, -0.1],
            [0.65, 0.65],
            [-0.1, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::TRI6,
            nd::arr2(&[[0, 1, 2, 3, 4, 5]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);
        let coords2 = nd::arr2(&[[0.3, -1.0], [0.3, 2.0]]);
        let mut mesh2 = UMesh::new(coords2.into_shared());
        mesh2.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        let cutted = mesh1.intersect_1d(mesh2);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        assert!(
            cutted
                .elements_of_dim(Dimension::D2)
                .any(|c| c.element_type() == ElementType::QPOLY)
        );
    }

    #[test]
    fn test_overlay_tri6_imprint() {
        // Straight-mid TRI6: the cut pieces must still be QPOLY.
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::TRI6,
            nd::arr2(&[[0, 1, 2, 3, 4, 5]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![0.25, 0.75])
            .add_axis(vec![-0.5, 0.5])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        assert!(
            cutted
                .elements_of_dim(Dimension::D2)
                .all(|c| c.element_type() == ElementType::QPOLY)
        );
    }

    /// Two QUAD8 cells sharing a curved vertical edge, with shared corner and mid nodes.
    fn two_quad8_squares(bulge: f64) -> UMesh {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -bulge],
            [1.0 + bulge, 0.5],
            [0.5, 1.0 + bulge],
            [-bulge, 0.5],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.5, -bulge],
            [2.0 + bulge, 0.5],
            [1.5, 1.0 + bulge],
        ]);
        let mut mesh = UMesh::new(coords.into_shared());
        mesh.add_regular_block(
            ElementType::QUAD8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 10, 11, 12, 5]]).to_shared(),
            None,
        );
        mesh
    }

    #[test]
    fn test_overlay_shared_curved_edge_mid_is_shared() {
        let mesh1 = two_quad8_squares(0.15);
        let before = exact_mesh_area(&mesh1);
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![0.4, 0.6])
            .add_axis(vec![-0.5, 0.5, 1.5])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);

        // Each synthesized mid node id must be unique to one coordinate: shared sub-arcs
        // must reuse one node id.
        let coords = cutted.coords();
        let mut by_pos: FxHashMap<(u64, u64), usize> = FxHashMap::default();
        for cell in cutted.elements_of_dim(Dimension::D2) {
            if cell.element_type() != ElementType::QPOLY {
                continue;
            }
            let conn = cell.connectivity();
            let n = conn.len() / 2;
            for &mid in &conn[n..] {
                let key = (coords[(mid, 0)].to_bits(), coords[(mid, 1)].to_bits());
                if let Some(&prev) = by_pos.get(&key) {
                    assert_eq!(prev, mid, "two mid nodes share one coordinate");
                }
                by_pos.insert(key, mid);
            }
        }
    }

    #[test]
    fn test_overlay_cut_at_mid_node() {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.2],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::QUAD8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);
        // Vertical cutter through the bottom-edge mid node.
        let coords2 = nd::arr2(&[[0.5, -1.0], [0.5, 2.0]]);
        let mut mesh2 = UMesh::new(coords2.into_shared());
        mesh2.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        let cutted = mesh1.intersect_1d(mesh2);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        // The parent mid node is now a corner; no piece may reuse it as a mid node.
        for cell in cutted.elements_of_dim(Dimension::D2) {
            if cell.element_type() != ElementType::QPOLY {
                continue;
            }
            let conn = cell.connectivity();
            let n = conn.len() / 2;
            assert!(!conn[n..].contains(&4), "parent mid must not be reused");
        }
    }

    #[test]
    fn test_overlay_quad9_center_node_is_ignored() {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.15],
            [1.15, 0.5],
            [0.5, 1.15],
            [-0.15, 0.5],
            [0.5, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::QUAD9,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7, 8]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.25, 0.5, 1.25])
            .add_axis(vec![0.25, 0.75])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
        assert!(
            cutted
                .elements_of_dim(Dimension::D2)
                .any(|c| c.element_type() == ElementType::QPOLY)
        );
    }

    /// A unit-square quadrilateral whose top edge curves towards the interior by `inward`.
    /// That edge has a clockwise sweep; the other three are straight. The boundary stays
    /// simple for the values used below. `centre` adds the QUAD9 centre node.
    fn concave_quad(inward: f64, centre: bool) -> UMesh {
        let (et, coords, n) = if centre {
            (
                ElementType::QUAD9,
                nd::arr2(&[
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.5, 0.0],
                    [1.0, 0.5],
                    [0.5, 1.0 - inward],
                    [0.0, 0.5],
                    [0.5, 0.5],
                ]),
                9,
            )
        } else {
            (
                ElementType::QUAD8,
                nd::arr2(&[
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [0.5, 0.0],
                    [1.0, 0.5],
                    [0.5, 1.0 - inward],
                    [0.0, 0.5],
                ]),
                8,
            )
        };
        let mut mesh = UMesh::new(coords.into_shared());
        let conn = nd::Array2::from_shape_vec((1, n), (0..n).collect::<Vec<usize>>()).unwrap();
        mesh.add_regular_block(et, conn.into_shared(), None);
        mesh
    }

    /// A TRI6 whose bottom edge curves towards the interior of the triangle.
    fn concave_tri(inward: f64) -> UMesh {
        let coords = nd::arr2(&[
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, inward],
            [0.5, 0.5],
            [0.0, 0.5],
        ]);
        let mut mesh = UMesh::new(coords.into_shared());
        mesh.add_regular_block(
            ElementType::TRI6,
            nd::arr2(&[[0, 1, 2, 3, 4, 5]]).to_shared(),
            None,
        );
        mesh
    }

    fn concave_cutter(kind: usize) -> UMesh {
        let b = RegularUMeshBuilder::new();
        match kind {
            0 => b
                .add_axis(vec![-0.25, 0.5, 1.25])
                .add_axis(vec![0.25, 0.75]),
            1 => b.add_axis(vec![0.25, 0.5, 0.75]).add_axis(vec![-0.5, 1.5]),
            2 => b.add_axis(vec![-0.5, 1.5]).add_axis(vec![0.25, 0.5, 0.75]),
            3 => b
                .add_axis(vec![-0.5, 0.3, 0.7, 1.5])
                .add_axis(vec![0.3, 0.7]),
            4 => b.add_axis(vec![0.4, 0.6]).add_axis(vec![-0.5, 0.5, 1.5]),
            // A horizontal edge crossing the concave arc twice, plus vertical edges so
            // the cutter has no dangling segment inside the cell.
            5 => b
                .add_axis(vec![-0.5, 0.3, 0.7, 1.5])
                .add_axis(vec![0.6, 0.85]),
            _ => unreachable!(),
        }
        .build()
    }

    /// Every cut piece tiles the original cell: the area is preserved, its QPOLY ring is
    /// well formed and none of its nodes escapes the input footprint (a synthesized
    /// mid-side node on the wrong circle branch would).
    fn assert_imprint_valid(out: &UMesh, before: f64) {
        assert_abs_diff_eq!(exact_mesh_area(out), before, epsilon = 1e-9);
        let coords = out.coords();
        for cell in out.elements_of_dim(Dimension::D2) {
            if cell.element_type() != ElementType::QPOLY {
                continue;
            }
            let conn = cell.connectivity();
            assert!(conn.len() % 2 == 0 && conn.len() >= 4, "odd QPOLY {conn:?}");
            let n = conn.len() / 2;
            let mut verts = conn[..n].to_vec();
            verts.sort_unstable();
            verts.dedup();
            assert_eq!(verts.len(), n, "repeated vertex in {conn:?}");
            for &node in conn {
                let x = coords[(node, 0)];
                let y = coords[(node, 1)];
                assert!(x.is_finite() && y.is_finite());
                assert!(
                    (-0.5..=1.5).contains(&x) && (-0.5..=1.5).contains(&y),
                    "node {node} escaped the cell footprint: ({x}, {y})"
                );
            }
        }
    }

    /// Concave (clockwise-sweep) arcs must be cut exactly like convex ones: area is
    /// conserved for every inward bulge, every cutter layout and every quadratic type.
    #[test]
    fn test_overlay_concave_arcs_imprint_preserves_area() {
        // 0.25 makes a cutter edge exactly tangent to the arc: the piece is split at
        // the pinch instead of being emitted as a self-touching QPOLY.
        for inward in [0.02, 0.1, 0.2, 0.25, 0.30] {
            for centre in [false, true] {
                let mesh1 = concave_quad(inward, centre);
                let before = exact_mesh_area(&mesh1);
                for kind in 0..6 {
                    let out = mesh1.overlay(concave_cutter(kind), OverlayOperation::Imprint);
                    assert_imprint_valid(&out, before);
                }
            }
        }
        // A larger inward bulge would make the TRI6 arc cross its hypotenuse.
        for inward in [0.02, 0.08, 0.15] {
            let tri = concave_tri(inward);
            let before = exact_mesh_area(&tri);
            for kind in 0..6 {
                let out = tri.overlay(concave_cutter(kind), OverlayOperation::Imprint);
                assert_imprint_valid(&out, before);
            }
            let qpoly = crate::tools::polyze(&tri.view());
            for kind in 0..6 {
                let out = qpoly.overlay(concave_cutter(kind), OverlayOperation::Imprint);
                assert_imprint_valid(&out, before);
            }
        }
    }

    /// The intersection nodes created where a straight cutter edge crosses a concave
    /// arc must lie on that arc's circle.
    #[test]
    fn test_overlay_concave_arc_intersections_lie_on_the_circle() {
        let mesh1 = concave_quad(0.25, false);
        let before = exact_mesh_area(&mesh1);
        // The horizontal edge at y = 0.85 crosses the inward top arc twice, at
        // x = 0.5 +/- sqrt(0.625^2 - 0.525^2).
        let cutter = concave_cutter(5);
        let out = mesh1.overlay(cutter, OverlayOperation::Imprint);
        assert_imprint_valid(&out, before);

        // Circle through (1,1), (0.5,0.75), (0,1): centre (0.5,1.375), radius 0.625.
        let coords = out.coords();
        let on_arc: Vec<[f64; 2]> = (0..coords.nrows())
            .map(|i| [coords[(i, 0)], coords[(i, 1)]])
            .filter(|p| (p[1] - 0.85).abs() < 1e-9 && p[0] > 0.0 && p[0] < 1.0)
            .filter(|p| {
                (((p[0] - 0.5).powi(2) + (p[1] - 1.375).powi(2)).sqrt() - 0.625).abs() < 1e-9
            })
            .collect();
        assert_eq!(
            on_arc.len(),
            2,
            "expected two arc/line intersections: {on_arc:?}"
        );
    }

    /// One straight cutter edge crossing the same concave arc twice: the sub-segment
    /// between the crossings lies outside the cell and must not become a dart.
    #[test]
    fn test_overlay_concave_arc_crossed_twice_by_one_edge() {
        let mesh1 = concave_quad(0.30, false);
        let before = exact_mesh_area(&mesh1);
        // No vertical subdivision: a single horizontal edge at y = 0.75 crosses the
        // arc (which dips to y = 0.70) twice.
        let cutter = RegularUMeshBuilder::new()
            .add_axis(vec![-0.5, 1.5])
            .add_axis(vec![0.25, 0.75])
            .build();
        let out = mesh1.overlay(cutter, OverlayOperation::Imprint);
        assert_imprint_valid(&out, before);
    }

    /// An exact tangency produces a pinched ring; it must be emitted as several
    /// simple cells meeting at the pinch node, never as one self-touching QPOLY.
    #[test]
    fn test_overlay_tangent_cutter_vertex_splits_at_the_pinch() {
        let mesh1 = concave_quad(0.25, false);
        let before = exact_mesh_area(&mesh1);
        // The cutter has a vertex exactly at the tangent point (0.5, 0.75).
        let cutter = RegularUMeshBuilder::new()
            .add_axis(vec![-0.5, 0.5, 1.5])
            .add_axis(vec![0.25, 0.75])
            .build();
        let out = mesh1.overlay(cutter, OverlayOperation::Imprint);
        assert_imprint_valid(&out, before);
        assert!(
            out.num_elements_of_dim(Dimension::D2) > 1,
            "cell was not split"
        );
    }

    /// A closed cutter loop tangent to the boundary must not leave a dangling dart;
    /// the region outside the loop is split at the contact node.
    #[test]
    fn test_overlay_tangent_closed_cutter_loop_splits_at_the_pinch() {
        let mesh1 = concave_quad(0.20, false);
        let before = exact_mesh_area(&mesh1);
        let cutter = RegularUMeshBuilder::new()
            .add_axis(vec![0.2, 0.8])
            .add_axis(vec![0.2, 0.8])
            .build();
        let out = mesh1.overlay(cutter, OverlayOperation::Imprint);
        assert_imprint_valid(&out, before);
        assert!(
            out.num_elements_of_dim(Dimension::D2) > 1,
            "cell was not split"
        );
    }

    /// Quadratic cells whose node ids are not `0..n-1` must still be cut: `bounds2`
    /// used to feed `boundary_edges()`' global ids into the local `coord2_ref`.
    #[test]
    fn test_overlay_quadratic_cell_with_global_node_ids() {
        let coords = nd::arr2(&[
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.15],
            [1.15, 0.5],
            [0.5, 1.15],
            [-0.15, 0.5],
        ]);
        let mut mesh1 = UMesh::new(coords.into_shared());
        mesh1.add_regular_block(
            ElementType::QUAD8,
            nd::arr2(&[[8usize, 9, 10, 11, 12, 13, 14, 15]]).to_shared(),
            None,
        );
        let before = exact_mesh_area(&mesh1);
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![-0.25, 0.5, 1.25])
            .add_axis(vec![0.25, 0.75])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&cutted), before, epsilon = 1e-12);
    }

    /// A horizontal chain of three points, given either as SEG2 edges or as PLINEs.
    fn intersect_1d_cutter(style: usize) -> UMesh {
        let coords = nd::arr2(&[
            [-0.5, 0.5],
            [0.5, 0.5],
            [1.5, 0.5],
            [-0.5, 0.25],
            [0.5, 0.25],
            [1.5, 0.25],
        ]);
        let mut mesh = UMesh::new(coords.into_shared());
        match style {
            // Two SEG2 edges.
            0 => mesh.add_regular_block(
                ElementType::SEG2,
                nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
                None,
            ),
            // One PLINE with the same three points.
            1 => mesh.add_poly_block(
                ElementType::PLINE,
                nd::arr1(&[0, 1, 2]).to_shared(),
                nd::arr1(&[3]).to_shared(),
                None,
            ),
            // Two PLINEs (four SEG2 edges in total).
            _ => mesh.add_poly_block(
                ElementType::PLINE,
                nd::arr1(&[0, 1, 2, 3, 4, 5]).to_shared(),
                nd::arr1(&[3, 6]).to_shared(),
                None,
            ),
        }
        mesh
    }

    #[test]
    fn test_intersect_1d_seg2_and_pline() {
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let before = exact_mesh_area(&mesh1);
        let seg2 = mesh1.intersect_1d(intersect_1d_cutter(0));
        let pline = mesh1.intersect_1d(intersect_1d_cutter(1));
        let two_plines = mesh1.intersect_1d(intersect_1d_cutter(2));

        for out in [&seg2, &pline, &two_plines] {
            assert_abs_diff_eq!(exact_mesh_area(out), before, epsilon = 1e-12);
            assert!(out.num_elements_of_dim(Dimension::D2) > 1, "not cut");
        }
        assert_eq!(
            seg2.num_elements_of_dim(Dimension::D2),
            pline.num_elements_of_dim(Dimension::D2),
            "a PLINE must produce the same cut as its SEG2 chain"
        );
        assert!(
            two_plines.num_elements_of_dim(Dimension::D2)
                > pline.num_elements_of_dim(Dimension::D2)
        );
    }

    #[test]
    fn test_intersect_1d_quadratic_subject() {
        let mesh1 = quad8_square(0.15);
        let before = exact_mesh_area(&mesh1);
        let out = mesh1.intersect_1d(intersect_1d_cutter(1));
        assert_abs_diff_eq!(exact_mesh_area(&out), before, epsilon = 1e-12);
    }

    #[test]
    fn test_intersect_1d_empty_cutter_is_noop() {
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let before = exact_mesh_area(&mesh1);
        let empty = UMesh::new(mesh1.coords().to_shared());
        let out = mesh1.intersect_1d(empty);
        assert_abs_diff_eq!(exact_mesh_area(&out), before, epsilon = 1e-12);
        assert_eq!(out.num_elements_of_dim(Dimension::D2), 1);
    }

    #[test]
    #[should_panic(expected = "expects a 1D mesh of SEG2 or PLINE")]
    fn test_intersect_1d_rejects_2d_cutter() {
        let mesh1 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let _ = mesh1.intersect_1d(mesh2);
    }

    /// IMPRINT accepts a quadratic cutter; the four boolean operations are
    /// explicitly rejected for quadratic meshes until their classification is exact.
    #[test]
    #[should_panic(expected = "not supported for quadratic meshes")]
    fn test_overlay_booleans_reject_quadratic_meshes() {
        let mesh1 = quad8_square(0.15);
        let mesh2 = make_imesh_2d(1);
        let _ = mesh1.overlay(mesh2, OverlayOperation::Union);
    }

    /// A quadratic cutter is supported for IMPRINT: the arc/arc intersections must
    /// refine the subject without changing its area, for both convex and concave arcs.
    #[test]
    fn test_overlay_quadratic_cutter() {
        let mesh1 = make_imesh_2d(1);
        let before = exact_mesh_area(&mesh1);
        let mesh2 = quad8_square(0.05);
        let out = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_abs_diff_eq!(exact_mesh_area(&out), before, epsilon = 1e-12);

        for bulge in [0.05, 0.15, 0.25] {
            let quad = quad8_square(bulge);
            let before = exact_mesh_area(&quad);
            for cutter in [quad8_square(0.10), concave_quad(bulge, false)] {
                let out = quad.overlay(cutter, OverlayOperation::Imprint);
                assert_abs_diff_eq!(exact_mesh_area(&out), before, epsilon = 1e-12);
                assert_imprint_valid(&out, before);
            }
        }
    }

    #[test]
    fn test_overlay_quad8_untouched_is_verbatim() {
        let mesh1 = quad8_square(0.15);
        // A cutter box strictly outside mesh1.
        let mesh2 = RegularUMeshBuilder::new()
            .add_axis(vec![2.0, 3.0])
            .add_axis(vec![2.0, 3.0])
            .build();
        let cutted = mesh1.overlay(mesh2, OverlayOperation::Imprint);
        assert_eq!(cutted.num_elements_of_dim(Dimension::D2), 1);
        let cell = cutted.elements_of_dim(Dimension::D2).next().unwrap();
        assert_eq!(cell.element_type(), ElementType::QUAD8);
        assert_eq!(cell.connectivity(), &[0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
