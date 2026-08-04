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

use rustc_hash::FxHashMap;

use nalgebra::Point2;
use nalgebra::Vector2;
use ndarray as nd;

use crate::element_traits::cut::{
    IntersectionIds, M1M2Intersections, NodeId, SortedSegIntersections,
};
use crate::element_traits::is_in::{in_polygon_stable, strict_interior_point};
use crate::element_traits::{
    Cutable, Intersection, Intersections, PointId, SortedVecKey, intersect_seg_seg,
};
use crate::mesh::{Dimension, Element, ElementLike, ElementType, UMesh, UMeshView};
use crate::prelude::ElementGeo;
use crate::tools::duplicates_from;
use crate::tools::spatial_index::SpIdx2;
use crate::tools::{Descendable, spatial_index::SpatiallyIndexable};

/// Merges `subject` onto `reference` in a single coordinate space.
///
/// The merged mesh coordinates are `[reference coords; subject coords]` and the `subject` blocks
/// are remapped so that subject nodes coinciding with a reference node point to that reference
/// node id. The two meshes therefore share the same node ids in the merged space.
///
/// NOTE: must be called before computing intersections because the merged coords indexing is used
/// to resolve intersection node ids.
fn merge_on_reference_coords(subject: UMesh, reference: UMeshView) -> UMesh {
    // reference node id -> coincident subject node ids
    let ref_to_subject_nodes = duplicates_from(subject.view(), reference.clone(), 1e-12);
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
    merged
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
}

/// Computes the geometric intersection (overlay) of two 2D meshes.
///
/// Refines `mesh1`'s cells with the edges of `mesh2`.
fn intersect_2d2d(mesh1: &UMesh, mesh2: UMesh) -> UMesh {
    let mesh2 = merge_on_reference_coords(mesh2, mesh1.view());
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
    let mesh2 = merge_on_reference_coords(mesh2, mesh1.view());
    cut_2d_with_edges(mesh1, mesh2)
}

/// Cut the cells of `subject` with the edges of `cutter` and keep the pieces for which the
/// `keep` predicate returns `true`. The predicate receives `true` when the piece lies inside
/// `cutter`.
fn cut_and_classify(subject: &UMesh, cutter: UMesh, keep: impl Fn(bool) -> bool) -> UMesh {
    let cutter = merge_on_reference_coords(cutter, subject.view());

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
    let mesh2 = merge_on_reference_coords(mesh2, mesh1.view());

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
fn cut_cells_all(
    out: &mut UMesh,
    subject: &UMesh,
    cutting_edges: &UMeshView,
    cutting_bvh: &SpIdx2,
    seg_intersections: &SortedSegIntersections,
) {
    for cell in subject.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = cutting_bvh.in_bounds(bmin, bmax);
        let reconstructed = cell.cut_with_intersections(
            seg_intersections,
            cutting_edges,
            out.coords(),
            &candidates,
        );

        // If the cell was cut, I add new polys from the cut
        if let Some(polys) = reconstructed {
            for new_cell in polys {
                out.add_element(ElementType::PGON, &new_cell, Some(*cell.family), None);
            }
        } else {
            out.add_element(
                cell.element_type(),
                cell.connectivity(),
                Some(*cell.family),
                cell.fields.clone(),
            );
        }
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
                    out.add_element(ElementType::PGON, &new_cell, Some(*cell.family), None);
                }
            }
        } else if keep(cell_inside(&cell, cutter, &cutter_bvh)) {
            out.add_element(
                cell.element_type(),
                cell.connectivity(),
                Some(*cell.family),
                cell.fields.clone(),
            );
        }
    }
}

/// Returns the cell vertices in counter-clockwise order.
fn cell_points(cell: Element<'_>) -> Vec<[f64; 2]> {
    let points: Vec<[f64; 2]> = (0..cell.connectivity().len())
        .map(|i| cell.coord2(i).into())
        .collect();
    oriented_ccw(points)
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
    let Some(interior) = strict_interior_point(points) else {
        return false;
    };
    let candidates = cutter_bvh.intersects(interior);
    candidates
        .iter()
        .map(|eid| cutter.element(eid))
        .map(|e| cell_points(e))
        .any(|cell| in_polygon_stable(&interior, &cell))
}

/// Signed area of a polygon using the shoelace formula.
/// Positive result indicates counter-clockwise orientation.
fn shoelace_signed_area(points: &[[f64; 2]]) -> f64 {
    let n = points.len();
    let mut area2 = 0.0;
    for i in 0..n {
        let [x0, y0] = points[i];
        let [x1, y1] = points[(i + 1) % n];
        area2 += x0 * y1 - x1 * y0;
    }
    area2 / 2.0
}

/// Centroid of a polygon using the shoelace formula.
/// Returns the first vertex for a degenerate polygon.
#[allow(dead_code)]
fn shoelace_centroid(points: &[[f64; 2]]) -> [f64; 2] {
    let n = points.len();
    let mut area2 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let [x0, y0] = points[i];
        let [x1, y1] = points[(i + 1) % n];
        let cross = x0 * y1 - x1 * y0;
        area2 += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    if area2.abs() < 1e-30 {
        return points[0];
    }
    [cx / (3.0 * area2), cy / (3.0 * area2)]
}

/// Returns the polygon points in counter-clockwise order.
fn oriented_ccw(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    if shoelace_signed_area(&points) < 0.0 {
        points.reverse();
    }
    points
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
) {
    use crate::mesh::ElementType::*;
    let int = match (edge.element_type(), edge2.element_type()) {
        (SEG2, SEG2) => intersect_seg_seg(
            edge.coord2(0),
            edge.coord2(1),
            edge2.coord2(0),
            edge2.coord2(1),
        ),
        _ => todo!("Intersection with SEG3 is not yet implemented"),
    };
    match int {
        Intersections::None => (),
        Intersections::One(i) => {
            {
                let m1sgid = SortedVecKey::new(edge.connectivity().into());
                let v = intersections.entry(m1sgid).or_default();
                match i {
                    Intersection::Existing(p) => {
                        v.push((
                            edge2.id(),
                            IntersectionIds::One(intersection_p_to_global_index(p, edge, &edge2)),
                        ));
                    }
                    Intersection::New(coord) => {
                        v.push((edge2.id(), IntersectionIds::One(*new_coord_id)));
                        *new_coord_id += 1;
                        new_coords.extend_from_slice(&coord);
                    }
                }
            };
        }
        Intersections::Two([_i1, _i2]) => todo!("Arc intersections is not yet implemented."),
        Intersections::Segment([p1, p2]) => {
            let m1sgid = SortedVecKey::new(edge.connectivity().into());
            let v = intersections.entry(m1sgid).or_default();
            v.push((
                edge2.id(),
                IntersectionIds::Segment(
                    intersection_p_to_global_index(p1, edge, &edge2),
                    intersection_p_to_global_index(p2, edge, &edge2),
                ),
            ));
        }
    }
}

fn intersection_p_to_global_index(p: PointId, edge: &Element<'_>, edge2: &Element<'_>) -> usize {
    match p {
        PointId::P1 => edge.connectivity[0],
        PointId::P2 => edge.connectivity[1],
        PointId::P3 => edge2.connectivity[0],
        PointId::P4 => edge2.connectivity[1],
    }
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
                let points: Vec<[f64; 2]> = (0..cell.connectivity().len())
                    .map(|i| cell.coord2(i).into())
                    .collect();
                shoelace_signed_area(&points).abs()
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
}
