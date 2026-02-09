use rustc_hash::FxHashMap;

use crate::element_traits::{Cutable, Intersections, SortedVecKey, intersect_seg_seg};
use crate::mesh::{Dimension, ElementId, ElementLike, ElementType, UMesh};
use crate::prelude::ElementGeo;
use crate::tools::spatial_index::SpIdx2;
use crate::tools::{Descendable, snap, spatial_index::SpatiallyIndexable};
use crate::tools::{compute_hashsub_to_elem, compute_sub_to_elem};

type VertexId = usize;
type M1SgId = SortedVecKey;
type M2SgId = ElementId;
type M1M2Intersections = FxHashMap<M1SgId, Vec<(M2SgId, Intersections)>>;

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
pub fn intersect_meshes(mesh1: UMesh, mut mesh2: UMesh) -> UMesh {
    snap(&mut mesh2, mesh1.view(), 1e-12);

    let m1_edges = mesh1.descend(Some(Dimension::D2), Some(Dimension::D1));
    let m2_edges = mesh2.descend(Some(Dimension::D2), Some(Dimension::D1));

    //TODO: compute new stacked coords array with all coords from m1, then all coords except merged
    //coords from m2. This should almost entirely rewrite m2. It gives a common coordinates array
    //base. m1 does not really need to lye on this new array. m2 needs.

    let m2bvh = m2_edges.view().bvh2();

    let intersections = compute_intersections(&m1_edges, &m2_edges, &m2bvh);

    let mut cutted_mesh = UMesh::new(m1_edges.coords);

    //TODO: this should be replaced with an owned version of the coords table (so I can add many
    //coords to it).
    let mut intersection_added_points = 0;

    for cell in mesh1.elements() {
        //TODO: there should be a short circuit if cell is not touched.
        let reconstructed = cell.cut_with_intersections(
            &intersections,
            m2_edges.view(),
            &mut intersection_added_points,
        );

        // Whether the cell was modified or not, I should add it to the new mesh.
        // TODO: add the family and fields of the old cell when creating the new elements.
        for new_cell in reconstructed {
            cutted_mesh.add_element(ElementType::PGON, &new_cell, Some(*cell.family), None);
        }
    }
    cutted_mesh
}

/// Compute all intersections between mesh1 and mesh2 where mesh1 and mesh2 should be 2d mesh of
/// edges. The computation is done quite naively. A BVH is used to accelarate finding m2
/// intersecting edges.
/// The returned map can map back to segments from m1.
// TODO: easy to write as a rayon parallelized closure.
fn compute_intersections(m1_edges: &UMesh, m2_edges: &UMesh, m2bvh: &SpIdx2) -> M1M2Intersections {
    let mut intersections: M1M2Intersections = FxHashMap::default();

    for edge in m1_edges.elements() {
        let [min, max] = edge.bounds2();
        let candidates = m2bvh.in_bounds(min, max);
        for c in candidates.into_iter() {
            let edge2 = m2_edges.element(c);
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
            let m1sgid = SortedVecKey::new(edge.connectivity().into());
            if let Some(v) = intersections.get_mut(&m1sgid) {
                v.push((c, int));
            } else {
                intersections.insert(m1sgid, vec![(c, int)]);
            }
        }
    }
    intersections
}

/// Assembles reconstructed cells into a global mesh.
///
/// # Responsibilities
/// - Merge identical vertices
/// - Assign new IDs
/// - Validate manifoldness
fn assemble_mesh(_cells: Vec<Vec<VertexId>>) -> UMesh {
    todo!("vertex unification, topology validation")
}
