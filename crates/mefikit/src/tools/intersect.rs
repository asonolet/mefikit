use rustc_hash::FxHashMap;

use ndarray as nd;

use crate::element_traits::cut;
use crate::element_traits::{
    Cutable, Intersection, Intersections, PointId, SortedVecKey, intersect_seg_seg,
};
use crate::mesh::{Dimension, Element, ElementLike, ElementType, UMesh, UMeshView};
use crate::prelude::ElementGeo;
use crate::tools::duplicates_from;
use crate::tools::spatial_index::SpIdx2;
use crate::tools::{Descendable, spatial_index::SpatiallyIndexable};

fn concat_merge_on_ref_coords(subject: UMesh, reference: UMeshView) -> UMesh {
    let m1_to_m2_nodes = duplicates_from(subject.view(), reference.clone(), 1e-12);
    let shift = reference.coords().nrows();
    let mut m2_to_m1_nodes: FxHashMap<usize, usize> = FxHashMap::default();
    for (n1, ns2) in m1_to_m2_nodes {
        for n2 in ns2 {
            m2_to_m1_nodes.insert(n2 + shift, n1);
        }
    }
    let new_coords = nd::concatenate![nd::Axis(0), reference.coords(), subject.coords()];
    let mut new_mesh2 = UMesh::new(new_coords.into_shared());
    for (&et, b) in subject.blocks() {
        let mut new_block = b.clone();
        let co = &mut new_block.connectivity;
        co.shift_index(shift);
        co.replace(&m2_to_m1_nodes);
        new_mesh2.element_blocks.insert(et, new_block);
    }
    new_mesh2
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
pub fn intersect_meshes(mesh1: UMesh, mesh2: UMesh) -> UMesh {
    //NOTE: must be before the compute_intersections because mesh2 coords indexing is used.
    let mesh2 = concat_merge_on_ref_coords(mesh2, mesh1.view());

    let m1_edges = mesh1.descend(Some(Dimension::D2), Some(Dimension::D1));
    let m2_edges = mesh2.descend(Some(Dimension::D2), Some(Dimension::D1));

    let m2bvh = m2_edges.view().bvh2();

    let (intersections, added_coords) = compute_intersections(&m1_edges, &m2_edges, &m2bvh);

    // Concatenates m1 coords, m2 coords, new intersections coords
    let new_coords = nd::concatenate![nd::Axis(0), mesh2.coords(), added_coords];

    let mut cutted_mesh = UMesh::new(new_coords.into_shared());

    for cell in mesh1.elements() {
        let reconstructed =
            cell.cut_with_intersections(&intersections, m2_edges.view(), cutted_mesh.coords());

        // If the cell was cut, I add new polys from the cut
        if let Some(polys) = reconstructed {
            for new_cell in polys {
                cutted_mesh.add_element(ElementType::PGON, &new_cell, Some(*cell.family), None);
            }
        } else {
            cutted_mesh.add_element(
                cell.element_type(),
                cell.connectivity(),
                Some(*cell.family),
                cell.fields.clone(),
            );
        }
    }
    cutted_mesh
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
) -> (cut::M1M2Intersections, nd::ArcArray2<f64>) {
    let mut intersections: cut::M1M2Intersections = FxHashMap::default();
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
    intersections: &mut cut::M1M2Intersections,
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
            let m1sgid = cut::M1SgId(SortedVecKey::new(edge.connectivity().into()));
            let v = intersections.entry(m1sgid).or_default();
            match i {
                Intersection::Existing(PointId::P1) => {
                    v.push((cut::M2SgId(edge2.id()), cut::NodeId(edge.connectivity[0])));
                }
                Intersection::Existing(PointId::P2) => {
                    v.push((cut::M2SgId(edge2.id()), cut::NodeId(edge.connectivity[1])));
                }
                Intersection::Existing(PointId::P3) => {
                    v.push((cut::M2SgId(edge2.id()), cut::NodeId(edge2.connectivity[0])));
                }
                Intersection::Existing(PointId::P4) => {
                    v.push((cut::M2SgId(edge2.id()), cut::NodeId(edge2.connectivity[1])));
                }
                Intersection::New(coord) => {
                    v.push((cut::M2SgId(edge2.id()), cut::NodeId(*new_coord_id)));
                    *new_coord_id += 1;
                    new_coords.extend_from_slice(&coord);
                }
            }
        }
        Intersections::Two([_i1, _i2]) => todo!(),
        Intersections::Segment([_i1, _i2]) => todo!(),
    }
}
