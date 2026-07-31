use rustc_hash::FxHashMap;

use nalgebra::Point2;
use nalgebra::Vector2;
use ndarray as nd;

use crate::element_traits::cut::{
    IntersectionIds, M1M2Intersections, NodeId, SortedSegIntersections,
};
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
pub fn intersect_2d2d(mesh1: UMesh, mesh2: UMesh) -> UMesh {
    //NOTE: must be before the compute_intersections because mesh2 coords indexing is used.
    let mesh2 = concat_merge_on_ref_coords(mesh2, mesh1.view());

    let m1_edges = mesh1.descend(Some(Dimension::D2), Some(Dimension::D1));
    let m2_edges = mesh2.descend(Some(Dimension::D2), Some(Dimension::D1));

    let m2bvh = m2_edges.view().bvh2();

    let (intersections, added_coords) = compute_intersections(&m1_edges, &m2_edges, &m2bvh);

    // Concatenates m1 coords, m2 coords, new intersections coords
    let new_coords = nd::concatenate![nd::Axis(0), mesh2.coords(), added_coords];

    let mut cutted_mesh = UMesh::new(new_coords.into_shared());

    let seg_intersections =
        to_sorted_intersections(&intersections, &m2_edges.view(), &cutted_mesh.coords());

    for cell in mesh1.elements_of_dim(Dimension::D2) {
        let [bmin, bmax] = cell.bounds2();
        let candidates = m2bvh.in_bounds(bmin, bmax);
        let reconstructed = cell.cut_with_intersections(
            &seg_intersections,
            m2_edges.view(),
            cutted_mesh.coords(),
            &candidates,
        );

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

        let mut sorted_ints: Vec<NodeId> = Vec::new();

        // First point
        sorted_ints.push(eid[0]);
        // Intersection points
        sorted_ints.append(v);
        // Sorting all intersections points
        sorted_ints.sort_by(|a, b| {
            let va: Vector2<f64> = Point2::from_slice(coords.row(*a).as_slice().unwrap()) - p1;
            let vb: Vector2<f64> = Point2::from_slice(coords.row(*b).as_slice().unwrap()) - p1;
            let da = oriented_vec.dot(&va);
            let db = oriented_vec.dot(&vb);
            da.total_cmp(&db)
        });
        // Adding last point (known)
        sorted_ints.push(eid[1]);

        // Removing duplicates
        sorted_ints.dedup();
        v.append(&mut sorted_ints);
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
    // use crate::io::write;
    use crate::mesh_examples::{make_imesh_2d, make_mesh_2d_multi_simple};
    // use std::path::Path;

    #[test]
    fn test_intersect_meshe_square1() {
        let mesh1 = make_imesh_2d(2);
        let mut mesh2 = make_imesh_2d(1);
        mesh2.coords *= 1. / 3.;

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
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

        let mesh_cutted = intersect_2d2d(mesh1, mesh2);
        // let p = Path::new("test_intersect_meshes3.vtk");
        // let _ = write(p, mesh_cutted.view());
        assert_eq!(mesh_cutted.coords().nrows(), 29);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D0), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D1), 0);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D2), 16);
        assert_eq!(mesh_cutted.num_elements_of_dim(Dimension::D3), 0);
    }
}
