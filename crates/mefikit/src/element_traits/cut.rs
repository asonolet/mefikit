use nalgebra::Point2;
use nalgebra::Vector2;
use ndarray as nd;

use rustc_hash::FxHashMap;

use crate::element_traits::SortedVecKey;
use crate::mesh::ElementId;
use crate::mesh::ElementIds;
use crate::mesh::ElementLike;
use crate::mesh::IndirectIndexOwned;
use crate::mesh::UMeshView;

pub type M1SgId = SortedVecKey;
pub type M2SgId = ElementId;
pub type NodeId = usize;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum IntersectionIds {
    One(NodeId),
    Segment(NodeId, NodeId),
}

pub type M1M2Intersections = FxHashMap<M1SgId, Vec<(M2SgId, IntersectionIds)>>;

type SortedSegIntersections = FxHashMap<SortedVecKey, Vec<NodeId>>;

/// Build edge to sorted intersections map.
/// Sort order is taken from the SortedVecKey order (from node with lower id to node with higer
/// id).
/// The SortedVecKey of the map is independent from mesh1/mesh2 distinction as NodeIds are common
/// between the two meshes.
/// TODO: this is half of the work, I need also seg2 intersected with seg1 by reversing the map.
fn to_sorted_intersections(
    intersections: &M1M2Intersections,
    _mesh2: UMeshView,
    global_coords: nd::ArrayView2<'_, f64>,
) -> SortedSegIntersections {
    let coords = global_coords;
    let mut sorted_intersections: SortedSegIntersections = FxHashMap::default();
    for (seg1_id, seg2_ints) in intersections {
        let p1: Point2<f64> = Point2::from_slice(coords.row(seg1_id[0]).as_slice().unwrap());
        let p2: Point2<f64> = Point2::from_slice(coords.row(seg1_id[1]).as_slice().unwrap());
        let oriented_vec: Vector2<f64> = p2 - p1;
        let mut sorted_ints: Vec<NodeId> = seg2_ints
            .iter()
            .map(|(_, intersection_ids)| match intersection_ids {
                IntersectionIds::One(i) => *i,
                IntersectionIds::Segment(_, _) => todo!("TODO: manage colinear case"),
            })
            .collect();
        sorted_ints.sort_by(|a, b| {
            let va: Vector2<f64> = Point2::from_slice(coords.row(*a).as_slice().unwrap()) - p1;
            let vb: Vector2<f64> = Point2::from_slice(coords.row(*b).as_slice().unwrap()) - p1;
            let da = oriented_vec.dot(&va);
            let db = oriented_vec.dot(&vb);
            da.total_cmp(&db)
        });
        sorted_intersections.insert(seg1_id.clone(), sorted_ints);
    }
    sorted_intersections
}

pub trait Cutable {
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<IndirectIndexOwned<usize>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    //TODO: replace added_intersection with a full mutable coords table.
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        _m2_candidates: &ElementIds,
    ) -> Option<IndirectIndexOwned<usize>> {
        let _seg_intersections = to_sorted_intersections(intersections, m2_edges, coords);
        todo!("build local node planar graph, extract faces, assemble mesh")
    }
}
