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
fn to_sorted_intersections(
    intersections: &M1M2Intersections,
    mesh2: UMeshView,
    coords: nd::ArrayView2<'_, f64>,
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
    mesh2: UMeshView,
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

/// Build the map n1 -> Vec<(n2, reached)>
/// This map should be build locally (by cell).
/// This map is sorted in counter clockwise order, ie n1n2 for a given n1 have increasing angle to Ox.
#[allow(unused)]
fn build_sorted_dart_map(sint: SortedSegIntersections) -> FxHashMap<NodeId, Vec<(NodeId, bool)>> {
    todo!()
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
