use itertools::Itertools;
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

pub type SortedSegIntersections = FxHashMap<SortedVecKey, Vec<NodeId>>;

/// Build the map n1 -> Vec<(n2, reached)>
/// This map should be build locally (by cell).
/// The cell must be counter-clockwise oriented
fn build_cell_dart_map(
    s_int: &SortedSegIntersections,
    mesh2: &UMeshView,
    candidate: &ElementIds,
    cell: &[usize],
) -> FxHashMap<NodeId, Vec<(NodeId, bool)>> {
    let mut graph: FxHashMap<NodeId, Vec<(NodeId, bool)>> = FxHashMap::default();
    for (&p0, &p1) in cell.iter().circular_tuple_windows() {
        let co = [p0, p1];
        let eid = SortedVecKey::new(co.as_slice().into());
        let pieces = s_int
            .get(&eid)
            .expect("The cell boundary should be in the svk intersection map.");
        for w in pieces.windows(2) {
            let n0 = w[0];
            let n1 = w[1];
            // Inner dart
            graph.entry(n0).or_default().push((n1, false));
            // Outer dart
            graph.entry(n1).or_default().push((n0, true));
        }
    }
    for c in candidate.iter() {
        let eid = SortedVecKey::new(mesh2.element(c).connectivity().into());
        let ints = s_int
            .get(&eid)
            .expect("candidate should be in intersection map");
        for w in ints.windows(2) {
            let n0 = w[0];
            let n1 = w[1];
            graph.entry(n0).or_default().push((n1, false));
            graph.entry(n1).or_default().push((n0, false));
        }
    }
    graph
}

pub trait Cutable {
    fn cut_with_intersections(
        &self,
        seg_intersections: &SortedSegIntersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<IndirectIndexOwned<usize>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    //TODO: replace added_intersection with a full mutable coords table.
    fn cut_with_intersections(
        &self,
        seg_intersections: &SortedSegIntersections,
        m2_edges: UMeshView,
        _coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<IndirectIndexOwned<usize>> {
        let _dart_map = build_cell_dart_map(
            seg_intersections,
            &m2_edges,
            m2_candidates,
            self.connectivity(),
        );
        todo!("build local node planar graph, extract faces, assemble mesh")
    }
}
