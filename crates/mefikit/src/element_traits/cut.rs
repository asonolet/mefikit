use itertools::Itertools;
use nalgebra::Point2;
use ndarray as nd;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::element_traits::SortedVecKey;
use crate::mesh::ElementId;
use crate::mesh::ElementIds;
use crate::mesh::ElementLike;
use crate::mesh::IndirectIndexOwned;
use crate::mesh::UMeshView;

pub type M1SgId = SortedVecKey;
pub type M2SgId = ElementId;
pub type NodeId = usize;
pub type M1M2Intersections = FxHashMap<M1SgId, Vec<(M2SgId, IntersectionIds)>>;
pub type SortedSegIntersections = FxHashMap<SortedVecKey, Vec<NodeId>>;
type Dart = [NodeId; 2];
type DartMap = FxHashMap<NodeId, Vec<(NodeId, bool)>>;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum IntersectionIds {
    One(NodeId),
    Segment(NodeId, NodeId),
}

/// Build the map n1 -> Vec<(n2, reached)>
/// This map should be build locally (by cell).
/// The cell must be counter-clockwise oriented
fn build_cell_dart_map(
    s_int: &SortedSegIntersections,
    mesh2: &UMeshView,
    candidate: &ElementIds,
    cell: &[usize],
) -> (Vec<Dart>, DartMap) {
    let mut graph: FxHashMap<NodeId, Vec<(NodeId, bool)>> = FxHashMap::default();
    let mut heap: Vec<Dart> = Vec::new();
    let mut m1_dart_set: FxHashSet<SortedVecKey> = FxHashSet::default();
    for (&p0, &p1) in cell.iter().circular_tuple_windows() {
        let co = [p0, p1];
        let eid = SortedVecKey::new(co.as_slice().into());
        let pieces = s_int.get(&eid);
        let pieces = match pieces {
            Some(a) => a,
            None => &Vec::from(co),
        };
        let inv = p1 == pieces[0];
        for w in pieces.windows(2) {
            let (n0, n1) = match !inv {
                true => (w[0], w[1]),
                false => (w[1], w[0]),
            };
            // Push to heap because I must go through it later
            heap.push([n0, n1]);
            // Push first node to frontier
            m1_dart_set.insert(SortedVecKey::new([n0, n1].as_slice().into()));
            // Inner dart
            graph.entry(n0).or_default().push((n1, false));
            // Outer dart
            graph.entry(n1).or_default().push((n0, true));
        }
    }
    for c in candidate.iter() {
        let eid = SortedVecKey::new(mesh2.element(c).connectivity().into());
        let ints = s_int.get(&eid);
        if let Some(ints) = ints {
            for w in ints.windows(2) {
                let n0 = w[0];
                let n1 = w[1];
                // NOTE: push only if new
                let dart_svk = SortedVecKey::new([n0, n1].as_slice().into());
                if !m1_dart_set.contains(&dart_svk) {
                    graph.entry(n0).or_default().push((n1, false));
                    graph.entry(n1).or_default().push((n0, false));
                }
            }
        } else {
            // NOTE: edge is either completly inside of completly outside but very near, in both
            // cases I should add it.
            let (n0, n1) = (eid[0], eid[1]);
            graph.entry(n0).or_default().push((n1, false));
            graph.entry(n1).or_default().push((n0, false));
        }
    }
    (heap, graph)
}

fn sort_cell_dart_map(dart_map: &mut DartMap, coords: nd::ArrayView2<'_, f64>) {
    for (k, v) in dart_map.iter_mut() {
        v.sort_by(|(n1, _), (n2, _)| {
            let ori = Point2::from_slice(coords.row(*k).as_slice().unwrap());
            let v0 = Point2::from_slice(coords.row(*n1).as_slice().unwrap()) - ori;
            let v1 = Point2::from_slice(coords.row(*n2).as_slice().unwrap()) - ori;
            let angle1 = v0.y.atan2(v0.x);
            let angle2 = v1.y.atan2(v1.x);
            angle1.total_cmp(&angle2).reverse()
        });
    }
}

fn walk_dart_map(mut dart_map: DartMap, mut heap: Vec<Dart>) -> IndirectIndexOwned<usize> {
    let mut pgons = IndirectIndexOwned::new();
    let mut pgon: Vec<NodeId> = Vec::new();
    loop {
        // je repasse là à chaque nouveau pgon à fermer
        let mut initial = heap.pop();
        // Getting the next dart to follow, skipping already walked through darts
        while initial.is_some() && dart_map[&initial.unwrap()[0]][initial.unwrap()[1]].1 {
            initial = heap.pop();
        }
        let Some(mut dart) = initial else {
            break;
        };
        pgon.push(dart[0]);
        loop {
            // je repasse là à chaque nouveau dart à suivre
            let p1 = dart_map
                .get_mut(&dart[0])
                .expect("The dart map should contain the last node")
                .get_mut(dart[1])
                .expect("The dart vec should contain the last dart");
            debug_assert!(!p1.1);
            p1.1 = true; // Mark the dart as walked through
            let next_node = p1.0;

            let next_darts = dart_map
                .get_mut(&next_node)
                .expect("The dart map should contain the next node");
            let last_opposed_dart_pos = next_darts
                .iter()
                .position(|(n, _)| *n == dart[0])
                .expect("The dart map should contain the previous contrary dart");

            // Adding the opposed dart into heap
            heap.push([next_node, last_opposed_dart_pos]);

            if next_node == pgon[0] {
                break; // closes the pgon
            }
            pgon.push(next_node);

            // Search for next dart not marked as walked through
            // This should almost always be the first one. The only exception is when there is an
            // unfinished seg which enters the pgon and does not cut it.
            let n = next_darts.len();
            let mut next_pos = 0;
            let mut found = false;
            for i in 1..n {
                next_pos = (last_opposed_dart_pos + i) % n;
                if !next_darts[next_pos].1 {
                    found = true;
                    break;
                }
            }
            if !found {
                // TODO: I must rewind, the heap (removing useless opposed dart), and the pgon
                todo!(
                    "Handle the case where there is no next dart to follow. This is when there is a piece of m2 which does not cut the cell."
                );
            }
            dart = [next_node, next_pos];
        }
        pgons.push(&pgon);
        pgon.clear();
    }
    pgons
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
    fn cut_with_intersections(
        &self,
        seg_intersections: &SortedSegIntersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<IndirectIndexOwned<usize>> {
        //TODO: early detection that there is no intersections returns None
        if m2_candidates.is_empty() {
            return None;
        }

        let (initial_darts, mut dart_map) = build_cell_dart_map(
            seg_intersections,
            &m2_edges,
            m2_candidates,
            self.connectivity(),
        );
        sort_cell_dart_map(&mut dart_map, coords);
        // NOTE: dart heap trick: le 2eme usize peut servir a indiquer non l'indice du 2eme noeud
        // mais l'indice dans le vecteur de Darts qui me donne celui qui a le bon noeud.
        let mut heap = initial_darts;
        for [p1, p2] in &mut heap {
            let darts = dart_map
                .get(p1)
                .expect("The dart map should contain the next node");
            let dart_pos = darts
                .iter()
                .position(|(n, _)| *n == *p2)
                .expect("The dart map should contain the previous contrary dart");
            *p2 = dart_pos;
        }

        Some(walk_dart_map(dart_map, heap))
    }
}
