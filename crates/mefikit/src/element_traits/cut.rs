use ndarray as nd;
use smallvec::SmallVec;

use rustc_hash::FxHashMap;
use rustc_hash::FxHashSet;

use crate::element_traits::ElementTopo;
use crate::element_traits::SortedVecKey;
use crate::geometry::{ArcEdge, in_quadratic_polygon};
use crate::mesh::ElementId;
use crate::mesh::ElementIds;
use crate::mesh::ElementLike;
use crate::mesh::UMeshView;

pub type M1SgId = SortedVecKey;
pub type M2SgId = ElementId;
pub type NodeId = usize;
pub type M1M2Intersections = FxHashMap<M1SgId, Vec<(M2SgId, IntersectionIds)>>;
pub type SortedSegIntersections = FxHashMap<SortedVecKey, Vec<NodeId>>;
type Dart = [NodeId; 2];

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum IntersectionIds {
    One(NodeId),
    Segment(NodeId, NodeId),
}

/// The curve that generated one edge of a cut piece.
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum CutEdgeSource {
    /// Index into the subject cell's [`ElementTopo::boundary_edges`] list.
    Boundary(usize),
    /// Edge of the cutting mesh.
    Cutter(ElementId),
}

/// One polygon produced by cutting a cell.
#[derive(Debug, Clone)]
pub struct CutPiece {
    /// Polygon ring in counter-clockwise order.
    pub ring: Vec<NodeId>,
    /// Source curve of each ring edge: `edges[i]` is the edge
    /// `ring[i] -> ring[(i + 1) % ring.len()]`.
    pub edges: Vec<CutEdgeSource>,
}

#[derive(Debug, Clone)]
struct DartInfo {
    next: NodeId,
    visited: bool,
    angle: f64,
    source: CutEdgeSource,
}

type DartMap = FxHashMap<NodeId, Vec<DartInfo>>;

#[inline]
fn edge_key(conn: &[usize]) -> SortedVecKey {
    SortedVecKey::new(conn.iter().copied().collect())
}

#[inline]
fn point(coords: nd::ArrayView2<'_, f64>, node: NodeId) -> [f64; 2] {
    [coords[(node, 0)], coords[(node, 1)]]
}

struct BoundaryGeometry {
    conn: SmallVec<[usize; 3]>,
    arc: ArcEdge,
}

fn boundary_geometries(
    boundary_edges: &[SmallVec<[usize; 3]>],
    coords: nd::ArrayView2<'_, f64>,
) -> Vec<BoundaryGeometry> {
    boundary_edges
        .iter()
        .map(|conn| {
            let p0 = point(coords, conn[0]);
            let p1 = point(coords, conn[1]);
            let arc = if conn.len() == 3 {
                ArcEdge::new(p0, point(coords, conn[2]), p1)
            } else {
                ArcEdge::straight(p0, p1)
            };
            BoundaryGeometry {
                conn: conn.clone(),
                arc,
            }
        })
        .collect()
}

#[inline]
fn angle_of(dir: [f64; 2]) -> f64 {
    dir[1].atan2(dir[0])
}

struct DartSpec {
    n0: NodeId,
    n1: NodeId,
    angle0: f64,
    angle1: f64,
    source: CutEdgeSource,
    outer_visited: bool,
    push_heap: bool,
}

fn push_dart(graph: &mut DartMap, heap: &mut Vec<Dart>, spec: DartSpec) {
    if spec.push_heap {
        heap.push([spec.n0, spec.n1]);
    }
    // Inner dart.
    graph.entry(spec.n0).or_default().push(DartInfo {
        next: spec.n1,
        visited: false,
        angle: spec.angle0,
        source: spec.source,
    });
    // Outer dart. Boundary darts point outside the cell and are marked as already
    // walked; internal cutter darts have a face on both sides and stay available.
    graph.entry(spec.n1).or_default().push(DartInfo {
        next: spec.n0,
        visited: spec.outer_visited,
        angle: spec.angle1,
        source: spec.source,
    });
}

fn push_cutter_dart(
    graph: &mut DartMap,
    heap: &mut Vec<Dart>,
    n0: NodeId,
    n1: NodeId,
    coords: nd::ArrayView2<'_, f64>,
    source: CutEdgeSource,
) {
    let p0 = point(coords, n0);
    let p1 = point(coords, n1);
    let dir = [p1[0] - p0[0], p1[1] - p0[1]];
    let angle0 = angle_of(dir);
    let angle1 = angle_of([-dir[0], -dir[1]]);
    push_dart(
        graph,
        heap,
        DartSpec {
            n0,
            n1,
            angle0,
            angle1,
            source,
            outer_visited: false,
            push_heap: false,
        },
    );
}

/// Build the map n1 -> Vec<DartInfo>.
/// This map should be build locally (by cell).
/// The cell must be counter-clockwise oriented.
fn build_cell_dart_map(
    s_int: &SortedSegIntersections,
    mesh2: &UMeshView,
    candidate: &ElementIds,
    boundary_edges: &[SmallVec<[usize; 3]>],
    coords: nd::ArrayView2<'_, f64>,
) -> (Vec<Dart>, DartMap) {
    let mut graph: DartMap = FxHashMap::default();
    let mut heap: Vec<Dart> = Vec::new();
    // Only straight boundary darts can safely absorb a coincident cutter dart. A curved
    // boundary and its chord are distinct curves even when they share endpoints.
    let mut straight_boundary_pairs: FxHashSet<SortedVecKey> = FxHashSet::default();
    let mut cutter_pairs: FxHashSet<SortedVecKey> = FxHashSet::default();
    let geometries = boundary_geometries(boundary_edges, coords);

    // A straight cutter edge can cross the same curved boundary twice. Only the
    // sub-segments that actually lie inside the cell may contribute darts: the others
    // would open phantom faces and leave the dart walker with no next dart to follow.
    // Linear cells never hit that situation, so they keep the untouched fast path.
    let curved = geometries.iter().any(|g| g.arc.is_curved());
    let pgon: Vec<[f64; 2]> = if curved {
        let n = geometries.len();
        let mut p = Vec::with_capacity(2 * n);
        for g in &geometries {
            p.push(point(coords, g.conn[0]));
        }
        for g in &geometries {
            let mid = g
                .conn
                .get(2)
                .map(|&nd| point(coords, nd))
                .unwrap_or_else(|| {
                    let a = point(coords, g.conn[0]);
                    let b = point(coords, g.conn[1]);
                    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
                });
            p.push(mid);
        }
        p
    } else {
        Vec::new()
    };
    let inside = |n0: usize, n1: usize| -> bool {
        if !curved {
            return true;
        }
        let a = point(coords, n0);
        let b = point(coords, n1);
        in_quadratic_polygon(&[0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])], &pgon)
    };

    for (edge_idx, edge) in geometries.iter().enumerate() {
        let p0 = edge.conn[0];
        let p1 = edge.conn[1];
        let eid = edge_key(&edge.conn);
        let pieces = s_int.get(&eid);
        let default = [p0, p1];
        let pieces = pieces.map(|p| p.as_slice()).unwrap_or(&default);
        let inv = p1 == pieces[0];
        for w in pieces.windows(2) {
            let (n0, n1) = if inv { (w[1], w[0]) } else { (w[0], w[1]) };
            let from = point(coords, n0);
            let to = point(coords, n1);
            let dir0 = edge.arc.direction(&from, &to);
            let dir1 = edge.arc.direction(&to, &from);
            let angle0 = angle_of(dir0);
            let angle1 = angle_of(dir1);
            if !edge.arc.is_curved() {
                straight_boundary_pairs.insert(SortedVecKey::new(SmallVec::from_slice(&[n0, n1])));
            }
            push_dart(
                &mut graph,
                &mut heap,
                DartSpec {
                    n0,
                    n1,
                    angle0,
                    angle1,
                    source: CutEdgeSource::Boundary(edge_idx),
                    outer_visited: true,
                    push_heap: true,
                },
            );
        }
    }

    for c in candidate.iter() {
        let cutter = mesh2.element(c);
        let conn = cutter.connectivity();
        if conn.len() == 3 {
            panic!(
                "quadratic cutter edges are not supported: the quadratic mesh must be the \
                 subject (mesh1)"
            );
        }
        let eid = edge_key(conn);
        let ints = s_int.get(&eid);
        if let Some(ints) = ints {
            for w in ints.windows(2) {
                let (n0, n1) = (w[0], w[1]);
                let pair = SortedVecKey::new(SmallVec::from_slice(&[n0, n1]));
                if straight_boundary_pairs.contains(&pair) || !cutter_pairs.insert(pair) {
                    continue;
                }
                if !inside(n0, n1) {
                    continue;
                }
                push_cutter_dart(
                    &mut graph,
                    &mut heap,
                    n0,
                    n1,
                    coords,
                    CutEdgeSource::Cutter(cutter.id()),
                );
            }
        } else {
            // Edge is either completely inside or completely outside but very near. In both
            // cases it is added to the graph, unless it coincides with a straight boundary.
            let (n0, n1) = (conn[0], conn[1]);
            let pair = SortedVecKey::new(SmallVec::from_slice(&[n0, n1]));
            if straight_boundary_pairs.contains(&pair) || !cutter_pairs.insert(pair) {
                continue;
            }
            if !inside(n0, n1) {
                continue;
            }
            push_cutter_dart(
                &mut graph,
                &mut heap,
                n0,
                n1,
                coords,
                CutEdgeSource::Cutter(cutter.id()),
            );
        }
    }
    (heap, graph)
}

fn sort_cell_dart_map(dart_map: &mut DartMap) {
    for darts in dart_map.values_mut() {
        darts.sort_by(|a, b| a.angle.total_cmp(&b.angle).reverse());
    }
}

fn walk_dart_map(mut dart_map: DartMap, mut heap: Vec<Dart>) -> Vec<CutPiece> {
    let mut pieces = Vec::new();
    let mut ring: Vec<NodeId> = Vec::new();
    let mut sources: Vec<CutEdgeSource> = Vec::new();
    loop {
        // je repasse là à chaque nouveau pgon à fermer
        let mut initial = heap.pop();
        // Getting the next dart to follow, skipping already walked through darts
        while initial.is_some() {
            let [n, pos] = initial.unwrap();
            if !dart_map[&n][pos].visited {
                initial = Some([n, pos]);
                break;
            }
            initial = heap.pop();
        }
        let Some(mut dart) = initial else {
            break;
        };
        let start = dart[0];
        ring.push(start);
        loop {
            // je repasse là à chaque nouveau dart à suivre
            let info = dart_map
                .get_mut(&dart[0])
                .expect("The dart map should contain the last node")
                .get_mut(dart[1])
                .expect("The dart vec should contain the last dart");
            debug_assert!(!info.visited);
            info.visited = true; // Mark the dart as walked through
            let next_node = info.next;
            let source = info.source;
            sources.push(source);

            let next_darts = dart_map
                .get_mut(&next_node)
                .expect("The dart map should contain the next node");
            let last_opposed_dart_pos = next_darts
                .iter()
                .position(|d| d.next == dart[0] && d.source == source)
                .expect("The dart map should contain the previous contrary dart");

            // Adding the opposed dart into heap
            heap.push([next_node, last_opposed_dart_pos]);

            if next_node == start {
                break; // closes the pgon
            }
            ring.push(next_node);

            // Search for next dart not marked as walked through
            // This should almost always be the first one. The only exception is when there is an
            // unfinished seg which enters the pgon and does not cut it.
            let n = next_darts.len();
            let mut next_pos = 0;
            let mut found = false;
            for i in 1..n {
                next_pos = (last_opposed_dart_pos + i) % n;
                if !next_darts[next_pos].visited {
                    found = true;
                    break;
                }
            }
            if !found {
                // A dead end: the only unvisited dart here is the one opposed to the
                // dart we just walked, so this walk is a dangling spur rather than a
                // face. Discard it. The opposed darts pushed onto the heap above are
                // still available, so the real faces are enumerated from their other
                // side and no area is lost.
                ring.clear();
                sources.clear();
                break;
            }
            dart = [next_node, next_pos];
        }
        if !ring.is_empty() {
            pieces.push(CutPiece {
                ring: std::mem::take(&mut ring),
                edges: std::mem::take(&mut sources),
            });
        }
    }
    pieces
}

/// Whether a ring visits the same node more than once (a "pinched" ring).
#[inline]
fn is_pinched(ring: &[NodeId]) -> bool {
    for (i, &node) in ring.iter().enumerate() {
        if ring[..i].contains(&node) {
            return true;
        }
    }
    false
}

/// Splits pinched rings into simple rings at the pinch nodes.
///
/// A tangential contact makes two parts of a cut piece meet at a single node; emitting
/// them as one cell would produce a non-conformal, self-touching polygon. A closed walk
/// decomposes into simple cycles: whenever it reaches a node it already visited, the
/// portion walked since then is one cycle. Every edge is used exactly once, so the
/// pieces tile the original walk exactly and preserve its area.
fn split_pinched_pieces(pieces: Vec<CutPiece>) -> Vec<CutPiece> {
    let mut out = Vec::with_capacity(pieces.len());
    for piece in pieces {
        if !is_pinched(&piece.ring) {
            out.push(piece);
            continue;
        }
        let n = piece.ring.len();
        let mut path: Vec<NodeId> = Vec::with_capacity(n);
        let mut path_edges: Vec<CutEdgeSource> = Vec::with_capacity(n);
        path.push(piece.ring[0]);
        for i in 0..n {
            let next = piece.ring[(i + 1) % n];
            let edge = piece.edges[i];
            if let Some(k) = path.iter().rposition(|&nd| nd == next) {
                let mut edges = path_edges[k..].to_vec();
                edges.push(edge);
                let ring = path[k..].to_vec();
                // A self-loop bounds no area; never emit a degenerate cell for it.
                if ring.len() >= 2 {
                    out.push(CutPiece { ring, edges });
                }
                path.truncate(k + 1);
                path_edges.truncate(k);
            } else {
                path.push(next);
                path_edges.push(edge);
            }
        }
    }
    out
}

pub trait Cutable {
    fn cut_with_intersections(
        &self,
        seg_intersections: &SortedSegIntersections,
        m2_edges: &UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<Vec<CutPiece>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    fn cut_with_intersections(
        &self,
        seg_intersections: &SortedSegIntersections,
        m2_edges: &UMeshView,
        coords: nd::ArrayView2<'_, f64>,
        m2_candidates: &ElementIds,
    ) -> Option<Vec<CutPiece>> {
        //TODO: early detection that there is no intersections returns None
        if m2_candidates.is_empty() {
            return None;
        }

        let boundary_edges = self.boundary_edges();
        let (initial_darts, mut dart_map) = build_cell_dart_map(
            seg_intersections,
            m2_edges,
            m2_candidates,
            &boundary_edges,
            coords,
        );
        sort_cell_dart_map(&mut dart_map);
        // NOTE: dart heap trick: le 2eme usize peut servir a indiquer non l'indice du 2eme noeud
        // mais l'indice dans le vecteur de Darts qui me donne celui qui a le bon noeud.
        let mut heap = initial_darts;
        for [p1, p2] in &mut heap {
            let darts = dart_map
                .get(p1)
                .expect("The dart map should contain the next node");
            let dart_pos = darts
                .iter()
                .position(|d| d.next == *p2)
                .expect("The dart map should contain the previous contrary dart");
            *p2 = dart_pos;
        }

        let pieces = walk_dart_map(dart_map, heap);
        // Pinched rings only come from curved boundaries (a tangential contact), so
        // linear cells never pay for the decomposition below.
        let curved = boundary_edges.iter().any(|edge| edge.len() == 3);
        Some(if curved {
            split_pinched_pieces(pieces)
        } else {
            pieces
        })
    }
}

#[cfg(test)]
mod quadratic_cut_tests {
    use super::{CutEdgeSource, CutPiece, is_pinched, split_pinched_pieces};
    use crate::element_traits::ElementTopo;
    use crate::mesh::{ArcGroups, Element, ElementType::*};
    use ndarray as nd;

    /// A closed walk visiting node 2 three times decomposes into four simple rings.
    /// Every edge is used exactly once, so the pieces tile the original walk.
    #[test]
    fn split_pinched_ring_into_simple_cycles() {
        let ring = vec![16, 9, 6, 3, 17, 6, 15, 2, 6, 9, 14, 15, 6, 17];
        let edges: Vec<CutEdgeSource> = (0..ring.len())
            .map(|i| {
                if i % 2 == 0 {
                    CutEdgeSource::Boundary(i / 2)
                } else {
                    CutEdgeSource::Cutter(crate::mesh::ElementId::new(SEG2, i / 2))
                }
            })
            .collect();
        assert!(is_pinched(&ring));
        let pieces = split_pinched_pieces(vec![CutPiece { ring, edges }]);
        assert_eq!(pieces.len(), 4);
        let mut edge_count = 0;
        for piece in &pieces {
            assert!(!is_pinched(&piece.ring), "pinched ring: {:?}", piece.ring);
            assert!(piece.ring.len() >= 2);
            assert_eq!(piece.ring.len(), piece.edges.len());
            edge_count += piece.edges.len();
        }
        assert_eq!(edge_count, 14);
    }

    fn element<'a>(
        coords: &'a nd::Array2<f64>,
        conn: &'a [usize],
        et: crate::mesh::ElementType,
    ) -> Element<'a> {
        let family = Box::leak(Box::new(0usize));
        let groups = Box::leak(Box::new(ArcGroups::new()));
        Element::new(0, coords.view(), family, conn, et, groups)
    }

    #[test]
    fn boundary_edges_tri6_corners_only() {
        let coords = nd::Array2::zeros((6, 2));
        let conn = [10, 11, 12, 13, 14, 15];
        let elem = element(&coords, &conn, TRI6);
        let edges: Vec<Vec<usize>> = elem.boundary_edges().iter().map(|e| e.to_vec()).collect();
        assert_eq!(
            edges,
            vec![vec![10, 11, 13], vec![11, 12, 14], vec![12, 10, 15]]
        );
    }

    #[test]
    fn boundary_edges_quad8_corners_only() {
        let coords = nd::Array2::zeros((8, 2));
        let conn = [0, 1, 2, 3, 4, 5, 6, 7];
        let elem = element(&coords, &conn, QUAD8);
        let edges: Vec<Vec<usize>> = elem.boundary_edges().iter().map(|e| e.to_vec()).collect();
        assert_eq!(
            edges,
            vec![vec![0, 1, 4], vec![1, 2, 5], vec![2, 3, 6], vec![3, 0, 7]]
        );
    }

    #[test]
    fn boundary_edges_qpoly_corners_only() {
        let coords = nd::Array2::zeros((10, 2));
        let conn = [1, 2, 3, 4, 5, 6, 10, 11, 12, 13];
        let elem = element(&coords, &conn, QPOLY);
        let edges: Vec<Vec<usize>> = elem.boundary_edges().iter().map(|e| e.to_vec()).collect();
        assert_eq!(
            edges,
            vec![
                vec![1, 2, 6],
                vec![2, 3, 10],
                vec![3, 4, 11],
                vec![4, 5, 12],
                vec![5, 1, 13]
            ]
        );
    }

    #[test]
    fn boundary_edges_linear_unchanged() {
        let coords = nd::Array2::zeros((4, 2));
        let conn = [0, 1, 2, 3];
        let elem = element(&coords, &conn, QUAD4);
        let edges: Vec<Vec<usize>> = elem.boundary_edges().iter().map(|e| e.to_vec()).collect();
        assert_eq!(edges, vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]]);
    }
}
