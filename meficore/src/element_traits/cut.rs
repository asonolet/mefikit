use rustc_hash::FxHashMap;

use super::{Intersections, intersect_seg_seg};
use crate::element_traits::SortedVecKey;
use crate::mesh::ElementId;
use crate::mesh::ElementLike;
use crate::mesh::IndirectIndexOwned;
use crate::mesh::UMeshView;
use crate::prelude::ElementTopo;

#[repr(transparent)]
pub struct NodeId(pub usize);

#[repr(transparent)]
struct DartId(usize);

#[repr(transparent)]
struct PGSegId(usize);

#[repr(transparent)]
struct PGNodeId(usize);

/// This graph uses PGNodes as the nodes of the graph.
/// Each node stores the edges leaving from it in order to reach other nodes.
struct PlanarGraphNode {
    nodes: Vec<PGNode>,
}

struct PGNode {
    node: NodeId, // This is the global mesh node id.
    darts: Vec<Dart>,
}

enum MeshId {
    M1,
    M2,
}

/// This represent an oriented edge. It can be easly used to access other nodes and therefore
/// follow the CCW path.
struct Dart {
    angle: f32,       // used to sort Darts in CCW order in a PGNode
    origin: PGNodeId, // usefull ?
    end: PGNodeId,    // allows to search for next PGNode
    edge: ElementId,  // references the original edge. It is useless for segments as a segment is
    // well defined with its extremities, but usefull to compute mid points for SEG3 elements.
    m: MeshId, // The dart could be issued from mesh 1 (reference) or mesh2 (tool). This allow to
               // discriminate and use the edge id appropriatly.
}

/// Build edge to sorted intersections map.
/// The key of the map is independent from mesh1/mesh2 distinction.
/// Hence the rest of the algorithm is independent from it.
fn edge_to_intersections<'a, E: ElementLike<'a>>(
    intersections: &M1M2Intersections,
) -> FxHashMap<SortedVecKey, Vec<NodeId>> {
    todo!()
}

/// Builds a local node planar graph for a cell.
///
/// # Nodes
/// - Cell vertices
/// - Intersection points
///
/// # Edges
/// - Subdivided cell edges
/// - Portions of mesh2 edges inside the cell
///
/// # Guarantees
/// - Graph is planar
/// - All edges are paired (half-edges)
fn build_local_node_planar_graph<'a, E: ElementLike<'a>>(
    cell: &E,
    mesh2: UMeshView,
    intersections: &M1M2Intersections,
) -> PlanarGraphNode {
    let mut pgl: Vec<PGNode> = Vec::new();
    let conn_cell = cell.connectivity();
    let len_cell = conn_cell.len();
    for (i, &n) in conn_cell.iter().enumerate() {
        let e1 = [conn_cell[(i - 1) % len_cell], n];
        let e2 = [n, conn_cell[(i + 1) % len_cell]];
        let e1_id = SortedVecKey::new(e1.as_slice().into());
        let e2_id = SortedVecKey::new(e2.as_slice().into());
        let mut darts = vec![];
        let pgn = PGNode {
            node: NodeId(n),
            darts,
        };
        pgl.push(pgn);
    }
    PlanarGraphNode { nodes: pgl }
}

/// Extracts closed CCW faces from a local planar graph.
///
/// # Algorithm
/// - Angle-sorted half-edge traversal
/// - DCEL-style face walk
///
/// # Output
/// - Valid polygonal cells
fn extract_faces_from_graph(graph: PlanarGraphNode) -> IndirectIndexOwned<usize> {
    todo!("DCEL traversal, loop detection, orientation check")
}

/// Assembles reconstructed cells into a global mesh.
///
/// # Responsibilities
/// - Merge identical vertices
/// - Assign new IDs
/// - Validate manifoldness
fn assemble_mesh(cells: IndirectIndexOwned<usize>) {
    todo!("vertex unification, topology validation")
}

#[repr(transparent)]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct M1SgId(pub SortedVecKey);

#[repr(transparent)]
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct M2SgId(pub ElementId);

pub type M1M2Intersections = FxHashMap<M1SgId, Vec<(M2SgId, NodeId)>>;

struct Intersection(NodeId, [f64; 2]);

pub trait Cutable {
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
    ) -> Option<IndirectIndexOwned<usize>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    //TODO: replace added_intersection with a full mutable coords table.
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
    ) -> Option<IndirectIndexOwned<usize>> {
        for (_, seg) in self.subentities(None) {
            //TODO: get m1 seg ends (0 and 1 index)
            //find intersections with m2_seg in intersections
            //create SegParts and add them to the PlanarGraph.
        }
        todo!()
    }
}
