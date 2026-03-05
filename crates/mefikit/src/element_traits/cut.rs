use itertools::Itertools;
use ndarray as nd;

use rustc_hash::FxHashMap;

use crate::element_traits::SortedVecKey;
use crate::mesh::ElementId;
use crate::mesh::ElementLike;
use crate::mesh::IndirectIndexOwned;
use crate::mesh::UMeshView;
use crate::prelude::ElementTopo;

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
#[repr(transparent)]
pub struct NodeId(pub usize);

#[repr(transparent)]
struct PGNodeId(usize);

/// This graph uses PGNodes as the nodes of the graph.
/// Each node stores the edges leaving from it in order to reach other nodes.
struct PlanarGraphNode {
    nodes: FxHashMap<NodeId, PGNode>,
}

struct PGNode {
    darts: Vec<Dart>,
}

enum MeshId {
    M1,
    M2,
}

/// This represent an oriented edge. It can be easly used to access other nodes and therefore
/// follow the CCW path.
struct Dart {
    angle: f32,         // used to sort Darts in CCW order in a PGNode
    end: NodeId,        // allows to search for next PGNode
    edge: SortedVecKey, // references the original edge. It is useless for segments as a segment is
    // well defined with its extremities, but usefull to compute mid points for SEG3 elements.
    m: MeshId, // The dart could be issued from mesh 1 (reference) or mesh2 (tool). This allow to
               // discriminate and use the edge id appropriatly.
}

type SortedSegIntersections = FxHashMap<SortedVecKey, Vec<(SortedVecKey, NodeId)>>;

/// Build edge to sorted intersections map.
/// The key of the map is independent from mesh1/mesh2 distinction.
/// Hence the rest of the algorithm is independent from it.
fn edge_to_intersections<'a, E: ElementLike<'a>>(
    intersections: &M1M2Intersections,
) -> SortedSegIntersections {
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
    intersections: &SortedSegIntersections,
) -> PlanarGraphNode {
    let mut pgl: FxHashMap<NodeId, PGNode> = FxHashMap::default();
    let conn_cell = cell.connectivity();

    //TODO: if cell is QUADPGON, shortens this to the half
    let len_cell = conn_cell.len();

    // Border nodes
    for i in 0..len_cell {
        let e1 = SortedVecKey::new(
            [conn_cell[i % len_cell], conn_cell[(i + 1) % len_cell]]
                .as_slice()
                .into(),
        );
        let seg_int = &intersections[&e1];
        let mut i = 0;
        // A chaque intersection je construis tous les darts
        while i < seg_int.len() {
            let mut darts = vec![];
            let int_node = seg_int[i].1;
            if i > 0 {
                // J'ajoute en darts le lien vers le noeud précédent sur le segment que je parcours
                let bfore = Dart {
                    angle: 0.0,
                    end: seg_int[i - 1].1,
                    edge: e1.clone(),
                    m: MeshId::M1,
                };
                darts.push(bfore);
            }
            // J'ajoute successivement les liens version les autres segments
            while seg_int[i].1 == int_node {
                let s = &seg_int[i].0;
                let seg_s = &intersections[&s];
                // TODO: Attention, ici c'est le cas d'une intersection pure, pas d'un noeud tangent donc
                // fusionné !
                let k = seg_s
                    .iter()
                    .find_position(|int| int.1 == int_node)
                    .unwrap()
                    .0;
                darts.push(Dart {
                    angle: 0.0,
                    end: seg_s[k - 1].1,
                    edge: s.clone(),
                    m: MeshId::M2,
                });
                darts.push(Dart {
                    angle: 0.0,
                    end: seg_s[k + 1].1,
                    edge: s.clone(),
                    m: MeshId::M2,
                });
                i += 1;
            }
            // J'ajout le lien avec le noeud suivant
            if i < seg_int.len() {
                darts.push(Dart {
                    angle: 0.0,
                    end: seg_int[i].1,
                    edge: e1.clone(),
                    m: MeshId::M1,
                });
            }

            let pgn = PGNode { darts };
            pgl.insert(int_node, pgn);
        }
    }
    // Identifier les noeuds qui sont à l'intérieur du PGON
    // Créer les PGNodes associées à ces noeuds :
    // - je regarde la connectivité nodale n2n
    // - j'en déduis un segment, et donc un id, donc je remonte aux intersections triées
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

pub trait Cutable {
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
    ) -> Option<IndirectIndexOwned<usize>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    //TODO: replace added_intersection with a full mutable coords table.
    fn cut_with_intersections(
        &self,
        intersections: &M1M2Intersections,
        m2_edges: UMeshView,
        coords: nd::ArrayView2<'_, f64>,
    ) -> Option<IndirectIndexOwned<usize>> {
        todo!()
    }
}
