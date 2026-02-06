use crate::element_traits::{ElementGeo, Intersections, intersect_seg_seg};
use crate::mesh::{Element, ElementId, ElementLike, ElementType, UMesh};
use crate::tools::snap;

use arrayvec::ArrayVec;
use nalgebra::Point2;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct DirectedEdge(usize, usize);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct UndirectedEdge(usize, usize);

impl UndirectedEdge {
    fn new(a: usize, b: usize) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

impl From<DirectedEdge> for UndirectedEdge {
    fn from(e: DirectedEdge) -> Self {
        Self::new(e.0, e.1)
    }
}

// impl From<UndirectedEdge> for DirectedEdge {
//     fn from(e: UndirectedEdge) -> Self {
//         Self(e.0, e.1)
//     }
// }

/// In the general case, there could be multiple intersections beween 1d elements (ex SEG2 and
/// SEG3).
///
/// The implementation should be completly symmetric for it to be correct.
fn intersect_1d_elems(
    e1: &Element,
    e2: &Element,
    _next_node_index: usize,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    match (e1.element_type(), e2.element_type()) {
        (ElementType::SEG2, ElementType::SEG2) => {
            let _seg1_nodes = [e1.connectivity()[0], e1.connectivity()[1]];
            let _seg2_nodes = [e2.connectivity()[0], e2.connectivity()[1]];
            let p1 = e1.coord2(0);
            let p2 = e1.coord2(1);
            let p3 = e2.coord2(0);
            let p4 = e2.coord2(1);
            let _inters = intersect_seg_seg(p1, p2, p3, p4);
            todo!()
        }
        _ => todo!(),
    }
}

// Cette méthode permet de découper un maillage 2d potentiellement non conforme avec un maillage
// de segments propres (sans noeuds non fusionnés).
// pub fn cut_2d_mesh_with_1d_mesh(mesh: &UMesh, tool_mesh: UMesh) -> Result<UMesh, String> {
//     // tool_mesh.merge_nodes();
//     let tool_mesh = tool_mesh.prepend_coords(mesh.coords());
//
//     let (m1d, mgraph) = compute_neighbours(mesh, Some(D2), None);
//
//     // build R*-tree with 1d tool mesh
//     let segs: Vec<_> = tool_mesh
//         .elements()
//         .map(|seg| Segment::from(&seg).0)
//         .collect();
//     let cutter_tree = RTree::bulk_load(segs);
//
//     // Maintenant je travaille uniquement en terme de connectivité, j'utilise la table de
//     // coordonnées suivante pour les identifiants de noeuds:
//     // Table de mesh + table de tool_mesh après merge + nouveaux noeuds différents
//     // Je créé la liste des segments de tool mesh découpés (avec les noeuds d'intersection)
//     let mut e2int: HashMap<ElementId, Vec<[usize; 2]>> = HashMap::new();
//     for el in mesh.elements_of_dim(D2) {
//         let segs_in_elem: Vec<_> = cutter_tree
//             .locate_in_envelope_intersecting(&el.to_aabb2())
//             .collect();
//         for (_, _, &eid) in mgraph.edges(el.id()) {
//             // TODO: edge could be SEG3!
//             let edge = m1d.element(eid);
//             let _intersections: &_ = e2int.entry(eid).or_insert_with(|| {
//                 let intersections = Vec::new();
//                 for seg in &segs_in_elem {
//                     let seg_elem = tool_mesh.element(seg.data);
//                     // Calcul des intersections avec edge
//                     // Une intersection est soit un Point, soit un Segment
//                     let _intersection_coords = intersect_1d_elems(&edge, &seg_elem, 4);
//                     todo!()
//                 }
//                 intersections
//             });
//         }
//     }
//     todo!()
// }

/// Mesh vertex
pub struct Vertex {
    pub id: VertexId,
    pub point: Point2<f64>,
}

/// Oriented edge (topological)
pub struct Edge {
    pub id: EdgeId,
    pub end: VertexId,
    pub start: VertexId,
    pub left_cell: Option<usize>,
    pub right_cell: Option<usize>,
}

type VertexId = usize;

type EdgeId = usize;

struct SpatialIndex {}

struct LocalGraph {}

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

    let mesh2_edges = extract_oriented_edges(&mesh2);

    let spatial_index = build_spatial_index(&mesh2_edges);

    let mut new_cells = Vec::new();

    for cell in mesh1.elements() {
        let reconstructed = intersect_cell_with_edges(cell, &mesh1, &mesh2_edges, &spatial_index);

        new_cells.extend(reconstructed);
    }

    assemble_mesh(new_cells)
}

/// Converts mesh cells into oriented edges with adjacency metadata.
///
/// # Guarantees
/// - All edges are consistently oriented (CCW)
/// - Each edge knows its left/right cell
///
/// # Purpose
/// - Turns mesh2 into a planar graph suitable for intersection tests
fn extract_oriented_edges(mesh: &UMesh) -> Vec<Edge> {
    todo!("iterate cells, build edges, merge duplicates")
}

/// Builds a spatial index for fast edge queries.
///
/// # Purpose
/// - Reduce intersection tests from O(N²) to O(N log N)
///
/// # Implementation
/// - R-tree, BVH, or uniform grid
fn build_spatial_index(edges: &[Edge]) -> SpatialIndex {
    todo!("choose and build spatial acceleration structure")
}

/// Intersects a single mesh1 cell with relevant mesh2 edges.
///
/// # Output
/// - Zero or more reconstructed sub-cells (closed polygons)
///
/// # Steps
/// 1. Find candidate edges overlapping the cell
/// 2. Compute all segment–segment intersections
/// 3. Build a local planar graph
/// 4. Extract faces (DCEL traversal)
fn intersect_cell_with_edges(
    cell: Element,
    mesh1: &UMesh,
    mesh2_edges: &[Edge],
    index: &SpatialIndex,
) -> Vec<Vec<VertexId>> {
    let candidates = query_candidate_edges(&cell, mesh1, index);

    let intersections = compute_intersections(&cell, mesh1, &candidates);

    let local_graph = build_local_planar_graph(&cell, mesh1, &intersections);

    extract_faces_from_graph(local_graph)
}

/// Finds mesh2 edges potentially intersecting a mesh1 cell.
///
/// # Strategy
/// - Bounding box overlap
/// - Conservative filtering (false positives allowed)
fn query_candidate_edges(cell: &Element, mesh1: &UMesh, index: &SpatialIndex) -> Vec<EdgeId> {
    todo!("bbox construction and spatial query")
}

/// Computes all geometric intersections between a cell and mesh2 edges.
///
/// # Output
/// - Unique intersection points with topological metadata
///
/// # Invariants
/// - No duplicate intersections
/// - All intersections lie on both primitives within tolerance
fn compute_intersections(
    cell: &Element,
    mesh1: &UMesh,
    candidate_edges: &[EdgeId],
) -> Vec<Intersections> {
    todo!("segment-segment intersection, de-duplication")
}

/// Builds a local planar graph for a cell.
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
fn build_local_planar_graph(
    cell: &Element,
    mesh1: &UMesh,
    intersections: &[Intersections],
) -> LocalGraph {
    todo!("vertex creation, edge splitting, half-edge linkage")
}

/// Extracts closed CCW faces from a local planar graph.
///
/// # Algorithm
/// - Angle-sorted half-edge traversal
/// - DCEL-style face walk
///
/// # Output
/// - Valid polygonal cells
fn extract_faces_from_graph(graph: LocalGraph) -> Vec<Vec<VertexId>> {
    todo!("DCEL traversal, loop detection, orientation check")
}

/// Assembles reconstructed cells into a global mesh.
///
/// # Responsibilities
/// - Merge identical vertices
/// - Assign new IDs
/// - Validate manifoldness
fn assemble_mesh(cells: Vec<Vec<VertexId>>) -> UMesh {
    todo!("vertex unification, topology validation")
}
