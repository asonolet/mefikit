use std::collections::BTreeMap;

use rustc_hash::FxHashMap;

use super::{Intersections, intersect_seg_seg};
use crate::element_traits::SortedVecKey;
use crate::mesh::ElementId;
use crate::mesh::ElementIds;
use crate::mesh::ElementLike;
use crate::mesh::UMesh;
use crate::mesh::UMeshView;
use crate::prelude::ElementTopo;

type VertexId = usize;
type PGSegId = usize;
type NodeId = usize;

struct PlanarGraph {
    segs: Vec<PGSeg>,
}

struct SegPart {
    m: bool, // if the seg comes from m1 or m2
    eid: ElementId,
    start: NodeId,
    end: NodeId,
}

struct PGSeg {
    seg: SegPart,
    start_segs: Vec<PGSegId>,
    end_segs: Vec<PGSegId>,
    angle: f32,
    visited: u8,
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
fn build_local_planar_graph<'a, E: ElementLike<'a>>(
    _cell: &E,
    _mesh1: UMeshView,
    _intersections: &[Intersections],
) -> PlanarGraph {
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
fn extract_faces_from_graph(graph: PlanarGraph) -> Vec<Vec<VertexId>> {
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

pub trait Cutable {
    // fn cut_with_edges(
    //     &self,
    //     mesh_edges: UMeshView,
    //     candidates: Option<ElementIds>,
    // ) -> Vec<Vec<VertexId>>;
    // fn compute_intersections(
    //     &self,
    //     mesh_edges: UMeshView,
    //     candidates: Option<ElementIds>,
    // ) -> Vec<Intersections>;
    fn cut_with_intersections(
        &self,
        intersections: &FxHashMap<SortedVecKey, Vec<(ElementId, Intersections)>>,
        m2_edges: UMeshView,
        added_intersection: &mut usize,
    ) -> Vec<Vec<VertexId>>;
}

impl<'a, T: ElementLike<'a>> Cutable for T {
    fn cut_with_intersections(
        &self,
        intersections: &FxHashMap<SortedVecKey, Vec<(ElementId, Intersections)>>,
        m2_edges: UMeshView,
        added_intersection: &mut usize,
    ) -> Vec<Vec<VertexId>> {
        todo!()
    }
    // /// Intersects a single mesh1 cell with relevant mesh2 edges.
    // ///
    // /// # Output
    // /// - Zero or more reconstructed sub-cells (closed polygons)
    // ///
    // /// # Steps
    // /// 1. Find candidate edges overlapping the cell
    // /// 2. Compute all segment–segment intersections
    // /// 3. Build a local planar graph
    // /// 4. Extract faces (DCEL traversal)
    // fn cut_with_edges(
    //     &self,
    //     mesh_edges: UMeshView,
    //     candidates: Option<ElementIds>,
    // ) -> Vec<Vec<VertexId>> {
    //     let intersections = self.compute_intersections(mesh_edges.clone(), candidates);

    //     let local_graph = build_local_planar_graph(self, mesh_edges, &intersections);

    //     extract_faces_from_graph(local_graph)
    // }

    // /// Computes all geometric intersections between a cell and mesh2 edges.
    // ///
    // /// # Output
    // /// - Unique intersection points with topological metadata
    // ///
    // /// # Invariants
    // /// - No duplicate intersections
    // /// - All intersections lie on both primitives within tolerance
    // fn compute_intersections(
    //     &self,
    //     mesh_edges: UMeshView,
    //     candidates: Option<ElementIds>,
    // ) -> Vec<Intersections> {
    //     let _candidates = match candidates {
    //         Some(c) => c,
    //         None => {
    //             let bt = mesh_edges
    //                 .blocks()
    //                 .map(|(&et, b)| (et, (0..b.len()).collect()))
    //                 .collect();
    //             ElementIds(bt)
    //         }
    //     };
    //     todo!()
    // }
}
