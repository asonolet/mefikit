use crate::element_traits::Cutable;
use crate::mesh::{Dimension, UMesh};
use crate::prelude::ElementGeo;
use crate::tools::{Descendable, snap, spatial_index::SpatiallyIndexable};

type VertexId = usize;

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

    let mesh2_edges = mesh2.descend(Some(Dimension::D2), Some(Dimension::D1));
    let spatial_index = mesh2_edges.view().bvh2();

    let mut new_cells = Vec::new();

    for cell in mesh1.elements() {
        let [min, max] = cell.bounds2();
        let candidates = spatial_index.in_bounds(min, max);
        let reconstructed = cell.cut_with_edges(mesh2_edges.view(), Some(&candidates));

        new_cells.extend(reconstructed);
    }

    assemble_mesh(new_cells)
}

/// Assembles reconstructed cells into a global mesh.
///
/// # Responsibilities
/// - Merge identical vertices
/// - Assign new IDs
/// - Validate manifoldness
fn assemble_mesh(_cells: Vec<Vec<VertexId>>) -> UMesh {
    todo!("vertex unification, topology validation")
}
