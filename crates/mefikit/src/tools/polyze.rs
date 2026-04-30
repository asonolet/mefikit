// This module converts any UMeshView into a UMesh having only one ElementBlock: the
// polyline/polygon/polyhedron block. The inverse operation can be used to simplify some cells
// from a mesh.

use std::collections::HashMap;

use rustc_hash::FxHashMap;

use crate::mesh::{ElementId, ElementType, UMesh};
use crate::tools::MeshSelect;

/// Transforms all non polygons/polyhedrons into polygons/polyhedrons.
///
/// Seg stays as seg, vertex as vertex.
/// TODO: quadratic polygons are not yet implemented.
#[allow(unused)]
pub fn polyze(mesh: &UMesh) -> (UMesh, FxHashMap<ElementId, ElementId>) {
    let mut polyzed_map: FxHashMap<_, _> = HashMap::default();
    let mut new_pgon_id = mesh.block(ElementType::PGON).map_or(0, |b| b.len());
    let mut new_phed_id = mesh.block(ElementType::PHED).map_or(0, |b| b.len());
    let sel = crate::tools::selection::types(vec![ElementType::PGON, ElementType::PHED]);
    let (_, non_poly_mesh) = mesh.select(!sel, false);

    for elem in non_poly_mesh.elements() {
        todo!()
    }
    (non_poly_mesh, polyzed_map)
}
