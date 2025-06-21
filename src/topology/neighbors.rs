/// This module provides functionality to compute the whole neighbours graph of a given mesh.
use std::collections::HashMap;

use crate::umesh::Dimension;
use crate::umesh::ElementId;
use crate::umesh::UMesh;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SortedVecKey(Vec<usize>);

impl SortedVecKey {
    fn new(mut vec: Vec<usize>) -> Self {
        vec.sort_unstable();
        SortedVecKey(vec)
    }
}

/// This method is used to compute a subentity mesh.
///
/// By default, the mesh computed as a codimension of 1 with the entry mesh. Meaning that there is
/// a difference of 1 in their dimensions. Hence volumes gives faces mesh, faces gives edges mesh
/// and edges mesh gives vertices.
/// If the codim asked for is too high, the function will panick.
/// For performance reason, two subentities are considered the same if they have the same nodes,
/// regardless of their order.
pub fn compute_submesh(
    mesh: &UMesh,
    codim: Option<Dimension>,
) -> (
    UMesh,
    HashMap<ElementId, ElementId>,
    HashMap<ElementId, ElementId>,
) {
    let codim = match codim {
        Some(c) => c,
        None => Dimension::D1,
    };
    let mut subentities_hash: HashMap<SortedVecKey, ElementId> = HashMap::new();
    let mut subentity_to_elem: HashMap<ElementId, ElementId> = HashMap::new();
    let mut elem_to_subentity: HashMap<ElementId, ElementId> = HashMap::new();
    let mut neighbors: UMesh = UMesh::new(mesh.coords().clone());

    for elem in mesh.elements() {
        // let faces = elem.faces();
        for (et, conn) in elem.subentities(Some(codim)).unwrap() {
            let subentity_id = match neighbors.element_block(et) {
                Some(block) => block.len(),
                None => 0,
            };
            let key = SortedVecKey::new(conn.clone());
            if let Some(val) = subentities_hash.get(&key) {
                // The subentity already exists
                subentity_to_elem.insert(*val, elem.id());
                elem_to_subentity.insert(elem.id(), *val);
            } else {
                // The subentity is new
                let new_id = ElementId::new(et, subentity_id);
                subentities_hash.insert(key, new_id);
                subentity_to_elem.insert(new_id, elem.id());
                elem_to_subentity.insert(elem.id(), new_id);
                neighbors.add_element(et, conn.as_slice(), None, None);
            }
        }
    }

    (neighbors, subentity_to_elem, elem_to_subentity)
}
