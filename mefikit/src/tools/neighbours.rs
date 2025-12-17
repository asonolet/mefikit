use itertools::Itertools;
use petgraph::prelude::UnGraphMap;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

use crate::mesh::{Dimension, ElementId, ElementIds, ElementLike, UMesh};
use crate::topology::ElementTopo;
use crate::topology::SortedVecKey;

/// This method is used to compute a subentity mesh in parallel.
///
/// By default, the mesh computed as a codimension of 1 with the entry mesh. Meaning that there
/// is a difference of 1 in their dimensions. Hence volumes gives faces mesh, faces gives edges
/// mesh and edges mesh gives vertices.  If the codim asked for is too high, the function will
/// panick.  For performance reason, two subentities are considered the same if they have the
/// same nodes, regardless of their order.
/// The output graph is a element to element graph (from input mesh), using subentities as edges (weight in
/// petgraph lang)
#[cfg(feature = "rayon")]
pub fn par_compute_neighbours(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    target_dim: Option<Dimension>,
) -> (
    UMesh,
    UnGraphMap<ElementId, ElementId>, // element to element with subelem as edges
) {
    let src_dim = match src_dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let codim = match target_dim {
        Some(t) => src_dim - t,
        None => Dimension::D1,
    };
    // let mut subentities_hash: HashMap<SortedVecKey, [ElementId; 2]> =
    //     HashMap::with_capacity(self.coords.shape()[0]); // FaceId, ElemId
    let mut elem_to_elem: UnGraphMap<ElementId, ElementId> =
        UnGraphMap::with_capacity(mesh.num_elements(), mesh.coords.shape()[0]); // Node is
    // ElemId, edge
    // is FaceId
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    type SubentityMap =
        HashMap<SortedVecKey, (SmallVec<[ElementId; 2]>, SmallVec<[usize; 4]>, ElementType)>;

    mesh.par_elements_of_dim(src_dim)
        .fold(HashMap::new, |mut subentities_hash: SubentityMap, elem| {
            for (et, conn) in elem.subentities(Some(codim)) {
                for co in conn.iter() {
                    let key = SortedVecKey::new(co.into());
                    match subentities_hash.get_mut(&key) {
                        // The subentity already exists
                        Some((ids, _, _)) => ids.push(elem.id()),
                        None => {
                            subentities_hash.insert(key, (smallvec![elem.id()], co.into(), et));
                        }
                    }
                }
            }
            subentities_hash
        })
        .reduce(HashMap::new, |mut a, b| {
            for (k, (ids, conn, et)) in b {
                match a.get_mut(&k) {
                    // The subentity already exists
                    Some((existing_ids, _, _)) => existing_ids.extend(ids),
                    None => {
                        a.insert(k, (ids, conn, et));
                    }
                }
            }
            a
        })
        .into_iter()
        .for_each(|(_key, (ids, conn, et))| {
            neighbors.add_element(et, conn.as_slice(), None, None);
            let subentity_id = neighbors.get_block(et).unwrap().len() - 1;
            let new_id = ElementId::new(et, subentity_id);
            ids.iter().tuple_combinations().for_each(|(eid_a, eid_b)| {
                elem_to_elem.add_edge(*eid_a, *eid_b, new_id);
            });
        });

    (neighbors, elem_to_elem)
}

/// This method is used to compute a subentity mesh.
///
/// By default, the mesh computed as a codimension of 1 with the entry mesh. Meaning that there
/// is a difference of 1 in their dimensions. Hence volumes gives faces mesh, faces gives edges
/// mesh and edges mesh gives vertices.  If the codim asked for is too high, the function will
/// panick.  For performance reason, two subentities are considered the same if they have the
/// same nodes, regardless of their order.
/// The output graph is a element to element graph (from input mesh), using subentities as edges (weight in
/// petgraph lang)
pub fn compute_neighbours(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    target_dim: Option<Dimension>,
) -> (
    UMesh,
    UnGraphMap<ElementId, ElementId>, // element to element with subelem as edges
) {
    let src_dim = match src_dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let codim = match target_dim {
        Some(t) => src_dim - t,
        None => Dimension::D1,
    };
    let mut subentities_hashmap: FxHashMap<SortedVecKey, (ElementId, SmallVec<[ElementId; 2]>)> =
        HashMap::default();
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(src_dim) {
        for (et, conn) in elem.subentities(Some(codim)) {
            for co in conn.iter() {
                let key = SortedVecKey::new(co.into());

                match subentities_hashmap.get_mut(&key) {
                    None => {
                        // The subentity is new
                        let subentity_id = match neighbors.get_block(et) {
                            Some(block) => block.len(),
                            None => 0,
                        };
                        let new_id = ElementId::new(et, subentity_id);
                        subentities_hashmap.insert(key, (new_id, smallvec![elem.id()]));
                        neighbors.add_element(et, co, None, None);
                    }
                    Some((_, eids)) => {
                        // The subentity already exists
                        eids.push(elem.id());
                    }
                }
            }
        }
    }
    // Node is ElemId, edge is FaceId
    let mut elem_to_elem: UnGraphMap<ElementId, ElementId> =
        UnGraphMap::with_capacity(mesh.num_elements(), mesh.coords.shape()[0]);
    for elem in mesh.elements_of_dim(src_dim) {
        elem_to_elem.add_node(elem.id());
    }
    for (_, (fid, eids)) in subentities_hashmap {
        eids.iter().tuple_combinations().for_each(|(eid_a, eid_b)| {
            elem_to_elem.add_edge(*eid_a, *eid_b, fid);
        });
    }

    (neighbors, elem_to_elem)
}

/// This method is used to compute a subentity mesh.
///
/// By default, the mesh computed has a codimension of 1 with the entry mesh. Meaning that there
/// is a difference of 1 in their dimensions. Hence volumes gives faces mesh, faces gives edges
/// mesh and edges mesh gives vertices.  If the codim asked for is too high, the function will
/// panick.  For performance reason, two subentities are considered the same if they have the
/// same nodes, regardless of their order.
/// The output graph is a element to element graph (from input mesh), using subentities as edges (weight in
/// petgraph lang)
pub fn compute_submesh(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    target_dim: Option<Dimension>,
) -> UMesh {
    let src_dim = match src_dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let codim = match target_dim {
        Some(t) => src_dim - t,
        None => Dimension::D1,
    };
    let mut subentities_hash: FxHashSet<SortedVecKey> = HashSet::default(); // Face
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(src_dim) {
        for (et, conn) in elem.subentities(Some(codim)) {
            for co in conn.iter() {
                let key = SortedVecKey::new(co.into());
                if !subentities_hash.contains(&key) {
                    // The subentity is new
                    subentities_hash.insert(key);
                    neighbors.add_element(et, co, None, None);
                }
            }
        }
    }

    neighbors
}

/// This method is used to compute the submesh and the map sub_elem_id to elem ids.
pub fn compute_sub_to_elem(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    target_dim: Option<Dimension>,
) -> (UMesh, FxHashMap<ElementId, Vec<ElementId>>) {
    let src_dim = match src_dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let codim = match target_dim {
        Some(t) => src_dim - t,
        None => Dimension::D1,
    };
    let mut hash_to_subid: FxHashMap<SortedVecKey, ElementId> = HashMap::default(); // Face
    let mut sub_to_elem: FxHashMap<ElementId, Vec<ElementId>> = HashMap::default(); // Face
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(src_dim) {
        for (et, conn) in elem.subentities(Some(codim)) {
            for co in conn.iter() {
                let key = SortedVecKey::new(co.into());
                if let Some(subid) = hash_to_subid.get(&key) {
                    // The subentity is already in the mesh
                    let elems = sub_to_elem.get_mut(subid).unwrap();
                    elems.push(elem.id());
                } else {
                    // The subentity is new
                    let subid = neighbors.add_element(et, co, None, None);
                    hash_to_subid.insert(key, subid);
                    sub_to_elem.insert(subid, vec![elem.id()]);
                }
            }
        }
    }

    (neighbors, sub_to_elem)
}

/// This method is used to compute the boundaries of a mesh.
pub fn compute_boundaries(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    target_dim: Option<Dimension>,
) -> UMesh {
    let (submesh, sub_to_elem) = compute_sub_to_elem(mesh, src_dim, target_dim);
    let boundaries_ids: ElementIds = sub_to_elem
        .iter()
        .filter_map(|(&sub, elems)| match elems.len() {
            1 => Some(sub),
            _ => None,
        })
        .collect();
    submesh.extract(&boundaries_ids)
}
