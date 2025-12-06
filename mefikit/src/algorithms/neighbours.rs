use itertools::Itertools;
use petgraph::prelude::UnGraphMap;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

use crate::topology::ElementTopo;
use crate::topology::SortedVecKey;
use crate::umesh::{Dimension, ElementId, ElementLike, ElementType, UMesh};

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
    dim: Option<Dimension>,
    codim: Option<Dimension>,
) -> (
    UMesh,
    UnGraphMap<ElementId, ElementId>, // element to element with subelem as edges
) {
    let codim = match codim {
        Some(c) => c,
        None => Dimension::D1,
    };
    let dim = match dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
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

    mesh.par_elements_of_dim(dim)
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
    dim: Option<Dimension>,
    codim: Option<Dimension>,
) -> (
    UMesh,
    UnGraphMap<ElementId, ElementId>, // element to element with subelem as edges
) {
    let codim = match codim {
        Some(c) => c,
        None => Dimension::D1,
    };
    let dim = match dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let mut subentities_hashmap: FxHashMap<SortedVecKey, (ElementId, SmallVec<[ElementId; 2]>)> =
        HashMap::default();
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(dim) {
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
    for (_, (fid, eids)) in subentities_hashmap {
        eids.iter().tuple_combinations().for_each(|(eid_a, eid_b)| {
            elem_to_elem.add_edge(*eid_a, *eid_b, fid);
        });
    }

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
pub fn compute_submesh(mesh: &UMesh, dim: Option<Dimension>, codim: Option<Dimension>) -> UMesh {
    let codim = match codim {
        Some(c) => c,
        None => Dimension::D1,
    };
    let dim = match dim {
        Some(c) => c,
        None => mesh.topological_dimension().unwrap(),
    };
    let mut subentities_hash: FxHashSet<SortedVecKey> = HashSet::default(); // Face
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(dim) {
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
