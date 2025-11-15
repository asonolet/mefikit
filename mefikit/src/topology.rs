mod symmetry;
mod utils;

use itertools::Itertools;
use ndarray::prelude::*;
use petgraph::prelude::UnGraphMap;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::{SmallVec, smallvec};
use std::collections::{HashMap, HashSet};

use self::utils::SortedVecKey;
use crate::umesh::Connectivity;
use crate::{Dimension, ElementId, ElementLike, ElementType, UMesh};

pub trait ElementTopo<'a>: ElementLike<'a> {
    /// This function returns the subentities of the element based on the codimension.
    fn subentities(&self, codim: Option<Dimension>) -> Vec<(ElementType, Connectivity)> {
        use ElementType::*;
        let codim = match codim {
            None => Dimension::D1,
            Some(c) => c,
        };
        let co = self.connectivity();
        let mut res = Vec::new();
        match self.element_type() {
            SEG2 | SEG3 | SEG4 => {
                // 1D elements have edges as subentities
                if codim == Dimension::D1 {
                    let conn = arr2(&[[co[0]], [co[1]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                    res
                } else {
                    todo!()
                }
            }
            TRI3 => {
                // 2D elements have edges as subentities
                if codim == Dimension::D1 {
                    let conn = arr2(&[[co[0], co[1]], [co[1], co[2]], [co[2], co[0]]]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                    res
                } else {
                    todo!()
                }
            }
            TRI6 | TRI7 => {
                // 2D Quad elements have edges3 as subentities
                if codim == Dimension::D1 {
                    let conn = arr2(&[
                        [co[0], co[1], co[3]],
                        [co[1], co[2], co[4]],
                        [co[2], co[0], co[5]],
                    ]);
                    res.push((SEG3, Connectivity::new_regular(conn.to_shared())));
                    res
                } else {
                    todo!()
                }
            }
            QUAD4 => {
                // 2D elements have edges as subentities
                if codim == Dimension::D1 {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[1], co[2]],
                        [co[2], co[3]],
                        [co[3], co[0]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                    res
                } else {
                    todo!()
                }
            }
            TET4 => {
                // 3D elements have faces as subentities
                if codim == Dimension::D1 {
                    let conn = arr2(&[
                        [co[0], co[1], co[2]],
                        [co[1], co[2], co[3]],
                        [co[2], co[3], co[0]],
                        [co[3], co[0], co[1]],
                    ]);
                    res.push((TRI3, Connectivity::new_regular(conn.to_shared())));
                    res
                } else if codim == Dimension::D2 {
                    todo!()
                } else {
                    todo!()
                }
            }
            HEX8 => {
                if codim == Dimension::D1 {
                    let conn = arr2(&[
                        [co[0], co[1], co[2], co[3]],
                        [co[0], co[3], co[7], co[4]],
                        [co[0], co[4], co[5], co[1]],
                        [co[1], co[5], co[6], co[2]],
                        [co[2], co[6], co[7], co[3]],
                        [co[4], co[7], co[6], co[5]],
                    ]);
                    res.push((QUAD4, Connectivity::new_regular(conn.to_shared())));
                    res
                } else if codim == Dimension::D2 {
                    todo!()
                } else {
                    todo!()
                }
            }
            PGON => {
                if codim == Dimension::D1 {
                    let mut conn: Vec<_> = co.windows(2).flatten().cloned().collect();
                    conn.push(co[co.len() - 1]);
                    conn.push(co[0]);
                    let conn = Array2::from_shape_vec([conn.len() / 2, 2], conn).unwrap();
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                    res
                } else if codim == Dimension::D2 {
                    let conn = Array2::from_shape_vec([co.len(), 1], co.to_vec()).unwrap();
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                    res
                } else {
                    todo!()
                }
            }
            PHED => {
                if codim == Dimension::D1 {
                    let mut conn = Vec::new();
                    let mut offsets = Vec::new();
                    let mut offset = 0;
                    co.split_inclusive(|&e| e == usize::MAX).for_each(|a| {
                        let len = a.len() - 1;
                        offset += len;
                        offsets.push(offset);
                        conn.append(&mut a[..len].to_vec())
                    });
                    let offsets = Array1::from_vec(offsets);
                    let conn = Array::from_vec(conn);
                    res.push((
                        PGON,
                        Connectivity::new_poly(conn.to_shared(), offsets.to_shared()),
                    ));
                    res
                } else {
                    todo!()
                }
            }
            _ => todo!(), // For other types, return empty vector
        }
    }

    fn to_simplexes(&self) -> Vec<(ElementType, Vec<usize>)> {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX => vec![(VERTEX, vec![co[0]])],
            SEG2 | SEG3 | SEG4 => vec![(SEG2, vec![co[0], co[1]])],
            TRI3 | TRI6 | TRI7 => vec![(TRI3, vec![co[0], co[1], co[2]])],
            QUAD4 | QUAD8 | QUAD9 => vec![
                (TRI3, vec![co[0], co[1], co[3]]),
                (TRI3, vec![co[2], co[3], co[1]]),
            ],
            TET4 | TET10 => vec![(TET4, vec![co[0], co[1], co[2], co[3]])],
            HEX8 | HEX21 => vec![
                (TET4, vec![co[0], co[1], co[3], co[4]]),
                (TET4, vec![co[2], co[3], co[1], co[6]]),
                (TET4, vec![co[7], co[6], co[4], co[3]]),
                (TET4, vec![co[5], co[4], co[6], co[1]]),
                (TET4, vec![co[4], co[6], co[3], co[1]]),
            ],
            _ => todo!(),
        }
    }
}

impl<'a, T> ElementTopo<'a> for T where T: ElementLike<'a> {}

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
        None => mesh.element_blocks.keys().max().unwrap().dimension(),
    };
    // let mut subentities_hash: HashMap<SortedVecKey, [ElementId; 2]> =
    //     HashMap::with_capacity(self.coords.shape()[0]); // FaceId, ElemId
    let mut elem_to_elem: UnGraphMap<ElementId, ElementId> =
        UnGraphMap::with_capacity(mesh.num_elements(), mesh.coords.shape()[0]); // Node is
    // ElemId, edge
    // is FaceId
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    mesh.par_elements_of_dim(dim)
        .fold(
            HashMap::new,
            |mut subentities_hash: HashMap<
                SortedVecKey,
                (SmallVec<[ElementId; 2]>, SmallVec<[usize; 4]>, ElementType),
            >,
             elem| {
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
            },
        )
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
            let subentity_id = neighbors.element_blocks.get(&et).unwrap().len() - 1;
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
        None => mesh.element_blocks.keys().max().unwrap().dimension(),
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
                        let subentity_id = match neighbors.element_blocks.get(&et) {
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
        None => mesh.element_blocks.keys().max().unwrap().dimension(),
    };
    let mut subentities_hash: FxHashSet<SortedVecKey> = HashSet::default(); // Face
    let mut neighbors: UMesh = UMesh::new(mesh.coords.to_shared());

    for elem in mesh.elements_of_dim(dim) {
        for (et, conn) in elem.subentities(Some(codim)) {
            for co in conn.iter() {
                let key = SortedVecKey::new(co.into());
                if subentities_hash.get(&key).is_none() {
                    // The subentity is new
                    subentities_hash.insert(key);
                    neighbors.add_element(et, co, None, None);
                }
            }
        }
    }

    neighbors
}
