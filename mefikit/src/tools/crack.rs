use std::collections::HashMap;

use petgraph::algo::tarjan_scc;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet, FxHasher};

// This algorithm duplicates some nodes in order to break connectivites between some cells.
use crate::mesh::{Dimension, ElementId, ElementLike, UMesh, UMeshView};
use crate::tools::compute_neighbours;
use crate::tools::neighbours::compute_sub_to_elem;
use crate::tools::selector::Selector;
use crate::topology::SortedVecKey;

/// Gets the ids of elements from partmesh which are in mesh_ref.
///
/// Supposes that the coordinates array is the same.
fn find_equals(mesh_ref: UMeshView, partmesh: UMeshView) -> Vec<Option<ElementId>> {
    let dims: FxHashSet<Dimension> = partmesh.blocks().map(|(&et, _)| et.dimension()).collect();
    let mut hash_to_ref: FxHashMap<SortedVecKey, ElementId> = HashMap::default();
    for dim in dims {
        let svk_eid: Vec<_> = mesh_ref
            .par_elements_of_dim(dim)
            .map(|e| (SortedVecKey::new(e.connectivity().into()), e.id()))
            .collect();
        hash_to_ref.extend(svk_eid.into_iter())
    }
    partmesh
        .elements()
        .map(|e| {
            hash_to_ref
                .get(&SortedVecKey::new(e.connectivity().into()))
                .copied()
        })
        .collect()
}

pub fn crack(mesh: UMesh, cut: UMeshView) -> UMesh {
    // First extract the vicinity of the cut
    let nodes = cut.used_nodes();
    let mut near_mesh = Selector::new(&mesh)
        .nodes(false)
        .id_in(nodes.as_slice())
        .select();
    let (submesh, f2c) = compute_sub_to_elem(&near_mesh, None, None);
    // Throws if some element in cut is not in submesh
    let cut_ids = find_equals(submesh.view(), cut.view());
    let cut_c2c: Vec<[ElementId; 2]> = cut_ids
        .into_iter()
        .map(|x| x.expect("cut elements should be found in mesh submesh."))
        .map(|f_id| f2c[&f_id].clone().try_into().unwrap())
        .collect();
    let mut new_node_id = mesh.coords().len();
    // I am gessing a capacity here as I cannot know it in advance
    let mut n2o_nodes: FxHashMap<usize, usize> =
        HashMap::with_capacity_and_hasher(2 * nodes.len(), FxBuildHasher);
    for n in nodes {
        // 1. Build mesh of cells touching node n
        let local_mesh = Selector::new(&near_mesh).nodes(false).id_in(&[n]).select();
        let (_, mut local_c2c) = compute_neighbours(&local_mesh, None, None);
        for edge in &cut_c2c {
            local_c2c.remove_edge(edge[0], edge[1]);
        }
        let compos = tarjan_scc(&local_c2c);
        // The node is not duplicated
        if compos.len() == 1 {
            continue;
        }
        // 2. Duplicate the node if there is more than one connex compo
        for compo in compos[1..].iter() {
            n2o_nodes.insert(new_node_id, n);
            for &eid in compo {
                let conn = near_mesh.element_mut(eid).connectivity;
                for c in conn.iter_mut() {
                    if *c == n {
                        *c = new_node_id;
                    }
                }
            }
            // TODO: Add the new node to the new coords array
            new_node_id += 1;
        }
    }

    todo!()
}
