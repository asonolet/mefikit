use std::collections::HashMap;

use petgraph::Direction::Outgoing;
use petgraph::algo::tarjan_scc;
use petgraph::prelude::UnGraphMap;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

// This algorithm duplicates some nodes in order to break connectivites between some cells.
use crate::element_traits::SortedVecKey;
use crate::mesh::{Dimension, ElementId, ElementIds, ElementLike, UMesh, UMeshView};
use crate::tools::neighbours::{compute_neighbours_graph, compute_sub_to_elem};
use crate::tools::selector::{MeshSelect, sel};

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

fn build_subgraph(
    graph: &UnGraphMap<ElementId, SortedVecKey>,
    elements: &ElementIds,
) -> UnGraphMap<ElementId, SortedVecKey> {
    let elements: FxHashSet<ElementId> = elements.iter().collect();
    let mut subgraph: UnGraphMap<ElementId, SortedVecKey> = UnGraphMap::default();
    for e in &elements {
        subgraph.add_node(*e);
        for edge in graph.edges_directed(*e, Outgoing) {
            if elements.contains(&edge.1) {
                subgraph.add_edge(edge.0, edge.1, edge.2.clone());
            }
        }
    }
    subgraph
}

fn compute_node_to_elems(mesh: UMeshView) -> FxHashMap<usize, ElementIds> {
    let mut node_to_elem: FxHashMap<usize, ElementIds> =
        FxHashMap::with_capacity_and_hasher(mesh.used_nodes().len(), FxBuildHasher);
    for e in mesh.elements() {
        for n in e.connectivity().iter() {
            if let Some(elem_ids) = node_to_elem.get_mut(n) {
                elem_ids.add(e.element_type(), e.index());
            } else {
                node_to_elem.insert(*n, std::iter::once(e.id()).collect());
            }
        }
    }
    node_to_elem
}

pub fn crack(mut mesh: UMesh, cut: UMeshView) -> UMesh {
    // First extract the vicinity of the cut
    let nodes = cut.used_nodes();
    let (index, _) = mesh.select(sel::nids(nodes.clone(), false));
    let mut near_mesh = mesh.extract(&index);
    let (descending_mesh, f2c) = compute_sub_to_elem(&near_mesh, None, None);
    // Throws if some element in cut is not in descending_mesh
    let cut_ids = find_equals(descending_mesh.view(), cut.view());
    let cut_c2c: Vec<[ElementId; 2]> = cut_ids
        .into_iter()
        .map(|x| x.expect("cut elements should be found in mesh descending_mesh."))
        .filter(|f_id| f2c[f_id].len() == 2)
        .map(|f_id| f2c[&f_id].clone().try_into().unwrap())
        .collect();

    let mut near_c2c = compute_neighbours_graph(&near_mesh, None, None);
    for edge in &cut_c2c {
        near_c2c.remove_edge(edge[0], edge[1]);
    }

    let mut new_node_id = mesh.coords().nrows();
    // I am gessing a capacity here as I cannot know it in advance
    // let mut n2o_nodes: FxHashMap<usize, usize> =
    //     FxHashMap::with_capacity_and_hasher(2 * nodes.len(), FxBuildHasher);

    let node_to_elem: FxHashMap<usize, ElementIds> = compute_node_to_elems(near_mesh.view());

    for n in nodes {
        // 1. Build graph of cells touching node n
        let local_c2c = build_subgraph(&near_c2c, &node_to_elem[&n]);
        // 2. Find connex components
        let compos = tarjan_scc(&local_c2c);
        // The node is not duplicated
        if compos.len() <= 1 {
            continue;
        }
        // 3. Duplicate the node if there is more than one connex compo
        for compo in compos[1..].iter() {
            // n2o_nodes.insert(new_node_id, n);
            for &eid in compo {
                let conn = near_mesh.element_mut(eid).connectivity;
                for c in conn.iter_mut() {
                    if *c == n {
                        *c = new_node_id;
                        break;
                    }
                }
            }
            let new_coord = mesh.coords().row(n).into_owned();
            let _ = mesh.append_coord(new_coord.view());
            new_node_id += 1;
        }
    }
    mesh.replace(&index, near_mesh.view())
}
