use crate::mesh::{ElementLike, UMesh, UMeshView};

use itertools::Itertools;
use nalgebra as na;
use rstar::{RTree, primitives::GeomWithData};

fn snap_dim_n<const T: usize>(mut subject: UMesh, reference: UMeshView, eps: f64) -> UMesh {
    let ref_points: Vec<[f64; T]> = reference
        .used_nodes()
        .into_iter()
        .map(|i| {
            reference
                .coords()
                .row(i)
                .to_slice()
                .unwrap()
                .try_into()
                .unwrap()
        })
        .collect();
    let rtree = RTree::bulk_load(ref_points);
    for node in subject.used_nodes() {
        let coord: &mut [f64; T] = subject
            .coords
            .row_mut(node)
            .into_slice()
            .unwrap()
            .try_into()
            .unwrap();
        let closest_points = rtree.locate_within_distance(*coord, f64::powi(eps, 2));
        let (_, closest) = closest_points
            .into_iter()
            .fold((f64::INFINITY, None), |acc, &p| {
                let (min_d2, closest_p) = acc;
                let na_p = p.into();
                let na_coord = (*coord).into();
                let d2 = na::distance_squared(&na_p, &na_coord);
                if d2 < min_d2 {
                    (d2, Some(p))
                } else {
                    (min_d2, closest_p)
                }
            });
        if let Some(c) = closest {
            coord.copy_from_slice(&c)
        }
    }
    subject
}

/// Snap coords of subject mesh onto used nodes of reference.
///
/// Be careful, the method could produce degenerated elements if eps is not lower than half the
/// smallest distance between two points from the same element.
pub fn snap(subject: UMesh, reference: UMeshView, eps: f64) -> UMesh {
    match subject.coords().ncols() {
        1 => snap_dim_n::<1>(subject, reference, eps),
        2 => snap_dim_n::<2>(subject, reference, eps),
        3 => snap_dim_n::<3>(subject, reference, eps),
        _ => panic!("Could not snap the mesh because of its dimension."),
    }
}

//TODO: replace Vec<Vec<usize>> with proper IndirectIndex type.
// This would allow for cache friendly linear search of data.

fn duplicates_dim_n<const T: usize>(mesh: UMeshView, eps: f64) -> Vec<Vec<usize>> {
    let used_nodes = mesh.used_nodes();
    let points: Vec<GeomWithData<[f64; T], usize>> = used_nodes
        .iter()
        .map(|&i| {
            GeomWithData::new(
                mesh.coords().row(i).to_slice().unwrap().try_into().unwrap(),
                i,
            )
        })
        .collect();
    let mut rtree = RTree::bulk_load(points);
    let mut res: Vec<Vec<usize>> = Vec::new();
    for &node in &used_nodes {
        let coord: [f64; T] = mesh
            .coords()
            .row(node)
            .as_slice()
            .unwrap()
            .try_into()
            .unwrap();
        // Points are drained so they are not counted twice
        let closest_points = rtree.drain_within_distance(coord, f64::powi(eps, 2));
        let node_group: Vec<usize> = closest_points.map(|p| p.data).sorted_unstable().collect();
        if node_group.len() > 1 {
            res.push(node_group);
        }
    }
    res
}

pub fn duplicates(mesh: UMeshView, eps: f64) -> Vec<Vec<usize>> {
    match mesh.coords().ncols() {
        1 => duplicates_dim_n::<1>(mesh, eps),
        2 => duplicates_dim_n::<2>(mesh, eps),
        3 => duplicates_dim_n::<3>(mesh, eps),
        _ => panic!("Could not snap the mesh because of its dimension."),
    }
}

fn find_group(n: &usize, nodes: &[usize], groups: &[usize]) -> Option<usize> {
    match nodes.binary_search(n) {
        Ok(i) => Some(groups[i]),
        Err(_) => None,
    }
}

/// Merge close nodes.
///
/// Be careful, this method can produce degenerated elements if used with an epsilon greater than
/// the distance between two nodes of the same element.
pub fn merge_nodes(mut mesh: UMesh, eps: f64) -> UMesh {
    let dups = duplicates(mesh.view(), eps);
    let sorted_nodes_dup: Vec<(usize, usize)> = dups
        .iter()
        .enumerate()
        .flat_map(|(i, ns)| ns.iter().cloned().zip(std::iter::repeat(i)))
        .sorted_unstable()
        .collect();
    let sorted_nodes: Vec<usize> = sorted_nodes_dup.iter().map(|t| t.0).collect();
    let sorted_grps: Vec<usize> = sorted_nodes_dup.iter().map(|t| t.1).collect();
    // Here the idea is to go once throught each element and to renumber all nodes presents in
    // duplicates to the first node of the duplicates group.
    // I suppose that the number of duplicates is small in front of the number of elements so I
    // only go throught all elements once and thought the number of duplicates many times.
    // TODO: build a parallel version of the ElementMut iterator
    let eids: Vec<_> = mesh.elements().map(|e| e.id()).collect();
    for e in eids {
        let elem = mesh.element_mut(e);
        for n in elem.connectivity {
            if let Some(grp) = find_group(n, &sorted_nodes, &sorted_grps) {
                *n = dups[grp][0];
            }
        }
    }
    mesh
}
