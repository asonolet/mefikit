use crate::{UMesh, UMeshView};

use rstar::RTree;

fn snap_points(subject: UMesh, reference: UMeshView, eps: f64) {
    let mut coords = subject.coords;
    let ref_points: Vec<[f64; 3]> = reference
        .coords
        .rows()
        .into_iter()
        .map(|e| e.to_slice().unwrap().try_into().unwrap())
        .collect();
    let rtree: RTree<[f64; 3]> = RTree::bulk_load(ref_points);
    for coo in coords.rows() {
        let coord: [f64; 3] = coo.to_slice().unwrap().try_into().unwrap();
        let closest_points = rtree.locate_within_distance(coord, f64::powi(eps, 2));
        //TODO: compute the closest point from all matched points and replace subject coord with
        //this match.
    }
}
