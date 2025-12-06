use crate::umesh::{UMesh, UMeshView};

use nalgebra as na;
use rstar::RTree;

pub fn snap_points<const T: usize>(subject: UMesh, reference: UMeshView, eps: f64) {
    let mut coords = subject.coords;
    let ref_points: Vec<[f64; T]> = reference
        .coords
        .rows()
        .into_iter()
        .map(|e| e.to_slice().unwrap().try_into().unwrap())
        .collect();
    let rtree: RTree<[f64; T]> = RTree::bulk_load(ref_points);
    for mut coo in coords.rows_mut() {
        let coord: &mut [f64; T] = coo.as_slice_mut().unwrap().try_into().unwrap();
        let closest_points = rtree.locate_within_distance(*coord, f64::powi(eps, 2));
        //TODO: compute the closest point from all matched points and replace subject coord with
        //this match.
        let (_, closest) = closest_points
            .into_iter()
            .fold((f64::INFINITY, None), |acc, &p| {
                let (min_d2, closest_p) = acc;
                let closest_p = closest_p.unwrap();
                let na_p = p.into();
                let na_coord = (*coord).into();
                let d2 = na::distance_squared(&na_p, &na_coord);
                if d2 < min_d2 {
                    (d2, Some(p))
                } else {
                    (min_d2, Some(closest_p))
                }
            });
        if let Some(c) = closest {
            *coord = c
        }
    }
}
