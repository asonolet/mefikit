use nalgebra as na;
use todo;

pub fn in_sphere(x: &[f64], center: &[f64], r: f64) -> bool {
    let x = na::Point3::from_slice(x);
    let c = na::Point3::from_slice(center);
    let d2: f64 = na::distance_squared(&x, &c);
    d2 <= r * r
}

pub fn in_circle(x: &[f64], center: &[f64], r: f64) -> bool {
    let x = na::Point2::from_slice(x);
    let c = na::Point2::from_slice(center);
    let d2: f64 = na::distance_squared(&x, &c);
    d2 <= r * r
}

fn dist_proj_along(
    x: &na::Point3<f64>,
    center: &na::Point3<f64>,
    vec: &na::Unit<na::Vector3<f64>>,
) -> f64 {
    let v = x - center;
    vec.dot(&v)
}

fn dist_to_line(
    x: &na::Point3<f64>,
    center: &na::Point3<f64>,
    vec: &na::Unit<na::Vector3<f64>>,
) -> f64 {
    todo!()
}

fn project_plane(
    x: &na::Point3<f64>,
    center: &na::Point3<f64>,
    vec: &na::Vector3<f64>,
) -> na::Point2<f64> {
    todo!()
}

pub fn in_tube(x: &[f64], center: &[f64], vec: &[f64], r: f64) -> bool {
    let x = na::Point3::from_slice(x);
    let vec = na::Vector3::from_row_slice(vec);
    let center = na::Point3::from_slice(center);
    let xp = project_plane(&x, &center, &vec);
    let d2: f64 = na::distance_squared(&xp, &xp);
    d2 <= r * r
}

///  p0 is lower min bound and p1 higher max
pub fn in_aa_bbox(x: &[f64], p0: &[f64], p1: &[f64]) -> bool {
    let x = na::Point3::from_slice(x);
    let p0 = na::Point3::from_slice(p0);
    let p1 = na::Point3::from_slice(p1);
    if (x.x < p0.x)
        || (x.x >= p1.x)
        || (x.y < p0.y)
        || (x.y >= p1.y)
        || (x.z < p0.z)
        || (x.z >= p1.z)
    {
        false
    } else {
        true
    }
}

///  p0 is lower min bound and p1 higher max
pub fn in_aa_rectangle(x: &[f64], p0: &[f64], p1: &[f64]) -> bool {
    let x = na::Point2::from_slice(x);
    let p0 = na::Point2::from_slice(p0);
    let p1 = na::Point2::from_slice(p1);
    if (x.x < p0.x) || (x.x >= p1.x) || (x.y < p0.y) || (x.y >= p1.y) {
        false
    } else {
        true
    }
}
