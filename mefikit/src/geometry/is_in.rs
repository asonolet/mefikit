use robust as ro;

pub fn in_sphere(x: &[f64; 3], center: &[f64; 3], r: f64) -> bool {
    let x = ro::Coord3D {
        x: x[0],
        y: x[1],
        z: x[2],
    };
    let pa = ro::Coord3D {
        x: center[0] + r,
        y: center[1],
        z: center[2],
    };
    let pb = ro::Coord3D {
        x: center[0],
        y: center[1] + r,
        z: center[2],
    };
    let pc = ro::Coord3D {
        x: center[0] - r,
        y: center[1],
        z: center[2],
    };
    let pd = ro::Coord3D {
        x: center[0],
        y: center[1],
        z: center[2] + r,
    };
    ro::insphere(pa, pb, pc, pd, x) > 0.0
}

pub fn in_circle(x: &[f64; 2], center: &[f64; 2], r: f64) -> bool {
    let x = ro::Coord { x: x[0], y: x[1] };
    let pa = ro::Coord {
        x: center[0] + r,
        y: center[1],
    };
    let pb = ro::Coord {
        x: center[0],
        y: center[1] + r,
    };
    let pc = ro::Coord {
        x: center[0] - r,
        y: center[1],
    };
    ro::incircle(pa, pb, pc, x) > 0.0
}

// fn dist_proj_along(
//     x: &na::Point3<f64>,
//     center: &na::Point3<f64>,
//     vec: &na::Unit<na::Vector3<f64>>,
// ) -> f64 {
//     let v = x - center;
//     vec.dot(&v)
// }

// fn dist_to_line(
//     x: &na::Point3<f64>,
//     center: &na::Point3<f64>,
//     vec: &na::Unit<na::Vector3<f64>>,
// ) -> f64 {
//     todo!()
// }

// fn project_plane(
//     x: &na::Point3<f64>,
//     center: &na::Point3<f64>,
//     vec: &na::Vector3<f64>,
// ) -> na::Point2<f64> {
//     todo!()
// }

// pub fn in_tube(x: &[f64], center: &[f64], vec: &[f64], r: f64) -> bool {
//     let x = na::Point3::from_slice(x);
//     let vec = na::Vector3::from_row_slice(vec);
//     let center = na::Point3::from_slice(center);
//     let xp = project_plane(&x, &center, &vec);
//     let d2: f64 = na::distance_squared(&xp, &xp);
//     d2 <= r * r
// }

///  p0 is lower min bound and p1 higher max
pub fn in_aa_bbox(x: &[f64; 3], p0: &[f64; 3], p1: &[f64; 3]) -> bool {
    !((x[0] < p0[0])
        || (x[0] >= p1[0])
        || (x[1] < p0[1])
        || (x[1] >= p1[1])
        || (x[2] < p0[2])
        || (x[2] >= p1[2]))
}

///  p0 is lower min bound and p1 higher max
pub fn in_aa_rectangle(x: &[f64; 2], p0: &[f64; 2], p1: &[f64; 2]) -> bool {
    !((x[0] < p0[0]) || (x[0] >= p1[0]) || (x[1] < p0[1]) || (x[1] >= p1[1]))
}
