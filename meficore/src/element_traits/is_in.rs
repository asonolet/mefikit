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
    let pd = ro::Coord3D {
        x: center[0] - r,
        y: center[1],
        z: center[2],
    };
    let pc = ro::Coord3D {
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

pub fn in_polygon(x: &[f64; 2], pgon: &[[f64; 2]]) -> bool {
    let px = x[0];
    let py = x[1];

    let n = pgon.len();
    if n < 3 {
        return false;
    }

    let mut inside = false;

    // Iterate edges
    for i in 0..n {
        let (x0, y0) = (pgon[i][0], pgon[i][1]);
        let (x1, y1) = (pgon[(i + 1) % n][0], pgon[(i + 1) % n][1]);

        // Check if edge straddles horizontal ray at py
        let cond = (y0 > py) != (y1 > py);
        if cond {
            // Compute intersection x coordinate
            let t = (py - y0) / (y1 - y0);
            let x_int = x0 + t * (x1 - x0);

            if px < x_int {
                inside = !inside;
            }
        }
    }

    inside
}

pub fn in_quadratic_polygon(x: &[f64; 2], pgon: &[[f64; 2]]) -> bool {
    let px = x[0];
    let py = x[1];

    let n = pgon.len() / 2;
    assert!(pgon.len().is_multiple_of(2));

    let vertices = &pgon[..n];
    let arcs = &pgon[n..];

    let mut inside = false;

    for i in 0..n {
        let p0 = vertices[i];
        let q = arcs[i];
        let p2 = vertices[(i + 1) % n];

        // Compute circle center
        let x0 = p0[0];
        let y0 = p0[1];
        let x1 = q[0];
        let y1 = q[1];
        let x2 = p2[0];
        let y2 = p2[1];

        let d = 2.0 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1));
        if d.abs() < 1e-14 {
            // Degenerate → treat as line segment
            continue;
        }

        let c_x = ((x0 * x0 + y0 * y0) * (y1 - y2)
            + (x1 * x1 + y1 * y1) * (y2 - y0)
            + (x2 * x2 + y2 * y2) * (y0 - y1))
            / d;

        let c_y = ((x0 * x0 + y0 * y0) * (x2 - x1)
            + (x1 * x1 + y1 * y1) * (x0 - x2)
            + (x2 * x2 + y2 * y2) * (x1 - x0))
            / d;

        let r2 = (x0 - c_x).powi(2) + (y0 - c_y).powi(2);
        let dy = py - c_y;
        let disc = r2 - dy * dy;

        if disc < 0.0 {
            continue;
        }

        let sqrt_d = disc.sqrt();
        let xs = [c_x - sqrt_d, c_x + sqrt_d];

        let theta0 = (y0 - c_y).atan2(x0 - c_x);
        let theta1 = (y1 - c_y).atan2(x1 - c_x);
        let theta2 = (y2 - c_y).atan2(x2 - c_x);

        for &xi in &xs {
            if xi <= px {
                continue;
            }

            let yi = py;
            let thetai = (yi - c_y).atan2(xi - c_x);

            let a = angle_between(theta0, theta2, thetai);
            let b = angle_between(theta0, theta2, theta1);

            if a == b {
                inside = !inside;
            }
        }
    }

    inside
}

fn angle_between(a: f64, b: f64, x: f64) -> bool {
    let mut ab = b - a;
    let mut ax = x - a;
    if ab < 0.0 {
        ab += 2.0 * std::f64::consts::TAU;
    }
    if ax < 0.0 {
        ax += 2.0 * std::f64::consts::TAU;
    }
    ax <= ab
}

pub fn in_bezier_polygon(x: &[f64; 2], pgon: &[[f64; 2]]) -> bool {
    let px = x[0];
    let py = x[1];

    let n = pgon.len() / 2;
    assert!(pgon.len().is_multiple_of(2));

    let vertices = &pgon[..n];
    let quads = &pgon[n..];

    let mut inside = false;

    for i in 0..n {
        let p0 = vertices[i];
        let p1 = quads[i];
        let p2 = vertices[(i + 1) % n];

        // Quadratic coefficients for y(t)
        let ay = p0[1] - 2.0 * p1[1] + p2[1];
        let by = 2.0 * (p1[1] - p0[1]);
        let cy = p0[1] - py;

        let scale = ay.abs().max(by.abs()).max(cy.abs()).max(1.0);
        let eps = 32.0 * f64::EPSILON * scale;

        // Solve ay*t^2 + by*t + cy = 0
        let mut roots = [0.0; 2];
        let mut count = 0;

        if ay.abs() < eps {
            // Linear case
            if by.abs() > eps {
                let t = -cy / by;
                if t > 0.0 && t <= 1.0 {
                    roots[0] = t;
                    count = 1;
                }
            }
        } else {
            let disc = by * by - 4.0 * ay * cy;
            if disc >= 0.0 {
                let s = disc.sqrt();
                let t1 = (-by - s) / (2.0 * ay);
                let t2 = (-by + s) / (2.0 * ay);

                if t1 > 0.0 && t1 <= 1.0 {
                    roots[count] = t1;
                    count += 1;
                }
                if t2 > 0.0 && t2 <= 1.0 {
                    roots[count] = t2;
                    count += 1;
                }
            }
        }

        for t in &roots[..count] {
            let mt = 1.0 - t;

            // Compute x(t)
            let xt = mt * mt * p0[0] + 2.0 * mt * t * p1[0] + t * t * p2[0];

            if xt > px {
                inside = !inside;
            }
        }
    }

    inside
}

pub fn point_in_phed(point: &[f64; 3], coords: &[[f64; 3]], connectivity: &[usize]) -> bool {
    let px = point[0];
    let py = point[1];
    let pz = point[2];

    let mut inside = false;

    let mut face_start = 0;
    let nconn = connectivity.len();

    while face_start < nconn {
        let mut face_end = face_start;
        while face_end < nconn && connectivity[face_end] != usize::MAX {
            face_end += 1;
        }

        // Face has at least 3 vertices
        if face_end - face_start >= 3 {
            let v0 = coords[connectivity[face_start]];

            // Fan triangulation: (v0, vi, vi+1)
            for i in face_start + 1..face_end - 1 {
                let v1 = coords[connectivity[i]];
                let v2 = coords[connectivity[i + 1]];

                if ray_intersects_triangle(px, py, pz, v0, v1, v2) {
                    inside = !inside;
                }
            }
        }

        face_start = face_end + 1;
    }

    inside
}

fn point_in_triangle_3d(
    px: f64,
    py: f64,
    pz: f64,
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> bool {
    let c0 = cross(
        [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
        [px - v0[0], py - v0[1], pz - v0[2]],
    );
    let c1 = cross(
        [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]],
        [px - v1[0], py - v1[1], pz - v1[2]],
    );
    let c2 = cross(
        [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]],
        [px - v2[0], py - v2[1], pz - v2[2]],
    );

    let d0 = dot(c0, c1);
    let d1 = dot(c1, c2);

    d0 >= 0.0 && d1 >= 0.0
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn ray_intersects_triangle(
    px: f64,
    py: f64,
    pz: f64,
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> bool {
    // Fast reject: all triangle vertices are behind the ray
    if v0[0] <= px && v1[0] <= px && v2[0] <= px {
        return false;
    }

    // Compute triangle normal
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    let n = [
        e1[1] * e2[2] - e1[2] * e2[1],
        e1[2] * e2[0] - e1[0] * e2[2],
        e1[0] * e2[1] - e1[1] * e2[0],
    ];

    let scale = e1[0].abs().max(e1[1].abs()).max(e1[2].abs())
        * e2[0].abs().max(e2[1].abs()).max(e2[2].abs());
    let eps = 64.0 * f64::EPSILON * scale.max(1.0);

    // Ray direction is (1, 0, 0)
    let denom = n[0];
    if denom.abs() < eps {
        // Triangle is parallel to ray
        return false;
    }

    let t = (n[0] * (v0[0] - px) + n[1] * (v0[1] - py) + n[2] * (v0[2] - pz)) / denom;

    if t <= 0.0 {
        return false;
    }

    // Intersection point
    let iy = py;
    let iz = pz;
    let ix = px + t;

    // Barycentric test in 3D (projected)
    point_in_triangle_3d(ix, iy, iz, v0, v1, v2)
}

pub fn point_in_phed2(point: &[f64; 3], coords: &[[f64; 3]], connectivity: &[usize]) -> bool {
    let px = point[0];
    let py = point[1];
    let pz = point[2];

    let mut inside = false;

    let mut face_start = 0;
    let nconn = connectivity.len();

    while face_start < nconn {
        let mut face_end = face_start;
        while face_end < nconn && connectivity[face_end] != usize::MAX {
            face_end += 1;
        }

        if face_end - face_start >= 3 {
            let v0 = coords[connectivity[face_start]];

            // Fan triangulation
            for i in face_start + 1..face_end - 1 {
                let v1 = coords[connectivity[i]];
                let v2 = coords[connectivity[i + 1]];

                if ray_intersects_triangle_half_open(px, py, pz, v0, v1, v2) {
                    inside = !inside;
                }
            }
        }

        face_start = face_end + 1;
    }

    inside
}

fn ray_intersects_triangle_half_open(
    px: f64,
    py: f64,
    pz: f64,
    v0: [f64; 3],
    v1: [f64; 3],
    v2: [f64; 3],
) -> bool {
    // Fast reject: triangle entirely behind ray
    if v0[0] <= px && v1[0] <= px && v2[0] <= px {
        return false;
    }

    // Edges
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];

    // Triangle normal
    let n = cross(e1, e2);

    // Ray direction = (1, 0, 0)
    let denom = n[0];

    // Geometry-scaled epsilon
    let scale = e1[0].abs().max(e1[1].abs()).max(e1[2].abs())
        * e2[0].abs().max(e2[1].abs()).max(e2[2].abs());

    let eps = 64.0 * f64::EPSILON * scale.max(1.0);

    if denom.abs() <= eps {
        // Triangle parallel to ray
        return false;
    }

    // Solve for intersection parameter t
    let t = (n[0] * (v0[0] - px) + n[1] * (v0[1] - py) + n[2] * (v0[2] - pz)) / denom;

    if t <= 0.0 {
        return false;
    }

    // Intersection point
    let ix = px + t;
    let iy = py;
    let iz = pz;

    // Half-open rule (symbolic perturbation)
    let ymin = v0[1].min(v1[1]).min(v2[1]);
    let ymax = v0[1].max(v1[1]).max(v2[1]);

    if !(iy > ymin && iy <= ymax) {
        return false;
    }

    let zmin = v0[2].min(v1[2]).min(v2[2]);
    let zmax = v0[2].max(v1[2]).max(v2[2]);

    if !(iz > zmin && iz <= zmax) {
        return false;
    }

    // Inside-triangle test
    point_in_triangle_3d(ix, iy, iz, v0, v1, v2)
}

#[cfg(test)]
mod tests {
    use super::in_polygon;
    use super::in_quadratic_polygon;

    fn square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    fn diamond() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]]
    }

    #[test]
    fn inside_diamond_point() {
        let pgon = diamond();
        let p = [0.0, 1.0];
        assert!(in_polygon(&p, &pgon));
    }

    #[test]
    fn outside_diamond_point() {
        let pgon = diamond();
        let p = [-2.0, 1.0];
        assert!(!in_polygon(&p, &pgon));
    }

    #[test]
    fn inside_point() {
        let pgon = square();
        let p = [0.5, 0.5];
        assert!(in_polygon(&p, &pgon));
    }

    #[test]
    fn outside_point() {
        let pgon = square();
        let p = [1.5, 0.5];
        assert!(!in_polygon(&p, &pgon));
    }

    #[test]
    fn outside_point_left() {
        let pgon = square();
        let p = [-1.5, 0.5];
        assert!(!in_polygon(&p, &pgon));
    }

    #[test]
    fn far_outside_point() {
        let pgon = square();
        let p = [10.0, -3.0];
        assert!(!in_polygon(&p, &pgon));
    }

    #[test]
    fn on_edge_horizontal() {
        let pgon = square();
        let p = [0.5, 0.0];
        // Parity ray-casting is undefined on boundary,
        // but this test ensures no panic / instability.
        let _ = in_polygon(&p, &pgon);
    }

    #[test]
    fn on_edge_vertical() {
        let pgon = square();
        let p = [1.0, 0.5];
        let _ = in_polygon(&p, &pgon);
    }

    #[test]
    fn on_vertex() {
        let pgon = square();
        let p = [0.0, 0.0];
        let _ = in_polygon(&p, &pgon);
    }

    #[test]
    fn concave_polygon_inside() {
        let pgon = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]];
        let p = [1.5, 1.5];
        assert!(in_polygon(&p, &pgon));
    }

    #[test]
    fn concave_polygon_outside() {
        let pgon = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]];
        let p = [0.75, 1.25];
        assert!(!in_polygon(&p, &pgon));
    }

    #[test]
    fn reversed_winding() {
        let mut pgon = square();
        pgon.reverse();

        let p = [0.5, 0.5];
        assert!(in_polygon(&p, &pgon));
    }

    fn quadratic_square() -> Vec<[f64; 2]> {
        // Vertices
        let v = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];

        // Quadratic control points (midpoints, slightly pushed outward)
        let q = vec![
            [0.5, -0.2], // bottom edge
            [1.2, 0.5],  // right edge
            [0.5, 1.2],  // top edge
            [-0.2, 0.5], // left edge
        ];

        [v, q].concat()
    }

    /// ```text
    ///  3
    ///  o__             __o 2
    ///  \  --__  6  __--  /
    ///   \     --+--     /
    ///  7 +             + 5
    ///    /    __+__     \
    ///   / __--  4   --__ \
    ///  o--              --o
    ///  0                  1
    ///  ```
    fn quadratic_concave() -> Vec<[f64; 2]> {
        let v = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];

        let q = vec![
            [1.0, 0.5], // bottom
            [1.5, 1.0], // right
            [1.0, 1.5], // top edge curves inward
            [0.5, 1.0], // left
        ];

        [v, q].concat()
    }

    #[test]
    fn inside_quadratic_square() {
        let pgon = quadratic_square();
        let p = [0.5, 0.5];
        assert!(in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn outside_quadratic_square() {
        let pgon = quadratic_square();
        let p = [1.5, 0.5];
        assert!(!in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn far_outside_quadratic_square() {
        let pgon = quadratic_square();
        let p = [10.0, -3.0];
        assert!(!in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn on_quadratic_edge_stability() {
        let pgon = quadratic_square();
        let p = [0.5, 0.0];
        // Boundary semantics are undefined;
        // this test ensures stability (no panic / NaN)
        let _ = in_quadratic_polygon(&p, &pgon);
    }

    #[test]
    fn inside_quadratic_concave() {
        let pgon = quadratic_concave();
        let p = [1., 1.];
        assert!(in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn outside_quadratic_concave1() {
        let pgon = quadratic_concave();
        let p = [1.0, 0.3];
        assert!(!in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn outside_quadratic_concave2() {
        let pgon = quadratic_concave();
        let p = [1.0, -0.3];
        assert!(!in_quadratic_polygon(&p, &pgon));
    }

    #[test]
    fn reversed_winding_quadratic() {
        let mut pgon = quadratic_square();
        let n = pgon.len() / 2;

        // reverse vertices
        pgon[..n].reverse();
        // reverse quadratic points to match edges
        pgon[n..].reverse();

        let p = [0.5, 0.5];
        assert!(in_quadratic_polygon(&p, &pgon));
    }
}
