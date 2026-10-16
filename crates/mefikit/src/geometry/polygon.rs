//! Owned polygons with area, centroid, convexity, and point-in-polygon tests.
//!
//! A [`Polygon`] stores the coordinates of its vertices in counter-clockwise order together with
//! its known [`Convexity`]. The convexity is used to select the fastest point-in-polygon test:
//! half-plane tests for convex polygons and ray casting for concave ones.

use itertools::Itertools;
use nalgebra as na;
use robust as ro;
use smallvec::SmallVec;
use std::sync::OnceLock;

use super::arc::ArcCircle;
use super::convexity::Convexity;

/// An owned polygon in `D`-dimensional space.
///
/// Vertices are stored in counter-clockwise order. Only `D = 2` and `D = 3` are supported.
#[derive(Clone, Debug)]
pub struct Polygon<const D: usize> {
    points: SmallVec<[[f64; D]; 4]>,
    convexity: Convexity,
    convexity_cache: OnceLock<bool>,
}

impl<const D: usize> Polygon<D> {
    /// Creates a polygon from the given vertices with an explicit convexity.
    pub fn with_convexity(
        points: impl IntoIterator<Item = [f64; D]>,
        convexity: Convexity,
    ) -> Self {
        Polygon {
            points: points.into_iter().collect(),
            convexity,
            convexity_cache: OnceLock::new(),
        }
    }

    /// Creates a polygon known to be convex.
    pub fn convex(points: impl IntoIterator<Item = [f64; D]>) -> Self {
        Self::with_convexity(points, Convexity::Convex)
    }

    /// Creates a polygon whose convexity is not known.
    pub fn unknown(points: impl IntoIterator<Item = [f64; D]>) -> Self {
        Self::with_convexity(points, Convexity::Unknown)
    }

    /// Returns the number of vertices.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the polygon has no vertices.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Iterates over the polygon vertices.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f64; D]> {
        self.points.iter()
    }

    /// Returns the axis-aligned bounding box as `[min, max]`.
    pub fn bounds(&self) -> [[f64; D]; 2] {
        bounds_iter(self.points.iter().copied())
    }

    /// Computes the vertex centroid of the polygon: the arithmetic mean of its vertices.
    ///
    /// This is the average of the node coordinates, not the area-weighted centroid (see
    /// [`Self::geometric_centroid`]).
    pub fn centroid(&self) -> [f64; D] {
        vertex_centroid(&self.points)
    }

    /// Returns the known convexity of the polygon.
    pub fn convexity(&self) -> Convexity {
        self.convexity
    }

    /// Returns `true` if the polygon is convex, computing the test on demand if the convexity was
    /// unknown.
    pub fn is_convex(&self) -> bool {
        match self.convexity {
            Convexity::Convex => true,
            Convexity::Concave => false,
            Convexity::Unknown => *self
                .convexity_cache
                .get_or_init(|| self.compute_convexity()),
        }
    }

    fn compute_convexity(&self) -> bool {
        let n = self.points.len();
        if n < 3 {
            return true;
        }
        let mut sign: f64 = 0.0;
        for i in 0..n {
            let o = polygon_orient(&self.points, i, (i + 1) % n, (i + 2) % n);
            if o != 0.0 {
                if sign != 0.0 && o.is_sign_positive() != sign.is_sign_positive() {
                    return false;
                }
                sign = o;
            }
        }
        true
    }
}

/// Exact sign of the 2D orientation `(b - a) × (c - a)` using adaptive-precision predicates
/// (Shewchuk via the `robust` crate).
///
/// Use this where the sign must be exact: convexity tests and boundary-sensitive containment.
/// For interior clipping/ear-selection where plain `f64` arithmetic is acceptable, use
/// [`cross2`] instead.
fn orient2d2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    ro::orient2d(
        ro::Coord { x: a[0], y: a[1] },
        ro::Coord { x: b[0], y: b[1] },
        ro::Coord { x: c[0], y: c[1] },
    )
}

/// Naive 2D cross product `(b - a) × (c - a)` in plain `f64` arithmetic.
///
/// Shared by the ear-clipping paths and the 2D polygon clipping in the conservative transfer.
/// Not exact near collinearity — prefer [`orient2d2`] when the sign is correctness-critical.
#[inline(always)]
pub(crate) fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Computes the area of a 2D triangle.
///
/// Bit-exact reproduction of the pre-refactoring `surf_tri2` formula.
#[inline(always)]
pub(crate) fn area_tri2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

/// Computes the area of a 2D quadrilateral.
///
/// Bit-exact reproduction of the pre-refactoring `surf_quad2` formula.
#[inline(always)]
pub(crate) fn area_quad2(p: &[[f64; 2]; 4]) -> f64 {
    let pxys = [
        p[0][0] * p[1][1],
        p[1][0] * p[2][1],
        p[2][0] * p[3][1],
        p[3][0] * p[0][1],
    ];
    let pyxs = [
        p[0][1] * p[1][0],
        p[1][1] * p[2][0],
        p[2][1] * p[3][0],
        p[3][1] * p[0][0],
    ];
    0.5 * (pxys.iter().sum::<f64>() - pyxs.iter().sum::<f64>()).abs()
}

/// Computes the un-normalized Newell normal of a planar polygon embedded in 3D space.
///
/// The result is the area vector: its magnitude is twice the polygon area and its direction
/// follows the right-hand rule around the vertex winding. For a degenerate (zero-area)
/// polygon the result is `[0.0, 0.0, 0.0]`.
///
/// Shared by [`Polygon::area`], [`Polygon::normal`] and the allocation-free element measure path.
pub(crate) fn newell_normal3(points: &[[f64; 3]]) -> [f64; 3] {
    let n = points.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        nx += a[1] * b[2] - a[2] * b[1];
        ny += a[2] * b[0] - a[0] * b[2];
        nz += a[0] * b[1] - a[1] * b[0];
    }
    [nx, ny, nz]
}

/// Computes the area of a planar polygon embedded in 3D space using Newell's method.
///
/// Shared by [`Polygon::area`] and the allocation-free element measure path.
pub(crate) fn area_polygon3(points: &[[f64; 3]]) -> f64 {
    let n = newell_normal3(points);
    0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
}

/// Computes the vertex centroid: the arithmetic mean of the given vertices.
///
/// Shared by [`Polygon::centroid`], [`crate::geometry::Polyhedron::centroid`] and the
/// allocation-free element centroid path.
pub(crate) fn vertex_centroid<const D: usize>(points: &[[f64; D]]) -> [f64; D] {
    let mut c = [0.0; D];
    for p in points {
        for (ck, pk) in c.iter_mut().zip(p.iter()) {
            *ck += *pk;
        }
    }
    let n = points.len() as f64;
    c.map(|v| v / n)
}

/// Returns `true` if `x` lies inside the convex polygon given in counter-clockwise order.
///
/// Half-plane test using exact orientation predicates; boundary semantics are not guaranteed.
/// Shared by [`Polygon::contains`] and the allocation-free element point-in-polygon path.
pub(crate) fn convex_polygon_contains2(points: &[[f64; 2]], x: &[f64; 2]) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    let mut sign: f64 = 0.0;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        let s = orient2d2(a, b, [x[0], x[1]]);
        if s != 0.0 {
            if sign != 0.0 && s.is_sign_positive() != sign.is_sign_positive() {
                return false;
            }
            sign = s;
        }
    }
    true
}

/// Signed area of a polygon in 2D using the shoelace formula.
///
/// Shared by [`Polygon::signed_area`] and the allocation-free element measure path.
pub(crate) fn signed_area2(points: &[[f64; 2]]) -> f64 {
    let n = points.len();
    let mut area2 = 0.0;
    for i in 0..n {
        let [x0, y0] = points[i];
        let [x1, y1] = points[(i + 1) % n];
        area2 += x0 * y1 - x1 * y0;
    }
    area2 / 2.0
}

/// Reverses the polygon in place when its winding is clockwise (negative signed area).
///
/// Shared by [`Polygon::into_ccw`] and the allocation-free element path.
pub(crate) fn into_ccw2(points: &mut [[f64; 2]]) {
    if signed_area2(points) < 0.0 {
        points.reverse();
    }
}

/// Axis-aligned bounding box of the given points as `[min, max]`.
///
/// Shared by [`Polygon::bounds`], [`crate::geometry::Polyhedron::bounds`] and the element
/// bounding-box paths.
pub(crate) fn bounds_iter<const D: usize>(
    points: impl IntoIterator<Item = [f64; D]>,
) -> [[f64; D]; 2] {
    points
        .into_iter()
        .fold([[f64::INFINITY; D], [-f64::INFINITY; D]], |acc, p| {
            let mut lo = acc[0];
            let mut hi = acc[1];
            for k in 0..D {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
            [lo, hi]
        })
}

/// Projects a point onto the dominant axis plane: the axis with the largest absolute coordinate
/// of the normal is dropped.
pub(crate) fn project2<const D: usize>(p: [f64; D], axis: usize) -> [f64; 2] {
    match axis {
        0 => [p[1], p[2]],
        1 => [p[0], p[2]],
        _ => [p[0], p[1]],
    }
}

pub(crate) fn newell_normal<const D: usize>(points: &[[f64; D]]) -> [f64; 3] {
    let mut n = [0.0; 3];
    let len = points.len();
    for i in 0..len {
        let a = points[i];
        let b = points[(i + 1) % len];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    n
}

/// Returns the index of the axis with the largest absolute component.
pub(crate) fn dominant_axis(n: [f64; 3]) -> usize {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        0
    } else if ay >= az {
        1
    } else {
        2
    }
}

/// Exact 2D orientation of the vertices `i, j, k` of `points`; for 3D input the vertices are
/// first projected onto the dominant axis plane of the Newell normal.
///
/// Exactness relies on [`orient2d2`], so the sign is robust for near-collinear and near-coplanar
/// configurations.
fn polygon_orient<const D: usize>(points: &[[f64; D]], i: usize, j: usize, k: usize) -> f64 {
    let a = points[i];
    let b = points[j];
    let c = points[k];
    match D {
        2 => orient2d2([a[0], a[1]], [b[0], b[1]], [c[0], c[1]]),
        3 => {
            let n = newell_normal(points);
            let axis = dominant_axis(n);
            orient2d2(project2(a, axis), project2(b, axis), project2(c, axis))
        }
        _ => panic!("Polygon only supports 2D or 3D coordinates"),
    }
}

/// Computes the squared distance of point `p` to the segment `[a, b]`, together with the clamped
/// projection parameter.
fn ortho_dist2(p: na::Point2<f64>, a: na::Point2<f64>, b: na::Point2<f64>) -> (f64, f64) {
    let ab = b - a;
    let ap = p - a;
    let t = ab.dot(&ap) / ab.dot(&ab);
    let tc = t.clamp(0.0, 1.0);
    let proj = a + tc * ab;
    (tc, (proj - p).norm_squared())
}

impl Polygon<2> {
    /// Computes the signed area using the shoelace formula.
    ///
    /// Positive result indicates counter-clockwise orientation.
    pub fn signed_area(&self) -> f64 {
        signed_area2(&self.points)
    }

    /// Computes the area of the polygon.
    pub fn area(&self) -> f64 {
        self.signed_area().abs()
    }

    /// Computes the geometric centroid of the polygon using the shoelace formula (area-weighted).
    ///
    /// This is the centroid of the area enclosed by the polygon, not the average of its vertices
    /// (see [`Self::centroid`]). Returns the first vertex for a degenerate polygon.
    pub fn geometric_centroid(&self) -> [f64; 2] {
        let n = self.points.len();
        let mut area2 = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        for i in 0..n {
            let [x0, y0] = self.points[i];
            let [x1, y1] = self.points[(i + 1) % n];
            let cross = x0 * y1 - x1 * y0;
            area2 += cross;
            cx += (x0 + x1) * cross;
            cy += (y0 + y1) * cross;
        }
        if area2.abs() < 1e-30 {
            return self.points[0];
        }
        [cx / (3.0 * area2), cy / (3.0 * area2)]
    }

    /// Returns the polygon vertices in counter-clockwise order.
    pub fn into_ccw(mut self) -> Self {
        into_ccw2(&mut self.points);
        self
    }

    /// Returns `true` if point `x` lies inside the polygon.
    ///
    /// Uses half-plane tests for convex polygons and ray casting for concave ones. Boundary
    /// semantics are not guaranteed.
    pub fn contains(&self, x: &[f64; 2]) -> bool {
        if self.is_convex() {
            self.convex_contains(x)
        } else {
            self.raycast_contains(x)
        }
    }

    fn convex_contains(&self, x: &[f64; 2]) -> bool {
        convex_polygon_contains2(&self.points, x)
    }

    fn raycast_contains(&self, x: &[f64; 2]) -> bool {
        let px = x[0];
        let py = x[1];

        let n = self.points.len();
        if n < 3 {
            return false;
        }

        let mut inside = false;

        // Iterate edges
        for i in 0..n {
            let p0 = &self.points[i];
            let p1 = &self.points[(i + 1) % n];
            if ray_crosses_segment(p0, p1, px, py) {
                inside = !inside;
            }
        }

        inside
    }

    /// Returns `true` if point `x` lies inside the polygon using the nearest point method.
    ///
    /// It might be slower than the ray casting method but is stable (no issue of near-crossing an
    /// edge which is far away).
    ///
    /// # Convention
    /// Polygon points are in anti-clockwise order.
    pub fn contains_stable(&self, x: &[f64; 2]) -> bool {
        let px = x[0];
        let py = x[1];

        enum Closest {
            Point(usize),
            OrthoProj(usize),
        }

        let n = self.points.len();
        if n < 3 {
            return false;
        }

        let mut min_dist2 = f64::INFINITY;
        let mut closest = Closest::Point(0);
        let p = na::Point2::new(px, py);
        // Iterate polygon points and get closest one.
        for (i, (a, b)) in self.points.iter().circular_tuple_windows().enumerate() {
            let pa = na::Point2::new(a[0], a[1]);
            let pb = na::Point2::new(b[0], b[1]);
            let (t, sqrt_dist) = ortho_dist2(p, pa, pb);
            if sqrt_dist < min_dist2 {
                min_dist2 = sqrt_dist;
                if t == 0.0 {
                    closest = Closest::Point(i);
                } else if t == 1.0 {
                    closest = Closest::Point((i + 1) % n);
                } else {
                    closest = Closest::OrthoProj(i);
                }
            }
        }

        match closest {
            Closest::Point(closest) => {
                let [xa, ya] = self.points[(closest + n - 1) % n];
                let [xb, yb] = self.points[closest];
                let [xc, yc] = self.points[(closest + 1) % n];
                let a = ro::Coord { x: xa, y: ya };
                let b = ro::Coord { x: xb, y: yb };
                let c = ro::Coord { x: xc, y: yc };
                let d = ro::Coord { x: px, y: py };
                if ro::orient2d(a, b, c) < 0. {
                    ro::orient2d(a, b, d) > 0. || ro::orient2d(b, c, d) > 0.
                } else {
                    ro::orient2d(a, b, d) > 0. && ro::orient2d(b, c, d) > 0.
                }
            }
            Closest::OrthoProj(closest) => {
                let [xa, ya] = self.points[closest];
                let [xb, yb] = self.points[(closest + 1) % n];
                let a = ro::Coord { x: xa, y: ya };
                let b = ro::Coord { x: xb, y: yb };
                let p = ro::Coord { x: px, y: py };
                ro::orient2d(a, b, p) > 0.
            }
        }
    }

    /// Returns a point strictly inside the polygon, or `None` if the polygon is degenerate
    /// (fewer than 3 points or zero area).
    ///
    /// The interior point is found with a horizontal scan-line at the midpoint of the widest gap
    /// between distinct vertex y-coordinates, so the line passes through no vertex. The edges
    /// strictly straddling the line give the x-coordinates of its crossings; the interior along the
    /// line is the union of the open segments between consecutive crossing pairs, and the returned
    /// point is the midpoint of the first such segment. It therefore lies strictly inside the
    /// polygon, never on a boundary edge or on a vertex line.
    pub fn strict_interior_point(&self) -> Option<[f64; 2]> {
        let n = self.points.len();
        if n < 3 {
            return None;
        }

        let mut area2 = 0.0;
        for i in 0..n {
            let [x0, y0] = self.points[i];
            let [x1, y1] = self.points[(i + 1) % n];
            area2 += x0 * y1 - x1 * y0;
        }
        if area2.abs() < 1e-30 {
            return None;
        }

        let mut ys: Vec<f64> = self.points.iter().map(|p| p[1]).collect();
        ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        ys.dedup();

        let mut y_mid = None;
        let mut widest = 0.0;
        for gap in ys.windows(2) {
            let g = gap[1] - gap[0];
            if g > widest {
                widest = g;
                y_mid = Some((gap[0] + gap[1]) / 2.0);
            }
        }
        let y_mid = y_mid?;

        let mut xs: Vec<f64> = Vec::with_capacity(n);
        for i in 0..n {
            let [x0, y0] = self.points[i];
            let [x1, y1] = self.points[(i + 1) % n];
            if (y0 < y_mid) != (y1 < y_mid) {
                xs.push(x0 + (y_mid - y0) * (x1 - x0) / (y1 - y0));
            }
        }
        if xs.len() < 2 {
            return None;
        }

        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some([(xs[0] + xs[1]) / 2.0, y_mid])
    }
}

impl Polygon<3> {
    /// Computes the area of the polygon using Newell's method.
    ///
    /// This is valid for planar polygons embedded in 3D space.
    pub fn area(&self) -> f64 {
        area_polygon3(&self.points)
    }

    /// Computes the unit normal of the planar polygon using Newell's method.
    ///
    /// The direction follows the right-hand rule around the vertex winding. Returns
    /// `[0.0, 0.0, 0.0]` for a degenerate (zero-area) polygon.
    pub fn normal(&self) -> [f64; 3] {
        let n = newell_normal3(&self.points);
        let norm = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        if norm == 0.0 {
            return [0.0; 3];
        }
        [n[0] / norm, n[1] / norm, n[2] / norm]
    }

    /// Computes the geometric centroid of the polygon as the area-weighted average of its
    /// fan-triangle centroids.
    ///
    /// This is the centroid of the area enclosed by the polygon, not the average of its vertices
    /// (see [`Self::centroid`]).
    pub fn geometric_centroid(&self) -> [f64; 3] {
        let n = self.points.len();
        if n < 3 {
            return self.points[0];
        }
        let v0 = self.points[0];
        let mut area_sum = 0.0;
        let mut c = [0.0; 3];
        for i in 1..n - 1 {
            let v1 = self.points[i];
            let v2 = self.points[i + 1];
            let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
            let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
            let cross = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            let a = 0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
            area_sum += a;
            for k in 0..3 {
                c[k] += a * (v0[k] + v1[k] + v2[k]) / 3.0;
            }
        }
        if area_sum < 1e-30 {
            return self.points[0];
        }
        [c[0] / area_sum, c[1] / area_sum, c[2] / area_sum]
    }
}

/// Returns `true` if point `x` is inside a quadratic polygon.
///
/// The polygon is specified as `[vertices..., mid-side nodes...]`, the layout of the
/// quadratic element types (TRI6, TRI7, QUAD8, QUAD9, QPOLY): edge `i` joins vertex `i`
/// to vertex `i + 1` and is the circular arc through its mid-side node. Collinear edges
/// fall back to the straight segment, so classification is always defined. Points on the
/// boundary are not classified.
pub fn in_quadratic_polygon(x: &[f64; 2], pgon: &[[f64; 2]]) -> bool {
    let n = pgon.len() / 2;
    assert!(pgon.len().is_multiple_of(2));

    let [px, py] = *x;
    let mut inside = false;
    for i in 0..n {
        let p0 = &pgon[i];
        let m = &pgon[n + i];
        let p1 = &pgon[(i + 1) % n];
        inside ^= match ArcCircle::through(p0, m, p1) {
            Some(circ) => circ.crosses_ray_right(p0, m, p1, px, py),
            None => ray_crosses_segment(p0, p1, px, py),
        };
    }
    inside
}

/// Whether the horizontal ray `(px, py) -> +inf` crosses the straight segment `p0 -> p1`.
#[inline]
fn ray_crosses_segment(p0: &[f64; 2], p1: &[f64; 2], px: f64, py: f64) -> bool {
    (p0[1] > py) != (p1[1] > py) && p0[0] + (py - p0[1]) / (p1[1] - p0[1]) * (p1[0] - p0[0]) > px
}

/// Returns `true` if point `x` is inside a Bezier polygon.
///
/// The polygon is specified as `[vertices..., Bezier_control_points...]`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;

    fn square() -> Polygon<2> {
        Polygon::unknown([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    }

    fn diamond() -> Polygon<2> {
        Polygon::unknown([[0.0, 0.0], [1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
    }

    /// Right triangle (counter-clockwise): interior = {x >= 0, y >= 0, x + y <= 1}.
    fn right_triangle() -> Polygon<2> {
        Polygon::unknown([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    }

    /// L-shaped polygon (counter-clockwise) with a reflex vertex at (1, 1).
    fn l_shape() -> Polygon<2> {
        Polygon::unknown([
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [0.0, 2.0],
        ])
    }

    fn ccw(p: Polygon<2>) -> Polygon<2> {
        p.into_ccw()
    }

    #[test]
    fn square_area() {
        assert_abs_diff_eq!(square().area(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn square_centroid() {
        let c = square().centroid();
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.5, epsilon = 1e-12);
        let g = square().geometric_centroid();
        assert_abs_diff_eq!(g[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn triangle_centroid() {
        let c = right_triangle().centroid();
        assert_abs_diff_eq!(c[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 1.0 / 3.0, epsilon = 1e-12);
        let g = right_triangle().geometric_centroid();
        assert_abs_diff_eq!(g[0], 1.0 / 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(g[1], 1.0 / 3.0, epsilon = 1e-12);
    }

    /// The geometric centroid of a concave polygon must lie inside its own interior (regression:
    /// the vertex centroid of an L-shape, the arithmetic mean of its nodes, falls in the notch).
    #[test]
    fn concave_geometric_centroid_inside() {
        let c = l_shape().geometric_centroid();
        let pgon = l_shape();
        assert!(
            pgon.contains_stable(&c),
            "geometric centroid {c:?} must be inside the L-shape"
        );
    }

    #[test]
    fn vertex_centroid_differs_from_geometric() {
        // The vertex centroid of the L-shape is the mean of its nodes: (1, 1), the reflex vertex.
        let vertex = l_shape().centroid();
        assert_eq!(vertex, [1.0, 1.0]);
        let geometric = l_shape().geometric_centroid();
        assert!(
            (vertex[0] - geometric[0]).abs() > 1e-9 && (vertex[1] - geometric[1]).abs() > 1e-9,
            "vertex centroid {vertex:?} must differ from geometric centroid {geometric:?}"
        );
    }

    #[test]
    fn convexity_detection() {
        assert!(square().is_convex());
        assert!(diamond().is_convex());
        assert!(right_triangle().is_convex());
        assert!(!l_shape().is_convex());
        let concave =
            Polygon::unknown([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]]);
        assert!(!concave.is_convex());
        assert!(!concave.is_convex(), "cache must be consistent");
    }

    #[test]
    fn known_convexity_shortcut() {
        let p = Polygon::convex([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        assert!(p.is_convex());
        let p = Polygon::with_convexity(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            Convexity::Concave,
        );
        assert!(!p.is_convex());
    }

    #[test]
    fn into_ccw_reverses_clockwise() {
        let cw = Polygon::unknown([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        assert!(cw.signed_area() < 0.0);
        let ccw = cw.into_ccw();
        assert!(ccw.signed_area() > 0.0);
    }

    #[test]
    fn outside_corner_diagonal() {
        let pgon = Polygon::unknown([
            [2.0 / 3.0, 1.0 / 3.0],
            [1.0, 1.0 / 3.0],
            [1.0, 2.0 / 3.0],
            [2.0 / 3.0, 2.0 / 3.0],
        ]);
        let p = [13.0 / 12.0, 1.0 / 3.0];
        assert!(
            !ccw(pgon.clone()).contains_stable(&p),
            "stable should be outside"
        );
        assert!(!pgon.contains(&p), "should be outside");
    }

    #[test]
    fn outside_right_of_cell() {
        let pgon = Polygon::unknown([[0.25, -0.25], [0.75, -0.25], [0.75, 0.25], [0.25, 0.25]]);
        let p = [0.875, 0.125];
        assert!(!ccw(pgon).contains_stable(&p), "stable should be outside");
    }

    #[test]
    fn outside_beyond_corner_above_edge() {
        let pgon = square();
        let p = [1.1, 0.05];
        assert!(
            !ccw(pgon.clone()).contains_stable(&p),
            "stable should be outside"
        );
        assert!(!pgon.contains(&p), "should be outside");
    }

    #[test]
    fn inside_near_corner() {
        let pgon = square();
        let p = [0.9, 0.1];
        assert!(
            ccw(pgon.clone()).contains_stable(&p),
            "stable should be inside"
        );
        assert!(pgon.contains(&p), "should be inside");
    }

    /// The closest feature of a point beyond an acute convex corner is the corner vertex.
    /// Such a point is outside the wedge unless it is to the left of both adjacent edges
    /// (this is the regression test for the inverted `&&`/`||` in the vertex wedge test).
    #[test]
    fn convex_acute_corner_exterior_wedge() {
        let pgon = right_triangle();
        // Beyond the apex (0, 1): to the left of edge (0,1)->(0,0) but right of edge (1,0)->(0,1).
        let p = [0.5, 1.5];
        assert!(
            !ccw(pgon.clone()).contains_stable(&p),
            "stable should be outside"
        );
        assert!(!pgon.contains(&p), "should be outside");
        // Beyond the corner (1, 0): to the left of edge (0,0)->(1,0) but right of edge (1,0)->(0,1).
        let p = [1.0, -1.0];
        assert!(!ccw(pgon).contains_stable(&p), "stable should be outside");
    }

    /// A point in the wide (reflex) wedge of a reflex vertex, on the side where it is left of
    /// only one of the two adjacent edges, must still be inside. Closest feature is the vertex.
    #[test]
    fn reflex_corner_interior_wedge() {
        let pgon = l_shape();
        // Below the reflex vertex (1, 1) of the L-shape: inside the horizontal bar.
        let p = [1.0, 0.6];
        assert!(
            ccw(pgon.clone()).contains_stable(&p),
            "stable should be inside"
        );
        assert!(pgon.contains(&p), "should be inside");
    }

    /// A point in the notch of a concave polygon, where the closest feature is an edge interior.
    #[test]
    fn reflex_corner_exterior_notch() {
        let pgon = l_shape();
        // The missing corner [1,2]x[1,2] of the L-shape.
        let p = [1.5, 1.5];
        assert!(
            !ccw(pgon.clone()).contains_stable(&p),
            "stable should be outside"
        );
        assert!(!pgon.contains(&p), "should be outside");
    }

    /// Point exactly aligned with a convex corner of an axis-aligned cell, beyond it, with the
    /// corner vertex as closest feature and the point to the right of both adjacent edges.
    #[test]
    fn convex_corner_exterior_aligned() {
        let pgon = square();
        let p = [1.0, -0.5];
        assert!(!ccw(pgon).contains_stable(&p), "stable should be outside");
    }

    #[test]
    fn inside_diamond_point() {
        let pgon = diamond();
        let p = [0.0, 1.0];
        assert!(pgon.contains(&p));
    }

    #[test]
    fn outside_diamond_point() {
        let pgon = diamond();
        let p = [-2.0, 1.0];
        assert!(!pgon.contains(&p));
    }

    #[test]
    fn inside_point() {
        let pgon = square();
        let p = [0.5, 0.5];
        assert!(pgon.contains(&p));
    }

    #[test]
    fn inside_point_stable() {
        let pgon = square();
        let p = [0.5, 0.5];
        assert!(ccw(pgon).contains_stable(&p));
    }

    #[test]
    fn outside_point_stable() {
        let pgon = square();
        let p = [2.5, 0.0];
        assert!(!ccw(pgon).contains_stable(&p));
    }

    #[test]
    fn outside_point() {
        let pgon = square();
        let p = [1.5, 0.5];
        assert!(!pgon.contains(&p));
    }

    #[test]
    fn outside_point_left() {
        let pgon = square();
        let p = [-1.5, 0.5];
        assert!(!pgon.contains(&p));
    }

    #[test]
    fn far_outside_point() {
        let pgon = square();
        let p = [10.0, -3.0];
        assert!(!pgon.contains(&p));
    }

    #[test]
    fn on_edge_horizontal() {
        let pgon = square();
        let p = [0.5, 0.0];
        // Parity ray-casting is undefined on boundary,
        // but this test ensures no panic / instability.
        let _ = pgon.contains(&p);
    }

    #[test]
    fn on_edge_vertical() {
        let pgon = square();
        let p = [1.0, 0.5];
        let _ = pgon.contains(&p);
    }

    #[test]
    fn on_vertex() {
        let pgon = square();
        let p = [0.0, 0.0];
        let _ = pgon.contains(&p);
    }

    #[test]
    fn concave_polygon_inside() {
        let pgon = Polygon::unknown([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]]);
        let p = [1.5, 1.5];
        assert!(pgon.contains(&p));
    }

    #[test]
    fn concave_polygon_outside() {
        let pgon = Polygon::unknown([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [1.0, 1.0], [0.0, 2.0]]);
        let p = [0.75, 1.25];
        assert!(!pgon.contains(&p));
    }

    #[test]
    fn reversed_winding() {
        let pgon = Polygon::unknown([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]]);
        let p = [0.5, 0.5];
        assert!(pgon.contains(&p));
    }

    /// `strict_interior_point` returns a point strictly inside a convex polygon.
    #[test]
    fn strict_interior_point_convex() {
        for pgon in [square(), diamond()] {
            let p = pgon
                .strict_interior_point()
                .expect("convex polygon has an interior point");
            assert!(
                ccw(pgon.clone()).contains_stable(&p),
                "interior point must be inside"
            );
        }
    }

    /// `strict_interior_point` returns a point strictly inside a concave polygon.
    #[test]
    fn strict_interior_point_concave() {
        let pgon = l_shape();
        let p = pgon
            .strict_interior_point()
            .expect("L-shape has an interior point");
        assert!(
            ccw(pgon.clone()).contains_stable(&p),
            "interior point must be inside"
        );
    }

    /// The non-convex L-shaped piece produced when a cell overlapping the `[0, 3]^2` boundary of
    /// the first mesh is cut must yield an interior point outside that mesh (`x > 3` or `y > 3`).
    /// Its centroid, in contrast, falls in the notch and lies inside `[0, 3]^2`, which is why the
    /// previous centroid-based classification dropped the piece from the union.
    #[test]
    fn strict_interior_point_notebook_l_piece() {
        let dec = 37.0 / 70.0;
        let pgon = Polygon::unknown([
            [3.0, 2.0 + dec],
            [2.5 + dec, 2.0 + dec],
            [2.5 + dec, 2.5 + dec],
            [2.0 + dec, 2.5 + dec],
            [2.0 + dec, 3.0],
            [3.0, 3.0],
        ]);
        let p = pgon
            .strict_interior_point()
            .expect("L-piece has an interior point");
        assert!(
            ccw(pgon.clone()).contains_stable(&p),
            "interior point must be inside the piece"
        );
        assert!(
            p[0] > 3.0 || p[1] > 3.0,
            "interior point must be outside the [0,3]^2 mesh, got {p:?}"
        );
    }

    /// Degenerate polygons have no strict interior point.
    #[test]
    fn strict_interior_point_degenerate() {
        assert_eq!(
            Polygon::unknown([] as [[f64; 2]; 0]).strict_interior_point(),
            None
        );
        assert_eq!(
            Polygon::unknown([[0.0, 0.0], [1.0, 1.0]]).strict_interior_point(),
            None
        );
        let flat = Polygon::unknown([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]);
        assert_eq!(flat.strict_interior_point(), None);
    }

    /// A planar 3D polygon (a quad) has the same area as its 2D counterpart.
    #[test]
    fn polygon3_area() {
        let p = Polygon::unknown([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        assert_abs_diff_eq!(p.area(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn polygon3_area_triangle() {
        let p = Polygon::unknown([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
        assert_abs_diff_eq!(p.area(), 0.5, epsilon = 1e-12);
    }

    /// Newell's method computes the area of a quad embedded in an oblique plane.
    #[test]
    fn polygon3_area_oblique_plane() {
        let p = Polygon::unknown([
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]);
        assert_abs_diff_eq!(p.area(), 2.0f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn polygon3_centroid() {
        let p = Polygon::unknown([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        let c = p.centroid();
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.5, epsilon = 1e-12);
        let g = p.geometric_centroid();
        assert_abs_diff_eq!(g[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn polygon3_is_convex() {
        let p = Polygon::unknown([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]);
        assert!(p.is_convex());
        let concave = Polygon::unknown([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
        ]);
        assert!(!concave.is_convex());
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

    #[test]
    fn bezier_polygon_basic() {
        // A Bezier polygon approximating a square with midpoints on the edges.
        let v = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let q = vec![[0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5]];
        let pgon = [v, q].concat();
        let p = [0.5, 0.5];
        assert!(in_bezier_polygon(&p, &pgon));
        let p = [2.0, 0.5];
        assert!(!in_bezier_polygon(&p, &pgon));
    }

    #[test]
    fn area_tri2_matches_polygon_area() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]];
        let poly = Polygon::unknown(pts);
        assert_eq!(
            area_tri2(&pts[0], &pts[1], &pts[2]),
            poly.area(),
            "TRI3 fast path must stay bit-exact with Polygon::area"
        );
    }

    #[test]
    fn area_quad2_matches_polygon_area() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let poly = Polygon::unknown(pts);
        assert_eq!(
            area_quad2(&pts),
            poly.area(),
            "QUAD4 fast path must stay bit-exact with Polygon::area"
        );
    }

    #[test]
    fn vertex_centroid_matches_polygon_centroid() {
        let pts = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let poly = Polygon::unknown(pts);
        assert_eq!(vertex_centroid(&pts), poly.centroid());
        let pts3 = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        assert_eq!(vertex_centroid(&pts3), [1.0, 1.0 / 3.0, 0.0]);
    }

    #[test]
    fn convex_polygon_contains2_matches_polygon_contains() {
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let quad = [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        for pts in [tri.as_slice(), quad.as_slice()] {
            let poly = Polygon::unknown(pts.to_vec());
            for p in [
                [0.5, 0.5],
                [0.1, 0.1],
                [0.9, 0.9],
                [0.0, 0.0],
                [1.0, 1.0],
                [1.5, 0.5],
                [-0.1, 0.5],
            ] {
                assert_eq!(
                    convex_polygon_contains2(pts, &p),
                    poly.contains(&p),
                    "convex fast path must match Polygon::contains for {p:?}"
                );
            }
        }
    }

    // -------------------------------------------------------------------------------------
    // Floating-point robustness properties for the ray-casting containment paths.
    //
    // The ray cast is an inherently floating-point computation: it locates a horizontal
    // crossing by interpolation and toggles a parity bit. These tests cross-check it against
    // oracles that are as exact as we can make them (`robust::orient2d` winding numbers, the
    // straight-chord fallback) and against rigid/affine invariance, and they deliberately
    // probe the cases where error is worst: vertices exactly on the ray, mid-side nodes
    // almost on their chords (where the arc centre escapes to infinity), and large
    // coordinate offsets. Points within a scale-aware tolerance of the boundary are skipped,
    // since no two correct implementations need to agree there.
    // -------------------------------------------------------------------------------------

    /// Largest coordinate magnitude seen in `pts` or `extra`, floored at `1.0`. Used to make
    /// tolerances scale-aware.
    fn coord_scale(pts: &[[f64; 2]], extra: [f64; 2]) -> f64 {
        let mut s = 1.0f64;
        for p in pts {
            s = s.max(p[0].abs()).max(p[1].abs());
        }
        s.max(extra[0].abs()).max(extra[1].abs())
    }

    /// Non-zero-winding containment using the exact orientation predicate. For simple
    /// polygons this agrees with the even-odd ray-casting rule, so it is a sound oracle.
    fn winding_contains(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
        let n = poly.len();
        let mut wn = 0i32;
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if a[1] <= p[1] {
                if b[1] > p[1] && orient2d2(a, b, p) > 0.0 {
                    wn += 1;
                }
            } else if b[1] <= p[1] && orient2d2(a, b, p) < 0.0 {
                wn -= 1;
            }
        }
        wn != 0
    }

    /// Euclidean distance from `p` to the segment `[a, b]`.
    fn dist_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let ap = [p[0] - a[0], p[1] - a[1]];
        let len2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if len2 > 0.0 {
            ((ap[0] * ab[0] + ap[1] * ab[1]) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dx = ap[0] - t * ab[0];
        let dy = ap[1] - t * ab[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// Distance from `p` to the closed polyline through `poly`.
    fn dist_to_boundary(poly: &[[f64; 2]], p: [f64; 2]) -> f64 {
        let n = poly.len();
        let mut d = f64::INFINITY;
        for i in 0..n {
            d = d.min(dist_to_segment(p, poly[i], poly[(i + 1) % n]));
        }
        d
    }

    /// A random star-shaped (hence simple) polygon: vertices sorted by angle around a random
    /// centre. This can be concave, so it exercises the ray-casting path rather than the
    /// convex fast path.
    fn star_polygon(min_n: usize, max_n: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        (min_n..=max_n).prop_flat_map(|n| {
            (
                prop::collection::vec(0.0f64..std::f64::consts::TAU, n),
                prop::collection::vec(0.3f64..2.0, n),
                prop::array::uniform2(-1.0f64..1.0),
            )
                .prop_map(|(angles, radii, center)| {
                    let mut pairs: Vec<(f64, f64)> = angles.into_iter().zip(radii).collect();
                    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                    pairs
                        .into_iter()
                        .map(|(a, r)| [center[0] + r * a.cos(), center[1] + r * a.sin()])
                        .collect()
                })
        })
    }

    /// Convex hull (monotone chain), returned counter-clockwise.
    fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
        pts.sort_by(|a, b| {
            a[0].partial_cmp(&b[0])
                .unwrap()
                .then(a[1].partial_cmp(&b[1]).unwrap())
        });
        pts.dedup();
        if pts.len() < 3 {
            return pts;
        }
        let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| {
            (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        };
        let mut lower: Vec<[f64; 2]> = Vec::new();
        for &p in &pts {
            while lower.len() >= 2
                && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0
            {
                lower.pop();
            }
            lower.push(p);
        }
        let mut upper: Vec<[f64; 2]> = Vec::new();
        for &p in pts.iter().rev() {
            while upper.len() >= 2
                && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0
            {
                upper.pop();
            }
            upper.push(p);
        }
        lower.pop();
        upper.pop();
        lower.extend(upper);
        lower
    }

    proptest! {
        /// The ray cast must agree with the exact winding test on a simple polygon whenever
        /// the query point is far enough from the boundary for the comparison to be meaningful.
        #[test]
        fn raycast_matches_exact_winding(
            poly in star_polygon(3, 8),
            q in prop::array::uniform2(-2.0f64..2.0),
        ) {
            let scale = coord_scale(&poly, q);
            prop_assume!(dist_to_boundary(&poly, q) > 64.0 * f64::EPSILON * scale);
            let got = Polygon::unknown(poly.clone()).raycast_contains(&q);
            prop_assert_eq!(got, winding_contains(&poly, q));
        }
    }

    proptest! {
        /// Classification is a geometric property, so translating the whole configuration must
        /// not change it — even when the translation moves coordinates to ~1e6, where the
        /// interpolated crossing absorbs a much larger absolute rounding error.
        #[test]
        fn raycast_is_translation_invariant(
            poly in star_polygon(3, 8),
            q in prop::array::uniform2(-2.0f64..2.0),
            t in prop::array::uniform2(-1e6f64..1e6),
        ) {
            let shifted: Vec<[f64; 2]> =
                poly.iter().map(|p| [p[0] + t[0], p[1] + t[1]]).collect();
            let qs = [q[0] + t[0], q[1] + t[1]];
            prop_assume!(dist_to_boundary(&poly, q) > 64.0 * f64::EPSILON * coord_scale(&shifted, qs));
            let before = Polygon::unknown(poly).raycast_contains(&q);
            let after = Polygon::unknown(shifted).raycast_contains(&qs);
            prop_assert_eq!(before, after);
        }
    }

    proptest! {
        /// Uniform scaling must not change the classification.
        #[test]
        fn raycast_is_scale_invariant(
            poly in star_polygon(3, 8),
            q in prop::array::uniform2(-2.0f64..2.0),
            s in 1e-3f64..1e3,
        ) {
            let scale = coord_scale(&poly, q);
            prop_assume!(dist_to_boundary(&poly, q) > 64.0 * f64::EPSILON * scale);
            let before = Polygon::unknown(poly.clone()).raycast_contains(&q);
            let scaled: Vec<[f64; 2]> = poly.iter().map(|p| [p[0] * s, p[1] * s]).collect();
            let after = Polygon::unknown(scaled).raycast_contains(&[q[0] * s, q[1] * s]);
            prop_assert_eq!(before, after);
        }
    }

    proptest! {
        /// The half-plane fast path (exact orientation predicates) and the ray cast must agree
        /// on a convex polygon away from the boundary.
        #[test]
        fn convex_fast_path_matches_raycast(
            pts in prop::collection::vec(prop::array::uniform2(-2.0f64..2.0), 3..=12),
            q in prop::array::uniform2(-3.0f64..3.0),
        ) {
            let hull = convex_hull(pts);
            prop_assume!(hull.len() >= 3);
            let scale = coord_scale(&hull, q);
            prop_assume!(dist_to_boundary(&hull, q) > 64.0 * f64::EPSILON * scale);
            let poly = Polygon::unknown(hull.clone());
            prop_assert_eq!(convex_polygon_contains2(&hull, &q), poly.raycast_contains(&q));
        }
    }

    /// Star-shaped vertices with one mid-side node per edge, offset along the chord normal by
    /// a bounded fraction of the chord length. The offset magnitude stays away from zero so the
    /// arcs are comfortably non-degenerate (near-degenerate arcs are covered separately).
    fn quadratic_polygon(min_n: usize, max_n: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        (
            star_polygon(min_n, max_n),
            prop::collection::vec(prop_oneof![-0.3f64..-0.05, 0.05f64..0.3], max_n),
        )
            .prop_map(|(verts, offsets)| {
                let n = verts.len();
                let mut out = verts.clone();
                for i in 0..n {
                    let p0 = verts[i];
                    let p1 = verts[(i + 1) % n];
                    let d = [p1[0] - p0[0], p1[1] - p0[1]];
                    let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
                    let off = offsets[i];
                    out.push([mid[0] + off * d[1], mid[1] - off * d[0]]);
                }
                out
            })
    }

    /// Circle through `p0`, `m`, `p1`, sampled into `k` equal-angle chords. Returns the
    /// polyline together with the maximum deviation between a chord and the true arc, or `None`
    /// when the three points are collinear at the edge's scale. This is an independent
    /// circumcentre computation, on purpose: it must not share code with the production arc.
    fn sample_arc(
        p0: [f64; 2],
        m: [f64; 2],
        p1: [f64; 2],
        k: usize,
    ) -> Option<(Vec<[f64; 2]>, f64)> {
        let (ax, ay) = (p0[0], p0[1]);
        let (bx, by) = (m[0], m[1]);
        let (cx, cy) = (p1[0], p1[1]);
        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        let scale2 =
            ((bx - ax).powi(2) + (by - ay).powi(2)).max((cx - ax).powi(2) + (cy - ay).powi(2));
        if d.abs() <= 64.0 * f64::EPSILON * scale2 {
            return None;
        }
        let a2 = ax * ax + ay * ay;
        let b2 = bx * bx + by * by;
        let c2 = cx * cx + cy * cy;
        let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
        let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
        let r = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();
        let a0 = (ay - uy).atan2(ax - ux);
        let am = (by - uy).atan2(bx - ux);
        let a1 = (cy - uy).atan2(cx - ux);
        let tau = std::f64::consts::TAU;
        let ccw = (a1 - a0).rem_euclid(tau);
        let sweep = if (am - a0).rem_euclid(tau) <= ccw {
            ccw
        } else {
            ccw - tau
        };
        let mut pts = Vec::with_capacity(k + 1);
        for j in 0..=k {
            let a = a0 + sweep * (j as f64) / (k as f64);
            pts.push([ux + r * a.cos(), uy + r * a.sin()]);
        }
        let gap = r * (1.0 - (sweep.abs() / (2.0 * k as f64)).cos());
        Some((pts, gap))
    }

    /// Dense boundary polyline of a quadratic polygon, with the largest chord deviation over
    /// all arcs.
    fn quadratic_boundary_polyline(pgon: &[[f64; 2]], k: usize) -> (Vec<[f64; 2]>, f64) {
        let n = pgon.len() / 2;
        let mut out = Vec::new();
        let mut gap = 0.0f64;
        for i in 0..n {
            let p0 = pgon[i];
            let m = pgon[n + i];
            let p1 = pgon[(i + 1) % n];
            match sample_arc(p0, m, p1, k) {
                Some((pts, g)) => {
                    gap = gap.max(g);
                    out.extend_from_slice(&pts[..pts.len() - 1]);
                }
                None => out.push(p0),
            }
        }
        (out, gap)
    }

    proptest! {
        /// With mid-side nodes exactly on the chords the arcs collapse to the straight
        /// segments, so the quadratic path must agree with the straight ray cast.
        #[test]
        fn straight_mids_match_straight_raycast(
            poly in star_polygon(3, 8),
            q in prop::array::uniform2(-2.0f64..2.0),
        ) {
            let n = poly.len();
            let mut pgon = poly.clone();
            for i in 0..n {
                let p0 = poly[i];
                let p1 = poly[(i + 1) % n];
                pgon.push([(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]);
            }
            let scale = coord_scale(&poly, q);
            prop_assume!(dist_to_boundary(&poly, q) > 64.0 * f64::EPSILON * scale);
            prop_assert_eq!(
                in_quadratic_polygon(&q, &pgon),
                Polygon::unknown(poly).raycast_contains(&q)
            );
        }
    }

    proptest! {
        /// The curved containment must agree with an exact winding test on a densely sampled
        /// boundary, once the query point is farther from the boundary than the sampling error.
        #[test]
        fn quadratic_matches_sampled_boundary(
            pgon in quadratic_polygon(3, 6),
            q in prop::array::uniform2(-2.5f64..2.5),
        ) {
            let (poly_pts, gap) = quadratic_boundary_polyline(&pgon, 64);
            prop_assume!(poly_pts.len() >= 3 && gap < 0.5);
            let scale = coord_scale(&pgon, q);
            prop_assume!(dist_to_boundary(&poly_pts, q) > 4.0 * gap + 64.0 * f64::EPSILON * scale);
            prop_assert_eq!(
                in_quadratic_polygon(&q, &pgon),
                winding_contains(&poly_pts, q)
            );
        }
    }

    proptest! {
        /// Curved containment is translation invariant, including at ~1e6 offsets.
        #[test]
        fn quadratic_is_translation_invariant(
            pgon in quadratic_polygon(3, 6),
            q in prop::array::uniform2(-2.5f64..2.5),
            t in prop::array::uniform2(-1e6f64..1e6),
        ) {
            let (poly_pts, gap) = quadratic_boundary_polyline(&pgon, 64);
            prop_assume!(poly_pts.len() >= 3 && gap < 0.5);
            let shifted: Vec<[f64; 2]> =
                pgon.iter().map(|p| [p[0] + t[0], p[1] + t[1]]).collect();
            let qs = [q[0] + t[0], q[1] + t[1]];
            prop_assume!(
                dist_to_boundary(&poly_pts, q)
                    > 4.0 * gap + 64.0 * f64::EPSILON * coord_scale(&shifted, qs)
            );
            prop_assert_eq!(
                in_quadratic_polygon(&q, &pgon),
                in_quadratic_polygon(&qs, &shifted)
            );
        }
    }

    proptest! {
        /// Mid-side nodes almost on their chords drive the arc centre to infinity and make
        /// `disc = r2 - dy^2` catastrophically cancelling. As long as the true sagitta stays
        /// below the skip margin, points away from the chord must still classify as if the
        /// edge were straight; a blow-up would show up as a disagreement far from the boundary.
        #[test]
        fn near_collinear_mids_stay_stable(
            poly in star_polygon(3, 8),
            q in prop::array::uniform2(-3.0f64..3.0),
            rel in 1e-12f64..1e-3,
            sign in proptest::bool::ANY,
        ) {
            let n = poly.len();
            let s = if sign { 1.0 } else { -1.0 };
            let mut pgon = poly.clone();
            let mut max_off = 0.0f64;
            for i in 0..n {
                let p0 = poly[i];
                let p1 = poly[(i + 1) % n];
                let d = [p1[0] - p0[0], p1[1] - p0[1]];
                let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
                let off = s * rel;
                max_off = max_off.max((off * d[0]).hypot(off * d[1]));
                pgon.push([mid[0] + off * d[1], mid[1] - off * d[0]]);
            }
            let scale = coord_scale(&poly, q);
            prop_assume!(dist_to_boundary(&poly, q) > 4.0 * max_off + 64.0 * f64::EPSILON * scale);
            prop_assert_eq!(in_quadratic_polygon(&q, &pgon), winding_contains(&poly, q));
        }
    }

    proptest! {
        /// A horizontal ray passing exactly through a vertex is the classic degenerate case
        /// for even-odd ray casting: whichever incident edge counts the vertex, it must be
        /// counted exactly once. The arc path must still match the robust reference.
        #[test]
        fn vertex_ray_matches_reference(
            pgon in quadratic_polygon(3, 6),
            idx in 0usize..6,
            side in proptest::bool::ANY,
            off in 0.05f64..2.5,
        ) {
            let n = pgon.len() / 2;
            let v = pgon[idx % n];
            let px = if side { v[0] + off } else { v[0] - off };
            let q = [px, v[1]];
            let (poly_pts, gap) = quadratic_boundary_polyline(&pgon, 128);
            prop_assume!(poly_pts.len() >= 3 && gap < 0.5);
            let scale = coord_scale(&pgon, q);
            prop_assume!(
                dist_to_boundary(&poly_pts, q) > 4.0 * gap + 64.0 * f64::EPSILON * scale
            );
            prop_assert_eq!(
                in_quadratic_polygon(&q, &pgon),
                winding_contains(&poly_pts, q)
            );
        }
    }
}
