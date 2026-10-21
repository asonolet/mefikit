//! Circular arcs through three points — the shape of quadratic element edges.
//!
//! A quadratic edge joins vertex `p0` to vertex `p1` and is shaped by its mid-side node
//! `m`: the edge is the circular arc through the three points. When the three points are
//! collinear the edge degenerates to the polyline `p0 -> m -> p1` (the straight segment
//! when `m` lies between the vertices).
//!
//! This module is pure geometry: it answers *where* curves meet (as coordinates /
//! [`Intersections`]) using exact orientation and in-circle predicates for the topological
//! decisions ([`robust::orient2d`], [`robust::incircle`]) and length-relative
//! floating-point tolerances for the numeric ones, following the same convention as
//! [`super::segment`]. Mesh-quality policy (snapping to shared nodes, discarding
//! near-coincident intersections, …) is the caller's responsibility.

use std::f64::consts::TAU;

use robust::{Coord, incircle, orient2d};

use super::segment::{Intersection, Intersections, PointId};
use super::signed_area2;

/// The circle through three points.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArcCircle {
    center: [f64; 2],
    r2: f64,
}

#[inline(always)]
fn coord(p: &[f64; 2]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

impl ArcCircle {
    /// The circle through `p0`, `m` and `p1`, or `None` when the three points are
    /// collinear at the scale of the edge.
    ///
    /// The centre is computed relative to `p0` (translation invariant). The degeneracy test
    /// uses the exact orientation determinant ([`robust::orient2d`]) compared against the
    /// edge's own squared scale, so there is no absolute epsilon.
    pub(crate) fn through(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> Option<Self> {
        let a = [m[0] - p0[0], m[1] - p0[1]];
        let b = [p1[0] - p0[0], p1[1] - p0[1]];
        let scale2 = (a[0] * a[0] + a[1] * a[1]).max(b[0] * b[0] + b[1] * b[1]);
        // Exact signed double area of the triangle (p0, m, p1).
        let cross = orient2d(coord(p0), coord(m), coord(p1));
        if cross.abs() <= 32.0 * f64::EPSILON * scale2 {
            return None;
        }
        let la = a[0] * a[0] + a[1] * a[1];
        let lb = b[0] * b[0] + b[1] * b[1];
        let d = 2.0 * cross;
        let center = [
            p0[0] + (la * b[1] - lb * a[1]) / d,
            p0[1] + (lb * a[0] - la * b[0]) / d,
        ];
        let r2 = (p0[0] - center[0]).powi(2) + (p0[1] - center[1]).powi(2);
        Some(Self { center, r2 })
    }

    /// Signed radial deviation of `pt` from the circle, as a length: negative inside, positive
    /// outside. Uses the exact in-circle determinant ([`robust::incircle`]) so the sign is
    /// exact even for nearly straight arcs; the determinant is converted to a length through
    /// `power / (2 r)` where `power` is the power of the point.
    fn signed_radial(&self, p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], pt: &[f64; 2]) -> f64 {
        // incircle(p0, m, p1, pt) = orient2d(p0, m, p1) * power(pt), with power = |pt-c|^2 - r^2.
        // Dividing by the orientation gives the power; dividing by 2r gives the radial length.
        let det = incircle(coord(p0), coord(m), coord(p1), coord(pt));
        let ori = orient2d(coord(p0), coord(m), coord(p1));
        if ori == 0.0 {
            return 0.0;
        }
        let r = self.r2.sqrt();
        if r == 0.0 {
            return 0.0;
        }
        let power = det / ori;
        power / (2.0 * r)
    }

    /// Signed angular sweep from `p0` to `p1` along the circle, in `(-tau, tau)`:
    /// positive counter-clockwise, choosing the direction that passes through `m`.
    pub(crate) fn sweep(&self, p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> f64 {
        let a0 = self.angle(p0);
        let ccw = (self.angle(p1) - a0).rem_euclid(TAU);
        if (self.angle(m) - a0).rem_euclid(TAU) <= ccw {
            ccw
        } else {
            ccw - TAU
        }
    }

    /// Whether the horizontal ray `(px, py) -> +inf` crosses the arc `p0 -> p1` through
    /// `m` an odd number of times. Points on the boundary are not classified.
    ///
    /// The arc is split at the interior points where its tangent is horizontal into
    /// y-monotone pieces, and each piece is tested with the same half-open rule as a straight
    /// segment (`min(y) <= py < max(y)`). This is what makes the parity robust: a vertex that
    /// lies exactly on the ray is counted by exactly one of its two incident edges, and a
    /// tangent touch (a horizontal extremum) cancels to zero instead of double counting.
    pub(crate) fn crosses_ray_right(
        &self,
        p0: &[f64; 2],
        m: &[f64; 2],
        p1: &[f64; 2],
        px: f64,
        py: f64,
    ) -> bool {
        if self.r2 <= 0.0 {
            return false;
        }
        let sweep = self.sweep(p0, m, p1);
        if sweep == 0.0 {
            return false;
        }
        let r = self.r2.sqrt();
        let a0 = self.angle(p0);
        let (lo_phi, hi_phi) = if sweep >= 0.0 {
            (a0, a0 + sweep)
        } else {
            (a0 + sweep, a0)
        };

        // Breakpoints in the normalised parameter `s in [0, 1]` at which `cos` vanishes.
        let pi = std::f64::consts::PI;
        let mut breaks: Vec<f64> = vec![0.0, 1.0];
        for k in (lo_phi / pi).floor() as i64 - 1..=(hi_phi / pi).ceil() as i64 + 1 {
            let phi = std::f64::consts::FRAC_PI_2 + (k as f64) * pi;
            let s = (phi - a0) / sweep;
            if s > 0.0 && s < 1.0 {
                breaks.push(s);
            }
        }
        breaks.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // `breaks` holds normalised parameters in `[0, 1]`; two breakpoints closer than the
        // floating-point resolution of that interval are the same breakpoint. This is a pure
        // roundoff tolerance, not a mesh-quality one, so it is derived solely from `f64::EPSILON`.
        breaks
            .dedup_by(|x, y| (*x - *y).abs() <= 64.0 * (1.0 + x.abs().max(y.abs())) * f64::EPSILON);

        let cy = self.center[1];
        let cx = self.center[0];
        // At the arc's own endpoints use the exact node ordinates, and at interior breakpoints
        // the exact top/bottom of the circle: recomputing an ordinate through the circle
        // (`cy + r * sin(angle)`) can land a rounding error on either side of `py` and
        // mis-count a vertex or a tangent extremum that lies exactly on the ray.
        let y_of = |s: f64| {
            if s == 0.0 {
                p0[1]
            } else if s == 1.0 {
                p1[1]
            } else if (a0 + sweep * s).sin() >= 0.0 {
                cy + r
            } else {
                cy - r
            }
        };
        let mut inside = false;
        for w in breaks.windows(2) {
            let (sa, sb) = (w[0], w[1]);
            if sb <= sa {
                continue;
            }
            // `cos` keeps its sign on the piece, which selects the near or far root.
            let cos_sign = if (a0 + sweep * (sa + sb) * 0.5).cos() < 0.0 {
                -1.0
            } else {
                1.0
            };
            let ya = y_of(sa);
            let yb = y_of(sb);
            let (ylo, yhi) = if ya <= yb { (ya, yb) } else { (yb, ya) };
            if py < ylo || py >= yhi {
                continue;
            }
            let dy = py - cy;
            let disc = self.r2 - dy * dy;
            if disc < 0.0 {
                continue;
            }
            let x = cx + cos_sign * disc.sqrt();
            if x > px {
                inside = !inside;
            }
        }
        inside
    }

    #[inline]
    fn angle(&self, p: &[f64; 2]) -> f64 {
        (p[1] - self.center[1]).atan2(p[0] - self.center[0])
    }
}

/// Length of the quadratic edge `p0 -> m -> p1`: the circular arc through the three
/// points, or the polyline through them when they are collinear.
pub(crate) fn arc_length2(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> f64 {
    match ArcCircle::through(p0, m, p1) {
        Some(circ) => circ.r2.sqrt() * circ.sweep(p0, m, p1).abs(),
        None => (m[0] - p0[0]).hypot(m[1] - p0[1]) + (p1[0] - m[0]).hypot(p1[1] - m[1]),
    }
}

/// Signed area between the chord `p0 -> p1` and the arc `p0 -> p1` through `m`:
/// positive when the arc bulges right of the chord, so it is added to the shoelace area
/// of a counter-clockwise polygon. Zero when the three points are collinear.
pub(crate) fn arc_segment_area2(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> f64 {
    match ArcCircle::through(p0, m, p1) {
        Some(circ) => {
            let sweep = circ.sweep(p0, m, p1);
            0.5 * circ.r2 * (sweep - sweep.sin())
        }
        None => 0.0,
    }
}

/// Exact area of the quadratic polygon given as `[v0..v{n-1}, m0..m{n-1}]`: the
/// shoelace area of the vertices plus the circular-segment area of every arc.
pub(crate) fn quadratic_polygon_area2(pgon: &[[f64; 2]]) -> f64 {
    let n = pgon.len() / 2;
    signed_area2(&pgon[..n])
        + (0..n)
            .map(|i| arc_segment_area2(&pgon[i], &pgon[n + i], &pgon[(i + 1) % n]))
            .sum::<f64>()
}

impl ArcCircle {
    /// Normalised parameter `s in [0, 1]` of the point `x` along the arc `p0 -> m -> p1`,
    /// measured in the arc's own sweep direction (the direction that passes through `m`).
    ///
    /// `x` is assumed to lie on the circle; the result is `0` at `p0` and `1` at `p1`.
    /// For a clock-wise sweep the parameter still increases from `p0` to `p1`.
    pub(crate) fn param_along(
        &self,
        p0: &[f64; 2],
        m: &[f64; 2],
        p1: &[f64; 2],
        x: &[f64; 2],
    ) -> f64 {
        let sweep = self.sweep(p0, m, p1);
        if sweep == 0.0 {
            return 0.0;
        }
        let a0 = self.angle(p0);
        let ccw = (self.angle(x) - a0).rem_euclid(TAU);
        if sweep > 0.0 {
            ccw / sweep
        } else {
            (TAU - ccw) / (-sweep)
        }
    }

    /// The point on the circle at the parameter midpoint between `a` and `b`, following the arc's
    /// own sweep direction. Used to give a sub-arc its own mid-side node when the parent mid does
    /// not fall strictly inside the sub-arc.
    pub(crate) fn midpoint_between(
        &self,
        p0: &[f64; 2],
        m: &[f64; 2],
        p1: &[f64; 2],
        a: &[f64; 2],
        b: &[f64; 2],
    ) -> [f64; 2] {
        let sa = self.param_along(p0, m, p1, a);
        let sb = self.param_along(p0, m, p1, b);
        let sm = 0.5 * (sa + sb);
        let a0 = self.angle(p0);
        let sweep = self.sweep(p0, m, p1);
        let ang = a0 + sweep * sm;
        let r = self.r2.sqrt();
        [
            self.center[0] + r * ang.cos(),
            self.center[1] + r * ang.sin(),
        ]
    }
}

/// Whether `pt` lies on the circular arc `p0 -> m -> p1` (within a floating-point tolerance),
/// treating a collinear edge as its polyline. Used by the overlay to tell which pieces of a cut
/// cell follow a curved boundary edge.
pub(crate) fn point_on_arc(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], pt: &[f64; 2]) -> bool {
    let mut scale = 1.0f64;
    for p in [p0, m, p1, pt] {
        scale = scale.max(p[0].abs()).max(p[1].abs());
    }
    let eps = 64.0 * scale * f64::EPSILON;
    match ArcCircle::through(p0, m, p1) {
        Some(circ) => on_arc(&circ, p0, m, p1, pt, eps),
        None => point_on_seg(p0, m, pt, eps) || point_on_seg(m, p1, pt, eps),
    }
}

/// Normalised parameter `s` of `pt` along the arc `p0 -> m -> p1` (see
/// [`ArcCircle::param_along`]); `None` when the edge is degenerate (collinear), in which case the
/// caller should treat it as straight.
pub(crate) fn arc_param(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], pt: &[f64; 2]) -> Option<f64> {
    ArcCircle::through(p0, m, p1).map(|circ| circ.param_along(p0, m, p1, pt))
}

/// The point on the arc `p0 -> m -> p1` at the parameter midpoint between `a` and `b`, giving a
/// sub-arc its own mid-side node. `None` when the edge is degenerate (collinear).
pub(crate) fn sub_arc_midpoint(
    p0: &[f64; 2],
    m: &[f64; 2],
    p1: &[f64; 2],
    a: &[f64; 2],
    b: &[f64; 2],
) -> Option<[f64; 2]> {
    ArcCircle::through(p0, m, p1).map(|circ| circ.midpoint_between(p0, m, p1, a, b))
}

/// Real roots of `a t^2 + b t + c = 0` (empty when there are none).
fn quad_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let mut roots = Vec::new();
    if a.abs() <= f64::MIN_POSITIVE {
        if b.abs() > f64::MIN_POSITIVE {
            roots.push(-c / b);
        }
        return roots;
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return roots;
    }
    let sq = disc.sqrt();
    roots.push((-b - sq) / (2.0 * a));
    roots.push((-b + sq) / (2.0 * a));
    roots
}

#[inline]
fn dedup_push(pts: &mut Vec<[f64; 2]>, pt: [f64; 2], eps: f64) {
    let eps2 = eps * eps;
    for p in pts.iter() {
        let dx = p[0] - pt[0];
        let dy = p[1] - pt[1];
        if dx * dx + dy * dy <= eps2 {
            return;
        }
    }
    pts.push(pt);
}

#[inline]
fn point_on_seg(a: &[f64; 2], b: &[f64; 2], p: &[f64; 2], eps: f64) -> bool {
    let ap = [p[0] - a[0], p[1] - a[1]];
    let bp = [p[0] - b[0], p[1] - b[1]];
    let ab = [b[0] - a[0], b[1] - a[1]];
    let cross = ap[0] * bp[1] - ap[1] * bp[0];
    if cross.abs() > eps {
        return false;
    }
    let dot = ap[0] * ab[0] + ap[1] * ab[1];
    if dot < 0.0 {
        return false;
    }
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    dot <= len_sq
}

/// Whether `pt` lies on the circular arc `p0 -> m -> p1`: on the circle within the length
/// tolerance `eps` and inside the angular sweep chosen through `m`.
///
/// `eps` is the radial (length) floating-point tolerance, derived from the coordinate
/// magnitude with the same convention as [`super::segment`]. The angular sweep is extended by
/// `eps / r`, the angular resolution implied by that same length tolerance.
fn on_arc(
    circ: &ArcCircle,
    p0: &[f64; 2],
    m: &[f64; 2],
    p1: &[f64; 2],
    pt: &[f64; 2],
    eps: f64,
) -> bool {
    if circ.signed_radial(p0, m, p1, pt).abs() > eps {
        return false;
    }
    let r = circ.r2.sqrt();
    let a0 = (p0[1] - circ.center[1]).atan2(p0[0] - circ.center[0]);
    let apt = (pt[1] - circ.center[1]).atan2(pt[0] - circ.center[0]);
    let sweep = circ.sweep(p0, m, p1);
    let diff = (apt - a0).rem_euclid(TAU);
    let tol_ang = eps / r.max(eps);
    if sweep >= 0.0 {
        diff <= sweep + tol_ang
    } else {
        diff <= tol_ang || diff >= TAU + sweep - tol_ang
    }
}

/// Intersection points of the circular arc `p0 -> m -> p1` with the straight segment
/// `l0 -> l1`. A degenerate (collinear) arc falls back to the polyline `p0 -> m -> p1`.
///
/// The result identifies intersections that fall on an endpoint of either operand
/// ([`Intersection::Existing`]); everything else is returned as coordinates
/// ([`Intersection::New`]). An arc and a segment meet in at most two points, so the result is
/// never [`Intersections::Segment`].
pub fn intersect_arc_seg(
    p0: &[f64; 2],
    m: &[f64; 2],
    p1: &[f64; 2],
    l0: &[f64; 2],
    l1: &[f64; 2],
) -> Intersections {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut scale = 1.0f64;
    for p in [p0, m, p1, l0, l1] {
        scale = scale.max(p[0].abs()).max(p[1].abs());
    }
    let eps = 64.0 * scale * f64::EPSILON;

    match ArcCircle::through(p0, m, p1) {
        None => {
            for (a, b) in [(p0, m), (m, p1)] {
                let res = super::segment::Segment::new(*a, *b)
                    .intersect(super::segment::Segment::new(*l0, *l1));
                for pt in seg_int_coords(res, a, b, l0, l1) {
                    dedup_push(&mut pts, pt, eps);
                }
            }
        }
        Some(circ) => {
            let d = [l1[0] - l0[0], l1[1] - l0[1]];
            let f = [l0[0] - circ.center[0], l0[1] - circ.center[1]];
            let qa = d[0] * d[0] + d[1] * d[1];
            let qb = 2.0 * (f[0] * d[0] + f[1] * d[1]);
            let qc = f[0] * f[0] + f[1] * f[1] - circ.r2;
            for t in quad_roots(qa, qb, qc) {
                let eps_t = 64.0 * (1.0 + t.abs()) * f64::EPSILON;
                if t >= -eps_t && t <= 1.0 + eps_t {
                    let t = t.clamp(0.0, 1.0);
                    let pt = [l0[0] + t * d[0], l0[1] + t * d[1]];
                    if on_arc(&circ, p0, m, p1, &pt, eps) {
                        dedup_push(&mut pts, pt, eps);
                    }
                }
            }
            for l in [l0, l1] {
                if on_arc(&circ, p0, m, p1, l, eps) {
                    dedup_push(&mut pts, *l, eps);
                }
            }
            for a in [p0, m, p1] {
                if point_on_seg(l0, l1, a, eps) {
                    dedup_push(&mut pts, *a, eps);
                }
            }
        }
    }
    classify_intersections(&pts, [p0, p1, l0, l1], eps)
}

/// Builds an [`Intersections`] from up to two intersection coordinates, mapping a coordinate
/// that coincides with one of the four operands' endpoints to [`Intersection::Existing`].
///
/// `endpoints` is `[P1, P2, P3, P4]`: the endpoints of the first operand (arc or segment),
/// then those of the second. Mid-side nodes are shape parameters, never endpoints.
fn classify_intersections(pts: &[[f64; 2]], endpoints: [&[f64; 2]; 4], eps: f64) -> Intersections {
    let ids = [PointId::P1, PointId::P2, PointId::P3, PointId::P4];
    let classify = |pt: [f64; 2]| -> Intersection {
        for (c, id) in endpoints.iter().zip(ids) {
            let dx = c[0] - pt[0];
            let dy = c[1] - pt[1];
            if dx * dx + dy * dy <= eps * eps {
                return Intersection::Existing(id);
            }
        }
        Intersection::New(pt)
    };
    match pts {
        [] => Intersections::None,
        [p] => Intersections::One(classify(*p)),
        [p, q, ..] => Intersections::Two([classify(*p), classify(*q)]),
    }
}

/// Intersection points of the circular arc `a0 -> am -> a1` with the circular
/// arc `b0 -> bm -> b1`. Degenerate (collinear) arcs fall back to their polyline.
///
/// The result identifies intersections that fall on an endpoint of either arc
/// ([`Intersection::Existing`], with `a0/a1` as `P1/P2` and `b0/b1` as `P3/P4`);
/// everything else is returned as coordinates. Two arcs meet in at most two points, so the
/// result is never [`Intersections::Segment`].
pub fn intersect_arc_arc(
    a0: &[f64; 2],
    am: &[f64; 2],
    a1: &[f64; 2],
    b0: &[f64; 2],
    bm: &[f64; 2],
    b1: &[f64; 2],
) -> Intersections {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let mut scale = 1.0f64;
    for p in [a0, am, a1, b0, bm, b1] {
        scale = scale.max(p[0].abs()).max(p[1].abs());
    }
    let eps = 64.0 * scale * f64::EPSILON;

    let ca = ArcCircle::through(a0, am, a1);
    let cb = ArcCircle::through(b0, bm, b1);

    match (ca, cb) {
        (None, None) | (None, Some(_)) => {
            for (x0, x1) in [(a0, am), (am, a1)] {
                for p in seg_int_coords(intersect_arc_seg(b0, bm, b1, x0, x1), b0, b1, x0, x1) {
                    dedup_push(&mut pts, p, eps);
                }
            }
        }
        (Some(_), None) => {
            for (x0, x1) in [(b0, bm), (bm, b1)] {
                for p in seg_int_coords(intersect_arc_seg(a0, am, a1, x0, x1), a0, a1, x0, x1) {
                    dedup_push(&mut pts, p, eps);
                }
            }
        }
        (Some(ca), Some(cb)) => {
            let dx = cb.center[0] - ca.center[0];
            let dy = cb.center[1] - ca.center[1];
            let d2 = dx * dx + dy * dy;
            let d = d2.sqrt();
            let ra = ca.r2.sqrt();
            let rb = cb.r2.sqrt();
            let eps_d = 64.0 * scale * f64::EPSILON;
            if d <= eps_d * (ra.max(rb).max(1.0)) {
                // Concentric: identical circles overlap everywhere, distinct radii never meet.
                // Discrete points are undefined here; treat as no intersections for imprint.
            } else {
                let a = (d2 + ra * ra - rb * rb) / (2.0 * d);
                let h2 = ra * ra - a * a;
                if h2 >= -eps * eps {
                    let h = h2.max(0.0).sqrt();
                    let xm = ca.center[0] + a * dx / d;
                    let ym = ca.center[1] + a * dy / d;
                    let rx = -dy * (h / d);
                    let ry = dx * (h / d);
                    let p1pt = [xm + rx, ym + ry];
                    if on_arc(&ca, a0, am, a1, &p1pt, eps) && on_arc(&cb, b0, bm, b1, &p1pt, eps) {
                        dedup_push(&mut pts, p1pt, eps);
                    }
                    if h > eps {
                        let p2pt = [xm - rx, ym - ry];
                        if on_arc(&ca, a0, am, a1, &p2pt, eps)
                            && on_arc(&cb, b0, bm, b1, &p2pt, eps)
                        {
                            dedup_push(&mut pts, p2pt, eps);
                        }
                    }
                }
            }
            // Explicit endpoint-on-other-arc checks (covers near-tangential/endpoint cases
            // the circle-circle solve may miss within tolerance).
            for (p, other) in [
                (a0, (&cb, b0, bm, b1)),
                (am, (&cb, b0, bm, b1)),
                (a1, (&cb, b0, bm, b1)),
            ] {
                if on_arc(other.0, other.1, other.2, other.3, p, eps) {
                    dedup_push(&mut pts, *p, eps);
                }
            }
            for (p, other) in [
                (b0, (&ca, a0, am, a1)),
                (bm, (&ca, a0, am, a1)),
                (b1, (&ca, a0, am, a1)),
            ] {
                if on_arc(other.0, other.1, other.2, other.3, p, eps) {
                    dedup_push(&mut pts, *p, eps);
                }
            }
        }
    }

    // Final dedup pass across all sources.
    let mut uniq: Vec<[f64; 2]> = Vec::with_capacity(pts.len());
    for p in pts {
        dedup_push(&mut uniq, p, eps);
    }
    classify_intersections(&uniq, [a0, a1, b0, b1], eps)
}

/// Converts an [`Intersections`] result between two specific straight segments into plain
/// coordinates (Existing ids are resolved against the endpoints used for that call).
fn seg_int_coords(
    res: Intersections,
    a1: &[f64; 2],
    a2: &[f64; 2],
    b1: &[f64; 2],
    b2: &[f64; 2],
) -> Vec<[f64; 2]> {
    let one = |i: Intersection| -> [f64; 2] {
        match i {
            Intersection::Existing(PointId::P1) => *a1,
            Intersection::Existing(PointId::P2) => *a2,
            Intersection::Existing(PointId::P3) => *b1,
            Intersection::Existing(PointId::P4) => *b2,
            Intersection::New(p) => p,
        }
    };
    let pid = |p: PointId| -> [f64; 2] {
        match p {
            PointId::P1 => *a1,
            PointId::P2 => *a2,
            PointId::P3 => *b1,
            PointId::P4 => *b2,
        }
    };
    match res {
        Intersections::None => vec![],
        Intersections::One(i) => vec![one(i)],
        Intersections::Two([i1, i2]) => vec![one(i1), one(i2)],
        Intersections::Segment([p1, p2]) => vec![pid(p1), pid(p2)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TOL: f64 = 1e-12;

    /// Unit quarter circle from `(1, 0)` to `(0, 1)` through the 45° point.
    fn quarter() -> ([f64; 2], [f64; 2], [f64; 2]) {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        ([1.0, 0.0], [s, s], [0.0, 1.0])
    }

    #[test]
    fn quarter_circle_length() {
        let (p0, m, p1) = quarter();
        assert!((arc_length2(&p0, &m, &p1) - std::f64::consts::FRAC_PI_2).abs() < TOL);
    }

    #[test]
    fn straight_midpoint_is_the_segment() {
        let p0 = [0.0, 0.0];
        let m = [0.5, 0.0];
        let p1 = [1.0, 0.0];
        assert!((arc_length2(&p0, &m, &p1) - 1.0).abs() < TOL);
        assert!(arc_segment_area2(&p0, &m, &p1).abs() < TOL);
    }

    #[test]
    fn collinear_midpoint_beyond_falls_back_to_polyline() {
        let p0 = [0.0, 0.0];
        let m = [2.0, 0.0];
        let p1 = [1.0, 0.0];
        assert!((arc_length2(&p0, &m, &p1) - 3.0).abs() < TOL);
    }

    #[test]
    fn segment_area_sign_follows_bulge_side() {
        // Right of the chord p0 -> p1 is positive, left is negative.
        let p0 = [0.0, 0.0];
        let p1 = [1.0, 0.0];
        assert!(arc_segment_area2(&p0, &[0.5, -0.2], &p1) > 0.0);
        assert!(arc_segment_area2(&p0, &[0.5, 0.2], &p1) < 0.0);
        // Symmetric bulges cancel out.
        let a = arc_segment_area2(&p0, &[0.5, -0.2], &p1);
        let b = arc_segment_area2(&p0, &[0.5, 0.2], &p1);
        assert!((a + b).abs() < TOL);
    }

    #[test]
    fn bulged_square_area_is_square_plus_segments() {
        // Unit square, each edge bulging outward by a symmetric amount: the exact area
        // exceeds 1 by four times the segment area of one edge.
        let v = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let m = [[0.5, -0.2], [1.2, 0.5], [0.5, 1.2], [-0.2, 0.5]];
        let mut pgon: Vec<[f64; 2]> = v.to_vec();
        pgon.extend_from_slice(&m);
        let seg = arc_segment_area2(&v[0], &m[0], &v[1]);
        assert!((quadratic_polygon_area2(&pgon) - (1.0 + 4.0 * seg)).abs() < TOL);
        assert!(seg > 0.0);
        assert!(quadratic_polygon_area2(&pgon) > 1.0);
    }

    #[test]
    fn sweep_follows_mid_side_node() {
        // CW and CCW placements of the mid-side node give opposite sweeps.
        let p0 = [1.0, 0.0];
        let p1 = [0.0, 1.0];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let circ = ArcCircle::through(&p0, &[s, s], &p1).unwrap();
        assert!(circ.sweep(&p0, &[s, s], &p1) > 0.0);
        let anti = [-s, -s];
        let circ = ArcCircle::through(&p0, &anti, &p1).unwrap();
        assert!(circ.sweep(&p0, &anti, &p1) < 0.0);
    }

    #[test]
    fn ray_crossing_parity_on_square_with_bulges() {
        let pgon = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.2],
            [1.2, 0.5],
            [0.5, 1.2],
            [-0.2, 0.5],
        ];
        // Segment-style classification through ArcCircle + in_quadratic_polygon agree
        // on a point well inside the bulged square.
        assert!(crate::geometry::in_quadratic_polygon(&[0.5, 0.5], &pgon));
        assert!(!crate::geometry::in_quadratic_polygon(&[1.5, 0.5], &pgon));
        // A point inside the bulge of the right edge is inside the shape.
        assert!(crate::geometry::in_quadratic_polygon(&[1.1, 0.5], &pgon));
        // A point inside the bulge of the bottom edge (but outside the chord) is inside
        // the shape; a point past the bulge is outside.
        assert!(crate::geometry::in_quadratic_polygon(&[0.5, -0.1], &pgon));
        assert!(!crate::geometry::in_quadratic_polygon(&[0.5, -0.3], &pgon));
    }

    #[test]
    fn tangent_bottom_extremum_is_not_a_crossing() {
        // A lens: the lower arc from (-1,0) to (1,0) through (0,-1), and the upper arc back
        // through (0,1). At `y = -1` the lower arc is tangent at its interior bottom extremum,
        // so a ray there must graze rather than cross: the left point is outside.
        let pgon = [[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]];
        assert!(crate::geometry::in_quadratic_polygon(&[0.0, 0.0], &pgon));
        assert!(crate::geometry::in_quadratic_polygon(&[0.0, -0.9], &pgon));
        assert!(!crate::geometry::in_quadratic_polygon(&[0.0, -1.1], &pgon));
        assert!(!crate::geometry::in_quadratic_polygon(&[-2.0, -1.0], &pgon));
        assert!(!crate::geometry::in_quadratic_polygon(&[2.0, 0.0], &pgon));
    }

    /// A random non-degenerate arc: a chord with the mid-side node offset along its normal by a
    /// bounded, non-vanishing fraction of the chord length.
    fn arb_arc() -> impl Strategy<Value = ([f64; 2], [f64; 2], [f64; 2])> {
        (
            prop::array::uniform2(-2.0f64..2.0),
            prop::array::uniform2(-2.0f64..2.0),
            prop_oneof![-0.4f64..-0.02, 0.02f64..0.4],
        )
            .prop_map(|(p0, p1, off)| {
                let d = [p1[0] - p0[0], p1[1] - p0[1]];
                let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
                (p0, [mid[0] + off * d[1], mid[1] - off * d[0]], p1)
            })
    }

    proptest! {
        /// The signed area of a circular segment is positive exactly when the mid-side node
        /// lies to the right of the chord `p0 -> p1` (i.e. the orientation is negative).
        #[test]
        fn segment_area_sign_matches_orientation((p0, m, p1) in arb_arc()) {
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let scale2 = d[0] * d[0] + d[1] * d[1];
            prop_assume!(scale2 > 1e-3);
            let cross = d[0] * (m[1] - p0[1]) - d[1] * (m[0] - p0[0]);
            prop_assume!(cross.abs() > 1e-4 * scale2);
            let area = arc_segment_area2(&p0, &m, &p1);
            prop_assert_eq!(area > 0.0, cross < 0.0);
            prop_assert_eq!(area < 0.0, cross > 0.0);
        }
    }

    proptest! {
        /// An arc is never shorter than its chord.
        #[test]
        fn arc_length_at_least_chord((p0, m, p1) in arb_arc()) {
            let chord = (p1[0] - p0[0]).hypot(p1[1] - p0[1]);
            prop_assume!(chord > 1e-3);
            let len = arc_length2(&p0, &m, &p1);
            prop_assert!(len >= chord - 1e-9 * chord.max(1.0));
        }
    }

    proptest! {
        /// Length and segment area are rigid (translation invariant) and homogeneous: length
        /// scales with `s`, area with `s^2`.
        #[test]
        fn arc_measures_are_rigid(
            (p0, m, p1) in arb_arc(),
            t in prop::array::uniform2(-1e6f64..1e6),
            s in 1e-3f64..1e3,
        ) {
            let len = arc_length2(&p0, &m, &p1);
            let area = arc_segment_area2(&p0, &m, &p1);

            let tp = |p: [f64; 2]| [p[0] + t[0], p[1] + t[1]];
            let sp = |p: [f64; 2]| [p[0] * s, p[1] * s];

            let len_t = arc_length2(&tp(p0), &tp(m), &tp(p1));
            let area_t = arc_segment_area2(&tp(p0), &tp(m), &tp(p1));
            let len_s = arc_length2(&sp(p0), &sp(m), &sp(p1));
            let area_s = arc_segment_area2(&sp(p0), &sp(m), &sp(p1));

            prop_assert!((len_t - len).abs() <= 1e-9 * len.abs().max(1.0));
            prop_assert!((area_t - area).abs() <= 1e-9 * area.abs().max(1.0));
            prop_assert!((len_s - len * s).abs() <= 1e-9 * (len * s).abs().max(1.0));
            prop_assert!((area_s - area * s * s).abs() <= 1e-9 * (area * s * s).abs().max(1.0));
        }
    }

    proptest! {
        /// With mid-side nodes exactly at the chord midpoints every segment area vanishes, so
        /// the exact quadratic area reduces to the shoelace area of the vertices.
        #[test]
        fn straight_mids_area_is_shoelace(
            verts in prop::collection::vec(prop::array::uniform2(-2.0f64..2.0), 3..=8),
        ) {
            let n = verts.len();
            let mut pgon = verts.clone();
            for i in 0..n {
                let p0 = verts[i];
                let p1 = verts[(i + 1) % n];
                pgon.push([(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]);
            }
            prop_assert_eq!(quadratic_polygon_area2(&pgon), signed_area2(&verts));
        }
    }

    #[allow(dead_code)]
    fn sample_arc(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], n: usize) -> Vec<[f64; 2]> {
        match ArcCircle::through(p0, m, p1) {
            Some(circ) => {
                let r = circ.r2.sqrt();
                let a0 = (p0[1] - circ.center[1]).atan2(p0[0] - circ.center[0]);
                let sweep = circ.sweep(p0, m, p1);
                (0..=n)
                    .map(|i| {
                        let t = i as f64 / n as f64;
                        let a = a0 + sweep * t;
                        [circ.center[0] + r * a.cos(), circ.center[1] + r * a.sin()]
                    })
                    .collect()
            }
            None => {
                // Degenerate: sample the polyline p0 -> m -> p1.
                let mut pts = Vec::with_capacity(2 * n + 1);
                for (a, b) in [(p0, m), (m, p1)] {
                    for i in 0..=n {
                        let t = i as f64 / n as f64;
                        pts.push([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
                    }
                }
                pts
            }
        }
    }

    #[allow(dead_code)]
    fn dist_point_to_seg(p: &[f64; 2], a: &[f64; 2], b: &[f64; 2]) -> f64 {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let ap = [p[0] - a[0], p[1] - a[1]];
        let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
        if len_sq <= f64::MIN_POSITIVE {
            return (ap[0] * ap[0] + ap[1] * ap[1]).sqrt();
        }
        let t = ((ap[0] * ab[0] + ap[1] * ab[1]) / len_sq).clamp(0.0, 1.0);
        let proj = [a[0] + t * ab[0], a[1] + t * ab[1]];
        let dx = p[0] - proj[0];
        let dy = p[1] - proj[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// The coordinates of an [`Intersections`] result, resolving `Existing` ids against the
    /// `[P1, P2, P3, P4]` endpoints supplied.
    fn int_coords(res: &Intersections, eps: [&[f64; 2]; 4]) -> Vec<[f64; 2]> {
        let one = |i: &Intersection| -> [f64; 2] {
            match i {
                Intersection::Existing(PointId::P1) => *eps[0],
                Intersection::Existing(PointId::P2) => *eps[1],
                Intersection::Existing(PointId::P3) => *eps[2],
                Intersection::Existing(PointId::P4) => *eps[3],
                Intersection::New(p) => *p,
            }
        };
        match res {
            Intersections::None => vec![],
            Intersections::One(i) => vec![one(i)],
            Intersections::Two([i1, i2]) => vec![one(i1), one(i2)],
            Intersections::Segment([p1, p2]) => {
                vec![
                    one(&Intersection::Existing(*p1)),
                    one(&Intersection::Existing(*p2)),
                ]
            }
        }
    }

    #[test]
    fn arc_seg_known_crossing() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let p0 = [1.0, 0.0];
        let m = [s, s];
        let p1 = [0.0, 1.0];
        let l0 = [0.5, -1.0];
        let l1 = [0.5, 1.0];
        let res = intersect_arc_seg(&p0, &m, &p1, &l0, &l1);
        let pts = int_coords(&res, [&p0, &p1, &l0, &l1]);
        assert_eq!(pts.len(), 1);
        let expect = [0.5, (1.0 - 0.25f64).sqrt()];
        assert!((pts[0][0] - expect[0]).abs() < 1e-12);
        assert!((pts[0][1] - expect[1]).abs() < 1e-12);
    }

    #[test]
    fn arc_seg_endpoint_contact_is_existing() {
        // A segment that ends exactly on the arc's endpoint `p0` must report `Existing(P1)`.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let p0 = [1.0, 0.0];
        let m = [s, s];
        let p1 = [0.0, 1.0];
        let l0 = [1.0, 1.0];
        let l1 = [1.0, 0.0];
        let res = intersect_arc_seg(&p0, &m, &p1, &l0, &l1);
        assert_eq!(res, Intersections::One(Intersection::Existing(PointId::P1)));
    }

    #[test]
    fn arc_arc_known_crossing() {
        // Unit circle arc (quarter, through 45deg) vs circle of radius 1 centred at (1, 0):
        // they intersect at (0.5, ±sqrt(3)/2); only the +one lies on the quarter arc.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let a0 = [1.0, 0.0];
        let am = [s, s];
        let a1 = [0.0, 1.0];
        // Second arc: circle centred (1,0) radius 1, arc from (0,0) through (1,-1) to (2,0)
        // i.e. the lower semicircle, which misses the quarter arc except possibly at endpoints.
        let b0 = [0.0, 0.0];
        let bm = [1.0, -1.0];
        let b1 = [2.0, 0.0];
        let res = intersect_arc_arc(&a0, &am, &a1, &b0, &bm, &b1);
        let pts = int_coords(&res, [&a0, &a1, &b0, &b1]);
        // The lower semicircle of circle(1,0,r=1) meets the unit-circle quarter arc only where
        // both circles meet within both sweeps: the circles themselves intersect at
        // (0.5, ±sqrt(3)/2); +sqrt(3)/2 ~ 0.866 is on the upper quarter arc (angle ~60deg),
        // but is it on the lower semicircle arc (sweep from angle 180deg through 270deg to 360deg
        // i.e. 180..360)? (0.5, 0.866) relative to center (1,0) is at angle ~120deg (upper-left),
        // NOT in 180..360 sweep. So no intersection is expected.
        assert!(pts.is_empty(), "expected no intersections, got {pts:?}");
    }

    /// Regression: an arc endpoint that lies on the segment's *line* but outside the segment
    /// must not be reported as an intersection. This arose when the left arc of a `QUAD8` cell
    /// `(0,0) -> (-0.1,0.5) -> (0,1)` met the segment `(0,-0.5) -> (0,0.5)`: the far endpoint
    /// `(0,1)` was wrongly reported, yielding a spurious `Intersections::Two`.
    #[test]
    fn arc_seg_ignores_arc_endpoint_outside_segment() {
        let p0 = [0.0, 0.0];
        let m = [-0.1, 0.5];
        let p1 = [0.0, 1.0];
        let l0 = [0.0, -0.5];
        let l1 = [0.0, 0.5];
        let res = intersect_arc_seg(&p0, &m, &p1, &l0, &l1);
        let pts = int_coords(&res, [&p0, &p1, &l0, &l1]);
        assert_eq!(pts.len(), 1, "expected a single crossing, got {pts:?}");
        assert!(
            (pts[0][0]).abs() < 1e-12 && (pts[0][1]).abs() < 1e-12,
            "expected the crossing at (0,0), got {pts:?}"
        );
    }

    proptest! {
        #[test]
        fn arc_seg_intersections_lie_on_both_curves(
            p0 in prop::array::uniform2(-2.0f64..2.0),
            m in prop::array::uniform2(-2.0f64..2.0),
            p1 in prop::array::uniform2(-2.0f64..2.0),
            l0 in prop::array::uniform2(-2.0f64..2.0),
            l1 in prop::array::uniform2(-2.0f64..2.0),
        ) {
            prop_assume!((p0[0]-p1[0]).hypot(p0[1]-p1[1]) > 1e-3);
            prop_assume!((l0[0]-l1[0]).hypot(l0[1]-l1[1]) > 1e-3);
            let res = intersect_arc_seg(&p0, &m, &p1, &l0, &l1);
            let pts = int_coords(&res, [&p0, &p1, &l0, &l1]);
            let mut scale = 1.0f64;
            for p in [&p0, &m, &p1, &l0, &l1] {
                scale = scale.max(p[0].abs()).max(p[1].abs());
            }
            let eps = 64.0 * scale * f64::EPSILON * 1e4; // looser tolerance for property check
            let circ = ArcCircle::through(&p0, &m, &p1);
            for pt in &pts {
                prop_assert!(point_on_seg(&l0, &l1, pt, eps), "point not on segment: {pt:?}");
                match &circ {
                    Some(c) => prop_assert!(on_arc(c, &p0, &m, &p1, pt, eps), "point not on arc: {pt:?}"),
                    None => {
                        let on_poly = point_on_seg(&p0, &m, pt, eps) || point_on_seg(&m, &p1, pt, eps);
                        prop_assert!(on_poly, "degenerate point not on polyline: {pt:?}");
                    }
                }
            }
        }
    }

    /// A point just past the end of the arc's sweep is rejected once the mesh-quality minimum-angle
    /// tolerance has been removed from [`on_arc`]: only floating-point slack remains.
    #[test]
    fn on_arc_rejects_point_just_outside_sweep() {
        let p0 = [1.0, 0.0];
        let m = [0.0, 1.0];
        let p1 = [-1.0, 0.0];
        let circ = ArcCircle::through(&p0, &m, &p1).unwrap();
        let ang = std::f64::consts::PI + 5e-7;
        let pt = [ang.cos(), ang.sin()];
        let r_eps = 64.0 * f64::EPSILON;
        assert!(!on_arc(&circ, &p0, &m, &p1, &pt, r_eps));
    }

    /// A point clearly off the arc must be rejected.
    #[test]
    fn on_arc_rejects_point_far_outside_sweep() {
        let p0 = [1.0, 0.0];
        let m = [0.0, 1.0];
        let p1 = [-1.0, 0.0];
        let circ = ArcCircle::through(&p0, &m, &p1).unwrap();
        let ang = std::f64::consts::PI + 0.1;
        let pt = [ang.cos(), ang.sin()];
        let r_eps = 64.0 * f64::EPSILON;
        assert!(!on_arc(&circ, &p0, &m, &p1, &pt, r_eps));
    }

    /// The arc/segment solver uses the same absolute `.max(1.0)` scale floor as
    /// [`super::segment`], so its tolerance stops shrinking once the geometry is smaller than
    /// unit size. The exact crossing must always be reported; only the unit-scale case is exactly
    /// a single point (below unit size the floor may additionally admit points within its
    /// absolute near-endpoint tolerance).
    #[test]
    fn arc_seg_intersection_with_unit_scale_floor() {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let p0 = [1.0, 0.0];
        let m = [s, s];
        let p1 = [0.0, 1.0];
        let l0 = [0.5, -1.0];
        let l1 = [0.5, 1.0];
        for k in [1e0, 1e-6, 1e-9] {
            let sp = |p: [f64; 2]| [p[0] * k, p[1] * k];
            let res = intersect_arc_seg(&sp(p0), &sp(m), &sp(p1), &sp(l0), &sp(l1));
            let pts = int_coords(&res, [&sp(p0), &sp(p1), &sp(l0), &sp(l1)]);
            let expect = [0.5 * k, (1.0 - 0.25f64).sqrt() * k];
            let tol = 1e-9 * k.abs().max(1.0);
            assert!(
                pts.iter()
                    .any(|pt| (pt[0] - expect[0]).abs() <= tol && (pt[1] - expect[1]).abs() <= tol),
                "true crossing not reported at scale {k}: {pts:?}"
            );
            if k >= 1.0 {
                assert_eq!(pts.len(), 1, "unexpected extra intersections at scale {k}");
            }
        }
    }
}
