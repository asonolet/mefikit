//! Circular arcs through three points — the shape of quadratic element edges.
//!
//! A quadratic edge joins vertex `p0` to vertex `p1` and is shaped by its mid-side node
//! `m`: the edge is the circular arc through the three points. When the three points are
//! collinear the edge degenerates to the straight segment `p0 -> p1`.
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

/// Characteristic length of a set of points: the largest pairwise distance.
///
/// Tolerances are derived from this length so all geometric predicates are translation and
/// scale aware.
fn points_scale(points: &[&[f64; 2]]) -> f64 {
    let mut scale = 0.0f64;
    for (i, a) in points.iter().enumerate() {
        for b in points.iter().skip(i + 1) {
            let dx = a[0] - b[0];
            let dy = a[1] - b[1];
            scale = scale.max(dx * dx + dy * dy);
        }
    }
    scale.sqrt()
}

#[inline(always)]
fn coord(p: &[f64; 2]) -> Coord<f64> {
    Coord { x: p[0], y: p[1] }
}

/// The circle through three points.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArcCircle {
    center: [f64; 2],
    r2: f64,
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
        // floating-point resolution of that interval are the same breakpoint.
        breaks
            .dedup_by(|x, y| (*x - *y).abs() <= 64.0 * (1.0 + x.abs().max(y.abs())) * f64::EPSILON);

        let cy = self.center[1];
        let cx = self.center[0];
        // At the arc's own endpoints use the exact node ordinates, and at interior breakpoints
        // the exact top/bottom of the circle.
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
    pub(crate) fn angle(&self, p: &[f64; 2]) -> f64 {
        (p[1] - self.center[1]).atan2(p[0] - self.center[0])
    }

    #[inline]
    pub(crate) fn center(&self) -> [f64; 2] {
        self.center
    }

    #[inline]
    pub(crate) fn radius(&self) -> f64 {
        self.r2.sqrt()
    }

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
    /// own sweep direction.
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

/// A quadratic edge: the circular arc through `p0`, `m` and `p1`.
///
/// Collinear triples are kept as straight edges; [`Self::circle`] is then `None`.
#[derive(Copy, Clone, Debug)]
pub(crate) struct ArcEdge {
    p0: [f64; 2],
    m: [f64; 2],
    p1: [f64; 2],
    circle: Option<ArcCircle>,
}

impl ArcEdge {
    /// Builds the curved edge through the three points.
    pub(crate) fn new(p0: [f64; 2], m: [f64; 2], p1: [f64; 2]) -> Self {
        Self {
            p0,
            m,
            p1,
            circle: ArcCircle::through(&p0, &m, &p1),
        }
    }

    /// Builds a straight edge between two points.
    pub(crate) fn straight(p0: [f64; 2], p1: [f64; 2]) -> Self {
        Self {
            p0,
            m: [0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1])],
            p1,
            circle: None,
        }
    }

    /// Whether the edge is a genuine circular arc (non-collinear triple).
    pub(crate) fn is_curved(&self) -> bool {
        self.circle.is_some()
    }

    /// Normalised parameter of `p` along the arc, or `None` for a straight edge.
    pub(crate) fn param(&self, p: &[f64; 2]) -> Option<f64> {
        self.circle
            .map(|circ| circ.param_along(&self.p0, &self.m, &self.p1, p))
    }

    /// Parameter of the parent mid-side node.
    pub(crate) fn mid_param(&self) -> Option<f64> {
        self.param(&self.m)
    }

    /// Outgoing direction at `from` along this edge towards `to`.
    ///
    /// For a circular arc this is the tangent at `from`, not the chord direction. The sign is
    /// chosen so the tangent points towards `to` along the arc sweep through the mid node.
    pub(crate) fn direction(&self, from: &[f64; 2], to: &[f64; 2]) -> [f64; 2] {
        let chord = [to[0] - from[0], to[1] - from[1]];
        let Some(circ) = self.circle else {
            return chord;
        };
        let sa = circ.param_along(&self.p0, &self.m, &self.p1, from);
        let sb = circ.param_along(&self.p0, &self.m, &self.p1, to);
        if sa == sb {
            return chord;
        }
        let sweep = circ.sweep(&self.p0, &self.m, &self.p1);
        let theta = circ.angle(from);
        let sign = if sweep >= 0.0 { 1.0 } else { -1.0 } * if sb > sa { 1.0 } else { -1.0 };
        [-theta.sin() * sign, theta.cos() * sign]
    }

    /// Point on the arc at the parameter midpoint between `a` and `b`, or `None` for a
    /// straight edge.
    pub(crate) fn sub_midpoint(&self, a: &[f64; 2], b: &[f64; 2]) -> Option<[f64; 2]> {
        self.circle
            .map(|circ| circ.midpoint_between(&self.p0, &self.m, &self.p1, a, b))
    }

    /// Axis-aligned bounding box of the whole arc, including tangent extrema.
    pub(crate) fn bounds(&self) -> [[f64; 2]; 2] {
        let mut bounds = [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]];
        let mut include = |p: [f64; 2]| {
            for k in 0..2 {
                bounds[0][k] = bounds[0][k].min(p[k]);
                bounds[1][k] = bounds[1][k].max(p[k]);
            }
        };
        include(self.p0);
        include(self.p1);
        include(self.m);
        if let Some(circ) = self.circle {
            let a0 = circ.angle(&self.p0);
            let sweep = circ.sweep(&self.p0, &self.m, &self.p1);
            let lo = a0.min(a0 + sweep);
            let hi = a0.max(a0 + sweep);
            let step = std::f64::consts::FRAC_PI_2;
            let mut k = (lo / step).floor() as i64 - 1;
            let last = (hi / step).ceil() as i64 + 1;
            while k <= last {
                let angle = k as f64 * step;
                if angle >= lo && angle <= hi {
                    let r = circ.radius();
                    let center = circ.center();
                    include([center[0] + r * angle.cos(), center[1] + r * angle.sin()]);
                }
                k += 1;
            }
        }
        bounds
    }
}

/// Length of the quadratic edge `p0 -> m -> p1`: the circular arc through the three
/// points, or the chord when they are collinear.
pub(crate) fn arc_length2(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> f64 {
    match ArcCircle::through(p0, m, p1) {
        Some(circ) => circ.r2.sqrt() * circ.sweep(p0, m, p1).abs(),
        None => (p1[0] - p0[0]).hypot(p1[1] - p0[1]),
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
    assert!(pgon.len().is_multiple_of(2));
    let n = pgon.len() / 2;
    signed_area2(&pgon[..n])
        + (0..n)
            .map(|i| arc_segment_area2(&pgon[i], &pgon[n + i], &pgon[(i + 1) % n]))
            .sum::<f64>()
}

/// Real roots of `a t^2 + b t + c = 0` (empty when there are none).
///
/// Near-zero discriminants are clamped to zero so a tangent line yields the touching point
/// despite floating-point roundoff.
fn quad_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let mut roots = Vec::new();
    if a <= 64.0 * f64::EPSILON * (b.abs().max(c.abs())).max(f64::MIN_POSITIVE) {
        if b.abs() > f64::MIN_POSITIVE {
            roots.push(-c / b);
        }
        return roots;
    }
    let disc = b * b - 4.0 * a * c;
    let disc_tol = 64.0 * f64::EPSILON * (b * b).max((4.0 * a * c).abs());
    if disc < -disc_tol {
        return roots;
    }
    let sq = disc.max(0.0).sqrt();
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
    let len_sq = ab[0] * ab[0] + ab[1] * ab[1];
    if len_sq <= eps * eps {
        return (p[0] - a[0]).hypot(p[1] - a[1]) <= eps;
    }
    let cross = ap[0] * bp[1] - ap[1] * bp[0];
    if cross.abs() > eps * len_sq.sqrt() {
        return false;
    }
    let dot = ap[0] * ab[0] + ap[1] * ab[1];
    dot >= -eps * len_sq.sqrt() && dot <= len_sq + eps * len_sq.sqrt()
}

/// Whether `pt` lies on the circular arc `p0 -> m -> p1`: on the circle within the length
/// tolerance `eps` and inside the angular sweep chosen through `m`.
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
    let a0 = circ.angle(p0);
    let apt = circ.angle(pt);
    let sweep = circ.sweep(p0, m, p1);
    let diff = (apt - a0).rem_euclid(TAU);
    let tol_ang = eps / r.max(eps);
    if sweep >= 0.0 {
        diff <= sweep + tol_ang
    } else {
        diff <= tol_ang || diff >= TAU + sweep - tol_ang
    }
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

/// Intersection points of the circular arc `p0 -> m -> p1` with the straight segment
/// `l0 -> l1`. A degenerate (collinear) arc falls back to the straight segment.
///
/// The result identifies intersections that fall on an endpoint of either operand
/// ([`Intersection::Existing`]); everything else is returned as coordinates
/// ([`Intersection::New`]). An arc and a segment meet in at most two points, so the result is
/// never [`Intersections::Segment`].
pub(crate) fn intersect_arc_seg(
    p0: &[f64; 2],
    m: &[f64; 2],
    p1: &[f64; 2],
    l0: &[f64; 2],
    l1: &[f64; 2],
) -> Intersections {
    let mut pts: Vec<[f64; 2]> = Vec::new();
    let scale = points_scale(&[p0, m, p1, l0, l1]);
    let eps = 64.0 * scale * f64::EPSILON;

    match ArcCircle::through(p0, m, p1) {
        None => {
            let res = super::segment::Segment::new(*p0, *p1)
                .intersect(super::segment::Segment::new(*l0, *l1));
            for pt in seg_int_coords(res, p0, p1, l0, l1) {
                dedup_push(&mut pts, pt, eps);
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
    fn collinear_midpoint_beyond_stays_straight() {
        let p0 = [0.0, 0.0];
        let m = [2.0, 0.0];
        let p1 = [1.0, 0.0];
        assert!((arc_length2(&p0, &m, &p1) - 1.0).abs() < TOL);
    }

    #[test]
    fn segment_area_sign_follows_bulge_side() {
        let p0 = [0.0, 0.0];
        let p1 = [1.0, 0.0];
        assert!(arc_segment_area2(&p0, &[0.5, -0.2], &p1) > 0.0);
        assert!(arc_segment_area2(&p0, &[0.5, 0.2], &p1) < 0.0);
    }

    #[test]
    fn bulged_square_area_is_square_plus_segments() {
        let v = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let m = [[0.5, -0.2], [1.2, 0.5], [0.5, 1.2], [-0.2, 0.5]];
        let mut pgon: Vec<[f64; 2]> = v.to_vec();
        pgon.extend_from_slice(&m);
        let seg = arc_segment_area2(&v[0], &m[0], &v[1]);
        assert!((quadratic_polygon_area2(&pgon) - (1.0 + 4.0 * seg)).abs() < TOL);
        assert!(seg > 0.0);
    }

    #[test]
    fn sweep_follows_mid_side_node() {
        let p0 = [1.0, 0.0];
        let p1 = [0.0, 1.0];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let circ = ArcCircle::through(&p0, &[s, s], &p1).unwrap();
        assert!(circ.sweep(&p0, &[s, s], &p1) > 0.0);
        let circ = ArcCircle::through(&p0, &[-s, -s], &p1).unwrap();
        assert!(circ.sweep(&p0, &[-s, -s], &p1) < 0.0);
    }

    #[test]
    fn arc_seg_two_intersections() {
        // Unit circle: the horizontal line y = 0.5 crosses the upper semicircle twice.
        let p0 = [-1.0, 0.0];
        let m = [0.0, 1.0];
        let p1 = [1.0, 0.0];
        let res = intersect_arc_seg(&p0, &m, &p1, &[-2.0, 0.5], &[2.0, 0.5]);
        let Intersections::Two([a, b]) = res else {
            panic!("expected two intersections, got {res:?}");
        };
        let coords = seg_int_coords(
            Intersections::Two([a, b]),
            &p0,
            &p1,
            &[-2.0, 0.5],
            &[2.0, 0.5],
        );
        assert_eq!(coords.len(), 2);
        for p in coords {
            assert!((p[1] - 0.5).abs() < 1e-12);
            assert!((p[0].abs() - 3f64.sqrt() / 2.0).abs() < 1e-12);
        }
    }

    #[test]
    fn arc_seg_tangent_is_one_point() {
        let p0 = [-1.0, 0.0];
        let m = [0.0, 1.0];
        let p1 = [1.0, 0.0];
        let res = intersect_arc_seg(&p0, &m, &p1, &[-2.0, 1.0], &[2.0, 1.0]);
        assert!(matches!(res, Intersections::One(Intersection::New(_))));
    }

    #[test]
    fn tangent_follows_circle() {
        let p0 = [1.0, 0.0];
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let edge = ArcEdge::new(p0, [s, s], [0.0, 1.0]);
        let dir = edge.direction(&p0, &[s, s]);
        assert!(dir[0].abs() < 1e-12 && (dir[1] - 1.0).abs() < 1e-12);
    }

    fn on_arc_tol(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], p: &[f64; 2]) -> bool {
        let scale = points_scale(&[p0, m, p1, p]);
        let eps = 64.0 * scale * f64::EPSILON;
        match ArcCircle::through(p0, m, p1) {
            Some(circ) => on_arc(&circ, p0, m, p1, p, eps),
            None => point_on_seg(p0, p1, p, eps),
        }
    }

    #[test]
    fn arc_bounds_include_bulge() {
        let edge = ArcEdge::new([0.0, 0.0], [0.5, -0.5], [1.0, 0.0]);
        let b = edge.bounds();
        assert!((b[0][1] + 0.5).abs() < 1e-12);
        assert!((b[1][1] - 0.0).abs() < 1e-12);
    }

    proptest! {
        #[test]
        fn arc_seg_intersections_lie_on_both(
            p0 in prop::array::uniform2(-2.0f64..2.0),
            p1 in prop::array::uniform2(-2.0f64..2.0),
            l0 in prop::array::uniform2(-2.0f64..2.0),
            l1 in prop::array::uniform2(-2.0f64..2.0),
            off in prop_oneof![-0.4f64..-0.02, 0.02f64..0.4],
        ) {
            let d = [p1[0] - p0[0], p1[1] - p0[1]];
            let len2 = d[0] * d[0] + d[1] * d[1];
            prop_assume!(len2 > 1e-4);
            let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
            let m = [mid[0] + off * d[1], mid[1] - off * d[0]];
            let res = intersect_arc_seg(&p0, &m, &p1, &l0, &l1);
            for p in seg_int_coords(res, &p0, &p1, &l0, &l1) {
                prop_assert!(on_arc_tol(&p0, &m, &p1, &p));
                prop_assert!(point_on_seg(&l0, &l1, &p, 1e-9));
            }
        }
    }
}
