//! Circular arcs through three points — the shape of quadratic element edges.
//!
//! A quadratic edge joins vertex `p0` to vertex `p1` and is shaped by its mid-side node
//! `m`: the edge is the circular arc through the three points. When the three points are
//! collinear the edge degenerates to the polyline `p0 -> m -> p1` (the straight segment
//! when `m` lies between the vertices).

use std::f64::consts::TAU;

use super::signed_area2;

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
    /// The centre is computed relative to `p0` (translation invariant) and the
    /// degeneracy test compares the doubled triangle area against the edge's own squared
    /// scale, so there is no absolute epsilon.
    pub(crate) fn through(p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2]) -> Option<Self> {
        let a = [m[0] - p0[0], m[1] - p0[1]];
        let b = [p1[0] - p0[0], p1[1] - p0[1]];
        let cross = a[0] * b[1] - a[1] * b[0];
        let scale2 = (a[0] * a[0] + a[1] * a[1]).max(b[0] * b[0] + b[1] * b[1]);
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
    pub(crate) fn crosses_ray_right(
        &self,
        p0: &[f64; 2],
        m: &[f64; 2],
        p1: &[f64; 2],
        px: f64,
        py: f64,
    ) -> bool {
        let dy = py - self.center[1];
        let disc = self.r2 - dy * dy;
        if disc <= 0.0 {
            return false;
        }
        let sx = disc.sqrt();
        [self.center[0] - sx, self.center[0] + sx]
            .into_iter()
            .filter(|&x| x > px)
            .filter(|&x| self.on_arc(p0, m, p1, dy.atan2(x - self.center[0])))
            .count()
            % 2
            == 1
    }

    /// Whether the angle `a` (about this circle's centre) lies on the arc from `p0` to
    /// `p1` through `m`.
    fn on_arc(&self, p0: &[f64; 2], m: &[f64; 2], p1: &[f64; 2], a: f64) -> bool {
        let a0 = self.angle(p0);
        let sweep = self.sweep(p0, m, p1);
        if sweep >= 0.0 {
            (a - a0).rem_euclid(TAU) <= sweep
        } else {
            (a0 - a).rem_euclid(TAU) <= -sweep
        }
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
