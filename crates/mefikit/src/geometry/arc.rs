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
        breaks.dedup_by(|x, y| (*x - *y).abs() <= 1e-12);

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
}
