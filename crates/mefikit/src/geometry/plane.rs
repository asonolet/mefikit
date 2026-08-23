//! Oriented planar frame with projection/deprojection between 3D and its local 2D basis.
//!
//! A [`PlaneFrame`] fits an oriented plane through a set of points (Newell's method) and
//! provides an orthonormal in-plane basis `(u, v, n)` with `u × v = n`. Projection preserves
//! orientation: a vertex ring winding counter-clockwise around `n` maps to a
//! counter-clockwise ring in the projected 2D coordinates.

use super::{newell_normal3, vertex_centroid};

/// An oriented plane with an orthonormal 2D frame.
///
/// The frame is defined by an origin point and three unit vectors: `u` and `v` spanning the
/// plane, and the normal `n = u × v`.
#[derive(Clone, Copy, Debug)]
pub struct PlaneFrame {
    origin: [f64; 3],
    u: [f64; 3],
    v: [f64; 3],
    n: [f64; 3],
}

impl PlaneFrame {
    /// Fits a plane through `points` and builds the associated frame.
    ///
    /// The plane normal is obtained with Newell's method and the origin is the vertex
    /// centroid. The in-plane axis `u` is chosen among the coordinate axes (the one least
    /// aligned with the normal) to build a well-conditioned basis.
    ///
    /// For a degenerate point set (collinear or repeated points) the normal is null and an
    /// arbitrary axis-aligned basis is returned; callers are expected to reject degenerate
    /// inputs beforehand (e.g. through an area test).
    pub fn from_points(points: &[[f64; 3]]) -> Self {
        let n_raw = newell_normal3(points);
        let norm = (n_raw[0] * n_raw[0] + n_raw[1] * n_raw[1] + n_raw[2] * n_raw[2]).sqrt();
        let mut n = if norm > 0.0 {
            [n_raw[0] / norm, n_raw[1] / norm, n_raw[2] / norm]
        } else {
            [0.0; 3]
        };

        // Pick the coordinate axis least aligned with the normal as a seed for `u`.
        let seed_axis = if n[0].abs() <= n[1].abs() && n[0].abs() <= n[2].abs() {
            [1.0, 0.0, 0.0]
        } else if n[1].abs() <= n[2].abs() {
            [0.0, 1.0, 0.0]
        } else {
            [0.0, 0.0, 1.0]
        };
        // Gram-Schmidt: remove the component along `n` and normalize.
        let dot = seed_axis[0] * n[0] + seed_axis[1] * n[1] + seed_axis[2] * n[2];
        let mut u = [
            seed_axis[0] - dot * n[0],
            seed_axis[1] - dot * n[1],
            seed_axis[2] - dot * n[2],
        ];
        let u_norm = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        // Fallback for a null normal (degenerate input).
        if u_norm == 0.0 || norm == 0.0 {
            n = [0.0, 0.0, 1.0];
            u = [1.0, 0.0, 0.0];
        } else {
            u = [u[0] / u_norm, u[1] / u_norm, u[2] / u_norm];
        }
        let v = [
            n[1] * u[2] - n[2] * u[1],
            n[2] * u[0] - n[0] * u[2],
            n[0] * u[1] - n[1] * u[0],
        ];

        Self {
            origin: vertex_centroid(points),
            u,
            v,
            n,
        }
    }

    /// Returns the frame origin (the vertex centroid used at construction).
    pub fn origin(&self) -> [f64; 3] {
        self.origin
    }

    /// Returns the unit normal of the frame (`u × v`).
    pub fn normal(&self) -> [f64; 3] {
        self.n
    }

    /// Returns the maximum absolute distance of `points` to the plane.
    ///
    /// This is the planarity deviation of the point set w.r.t. this frame.
    pub fn max_deviation(&self, points: &[[f64; 3]]) -> f64 {
        points
            .iter()
            .map(|p| {
                let d = [
                    p[0] - self.origin[0],
                    p[1] - self.origin[1],
                    p[2] - self.origin[2],
                ];
                (d[0] * self.n[0] + d[1] * self.n[1] + d[2] * self.n[2]).abs()
            })
            .fold(0.0, f64::max)
    }

    /// Projects a 3D point onto the plane local 2D basis.
    ///
    /// Orientation preserving: winding around [`Self::normal`] is kept.
    #[inline]
    pub fn project(&self, p: &[f64; 3]) -> [f64; 2] {
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        [
            d[0] * self.u[0] + d[1] * self.u[1] + d[2] * self.u[2],
            d[0] * self.v[0] + d[1] * self.v[1] + d[2] * self.v[2],
        ]
    }

    /// Deprojects a 2D point of the plane local basis back into 3D space.
    #[inline]
    pub fn deproject(&self, q: &[f64; 2]) -> [f64; 3] {
        [
            self.origin[0] + q[0] * self.u[0] + q[1] * self.v[0],
            self.origin[1] + q[0] * self.u[1] + q[1] * self.v[1],
            self.origin[2] + q[0] * self.u[2] + q[1] * self.v[2],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn project_deproject_roundtrip_xy_plane() {
        let pts = vec![[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]];
        let f = PlaneFrame::from_points(&pts);
        assert_abs_diff_eq!(f.normal()[0], 0.);
        assert_abs_diff_eq!(f.normal()[1], 0.);
        assert_abs_diff_eq!(f.normal()[2], 1.);
        for p in &pts {
            let q = f.project(p);
            assert_abs_diff_eq!(f.deproject(&q)[..], p[..]);
        }
    }

    #[test]
    fn project_deproject_roundtrip_arbitrary_plane() {
        // Four coplanar points on the oblique plane x + y + z = 3.
        let pts = vec![[1., 1., 1.], [2., 1., 0.], [1., 2., 0.], [0., 1., 2.]];
        assert_abs_diff_eq!(
            PlaneFrame::from_points(&pts).max_deviation(&pts),
            0.,
            epsilon = 1e-12
        );
        let f = PlaneFrame::from_points(&pts);
        for p in &pts {
            let q = f.project(p);
            assert_abs_diff_eq!(f.deproject(&q)[..], p[..], epsilon = 1e-12);
        }
        // Orthonormality.
        let dot_uv = f.u[0] * f.v[0] + f.u[1] * f.v[1] + f.u[2] * f.v[2];
        assert_abs_diff_eq!(dot_uv, 0., epsilon = 1e-12);
        let cross = [
            f.u[1] * f.v[2] - f.u[2] * f.v[1],
            f.u[2] * f.v[0] - f.u[0] * f.v[2],
            f.u[0] * f.v[1] - f.u[1] * f.v[0],
        ];
        for (ck, nk) in cross.iter().zip(&f.n) {
            assert_abs_diff_eq!(*ck, *nk, epsilon = 1e-12);
        }
    }

    #[test]
    fn projection_preserves_winding() {
        // CCW square seen from +z must stay CCW after projection.
        let pts = vec![[0., 0., 5.], [1., 0., 5.], [1., 1., 5.], [0., 1., 5.]];
        let f = PlaneFrame::from_points(&pts);
        let ring: Vec<[f64; 2]> = pts.iter().map(|p| f.project(p)).collect();
        let signed: f64 = (0..ring.len())
            .map(|i| {
                let a = ring[i];
                let b = ring[(i + 1) % ring.len()];
                a[0] * b[1] - b[0] * a[1]
            })
            .sum::<f64>()
            / 2.0;
        assert!(signed > 0.0);
    }

    #[test]
    fn max_deviation_detects_offplane_point() {
        let pts = vec![[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [0., 1., 0.]];
        let f = PlaneFrame::from_points(&pts);
        assert_abs_diff_eq!(f.max_deviation(&pts), 0.);
        let off = vec![[1., 0.5, 0.25]];
        assert_abs_diff_eq!(f.max_deviation(&off), 0.25);
    }
}
