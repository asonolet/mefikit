//! Generic convex clipping primitives used by the cross-dimensional conservative transfer.
//!
//! The helpers here operate on convex sets described by a list of half-spaces
//! `n · x <= d` (unit normals). They are deliberately dimension-generic: the same
//! Sutherland–Hodgman loop clips a 2D polygon by 2D half-planes and a 3D polygon by 3D
//! half-spaces, and the same segment clipping works in 2D and 3D.

/// A half-space `n · x <= d` with a unit normal `n`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct HalfSpace<const D: usize> {
    pub(crate) n: [f64; D],
    pub(crate) d: f64,
}

impl<const D: usize> HalfSpace<D> {
    /// Builds the half-space `n · x <= n · point`, normalizing `n`.
    pub(crate) fn new(n: [f64; D], point: [f64; D]) -> Self {
        let norm = n.iter().map(|x| x * x).sum::<f64>().sqrt();
        let inv = if norm > 0.0 { 1.0 / norm } else { 0.0 };
        let n = std::array::from_fn(|i| n[i] * inv);
        let d = (0..D).map(|i| n[i] * point[i]).sum();
        Self { n, d }
    }

    /// Signed distance to the boundary: positive outside, negative inside.
    pub(crate) fn distance(&self, point: &[f64; D]) -> f64 {
        (0..D).map(|i| self.n[i] * point[i]).sum::<f64>() - self.d
    }

    /// Reverses the half-space, keeping the same boundary plane.
    pub(crate) fn flipped(self) -> Self {
        Self {
            n: std::array::from_fn(|i| -self.n[i]),
            d: -self.d,
        }
    }
}

/// Clips a convex polygon by a list of half-spaces.
///
/// Vertices at a distance `<= eps` are considered inside, which keeps boundary-touching
/// intersections (shared faces, edges, vertices). Returns the clipped vertex ring, which may
/// contain fewer than three vertices when the intersection is empty or degenerate.
pub(crate) fn clip_polygon<const D: usize>(
    polygon: &[[f64; D]],
    half_spaces: &[HalfSpace<D>],
    eps: f64,
) -> Vec<[f64; D]> {
    let mut current = polygon.to_vec();
    for half_space in half_spaces {
        if current.len() < 3 {
            return Vec::new();
        }

        let mut all_inside = true;
        let mut all_outside = true;
        for vertex in &current {
            if half_space.distance(vertex) > eps {
                all_inside = false;
            } else {
                all_outside = false;
            }
        }
        if all_inside {
            continue;
        }
        if all_outside {
            return Vec::new();
        }

        let mut output = Vec::with_capacity(current.len() + 2);
        for k in 0..current.len() {
            let p = current[k];
            let q = current[(k + 1) % current.len()];
            let dp = half_space.distance(&p);
            let dq = half_space.distance(&q);
            let p_in = dp <= eps;
            let q_in = dq <= eps;
            match (p_in, q_in) {
                (true, true) => output.push(q),
                (true, false) | (false, true) => {
                    let denom = dp - dq;
                    if denom != 0.0 {
                        let t = dp / denom;
                        output.push(std::array::from_fn(|i| p[i] + t * (q[i] - p[i])));
                    }
                    if !p_in {
                        output.push(q);
                    }
                }
                (false, false) => {}
            }
        }
        current = output;
    }
    current
}

/// Clips a segment `[a, b]` by a list of half-spaces.
///
/// Returns the retained endpoints, or `None` when the segment does not overlap the
/// intersection of the half-spaces.
pub(crate) fn clip_segment<const D: usize>(
    a: [f64; D],
    b: [f64; D],
    half_spaces: &[HalfSpace<D>],
    eps: f64,
) -> Option<([f64; D], [f64; D])> {
    let delta: [f64; D] = std::array::from_fn(|i| b[i] - a[i]);
    let mut t0 = 0.0_f64;
    let mut t1 = 1.0_f64;

    for half_space in half_spaces {
        let da = half_space.distance(&a);
        let db = half_space.distance(&b);
        if da > eps && db > eps {
            return None;
        }
        if da <= eps && db <= eps {
            continue;
        }
        let denom = da - db;
        if denom == 0.0 {
            continue;
        }
        let t = da / denom;
        if da > eps {
            t0 = t0.max(t);
        } else {
            t1 = t1.min(t);
        }
        if t0 > t1 {
            return None;
        }
    }

    if t1 - t0 <= 0.0 {
        return None;
    }
    Some((
        std::array::from_fn(|i| a[i] + t0 * delta[i]),
        std::array::from_fn(|i| a[i] + t1 * delta[i]),
    ))
}

/// Returns `true` when `point` lies in the intersection of the half-spaces.
pub(crate) fn point_inside<const D: usize>(
    point: &[f64; D],
    half_spaces: &[HalfSpace<D>],
    eps: f64,
) -> bool {
    half_spaces
        .iter()
        .all(|half_space| half_space.distance(point) <= eps)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::geometry::signed_area2;

    fn square() -> Vec<[f64; 2]> {
        vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    }

    #[test]
    fn half_space_distance_is_signed() {
        let hs = HalfSpace::new([1.0, 0.0], [0.5, 0.0]);
        assert_abs_diff_eq!(hs.distance(&[0.0, 0.0]), -0.5);
        assert_abs_diff_eq!(hs.distance(&[0.5, 0.25]), 0.0);
        assert_abs_diff_eq!(hs.distance(&[1.0, 0.0]), 0.5);
        let flipped = hs.flipped();
        assert_abs_diff_eq!(flipped.distance(&[0.0, 0.0]), 0.5);
    }

    #[test]
    fn clip_polygon_2d_halves_square() {
        let hs = HalfSpace::new([1.0, 0.0], [0.5, 0.0]);
        let clipped = clip_polygon(&square(), &[hs], 0.0);
        assert_eq!(clipped.len(), 4);
        assert_abs_diff_eq!(signed_area2(&clipped).abs(), 0.5, epsilon = 1e-15);
    }

    #[test]
    fn clip_polygon_2d_outside_is_empty() {
        let hs = HalfSpace::new([1.0, 0.0], [-1.0, 0.0]);
        assert!(clip_polygon(&square(), &[hs], 0.0).is_empty());
    }

    #[test]
    fn clip_polygon_3d_halves_square() {
        let square3 = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let hs = HalfSpace::new([1.0, 0.0, 0.0], [0.5, 0.0, 0.0]);
        let clipped = clip_polygon(&square3, &[hs], 0.0);
        assert_eq!(clipped.len(), 4);
        let area: f64 = (1..clipped.len() - 1)
            .map(|i| {
                let a = clipped[0];
                let b = clipped[i];
                let c = clipped[i + 1];
                let e1 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let e2 = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let n = [
                    e1[1] * e2[2] - e1[2] * e2[1],
                    e1[2] * e2[0] - e1[0] * e2[2],
                    e1[0] * e2[1] - e1[1] * e2[0],
                ];
                0.5 * (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt()
            })
            .sum();
        assert_abs_diff_eq!(area, 0.5, epsilon = 1e-15);
    }

    #[test]
    fn clip_segment_2d() {
        let planes = [
            HalfSpace::new([1.0, 0.0], [0.25, 0.0]).flipped(),
            HalfSpace::new([1.0, 0.0], [0.75, 0.0]),
        ];
        let (a, b) = clip_segment([0.0, 0.0], [1.0, 0.0], &planes, 0.0).unwrap();
        assert_abs_diff_eq!(a[0], 0.25, epsilon = 1e-15);
        assert_abs_diff_eq!(b[0], 0.75, epsilon = 1e-15);
    }

    #[test]
    fn clip_segment_3d_and_disjoint() {
        let planes = [HalfSpace::new([1.0, 0.0, 0.0], [0.5, 0.0, 0.0])];
        let (a, b) = clip_segment([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &planes, 0.0).unwrap();
        assert_abs_diff_eq!(a[0], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(b[0], 0.5, epsilon = 1e-15);
        let outside = [HalfSpace::new([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0])];
        assert!(clip_segment([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &outside, 0.0).is_none());
    }

    #[test]
    fn point_inside_2d_and_3d() {
        let square_planes = [
            HalfSpace::new([0.0, -1.0], [0.0, 0.0]),
            HalfSpace::new([1.0, 0.0], [1.0, 0.0]),
            HalfSpace::new([0.0, 1.0], [0.0, 1.0]),
            HalfSpace::new([-1.0, 0.0], [0.0, 0.0]),
        ];
        assert!(point_inside(&[0.5, 0.5], &square_planes, 0.0));
        assert!(!point_inside(&[1.5, 0.5], &square_planes, 0.0));

        // Unit tetrahedron x >= 0, y >= 0, z >= 0, x + y + z <= 1.
        let tetra_planes = [
            HalfSpace::new([-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
            HalfSpace::new([0.0, -1.0, 0.0], [0.0, 0.0, 0.0]),
            HalfSpace::new([0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
            HalfSpace::new([1.0, 1.0, 1.0], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        ];
        assert!(point_inside(&[0.1, 0.1, 0.1], &tetra_planes, 1e-12));
        assert!(!point_inside(&[0.5, 0.5, 0.5], &tetra_planes, 1e-12));
    }
}
