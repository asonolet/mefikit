//! Owned polyhedra with volume, centroid, and point-in-polyhedron tests.

use std::sync::OnceLock;

use super::convexity::Convexity;
use super::polygon::{Polygon, dominant_axis, newell_normal, project2};

/// An owned polyhedron in 3D space.
///
/// The faces are lists of vertex indices into [`Self::points`]. They are expected to be
/// consistently oriented so that the signed volume computed by the divergence theorem is
/// non-zero; the magnitude of [`Self::volume`] and [`Self::centroid`] do not depend on the
/// orientation, only on its consistency.
#[derive(Clone, Debug)]
pub struct Polyhedron {
    points: Vec<[f64; 3]>,
    faces: Vec<Vec<usize>>,
    convexity: Convexity,
    convexity_cache: OnceLock<bool>,
}

impl Polyhedron {
    /// Creates a polyhedron from the given vertices and faces with an explicit convexity.
    pub fn with_convexity(
        points: impl IntoIterator<Item = [f64; 3]>,
        faces: impl IntoIterator<Item = Vec<usize>>,
        convexity: Convexity,
    ) -> Self {
        Polyhedron {
            points: points.into_iter().collect(),
            faces: faces.into_iter().collect(),
            convexity,
            convexity_cache: OnceLock::new(),
        }
    }

    /// Creates a polyhedron known to be convex.
    pub fn convex(
        points: impl IntoIterator<Item = [f64; 3]>,
        faces: impl IntoIterator<Item = Vec<usize>>,
    ) -> Self {
        Self::with_convexity(points, faces, Convexity::Convex)
    }

    /// Creates a polyhedron whose convexity is not known.
    pub fn unknown(
        points: impl IntoIterator<Item = [f64; 3]>,
        faces: impl IntoIterator<Item = Vec<usize>>,
    ) -> Self {
        Self::with_convexity(points, faces, Convexity::Unknown)
    }

    /// Returns the number of vertices.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the polyhedron has no vertices.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Returns the number of faces.
    pub fn num_faces(&self) -> usize {
        self.faces.len()
    }

    /// Iterates over the polyhedron vertices.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[f64; 3]> {
        self.points.iter()
    }

    /// Returns the axis-aligned bounding box as `[min, max]`.
    pub fn bounds(&self) -> [[f64; 3]; 2] {
        self.points
            .iter()
            .fold([[f64::INFINITY; 3], [-f64::INFINITY; 3]], |acc, p| {
                let mut lo = acc[0];
                let mut hi = acc[1];
                for k in 0..3 {
                    lo[k] = lo[k].min(p[k]);
                    hi[k] = hi[k].max(p[k]);
                }
                [lo, hi]
            })
    }

    /// Returns the `i`-th face as a 3D polygon.
    pub fn face(&self, i: usize) -> Polygon<3> {
        Polygon::unknown(self.faces[i].iter().map(|&j| self.points[j]))
    }

    /// Iterates over the polyhedron faces as 3D polygons.
    pub fn faces(&self) -> impl ExactSizeIterator<Item = Polygon<3>> {
        (0..self.faces.len()).map(|i| self.face(i))
    }

    /// Returns the known convexity of the polyhedron.
    pub fn convexity(&self) -> Convexity {
        self.convexity
    }

    /// Returns `true` if the polyhedron is convex, computing the test on demand if the convexity
    /// was unknown.
    ///
    /// A polyhedron is convex iff every vertex lies on the same side of the plane of each of its
    /// faces. This assumes the faces are planar.
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
        if self.points.len() < 4 {
            return true;
        }
        for face in &self.faces {
            if face.len() < 3 {
                continue;
            }
            let (n, d) = face_plane(&self.points, face);
            let scale = n[0].abs().max(n[1].abs()).max(n[2].abs()).max(1.0);
            let eps = 1e-12 * scale;
            let mut has_pos = false;
            let mut has_neg = false;
            for (i, p) in self.points.iter().enumerate() {
                if face.contains(&i) {
                    continue;
                }
                let v = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] + d;
                if v > eps {
                    has_pos = true;
                } else if v < -eps {
                    has_neg = true;
                }
                if has_pos && has_neg {
                    return false;
                }
            }
        }
        true
    }

    /// Computes the volume of the polyhedron using the divergence theorem.
    pub fn volume(&self) -> f64 {
        self.signed_volume6().abs() / 6.0
    }

    /// Triangulates every face into triangles that exactly tile the face, so the signed volume
    /// and geometric centroid are correct even for concave faces.
    fn face_triangles(&self) -> Vec<[usize; 3]> {
        let mut tris = Vec::new();
        for face in &self.faces {
            tris.extend(ear_clip_triangles(face, &self.points));
        }
        tris
    }

    fn signed_volume6(&self) -> f64 {
        let mut v6 = 0.0;
        for [i, j, k] in self.face_triangles() {
            v6 += triple_product(self.points[i], self.points[j], self.points[k]);
        }
        v6
    }

    /// Computes the vertex centroid of the polyhedron: the arithmetic mean of its vertices.
    ///
    /// This is the average of the node coordinates, not the volume-weighted centroid (see
    /// [`Self::geometric_centroid`]).
    pub fn centroid(&self) -> [f64; 3] {
        let mut c = [0.0; 3];
        for p in &self.points {
            for (ck, pk) in c.iter_mut().zip(p.iter()) {
                *ck += *pk;
            }
        }
        let n = self.points.len() as f64;
        c.map(|v| v / n)
    }

    /// Computes the geometric centroid of the polyhedron as the volume-weighted average of the
    /// signed tetrahedra formed by the origin and each face triangle.
    ///
    /// This is the centroid of the volume enclosed by the polyhedron, not the average of its
    /// vertices (see [`Self::centroid`]).
    pub fn geometric_centroid(&self) -> [f64; 3] {
        let mut v6 = 0.0;
        let mut c = [0.0; 3];
        for [i, j, k] in self.face_triangles() {
            let a = self.points[i];
            let b = self.points[j];
            let cpt = self.points[k];
            let det = triple_product(a, b, cpt);
            v6 += det;
            for kk in 0..3 {
                c[kk] += (a[kk] + b[kk] + cpt[kk]) * det;
            }
        }
        if v6.abs() < 1e-30 {
            return self.points[0];
        }
        [c[0] / (4.0 * v6), c[1] / (4.0 * v6), c[2] / (4.0 * v6)]
    }

    /// Returns `true` if `point` lies inside the polyhedron using half-open ray casting.
    pub fn contains(&self, point: &[f64; 3]) -> bool {
        let px = point[0];
        let py = point[1];
        let pz = point[2];
        let mut inside = false;
        for face in &self.faces {
            let face_points: Vec<[f64; 3]> = face.iter().map(|&j| self.points[j]).collect();
            if ray_crosses_face(px, py, pz, &face_points) {
                inside = !inside;
            }
        }
        inside
    }
}

fn face_plane(points: &[[f64; 3]], face: &[usize]) -> ([f64; 3], f64) {
    let mut n = [0.0; 3];
    let len = face.len();
    for i in 0..len {
        let a = points[face[i]];
        let b = points[face[(i + 1) % len]];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let d = -(n[0] * points[face[0]][0] + n[1] * points[face[0]][1] + n[2] * points[face[0]][2]);
    (n, d)
}

fn triple_product(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]
}

/// Returns `true` if a point is inside a polyhedron using ray-casting.
///
/// The polyhedron is defined by `coords` (vertex positions) and `connectivity`
/// (face indices separated by `usize::MAX`).
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
            let face_points: Vec<[f64; 3]> = connectivity[face_start..face_end]
                .iter()
                .map(|&j| coords[j])
                .collect();

            if ray_crosses_face(px, py, pz, &face_points) {
                inside = !inside;
            }
        }

        face_start = face_end + 1;
    }

    inside
}

/// Provided for backward compatibility; behaves identically to [`point_in_phed`].
pub fn point_in_phed2(point: &[f64; 3], coords: &[[f64; 3]], connectivity: &[usize]) -> bool {
    point_in_phed(point, coords, connectivity)
}

/// Returns `true` if the ray from `(px, py, pz)` in the `+x` direction crosses the (planar)
/// polygon `face_points`.
///
/// The crossing is tested once per face on the face's plane: a plane-parallel face is skipped,
/// the ray-plane intersection point is computed, and the point is tested against the face polygon
/// projected onto its dominant axis plane. Counting each face once avoids double-counting the
/// coplanar triangles of a fan triangulation.
fn ray_crosses_face(px: f64, py: f64, pz: f64, face_points: &[[f64; 3]]) -> bool {
    let n = newell_normal(face_points);
    let denom = n[0];

    let scale = n[0].abs().max(n[1].abs()).max(n[2].abs()).max(1.0);
    let eps = 64.0 * f64::EPSILON * scale;

    if denom.abs() <= eps {
        return false;
    }

    let v0 = face_points[0];
    let t = (n[0] * (v0[0] - px) + n[1] * (v0[1] - py) + n[2] * (v0[2] - pz)) / denom;

    if t <= 0.0 {
        return false;
    }

    let ip = [px + t, py, pz];
    let axis = dominant_axis(n);
    let p2d = project2(ip, axis);
    let face2d: Vec<[f64; 2]> = face_points.iter().map(|&p| project2(p, axis)).collect();

    even_odd_contains(&p2d, &face2d)
}

/// Standard even-odd ray-casting point-in-polygon test in 2D.
fn even_odd_contains(x: &[f64; 2], pgon: &[[f64; 2]]) -> bool {
    let px = x[0];
    let py = x[1];
    let n = pgon.len();
    let mut inside = false;
    for i in 0..n {
        let a = pgon[i];
        let b = pgon[(i + 1) % n];
        if (a[1] > py) != (b[1] > py) {
            let t = (py - a[1]) / (b[1] - a[1]);
            let xi = a[0] + t * (b[0] - a[0]);
            if px < xi {
                inside = !inside;
            }
        }
    }
    inside
}

/// Triangulates a planar face by ear clipping, returning triangles that exactly tile the face
/// while preserving its winding, so concave faces are handled correctly.
fn ear_clip_triangles(face: &[usize], points: &[[f64; 3]]) -> Vec<[usize; 3]> {
    let mut tris = Vec::with_capacity(face.len().saturating_sub(2));
    if face.len() < 3 {
        return tris;
    }
    let pts: Vec<[f64; 3]> = face.iter().map(|&i| points[i]).collect();
    let axis = dominant_axis(newell_normal(&pts));
    let mut ring: Vec<(usize, [f64; 2])> = pts
        .iter()
        .copied()
        .zip(face.iter().copied())
        .map(|(p, i)| (i, project2(p, axis)))
        .collect();
    let mut area = 0.0;
    for k in 0..ring.len() {
        let a = ring[k].1;
        let b = ring[(k + 1) % ring.len()].1;
        area += a[0] * b[1] - a[1] * b[0];
    }
    let orient = if area >= 0.0 { 1.0 } else { -1.0 };
    while ring.len() > 3 {
        let m = ring.len();
        let mut clipped = false;
        for k in 0..m {
            let (ai, a) = ring[(k + m - 1) % m];
            let (bi, b) = ring[k];
            let (ci, c) = ring[(k + 1) % m];
            if (cross2(a, b, c)) * orient < 0.0 {
                continue;
            }
            let mut blocked = false;
            for &(_, p) in &ring {
                if p == a || p == b || p == c {
                    continue;
                }
                if point_in_triangle_strict(p, a, b, c, orient) {
                    blocked = true;
                    break;
                }
            }
            if !blocked {
                tris.push([ai, bi, ci]);
                ring.remove(k);
                clipped = true;
                break;
            }
        }
        if !clipped {
            for k in 0..ring.len() {
                let m = ring.len();
                let (ai, a) = ring[(k + m - 1) % m];
                let (bi, b) = ring[k];
                let (ci, c) = ring[(k + 1) % m];
                if cross2(a, b, c) * orient >= 0.0 {
                    tris.push([ai, bi, ci]);
                    ring.remove(k);
                    clipped = true;
                    break;
                }
            }
        }
        if !clipped {
            break;
        }
    }
    if ring.len() == 3 {
        tris.push([ring[0].0, ring[1].0, ring[2].0]);
    }
    tris
}

/// Cross product `(b - a) × (c - a)` in 2D.
fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Returns `true` if `p` lies strictly inside triangle `(a, b, c)` of the given orientation.
fn point_in_triangle_strict(
    p: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    orient: f64,
) -> bool {
    cross2(a, b, p) * orient > 0.0
        && cross2(b, c, p) * orient > 0.0
        && cross2(c, a, p) * orient > 0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn unit_cube() -> Polyhedron {
        Polyhedron::unknown(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            [
                vec![0, 1, 2, 3],
                vec![0, 3, 7, 4],
                vec![0, 4, 5, 1],
                vec![1, 5, 6, 2],
                vec![2, 6, 7, 3],
                vec![4, 7, 6, 5],
            ],
        )
    }

    fn unit_tet() -> Polyhedron {
        Polyhedron::unknown(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [vec![0, 1, 2], vec![1, 2, 3], vec![2, 3, 0], vec![3, 0, 1]],
        )
    }

    /// Concave L-shaped polyhedron: the notch at `[1,2]^2 x [0,1]` on the bottom face.
    fn l_polyhedron() -> Polyhedron {
        let points = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [0.0, 2.0, 1.0],
        ];
        let faces = vec![
            vec![0, 1, 2, 3, 4, 5],
            vec![0, 5, 11, 6],
            vec![5, 4, 10, 11],
            vec![4, 3, 9, 10],
            vec![3, 2, 8, 9],
            vec![2, 1, 7, 8],
            vec![1, 0, 6, 7],
            vec![6, 11, 10, 9, 8, 7],
        ];
        Polyhedron::unknown(points, faces)
    }

    #[test]
    fn cube_volume() {
        assert_abs_diff_eq!(unit_cube().volume(), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cube_centroid() {
        let c = unit_cube().centroid();
        assert_abs_diff_eq!(c[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(c[2], 0.5, epsilon = 1e-12);
        let g = unit_cube().geometric_centroid();
        assert_abs_diff_eq!(g[0], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g[1], 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g[2], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn tet_volume() {
        assert_abs_diff_eq!(unit_tet().volume(), 1.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn tet_centroid() {
        let c = unit_tet().centroid();
        assert_abs_diff_eq!(c[0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(c[1], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(c[2], 0.25, epsilon = 1e-12);
        let g = unit_tet().geometric_centroid();
        assert_abs_diff_eq!(g[0], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(g[1], 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(g[2], 0.25, epsilon = 1e-12);
    }

    #[test]
    fn centroid_differs_from_geometric_on_concave() {
        // Vertex centroid of the L-polyhedron: the arithmetic mean of its nodes.
        let vertex = l_polyhedron().centroid();
        assert_eq!(vertex, [1.0, 1.0, 0.5]);
        let geometric = l_polyhedron().geometric_centroid();
        assert!(
            (vertex[0] - geometric[0]).abs() > 1e-9
                || (vertex[1] - geometric[1]).abs() > 1e-9
                || (vertex[2] - geometric[2]).abs() > 1e-9,
            "vertex centroid {vertex:?} must differ from geometric centroid {geometric:?}"
        );
        assert!(
            l_polyhedron().contains(&geometric),
            "geometric centroid {geometric:?} must be inside the L-polyhedron (vertex {vertex:?})"
        );
    }

    #[test]
    fn cube_contains() {
        let phed = unit_cube();
        assert!(phed.contains(&[0.5, 0.5, 0.5]));
        assert!(!phed.contains(&[1.5, 0.5, 0.5]));
        assert!(!phed.contains(&[0.5, -0.5, 0.5]));
    }

    #[test]
    fn tet_contains() {
        let phed = unit_tet();
        assert!(phed.contains(&[0.25, 0.25, 0.25]));
        assert!(!phed.contains(&[0.75, 0.75, 0.75]));
    }

    #[test]
    fn cube_is_convex() {
        assert!(unit_cube().is_convex());
        assert!(unit_tet().is_convex());
        assert!(!l_polyhedron().is_convex());
    }

    #[test]
    fn polyhedron_faces() {
        let phed = unit_cube();
        assert_eq!(phed.num_faces(), 6);
        let face = phed.face(0);
        assert_abs_diff_eq!(face.area(), 1.0, epsilon = 1e-12);
        assert_eq!(phed.faces().len(), 6);
    }

    #[test]
    fn flat_connectivity_contains() {
        let points = unit_cube().iter().copied().collect::<Vec<_>>();
        let conns = [
            [0, 1, 2, 3],
            [0, 3, 7, 4],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [4, 7, 6, 5],
        ];
        let flat: Vec<usize> = conns
            .iter()
            .flat_map(|face| face.iter().copied().chain([usize::MAX]))
            .collect();
        assert!(point_in_phed2(&[0.5, 0.5, 0.5], &points, &flat));
        assert!(!point_in_phed2(&[1.5, 0.5, 0.5], &points, &flat));
        assert!(point_in_phed(&[0.5, 0.5, 0.5], &points, &flat));
        assert!(!point_in_phed(&[1.5, 0.5, 0.5], &points, &flat));
    }
}
