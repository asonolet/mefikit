//! Owned polyhedra with volume, centroid, and point-in-polyhedron tests.

use std::sync::OnceLock;

use super::convexity::Convexity;
use super::polygon::{Polygon, cross2, dominant_axis, newell_normal, project2};
use super::{bounds_iter, vertex_centroid};
use crate::mesh::IndirectIndexOwned;

/// An owned polyhedron in 3D space.
///
/// The faces are lists of vertex indices into `Self::points`, stored as an
/// [`IndirectIndexOwned`]: a flat data array plus cumulative offsets, so variable-length faces
/// use one contiguous allocation instead of one per face. They are expected to be consistently
/// oriented so that the signed volume computed by the divergence theorem is non-zero; the
/// magnitude of [`Self::volume`] and [`Self::centroid`] do not depend on the orientation, only on
/// its consistency.
#[derive(Clone, Debug)]
pub struct Polyhedron {
    points: Vec<[f64; 3]>,
    faces: IndirectIndexOwned<usize>,
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
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for face in faces {
            data.extend_from_slice(&face);
            offsets.push(data.len());
        }
        Polyhedron {
            points: points.into_iter().collect(),
            faces: IndirectIndexOwned {
                data: data.into(),
                offsets: offsets.into(),
            },
            convexity,
            convexity_cache: OnceLock::new(),
        }
    }

    /// Creates a polyhedron from the given vertices and an indirect index of face vertex
    /// indices, with an explicit convexity.
    pub(crate) fn with_indirect_index(
        points: impl IntoIterator<Item = [f64; 3]>,
        faces: IndirectIndexOwned<usize>,
        convexity: Convexity,
    ) -> Self {
        Polyhedron {
            points: points.into_iter().collect(),
            faces,
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
        bounds_iter(self.points.iter().copied())
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
        for fi in 0..self.faces.len() {
            let face = &self.faces[fi];
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
        for fi in 0..self.faces.len() {
            tris.extend(ear_clip_triangles(&self.faces[fi], &self.points));
        }
        tris
    }

    fn signed_volume6(&self) -> f64 {
        // Sum the triple products about the vertex centroid rather than the origin. By the
        // divergence theorem the signed volume is independent of the reference point, but summing
        // over *absolute* coordinates is numerically unstable for a polyhedron translated far from
        // the origin: it mixes ~M^3 terms to recover a ~L^3 result (M the coordinate magnitude,
        // L the shape size), silently losing most significant digits. Recentring on the centroid
        // makes the operands ~L in magnitude and translation exact, recovering full precision.
        let c = self.centroid();
        let mut v6 = 0.0;
        for [i, j, k] in self.face_triangles() {
            let a = [
                self.points[i][0] - c[0],
                self.points[i][1] - c[1],
                self.points[i][2] - c[2],
            ];
            let b = [
                self.points[j][0] - c[0],
                self.points[j][1] - c[1],
                self.points[j][2] - c[2],
            ];
            let dpt = [
                self.points[k][0] - c[0],
                self.points[k][1] - c[1],
                self.points[k][2] - c[2],
            ];
            v6 += triple_product(a, b, dpt);
        }
        v6
    }
    /// Computes the vertex centroid of the polyhedron: the arithmetic mean of its vertices.
    ///
    /// This is the average of the node coordinates, not the volume-weighted centroid (see
    /// [`Self::geometric_centroid`]).
    pub fn centroid(&self) -> [f64; 3] {
        vertex_centroid(&self.points)
    }

    /// Reorients every face in place so that each face normal (right-hand rule over the
    /// face winding) points outward: faces are wound CCW viewed from outside the
    /// polyhedron.
    ///
    /// Uses the vertex centroid as the interior reference point, which is correct for
    /// convex polyhedra (the case the face-shell machinery targets). After this call
    /// [`Self::volume`] and [`Self::geometric_centroid`] are exact for any input winding,
    /// and the `face_plane` normals/offsets are all outward.
    ///
    /// This is an explicit, coordinate-aware repair step; topology-only paths that build
    /// well-formed shells (e.g. `to_poly`/`subentities`) do not pay for it.
    pub fn into_ccw(&mut self) {
        let refp = self.centroid();
        for face in &mut self.faces {
            let (n, _d) = face_plane(&self.points, face);
            let p0 = self.points[face[0]];
            // n·x + d = 0 is the face plane. With `refp` strictly inside, the winding is
            // outward when the normal points away from `refp`, i.e. n·(p0 - refp) > 0.
            let dot =
                n[0] * (p0[0] - refp[0]) + n[1] * (p0[1] - refp[1]) + n[2] * (p0[2] - refp[2]);
            debug_assert!(
                dot.abs() > 1e-14,
                "face plane contains the centroid reference point"
            );
            if dot < 0.0 {
                face.reverse();
            }
        }
    }

    /// Computes the geometric centroid of the polyhedron as the volume-weighted average of the
    /// signed tetrahedra formed by the origin and each face triangle.
    ///
    /// This is the centroid of the volume enclosed by the polyhedron, not the average of its
    /// vertices (see [`Self::centroid`]).
    pub fn geometric_centroid(&self) -> [f64; 3] {
        // Reduce to the recentred frame (the vertex centroid at the origin) before accumulating,
        // so the ~M^4 cancellations of the absolute-coordinate sum are avoided for shapes translated
        // far from the origin. See `signed_volume6` for the same rationale.
        let refp = self.centroid();
        let mut v6 = 0.0;
        let mut c = [0.0; 3];
        for [i, j, k] in self.face_triangles() {
            let a = [
                self.points[i][0] - refp[0],
                self.points[i][1] - refp[1],
                self.points[i][2] - refp[2],
            ];
            let b = [
                self.points[j][0] - refp[0],
                self.points[j][1] - refp[1],
                self.points[j][2] - refp[2],
            ];
            let cpt = [
                self.points[k][0] - refp[0],
                self.points[k][1] - refp[1],
                self.points[k][2] - refp[2],
            ];
            let det = triple_product(a, b, cpt);
            v6 += det;
            for kk in 0..3 {
                c[kk] += (a[kk] + b[kk] + cpt[kk]) * det;
            }
        }
        if v6.abs() < 1e-30 {
            return self.points[0];
        }
        let g = [c[0] / (4.0 * v6), c[1] / (4.0 * v6), c[2] / (4.0 * v6)];
        [g[0] + refp[0], g[1] + refp[1], g[2] + refp[2]]
    }

    /// Returns `true` if `point` lies inside the polyhedron using half-open ray casting.
    pub fn contains(&self, point: &[f64; 3]) -> bool {
        let px = point[0];
        let py = point[1];
        let pz = point[2];
        let mut inside = false;
        for fi in 0..self.faces.len() {
            if ray_crosses_face(px, py, pz, &self.faces[fi], &self.points) {
                inside = !inside;
            }
        }
        inside
    }

    /// Computes the volume of the intersection of two convex polyhedra without constructing the
    /// intersection polyhedron.
    ///
    /// Every face of `self` is clipped against the half-spaces of `other` and vice versa, and the
    /// signed volume contribution of each resulting boundary polygon is accumulated directly via a
    /// triangle fan (divergence theorem).
    ///
    /// Both inputs are assumed convex; the convexity is not checked. Empty or non-overlapping
    /// inputs yield `0.0`.
    ///
    /// Faces lying in the same plane with the same interior side appear on the boundary of both
    /// polyhedra and would be counted twice; only the first is accumulated (the two clipped
    /// patches coincide exactly).
    pub fn convex_intersection_volume(&self, other: &Self) -> f64 {
        let [s_min, s_max] = self.bounds();
        let [o_min, o_max] = other.bounds();
        for k in 0..3 {
            if s_max[k] < o_min[k] || o_max[k] < s_min[k] {
                return 0.0;
            }
        }
        self.convex_intersection_volume_impl(other)
    }

    /// Core of [`Self::convex_intersection_volume`] without the axis-aligned bounding-box reject.
    ///
    /// The AABB reject is the only difference from [`Self::convex_intersection_volume`]: it spends
    /// roughly the cost of two `bounds()` folds on every call, so callers that already cull pairs
    /// by AABB (e.g. a BVH-driven boundary-element search) should call this to avoid the redundant
    /// work. The inputs are still assumed convex and the clip itself is unchanged.
    pub(crate) fn convex_intersection_volume_impl(&self, other: &Self) -> f64 {
        let scale = self
            .points
            .iter()
            .chain(other.points.iter())
            .map(|p| p[0].abs().max(p[1].abs()).max(p[2].abs()))
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let eps = 64.0 * f64::EPSILON * scale;

        // Translation-invariant length scale: the maximum distance of any vertex from the centroid
        // of both polyhedra. Used for the coplanarity tolerance so that `plane_coplanar_opposite`
        // does not depend on where the geometry is placed in space.
        let centroid = {
            let mut c = [0.0; 3];
            let mut n = 0.0;
            for p in self.points.iter().chain(other.points.iter()) {
                for (ci, xi) in c.iter_mut().zip(p.iter()) {
                    *ci += xi;
                }
                n += 1.0;
            }
            for xi in c.iter_mut() {
                *xi /= n;
            }
            c
        };
        let radial = self
            .points
            .iter()
            .chain(other.points.iter())
            .map(|p| {
                ((p[0] - centroid[0]).powi(2)
                    + (p[1] - centroid[1]).powi(2)
                    + (p[2] - centroid[2]).powi(2))
                .sqrt()
            })
            .fold(0.0_f64, f64::max)
            .max(1.0);
        let coplanar_tol = 1e-4 * radial;

        let other_planes = other.planes();
        let self_planes = self.planes();
        let reference = self.centroid();

        let mut volume = 0.0;
        for (fi, self_plane) in self_planes.iter().copied().enumerate() {
            // A face of `self` that is coplanar with a face of `other` lies on `other`'s boundary
            // (a shared separating face), so it contributes no 3D overlap regardless of its
            // orientation: the two cells only touch along a 2D face. Callers in remapping expect
            // adjacent cells to yield a zero intersection. Skipping it here (instead of clipping) is
            // also what makes the result independent of tiny warp/tilt of the shared face.
            if other_planes
                .iter()
                .any(|other_plane| plane_coplanar_opposite(self_plane, *other_plane, coplanar_tol))
            {
                continue;
            }
            let poly = self.face_polygon(fi);
            // Do not clip a boundary face against a plane that coincides with its own plane: a
            // (possibly slightly non-planar) face's vertices may straddle its best-fit plane, and
            // clipping it against that same plane — its own boundary — wrongly rejects the face.
            let planes: Vec<Plane> = other_planes
                .iter()
                .filter(|plane| !plane_same_side(self_plane, **plane, coplanar_tol))
                .copied()
                .collect();
            if let Some(clipped) = clip_face_by_polyhedron(&poly, &planes, eps) {
                let v = polygon_volume_contribution(&clipped, reference);
                volume += v;
            }
        }
        for (fi, plane) in other_planes.iter().enumerate() {
            // Symmetric skip: a face of `other` coplanar-opposite with a face of `self` is the
            // shared separating face and contributes no overlap.
            if self_planes
                .iter()
                .any(|p| plane_coplanar_opposite(*p, *plane, coplanar_tol))
            {
                continue;
            }
            if self_planes
                .iter()
                .any(|p| plane_same_side(*p, *plane, coplanar_tol))
            {
                continue;
            }
            let poly = other.face_polygon(fi);
            if let Some(clipped) = clip_face_by_polyhedron(&poly, &self_planes, eps) {
                let v = polygon_volume_contribution(&clipped, reference);
                volume += v;
            }
        }
        volume.abs()
    }

    /// Plane equation of each face: the outward unit normal `n` and offset `d` such that the
    /// interior of the polyhedron satisfies `n · x <= d`.
    ///
    /// The outward direction is inherited from the face winding (the faces are expected to be
    /// consistently outward-oriented, matching the divergence-theorem volume convention).
    fn planes(&self) -> Vec<Plane> {
        (0..self.faces.len())
            .map(|i| plane_of_face(&self.points, &self.faces[i]))
            .collect()
    }

    /// Gathers the vertex coordinates of face `i` into a fresh polygon.
    fn face_polygon(&self, i: usize) -> Vec<[f64; 3]> {
        self.faces[i].iter().map(|&j| self.points[j]).collect()
    }
}

/// A plane with outward unit normal `n` and offset `d`; the interior half-space is `n · x <= d`.
#[derive(Clone, Copy)]
struct Plane {
    n: [f64; 3],
    d: f64,
}

/// Computes the outward plane equation of a face by Newell's method.
///
/// This is the same normal as [`face_plane`] but expressed as `n · x <= d` (interior) instead of
/// `n · x + d = 0`.
fn plane_of_face(points: &[[f64; 3]], face: &[usize]) -> Plane {
    let (n, d) = face_plane(points, face);
    let norm = n[0].hypot(n[1]).hypot(n[2]);
    let inv = 1.0 / norm;
    Plane {
        n: [n[0] * inv, n[1] * inv, n[2] * inv],
        d: -d * inv,
    }
}

/// Returns `true` if the two planes coincide with the interior on the same side, i.e. the outward
/// normals point the same way and the offsets agree.
///
/// Used to skip the second half of coincident coplanar face contributions: when a face of `P` and
/// a face of `Q` are `same_side` coplanar, their clipped patches in the intersection boundary are
/// identical (`F ∩ Q = G ∩ P = (P ∩ Q) ∩ H`), so accumulating both would double the volume.
///
/// `tol` is the same warp-aware, translation-invariant tolerance as [`plane_coplanar_opposite`]:
/// two cells sharing a (possibly slightly warped) boundary face fit their planes independently, so
/// their offsets can agree only to within the face's non-planarity. Without absorbing that warp the
/// coincident faces would not be recognized, and their (identical) patches would be counted twice,
/// overstating the intersection volume.
fn plane_same_side(a: Plane, b: Plane, tol: f64) -> bool {
    let ndot = a.n[0] * b.n[0] + a.n[1] * b.n[1] + a.n[2] * b.n[2];
    if ndot <= 1.0 - 1e-10 {
        return false;
    }
    (a.d - b.d).abs() <= tol
}

/// Returns `true` if the two planes are (approximately) coincident with the interior on opposite
/// sides, i.e. the outward normals point in opposite directions and the offsets agree. This is the
/// signature of two adjacent cells sharing a separating face: `self`'s face is coplanar with a face
/// of `other`, so it lies on `other`'s boundary and contributes no 3D overlap.
///
/// `tol` is a translation-invariant length scale (the radial size of the geometry) times a relative
/// factor. The offset agreement tolerance is expressed relative to it so that the non-planarity
/// (warp) of a shared face — which shifts the individually fitted planes by an amount proportional
/// to the geometry size — is absorbed: two faces built from the same points but in opposite winding
/// can disagree by a small relative amount even though they are geometrically the same plane.
fn plane_coplanar_opposite(a: Plane, b: Plane, tol: f64) -> bool {
    let ndot = a.n[0] * b.n[0] + a.n[1] * b.n[1] + a.n[2] * b.n[2];
    if ndot >= -1.0 + 1e-10 {
        return false;
    }
    (a.d + b.d).abs() <= tol
}

/// Clips the convex polygon `poly` against each half-space of `planes`, keeping the part where
/// `n · x <= d` for every plane.
///
/// Returns `None` if the polygon is entirely clipped away (fewer than 3 vertices remain).
fn clip_face_by_polyhedron(poly: &[[f64; 3]], planes: &[Plane], eps: f64) -> Option<Vec<[f64; 3]>> {
    let mut current = poly.to_vec();
    for plane in planes {
        if current.len() < 3 {
            return None;
        }
        let n = plane.n;
        let d = plane.d;
        let mut all_inside = true;
        let mut all_outside = true;
        for v in &current {
            let dist = n[0] * v[0] + n[1] * v[1] + n[2] * v[2] - d;
            if dist > eps {
                all_inside = false;
            } else {
                all_outside = false;
            }
        }
        if all_inside {
            continue;
        }
        if all_outside {
            return None;
        }
        current = clip_polygon_by_plane(&current, plane, eps);
        if current.len() < 3 {
            return None;
        }
    }
    Some(current)
}

/// Sutherland–Hodgman clipping of a polygon against the single half-space `n · x <= d`.
fn clip_polygon_by_plane(poly: &[[f64; 3]], plane: &Plane, eps: f64) -> Vec<[f64; 3]> {
    let n = plane.n;
    let d = plane.d;
    let mut result = Vec::with_capacity(poly.len() + 2);

    let m = poly.len();
    let mut prev = m - 1;
    let mut prev_dist = {
        let pr = poly[m - 1];
        n[0] * pr[0] + n[1] * pr[1] + n[2] * pr[2] - d
    };
    let mut prev_inside = prev_dist <= eps;

    for ci in 0..m {
        let cur = poly[ci];
        let cur_dist = n[0] * cur[0] + n[1] * cur[1] + n[2] * cur[2] - d;
        let cur_inside = cur_dist <= eps;

        if prev_inside && cur_inside {
            result.push(cur);
        } else if prev_inside && !cur_inside {
            result.push(segment_plane_intersection(
                poly[prev], cur, prev_dist, cur_dist,
            ));
        } else if !prev_inside && cur_inside {
            result.push(segment_plane_intersection(
                poly[prev], cur, prev_dist, cur_dist,
            ));
            result.push(cur);
        }

        prev = ci;
        prev_dist = cur_dist;
        prev_inside = cur_inside;
    }
    remove_consecutive_duplicates(&mut result);
    result
}

/// Intersection of the segment `a -> b` with the plane, given the signed distances `da = n·a - d`
/// and `db = n·b - d`.
#[inline(always)]
fn segment_plane_intersection(a: [f64; 3], b: [f64; 3], da: f64, db: f64) -> [f64; 3] {
    let t = da / (da - db);
    [
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    ]
}

/// Removes consecutive duplicate vertices produced by clipping, and the cyclic first/last pair.
fn remove_consecutive_duplicates(vertices: &mut Vec<[f64; 3]>) {
    if vertices.len() < 2 {
        return;
    }
    let scale = vertices
        .iter()
        .map(|v| v[0].abs().max(v[1].abs()).max(v[2].abs()))
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let eps = 64.0 * f64::EPSILON * scale;
    let mut out = 0;
    for i in 0..vertices.len() {
        let v = vertices[i];
        if out == 0 || dist3(v, vertices[out - 1]) > eps {
            vertices[out] = v;
            out += 1;
        }
    }
    if out > 1 && dist3(vertices[0], vertices[out - 1]) <= eps {
        out -= 1;
    }
    vertices.truncate(out);
}

#[inline(always)]
fn dist3(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

/// Signed volume contribution of an outward-oriented boundary polygon to the intersection volume,
/// via a triangle fan about `reference` (divergence theorem).
fn polygon_volume_contribution(poly: &[[f64; 3]], reference: [f64; 3]) -> f64 {
    let n = poly.len();
    if n < 3 {
        return 0.0;
    }
    let p0 = poly[0];
    let mut sum = 0.0;
    for i in 1..n - 1 {
        let p1 = poly[i];
        let p2 = poly[i + 1];
        let a = [
            p0[0] - reference[0],
            p0[1] - reference[1],
            p0[2] - reference[2],
        ];
        let b = [
            p1[0] - reference[0],
            p1[1] - reference[1],
            p1[2] - reference[2],
        ];
        let c = [
            p2[0] - reference[0],
            p2[1] - reference[1],
            p2[2] - reference[2],
        ];
        sum += triple_product(a, b, c);
    }
    sum / 6.0
}

pub(crate) fn face_plane(points: &[[f64; 3]], face: &[usize]) -> ([f64; 3], f64) {
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

#[inline(always)]
fn triple_product(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let cross = [
        b[1] * c[2] - b[2] * c[1],
        b[2] * c[0] - b[0] * c[2],
        b[0] * c[1] - b[1] * c[0],
    ];
    a[0] * cross[0] + a[1] * cross[1] + a[2] * cross[2]
}

/// The six quadrilateral faces of a HEX8 in `subentities(D1)` connectivity order (VTK convention).
const HEX8_FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [1, 2, 6, 5],
    [3, 0, 4, 7],
];

/// The six quadrilateral faces of a HEX8 in `subentities(D1)` connectivity order (VTK convention).
const HEX8_TET: [[usize; 4]; 6] = [
    [0, 5, 1, 7],
    [0, 4, 5, 7],
    [1, 6, 2, 7],
    [1, 5, 6, 7],
    [0, 2, 3, 7],
    [0, 1, 2, 7],
];

/// Computes the volume of a tetrahedron.
#[inline(always)]
pub(crate) fn tet_volume(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    triple_product(ab, ac, ad) / 6.0
}

/// Computes the volume of a hexahedron.
///
/// Bit-exact with [`Polyhedron::volume`] on a hexahedron: each of the six quad faces is
/// ear-clipped into two triangles, as in `ear_clip_triangles`, and the triple products are summed
/// over the face layout produced by `as_polyhedron`.
pub(crate) fn hex_volume(p: &[[f64; 3]; 8]) -> f64 {
    let mut v = 0.0;
    for tet in HEX8_TET {
        v += tet_volume(&p[tet[0]], &p[tet[1]], &p[tet[2]], &p[tet[3]]);
    }
    v
}

/// The four triangular faces of a TET4 in `subentities(D1)` connectivity order.
const TET4_FACES: [[usize; 3]; 4] = [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]];

/// Returns `true` if `point` lies inside the tetrahedron with vertices `a, b, c, d`.
///
/// Bit-exact with [`Polyhedron::contains`] on a tetrahedron (same ray casting and plane tests over
/// the face layout produced by `as_polyhedron`) but allocates nothing.
#[inline(always)]
pub(crate) fn tet_contains(
    point: &[f64; 3],
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
    d: &[f64; 3],
) -> bool {
    let coords = [*a, *b, *c, *d];
    even_odd_contains_phed(point, &coords, &TET4_FACES)
}

/// Returns `true` if `point` lies inside the hexahedron `p`.
///
/// Bit-exact with [`Polyhedron::contains`] on a HEX8 (same ray casting over the six quad faces
/// produced by `as_polyhedron`) but allocates nothing.
#[inline(always)]
pub(crate) fn hex_contains(point: &[f64; 3], p: &[[f64; 3]; 8]) -> bool {
    even_odd_contains_phed(point, p, &HEX8_FACES)
}

/// Half-open ray-casting point-in-polyhedron test over `faces`, sharing the per-face plane test
/// and projected even-odd face test of [`Polyhedron::contains`].
fn even_odd_contains_phed<const NF: usize, const NV: usize>(
    point: &[f64; 3],
    coords: &[[f64; 3]],
    faces: &[[usize; NV]; NF],
) -> bool {
    let px = point[0];
    let py = point[1];
    let pz = point[2];
    let mut inside = false;
    for face in faces {
        if ray_crosses_face(px, py, pz, face, coords) {
            inside = !inside;
        }
    }
    inside
}

/// Triangulates a quad face with the same deterministic ear-clipping used by
/// `ear_clip_triangles`, writing at most two triangles into `tris` and returning the count.
#[allow(unused)]
fn ear_clip_quad(face: [usize; 4], points: &[[f64; 3]]) -> ([[usize; 3]; 2], usize) {
    let axis = dominant_axis(newell_normal(&[
        points[face[0]],
        points[face[1]],
        points[face[2]],
        points[face[3]],
    ]));
    let mut ring = [
        (face[0], project2(points[face[0]], axis)),
        (face[1], project2(points[face[1]], axis)),
        (face[2], project2(points[face[2]], axis)),
        (face[3], project2(points[face[3]], axis)),
    ];
    let mut len = 4;
    let mut tris = [[0; 3]; 2];
    let mut n = 0;
    ear_clip_ring(&mut ring, &mut len, |t| {
        tris[n] = t;
        n += 1;
    });
    (tris, n)
}

/// Deterministic ear-clipping of a planar face, emitting triangles that exactly tile the face
/// while preserving its winding, so concave faces are handled correctly.
///
/// The live vertices form the prefix `ring[..len]`; clipping shifts the tail left and decrements
/// `len`. Shared by [`ear_clip_triangles`] (arbitrary faces) and [`ear_clip_quad`] (stack-only
/// quads), so both produce bit-identical triangulations.
fn ear_clip_ring(
    ring: &mut [(usize, [f64; 2])],
    len: &mut usize,
    mut emit: impl FnMut([usize; 3]),
) {
    let mut area = 0.0;
    for k in 0..*len {
        let a = ring[k].1;
        let b = ring[(k + 1) % *len].1;
        area += a[0] * b[1] - a[1] * b[0];
    }
    let orient = if area >= 0.0 { 1.0 } else { -1.0 };
    while *len > 3 {
        let m = *len;
        let mut clipped = false;
        for k in 0..m {
            let (ai, a) = ring[(k + m - 1) % m];
            let (bi, b) = ring[k];
            let (ci, c) = ring[(k + 1) % m];
            if cross2(a, b, c) * orient < 0.0 {
                continue;
            }
            let blocked = ring[..*len].iter().any(|&(_, p)| {
                p != a && p != b && p != c && point_in_triangle_strict(p, a, b, c, orient)
            });
            if !blocked {
                emit([ai, bi, ci]);
                for j in k..*len - 1 {
                    ring[j] = ring[j + 1];
                }
                *len -= 1;
                clipped = true;
                break;
            }
        }
        if !clipped {
            let m = *len;
            for k in 0..m {
                let (ai, a) = ring[(k + m - 1) % m];
                let (bi, b) = ring[k];
                let (ci, c) = ring[(k + 1) % m];
                if cross2(a, b, c) * orient >= 0.0 {
                    emit([ai, bi, ci]);
                    for j in k..m - 1 {
                        ring[j] = ring[j + 1];
                    }
                    *len -= 1;
                    clipped = true;
                    break;
                }
            }
        }
        if !clipped {
            break;
        }
    }
    if *len == 3 {
        emit([ring[0].0, ring[1].0, ring[2].0]);
    }
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
        if face_end - face_start >= 3
            && ray_crosses_face(px, py, pz, &connectivity[face_start..face_end], coords)
        {
            inside = !inside;
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
/// polygon `face`.
///
/// The crossing is tested once per face on the face's plane: a plane-parallel face is skipped,
/// the ray-plane intersection point is computed, and the point is tested against the face polygon
/// projected onto its dominant axis plane. Counting each face once avoids double-counting the
/// coplanar triangles of a fan triangulation. Vertices are read from `coords` directly, so no
/// per-face buffer is allocated.
fn ray_crosses_face(px: f64, py: f64, pz: f64, face: &[usize], coords: &[[f64; 3]]) -> bool {
    let mut n = [0.0; 3];
    let len = face.len();
    for i in 0..len {
        let a = coords[face[i]];
        let b = coords[face[(i + 1) % len]];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    let denom = n[0];

    let scale = n[0].abs().max(n[1].abs()).max(n[2].abs()).max(1.0);
    let eps = 64.0 * f64::EPSILON * scale;

    if denom.abs() <= eps {
        return false;
    }

    let v0 = coords[face[0]];
    let t = (n[0] * (v0[0] - px) + n[1] * (v0[1] - py) + n[2] * (v0[2] - pz)) / denom;

    if t <= 0.0 {
        return false;
    }

    let axis = dominant_axis(n);
    let p2 = project2([px + t, py, pz], axis);

    // Standard even-odd ray-casting point-in-polygon test over the projected face.
    let mut inside = false;
    for i in 0..len {
        let a = project2(coords[face[i]], axis);
        let b = project2(coords[face[(i + 1) % len]], axis);
        if (a[1] > p2[1]) != (b[1] > p2[1]) {
            let t = (p2[1] - a[1]) / (b[1] - a[1]);
            let xi = a[0] + t * (b[0] - a[0]);
            if p2[0] < xi {
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
    let mut len = ring.len();
    ear_clip_ring(&mut ring, &mut len, |t| tris.push(t));
    tris
}

/// Returns `true` if `p` lies strictly inside triangle `(a, b, c)` of the given orientation.
///
/// Uses the naive [`cross2`] orientation: points on the triangle edges are excluded (strict
/// inside), and a point is reported inside only when it is strictly on the correct side of all
/// three edges. Boundary and collinear cases are resolved by the caller's fallback clipping.
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
    use proptest::prelude::*;

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
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![2, 3, 7, 6],
                vec![1, 2, 6, 5],
                vec![3, 0, 4, 7],
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
            [vec![0, 1, 3], vec![1, 2, 3], vec![2, 0, 3], vec![0, 2, 1]],
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
    fn into_ccw_repairs_inverted_face() {
        // Unit cube translated away from the origin, whose bottom face is wound inward
        // (the pre-fix `to_poly` layout). Because the divergent-theorem sum is sensitive
        // to the face orientation, the signed volume is position-dependent garbage.
        let mut p = Polyhedron::unknown(
            [
                [1.0, 1.0, 1.0],
                [2.0, 1.0, 1.0],
                [2.0, 2.0, 1.0],
                [1.0, 2.0, 1.0],
                [1.0, 1.0, 2.0],
                [2.0, 1.0, 2.0],
                [2.0, 2.0, 2.0],
                [1.0, 2.0, 2.0],
            ],
            [
                vec![0, 1, 2, 3], // inward (old to_poly winding)
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![2, 3, 7, 6],
                vec![1, 2, 6, 5],
                vec![3, 0, 4, 7],
            ],
        );
        assert!(p.volume() != 1.0);
        p.into_ccw();
        assert_abs_diff_eq!(p.volume(), 1.0, epsilon = 1e-12);
        // All face normals now point outward (away from the centroid).
        let c = p.centroid();
        for fi in 0..p.num_faces() {
            let (n, _d) = face_plane(&p.points, &p.faces[fi]);
            let scale = n[0].abs().max(n[1].abs()).max(n[2].abs());
            let p0 = p.points[p.faces[fi][0]];
            assert!(
                (n[0] * (p0[0] - c[0]) + n[1] * (p0[1] - c[1]) + n[2] * (p0[2] - c[2]))
                    > scale * 1e-12
            );
        }
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
    fn tet_polyhedron_contains() {
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
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
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

    #[test]
    fn tet_volume_helper_matches_polyhedron_volume() {
        let p = unit_tet();
        assert_eq!(
            super::tet_volume(&p.points[0], &p.points[1], &p.points[2], &p.points[3]),
            p.volume(),
            "TET4 fast path must stay bit-exact with Polyhedron::volume"
        );
    }

    #[test]
    fn hex_volume_helper_matches_polyhedron_volume() {
        let p = unit_cube();
        assert_abs_diff_eq!(
            super::hex_volume(&[
                p.points[0],
                p.points[1],
                p.points[2],
                p.points[3],
                p.points[4],
                p.points[5],
                p.points[6],
                p.points[7]
            ]),
            p.volume(),
        );
    }

    #[test]
    fn hex_volume_helper_on_skewed_hex() {
        let pts = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.5, 0.5, 3.0],
            [2.5, 0.5, 3.0],
            [2.5, 2.5, 3.0],
            [0.5, 2.5, 3.0],
        ];
        let phed = Polyhedron::convex(
            pts.to_vec(),
            vec![
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![2, 3, 7, 6],
                vec![1, 2, 6, 5],
                vec![3, 0, 4, 7],
            ],
        );
        assert_abs_diff_eq!(super::hex_volume(&pts), phed.volume());
    }

    #[test]
    fn tet_contains_matches_polyhedron_contains() {
        let p = unit_tet();
        let points: [[f64; 3]; 7] = [
            [0.25, 0.25, 0.25],
            [0.1, 0.1, 0.1],
            [0.75, 0.75, 0.75],
            [1.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0],
        ];
        for pt in points {
            assert_eq!(
                super::tet_contains(&pt, &p.points[0], &p.points[1], &p.points[2], &p.points[3]),
                p.contains(&pt),
                "TET4 contains fast path must stay bit-exact with Polyhedron::contains at {pt:?}"
            );
        }
    }

    #[test]
    fn hex_contains_matches_polyhedron_contains() {
        let p = unit_cube();
        let points: [[f64; 3]; 9] = [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.5, 0.0, 0.5],
            [0.25, 0.75, 0.25],
            [2.0, 2.0, 2.0],
            [-0.5, 0.5, 0.5],
        ];
        for pt in points {
            assert_eq!(
                super::hex_contains(
                    &pt,
                    &[
                        p.points[0],
                        p.points[1],
                        p.points[2],
                        p.points[3],
                        p.points[4],
                        p.points[5],
                        p.points[6],
                        p.points[7]
                    ]
                ),
                p.contains(&pt),
                "HEX8 contains fast path must stay bit-exact with Polyhedron::contains at {pt:?}"
            );
        }
    }

    #[test]
    fn indirect_index_faces_match_vec_of_vec_construction() {
        let points: Vec<[f64; 3]> = unit_cube().points.clone();
        let faces: Vec<Vec<usize>> = vec![
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ];
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        for face in &faces {
            data.extend_from_slice(face);
            offsets.push(data.len());
        }
        let indirect = Polyhedron::with_indirect_index(
            points.clone(),
            IndirectIndexOwned {
                data: data.into(),
                offsets: offsets.into(),
            },
            Convexity::Unknown,
        );
        let vec_of_vec = Polyhedron::with_convexity(points, faces, Convexity::Unknown);
        assert_eq!(indirect.volume(), vec_of_vec.volume());
        assert_eq!(indirect.centroid(), vec_of_vec.centroid());
        assert_eq!(
            indirect.geometric_centroid(),
            vec_of_vec.geometric_centroid()
        );
        assert_eq!(indirect.is_convex(), vec_of_vec.is_convex());
        for pt in [
            [0.5, 0.5, 0.5],
            [1.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ] {
            assert_eq!(
                indirect.contains(&pt),
                vec_of_vec.contains(&pt),
                "IndirectIndex-backed faces must match Vec<Vec> at {pt:?}"
            );
        }
    }

    #[test]
    fn point_in_phed_with_shuffled_global_indices() {
        // Unit cube whose 8 vertices sit at scattered rows of a larger coords array, so the
        // connectivity indices (4, 9, 2, ...) do not equal the row order.
        let coords: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 1.0], // row 0: cube vertex 4
            [9.0, 9.0, 9.0], // row 1: filler
            [1.0, 1.0, 0.0], // row 2: cube vertex 2
            [9.0, 9.0, 9.0], // row 3: filler
            [0.0, 0.0, 0.0], // row 4: cube vertex 0
            [1.0, 1.0, 1.0], // row 5: cube vertex 6
            [9.0, 9.0, 9.0], // row 6: filler
            [0.0, 1.0, 0.0], // row 7: cube vertex 3
            [0.0, 1.0, 1.0], // row 8: cube vertex 7
            [1.0, 0.0, 0.0], // row 9: cube vertex 1
            [9.0, 9.0, 9.0], // row 10: filler
            [1.0, 0.0, 1.0], // row 11: cube vertex 5
        ];
        let flat: Vec<usize> = vec![
            4,
            9,
            2,
            7,
            usize::MAX, //
            4,
            7,
            8,
            0,
            usize::MAX, //
            4,
            0,
            11,
            9,
            usize::MAX, //
            9,
            11,
            5,
            2,
            usize::MAX, //
            2,
            5,
            8,
            7,
            usize::MAX, //
            0,
            8,
            5,
            11,
            usize::MAX, //
        ];
        let inside = [0.5, 0.5, 0.5];
        let outside = [1.5, 0.5, 0.5];
        assert!(point_in_phed(&inside, &coords, &flat));
        assert!(point_in_phed2(&inside, &coords, &flat));
        assert!(!point_in_phed(&outside, &coords, &flat));
        assert!(!point_in_phed2(&outside, &coords, &flat));
    }

    fn translated(p: &Polyhedron, t: [f64; 3]) -> Polyhedron {
        Polyhedron::unknown(
            p.iter().map(|v| [v[0] + t[0], v[1] + t[1], v[2] + t[2]]),
            (0..p.num_faces()).map(|i| p.faces[i].to_vec()),
        )
    }

    /// Applies an independent axis-aligned scale to each of the three coordinates.
    fn dilated(p: &Polyhedron, s: [f64; 3]) -> Polyhedron {
        Polyhedron::unknown(
            p.iter().map(|v| [v[0] * s[0], v[1] * s[1], v[2] * s[2]]),
            (0..p.num_faces()).map(|i| p.faces[i].to_vec()),
        )
    }

    /// Unit cube centered at the origin: `[-0.5, 0.5]^3`.
    fn centered_cube() -> Polyhedron {
        let c = unit_cube();
        translated(&c, [-0.5, -0.5, -0.5])
    }

    /// Perturbs every vertex of every face slightly out of the face plane, as real meshes' faces
    /// are never perfectly coplanar. Each face's vertices are displaced by `delta` along that
    /// face's Newell normal, breaking planarity while keeping the vertex topology intact. A `delta`
    /// of zero leaves the polyhedron unchanged.
    fn warped_faces(p: &Polyhedron, delta: f64) -> Polyhedron {
        let mut pts = p.iter().copied().collect::<Vec<_>>();
        for fi in 0..p.num_faces() {
            let face: Vec<usize> = p.faces[fi].to_vec();
            let (n, _d) = face_plane(&pts, &face);
            let norm = n[0].hypot(n[1]).hypot(n[2]);
            if norm == 0.0 {
                continue;
            }
            let u = [n[0] / norm, n[1] / norm, n[2] / norm];
            for &vi in &face {
                pts[vi][0] += delta * u[0];
                pts[vi][1] += delta * u[1];
                pts[vi][2] += delta * u[2];
            }
        }
        Polyhedron::unknown(pts, (0..p.num_faces()).map(|i| p.faces[i].to_vec()))
    }

    #[test]
    fn intersection_disjoint_cubes() {
        let a = unit_cube();
        let b = translated(&a, [2.0, 0.0, 0.0]);
        assert_eq!(a.convex_intersection_volume(&b), 0.0);
        assert_eq!(b.convex_intersection_volume(&a), 0.0);
    }

    /// Two warped cubes whose interiors genuinely overlap in a 3D region (offset along x by half a
    /// side) must still yield the full overlap volume — the coplanar-opposite skip must NOT eat a
    /// real intersection just because faces are warped. Warping both cubes the same way does not
    /// create antiparallel-coplanar shared faces, so the overlap must be preserved. (The warp is
    /// kept small enough to stay in the intersector's reliable range; larger warp degrades the
    /// intersection integration independently of the shared-face skip.)
    #[test]
    fn overlapping_warped_cubes_keep_volume() {
        for delta in [0.0, 1e-12, 1e-10, 1e-8, 1e-7] {
            let a = warped_faces(&unit_cube(), delta);
            let b = warped_faces(&translated(&unit_cube(), [0.5, 0.0, 0.0]), delta);
            let expected = 0.5;
            let got = a.convex_intersection_volume(&b);
            assert!(
                (got - expected).abs() <= 1e-5,
                "overlap {got} != {expected} with warp {delta}"
            );
            let got_ba = b.convex_intersection_volume(&a);
            assert!(
                (got_ba - expected).abs() <= 1e-5,
                "overlap (sym) {got_ba} != {expected} with warp {delta}"
            );
        }
    }

    /// Adjacent cubes (`[0,1]^3` and `[1,2]^3`) share the face `x == 1` and thus have zero overlap
    /// in 3D. In a real Voronoi/PHED mesh the two neighboring cells not only share that face but
    /// reference the *same* (possibly non-planar) polygon for it: cell 80's face 4 and its neighbor
    /// cell 468's face 1 are the identical 5 vertices in opposite winding. This is the regression
    /// guard for that bug — a genuinely shared, warped face must not leak a spurious pyramid.
    #[test]
    fn adjacent_cells_share_warped_face_no_overlap() {
        // Shared (warped) quad in the plane x = 1, made non-planar by pushing corners out of it.
        let mut shared = [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ];
        // Perturb a pair of opposite corners out of the plane to break planarity by `delta`.
        for delta in [0.0, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3] {
            shared[0][1] += delta;
            shared[0][2] += delta;
            shared[2][1] -= delta;
            shared[2][2] -= delta;
            // Cell A = cube [0,1]^3 with the warped shared face at x = 1.
            let a = Polyhedron::unknown(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0],
                    shared[0],
                    shared[1],
                    shared[2],
                    shared[3],
                ],
                vec![
                    vec![0, 3, 2, 1], // x = 0 (outward -x)
                    vec![4, 5, 6, 7], // x = 1, shared warped face (outward +x)
                    vec![0, 1, 5, 4], // y = 0
                    vec![2, 3, 7, 6], // y = 1
                    vec![1, 2, 6, 5], // z = 1
                    vec![3, 0, 4, 7], // z = 0
                ],
            );
            // Cell B = cube [1,2]^3 using the SAME warped shared face at x = 1 (opposite winding).
            let b = Polyhedron::unknown(
                [
                    [2.0, 0.0, 0.0],
                    [2.0, 1.0, 0.0],
                    [2.0, 1.0, 1.0],
                    [2.0, 0.0, 1.0],
                    shared[0],
                    shared[1],
                    shared[2],
                    shared[3],
                ],
                vec![
                    vec![4, 7, 6, 5], // x = 1, shared warped face (outward -x)
                    vec![0, 1, 2, 3], // x = 2
                    vec![0, 1, 5, 4], // y = 0
                    vec![2, 3, 7, 6], // y = 1
                    vec![1, 2, 6, 5], // z = 1
                    vec![3, 0, 4, 7], // z = 0
                ],
            );
            let got = a.convex_intersection_volume(&b);
            assert!(
                got.abs() < 1e-12,
                "shared warped face overlap {got} with warp {delta}"
            );
            let got_ba = b.convex_intersection_volume(&a);
            assert!(
                got_ba.abs() < 1e-12,
                "shared warped face overlap (sym) {got_ba} with warp {delta}"
            );
        }
    }

    #[test]
    fn intersection_identical_cubes() {
        let a = unit_cube();
        assert_abs_diff_eq!(a.convex_intersection_volume(&a), 1.0, epsilon = 1e-12);
    }

    /// Regression for the (tgt1399, src879) over-count. Two cells whose shared capacity face is
    /// fitted independently by each cell can disagree by a small warp: cell 1399's y=0 plane fitted
    /// to d = +2.877e-6 while its neighbor's identical face sat at d = 0 (both outward [0,-1,0]).
    /// The old hard `1e-9 * scale` offset tolerance failed to recognise these as the same coincident
    /// same-side face, so the identical clipped patch on the shared face was accumulated twice,
    /// over-stating the intersection volume by ~2.8e-5 (the +7.6% over-count). The warp-aware
    /// tolerance used by `plane_same_side` must absorb such small shared-face warps — while still
    /// rejecting faces that are genuinely tilted apart or offset beyond the warp scale.
    #[test]
    fn plane_same_side_absorbs_shared_face_warp() {
        let tol = 1e-4; // coplanar_tol with unit-scale geometry (radial >= 1)
        let a = Plane {
            n: [0.0, -1.0, 0.0],
            d: 2.877e-6,
        };
        let b = Plane {
            n: [0.0, -1.0, 0.0],
            d: 0.0,
        };
        // Coincident same-orientation faces offset only by the shared-face warp are the same plane.
        assert!(plane_same_side(a, b, tol));
        assert!(plane_same_side(b, a, tol));
        // A genuinely different orientation is not coincident.
        assert!(!plane_same_side(
            a,
            Plane {
                n: [1.0, 0.0, 0.0],
                d: 0.0
            },
            tol
        ));
        // A real offset (larger than the warp scale) is not coincident.
        assert!(!plane_same_side(
            a,
            Plane {
                n: [0.0, -1.0, 0.0],
                d: 1e-2
            },
            tol
        ));
    }

    proptest! {
        /// Self-intersection (a polyhedron intersected with itself) must return its own volume,
        /// whatever rigid or affine transform is applied. This exercises the epsilon handling in
        /// `convex_intersection_volume`, which must be scale-aware to survive large coordinates
        /// (floating-point error in the plane signed distances scales with the coordinate
        /// magnitude) and anisotropic dilation (which skews the box into a general parallelepiped).
        ///
        /// The ground truth is the analytic volume `sx * sy * sz`: starting from the unit cube
        /// (volume 1), rotation and translation preserve volume while a dilation by `[sx, sy, sz]`
        /// multiplies it by `sx * sy * sz`. This cross-check against an exact value is independent
        /// of the divergence-theorem sum in `volume()`, so it would catch a regression in either
        /// the clip-based intersector or the measure.
        #[test]
        fn intersection_self_volume_preserved(
            tx in -1e6f64..1e6,
            ty in -1e6f64..1e6,
            tz in -1e6f64..1e6,
            angle in -4.0f64..4.0,
            sx in 0.1f64..10.0,
            sy in 0.1f64..10.0,
            sz in 0.1f64..10.0,
        ) {
            let a = centered_cube();
            let a = rotated_about(&a, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], angle);
            let a = dilated(&a, [sx, sy, sz]);
            let a = translated(&a, [tx, ty, tz]);

            let expected = sx * sy * sz;
            let got = a.convex_intersection_volume(&a);
            let tol = 1e-9 * (expected.abs().max(1.0));
            prop_assert!(
                (got - expected).abs() <= tol,
                "self-intersection {got} != volume {expected} (tol {tol})"
            );
        }
    }

    proptest! {
        /// The measure (`Polyhedron::volume`) of a dilated, rotated, translated cube must equal the
        /// analytic volume `sx * sy * sz` (rotation and translation are volume-preserving, an
        /// anisotropic dilation multiplies the unit-cube volume by the product of the three factors).
        ///
        /// `volume()` recentres its divergence-theorem sum on the vertex centroid before summing
        /// (see `signed_volume6`), so it is translation-exact; this test verifies that recentring
        /// actually keeps the measure precise even for a small shape at large coordinates.
        #[test]
        fn measure_volume_preserved(
            tx in -1e6f64..1e6,
            ty in -1e6f64..1e6,
            tz in -1e6f64..1e6,
            angle in -4.0f64..4.0,
            sx in 0.1f64..10.0,
            sy in 0.1f64..10.0,
            sz in 0.1f64..10.0,
        ) {
            let a = centered_cube();
            let a = rotated_about(&a, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], angle);
            let a = dilated(&a, [sx, sy, sz]);
            let a = translated(&a, [tx, ty, tz]);

            let expected = sx * sy * sz;
            let got = a.volume();
            let tol = 1e-9 * (expected.abs().max(1.0));
            prop_assert!(
                (got - expected).abs() <= tol,
                "volume {got} != expected {expected} (tol {tol})"
            );
        }
    }

    proptest! {
        /// Cross-validates the two measure/intersection computations on the same transformed cube:
        /// `volume()` and `convex_intersection_volume(poly, poly)` must agree. This exercises both
        /// the divergence-theorem sum and the clip-based boundary integration on identical input.
        #[test]
        fn measure_matches_self_intersection(
            tx in -1e6f64..1e6,
            ty in -1e6f64..1e6,
            tz in -1e6f64..1e6,
            angle in -4.0f64..4.0,
            sx in 0.1f64..5.0,
            sy in 0.1f64..5.0,
            sz in 0.1f64..5.0,
        ) {
            let a = centered_cube();
            let a = rotated_about(&a, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], angle);
            let a = dilated(&a, [sx, sy, sz]);
            let a = translated(&a, [tx, ty, tz]);

            let vol = a.volume();
            let inter = a.convex_intersection_volume(&a);
            let tol = 1e-6 * (vol.abs().max(1.0));
            prop_assert!(
                (vol - inter).abs() <= tol,
                "volume {vol} != self-intersection {inter} (tol {tol})"
            );
        }
    }

    proptest! {
        /// Real meshes' faces are never perfectly planar; every vertex is usually a hair off its
        /// face's best-fit plane. This checks that a mildly warped (non-planar-faced) convex cube
        /// still satisfies the two invariants that matter for the transfer:
        ///   - `volume()` equals `convex_intersection_volume(poly, poly)` (both triangulate the
        ///     same warped faces with the divergence theorem),
        ///   - both are invariant under translation, despite the faces being non-planar.
        ///
        /// `delta` is a fraction of the cube's size: an absolute warp between 1e-12 and 1e-2 times
        /// the side length.
        #[test]
        fn non_planar_face_robust(
            delta_rel in 1e-12f64..1e-2,
            tx in -1e4f64..1e4,
            ty in -1e4f64..1e4,
            tz in -1e4f64..1e4,
            angle in -4.0f64..4.0,
            sx in 1.0f64..5.0,
            sy in 1.0f64..5.0,
            sz in 1.0f64..5.0,
        ) {
            let base = centered_cube();
            let base = rotated_about(&base, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], angle);
            let base = dilated(&base, [sx, sy, sz]);
            let delta = delta_rel;
            let a = warped_faces(&base, delta);
            let vol = a.volume();
            let inter = a.convex_intersection_volume(&a);
            let tol = 1e-6 * (vol.abs().max(1.0));

            // The two computations agree on the same warped shape.
            prop_assert!(
                (vol - inter).abs() <= tol,
                "with warp {delta}: volume {vol} != self-intersection {inter}"
            );

            // Both are translation-invariant: translating the warped shape must not change them.
            let at = translated(&a, [tx, ty, tz]);
            let vol_t = at.volume();
            prop_assert!(
                (vol_t - vol).abs() <= tol,
                "volume {vol} changed to {vol_t} under translation (warp {delta})"
            );
            let inter_t = at.convex_intersection_volume(&at);
            prop_assert!(
                (inter_t - inter).abs() <= tol,
                "self-intersection {inter} changed to {inter_t} under translation (warp {delta})"
            );
        }
    }

    #[test]
    fn intersection_half_offset_along_x() {
        let a = unit_cube();
        let b = translated(&a, [0.5, 0.0, 0.0]);
        assert_abs_diff_eq!(a.convex_intersection_volume(&b), 0.5, epsilon = 1e-12);
    }

    #[test]
    fn intersection_cube_fully_inside_larger() {
        let a = centered_cube();
        let big = Polyhedron::convex(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
            ],
            [
                vec![0, 3, 2, 1],
                vec![4, 5, 6, 7],
                vec![0, 1, 5, 4],
                vec![2, 3, 7, 6],
                vec![1, 2, 6, 5],
                vec![3, 0, 4, 7],
            ],
        );
        assert_abs_diff_eq!(a.convex_intersection_volume(&big), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(big.convex_intersection_volume(&a), 1.0, epsilon = 1e-12);
    }

    /// Two identical unit cubes, one rotated 45° about the z axis around their shared center
    /// (0.5, 0.5, 0.5). The intersection is an octagon-base column extruded in z with volume
    /// `2 * (sqrt(2) - 1)` (the octagon is the intersection of two centered unit squares at 45°).
    #[test]
    fn intersection_rotated_cube_known_value() {
        let a = unit_cube();
        let b = rotated_about(
            &a,
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_4,
        );
        let expected = 2.0 * (2.0f64.sqrt() - 1.0);
        assert_abs_diff_eq!(a.convex_intersection_volume(&b), expected, epsilon = 1e-9);
    }

    fn rotated_about(p: &Polyhedron, center: [f64; 3], axis: [f64; 3], angle: f64) -> Polyhedron {
        let (c, s) = (angle.cos(), angle.sin());
        // Normalize the axis: Rodrigues' formula requires a unit vector, otherwise the transform
        // is not a pure rotation and does not preserve volume.
        let norm = axis[0].hypot(axis[1]).hypot(axis[2]);
        let (ax, ay, az) = (axis[0] / norm, axis[1] / norm, axis[2] / norm);
        let rot = |v: [f64; 3]| -> [f64; 3] {
            let w = [v[0] - center[0], v[1] - center[1], v[2] - center[2]];
            // Rodrigues' rotation formula.
            let dot = ax * w[0] + ay * w[1] + az * w[2];
            let cross = [
                ay * w[2] - az * w[1],
                az * w[0] - ax * w[2],
                ax * w[1] - ay * w[0],
            ];
            let r = [
                w[0] * c + cross[0] * s + ax * dot * (1.0 - c),
                w[1] * c + cross[1] * s + ay * dot * (1.0 - c),
                w[2] * c + cross[2] * s + az * dot * (1.0 - c),
            ];
            [r[0] + center[0], r[1] + center[1], r[2] + center[2]]
        };
        Polyhedron::unknown(
            p.iter().map(|&v| rot(v)),
            (0..p.num_faces()).map(|i| p.faces[i].to_vec()),
        )
    }

    #[test]
    fn intersection_symmetry() {
        let a = unit_cube();
        let b = rotated_about(
            &a,
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_4,
        );
        let v_ab = a.convex_intersection_volume(&b);
        let v_ba = b.convex_intersection_volume(&a);
        assert_abs_diff_eq!(v_ab, v_ba, epsilon = 1e-12);
    }

    /// Cross-validates the analytic volume against a Monte-Carlo sampling over the bounding box of
    /// an oblique (rotated) overlap where a closed-form value is derived independently.
    #[test]
    fn intersection_matches_monte_carlo() {
        let a = unit_cube();
        let b = rotated_about(
            &a,
            [0.5, 0.5, 0.5],
            [0.0, 0.0, 1.0],
            std::f64::consts::FRAC_PI_4,
        );
        let exact = a.convex_intersection_volume(&b);
        let expected = 2.0 * (2.0f64.sqrt() - 1.0);
        assert_abs_diff_eq!(exact, expected, epsilon = 1e-9);

        // Deterministic pseudo-random sampler over the unit cube [0,1]^3, which contains the whole
        // intersection (it is a subset of `a`). Count points inside both polyhedra.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };
        let na = 4_000_000usize;
        let mut hits = 0usize;
        for _ in 0..na {
            let pt = [next(), next(), next()];
            if a.contains(&pt)
                && (0..b.num_faces()).all(|fi| {
                    let (n, d) = super::face_plane(&b.points, &b.faces[fi]);
                    n[0] * pt[0] + n[1] * pt[1] + n[2] * pt[2] + d <= 0.0
                })
            {
                hits += 1;
            }
        }
        let estimate = (hits as f64) / (na as f64);
        assert!(
            (estimate - exact).abs() < 0.02,
            "Monte-Carlo estimate {estimate} should be within 0.02 of exact {exact}"
        );
    }

    /// Cross-validates the intersector on real hexa-mesh elements (`make_imesh_3d` + `to_polyhedron`
    /// is the `polyze`-style pipeline that turns each HEX8 cell into a convex polyhedron).
    ///
    /// Two axis-aligned hexa meshes overlap; the cell-pair intersections partition the mesh
    /// intersection, so the sum over all pairs equals the overlap of the two cube boxes, and every
    /// pair is itself an axis-aligned box with a closed-form volume.
    #[test]
    fn intersection_hexa_mesh_cell_pairs() {
        use crate::element_traits::ElementGeo;
        use crate::mesh_examples::make_imesh_3d;

        let mesh_a = make_imesh_3d(2); // [0,1]^3, 8 cells of side 0.5
        let mesh_b: crate::mesh::UMesh = crate::tools::grid::RegularUMeshBuilder::new()
            .add_axis(vec![0.25, 0.75, 1.25]) // x shifted +0.25
            .add_axis(vec![0.0, 0.5, 1.0])
            .add_axis(vec![0.0, 0.5, 1.0])
            .build();

        let cells_a: Vec<Polyhedron> = mesh_a.elements().map(|e| e.to_polyhedron()).collect();
        let cells_b: Vec<Polyhedron> = mesh_b.elements().map(|e| e.to_polyhedron()).collect();

        let mut total = 0.0;
        for pa in &cells_a {
            for pb in &cells_b {
                let [a_lo, a_hi] = pa.bounds();
                let [b_lo, b_hi] = pb.bounds();
                let mut vol = 1.0;
                for k in 0..3 {
                    let lo = a_lo[k].max(b_lo[k]);
                    let hi = a_hi[k].min(b_hi[k]);
                    vol *= (hi - lo).max(0.0);
                }
                let got = pa.convex_intersection_volume(pb);
                assert_abs_diff_eq!(got, vol, epsilon = 1e-10);
                total += got;
                // Symmetry must hold per pair.
                assert_abs_diff_eq!(pb.convex_intersection_volume(pa), got, epsilon = 1e-12);
            }
        }
        // The union over cell pairs is the overlap of [0,1]^3 with [0.25,1.25]x[0,1]x[0,1].
        assert_abs_diff_eq!(total, 0.75, epsilon = 1e-10);
    }
}
