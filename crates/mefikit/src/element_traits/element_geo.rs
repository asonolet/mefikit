//! Geometric operations for mesh elements.
//!
//! Provides the [`ElementGeo`] trait for coordinate access, measures,
//! bounding boxes, and centroid calculations. All computations delegate to the owned geometry
//! types of the [`crate::geometry`] module.

use super::element_topo::ElementTopo;
use crate::geometry::{
    Polygon, Polyhedron, Segment, arc_length2, area_polygon3, area_quad2, area_tri2, bounds_iter,
    convex_polygon_contains2, hex_contains, hex_volume, in_quadratic_polygon,
    quadratic_polygon_area2, signed_area2, tet_contains, tet_volume, vertex_centroid,
};
use crate::mesh::{Dimension, ElementLike, ElementType};

use nalgebra as na;

/// Geometric operations for mesh elements.
///
/// Extends [`ElementLike`] with methods for accessing coordinates as nalgebra
/// points, computing measures (length/area/volume), bounding boxes, and centroids.
pub trait ElementGeo<'a>: ElementLike<'a> + ElementTopo<'a> {
    /// Returns the i-th coordinate as a 1D point.
    ///
    /// # Panics
    /// Panics if the coordinate is not 1D.
    #[inline(always)]
    fn coord1(&self, i: usize) -> na::Point1<f64> {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 1);
        na::Point1::from_slice(coord)
    }
    /// Returns the i-th coordinate as a 2D point.
    ///
    /// # Panics
    /// Panics if the coordinate is not 2D.
    #[inline(always)]
    fn coord2(&self, i: usize) -> na::Point2<f64> {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 2);
        na::Point2::from_slice(coord)
    }

    /// Returns the i-th coordinate as a 2D array reference.
    ///
    /// # Panics
    /// Panics if the coordinate is not 2D.
    #[inline(always)]
    fn coord2_ref(&self, i: usize) -> &[f64; 2] {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 2);
        coord.try_into().unwrap()
    }

    /// Returns an iterator over all coordinates as 2D array references.
    fn coords2(&self) -> impl ExactSizeIterator<Item = &[f64; 2]> {
        (0..self.connectivity().len()).map(|i| self.coord2_ref(i))
    }

    /// Returns the i-th coordinate as a 3D point.
    ///
    /// # Panics
    /// Panics if the coordinate is not 3D.
    #[inline(always)]
    fn coord3(&self, i: usize) -> na::Point3<f64> {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 3);
        na::Point3::from_slice(coord)
    }

    /// Returns the i-th coordinate as a 3D array reference.
    ///
    /// # Panics
    /// Panics if the coordinate is not 3D.
    #[inline(always)]
    fn coord3_ref(&self, i: usize) -> &[f64; 3] {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 3);
        coord.try_into().unwrap()
    }

    /// Returns an iterator over all coordinates as 3D array references.
    fn coords3(&self) -> impl Iterator<Item = &[f64; 3]> {
        self.connectivity()
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                match *c {
                    usize::MAX => None, // Skip face boundaries in polyhedra
                    _ => Some(i),
                }
            })
            .map(|i| self.coord3_ref(i))
    }

    /// Returns an iterator over all coordinates as slices.
    fn coords(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        (0..self.connectivity().len()).map(|i| self.coord(i))
    }

    /// Builds the element as a 2D polygon.
    ///
    /// For the quadratic element types this only uses every connectivity node as a vertex,
    /// so the mid-side nodes become spurious vertices; use [`Self::to_quadratic_polygon2`]
    /// instead.
    fn to_polygon2(&self) -> Polygon<2> {
        Polygon::with_convexity(
            self.coords2().copied(),
            self.element_type().known_convexity(),
        )
    }

    /// The element's vertex coordinates: the first [`ElementLike::num_vertices`] nodes of
    /// the connectivity (mid-side nodes and the TRI7/QUAD9 centre node excluded).
    fn vertices2(&self) -> Vec<[f64; 2]> {
        (0..self.num_vertices())
            .map(|i| *self.coord2_ref(i))
            .collect()
    }

    /// The element's vertex coordinates in 3D (see [`Self::vertices2`]).
    fn vertices3(&self) -> Vec<[f64; 3]> {
        (0..self.num_vertices())
            .map(|i| *self.coord3_ref(i))
            .collect()
    }

    /// Builds the element as its quadratic polygon `[v0..v{n-1}, m0..m{n-1}]`: the `n`
    /// vertices first, then the `n` mid-side nodes in edge order (edge `i` joins vertex `i`
    /// to vertex `i+1` and is shaped by mid-side node `i`). A TRI7/QUAD9 centre node is
    /// excluded, as it takes no part in the boundary.
    ///
    /// Only meaningful for the quadratic 2D types (TRI6, TRI7, QUAD8, QUAD9, QPOLY).
    fn to_quadratic_polygon2(&self) -> Vec<[f64; 2]> {
        let n = self.num_vertices();
        debug_assert!(self.space_dimension() == 2 && self.connectivity().len() >= 2 * n);
        let mut pgon = self.vertices2();
        pgon.extend((n..2 * n).map(|i| *self.coord2_ref(i)));
        pgon
    }

    /// Builds the element as a 3D polyhedron from its faces.
    fn to_polyhedron(&self) -> Polyhedron {
        let coords: Vec<[f64; 3]> = self.coords3().copied().collect();
        let mut local: Vec<usize> = Vec::with_capacity(self.num_nodes());
        for &node in self.connectivity() {
            if node != usize::MAX {
                local.push(node);
            }
        }
        let mut faces = crate::mesh::IndirectIndexOwned::new();
        for (_, face_conn) in self.subentities(Some(Dimension::D1)) {
            for face in face_conn.iter() {
                let face: Vec<usize> = face
                    .iter()
                    .map(|&node| local.iter().position(|&n| n == node).unwrap())
                    .collect();
                faces.push(&face);
            }
        }
        Polyhedron::with_indirect_index(coords, faces, self.element_type().known_convexity())
    }

    /// Computes the geometric measure of the element in 1D space.
    ///
    /// Returns length for 1D elements. A SEG3 length is the polyline through its mid-side
    /// node (in 1D space the three points are always collinear, so the arc degenerates).
    fn measure1(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => Segment::new(self.coord1(0).into(), self.coord1(1).into()).length(),
            SEG3 => {
                let [a, b, m] = [self.coord1(0).x, self.coord1(1).x, self.coord1(2).x];
                (m - a).abs() + (b - m).abs()
            }
            other => unimplemented!("measure1 is not implemented for element type {other:?}"),
        }
    }

    /// Computes the geometric measure of the element in 2D space.
    ///
    /// Returns length for 1D elements and area for 2D elements. Quadratic elements use
    /// their exact arc-bounded SEG3 length and quadratic-polygon area.
    fn measure2(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => Segment::new(*self.coord2_ref(0), *self.coord2_ref(1)).length(),
            SEG3 => arc_length2(self.coord2_ref(0), self.coord2_ref(2), self.coord2_ref(1)),
            TRI3 => area_tri2(self.coord2_ref(0), self.coord2_ref(1), self.coord2_ref(2)),
            QUAD4 => area_quad2(&[
                *self.coord2_ref(0),
                *self.coord2_ref(1),
                *self.coord2_ref(2),
                *self.coord2_ref(3),
            ]),
            PGON => self.to_polygon2().area(),
            TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY => {
                quadratic_polygon_area2(&self.to_quadratic_polygon2())
            }
            other => unimplemented!("measure2 is not implemented for element type {other:?}"),
        }
    }

    /// Computes the geometric measure of the element in 3D space.
    ///
    /// Returns length for 1D elements, area for 2D elements, and volume for 3D elements.
    fn measure3(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => Segment::new(*self.coord3_ref(0), *self.coord3_ref(1)).length(),
            TRI3 => area_polygon3(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
            ]),
            QUAD4 => area_polygon3(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
                *self.coord3_ref(3),
            ]),
            TET4 => tet_volume(
                self.coord3_ref(0),
                self.coord3_ref(1),
                self.coord3_ref(2),
                self.coord3_ref(3),
            ),
            HEX8 => hex_volume(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
                *self.coord3_ref(3),
                *self.coord3_ref(4),
                *self.coord3_ref(5),
                *self.coord3_ref(6),
                *self.coord3_ref(7),
            ]),
            PHED => self.to_polyhedron().volume(),
            other => unimplemented!("measure3 is not implemented for element type {other:?}"),
        }
    }

    /// Returns the element's connectivity reordered to the canonical positive-volume
    /// (VTK) convention expected by [`Self::measure`], `tet_volume` and `hex_volume`:
    /// 2D cells wound counter-clockwise and 3D cells with a right-handed base face.
    ///
    /// The decision is made from the coordinates, so this *repairs* ill-formed windings
    /// (e.g. the left-handed HEX8 a purely topological `from_poly` produces from an
    /// outward-wound PHED) and leaves well-formed elements unchanged. The topology-only
    /// hot paths (`unpolyze`, `to_poly`, `subentities`) never pay for this.
    fn oriented_positive_connectivity(&self) -> (ElementType, Vec<usize>) {
        use ElementType::*;
        let et = self.element_type();
        let co = self.connectivity();
        match et {
            TRI3 | QUAD4 | PGON => {
                let pts: Vec<[f64; 2]> = self.coords2().copied().collect();
                if signed_area2(&pts) < 0.0 {
                    (et, co.iter().rev().copied().collect())
                } else {
                    (et, co.to_vec())
                }
            }
            TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY => {
                if quadratic_polygon_area2(&self.to_quadratic_polygon2()) < 0.0 {
                    // Reverse the vertex block, then re-pair the mid-side nodes with the
                    // reversed edges: the edge `(v_i, v_{i+1})` of the reversed cycle is the
                    // reverse of edge `(v_{n-2-i}, v_{n-1-i})`, shaped by mid-side node
                    // `m_{n-2-i}`. A TRI7/QUAD9 centre node (if any) stays last.
                    let n = self.num_vertices();
                    let mut conn = co.to_vec();
                    conn[..n].reverse();
                    conn[n..2 * n].reverse();
                    conn[n..2 * n].rotate_left(1);
                    (et, conn)
                } else {
                    (et, co.to_vec())
                }
            }
            TET4 => {
                let v = tet_volume(
                    self.coord3_ref(0),
                    self.coord3_ref(1),
                    self.coord3_ref(2),
                    self.coord3_ref(3),
                );
                if v < 0.0 {
                    (et, vec![co[0], co[1], co[3], co[2]])
                } else {
                    (et, co.to_vec())
                }
            }
            HEX8 => {
                let v = hex_volume(&[
                    *self.coord3_ref(0),
                    *self.coord3_ref(1),
                    *self.coord3_ref(2),
                    *self.coord3_ref(3),
                    *self.coord3_ref(4),
                    *self.coord3_ref(5),
                    *self.coord3_ref(6),
                    *self.coord3_ref(7),
                ]);
                if v < 0.0 {
                    // Swap the two bottom corners and the matching top corners to undo
                    // the left-handed inversion (involution on the index order).
                    (
                        et,
                        vec![co[0], co[3], co[2], co[1], co[4], co[7], co[6], co[5]],
                    )
                } else {
                    (et, co.to_vec())
                }
            }
            _ => (et, co.to_vec()),
        }
    }

    /// Returns `true` if the given point lies inside the element.
    fn is_point_inside(&self, point: &[f64]) -> bool {
        use ElementType::*;
        match self.element_type() {
            VERTEX => self.coord(0) == point,
            TRI3 => convex_polygon_contains2(
                &[
                    *self.coord2_ref(0),
                    *self.coord2_ref(1),
                    *self.coord2_ref(2),
                ],
                &[point[0], point[1]],
            ),
            QUAD4 => convex_polygon_contains2(
                &[
                    *self.coord2_ref(0),
                    *self.coord2_ref(1),
                    *self.coord2_ref(2),
                    *self.coord2_ref(3),
                ],
                &[point[0], point[1]],
            ),
            PGON => self.to_polygon2().contains(&[point[0], point[1]]),
            TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY => {
                in_quadratic_polygon(&[point[0], point[1]], &self.to_quadratic_polygon2())
            }
            TET4 => tet_contains(
                &[point[0], point[1], point[2]],
                self.coord3_ref(0),
                self.coord3_ref(1),
                self.coord3_ref(2),
                self.coord3_ref(3),
            ),
            HEX8 => hex_contains(
                &[point[0], point[1], point[2]],
                &[
                    *self.coord3_ref(0),
                    *self.coord3_ref(1),
                    *self.coord3_ref(2),
                    *self.coord3_ref(3),
                    *self.coord3_ref(4),
                    *self.coord3_ref(5),
                    *self.coord3_ref(6),
                    *self.coord3_ref(7),
                ],
            ),
            PHED => self
                .to_polyhedron()
                .contains(&[point[0], point[1], point[2]]),
            other => {
                unimplemented!("is_point_inside is not implemented for element type {other:?}")
            }
        }
    }

    fn bounds2(&self) -> [[f64; 2]; 2] {
        bounds_iter(self.coords2().copied())
    }

    fn bounds3(&self) -> [[f64; 3]; 2] {
        bounds_iter(self.coords3().copied())
    }

    /// Computes the 2D vertex centroid of the element: the arithmetic mean of its
    /// [`Self::vertices2`] (mid-side nodes and the TRI7/QUAD9 centre node are excluded).
    fn centroid2(&self) -> [f64; 2] {
        use ElementType::*;
        match self.element_type() {
            VERTEX => *self.coord2_ref(0),
            SEG2 | SEG3 => Segment::new(*self.coord2_ref(0), *self.coord2_ref(1)).midpoint(),
            TRI3 => vertex_centroid(&[
                *self.coord2_ref(0),
                *self.coord2_ref(1),
                *self.coord2_ref(2),
            ]),
            QUAD4 => vertex_centroid(&[
                *self.coord2_ref(0),
                *self.coord2_ref(1),
                *self.coord2_ref(2),
                *self.coord2_ref(3),
            ]),
            PGON => self.to_polygon2().centroid(),
            // The higher-order regulars and the 1D types resolve their vertex block
            // through `num_vertices()`.
            SEG4 | SPLINE | TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY => {
                vertex_centroid(&self.vertices2())
            }
            other => unimplemented!("centroid2 is not implemented for element type {other:?}"),
        }
    }

    /// Computes the 3D vertex centroid of the element: the arithmetic mean of its
    /// [`Self::vertices3`] (mid-side nodes are excluded).
    fn centroid3(&self) -> [f64; 3] {
        use ElementType::*;
        match self.element_type() {
            VERTEX => *self.coord3_ref(0),
            SEG2 | SEG3 => Segment::new(*self.coord3_ref(0), *self.coord3_ref(1)).midpoint(),
            TRI3 => vertex_centroid(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
            ]),
            QUAD4 => vertex_centroid(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
                *self.coord3_ref(3),
            ]),
            TET4 => vertex_centroid(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
                *self.coord3_ref(3),
            ]),
            HEX8 => vertex_centroid(&[
                *self.coord3_ref(0),
                *self.coord3_ref(1),
                *self.coord3_ref(2),
                *self.coord3_ref(3),
                *self.coord3_ref(4),
                *self.coord3_ref(5),
                *self.coord3_ref(6),
                *self.coord3_ref(7),
            ]),
            PHED => self.to_polyhedron().centroid(),
            // The higher-order regulars, the quadratic 2D cells and the 1D types resolve
            // their vertex block through `num_vertices()`.
            SEG4 | SPLINE | TRI6 | TRI7 | QUAD8 | QUAD9 | QPOLY | TET10 | HEX21 => {
                vertex_centroid(&self.vertices3())
            }
            other => unimplemented!("centroid3 is not implemented for element type {other:?}"),
        }
    }
}

impl<'a, T> ElementGeo<'a> for T where T: ElementLike<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Element, ElementType};
    use approx::assert_abs_diff_eq;
    use ndarray as nd;
    use proptest::prelude::*;

    #[test]
    fn test_coord2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let p0 = elem.coord2(0);
        assert_eq!(p0, na::Point2::new(0.0, 0.0));
        let p1 = elem.coord2(1);
        assert_eq!(p1, na::Point2::new(1.0, 0.0));
    }

    #[test]
    fn test_coord2_ref() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let c0: &[f64; 2] = elem.coord2_ref(0);
        assert_eq!(c0, &[0.0, 0.0]);
    }

    #[test]
    fn test_coords2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let coords: Vec<_> = elem.coords2().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coord3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let p0 = elem.coord3(0);
        assert_eq!(p0, na::Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_coords3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let coords: Vec<_> = elem.coords3().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coords() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let coords: Vec<_> = elem.coords().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_measure2_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = &[0, 1, 3, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::QUAD4, &groups);
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure2_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::SEG2, &groups);
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centroid2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TRI3, &groups);
        let centroid = elem.centroid2();
        assert_abs_diff_eq!(centroid[0], 1.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(centroid[1], 1.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centroid3() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TET4, &groups);
        let centroid = elem.centroid3();
        assert_abs_diff_eq!(centroid[0], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(centroid[1], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(centroid[2], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn is_point_inside_tet4_matches_as_polyhedron() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::TET4, &groups);
        let phed = elem.to_polyhedron();
        for p in [
            [0.25, 0.25, 0.25],
            [0.1, 0.1, 0.1],
            [0.75, 0.75, 0.75],
            [1.5, 0.5, 0.5],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.5],
        ] {
            assert_eq!(
                elem.is_point_inside(&p),
                phed.contains(&p),
                "TET4 is_point_inside must match as_polyhedron().contains at {p:?}"
            );
        }
    }

    #[test]
    fn is_point_inside_hex8_matches_as_polyhedron() {
        let mesh = crate::mesh_examples::make_imesh_3d(2);
        for elem in mesh.elements() {
            let phed = elem.to_polyhedron();
            for p in [
                [0.25, 0.25, 0.25],
                [0.75, 0.75, 0.75],
                [1.5, 1.5, 1.5],
                [0.0, 0.0, 0.0],
                [-0.5, 0.5, 0.5],
                [0.5, 0.0, 0.5],
            ] {
                assert_eq!(
                    elem.is_point_inside(&p),
                    phed.contains(&p),
                    "HEX8 is_point_inside must match as_polyhedron().contains at {p:?}"
                );
            }
        }
    }

    #[test]
    fn as_polyhedron_maps_global_to_local_faces() {
        let expected_faces = [
            [0, 3, 2, 1],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [2, 3, 7, 6],
            [1, 2, 6, 5],
            [3, 0, 4, 7],
        ];
        let mesh = crate::mesh_examples::make_imesh_3d(2);
        for elem in mesh.elements() {
            let phed = elem.to_polyhedron();
            assert_eq!(phed.num_faces(), 6);
            let coords: Vec<[f64; 3]> = elem.coords3().copied().collect();
            assert_abs_diff_eq!(phed.volume(), hex_volume(&coords.try_into().unwrap()),);
            for (i, row) in expected_faces.iter().enumerate() {
                let expected: Vec<[f64; 3]> = row.iter().map(|&k| *elem.coord3_ref(k)).collect();
                let actual: Vec<[f64; 3]> = phed.face(i).iter().copied().collect();
                assert_eq!(
                    actual,
                    expected,
                    "face {i} must resolve the element's global connectivity to local vertex \
                     positions (element {:?}, local face {row:?})",
                    elem.connectivity()
                );
            }
        }
    }

    #[test]
    fn as_polyhedron_phed_uses_local_connectivity() {
        // Unit cube whose 8 vertices sit at scattered rows of the global coords array, so the
        // global node ids (4, 9, 2, ...) do not match local indices (0..8).
        let coords = nd::array![
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
        let conn: &[usize] = &[
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &family, conn, ElementType::PHED, &groups);
        let phed = elem.to_polyhedron();
        assert_eq!(phed.num_faces(), 6);
        assert_abs_diff_eq!(phed.volume(), 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(phed.face(0).area(), 1.0, epsilon = 1e-12);
        assert!(elem.is_point_inside(&[0.5, 0.5, 0.5]));
        assert!(!elem.is_point_inside(&[1.5, 0.5, 0.5]));
        assert!(!elem.is_point_inside(&[0.5, 1.5, 0.5]));
    }

    #[test]
    fn test_oriented_positive_connectivity_2d() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        // A clockwise quad must be repaired to a counter-clockwise one; the exact cyclic
        // rotation is free, only the winding (positive signed area) is guaranteed.
        let elem = Element::new(
            0,
            coords.view(),
            &family,
            &[0, 3, 2, 1],
            ElementType::QUAD4,
            &groups,
        );
        let (et, conn) = elem.oriented_positive_connectivity();
        assert_eq!(et, ElementType::QUAD4);
        let mut sorted = conn.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
        // Reconstructed from the returned connectivity, the quad has positive area.
        let order: Vec<[f64; 2]> = conn
            .iter()
            .map(|&i| {
                let c = coords.row(i);
                [c[0], c[1]]
            })
            .collect();
        assert!(signed_area2(&order) > 0.0);
        // The counter-clockwise ordering is left untouched.
        let elem = Element::new(
            0,
            coords.view(),
            &family,
            &[0, 1, 2, 3],
            ElementType::QUAD4,
            &groups,
        );
        assert_eq!(
            elem.oriented_positive_connectivity(),
            (ElementType::QUAD4, vec![0, 1, 2, 3])
        );
    }

    #[test]
    fn test_oriented_positive_connectivity_tet4() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        // Left-handed [0, 1, 3, 2] is repaired to the right-handed [0, 1, 2, 3].
        for (conn, expected) in [
            (&[0usize, 1, 2, 3][..], vec![0, 1, 2, 3]),
            (&[0, 1, 3, 2][..], vec![0, 1, 2, 3]),
        ] {
            let elem = Element::new(0, coords.view(), &family, conn, ElementType::TET4, &groups);
            assert_eq!(
                elem.oriented_positive_connectivity(),
                (ElementType::TET4, expected)
            );
        }
    }

    #[test]
    fn test_oriented_positive_connectivity_hex8() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let canonical = vec![0usize, 1, 2, 3, 4, 5, 6, 7];
        // A well-formed hexa is returned unchanged...
        let elem = Element::new(
            0,
            coords.view(),
            &family,
            &[0usize, 1, 2, 3, 4, 5, 6, 7],
            ElementType::HEX8,
            &groups,
        );
        assert_eq!(
            elem.oriented_positive_connectivity(),
            (ElementType::HEX8, canonical.clone())
        );
        // ...and the left-handed inversion produced by the topological from_poly of an
        // outward-wound PHED is repaired to the canonical VTK ordering.
        let elem = Element::new(
            0,
            coords.view(),
            &family,
            &[0usize, 3, 2, 1, 4, 7, 6, 5],
            ElementType::HEX8,
            &groups,
        );
        assert_eq!(
            elem.oriented_positive_connectivity(),
            (ElementType::HEX8, canonical)
        );
    }

    #[test]
    fn test_measure2_seg3_arc() {
        // Quarter circle of radius 1 centred at the origin; SEG3 layout is [v0, v1, m].
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let coords = nd::array![[1.0, 0.0], [0.0, 1.0], [s, s]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(0, coords.view(), &0, &[0, 1, 2], ElementType::SEG3, &groups);
        assert_abs_diff_eq!(
            elem.measure2(),
            std::f64::consts::FRAC_PI_2,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_quad8_curved_edge_measure_and_containment() {
        // The bottom edge bulges down to y = -0.4 at its midpoint; the other edges are
        // straight. The shape is the flat unit square minus the chord plus the arc bulge.
        let coords = nd::array![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, -0.4],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            &0,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            ElementType::QUAD8,
            &groups,
        );
        assert!(elem.measure2() > 1.0);
        assert!(elem.is_point_inside(&[0.5, -0.1]));
        assert!(!elem.is_point_inside(&[0.5, -0.6]));
    }

    #[test]
    fn test_centroid2_quadratic_ignores_mid_nodes() {
        // TRI6 vertices at (0,0), (3,0), (0,3); mid-side nodes are irrelevant to the
        // vertex centroid.
        let coords = nd::array![
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 3.0],
            [9.0, 9.0],
            [9.0, 9.0],
            [9.0, 9.0]
        ];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            &0,
            &[0, 1, 2, 3, 4, 5],
            ElementType::TRI6,
            &groups,
        );
        assert_abs_diff_eq!(elem.centroid2()[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(elem.centroid2()[1], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_oriented_positive_connectivity_quad8() {
        let coords = nd::array![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ];
        let groups = crate::mesh::ArcGroups::new();
        // Clockwise vertex block with the mid-side nodes matching the reversed edges.
        let elem = Element::new(
            0,
            coords.view(),
            &0,
            &[3, 2, 1, 0, 6, 5, 4, 7],
            ElementType::QUAD8,
            &groups,
        );
        assert_eq!(
            elem.oriented_positive_connectivity(),
            (ElementType::QUAD8, vec![0, 1, 2, 3, 4, 5, 6, 7])
        );
    }

    // -------------------------------------------------------------------------------------
    // `oriented_positive_connectivity` must reverse a clockwise quadratic cell without ever
    // mis-pairing its mid-side nodes with the wrong edges. The earlier bug reversed the whole
    // mid-node block instead of re-pairing it, silently changing the geometry — which none of
    // the fixed hand-written tests caught because their shapes were too symmetric. These
    // properties compare the arc set directly and check area/idempotence on random shapes.
    // -------------------------------------------------------------------------------------

    /// A random star-shaped quadratic cell: `n` vertices in counter-clockwise angular order
    /// plus one mid-side node per edge, offset along the chord normal by a bounded, non-zero
    /// fraction of the chord length.
    fn arb_quadratic_coords(n: usize) -> impl Strategy<Value = Vec<[f64; 2]>> {
        (
            prop::collection::vec(0.0f64..std::f64::consts::TAU, n),
            prop::collection::vec(0.5f64..2.0, n),
            prop::array::uniform2(-1.0f64..1.0),
            prop::collection::vec(prop_oneof![-0.3f64..-0.05, 0.05f64..0.3], n),
        )
            .prop_map(move |(angles, radii, center, offs)| {
                let mut pairs: Vec<(f64, f64)> = angles.into_iter().zip(radii).collect();
                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let verts: Vec<[f64; 2]> = pairs
                    .iter()
                    .map(|&(a, r)| [center[0] + r * a.cos(), center[1] + r * a.sin()])
                    .collect();
                let mut out = verts.clone();
                for i in 0..n {
                    let p0 = verts[i];
                    let p1 = verts[(i + 1) % n];
                    let d = [p1[0] - p0[0], p1[1] - p0[1]];
                    let mid = [(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0];
                    let off = offs[i];
                    out.push([mid[0] + off * d[1], mid[1] - off * d[0]]);
                }
                out
            })
    }

    fn coords_array(pts: &[[f64; 2]]) -> nd::Array2<f64> {
        nd::Array2::from_shape_vec(
            (pts.len(), 2),
            pts.iter().flat_map(|p| p.iter().copied()).collect(),
        )
        .unwrap()
    }

    /// Normalised key of the arc joining `a` to `b` through `m`: the endpoints are sorted so
    /// that a reversed edge keys identically.
    fn edge_key(a: [f64; 2], m: [f64; 2], b: [f64; 2]) -> [f64; 6] {
        let (lo, hi) = if a[0] < b[0] || (a[0] == b[0] && a[1] <= b[1]) {
            (a, b)
        } else {
            (b, a)
        };
        [lo[0], lo[1], m[0], m[1], hi[0], hi[1]]
    }

    /// The multiset of arcs (as normalised keys), sorted for comparison.
    fn sorted_edge_keys(pgon: &[[f64; 2]]) -> Vec<[f64; 6]> {
        let n = pgon.len() / 2;
        let mut keys: Vec<[f64; 6]> = (0..n)
            .map(|i| edge_key(pgon[i], pgon[n + i], pgon[(i + 1) % n]))
            .collect();
        keys.sort_by(|x, y| {
            for k in 0..6 {
                match x[k].partial_cmp(&y[k]).unwrap() {
                    std::cmp::Ordering::Equal => continue,
                    o => return o,
                }
            }
            std::cmp::Ordering::Equal
        });
        keys
    }

    fn reorient_case(pts: &[[f64; 2]], etype: ElementType, n: usize, flip: bool) {
        let coords = coords_array(pts);
        let groups = crate::mesh::ArcGroups::new();
        let identity: Vec<usize> = (0..2 * n).collect();

        // A clockwise traversal of the *same* shape: reverse the vertex block, then re-pair
        // the mid-side nodes with the reversed edges.
        let conn: Vec<usize> = if flip {
            let mut c: Vec<usize> = (0..n).rev().collect();
            let mut mids: Vec<usize> = (n..2 * n).rev().collect();
            mids.rotate_left(1);
            c.extend(mids);
            c
        } else {
            identity.clone()
        };

        let elem = Element::new(0, coords.view(), &0, &conn, etype, &groups);
        let (et, conn2) = elem.oriented_positive_connectivity();
        assert_eq!(et, etype);

        let canonical = Element::new(0, coords.view(), &0, &identity, etype, &groups);
        let reoriented = Element::new(0, coords.view(), &0, &conn2, etype, &groups);

        // Orientation is fixed, and the measure (area) is preserved in magnitude.
        let area = canonical.measure2().abs();
        assert!(reoriented.measure2() >= -1e-9 * area.max(1.0));
        assert!((reoriented.measure2().abs() - area).abs() <= 1e-9 * area.max(1.0));

        // The reoriented cell is the same shape: the arc set is unchanged.
        assert_eq!(
            sorted_edge_keys(&canonical.to_quadratic_polygon2()),
            sorted_edge_keys(&reoriented.to_quadratic_polygon2()),
            "mid-side nodes must stay paired with their edges"
        );

        // Reorienting an already-positive cell is a no-op.
        let (et2, conn3) = reoriented.oriented_positive_connectivity();
        assert_eq!((et2, conn3), (et, conn2));
    }

    proptest! {
        #[test]
        fn reorient_preserves_quad8_shape(
            pts in arb_quadratic_coords(4),
            flip in proptest::bool::ANY,
        ) {
            reorient_case(&pts, ElementType::QUAD8, 4, flip);
        }
    }

    proptest! {
        #[test]
        fn reorient_preserves_tri6_shape(
            pts in arb_quadratic_coords(3),
            flip in proptest::bool::ANY,
        ) {
            reorient_case(&pts, ElementType::TRI6, 3, flip);
        }
    }
}
