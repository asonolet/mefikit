//! Geometric operations for mesh elements.
//!
//! Provides the [`ElementGeo`] trait for coordinate access, measures,
//! bounding boxes, and centroid calculations. All computations delegate to the owned geometry
//! types of the [`crate::geometry`] module.

use super::element_topo::ElementTopo;
use crate::geometry::{
    Polygon, Polyhedron, Segment, area_polygon3, area_quad2, area_tri2, bounds_iter,
    convex_polygon_contains2, hex_contains, hex_volume, signed_area2, tet_contains, tet_volume,
    vertex_centroid,
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
    fn to_polygon2(&self) -> Polygon<2> {
        Polygon::with_convexity(
            self.coords2().copied(),
            self.element_type().known_convexity(),
        )
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
    /// Returns length for 1D elements.
    fn measure1(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => Segment::new(self.coord1(0).into(), self.coord1(1).into()).length(),
            other => unimplemented!("measure1 is not implemented for element type {other:?}"),
        }
    }

    /// Computes the geometric measure of the element in 2D space.
    ///
    /// Returns length for 1D elements and area for 2D elements.
    fn measure2(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => Segment::new(*self.coord2_ref(0), *self.coord2_ref(1)).length(),
            TRI3 => area_tri2(self.coord2_ref(0), self.coord2_ref(1), self.coord2_ref(2)),
            QUAD4 => area_quad2(&[
                *self.coord2_ref(0),
                *self.coord2_ref(1),
                *self.coord2_ref(2),
                *self.coord2_ref(3),
            ]),
            PGON => self.to_polygon2().area(),
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

    /// Computes the 2D vertex centroid of the element: the arithmetic mean of its node
    /// coordinates.
    fn centroid2(&self) -> [f64; 2] {
        use ElementType::*;
        match self.element_type() {
            VERTEX => *self.coord2_ref(0),
            SEG2 => Segment::new(*self.coord2_ref(0), *self.coord2_ref(1)).midpoint(),
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
            other => unimplemented!("centroid2 is not implemented for element type {other:?}"),
        }
    }

    /// Computes the 3D vertex centroid of the element: the arithmetic mean of its node
    /// coordinates.
    fn centroid3(&self) -> [f64; 3] {
        use ElementType::*;
        match self.element_type() {
            VERTEX => *self.coord3_ref(0),
            SEG2 => Segment::new(*self.coord3_ref(0), *self.coord3_ref(1)).midpoint(),
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

    #[test]
    fn test_coord2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
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
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let c0: &[f64; 2] = elem.coord2_ref(0);
        assert_eq!(c0, &[0.0, 0.0]);
    }

    #[test]
    fn test_coords2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let coords: Vec<_> = elem.coords2().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coord3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let p0 = elem.coord3(0);
        assert_eq!(p0, na::Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_coords3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let coords: Vec<_> = elem.coords3().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coords() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let coords: Vec<_> = elem.coords().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_measure2_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = &[0, 1, 3, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure2_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SEG2,
            &groups,
        );
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centroid2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
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
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TET4,
            &groups,
        );
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
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TET4,
            &groups,
        );
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
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::PHED,
            &groups,
        );
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
            None,
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
            None,
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
            let elem = Element::new(
                0,
                coords.view(),
                None,
                &family,
                conn,
                ElementType::TET4,
                &groups,
            );
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
            None,
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
            None,
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
}
