//! Geometric operations for mesh elements.
//!
//! Provides the [`ElementGeo`] trait for coordinate access, measures,
//! bounding boxes, and centroid calculations.

use super::measures as mes;
use crate::mesh::{ElementLike, ElementType};

use nalgebra as na;
use rstar::AABB;

/// Geometric operations for mesh elements.
///
/// Extends [`ElementLike`] with methods for accessing coordinates as nalgebra
/// points, computing measures (length/area/volume), bounding boxes, and centroids.
pub trait ElementGeo<'a>: ElementLike<'a> {
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
    fn coords3(&self) -> impl ExactSizeIterator<Item = &[f64; 3]> {
        (0..self.connectivity().len()).map(|i| self.coord3_ref(i))
    }

    /// Returns an iterator over all coordinates as slices.
    fn coords(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        (0..self.connectivity().len()).map(|i| self.coord(i))
    }

    /// Computes the geometric measure of the element in 1D space.
    ///
    /// Returns length for 1D elements.
    fn measure1(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => mes::dist1(self.coord1(0), self.coord1(1)),
            _ => todo!(),
        }
    }

    /// Computes the geometric measure of the element in 2D space.
    ///
    /// Returns length for 1D elements and area for 2D elements.
    fn measure2(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => mes::dist2(self.coord2(0), self.coord2(1)),
            TRI3 => mes::surf_tri2(self.coord2(0), self.coord2(1), self.coord2(2)),
            QUAD4 => mes::surf_quad2(
                &self.coord2(0),
                &self.coord2(1),
                &self.coord2(2),
                &self.coord2(3),
            ),
            _ => todo!(),
        }
    }

    /// Computes the geometric measure of the element in 3D space.
    ///
    /// Returns length for 1D elements, area for 2D elements, and volume for 3D elements.
    fn measure3(&self) -> f64 {
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => mes::dist3(self.coord3_ref(0), self.coord3_ref(1)),
            TRI3 => mes::surf_tri3(
                self.coord3(0).into(),
                self.coord3(1).into(),
                self.coord3(2).into(),
            ),
            QUAD4 => mes::surf_quad3(
                self.coord3_ref(0),
                self.coord3_ref(1),
                self.coord3_ref(2),
                self.coord3_ref(3),
            ),
            _ => todo!(),
        }
    }

    /// Returns `true` if the given point lies inside the element.
    ///
    /// # Note
    /// This method is not yet implemented.
    fn is_point_inside(&self, _point: &[f64]) -> bool {
        todo!()
    }

    /// Computes the 2D axis-aligned bounding box of the element.
    fn to_aabb2(&self) -> AABB<[f64; 2]> {
        AABB::from_points(self.coords2())
    }

    /// Computes the 3D axis-aligned bounding box of the element.
    fn to_aabb(&self) -> AABB<[f64; 3]> {
        AABB::from_points(self.coords3())
    }

    /// Computes the 2D centroid of the element.
    fn centroid2(&self) -> [f64; 2] {
        let mut p: na::Point2<f64> = na::Point2::origin();
        for i in 0..self.connectivity().len() {
            p += self.coord2(i) - na::Point2::origin();
        }
        (p / (self.connectivity().len() as f64)).into()
    }

    /// Computes the 3D centroid of the element.
    fn centroid3(&self) -> [f64; 3] {
        let mut p: na::Point3<f64> = na::Point3::origin();
        for i in 0..self.connectivity().len() {
            p += self.coord3(i) - na::Point3::origin();
        }
        (p / (self.connectivity().len() as f64)).into()
    }
}

impl<'a, T> ElementGeo<'a> for T where T: ElementLike<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Element, ElementType};
    use approx::assert_abs_diff_eq;
    use ndarray as nd;
    use std::collections::BTreeMap;

    #[test]
    fn test_coord2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
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
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let c0: &[f64; 2] = elem.coord2_ref(0);
        assert_eq!(c0, &[0.0, 0.0]);
    }

    #[test]
    fn test_coords2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let coords: Vec<_> = elem.coords2().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coord3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let p0 = elem.coord3(0);
        assert_eq!(p0, na::Point3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_coords3() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let coords: Vec<_> = elem.coords3().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_coords() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let coords: Vec<_> = elem.coords().collect();
        assert_eq!(coords.len(), 3);
    }

    #[test]
    fn test_measure2_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = &[0, 1, 3, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::QUAD4,
        );
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_measure2_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::SEG2,
        );
        assert_abs_diff_eq!(elem.measure2(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_centroid2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
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
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TET4,
        );
        let centroid = elem.centroid3();
        assert_abs_diff_eq!(centroid[0], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(centroid[1], 0.25, epsilon = 1e-10);
        assert_abs_diff_eq!(centroid[2], 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_to_aabb2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let aabb = elem.to_aabb2();
        assert_eq!(aabb.lower(), [0.0, 0.0]);
        assert_eq!(aabb.upper(), [1.0, 1.0]);
    }

    #[test]
    fn test_to_aabb() {
        let coords = nd::array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let aabb = elem.to_aabb();
        assert_eq!(aabb.lower(), [0.0, 0.0, 0.0]);
        assert_eq!(aabb.upper(), [1.0, 1.0, 0.0]);
    }
}
