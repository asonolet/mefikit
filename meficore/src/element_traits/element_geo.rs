use super::measures as mes;
use crate::mesh::{ElementLike, ElementType};

use nalgebra as na;

pub trait ElementGeo<'a>: ElementLike<'a> {
    #[inline(always)]
    fn coord2(&self, i: usize) -> na::Point2<f64> {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 2);
        na::Point2::from_slice(coord)
    }
    #[inline(always)]
    fn coord2_ref(&self, i: usize) -> &[f64; 2] {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 2);
        coord.try_into().unwrap()
    }
    fn coords2(&self) -> impl ExactSizeIterator<Item = &[f64; 2]> {
        (0..self.connectivity().len()).map(|i| self.coord2_ref(i))
    }
    #[inline(always)]
    fn coord3(&self, i: usize) -> na::Point3<f64> {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 3);
        na::Point3::from_slice(coord)
    }
    #[inline(always)]
    fn coord3_ref(&self, i: usize) -> &[f64; 3] {
        let coord = self.coord(i);
        assert_eq!(coord.len(), 3);
        coord.try_into().unwrap()
    }
    fn coords3(&self) -> impl ExactSizeIterator<Item = &[f64; 3]> {
        (0..self.connectivity().len()).map(|i| self.coord3_ref(i))
    }
    fn coords(&self) -> impl ExactSizeIterator<Item = &[f64]> {
        (0..self.connectivity().len()).map(|i| self.coord(i))
    }

    fn measure2(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
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

    fn measure3(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
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

    fn is_point_inside(&self, _point: &[f64]) -> bool {
        // Returns true if the point is inside the element
        // For 0D elements, return true if the point is equal to the element's coordinates
        // For 1D elements, return true if the point is between the two nodes
        // For 2D elements, return true if the point is inside the polygon
        // For 3D elements, return true if the point is inside the polyhedron
        todo!()
    }

    fn bounds2(&self) -> [[f64; 2]; 2] {
        self.coords2()
            .fold([[f64::INFINITY; 2], [-f64::INFINITY; 2]], |a, c| {
                [
                    [f64::min(a[0][0], c[0]), f64::min(a[0][1], c[1])],
                    [f64::max(a[1][0], c[0]), f64::max(a[1][1], c[1])],
                ]
            })
    }

    fn bounds3(&self) -> [[f64; 3]; 2] {
        self.coords3()
            .fold([[f64::INFINITY; 3], [-f64::INFINITY; 3]], |a, c| {
                [
                    [
                        f64::min(a[0][0], c[0]),
                        f64::min(a[0][1], c[1]),
                        f64::min(a[0][2], c[2]),
                    ],
                    [
                        f64::max(a[1][0], c[0]),
                        f64::max(a[1][1], c[1]),
                        f64::max(a[1][2], c[2]),
                    ],
                ]
            })
    }

    fn centroid2(&self) -> [f64; 2] {
        let mut p: na::Point2<f64> = na::Point2::origin();
        for i in 0..self.connectivity().len() {
            p += self.coord2(i) - na::Point2::origin();
        }
        (p / (self.connectivity().len() as f64)).into()
    }

    fn centroid3(&self) -> [f64; 3] {
        let mut p: na::Point3<f64> = na::Point3::origin();
        for i in 0..self.connectivity().len() {
            p += self.coord3(i) - na::Point3::origin();
        }
        (p / (self.connectivity().len() as f64)).into()
    }
}

impl<'a, T> ElementGeo<'a> for T where T: ElementLike<'a> {}
