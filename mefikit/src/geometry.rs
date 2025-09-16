pub mod is_in;
pub mod measures;
pub mod seg_intersect;

use self::measures as mes;
use crate::{ElementLike, ElementType, UMeshView};

use nalgebra as na;
use ndarray as nd;
use ndarray::prelude::*;
use rayon::prelude::*;
use rstar::AABB;
use std::collections::BTreeMap;

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
    fn coords2(&self) -> impl Iterator<Item = &[f64; 2]> {
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
    fn coords3(&self) -> impl Iterator<Item = &[f64; 3]> {
        (0..self.connectivity().len()).map(|i| self.coord3_ref(i))
    }
    fn coords(&self) -> impl Iterator<Item = &[f64]> {
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
            SEG2 => todo!(),
            TRI3 => mes::surf_tri3(
                self.coord3(0).into(),
                self.coord3(1).into(),
                self.coord3(2).into(),
            ),
            QUAD4 => mes::surf_quad3(
                &self.coord3(0).into(),
                &self.coord3(1).into(),
                &self.coord3(2).into(),
                &self.coord3(3).into(),
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

    fn to_aabb2(&self) -> AABB<[f64; 2]> {
        AABB::from_points(self.coords2())
    }

    fn to_aabb(&self) -> AABB<[f64; 3]> {
        AABB::from_points(self.coords3())
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

pub fn measure(mesh: UMeshView) -> BTreeMap<ElementType, Array1<f64>> {
    mesh
        .element_blocks
        .iter()
        .map(|(&k, v)| {
            (
                k,
                match mesh.space_dimension() {
                    0 => nd::Array1::from_vec(vec![0.0; v.len()]),
                    1 => todo!(),
                    2 => nd::Array1::from_vec(
                        v.par_iter(mesh.coords.view())
                        .map(|e| e.measure2())
                        .collect()
                    ),
                    3 => nd::Array1::from_vec(
                        v.par_iter(mesh.coords.view())
                        .map(|e| e.measure3())
                        .collect()
                    ),
                    c => panic!( "{c} is not a valid space dimension. Space (coordinates) dimension must be 0, 1, 2 ou 3.")
                }
            )
        })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::mesh_examples as me;
    use approx::*;

    #[test]
    fn test_umesh_measure() {
        let mesh = me::make_mesh_2d_quad();
        let measures = measure(mesh.view());
        assert_eq!(measures.len(), 1);
        assert!(measures.contains_key(&ElementType::QUAD4));
        let measure_values = measures.get(&ElementType::QUAD4).unwrap();
        assert_eq!(measure_values.len(), 1);
        assert_abs_diff_eq!(measure_values[0], 1.0); // Area of the quad is 1.0
    }
}
