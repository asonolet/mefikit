pub mod is_in;
mod measure;

use self::measure as mes;
use crate::{ElementLike, ElementType, UMeshView};
use ndarray as nd;
use ndarray::prelude::*;
use std::collections::BTreeMap;

pub trait ElementGeo<'a>: ElementLike<'a> {
    fn measure2(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => mes::dist(self.coords().row(0), self.coords().row(1)),
            TRI3 => mes::surf_tri2(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
            ),
            QUAD4 => mes::surf_quad2(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(3).as_slice().unwrap().try_into().unwrap(),
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
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
            ),
            QUAD4 => mes::surf_quad3(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(3).as_slice().unwrap().try_into().unwrap(),
            ),
            _ => todo!(),
        }
    }

    fn is_point_inside(&self, point: &[f64]) -> bool {
        // Returns true if the point is inside the element
        // For 0D elements, return true if the point is equal to the element's coordinates
        // For 1D elements, return true if the point is between the two nodes
        // For 2D elements, return true if the point is inside the polygon
        // For 3D elements, return true if the point is inside the polyhedron
        todo!()
    }
}

impl<'a, T> ElementGeo<'a> for T where T: ElementLike<'a> {}

pub fn measure(mesh: UMeshView) -> BTreeMap<ElementType, Array1<f64>> {
    match mesh.space_dimension() {
        0 => mesh
            .element_blocks
            .iter()
            .map(|(&k, v)| (k, nd::arr1(&vec![0.0; v.len()])))
            .collect(),
        1 => todo!(),
        2 => mesh
            .element_blocks
            .iter()
            .map(|(&k, v)| {
                (
                    k,
                    nd::arr1(
                        &v.iter(mesh.coords.view())
                            .map(|e| e.measure2())
                            .collect::<Vec<f64>>(),
                    ),
                )
            })
            .collect(),
        3 => mesh
            .element_blocks
            .iter()
            .map(|(&k, v)| {
                (
                    k,
                    nd::arr1(
                        &v.iter(mesh.coords.view())
                            .map(|e| e.measure3())
                            .collect::<Vec<f64>>(),
                    ),
                )
            })
            .collect(),
        c => panic!(
            "{c} is not a valid space dimension. Space (coordinates) dimension must be 0, 1, 2 ou 3."
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ElementType, UMesh};
    use approx::*;
    use ndarray as nd;

    fn make_test_2d_mesh() -> UMesh {
        let coords =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::QUAD4, nd::arr2(&[[0, 1, 3, 2]]));
        mesh
    }

    #[test]
    fn test_umesh_measure() {
        let mesh = make_test_2d_mesh();
        let measures = measure(mesh.view());
        assert_eq!(measures.len(), 1);
        assert!(measures.contains_key(&ElementType::QUAD4));
        let measure_values = measures.get(&ElementType::QUAD4).unwrap();
        assert_eq!(measure_values.len(), 1);
        assert_abs_diff_eq!(measure_values[0], 1.0); // Area of the quad is 1.0
    }
}
