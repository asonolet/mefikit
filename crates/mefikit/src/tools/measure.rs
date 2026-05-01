//! Geometric measurements for mesh elements.
//!
//! Computes element measures (length, area, volume) and stores them as fields.

use crate::element_traits::ElementGeo;
use crate::mesh::ElementType;
use crate::mesh::FieldOwned;
use crate::mesh::UMesh;
use crate::mesh::{Dimension, UMeshView};

use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Computes the geometric measure of each element in the mesh.
///
/// Returns a map of element types to arrays of measure values.
pub fn measure(mesh: UMeshView, dim: Option<Dimension>) -> BTreeMap<ElementType, nd::Array1<f64>> {
    let dim = dim.unwrap_or_else(|| mesh.topological_dimension().unwrap());
    mesh
        .par_blocks()
        .filter(|(et, _)| et.dimension() == dim)
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

/// Trait for computing and storing element measures as fields.
pub trait Measurable {
    /// Computes element measures and returns them as a field.
    fn measure(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1>;
    /// Computes element measures and stores them as a named field in the mesh.
    fn measure_update(&mut self, name: &str, dim: Option<Dimension>);
}

impl Measurable for UMesh {
    fn measure(&self, dim: Option<Dimension>) -> FieldOwned<ndarray::Ix1> {
        FieldOwned::new(measure(self.view(), dim))
    }
    fn measure_update(&mut self, name: &str, dim: Option<Dimension>) {
        let field = FieldOwned::new(measure(self.view(), dim));
        self.update_field(name, field.into_shared().into_dyn(), dim);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use crate::mesh_examples as me;
    use approx::*;

    #[test]
    fn test_umesh_measure() {
        let mesh = me::make_mesh_2d_quad();
        let measures = measure(mesh.view(), None);
        assert_eq!(measures.len(), 1);
        assert!(measures.contains_key(&ElementType::QUAD4));
        let measure_values = measures.get(&ElementType::QUAD4).unwrap();
        assert_eq!(measure_values.len(), 1);
        assert_abs_diff_eq!(measure_values[0], 1.0); // Area of the quad is 1.0
    }
}
