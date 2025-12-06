use crate::geometry::ElementGeo;
use crate::mesh::ElementType;
use crate::mesh::UMeshView;

use ndarray as nd;
use ndarray::prelude::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

pub fn measure(mesh: UMeshView) -> BTreeMap<ElementType, Array1<f64>> {
    mesh
        .par_blocks()
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
    use crate::mesh::ElementType;
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
