use ndarray::prelude::*;
use ndarray::ArcArray2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::umesh::element::{Element, Regularity};
use crate::umesh::element_block::{ElementBlock, IntoElementBlockEntry};
use crate::umesh::ElementType;

#[derive(Debug, Clone, Serialize, Deserialize)]
/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements kinds and fields associated.
pub struct UMesh {
    coords: ArcArray2<f64>,
    element_blocks: BTreeMap<ElementType, ElementBlock>,
}

// pub struct UMeshView<'a> {
//     pub coords: &'a Array2<f64>,
//     pub elements: HashMap<ElementType, ElementBlockView<'a>>,
// }

impl UMesh {
    pub fn new(coords: ArcArray2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_block<T: IntoElementBlockEntry>(&mut self, block: T) {
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }

    pub fn add_element(
        &mut self,
        element_type: ElementType,
        connectivity: &[usize],
        family: Option<usize>,
        fields: Option<BTreeMap<String, ArrayViewD<f64>>>,
    ) {
        match element_type.regularity() {
            Regularity::Regular => {
                if connectivity.len() != element_type.num_nodes().unwrap() {
                    panic!("Connectivity length does not match the number of nodes for element type {:?}", element_type);
                }
                self.element_blocks
                    .entry(element_type)
                    .or_insert_with(|| ElementBlock::new_regular(element_type, arr2(&[[]])));
            }
            Regularity::Poly => {
                self.element_blocks
                    .entry(element_type)
                    .or_insert_with(|| ElementBlock::new_poly(element_type, arr1(&[]), arr1(&[])));
            }
        }

        self.element_blocks
            .get_mut(&element_type)
            .unwrap()  // This unwrap is safe because we just inserted the element type
            .add_element(ArrayView1::from(connectivity), family, fields);
    }

    pub fn coords(&self) -> &ArcArray2<f64> {
        &self.coords
    }

    pub fn elements(&self) -> impl Iterator<Item = Element> {
        self.element_blocks
            .values()
            .flat_map(|block| block.iter(self.coords.view()))
    }

    pub fn element_blocks(&self) -> &BTreeMap<ElementType, ElementBlock> {
        &self.element_blocks
    }

    pub fn element_block(&self, element_type: ElementType) -> Option<&ElementBlock> {
        self.element_blocks.get(&element_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::umesh::ElementType;
    use ndarray as nd;

    fn make_test_2d_mesh() -> UMesh {
        let coords =
            ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_block(ElementBlock::new_regular(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]),
        ));
        mesh
    }

    #[test]
    fn test_umesh_creation() {
        let coords = ArcArray2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_block(ElementBlock::new_regular(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]),
        ));
        assert_eq!(mesh.coords().shape(), &[3, 1]);
        assert_eq!(mesh.element_blocks().len(), 1);
        assert!(mesh.element_blocks().contains_key(&ElementType::SEG2));
    }
    #[test]
    fn test_umesh_element_iteration() {
        let mesh = make_test_2d_mesh();

        let elements: Vec<Element> = mesh.elements().collect();
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].element_type, ElementType::QUAD4);
        assert_eq!(elements[0].connectivity, nd::arr1(&[0, 1, 3, 2]));
    }
}
