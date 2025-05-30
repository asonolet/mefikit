use ndarray::prelude::*;
use ndarray::ArcArray2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::umesh::element::Element;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId {
    pub element_type: ElementType,
    pub local_index: usize,
}

impl UMesh {
    pub fn new(coords: ArcArray2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_block<T: IntoElementBlockEntry>(&mut self, compo: T) {
        let (key, wrapped) = compo.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
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
