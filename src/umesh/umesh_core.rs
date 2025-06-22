use ndarray as nd;
use ndarray::prelude::*;
use std::collections::BTreeMap;
use todo;

use crate::umesh::element::{Dimension, Element, ElementId, Regularity};
use crate::umesh::element_block::{ElementBlock, ElementBlockBase, IntoElementBlockEntry};
use crate::umesh::selector::Selector;
use crate::umesh::ElementType;

/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements
/// kinds and fields associated.
pub struct UMeshBase<CooData, ConnData, FieldData, GroupData>
where
    CooData: nd::RawData<Elem = f64> + nd::Data + Sync,
    ConnData: nd::RawData<Elem = usize> + nd::Data + Sync,
    FieldData: nd::RawData<Elem = f64> + nd::Data + Sync,
    GroupData: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    coords: ArrayBase<CooData, Ix2>, // TODO: Use ArcArray2 for shared ownership
    element_blocks: BTreeMap<ElementType, ElementBlockBase<ConnData, FieldData, GroupData>>,
}

pub type UMesh =
    UMeshBase<nd::OwnedRepr<f64>, nd::OwnedRepr<usize>, nd::OwnedRepr<f64>, nd::OwnedRepr<usize>>;

pub type UMeshView<'a> = UMeshBase<
    nd::ViewRepr<&'a f64>,
    nd::ViewRepr<&'a usize>,
    nd::ViewRepr<&'a f64>,
    nd::ViewRepr<&'a usize>,
>;

impl<CooData, ConnData, FieldData, GroupData> UMeshBase<CooData, ConnData, FieldData, GroupData>
where
    CooData: nd::RawData<Elem = f64> + nd::Data + Sync,
    ConnData: nd::RawData<Elem = usize> + nd::Data + Sync,
    FieldData: nd::RawData<Elem = f64> + nd::Data + Sync,
    GroupData: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn coords(&self) -> &ArrayBase<CooData, Ix2> {
        &self.coords
    }

    pub fn elements(&self) -> impl Iterator<Item = Element> {
        self.element_blocks
            .values()
            .flat_map(|block| block.iter(self.coords.view()))
    }

    pub fn elements_of_dim(&self, dim: Dimension) -> impl Iterator<Item = Element> {
        self.element_blocks
            .iter()
            .filter(move |(&k, _)| k.dimension() == dim)
            .flat_map(|(_, block)| block.iter(self.coords.view()))
    }

    pub fn element_blocks(
        &self,
    ) -> &BTreeMap<ElementType, ElementBlockBase<ConnData, FieldData, GroupData>> {
        &self.element_blocks
    }

    pub fn element_block(
        &self,
        element_type: ElementType,
    ) -> Option<&ElementBlockBase<ConnData, FieldData, GroupData>> {
        self.element_blocks.get(&element_type)
    }

    pub fn select_ids(&self) -> Selector<CooData, ConnData, FieldData, GroupData> {
        Selector::new(&self)
    }

    pub fn extract_mesh(&self, ids: &[ElementId]) -> UMesh {
        todo!();
    }
}

impl UMesh {
    pub fn new(coords: nd::Array2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_block(&mut self, block: ElementBlock) {
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
                self.element_blocks.entry(element_type).or_insert_with(|| {
                    ElementBlock::new_regular(
                        element_type,
                        Array2::zeros((0, element_type.num_nodes().unwrap())),
                    )
                });
            }
            Regularity::Poly => {
                self.element_blocks
                    .entry(element_type)
                    .or_insert_with(|| ElementBlock::new_poly(element_type, arr1(&[]), arr1(&[])));
            }
        }

        self.element_blocks
            .get_mut(&element_type)
            .unwrap() // This unwrap is safe because we just inserted the element type
            .add_element(ArrayView1::from(connectivity), family, fields);
    }

    // pub fn families(&self, element_type: ElementType) -> Option<&[usize]> {
    //     let eb = self.element_block(element_type);
    //     match eb {
    //         Some(eb) => Some(&eb.families),
    //         None => None,
    //     }
    // }

    pub fn replace(&mut self, ids: &[ElementId]) {
        todo!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::umesh::ElementType;
    use ndarray as nd;

    fn make_test_2d_mesh() -> UMesh {
        let coords =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_block(ElementBlock::new_regular(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]),
        ));
        mesh
    }

    #[test]
    fn test_umesh_creation() {
        let coords = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
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
