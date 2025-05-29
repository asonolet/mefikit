use ndarray::ArcArray2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::umesh::element::Element;
use crate::umesh::ElementType;
use crate::umesh::element_block::{ElementBlock, IntoElementBlockEntry};

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

