use ndarray::Array2;
use std::collections::HashMap;

use crate::element::ElementType;
use crate::element_block::{ElementBlock, IntoElementBlockEntry};


pub struct UMesh {
    pub coords: Array2<f64>,
    pub element_blocks: HashMap<ElementType, ElementBlock>,
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
    pub fn add_compo<T: IntoElementBlockEntry>(&mut self, compo: T) {
        let (key, wrapped) = compo.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }
}

#[cfg(test)]
mod tests {
}
