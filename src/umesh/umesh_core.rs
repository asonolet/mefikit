use ndarray::ArcArray2;
use std::collections::BTreeMap;

use crate::umesh::element::ElementType;
use crate::umesh::element_block::{ElementBlock, IntoElementBlockEntry};

/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements kinds and fields associated.
pub struct UMesh {
    pub coords: ArcArray2<f64>,
    pub element_blocks: BTreeMap<ElementType, ElementBlock>,
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
