mod mesh_element;
mod element_block;
mod element_block_like;

pub use crate::mesh_element::ElementType;
pub use crate::element_block::{ElementBlock, IntoElementBlockEntry, RegularCells, PolyCells};
pub use crate::element_block_like::ElementBlockLike;

pub mod umesh {

    use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, Axis};
    use std::collections::HashMap;
    use std::collections::HashSet;
    
    pub use crate::mesh_element::ElementType;
    pub use crate::element_block::{ElementBlock, IntoElementBlockEntry, RegularCells, PolyCells};
    pub use crate::element_block_like::ElementBlockLike;

    pub struct UMesh {
        coords: Array2<f64>,
        element_blocks: HashMap<ElementType, ElementBlock>,
    }

    pub struct UMeshView<'a> {
        pub coords: &'a Array2<f64>,
        pub elements: HashMap<ElementType, ElementBlockView<'a>>,
    }
    pub enum ElementBlockView<'a> {
        RegularCells(&'a RegularCells),
        PolyCells(&'a PolyCells),
    }

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
}
