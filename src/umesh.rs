use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, Axis};
use std::collections::HashMap;
use std::collections::HashSet;

use crate::mesh_element::ElementType;
use crate::element_block::{ElementBlock, IntoElementBlockEntry, RegularCells, PolyCells};
use crate::element_block_like::ElementBlockLike;


pub struct UMesh {
    coords: Array2<f64>,
    element_blocks: HashMap<ElementType, ElementBlock>,
}

pub struct UMeshView<'a> {
    pub coords: &'a Array2<f64>,
    pub elements: HashMap<ElementType, ElementBlockView<'a>>,
}

// TODO: write a real view (down to the ndarray view)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_block::{ElementBlock, RegularCells, RegularCellType};
    use ndarray::{array, Array2, Array1};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn dummy_regular_cells() -> RegularCells {
        let connectivity = array![[0, 1, 2]];
        let families = Array1::from(vec![0]);

        RegularCells {
            cell_type: RegularCellType::TRI3,
            connectivity,
            params: HashMap::new(),
            fields: HashMap::new(),
            families,
            groups: HashMap::new(),
        }
    }

    #[test]
    fn test_umesh_insert_and_retrieve() {
        let coords = Arc::new(array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        let mut mesh = UMesh {
            coords: coords.clone(),
            element_blocks: HashMap::new(),
        };

        let rc = dummy_regular_cells();
        mesh.element_blocks.insert(ElementType::TRI3, ElementBlock::RegularCells(rc));

        assert!(mesh.element_blocks.contains_key(&ElementType::TRI3));
    }
}
