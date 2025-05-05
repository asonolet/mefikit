use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis};
use std::collections::HashMap;
use std::collections::HashSet;
use rayon::prelude::*;

use crate::mesh_element::{ElementType, ElementView};

pub trait ElementBlockLike: Sync {
    fn len(&self) -> usize;
    fn params(&self) -> &HashMap<String, f64>;
    fn fields(&self) -> &HashMap<String, ArrayD<f64>>;
    fn families(&self) -> &Array1<usize>;
    fn groups(&self) -> &HashMap<String, HashSet<usize>>;
    fn compo_type(&self) -> ElementType;
    fn element_connectivity<'a>(&'a self, index: usize) -> ArrayView1<'a, usize>;

    fn iter<'a>(&'a self, coords: &'a Array2<f64>) -> Box<dyn Iterator<Item = Element<'a>> + 'a> {
        Box::new((0..self.len()).map(move |i| {
            let connectivity = self.element_connectivity(i);
            let family = &self.families()[i];
            let fields = self.fields().iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();
            Element {
                global_index: i,
                coords,
                fields,
                family,
                groups: self.groups(),
                connectivity,
                compo_type: self.compo_type(),
            }
        }))
    }
    fn par_iter<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> impl ParallelIterator<Item = Element<'a>> + 'a {
        (0..self.len()).into_par_iter().map(move |i| {
            let connectivity = self.element_connectivity(i);
            let fields = self.fields()
                .iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();

            Element {
                global_index: i,
                coords,
                fields,
                family: &self.families()[i],
                groups: self.groups(),
                connectivity,
                compo_type: self.compo_type(),
            }
        })
    }

    // /// Get neighbors of an element by kind
//     pub fn neighbors(
//         &self,
//         element_id: usize,
//         kind: NeighborKind,
//     ) -> Vec<usize>;
// 
//     /// Compute lower-dimensional block (e.g., faces of a 3D block)
//     pub fn compute_descending_connectivity(&self) -> (ElementBlock, HashMap<usize, HashSet<usize>>, HashMap<usize, HashSet<usize>>);
// 
//     /// Append new elements
//     pub fn append(
//         &self,
//         new_connectivity: &[usize],
//         offsets: Option<&[usize]>,
//         fields: Option<HashMap<String, ArrayD<f64>>>,
//         family: Option<usize>,
//         groups: Option<HashMap<String, HashSet<usize>>>,
//     ) -> ElementBlock;
// 
//     /// Insert elements at index
//     pub fn insert(
//         &self,
//         index: usize,
//         new_connectivity: &[usize],
//         offsets: Option<&[usize]>,
//         fields: Option<HashMap<String, ArrayD<f64>>>,
//         family: Option<usize>,
//         groups: Option<HashMap<String, HashSet<usize>>>,
//     ) -> ElementBlock;
// 
//     /// Remove elements by indices
//     pub fn remove(&self, indices: &[usize]) -> ElementBlock;
// 
//     /// Replace node ids in-place or return new block
//     pub fn replace_node_ids(
//         &self,
//         mapping: &HashMap<usize, usize>,
//     ) -> ElementBlock;
// 
//     /// Remove duplicates (optionally considering orientation)
//     pub fn remove_duplicates(&self, consider_orientation: bool) -> ElementBlock;
}

pub trait ElementBlockLikeMut: ElementBlockLike + Send {
    fn element_connectivity_mut(&mut self, index: usize) -> ArrayViewMut1<usize>;

    fn iter_mut<'a>(
        &'a mut self,
        coords: &'a Array2<f64>,
    ) -> impl Iterator<Item = ElementMut<'a>> + 'a {
        (0..self.len()).map(move |i| {
            let connectivity = self.element_connectivity_mut(i);
            let family = &mut self.families_mut()[i]; // Access mutable family

            let fields = self
                .fields_mut()
                .iter_mut()
                .map(|(k, v)| (k.as_str(), v.index_axis_mut(Axis(0), i)))
                .collect();

            ElementMut {
                global_index: i,
                coords,
                connectivity,
                family,
                fields,
                groups: self.groups(),
                element_type: self.compo_type(),
            }
        })
    }

    fn par_iter_mut<'a>(
        &'a mut self,
        coords: &'a Array2<f64>,
    ) -> impl ParallelIterator<Item = ElementMut<'a>> + 'a {
        let len = self.len();
        let families_ptr = self.families_mut() as *mut Array1<usize>;
        let fields_ptr = self.fields_mut() as *mut HashMap<String, ArrayD<f64>>;

        (0..len).into_par_iter().map(move |i| {
            let connectivity = unsafe { self.element_connectivity_mut(i) };
            let family = unsafe { &mut (*families_ptr)[i] };
            let fields = unsafe {
                (*fields_ptr)
                    .iter_mut()
                    .map(|(k, v)| (k.as_str(), v.index_axis_mut(Axis(0), i)))
                    .collect()
            };

            ElementMut {
                global_index: i,
                coords,
                connectivity,
                family,
                fields,
                groups: self.groups(),
                element_type: self.compo_type(),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_block::RegularCells;
    use ndarray::{array, Array1, Array2};
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;

    fn dummy_regular_cells() -> RegularCells {
        let connectivity = array![[0, 1, 2], [2, 3, 0]];
        let families = Array1::from(vec![0, 1]);
        let mut groups = HashMap::new();
        groups.insert("groupA".into(), HashSet::from([0]));
        groups.insert("groupB".into(), HashSet::from([1]));

        RegularCells {
            cell_type: RegularCellType::TRI3,
            connectivity,
            params: HashMap::new(),
            fields: HashMap::new(),
            families,
            groups,
        }
    }

    #[test]
    fn test_element_block_like_iter() {
        let rc = dummy_regular_cells();
        let coords = Arc::new(array![
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0]
        ]);

        let indices: Vec<_> = rc.iter(&coords).map(|el| el.global_index).collect();
        assert_eq!(indices, vec![0, 1]);
    }
}
