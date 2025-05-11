use ndarray::{s, Array1, Array2, ArrayD, ArrayView1, ArrayViewMut1, ArrayViewMutD, Axis};
use rayon::prelude::*;
use std::collections::HashMap;
use std::collections::HashSet;

use crate::element::{Element, ElementMut, ElementType};
use crate::connectivity::Connectivity;

/// The part of a mesh constituted by one kind of element.
///
/// The element block is the base structure to hold connectivity, fields, groups and params.
/// It is used to hold all cell information and allows cell iteration.
/// The only data not included for an element block to be standalone is the coordinates array.
pub struct ElementBlock {
    pub cell_type: ElementType,
    pub connectivity: Connectivity,
    pub params: HashMap<String, f64>,
    pub fields: HashMap<String, ArrayD<f64>>,
    pub families: Array1<usize>,
    pub groups: HashMap<String, HashSet<usize>>,
}

impl<'a> ElementBlock {
    fn len(&self) -> usize {
        self.connectivity.len()
    }
    fn element_connectivity(&'a self, index: usize) -> ArrayView1<'a, usize> {
        self.connectivity.get(index)
    }
    fn iter(&'a self, coords: &'a Array2<f64>) -> impl Iterator<Item = Element<'a>> + 'a {
        (0..self.len()).map(move |i| {
            let connectivity = self.element_connectivity(i);
            let fields = self
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();
            Element::new(
                i,
                coords,
                fields,
                &self.families[i],
                &self.groups,
                connectivity,
                self.cell_type,
            )
        })
    }
    fn par_iter(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> impl ParallelIterator<Item = Element<'a>> + 'a {
        (0..self.len()).into_par_iter().map(move |i| {
            let connectivity = self.element_connectivity(i);
            let fields = self
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();

            Element::new(
                i,
                coords,
                fields,
                &self.families[i],
                &self.groups,
                connectivity,
                self.cell_type,
            )
        })
    }

    fn element_connectivity_mut(&mut self, index: usize) -> ArrayViewMut1<usize> {
        self.connectivity.get_mut(index)
    }

    // pub fn iter_mut(
    //     &'a mut self,
    //     coords: &'a Array2<f64>,
    // ) -> impl Iterator<Item = ElementMut<'a>> + 'a {
    //     let ElementBlock {
    //         cell_type,
    //         connectivity,
    //         params: _,
    //         fields,
    //         families,
    //         groups,
    //     } = self;

    //     let num_elems = connectivity.len();

    //     // Step 1: Prepare a Vec of empty HashMaps
    //     let mut per_element_field_views: Vec<HashMap<&'a str, ArrayViewMutD<'a, f64>>> =
    //         Vec::with_capacity(num_elems);
    //     for _ in 0..num_elems {
    //         per_element_field_views.push(HashMap::new());
    //     }

    //     // Step 2: Fill each element's field map with views
    //     for (key, array) in fields.iter_mut() {
    //         let key_str: &'a str = key.as_str();
    //         let views = array.axis_iter_mut(Axis(0));
    //         let mut i = 0;
    //         for view in views {
    //             per_element_field_views[i].insert(key_str, view);
    //             i += 1;
    //         }
    //     }

    //     families
    //         .iter_mut()
    //         .zip(per_element_field_views.into_iter())
    //         .zip(connectivity.iter_mut())
    //         .enumerate()
    //         .map(|(i, ((fam, per_elem_fields), conn))| ElementMut {
    //             global_index: i,
    //             coords,
    //             connectivity: conn,
    //             family: fam,
    //             fields: per_elem_fields,
    //             groups: &*groups,
    //             element_type: *cell_type,
    //         })

    //     // Step 3: Preconstruct all ElementMut values and return iterator over them
    //     for i in 0..num_elems {
    //         elements.push(ElementMut {
    //             global_index: i,
    //             coords,
    //             connectivity: connectivity.element_connectivity_mut(i),
    //             family: std::mem::take(&mut families_mut_view[i]),
    //             fields: per_element_field_views.remove(0), // always remove first
    //             groups,
    //             element_type: *cell_type,
    //         });
    //     }
    // }

    // pub fn par_iter_mut<'a>(
    //     &'a mut self,
    //     coords: &'a Array2<f64>,
    // ) -> impl ParallelIterator<Item = ElementMut<'a>> {
    //     let num_elems = self.connectivity.len();

    //     // SAFETY: We split interior mutability manually into chunks
    //     // and make sure each index `i` is accessed only once.
    //     (0..num_elems).into_par_iter().map(move |i| {
    //         let connectivity = self.connectivity.element_connectivity_mut(i);

    //         let family = &mut self.families[i];

    //         let fields: HashMap<&'a str, ArrayViewMutD<'a, f64>> = self
    //             .fields
    //             .iter_mut()
    //             .map(|(k, v)| (k.as_str(), v.index_axis_mut(ndarray::Axis(0), i)))
    //             .collect();

    //         ElementMut {
    //             global_index: i,
    //             coords,
    //             connectivity,
    //             family,
    //             fields,
    //             groups: &self.groups,
    //             element_type: self.cell_type,
    //         }
    //     })
    // }
}

pub trait IntoElementBlockEntry {
    fn into_entry(self) -> (ElementType, ElementBlock);
}

impl IntoElementBlockEntry for ElementBlock {
    fn into_entry(self) -> (ElementType, ElementBlock) {
        (self.cell_type.into(), self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use std::collections::HashMap;

    use crate::element::Element;
    use crate::element::ElementType;

    #[test]
    fn test_element_block() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]]);
        let params = HashMap::new();
        let fields = HashMap::new();
        let families = Array1::from_vec(vec![0, 1, 2]);
        let groups = HashMap::new();

        let element_block = ElementBlock {
            cell_type: ElementType::TRI3,
            connectivity,
            params,
            fields,
            families,
            groups,
        };

        assert_eq!(element_block.len(), 3);
        assert_eq!(element_block.params.len(), 0);
        assert_eq!(element_block.fields.len(), 0);
        assert_eq!(element_block.families.len(), 3);
        assert_eq!(element_block.groups.len(), 0);
    }

    #[test]
    fn test_element_block_iter() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]]);
        let params = HashMap::new();
        let fields = HashMap::new();
        let families = Array1::from_vec(vec![0, 1, 2]);
        let groups = HashMap::new();

        let element_block = ElementBlock {
            cell_type: ElementType::TRI3,
            connectivity,
            params,
            fields,
            families,
            groups,
        };

        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elements: Vec<Element> = element_block.iter(&coords).collect();

        assert_eq!(elements.len(), 3);
    }
}
