use ndarray::{Array1, Array2, ArrayD, ArrayView1, Axis, s};
use std::collections::HashMap;
use std::collections::HashSet;
use rayon::prelude::*;

use crate::element::{ElementType, Element};


pub enum Connectivity {
    Regular(Array2<usize>),
    Poly {
        data: Array1<usize>,
        offsets: Array1<usize>,
    },
}

impl Connectivity {
    pub fn len(&self) -> usize {
        match self {
            Connectivity::Regular(conn) => conn.nrows(),
            Connectivity::Poly {offsets, ..} => offsets.len() - 1,
        }
    }

    pub fn element_connectivity(&self, index: usize) -> ArrayView1<'_, usize> {
        match self {
            Connectivity::Regular(conn) => conn.row(index),
            Connectivity::Poly { data, offsets } => {
                let start = offsets[index];
                let end = offsets[index + 1];
                data.slice(s![start..end])
            }
        }
    }

    // pub fn element_connectivity_mut<'a>(& mut self, index: usize) -> ArrayViewMut1<'a, usize> {
    //     match self {
    //         Connectivity::Regular(conn) => conn.row_mut(index),
    //         Connectivity::Poly { data, offsets } => {
    //             let start = offsets[index];
    //             let end = offsets[index + 1];
    //             // data.slice_mut<'a>(s![start..end])
    //             data.slice_mut<s![start..end]>()
    //         }
    //     }
    // }
}

pub struct ElementBlock {
    pub cell_type: ElementType,
    pub connectivity: Connectivity,
    pub params: HashMap<String, f64>,
    pub fields: HashMap<String, ArrayD<f64>>,
    pub families: Array1<usize>,
    pub groups: HashMap<String, HashSet<usize>>,
}

impl ElementBlock {
    fn len(&self) -> usize {
        self.connectivity.len()
    }
    fn params(&self) -> &HashMap<String, f64> {
        &self.params
    }
    fn fields(&self) -> &HashMap<String, ArrayD<f64>> {
        &self.fields
    }
    fn families(&self) -> &Array1<usize> {
        &self.families
    }
    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        &self.groups
    }
    fn compo_type(&self) -> ElementType {
        self.cell_type.into()
    }
    fn element_connectivity<'a>(&'a self, index: usize) -> ArrayView1<'a, usize> {
        self.connectivity.element_connectivity(index)
    }
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
    // fn element_connectivity_mut(&mut self, index: usize) -> ArrayViewMut1<usize> {
    //     self.connectivity.element_connectivity_mut(index)
    // }

    // fn iter_mut<'a>(
    //     &'a mut self,
    //     coords: &'a Array2<f64>,
    // ) -> impl Iterator<Item = ElementMut<'a>> + 'a {
    //     (0..self.len()).map(move |i| {
    //         let connectivity = self.element_connectivity_mut(i);
    //         let family = &mut self.families_mut()[i]; // Access mutable family

    //         let fields = self
    //             .fields_mut()
    //             .iter_mut()
    //             .map(|(k, v)| (k.as_str(), v.index_axis_mut(Axis(0), i)))
    //             .collect();

    //         ElementMut {
    //             global_index: i,
    //             coords,
    //             connectivity,
    //             family,
    //             fields,
    //             groups: self.groups(),
    //             element_type: self.compo_type(),
    //         }
    //     })
    // }

    // fn par_iter_mut<'a>(
    //     &'a mut self,
    //     coords: &'a Array2<f64>,
    // ) -> impl ParallelIterator<Item = ElementMut<'a>> + 'a {
    //     let len = self.len();
    //     let families_ptr = self.families_mut() as *mut Array1<usize>;
    //     let fields_ptr = self.fields_mut() as *mut HashMap<String, ArrayD<f64>>;

    //     (0..len).into_par_iter().map(move |i| {
    //         let connectivity = unsafe { self.element_connectivity_mut(i) };
    //         let family = unsafe { &mut (*families_ptr)[i] };
    //         let fields = unsafe {
    //             (*fields_ptr)
    //                 .iter_mut()
    //                 .map(|(k, v)| (k.as_str(), v.index_axis_mut(Axis(0), i)))
    //                 .collect()
    //         };

    //         ElementMut {
    //             global_index: i,
    //             coords,
    //             connectivity,
    //             family,
    //             fields,
    //             groups: self.groups(),
    //             element_type: self.compo_type(),
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
    use ndarray::{array, Array1, Array2};
    use std::collections::{HashMap, HashSet};

    fn dummy_regular_cells() -> RegularCells {
        let connectivity = array![[0, 1, 2], [2, 3, 0]];
        let families = Array1::from(vec![0, 1]);

        let mut groups = HashMap::new();
        groups.insert("groupA".into(), HashSet::from([0]));
        groups.insert("groupB".into(), HashSet::from([1]));

        RegularCells {
            cell_type: ElementType::TRI3,
            connectivity,
            params: HashMap::new(),
            fields: HashMap::new(),
            families,
            groups,
        }
    }

    #[test]
    fn test_regular_cells_len() {
        let rc = dummy_regular_cells();
        assert_eq!(rc.len(), 2);
    }

    #[test]
    fn test_element_block_variant() {
        let rc = dummy_regular_cells();
        let eb = ElementBlock::RegularCells(rc);
        if let ElementBlock::RegularCells(inner) = eb {
            assert_eq!(inner.connectivity.nrows(), 2);
        } else {
            panic!("Expected RegularCells variant");
        }
    }
}
