use derive_where::derive_where;
use ndarray as nd;
use ndarray::prelude::*;
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use crate::umesh::connectivity::{Connectivity, ConnectivityBase, ConnectivityView};
use crate::umesh::element::{Element, ElementType};

/// The part of a mesh constituted by one kind of element.
///
/// The element block is the base structure to hold connectivity, fields, groups.
/// It is used to hold all cell information and allows cell iteration.
/// The only data not included for an element block to be standalone is the coordinates array.
#[derive_where(Clone; C: nd::RawDataClone, F: nd::RawDataClone, G: nd::RawDataClone)]
#[derive_where(Debug, Serialize, PartialEq)]
#[derive_where(Deserialize; C: nd::DataOwned, F: nd::DataOwned, G: nd::DataOwned)]
pub struct ElementBlockBase<C, F, G>
where
    C: nd::RawData<Elem = usize> + nd::Data,
    F: nd::RawData<Elem = f64> + nd::Data,
    G: nd::RawData<Elem = usize> + nd::Data,
{
    pub cell_type: ElementType,
    pub connectivity: ConnectivityBase<C>,
    pub fields: BTreeMap<String, ArrayBase<F, nd::IxDyn>>,
    pub families: ArrayBase<G, nd::Ix1>,
    pub groups: BTreeMap<String, BTreeSet<usize>>,
}

pub type ElementBlock =
    ElementBlockBase<nd::OwnedRepr<usize>, nd::OwnedRepr<f64>, nd::OwnedRepr<usize>>;

pub type ElementBlockView<'a> =
    ElementBlockBase<nd::ViewRepr<&'a usize>, nd::ViewRepr<&'a f64>, nd::ViewRepr<&'a usize>>;

impl<C, F, G> ElementBlockBase<C, F, G>
where
    C: nd::RawData<Elem = usize> + nd::Data,
    F: nd::RawData<Elem = f64> + nd::Data,
    G: nd::RawData<Elem = usize> + nd::Data,
{
    pub fn len(&self) -> usize {
        self.connectivity.len()
    }

    pub fn element_connectivity(&self, index: usize) -> ArrayView1<'_, usize>
    where
        C: nd::Data,
    {
        self.connectivity.get(index)
    }

    pub fn element_connectivity_mut(&mut self, index: usize) -> ArrayViewMut1<usize>
    where
        C: nd::DataMut,
    {
        self.connectivity.get_mut(index)
    }

    pub fn get<'a>(&'a self, index: usize, coords: ArrayView2<'a, f64>) -> Element<'a>
    where
        C: nd::Data,
        G: nd::Data,
        F: nd::Data,
    {
        let connectivity = self.element_connectivity(index);
        let fields = self
            .fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), index)))
            .collect();
        Element::new(
            index,
            coords,
            fields,
            &self.families[index],
            &self.groups,
            connectivity,
            self.cell_type,
        )
    }

    pub fn iter<'a>(&'a self, coords: ArrayView2<'a, f64>) -> impl Iterator<Item = Element<'a>> + 'a
    where
        C: nd::Data,
        G: nd::Data,
        F: nd::Data,
    {
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

    pub fn par_iter<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> impl ParallelIterator<Item = Element<'a>> + 'a
    where
        C: nd::Data + Sync,
        G: nd::Data + Sync,
        F: nd::Data + Sync,
    {
        (0..self.len()).into_par_iter().map(move |i| {
            let connectivity = self.element_connectivity(i);
            let fields = self
                .fields
                .iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();

            Element::new(
                i,
                coords.view(),
                fields,
                &self.families[i],
                &self.groups,
                connectivity,
                self.cell_type,
            )
        })
    }

    // pub fn iter_mut(
    //     &'a mut self,
    //     coords: &'a Array2<f64>,
    // ) -> impl Iterator<Item = ElementMut<'a>> + 'a {
    //     let fields: Vec<BTreeMap<&'a str, ArrayViewMutD<'a, f64>>> = (0..self.len()).map(|i| {
    //         self
    //         .fields
    //         .iter_mut()
    //         .map(|(k, v)| (k.as_str(), v.index_axis_mut(Axis(0), i)))
    //         .collect()
    //     }).collect();

    //     self.connectivity.iter_mut().zip(self.families.iter_mut()).zip(fields.iter()).enumerate().map(|(i, ((conn, fam), fields))| {

    //         ElementMut::new(
    //             i,
    //             coords,
    //             conn,
    //             fam,
    //             *fields,
    //             &self.groups,
    //             self.cell_type,
    //         )
    //     })
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

    //         let fields: BTreeMap<&'a str, ArrayViewMutD<'a, f64>> = self
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
    //          , ConnectivityBase   groups: &self.groups,
    //             element_type: self.cell_type,
    //         }
    //     })
    // }
}

impl<'a> ElementBlock {
    /// Create a new regular element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `fields` - A map of field names to their values for each element.
    /// * `families` - An array of family indices for each element.
    /// * `groups` - A map of group names to sets of element indices.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_regular(cell_type: ElementType, connectivity: Array2<usize>) -> Self {
        let conn_len = connectivity.len();
        Self {
            cell_type,
            connectivity: Connectivity::Regular(connectivity),
            fields: BTreeMap::new(),
            families: Array1::from(vec![0; conn_len]),
            groups: BTreeMap::new(),
        }
    }

    /// Create a new poly element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `fields` - A map of field names to their values for each element.
    /// * `families` - An array of family indices for each element.
    /// * `groups` - A map of group names to sets of element indices.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_poly(
        cell_type: ElementType,
        connectivity: Array1<usize>,
        offsets: Array1<usize>,
    ) -> Self {
        let conn_len = connectivity.len();
        Self {
            cell_type,
            connectivity: Connectivity::new_poly(connectivity, offsets),
            fields: BTreeMap::new(),
            families: Array1::from(vec![0; conn_len]),
            groups: BTreeMap::new(),
        }
    }

    pub fn add_element(
        &mut self,
        connectivity: ArrayView1<usize>,
        family: Option<usize>,
        fields: Option<BTreeMap<String, ArrayViewD<f64>>>,
    ) {
        self.connectivity.append(connectivity);
        let family = family.unwrap_or_default();
        self.families.append(Axis(0), array![family].view());

        if let Some(fields) = fields {
            todo!();
        }
    }
}

impl<'a> ElementBlockView<'a> {
    /// Create a new regular element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `fields` - A map of field names to their values for each element.
    /// * `families` - An array of family indices for each element.
    /// * `groups` - A map of group names to sets of element indices.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_regular(cell_type: ElementType, connectivity: ArrayView2<'a, usize>) -> Self {
        let conn_len = connectivity.len();
        let reg_vec = Box::new(Array1::from(vec![0; conn_len]));
        Self {
            cell_type,
            connectivity: ConnectivityView::Regular(connectivity),
            fields: BTreeMap::new(),
            families: Box::leak(reg_vec).view(),
            groups: BTreeMap::new(),
        }
    }

    /// Create a new poly element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `fields` - A map of field names to their values for each element.
    /// * `families` - An array of family indices for each element.
    /// * `groups` - A map of group names to sets of element indices.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_poly(
        cell_type: ElementType,
        connectivity: ArrayView1<'a, usize>,
        offsets: ArrayView1<'a, usize>,
    ) -> Self {
        let conn_len = connectivity.len();
        let reg_vec = Box::new(Array1::from(vec![0; conn_len]));
        Self {
            cell_type,
            connectivity: ConnectivityView::Poly {
                data: connectivity,
                offsets,
            },
            fields: BTreeMap::new(),
            families: Box::leak(reg_vec).view(),
            groups: BTreeMap::new(),
        }
    }
    pub fn into_entry(self) -> (ElementType, ElementBlockView<'a>) {
        (self.cell_type, self)
    }
}

pub trait IntoElementBlockEntry {
    fn into_entry(self) -> (ElementType, ElementBlock);
}

impl IntoElementBlockEntry for ElementBlock {
    fn into_entry(self) -> (ElementType, ElementBlock) {
        (self.cell_type, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::collections::BTreeMap;

    use crate::umesh::element::Element;
    use crate::umesh::element::ElementType;

    #[test]
    fn test_element_block() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]]);
        let fields = BTreeMap::new();
        let families = vec![0, 1, 2];
        let groups = BTreeMap::new();

        let element_block = ElementBlock {
            cell_type: ElementType::TRI3,
            connectivity,
            fields,
            families: families.into(),
            groups,
        };

        assert_eq!(element_block.len(), 3);
        assert_eq!(element_block.fields.len(), 0);
        assert_eq!(element_block.families.len(), 3);
        assert_eq!(element_block.groups.len(), 0);
    }

    #[test]
    fn test_element_block_iter() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]]);
        let fields = BTreeMap::new();
        let families = vec![0, 1, 2];
        let groups = BTreeMap::new();

        let element_block = ElementBlock {
            cell_type: ElementType::TRI3,
            connectivity,
            fields,
            families: families.into(),
            groups,
        };

        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let elements: Vec<Element> = element_block.iter(coords.view()).collect();

        assert_eq!(elements.len(), 3);
    }
}
