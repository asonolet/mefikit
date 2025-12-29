use derive_where::derive_where;
use ndarray as nd;
use ndarray::prelude::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use super::connectivity::{Connectivity, ConnectivityBase, ConnectivityView};
use super::element::{Element, ElementMut, ElementType};

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
    ElementBlockBase<nd::OwnedArcRepr<usize>, nd::OwnedArcRepr<f64>, nd::OwnedArcRepr<usize>>;

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

    pub fn element_connectivity(&self, index: usize) -> &[usize] {
        self.connectivity.get(index)
    }

    pub fn get<'a>(&'a self, index: usize, coords: ArrayView2<'a, f64>) -> Element<'a> {
        let connectivity = self.element_connectivity(index);
        // let fields = self
        //     .fields
        //     .iter()
        //     .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), index)))
        //     .collect();
        Element::new(
            index,
            coords,
            None,
            &self.families[index],
            &self.groups,
            connectivity,
            self.cell_type,
        )
    }

    pub fn iter<'a>(
        &'a self,
        coords: ArrayView2<'a, f64>,
    ) -> impl ExactSizeIterator<Item = Element<'a>> + 'a {
        self.connectivity
            .iter()
            .enumerate()
            .map(move |(i, connectivity)| {
                Element::new(
                    i,
                    coords,
                    None,
                    &self.families[i],
                    &self.groups,
                    connectivity,
                    self.cell_type,
                )
            })
    }
    #[cfg(not(feature = "rayon"))]
    pub fn par_iter<'a>(
        &'a self,
        coords: ArrayView2<'a, f64>,
    ) -> impl Iterator<Item = Element<'a>> + 'a {
        self.iter(coords)
    }

    #[cfg(feature = "rayon")]
    pub fn par_iter<'a>(
        &'a self,
        coords: ArrayView2<'a, f64>,
    ) -> impl ParallelIterator<Item = Element<'a>> + 'a
    where
        C: Sync,
        G: Sync,
        F: Sync,
    {
        (0..self.len())
            .into_par_iter()
            .with_min_len(200)
            .map(move |i| {
                let connectivity = self.element_connectivity(i);
                // let fields = self
                //     .fields
                //     .iter()
                //     .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                //     .collect();

                Element::new(
                    i,
                    coords,
                    None,
                    &self.families[i],
                    &self.groups,
                    connectivity,
                    self.cell_type,
                )
            })
    }
}

impl ElementBlock {
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
    pub fn new_regular(
        cell_type: ElementType,
        connectivity: nd::ArcArray2<usize>,
        families: Option<nd::ArcArray1<usize>>,
    ) -> Self {
        let conn_len = connectivity.len();
        let families = match families {
            Some(fams) => Some(fams),
            None => Some(nd::ArcArray1::from(vec![0; conn_len])),
        };
        Self {
            cell_type,
            connectivity: Connectivity::Regular(connectivity),
            fields: BTreeMap::new(),
            families: families.unwrap(),
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
        connectivity: nd::ArcArray1<usize>,
        offsets: nd::ArcArray1<usize>,
    ) -> Self {
        let conn_len = connectivity.len();
        Self {
            cell_type,
            connectivity: Connectivity::new_poly(connectivity, offsets),
            fields: BTreeMap::new(),
            families: nd::ArcArray1::from(vec![0; conn_len]),
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
        let mut new_families = std::mem::take(&mut self.families).into_owned();
        new_families.append(Axis(0), array![family].view()).unwrap();
        self.families = new_families.into_shared();

        if let Some(_fields) = fields {
            todo!();
        }
    }

    pub fn get_mut<'a>(&'a mut self, index: usize, coords: ArrayView2<'a, f64>) -> ElementMut<'a> {
        let connectivity = self.connectivity.get_mut(index);
        // let fields = self
        //     .fields
        //     .iter()
        //     .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), index)))
        //     .collect();
        ElementMut::new(
            index,
            coords,
            None,
            self.families.get_mut(index).unwrap(),
            &self.groups,
            connectivity,
            self.cell_type,
        )
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
    pub fn new_regular(
        cell_type: ElementType,
        connectivity: ArrayView2<'a, usize>,
        families: Option<ArrayView1<'a, usize>>,
    ) -> Self {
        let conn_len = connectivity.len();
        let families = match families {
            Some(fams) => Some(fams),
            None => Some(Box::leak(Box::new(Array1::from(vec![0; conn_len]))).view()),
        };
        Self {
            cell_type,
            connectivity: ConnectivityView::Regular(connectivity),
            fields: BTreeMap::new(),
            families: families.unwrap(),
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

    use crate::mesh::element::Element;
    use crate::mesh::element::ElementType;

    #[test]
    fn test_element_block() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]].to_shared());
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
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]].to_shared());
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
