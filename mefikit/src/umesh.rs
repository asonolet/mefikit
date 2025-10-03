mod connectivity;
mod dataarray;
mod element;

pub(crate) use self::connectivity::{Connectivity, ConnectivityView};
pub use self::dataarray::DataArray;
pub use self::element::{
    Dimension, Element, ElementId, ElementIds, ElementLike, ElementMut, ElementType, Regularity,
};

use ndarray as nd;
use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// The part of a mesh constituted by one kind of element.
///
/// The element block is the base structure to hold connectivity, fields, groups.
/// It is used to hold all cell information and allows cell iteration.
/// The only data not included for an element block to be standalone is the coordinates array.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct ElementBlock {
    pub cell_type: ElementType,
    pub connectivity: Connectivity,
    pub fields: BTreeMap<String, ArcArray<f64, IxDyn>>,
    pub groups: BTreeMap<String, BTreeSet<usize>>,
}

#[derive(Clone, Debug, Serialize, PartialEq)]
struct ElementBlockView<'a> {
    pub cell_type: ElementType,
    pub connectivity: ConnectivityView<'a>,
    pub fields: BTreeMap<String, DataArray<'a, f64, IxDyn>>,
    pub groups: BTreeMap<String, BTreeSet<usize>>,
}

/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements
/// kinds and fields associated.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct UMesh {
    coords: nd::ArcArray2<f64>,
    element_blocks: BTreeMap<ElementType, ElementBlock>,
}

pub struct UMeshView<'a> {
    coords: DataArray<'a, f64, Ix2>,
    element_blocks: BTreeMap<ElementType, ElementBlockView<'a>>,
}

impl<'a> UMeshView<'a> {
    pub fn new(coords: DataArray<'a, f64, Ix2>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn to_owned(&self) -> UMesh {
        let mut umesh = UMesh::new(self.coords.to_shared());
        for (&et, eb) in &self.element_blocks {
            match eb.connectivity {
                ConnectivityView::Regular(r) => umesh.add_regular_block(et, r.to_owned()),
                ConnectivityView::Poly { data, offsets } => {
                    umesh.add_poly_block(et, data.to_owned(), offsets.to_owned())
                }
            }
        }
        umesh
    }

    pub fn add_regular_block(&mut self, et: ElementType, block: ArrayView2<'a, usize>) {
        let block = ElementBlockView {
            cell_type: et,
            connectivity: ConnectivityView::Regular(block),
            fields: BTreeMap::new(),
            groups: BTreeMap::new(),
        };
        self.element_blocks.entry(et).or_insert(block);
    }

    pub fn add_poly_block(
        &mut self,
        et: ElementType,
        conn: ArrayView1<'a, usize>,
        offsets: ArrayView1<'a, usize>,
    ) {
        let block = ElementBlockView {
            cell_type: et,
            connectivity: ConnectivityView::Poly {
                data: conn,
                offsets,
            },
            fields: BTreeMap::new(),
            groups: BTreeMap::new(),
        };
        self.element_blocks.entry(et).or_insert(block);
    }

    pub fn coords(&self) -> ArrayView2<'_, f64> {
        self.coords.view()
    }

    /// Low-level method to get view on the underlying connectivity array.
    ///
    /// Please consider using the elements() iterator which give the connectivity element by
    /// element with zero costs.
    pub fn regular_connectivity(&self, et: ElementType) -> Result<ArrayView2<'_, usize>, String> {
        match &self
            .element_blocks
            .get(&et)
            .ok_or_else(|| "Element is not in the mesh.".to_owned())?
            .connectivity
        {
            ConnectivityView::Regular(tab) => Ok(tab.view()),
            _ => Err(
                "This element type has poly connectivity, please use poly_connectivity(et) method."
                    .to_owned(),
            ),
        }
    }

    /// Low-level method to get view on the underlying connectivity arrays.
    ///
    /// Please consider using the elements() iterator which give the connectivity element by
    /// element with zero costs.
    pub fn poly_connectivity(
        &self,
        et: ElementType,
    ) -> Result<(ArrayView1<'_, usize>, ArrayView1<'_, usize>), String> {
        match &self
            .element_blocks
            .get(&et)
            .ok_or_else(|| "Element is not in the mesh.".to_owned())?
            .connectivity
        {
            ConnectivityView::Poly { data, offsets } => Ok((data.view(), offsets.view())),
            _ => Err(
                "This element type has regular connectivity, please use regular_connectivity(et) method."
                    .to_owned(),
            ),
        }
    }

    pub fn space_dimension(&self) -> usize {
        self.coords().shape()[1]
    }

    pub fn elements(&'a self) -> impl Iterator<Item = Element<'a>> {
        self.element_blocks.values().flat_map(move |block| {
            block
                .connectivity
                .iter()
                .enumerate()
                .map(move |(i, connectivity)| {
                    Element::new(
                        i,
                        self.coords().clone(),
                        None,
                        &0,
                        &block.groups,
                        connectivity,
                        block.cell_type,
                    )
                })
        })
    }

    pub fn par_elements(&self) -> impl ParallelIterator<Item = Element> {
        self.element_blocks.par_iter().flat_map(move |(_, block)| {
            (0..block.connectivity.len())
                .into_par_iter()
                .with_min_len(200)
                .map(move |i| {
                    let connectivity = block.connectivity.get(i);
                    // let fields = self
                    //     .fields
                    //     .iter()
                    //     .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                    //     .collect();

                    Element::new(
                        i,
                        self.coords().clone(),
                        None,
                        &0,
                        &block.groups,
                        connectivity,
                        block.cell_type,
                    )
                })
        })
    }

    pub fn num_elements(&self) -> usize {
        self.element_blocks
            .values()
            .map(|block| block.connectivity.len())
            .sum()
    }

    pub fn get_element(&self, id: ElementId) -> Element {
        let eb = self.element_blocks.get(&id.element_type()).unwrap();
        let connectivity = eb.connectivity.get(id.index());
        // let fields = self
        //     .fields
        //     .iter()
        //     .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), index)))
        //     .collect();
        Element::new(
            id.index(),
            self.coords().clone(),
            None,
            &0,
            &eb.groups,
            connectivity,
            id.element_type(),
        )
    }

    // pub fn elements_of_dim(&self, dim: Dimension) -> impl Iterator<Item = Element> {
    //     self.element_blocks
    //         .iter()
    //         .filter(move |(k, _)| k.dimension() == dim)
    //         .flat_map(|(_, block)| block.iter(self.coords.view()))
    // }

    // pub fn par_elements_of_dim(&self, dim: Dimension) -> impl ParallelIterator<Item = Element>
    // where
    //     N: Sync,
    //     C: Sync,
    //     F: Sync,
    //     G: Sync,
    // {
    //     self.element_blocks
    //         .par_iter()
    //         .filter(move |(k, _)| k.dimension() == dim)
    //         .flat_map(|(_, block)| block.par_iter(self.coords.view()))
    // }

    // /// Extracts a sub-mesh from the current mesh based on the provided element IDs.
    // ///
    // /// This method creates a new `UMesh`, owning its data (with copy) containing only the elements
    // /// specified by the IDs.
    // /// This method is low level and error prone in the case where ElementsIds are not directly
    // /// issued from a Selector. Please use Selector API if possible.
    // pub fn extract(&self, ids: &ElementIds) -> UMeshView<'a> {
    //     // Attention, dans le cas d'un UMeshView c'est une copie, passer des views partout ça
    //     // empèche de garder la ref vers le tableau des coordonnées.
    //     let mut extracted = UMeshView::new(self.coords.clone());
    //     for (t, block) in ids.iter() {
    //         if !self.element_blocks.contains_key(t) {
    //             continue;
    //         }
    //         match &self.element_blocks[t] {
    //             ElementBlockView {
    //                 connectivity: ConnectivityView::Regular(arr),
    //                 ..
    //             } => extracted.add_regular_block(*t, arr.select(Axis(0), block.as_slice())),
    //             _ => todo!(),
    //         };
    //     }
    //     extracted
    // }

    // pub fn families(&self, element_type: ElementType) -> Option<&[usize]> {
    //     let eb = self.element_block(element_type);
    //     match eb {
    //         Some(eb) => Some(&eb.families),
    //         None => None,
    //     }
    // }

    // /// This method is used to replace elements in the current mesh with another mesh, producing a
    // /// new mesh.
    // ///
    // /// Please mind what you are doing, this method wont check for mesh constitency.
    // ///
    // /// The element number is potentially different, in which case we need to remove the elements
    // /// from the current mesh and add the elements from the replace mesh. This creates a new mesh
    // /// because everything needs to be reallocated to be copied either way.
    // /// ElementIds are invalid on the new mesh.
    // pub fn replace(&self, _ids: &ElementIds, _replace_mesh: &UMesh) -> UMesh {
    //     todo!()
    // }
}

impl<'a> UMeshView<'a> {}

impl UMesh {
    pub fn new(coords: nd::ArcArray2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_regular_block(&mut self, et: ElementType, block: Array2<usize>) {
        let block = ElementBlock {
            cell_type: et,
            connectivity: Connectivity::Regular(block),
            fields: BTreeMap::new(),
            groups: BTreeMap::new(),
        };
        self.element_blocks.entry(et).or_insert(block);
    }

    pub fn add_poly_block(&mut self, et: ElementType, conn: Array1<usize>, offsets: Array1<usize>) {
        let block = ElementBlock {
            cell_type: et,
            connectivity: Connectivity::Poly {
                data: conn,
                offsets,
            },
            fields: BTreeMap::new(),
            groups: BTreeMap::new(),
        };
        self.element_blocks.entry(et).or_insert(block);
    }

    pub fn to_owned(self) -> UMesh {
        self
    }

    pub fn add_element(
        &mut self,
        element_type: ElementType,
        connectivity: &[usize],
        family: Option<usize>,
        fields: Option<BTreeMap<String, ArrayViewD<f64>>>,
    ) {
        match element_type.regularity() {
            Regularity::Regular => {
                if connectivity.len() != element_type.num_nodes().unwrap() {
                    panic!(
                        "Connectivity length does not match the number of nodes for element type {element_type:?}"
                    );
                }
                self.element_blocks.entry(element_type).or_insert_with(|| {
                    ElementBlock::new_regular(
                        element_type,
                        Array2::zeros((0, element_type.num_nodes().unwrap())),
                    )
                });
            }
            Regularity::Poly => {
                self.element_blocks
                    .entry(element_type)
                    .or_insert_with(|| ElementBlock::new_poly(element_type, arr1(&[]), arr1(&[])));
            }
        }

        self.element_blocks
            .get_mut(&element_type)
            .unwrap() // This unwrap is safe because we just inserted the element type
            .add_element(ArrayView1::from(connectivity), family, fields);
    }

    pub fn remove_elements(&mut self, _ids: &ElementIds) {
        todo!()
    }

    /// This is the most efficient way because it does not copy coords if no reallocation is
    /// needed if coords are not shared. When coords are shared it is copied either way.
    pub fn append_coords(
        mut self,
        added_coords: ArrayView2<'_, f64>,
    ) -> Result<Self, nd::ShapeError> {
        let mut coords = self.coords.into_owned();
        coords.append(Axis(0), added_coords)?;
        self.coords = coords.into_shared();
        Ok(self)
    }

    /// This is kind of efficient: coordinates are reallocated and copied but connectivities are
    /// modified inplace.
    pub fn prepend_coords(mut self, added_coords: ArrayView2<'_, f64>) -> Self {
        let n_coords = added_coords.len_of(Axis(0));
        self.coords = nd::concatenate![Axis(0), added_coords, self.coords].into_shared();
        for (_, eb) in self.element_blocks.iter_mut() {
            match &mut eb.connectivity {
                ConnectivityView::Regular(c) => *c += n_coords,
                ConnectivityBase::Poly { data, .. } => *data += n_coords,
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples as me;
    use crate::umesh::ElementType;

    #[test]
    fn test_umesh_creation() {
        let mesh = me::make_mesh_3d_seg2();
        assert_eq!(mesh.coords.shape(), &[3, 1]);
        assert_eq!(mesh.element_blocks.len(), 1);
        assert!(mesh.element_blocks.contains_key(&ElementType::SEG2));
    }
    #[test]
    fn test_umesh_element_iteration() {
        let mesh = me::make_mesh_2d_multi();

        let elements: Vec<Element> = mesh.elements().collect();
        assert_eq!(elements.len(), 4);
        assert_eq!(elements[0].element_type, ElementType::SEG2);
        assert_eq!(elements[0].connectivity, &[0, 1]);
        assert_eq!(elements[1].element_type, ElementType::SEG2);
        assert_eq!(elements[1].connectivity, &[1, 3]);
        assert_eq!(elements[2].element_type, ElementType::QUAD4);
        assert_eq!(elements[2].connectivity, &[0, 1, 3, 2]);
        assert_eq!(elements[3].element_type, ElementType::PGON);
        assert_eq!(elements[3].connectivity, &[0, 1, 5, 3, 2]);
    }
    // #[test]
    // fn test_umesh_element_block_addition() {
    //     let mut mesh = make_test_2d_mesh();
    //     assert_eq!(mesh.element_blocks().len(), 1);
    //     assert!(mesh.element_blocks().contains_key(&ElementType::QUAD4));

    //     mesh.add_element(
    //         ElementType::TRI3,
    //         &[0, 1, 2],
    //         Some(0),
    //         Some(BTreeMap::new()),
    //     );
    //     assert_eq!(mesh.element_blocks().len(), 2);
    //     assert!(mesh.element_blocks().contains_key(&ElementType::TRI3));
    // }
    #[test]
    fn test_umesh_element_retrieval() {
        let mesh = me::make_mesh_2d_multi();
        let element = mesh.get_element(ElementId::new(ElementType::QUAD4, 0));
        assert_eq!(element.element_type, ElementType::QUAD4);
        assert_eq!(element.connectivity, &[0, 1, 3, 2]);
    }
    // #[test]
    // fn test_umesh_extract_mesh() {
    //     let mesh = make_test_2d_mesh();
    //     let ids = vec![ElementId::new(ElementType::QUAD4, 0)];
    //     let sub_mesh = mesh.extract(&ids);
    //     assert_eq!(sub_mesh.element_blocks().len(), 1);
    //     assert!(sub_mesh.element_blocks().contains_key(&ElementType::QUAD4));
    //     assert_eq!(sub_mesh.coords().shape(), &[4, 2]);
    // }

    #[test]
    fn test_umesh_view() {
        let mesh = me::make_imesh_3d(40);
        mesh.view();
    }
}
