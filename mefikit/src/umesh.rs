mod connectivity;
mod element;
mod element_block;

pub use self::element::{
    Dimension, Element, ElementId, ElementIds, ElementLike, ElementMut, ElementType, Regularity,
};

pub(crate) use self::connectivity::Connectivity;

use derive_where::derive_where;
use ndarray as nd;
use ndarray::prelude::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

use self::connectivity::ConnectivityBase;
use self::element_block::{
    ElementBlock, ElementBlockBase, ElementBlockView, IntoElementBlockEntry,
};

/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements
/// kinds and fields associated.
#[derive_where(Debug, Serialize, PartialEq)]
#[derive_where(Deserialize; N: nd::DataOwned, C: nd::DataOwned, F: nd::DataOwned, G: nd::DataOwned)]
pub struct UMeshBase<N, C, F, G>
where
    N: nd::RawData<Elem = f64> + nd::Data,   // Nodes (Coords) data
    C: nd::RawData<Elem = usize> + nd::Data, // Connectivity data
    F: nd::RawData<Elem = f64> + nd::Data,   // Fields data
    G: nd::RawData<Elem = usize> + nd::Data, // Groups data
{
    pub coords: ArrayBase<N, Ix2>,
    pub element_blocks: BTreeMap<ElementType, ElementBlockBase<C, F, G>>,
}

pub type UMesh = UMeshBase<
    nd::OwnedArcRepr<f64>,
    nd::OwnedArcRepr<usize>,
    nd::OwnedArcRepr<f64>,
    nd::OwnedArcRepr<usize>,
>;

pub type UMeshView<'a> = UMeshBase<
    nd::ViewRepr<&'a f64>,
    nd::ViewRepr<&'a usize>,
    nd::ViewRepr<&'a f64>,
    nd::ViewRepr<&'a usize>,
>;

impl<N, C, F, G> UMeshBase<N, C, F, G>
where
    N: nd::RawData<Elem = f64> + nd::Data,
    C: nd::RawData<Elem = usize> + nd::Data,
    F: nd::RawData<Elem = f64> + nd::Data,
    G: nd::RawData<Elem = usize> + nd::Data,
{
    pub fn view(&self) -> UMeshView<'_> {
        let mut view = UMeshView::new(self.coords.view());
        for (&et, block) in self.element_blocks.iter() {
            match &block.connectivity {
                ConnectivityBase::Regular(arr) => view.add_regular_block(et, arr.view()),
                ConnectivityBase::Poly { data, offsets } => {
                    view.add_poly_block(et, data.view(), offsets.view())
                }
            };
        }
        view
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
            ConnectivityBase::Regular(tab) => Ok(tab.view()),
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
            ConnectivityBase::Poly { data, offsets } => Ok((data.view(), offsets.view())),
            _ => Err(
                "This element type has regular connectivity, please use regular_connectivity(et) method."
                    .to_owned(),
            ),
        }
    }

    pub fn space_dimension(&self) -> usize {
        self.coords.shape()[1]
    }

    pub fn elements(&self) -> impl Iterator<Item = Element<'_>> {
        self.element_blocks
            .values()
            .flat_map(|block| block.iter(self.coords.view()))
    }

    #[cfg(feature = "rayon")]
    pub fn par_elements(&self) -> impl ParallelIterator<Item = Element<'_>>
    where
        N: Sync,
        C: Sync,
        F: Sync,
        G: Sync,
    {
        self.element_blocks
            .par_iter()
            .flat_map(|(_, block)| block.par_iter(self.coords.view()))
    }

    #[cfg(not(feature = "rayon"))]
    pub fn par_elements(&self) -> impl Iterator<Item = Element<'_>> {
        self.elements()
    }

    pub fn num_elements(&self) -> usize {
        self.element_blocks.values().map(|block| block.len()).sum()
    }

    pub fn get_element(&self, id: ElementId) -> Element<'_> {
        let eb = self.element_blocks.get(&id.element_type()).unwrap();
        eb.get(id.index(), self.coords.view())
    }

    pub fn elements_of_dim(&self, dim: Dimension) -> impl Iterator<Item = Element<'_>> {
        self.element_blocks
            .iter()
            .filter(move |(k, _)| k.dimension() == dim)
            .flat_map(|(_, block)| block.iter(self.coords.view()))
    }

    #[cfg(feature = "rayon")]
    pub fn par_elements_of_dim(&self, dim: Dimension) -> impl ParallelIterator<Item = Element<'_>>
    where
        N: Sync,
        C: Sync,
        F: Sync,
        G: Sync,
    {
        self.element_blocks
            .par_iter()
            .filter(move |(k, _)| k.dimension() == dim)
            .flat_map(|(_, block)| block.par_iter(self.coords.view()))
    }

    #[cfg(not(feature = "rayon"))]
    pub fn par_elements_of_dim(&self, dim: Dimension) -> impl Iterator<Item = Element<'_>> {
        self.elements_of_dim(dim)
    }

    // pub fn families(&self, element_type: ElementType) -> Option<&[usize]> {
    //     let eb = self.element_block(element_type);
    //     match eb {
    //         Some(eb) => Some(&eb.families),
    //         None => None,
    //     }
    // }

    /// This method is used to replace elements in the current mesh with another mesh, producing a
    /// new mesh.
    ///
    /// Please mind what you are doing, this method wont check for mesh consistency.
    ///
    /// The element number is potentially different, in which case we need to remove the elements
    /// from the current mesh and add the elements from the replace mesh. This creates a new mesh
    /// because everything needs to be reallocated to be copied either way.
    /// `ElementIds` are invalid on the new mesh.
    pub fn replace(&self, _ids: &ElementIds, _replace_mesh: &UMesh) -> UMesh {
        todo!()
    }
}

impl<'a> UMeshView<'a> {
    pub fn new(coords: nd::ArrayView2<'a, f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn to_shared(&self) -> UMesh {
        let mut umesh = UMesh::new(self.coords.to_shared());
        for (&et, eb) in &self.element_blocks {
            match eb.connectivity {
                ConnectivityBase::Regular(r) => umesh.add_regular_block(et, r.to_shared()),
                ConnectivityBase::Poly { data, offsets } => {
                    umesh.add_poly_block(et, data.to_shared(), offsets.to_shared())
                }
            }
        }
        umesh
    }

    pub fn add_regular_block(&mut self, et: ElementType, block: ArrayView2<'a, usize>) {
        let block = ElementBlockView::new_regular(et, block);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }

    pub fn add_poly_block(
        &mut self,
        et: ElementType,
        conn: ArrayView1<'a, usize>,
        offsets: ArrayView1<'a, usize>,
    ) {
        let block = ElementBlockView::new_poly(et, conn, offsets);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }
}

impl UMesh {
    pub fn new(coords: nd::ArcArray2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_regular_block(&mut self, et: ElementType, block: nd::ArcArray2<usize>) {
        let block = ElementBlock::new_regular(et, block);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }

    pub fn add_poly_block(
        &mut self,
        et: ElementType,
        conn: nd::ArcArray1<usize>,
        offsets: nd::ArcArray1<usize>,
    ) {
        let block = ElementBlock::new_poly(et, conn, offsets);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
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
                        nd::ArcArray2::zeros((0, element_type.num_nodes().unwrap())),
                    )
                });
            }
            Regularity::Poly => {
                self.element_blocks.entry(element_type).or_insert_with(|| {
                    ElementBlock::new_poly(
                        element_type,
                        arr1(&[]).into_shared(),
                        arr1(&[]).into_shared(),
                    )
                });
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

    /// This is the most efficient way because it does not copy coordinates if no reallocation is
    /// needed if coordinates are not shared. When coordinates are shared it is copied either way.
    pub fn append_coords(
        &mut self,
        added_coords: ArrayView2<'_, f64>,
    ) -> Result<(), nd::ShapeError> {
        let mut coords = std::mem::take(&mut self.coords).into_owned();
        coords.append(Axis(0), added_coords)?;
        self.coords = coords.into_shared();
        Ok(())
    }

    /// This is kind of efficient: coordinates are reallocated and copied but connectivities are
    /// modified in-place.
    pub fn prepend_coords(mut self, added_coords: ArrayView2<'_, f64>) -> Self {
        let n_coords = added_coords.len_of(Axis(0));
        self.coords = nd::concatenate![Axis(0), added_coords, self.coords].into_shared();
        for (_, eb) in self.element_blocks.iter_mut() {
            match &mut eb.connectivity {
                ConnectivityBase::Regular(c) => *c += n_coords,
                ConnectivityBase::Poly { data, .. } => *data += n_coords,
            }
        }
        self
    }

    /// Extracts a sub-mesh from the current mesh based on the provided element IDs.
    ///
    /// This method creates a new `UMesh`, owning its data (with copy) containing only the elements
    /// specified by the IDs.
    /// This method is low level and error prone in the case where `ElementsIds` are not directly
    /// issued from a Selector. Please use Selector API if possible.
    pub fn extract(&self, ids: &ElementIds) -> UMesh {
        let mut extracted = UMesh::new(self.coords.clone());
        for (t, block) in ids.iter() {
            if !self.element_blocks.contains_key(t) {
                continue;
            }
            match &self.element_blocks[t] {
                ElementBlockBase {
                    connectivity: ConnectivityBase::Regular(arr),
                    ..
                } => extracted
                    .add_regular_block(*t, arr.select(Axis(0), block.as_slice()).into_shared()),
                _ => todo!(),
            };
        }
        extracted
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
