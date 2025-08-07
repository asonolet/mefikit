mod connectivity;
mod element;
mod element_block;
mod utils;

pub use self::element::{
    Dimension, Element, ElementId, ElementIds, ElementLike, ElementMut, ElementType, Regularity,
};

#[allow(unused_imports)]
pub(crate) use self::connectivity::Connectivity;

use derive_where::derive_where;
use ndarray as nd;
use ndarray::prelude::*;
use std::collections::{BTreeMap, HashMap};

use self::connectivity::ConnectivityBase;
use self::element_block::{
    ElementBlock, ElementBlockBase, ElementBlockView, IntoElementBlockEntry,
};
use self::utils::SortedVecKey;

/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements
/// kinds and fields associated.
#[derive_where(Clone; N: nd::RawDataClone, C: nd::RawDataClone, F: nd::RawDataClone, G: nd::RawDataClone)]
#[derive_where(Debug, Serialize, PartialEq)]
#[derive_where(Deserialize; N: nd::DataOwned, C: nd::DataOwned, F: nd::DataOwned, G: nd::DataOwned)]
pub struct UMeshBase<N, C, F, G>
where
    N: nd::RawData<Elem = f64> + nd::Data,   // Nodes (Coords) data
    C: nd::RawData<Elem = usize> + nd::Data, // Connectivity data
    F: nd::RawData<Elem = f64> + nd::Data,   // Fields data
    G: nd::RawData<Elem = usize> + nd::Data, // Groups data
{
    pub(crate) coords: ArrayBase<N, Ix2>, // TODO: Use ArcArray2 for shared ownership
    pub(crate) element_blocks: BTreeMap<ElementType, ElementBlockBase<C, F, G>>,
}

pub type UMesh =
    UMeshBase<nd::OwnedRepr<f64>, nd::OwnedRepr<usize>, nd::OwnedRepr<f64>, nd::OwnedRepr<usize>>;

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
        let mut view = UMeshView::new(self.coords());
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

    pub(crate) fn coords(&self) -> ArrayView2<'_, f64> {
        self.coords.view()
    }

    pub fn space_dimension(&self) -> usize {
        self.coords.shape()[1]
    }

    pub fn elements(&self) -> impl Iterator<Item = Element> {
        self.element_blocks
            .values()
            .flat_map(|block| block.iter(self.coords.view()))
    }

    pub fn num_elements(&self) -> usize {
        self.element_blocks.values().map(|block| block.len()).sum()
    }

    pub fn get_element(&self, id: ElementId) -> Element {
        let eb = self.element_blocks.get(&id.element_type()).unwrap();
        eb.get(id.index(), self.coords.view())
    }

    pub fn elements_of_dim(&self, dim: Dimension) -> impl Iterator<Item = Element> {
        self.element_blocks
            .iter()
            .filter(move |(k, _)| k.dimension() == dim)
            .flat_map(|(_, block)| block.iter(self.coords.view()))
    }

    // TODO: check that it is a good idea to return an ElementBlockBase rather than an
    // ElementBlockView
    // pub(crate) fn element_blocks(&self) -> &BTreeMap<ElementType, ElementBlockBase<C, F, G>> {
    //     &self.element_blocks
    // }

    // pub(crate) fn element_block(
    //     &self,
    //     element_type: ElementType,
    // ) -> Option<&ElementBlockBase<C, F, G>> {
    //     self.element_blocks.get(&element_type)
    // }

    /// Extracts a sub-mesh from the current mesh based on the provided element IDs.
    ///
    /// This method creates a new `UMesh`, owning its data (with copy) containing only the elements
    /// specified by the IDs.
    pub fn extract(&self, ids: &ElementIds) -> UMesh {
        todo!();
    }

    // pub fn families(&self, element_type: ElementType) -> Option<&[usize]> {
    //     let eb = self.element_block(element_type);
    //     match eb {
    //         Some(eb) => Some(&eb.families),
    //         None => None,
    //     }
    // }

    /// This method is used to replace elements in the current mesh with another mesh.
    ///
    /// Please mind what you are doing, this method wont check for mesh constitency.
    ///
    /// The element number is the same, in which case we just replace the elements inplace. It is
    /// efficient except in the case of poly elements. It can be used to reorder elements, change
    /// their fields or families, change their node order, etc. The ElementIds are still valid and
    /// can be used to access the elements in the mesh.
    pub fn replace_inplace(&mut self, ids: &ElementIds, replace_mesh: &UMesh) {
        todo!();
    }

    /// This method is used to replace elements in the current mesh with another mesh.
    ///
    /// Please mind what you are doing, this method wont check for mesh constitency.
    ///
    /// The element number is potentially different, in which case we need to remove the elements
    /// from the current mesh and add the elements from the replace mesh. This creates a new mesh
    /// because everything needs to be reallocated to be copied either way.
    /// ElementIds are invalid on the new mesh.
    pub fn replace(&self, ids: &ElementIds, replace_mesh: &UMesh) -> UMesh {
        todo!()
    }

    /// This method is used to compute a subentity mesh.
    ///
    /// By default, the mesh computed as a codimension of 1 with the entry mesh. Meaning that there
    /// is a difference of 1 in their dimensions. Hence volumes gives faces mesh, faces gives edges
    /// mesh and edges mesh gives vertices.  If the codim asked for is too high, the function will
    /// panick.  For performance reason, two subentities are considered the same if they have the
    /// same nodes, regardless of their order.
    pub fn compute_submesh(
        &self,
        codim: Option<Dimension>,
    ) -> (
        UMesh,
        HashMap<ElementId, ElementId>,
        HashMap<ElementId, ElementId>,
    ) {
        let codim = match codim {
            Some(c) => c,
            None => Dimension::D1,
        };
        let mut subentities_hash: HashMap<SortedVecKey, ElementId> = HashMap::new();
        let mut subentity_to_elem: HashMap<ElementId, ElementId> = HashMap::new();
        let mut elem_to_subentity: HashMap<ElementId, ElementId> = HashMap::new();
        let mut neighbors: UMesh = UMesh::new(self.coords().to_owned());

        for elem in self.elements() {
            for (et, conn) in elem.subentities(Some(codim)).unwrap() {
                let subentity_id = match neighbors.element_blocks.get(&et) {
                    Some(block) => block.len(),
                    None => 0,
                };
                let key = SortedVecKey::new(conn.clone());
                if let Some(val) = subentities_hash.get(&key) {
                    // The subentity already exists
                    subentity_to_elem.insert(*val, elem.id());
                    elem_to_subentity.insert(elem.id(), *val);
                } else {
                    // The subentity is new
                    let new_id = ElementId::new(et, subentity_id);
                    subentities_hash.insert(key, new_id);
                    subentity_to_elem.insert(new_id, elem.id());
                    elem_to_subentity.insert(elem.id(), new_id);
                    neighbors.add_element(et, conn.as_slice(), None, None);
                }
            }
        }

        (neighbors, subentity_to_elem, elem_to_subentity)
    }

    pub fn normal(&self) -> BTreeMap<ElementType, Array2<f64>> {
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
    pub fn new(coords: nd::Array2<f64>) -> Self {
        Self {
            coords,
            element_blocks: BTreeMap::new(),
        }
    }

    pub fn add_regular_block(&mut self, et: ElementType, block: Array2<usize>) {
        let block = ElementBlock::new_regular(et, block);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
    }

    pub fn add_poly_block(&mut self, et: ElementType, conn: Array1<usize>, offsets: Array1<usize>) {
        let block = ElementBlock::new_poly(et, conn, offsets);
        let (key, wrapped) = block.into_entry();
        self.element_blocks.entry(key).or_insert(wrapped);
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

    pub fn remove_elements(&mut self, ids: &ElementIds) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::umesh::ElementType;
    use approx::*;
    use ndarray as nd;

    fn make_test_2d_mesh() -> UMesh {
        let coords =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::QUAD4, nd::arr2(&[[0, 1, 3, 2]]));
        mesh
    }

    #[test]
    fn test_umesh_creation() {
        let coords = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1], [1, 2]]));
        assert_eq!(mesh.coords().shape(), &[3, 1]);
        assert_eq!(mesh.element_blocks.len(), 1);
        assert!(mesh.element_blocks.contains_key(&ElementType::SEG2));
    }
    #[test]
    fn test_umesh_element_iteration() {
        let mesh = make_test_2d_mesh();

        let elements: Vec<Element> = mesh.elements().collect();
        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].element_type, ElementType::QUAD4);
        assert_eq!(elements[0].connectivity, nd::arr1(&[0, 1, 3, 2]));
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
        let mesh = make_test_2d_mesh();
        let element = mesh.get_element(ElementId::new(ElementType::QUAD4, 0));
        assert_eq!(element.element_type, ElementType::QUAD4);
        assert_eq!(element.connectivity, nd::arr1(&[0, 1, 3, 2]));
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
        for i in [40] {
            let mesh = crate::RegularUMeshBuilder::new()
                .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
                .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
                .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
                .build();
            mesh.view();
        }
    }
}
