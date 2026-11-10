use derive_where::derive_where;
use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::connectivity::{Connectivity, ConnectivityBase, ConnectivityView};
use super::element::{Element, ElementMut, ElementType};
use super::indirect_index::IndirectIndex;

/// A wrapper around Arc<BTreeMap<String, BTreeSet<usize>>> that implements Serialize/Deserialize.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArcGroups(pub Arc<BTreeMap<String, BTreeSet<usize>>>);

impl Serialize for ArcGroups {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ArcGroups {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let map = BTreeMap::deserialize(deserializer)?;
        Ok(ArcGroups(Arc::new(map)))
    }
}

impl ArcGroups {
    pub fn new() -> Self {
        ArcGroups(Arc::new(BTreeMap::new()))
    }

    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, String, BTreeSet<usize>> {
        self.0.iter()
    }

    #[allow(dead_code)]
    pub fn get(&self, key: &str) -> Option<&BTreeSet<usize>> {
        self.0.get(key)
    }

    #[allow(dead_code)]
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[allow(dead_code)]
    pub fn keys(&self) -> std::collections::btree_map::Keys<'_, String, BTreeSet<usize>> {
        self.0.keys()
    }
}

impl Default for ArcGroups {
    fn default() -> Self {
        Self::new()
    }
}

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
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    pub cell_type: ElementType,
    pub connectivity: ConnectivityBase<C>,
    pub fields: BTreeMap<String, nd::ArrayBase<F, nd::IxDyn>>,
    /// Family IDs for each element (implementation detail for group partitioning).
    families: nd::ArrayBase<G, nd::Ix1>,
    /// Groups mapping: group_name -> set of family_ids.
    /// Wrapped in Arc for cheap cloning.
    groups: ArcGroups,
}

pub type ElementBlock =
    ElementBlockBase<nd::OwnedArcRepr<usize>, nd::OwnedArcRepr<f64>, nd::OwnedArcRepr<usize>>;

pub type ElementBlockView<'a> =
    ElementBlockBase<nd::ViewRepr<&'a usize>, nd::ViewRepr<&'a f64>, nd::ViewRepr<&'a usize>>;

impl<C, F, G> ElementBlockBase<C, F, G>
where
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    /// Returns the number of elements in this block.
    pub fn len(&self) -> usize {
        self.connectivity.len()
    }

    /// Returns the connectivity (node indices) for the element at `index`.
    pub fn element_connectivity(&self, index: usize) -> &[usize] {
        &self.connectivity[index]
    }

    pub fn element_type(&self) -> ElementType {
        self.cell_type
    }

    /// Returns an immutable view of the element at `index`.
    pub fn get<'a>(&'a self, index: usize, coords: nd::ArrayView2<'a, f64>) -> Element<'a> {
        Element::new(
            index,
            coords,
            &self.families[index],
            &self.connectivity[index],
            self.cell_type,
            &self.groups,
        )
    }

    /// Returns an iterator over all elements in this block.
    pub fn iter<'a>(
        &'a self,
        coords: nd::ArrayView2<'a, f64>,
    ) -> impl ExactSizeIterator<Item = Element<'a>> + 'a {
        self.connectivity
            .iter()
            .enumerate()
            .map(move |(i, connectivity)| {
                Element::new(
                    i,
                    coords,
                    &self.families[i],
                    connectivity,
                    self.cell_type,
                    &self.groups,
                )
            })
    }

    /// Parallel iterator over elements (serial fallback without `rayon`).
    #[cfg(not(feature = "rayon"))]
    pub fn par_iter<'a>(
        &'a self,
        coords: nd::ArrayView2<'a, f64>,
    ) -> impl Iterator<Item = Element<'a>> + 'a {
        self.iter(coords)
    }

    /// Parallel iterator over elements (requires `rayon` feature).
    #[cfg(feature = "rayon")]
    pub fn par_iter<'a>(
        &'a self,
        coords: nd::ArrayView2<'a, f64>,
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
                Element::new(
                    i,
                    coords,
                    &self.families[i],
                    &self.connectivity[i],
                    self.cell_type,
                    &self.groups,
                )
            })
    }

    /// Returns a reference to the families array.
    pub fn families(&self) -> nd::ArrayView1<'_, usize> {
        self.families.view()
    }

    /// Returns a reference to the groups map.
    pub(crate) fn groups(&self) -> &BTreeMap<String, BTreeSet<usize>> {
        &self.groups.0
    }

    /// Returns a reference to the groups wrapper (ArcGroups).
    pub(crate) fn arc_groups(&self) -> &ArcGroups {
        &self.groups
    }

    /// Set the groups on this block view.
    pub(crate) fn set_groups(&mut self, groups: ArcGroups) {
        self.groups = groups;
    }

    /// Get all element indices belonging to a group (local to this block).
    /// Scans all elements and returns those whose family ID is in the group's set.
    pub fn group_elements_local(&self, group: &str) -> Vec<usize> {
        match self.groups.0.get(group) {
            Some(family_ids) => (0..self.len())
                .filter(|&i| family_ids.contains(&self.families[i]))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get all group names for an element (local to this block).
    /// Uses the family-based indirection.
    pub fn element_groups_local(&self, elem_index: usize) -> Vec<String> {
        let fid = self.families[elem_index];
        self.groups
            .0
            .iter()
            .filter(|(_, family_ids)| family_ids.contains(&fid))
            .map(|(name, _)| name.clone())
            .collect()
    }

    /// Check if an element is in a group (local to this block).
    /// Uses the family-based indirection.
    pub fn in_group_local(&self, elem_index: usize, group: &str) -> bool {
        let fid = self.families[elem_index];
        self.groups
            .0
            .get(group)
            .map(|family_ids| family_ids.contains(&fid))
            .unwrap_or(false)
    }

    /// Check if a group exists in this block.
    pub fn has_group_local(&self, group: &str) -> bool {
        self.groups.0.contains_key(group)
    }

    pub fn view(&'_ self) -> ElementBlockView<'_> {
        ElementBlockView {
            cell_type: self.cell_type,
            connectivity: self.connectivity.view(),
            fields: self
                .fields
                .iter()
                .map(|(n, f)| (n.clone(), f.view()))
                .collect(),
            families: self.families.view(),
            groups: self.groups.clone(),
        }
    }
}

impl ElementBlock {
    /// Returns a clone of the families array (`Arc`-cheap for owned blocks).
    pub(crate) fn families_owned(&self) -> nd::ArcArray1<usize> {
        self.families.clone()
    }

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
        fields: Option<BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>>>,
    ) -> Self {
        let conn_len = connectivity.nrows();
        let families = match families {
            Some(fams) => Some(fams),
            None => Some(nd::ArcArray1::from(vec![0; conn_len])),
        };

        let fields = fields.unwrap_or_default();

        Self {
            cell_type,
            connectivity: Connectivity::Regular(connectivity),
            fields,
            families: families.unwrap(),
            groups: ArcGroups::new(),
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
        families: Option<nd::ArcArray1<usize>>,
        fields: Option<BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>>>,
    ) -> Self {
        let n_elements = offsets.len();
        let families = families.unwrap_or_else(|| nd::ArcArray1::from(vec![0; n_elements]));
        let fields = fields.unwrap_or_default();
        Self {
            cell_type,
            connectivity: Connectivity::new_poly(connectivity, offsets),
            fields,
            families,
            groups: ArcGroups::new(),
        }
    }

    /// Builds a block while preserving the element metadata (families, fields, groups).
    ///
    /// Used by out-of-place mesh tools that rebuild the connectivity but keep the element
    /// order unchanged (e.g. `reorient`), in which case the metadata arrays can be carried
    /// over with a cheap `Arc` clone.
    pub(crate) fn new_with_metadata(
        cell_type: ElementType,
        connectivity: Connectivity,
        families: nd::ArcArray1<usize>,
        fields: BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>>,
        groups: ArcGroups,
    ) -> Self {
        Self {
            cell_type,
            connectivity,
            fields,
            families,
            groups,
        }
    }

    /// Adds a new element to this block.
    ///
    /// The connectivity is appended to the block's connectivity array, and a
    /// new family entry is created. Field support is not yet implemented.
    pub fn add_element(&mut self, connectivity: nd::ArrayView1<usize>, family: Option<usize>) {
        self.connectivity.push(connectivity);
        let family = family.unwrap_or_default();
        let mut new_families = std::mem::take(&mut self.families).into_owned();
        new_families
            .append(nd::Axis(0), nd::array![family].view())
            .unwrap();
        self.families = new_families.into_shared();
    }

    /// Returns a mutable view of the element at `index`.
    pub fn get_mut<'a>(
        &'a mut self,
        index: usize,
        coords: nd::ArrayView2<'a, f64>,
    ) -> ElementMut<'a> {
        ElementMut::new(
            index,
            coords,
            self.families.get_mut(index).unwrap(),
            &mut self.connectivity[index],
            self.cell_type,
            &self.groups,
        )
    }

    /// Snapshot current per-element group memberships as group ids.
    ///
    /// Group ids refer to the keys of the block's group map, in name order.
    fn group_memberships(&self) -> (Vec<String>, Vec<Vec<usize>>) {
        use std::collections::HashMap;

        let group_names: Vec<String> = self.groups.0.keys().cloned().collect();
        let mut family_groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (group_id, name) in group_names.iter().enumerate() {
            if let Some(family_ids) = self.groups.0.get(name) {
                for &family in family_ids {
                    family_groups.entry(family).or_default().push(group_id);
                }
            }
        }

        let memberships = (0..self.len())
            .map(|i| {
                family_groups
                    .get(&self.families[i])
                    .cloned()
                    .unwrap_or_default()
            })
            .collect();
        (group_names, memberships)
    }

    /// Rebuild families and the group family sets from exact per-element group
    /// memberships. Group ids refer to `group_names`.
    fn rebuild_from_memberships(
        &mut self,
        group_names: &[String],
        mut memberships: Vec<Vec<usize>>,
    ) {
        use std::collections::HashMap;

        let n = self.len();
        debug_assert_eq!(memberships.len(), n);

        let mut signature_to_family: HashMap<Vec<usize>, usize> = HashMap::new();
        signature_to_family.insert(Vec::new(), 0);
        let mut next_family = 1usize;
        let mut families = vec![0usize; n];

        for (i, signature) in memberships.iter_mut().enumerate() {
            signature.sort_unstable();
            signature.dedup();
            let family = *signature_to_family
                .entry(signature.clone())
                .or_insert_with(|| {
                    let family = next_family;
                    next_family += 1;
                    family
                });
            families[i] = family;
        }

        let mut groups: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for (i, signature) in memberships.iter().enumerate() {
            for &group_id in signature {
                if let Some(name) = group_names.get(group_id) {
                    groups.entry(name.clone()).or_default().insert(families[i]);
                }
            }
        }

        self.families = nd::Array1::from_vec(families).into_shared();
        self.groups = ArcGroups(Arc::new(groups));
    }

    /// Add elements to a group.
    ///
    /// Rebuilds the block's family partition from the resulting exact
    /// membership. This is linear in the block size plus the number of group
    /// memberships, and never scans the block once per family.
    pub(crate) fn add_to_group_internal(&mut self, group: &str, elem_indices: &[usize]) {
        if elem_indices.is_empty() {
            return;
        }
        let (mut group_names, mut memberships) = self.group_memberships();
        let group_id = match group_names.iter().position(|name| name == group) {
            Some(group_id) => group_id,
            None => {
                group_names.push(group.to_string());
                group_names.len() - 1
            }
        };
        for &i in elem_indices {
            if i < memberships.len() {
                memberships[i].push(group_id);
            }
        }
        self.rebuild_from_memberships(&group_names, memberships);
    }

    /// Remove elements from a group.
    ///
    /// Rebuilds the block's family partition from the resulting exact membership.
    pub(crate) fn remove_from_group_internal(&mut self, group: &str, elem_indices: &[usize]) {
        if elem_indices.is_empty() || !self.groups.0.contains_key(group) {
            return;
        }
        let (group_names, mut memberships) = self.group_memberships();
        let Some(group_id) = group_names.iter().position(|name| name == group) else {
            return;
        };
        for &i in elem_indices {
            if i < memberships.len() {
                memberships[i].retain(|&id| id != group_id);
            }
        }
        self.rebuild_from_memberships(&group_names, memberships);
    }

    /// Delete a group from this block.
    pub(crate) fn delete_group_internal(&mut self, group: &str) {
        if !self.groups.0.contains_key(group) {
            return;
        }
        let (group_names, mut memberships) = self.group_memberships();
        let Some(group_id) = group_names.iter().position(|name| name == group) else {
            return;
        };
        for signature in &mut memberships {
            signature.retain(|&id| id != group_id);
        }
        self.rebuild_from_memberships(&group_names, memberships);
    }

    /// Rename a group, replacing any group that already has the target name.
    pub(crate) fn rename_group_internal(&mut self, old_name: &str, new_name: &str) {
        if old_name == new_name || !self.groups.0.contains_key(old_name) {
            return;
        }
        let (mut group_names, mut memberships) = self.group_memberships();
        let Some(old_id) = group_names.iter().position(|name| name == old_name) else {
            return;
        };
        if let Some(new_id) = group_names.iter().position(|name| name == new_name) {
            for signature in &mut memberships {
                signature.retain(|&id| id != new_id);
            }
        }
        group_names[old_id] = new_name.to_string();
        self.rebuild_from_memberships(&group_names, memberships);
    }

    /// Replace the families array for this block (crate-internal).
    #[cfg(feature = "io")]
    pub(crate) fn set_families(
        &mut self,
        families: nd::ArrayBase<nd::OwnedArcRepr<usize>, nd::Ix1>,
    ) {
        self.families = families;
    }

    /// Replace all groups with exact element-index membership.
    ///
    /// Families are rebuilt from the group signatures in a single pass. This is
    /// safe on a freshly built block where every element still carries family 0,
    /// and does not interpret indices through the current families.
    pub(crate) fn set_groups_internal(&mut self, new_groups: BTreeMap<String, Vec<usize>>) {
        let n = self.len();
        let group_names: Vec<String> = new_groups.keys().cloned().collect();
        let mut memberships: Vec<Vec<usize>> = vec![Vec::new(); n];

        for (group_id, indices) in new_groups.values().enumerate() {
            for &i in indices {
                if i < n {
                    memberships[i].push(group_id);
                }
            }
        }
        self.rebuild_from_memberships(&group_names, memberships);
    }
}

impl<'a> ElementBlockView<'a> {
    /// Create a new regular element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `families` - An array of family indices for each regular element; it must remain
    ///   alive for the lifetime of the returned view.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_regular(
        cell_type: ElementType,
        connectivity: nd::ArrayView2<'a, usize>,
        families: nd::ArrayView1<'a, usize>,
    ) -> Self {
        assert_eq!(
            families.len(),
            connectivity.nrows(),
            "family count must match the number of regular elements"
        );
        Self {
            cell_type,
            connectivity: ConnectivityView::Regular(connectivity),
            fields: BTreeMap::new(),
            families,
            groups: ArcGroups::new(),
        }
    }

    /// Create a new poly element block.
    ///
    /// # Arguments
    /// * `cell_type` - The type of the elements in this block.
    /// * `connectivity` - The connectivity of the elements in this block.
    /// * `offsets` - The offsets of the elements in this block.
    /// * `families` - An array of family indices for each polygonal element; it must remain
    ///   alive for the lifetime of the returned view.
    /// # Returns
    /// A new `ElementBlock` instance.
    pub fn new_poly(
        cell_type: ElementType,
        connectivity: nd::ArrayView1<'a, usize>,
        offsets: nd::ArrayView1<'a, usize>,
        families: nd::ArrayView1<'a, usize>,
    ) -> Self {
        assert_eq!(
            families.len(),
            offsets.len(),
            "family count must match the number of polygonal elements"
        );
        Self {
            cell_type,
            connectivity: ConnectivityView::Poly(IndirectIndex {
                data: connectivity,
                offsets,
            }),
            fields: BTreeMap::new(),
            families,
            groups: ArcGroups::new(),
        }
    }
    pub fn into_entry(self) -> (ElementType, ElementBlockView<'a>) {
        (self.cell_type, self)
    }
}

/// Trait for converting an element block into an (ElementType, block) tuple.
pub trait IntoElementBlockEntry {
    /// Consumes self and returns the element type and block.
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
        let groups = ArcGroups::new();

        let element_block = ElementBlock {
            cell_type: ElementType::TRI3,
            connectivity,
            fields,
            families: families.into(),
            groups,
        };

        assert_eq!(element_block.len(), 3);
        assert_eq!(element_block.fields.len(), 0);
        assert_eq!(element_block.families().len(), 3);
        assert_eq!(element_block.groups().len(), 0);
    }

    #[test]
    fn test_element_block_iter() {
        let connectivity = Connectivity::Regular(array![[0, 1], [1, 2], [2, 3]].to_shared());
        let fields = BTreeMap::new();
        let families = vec![0, 1, 2];
        let groups = ArcGroups::new();

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

    #[test]
    fn test_view_constructors_use_explicit_families() {
        let regular_connectivity = array![[0, 1, 2], [1, 2, 0]];
        let regular_families = array![0usize, 1usize];
        let regular = ElementBlockView::new_regular(
            ElementType::TRI3,
            regular_connectivity.view(),
            regular_families.view(),
        );
        assert_eq!(regular.families().to_vec(), vec![0, 1]);

        let poly_connectivity = array![0, 1, 2, 0, 1];
        let poly_offsets = array![3, 5];
        let poly_families = array![0usize, 1usize];
        let poly = ElementBlockView::new_poly(
            ElementType::QUAD4,
            poly_connectivity.view(),
            poly_offsets.view(),
            poly_families.view(),
        );
        assert_eq!(poly.families().to_vec(), vec![0, 1]);
    }

    #[test]
    #[should_panic(expected = "family count must match")]
    fn test_view_rejects_mismatched_family_count() {
        let connectivity = array![[0, 1, 2], [1, 2, 0]];
        let families = array![0usize];
        let _ =
            ElementBlockView::new_regular(ElementType::TRI3, connectivity.view(), families.view());
    }

    #[test]
    fn test_group_operations() {
        let mut element_block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Add elements 0 and 1 to group "wall"
        element_block.add_to_group_internal("wall", &[0, 1]);
        assert!(element_block.in_group_local(0, "wall"));
        assert!(element_block.in_group_local(1, "wall"));
        assert!(!element_block.in_group_local(2, "wall"));

        // Add elements 2 and 3 to group "interface"
        element_block.add_to_group_internal("interface", &[2, 3]);
        assert!(element_block.in_group_local(2, "interface"));
        assert!(element_block.in_group_local(3, "interface"));

        // Check group elements
        let wall_elements = element_block.group_elements_local("wall");
        assert_eq!(wall_elements.len(), 2);
        assert!(wall_elements.contains(&0));
        assert!(wall_elements.contains(&1));

        let interface_elements = element_block.group_elements_local("interface");
        assert_eq!(interface_elements.len(), 2);
        assert!(interface_elements.contains(&2));
        assert!(interface_elements.contains(&3));

        // Check element groups
        let elem0_groups = element_block.element_groups_local(0);
        assert_eq!(elem0_groups.len(), 1);
        assert!(elem0_groups.contains(&"wall".to_string()));

        let elem2_groups = element_block.element_groups_local(2);
        assert_eq!(elem2_groups.len(), 1);
        assert!(elem2_groups.contains(&"interface".to_string()));

        // Remove element 1 from wall
        element_block.remove_from_group_internal("wall", &[1]);
        assert!(element_block.in_group_local(0, "wall"));
        assert!(!element_block.in_group_local(1, "wall"));
    }

    #[test]
    fn test_group_families_are_partitioned() {
        let mut element_block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Initially all elements have family 0
        assert_eq!(element_block.families()[0], 0);
        assert_eq!(element_block.families()[1], 0);

        // Add elements 0,1 to "wall" and 2,3 to "interface"
        element_block.add_to_group_internal("wall", &[0, 1]);
        element_block.add_to_group_internal("interface", &[2, 3]);

        // After refinement, elements should have different families
        // Elements 0,1 should have same family (both in "wall" only)
        // Elements 2,3 should have same family (both in "interface" only)
        assert_eq!(element_block.families()[0], element_block.families()[1]);
        assert_eq!(element_block.families()[2], element_block.families()[3]);
        assert_ne!(element_block.families()[0], element_block.families()[2]);

        // Element 0 should be in "wall" but not "interface"
        assert!(element_block.in_group_local(0, "wall"));
        assert!(!element_block.in_group_local(0, "interface"));

        // Element 2 should be in "interface" but not "wall"
        assert!(!element_block.in_group_local(2, "wall"));
        assert!(element_block.in_group_local(2, "interface"));
    }

    #[test]
    fn test_intersecting_groups() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Elements 0,1,2 in "fluid", elements 1,2,3 in "boundary"
        // Element 1 and 2 are in both groups
        block.add_to_group_internal("fluid", &[0, 1, 2]);
        block.add_to_group_internal("boundary", &[1, 2, 3]);

        assert!(block.in_group_local(0, "fluid"));
        assert!(!block.in_group_local(0, "boundary"));

        assert!(block.in_group_local(1, "fluid"));
        assert!(block.in_group_local(1, "boundary"));

        assert!(block.in_group_local(2, "fluid"));
        assert!(block.in_group_local(2, "boundary"));

        assert!(!block.in_group_local(3, "fluid"));
        assert!(block.in_group_local(3, "boundary"));

        // Elements 0 and 3 have unique signatures (one group each)
        // Elements 1 and 2 have the same signature (both in fluid+boundary)
        assert_eq!(block.families()[1], block.families()[2]);
        assert_ne!(block.families()[0], block.families()[1]);
        assert_ne!(block.families()[3], block.families()[1]);
    }

    #[test]
    fn test_partial_family_split_add() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // All start in family 0. Add only elements 0,1 to "wall".
        // Elements 2,3 stay out — family must split.
        let family_before = block.families()[2]; // still 0
        block.add_to_group_internal("wall", &[0, 1]);

        assert!(block.in_group_local(0, "wall"));
        assert!(block.in_group_local(1, "wall"));
        assert!(!block.in_group_local(2, "wall"));
        assert!(!block.in_group_local(3, "wall"));

        // Elements 0,1 share a family, 2,3 share another
        assert_eq!(block.families()[0], block.families()[1]);
        assert_eq!(block.families()[2], block.families()[3]);
        assert_ne!(block.families()[0], block.families()[2]);

        // Elements 2,3 kept their original family
        assert_eq!(block.families()[2], family_before);
    }

    #[test]
    fn test_partial_family_split_remove() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Add 0,1,2,3 all to "wall" — no split needed since all have family 0
        block.add_to_group_internal("wall", &[0, 1, 2, 3]);
        assert_eq!(block.families()[0], block.families()[3]);

        // Remove only element 1 — family must split
        block.remove_from_group_internal("wall", &[1]);

        assert!(block.in_group_local(0, "wall"));
        assert!(!block.in_group_local(1, "wall"));
        assert!(block.in_group_local(2, "wall"));
        assert!(block.in_group_local(3, "wall"));

        // Elements 0,2,3 share a family, element 1 has its own
        assert_eq!(block.families()[0], block.families()[2]);
        assert_eq!(block.families()[0], block.families()[3]);
        assert_ne!(block.families()[0], block.families()[1]);
    }

    #[test]
    fn test_set_groups_replaces_all() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        block.add_to_group_internal("wall", &[0, 1]);
        assert!(block.in_group_local(0, "wall"));

        // set_groups replaces everything
        let mut new_groups = BTreeMap::new();
        new_groups.insert("outlet".to_string(), vec![2, 3]);
        block.set_groups_internal(new_groups);

        assert!(!block.in_group_local(0, "wall"));
        assert!(block.in_group_local(2, "outlet"));
        assert!(block.in_group_local(3, "outlet"));
        assert!(!block.has_group_local("wall"));
    }

    #[test]
    fn test_set_groups_preserves_overlaps() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]].to_shared(),
            None,
            None,
        );

        let mut groups = BTreeMap::new();
        groups.insert("a".to_string(), vec![0, 1, 2]);
        groups.insert("b".to_string(), vec![1, 2, 3]);
        block.set_groups_internal(groups);

        assert!(block.in_group_local(0, "a"));
        assert!(!block.in_group_local(0, "b"));
        assert!(block.in_group_local(1, "a"));
        assert!(block.in_group_local(1, "b"));
        assert!(block.in_group_local(2, "a"));
        assert!(block.in_group_local(2, "b"));
        assert!(!block.in_group_local(3, "a"));
        assert!(block.in_group_local(3, "b"));
        assert!(!block.in_group_local(4, "a"));
        assert!(!block.in_group_local(4, "b"));

        // Elements outside every group keep the reserved family 0.
        assert_eq!(block.families()[4], 0);
        // Elements with different signatures must have different families.
        assert_ne!(block.families()[0], block.families()[1]);
        assert_ne!(block.families()[1], block.families()[3]);
    }

    #[test]
    fn test_delete_and_rename_groups() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        let mut groups = BTreeMap::new();
        groups.insert("a".to_string(), vec![0, 1]);
        groups.insert("b".to_string(), vec![1, 2, 3]);
        block.set_groups_internal(groups);

        block.delete_group_internal("a");
        assert!(!block.has_group_local("a"));
        assert_eq!(block.group_elements_local("b"), vec![1, 2, 3]);

        block.rename_group_internal("b", "c");
        assert!(!block.has_group_local("b"));
        assert_eq!(block.group_elements_local("c"), vec![1, 2, 3]);

        // Renaming onto an existing group replaces its membership.
        let mut groups = BTreeMap::new();
        groups.insert("x".to_string(), vec![0]);
        groups.insert("y".to_string(), vec![0, 1]);
        block.set_groups_internal(groups);
        block.rename_group_internal("x", "y");
        assert!(!block.has_group_local("x"));
        assert_eq!(block.group_elements_local("y"), vec![0]);
    }

    #[test]
    fn test_set_groups_compacts_families() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        let mut groups = BTreeMap::new();
        groups.insert("wall".to_string(), vec![0, 1]);
        groups.insert("interface".to_string(), vec![2, 3]);
        block.set_groups_internal(groups);

        let fam_after = block.families().to_vec();

        // Two distinct families, one for wall, one for interface.
        assert_ne!(fam_after[0], fam_after[2]);
        assert_eq!(fam_after[0], fam_after[1]);
        assert_eq!(fam_after[2], fam_after[3]);

        // Group membership preserved.
        assert!(block.in_group_local(0, "wall"));
        assert!(!block.in_group_local(0, "interface"));
        assert!(!block.in_group_local(2, "wall"));
        assert!(block.in_group_local(2, "interface"));
    }

    #[test]
    fn test_group_signature_consistency() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Complex scenario: three groups with overlaps
        block.add_to_group_internal("A", &[0, 1, 2]);
        block.add_to_group_internal("B", &[1, 2, 3]);
        block.add_to_group_internal("C", &[0, 2]);

        // Expected signatures:
        // elem 0: {A, C}
        // elem 1: {A, B}
        // elem 2: {A, B, C}
        // elem 3: {B}
        let sig0 = block.element_groups_local(0);
        let sig1 = block.element_groups_local(1);
        let sig2 = block.element_groups_local(2);
        let sig3 = block.element_groups_local(3);

        assert_eq!(sig0, vec!["A", "C"]);
        assert_eq!(sig1, vec!["A", "B"]);
        assert_eq!(sig2, vec!["A", "B", "C"]);
        assert_eq!(sig3, vec!["B"]);

        // All four signatures are unique, so four distinct families
        let mut families: Vec<usize> = (0..4).map(|i| block.families()[i]).collect();
        families.sort();
        families.dedup();
        assert_eq!(families.len(), 4);
    }

    #[test]
    fn test_remove_group_and_re_add() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        block.add_to_group_internal("wall", &[0, 1]);
        assert!(block.in_group_local(0, "wall"));

        // Remove all elements from wall — group should disappear
        block.remove_from_group_internal("wall", &[0, 1]);
        assert!(!block.has_group_local("wall"));

        // Re-add to wall
        block.add_to_group_internal("wall", &[0, 1]);
        assert!(block.in_group_local(0, "wall"));
        assert!(block.in_group_local(1, "wall"));
        assert!(!block.in_group_local(2, "wall"));
    }

    #[test]
    fn test_single_element_groups() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // Each element in its own group
        block.add_to_group_internal("g0", &[0]);
        block.add_to_group_internal("g1", &[1]);
        block.add_to_group_internal("g2", &[2]);
        block.add_to_group_internal("g3", &[3]);

        // All four have unique signatures, four distinct families
        assert_ne!(block.families()[0], block.families()[1]);
        assert_ne!(block.families()[1], block.families()[2]);
        assert_ne!(block.families()[2], block.families()[3]);
    }

    #[test]
    fn test_no_groups_all_share_family() {
        let block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        // No groups — all elements have the same empty signature → same family
        assert_eq!(block.families()[0], block.families()[1]);
        assert_eq!(block.families()[1], block.families()[2]);
        assert_eq!(block.families()[2], block.families()[3]);
    }

    #[test]
    fn test_element_block_view_inherits_groups() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        block.add_to_group_internal("wall", &[0, 1]);

        let view = block.view();
        assert!(view.in_group_local(0, "wall"));
        assert!(!view.in_group_local(2, "wall"));

        let wall_elements = view.group_elements_local("wall");
        assert_eq!(wall_elements.len(), 2);
        assert!(wall_elements.contains(&0));
        assert!(wall_elements.contains(&1));
    }
}
