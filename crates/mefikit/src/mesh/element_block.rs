use derive_where::derive_where;
use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::mesh::IndirectIndexOwned;

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
    pub fn groups(&self) -> &BTreeMap<String, BTreeSet<usize>> {
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

    /// Returns a mutable reference to the groups map.
    pub fn groups_mut(&mut self) -> &mut BTreeMap<String, BTreeSet<usize>> {
        Arc::make_mut(&mut self.groups.0)
    }

    /// Returns the group signature for an element (sorted list of group names).
    /// Uses the family-based indirection: checks which groups contain this element's family ID.
    pub fn element_signature(&self, elem_index: usize) -> Vec<String> {
        let fid = self.families[elem_index];
        self.groups
            .iter()
            .filter(|(_, family_ids)| family_ids.contains(&fid))
            .map(|(name, _)| name.clone())
            .collect()
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
        fields: Option<BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>>>,
    ) -> Self {
        let n_elements = offsets.len();
        let fields = fields.unwrap_or_default();
        Self {
            cell_type,
            connectivity: Connectivity::new_poly(connectivity, offsets),
            fields,
            families: nd::ArcArray1::from(vec![0; n_elements]),
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

    /// Recompute families from scratch based on current groups.
    /// This ensures the invariant: elements with different group signatures have different family IDs.
    ///
    /// Uses a snapshot approach to avoid circular dependencies: first captures element-level
    /// group membership from the current (family-ID-based) groups, then assigns new families,
    /// then rebuilds groups with the new family IDs.
    pub fn recompute_families(&mut self) {
        use std::collections::HashMap;

        let n = self.len();
        if n == 0 {
            return;
        }

        // Step 1: Snapshot element-level group membership using current families + groups.
        // This avoids the circular dependency where element_signature reads groups
        // that reference old family IDs while we're trying to recompute families.
        let mut element_group_snapshot: IndirectIndexOwned<String> = IndirectIndexOwned::default();
        for fid in &self.families {
            let group_names = self
                .groups
                .iter()
                .filter(|(_, fids)| fids.contains(fid))
                .map(|(name, _)| name.clone());
            element_group_snapshot.push_iter(group_names);
        }

        // Step 2: Compute new family IDs from signatures (using the snapshot)
        let mut signature_to_family: HashMap<Vec<String>, usize> = HashMap::new();
        let mut new_families_vec = vec![0; n];

        for (i, sig) in element_group_snapshot.iter().enumerate() {
            let family_id = if let Some(&id) = signature_to_family.get(sig) {
                id
            } else {
                let id = signature_to_family.len();
                signature_to_family.insert(sig.to_vec(), id);
                id
            };
            new_families_vec[i] = family_id;
        }

        // Step 3: Update families
        let mut new_families = std::mem::take(&mut self.families).into_owned();
        for (i, &fid) in new_families_vec.iter().enumerate() {
            new_families[i] = fid;
        }
        self.families = new_families.into_shared();

        // Step 4: Rebuild groups from the snapshot + new family IDs
        let mut new_groups: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
        for (i, group_names) in element_group_snapshot.iter().enumerate() {
            let fid = self.families[i];
            for group_name in group_names {
                new_groups
                    .entry(group_name.clone())
                    .or_default()
                    .insert(fid);
            }
        }
        self.groups = ArcGroups(Arc::new(new_groups));
    }

    /// Add elements to a group with family splitting.
    ///
    /// When adding only some elements of a family, the family is split so the
    /// added elements get a new family ID that is placed in the group. This
    /// preserves the invariant without a full recompute.
    pub fn add_to_group_internal(&mut self, group: &str, elem_indices: &[usize]) {
        if elem_indices.is_empty() {
            return;
        }
        let n = self.len();

        // Check if any family among the elements being added needs splitting
        let mut needs_split = false;
        for &i in elem_indices {
            if i >= n {
                continue;
            }
            let fid = self.families[i];
            let has_non_added =
                (0..n).any(|j| self.families[j] == fid && !elem_indices.contains(&j));
            if has_non_added {
                needs_split = true;
                break;
            }
        }

        if needs_split {
            let max_family = self.families.iter().copied().max().unwrap_or(0);
            let mut next_fid = max_family + 1;
            let mut splits: BTreeMap<usize, usize> = BTreeMap::new();

            for &i in elem_indices {
                if i >= n {
                    continue;
                }
                let fid = self.families[i];
                if splits.contains_key(&fid) {
                    continue;
                }
                let has_non_added =
                    (0..n).any(|j| self.families[j] == fid && !elem_indices.contains(&j));
                if has_non_added {
                    splits.insert(fid, next_fid);
                    next_fid += 1;
                }
            }

            for &i in elem_indices {
                if let Some(&new_fid) = splits.get(&self.families[i]) {
                    self.families[i] = new_fid;
                }
            }

            // Propagate old family's group memberships to new families
            let groups = Arc::make_mut(&mut self.groups.0);
            for (&old_fid, &new_fid) in &splits {
                let group_names: Vec<String> = groups
                    .iter()
                    .filter(|(_, fids)| fids.contains(&old_fid))
                    .map(|(name, _)| name.clone())
                    .collect();
                for name in &group_names {
                    if let Some(fids) = groups.get_mut(name) {
                        fids.insert(new_fid);
                    }
                }
            }

            // Add the new family IDs to the target group
            let fids = groups.entry(group.to_string()).or_default();
            for &i in elem_indices {
                if i < n {
                    fids.insert(self.families[i]);
                }
            }
        } else {
            let groups = Arc::make_mut(&mut self.groups.0);
            let fids = groups.entry(group.to_string()).or_default();
            for &i in elem_indices {
                if i < n {
                    fids.insert(self.families[i]);
                }
            }
        }
    }

    /// Remove elements from a group with family splitting.
    ///
    /// When removing only some elements of a family, the family is split so the
    /// removed elements get a new family ID that is NOT in the group.
    pub fn remove_from_group_internal(&mut self, group: &str, elem_indices: &[usize]) {
        if elem_indices.is_empty() {
            return;
        }
        let n = self.len();

        let mut needs_split = false;
        for &i in elem_indices {
            if i >= n {
                continue;
            }
            let fid = self.families[i];
            let in_group = self
                .groups
                .0
                .get(group)
                .map(|fids| fids.contains(&fid))
                .unwrap_or(false);
            if !in_group {
                continue;
            }
            let has_non_removed =
                (0..n).any(|j| self.families[j] == fid && !elem_indices.contains(&j));
            if has_non_removed {
                needs_split = true;
                break;
            }
        }

        if needs_split {
            let max_family = self.families.iter().copied().max().unwrap_or(0);
            let mut next_fid = max_family + 1;
            let mut splits: BTreeMap<usize, usize> = BTreeMap::new();

            for &i in elem_indices {
                if i >= n {
                    continue;
                }
                let fid = self.families[i];
                let in_group = self
                    .groups
                    .0
                    .get(group)
                    .map(|fids| fids.contains(&fid))
                    .unwrap_or(false);
                if !in_group || splits.contains_key(&fid) {
                    continue;
                }
                let has_non_removed =
                    (0..n).any(|j| self.families[j] == fid && !elem_indices.contains(&j));
                if has_non_removed {
                    splits.insert(fid, next_fid);
                    next_fid += 1;
                }
            }

            for &i in elem_indices {
                if let Some(&new_fid) = splits.get(&self.families[i]) {
                    self.families[i] = new_fid;
                }
            }

            // The old family IDs stay in the group (non-removed elements keep them).
            // The new family IDs (for removed elements) were never added, so they
            // are effectively not in the group. No group mutation needed here.
        } else {
            let groups = Arc::make_mut(&mut self.groups.0);
            if let Some(fids) = groups.get_mut(group) {
                let removed_families: BTreeSet<usize> = elem_indices
                    .iter()
                    .filter(|&&i| i < n)
                    .map(|&i| self.families[i])
                    .collect();
                for fid in removed_families {
                    fids.remove(&fid);
                }
                if fids.is_empty() {
                    groups.remove(group);
                }
            }
        }
    }

    /// Replace the families array for this block (crate-internal).
    #[cfg(feature = "io")]
    pub(crate) fn set_families(
        &mut self,
        families: nd::ArrayBase<nd::OwnedArcRepr<usize>, nd::Ix1>,
    ) {
        self.families = families;
    }

    /// Set groups from a map of group_name -> element_indices.
    /// Converts element indices to family IDs, then calls recompute_families
    /// to establish the correct partition.
    pub fn set_groups_internal(&mut self, new_groups: BTreeMap<String, Vec<usize>>) {
        let mut groups: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();

        for (group_name, elem_indices) in new_groups {
            let family_ids: BTreeSet<usize> = elem_indices
                .into_iter()
                .filter(|&i| i < self.len())
                .map(|i| self.families[i])
                .collect();
            if !family_ids.is_empty() {
                groups.insert(group_name, family_ids);
            }
        }

        self.groups = ArcGroups(Arc::new(groups));
        self.recompute_families();
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
        connectivity: nd::ArrayView2<'a, usize>,
        families: Option<nd::ArrayView1<'a, usize>>,
    ) -> Self {
        let families = match families {
            Some(fams) => Some(fams),
            None => todo!("Implement something meaningful?"),
        };
        Self {
            cell_type,
            connectivity: ConnectivityView::Regular(connectivity),
            fields: BTreeMap::new(),
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
        connectivity: nd::ArrayView1<'a, usize>,
        offsets: nd::ArrayView1<'a, usize>,
    ) -> Self {
        let conn_len = connectivity.len();
        let reg_vec = Box::new(nd::Array1::from(vec![0; conn_len]));
        Self {
            cell_type,
            connectivity: ConnectivityView::Poly(IndirectIndex {
                data: connectivity,
                offsets,
            }),
            fields: BTreeMap::new(),
            families: Box::leak(reg_vec).view(),
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
    fn test_recompute_families() {
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
    fn test_recompute_families_compaction() {
        let mut block = ElementBlock::new_regular(
            ElementType::TRI3,
            array![[0, 1], [1, 2], [2, 3], [3, 0]].to_shared(),
            None,
            None,
        );

        block.add_to_group_internal("wall", &[0, 1]);
        block.add_to_group_internal("interface", &[2, 3]);

        let fam_before = block.families().to_vec();

        // recompute_families should compact but preserve the partition
        block.recompute_families();

        let fam_after = block.families().to_vec();

        // The partition structure should be the same:
        // two distinct families, one for wall, one for interface
        assert_ne!(fam_after[0], fam_after[2]);
        assert_eq!(fam_after[0], fam_after[1]);
        assert_eq!(fam_after[2], fam_after[3]);

        // Group membership preserved
        assert!(block.in_group_local(0, "wall"));
        assert!(!block.in_group_local(0, "interface"));
        assert!(!block.in_group_local(2, "wall"));
        assert!(block.in_group_local(2, "interface"));

        // Family IDs may be renumbered (compacted)
        let max_before = *fam_before.iter().max().unwrap();
        let max_after = *fam_after.iter().max().unwrap();
        assert!(max_after <= max_before);
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
        let sig0 = block.element_signature(0);
        let sig1 = block.element_signature(1);
        let sig2 = block.element_signature(2);
        let sig3 = block.element_signature(3);

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
