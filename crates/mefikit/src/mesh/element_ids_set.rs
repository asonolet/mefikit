//! A set-based collection of element identifiers for efficient set operations.
//!
//! [`ElementIdsSet`] uses hash sets for each element type, enabling fast
//! union, intersection, difference, and membership operations.

use itertools::Itertools;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;

use super::element::{ElementId, ElementType};
use super::element_ids::ElementIds;

/// A set of element identifiers organized by element type.
///
/// Unlike [`ElementIds`], this uses [`FxHashSet`] for indices, making it
/// suitable for set operations like union, intersection, and difference.
#[derive(Debug, Clone)]
pub struct ElementIdsSet(pub BTreeMap<ElementType, FxHashSet<usize>>);

impl From<ElementIdsSet> for ElementIds {
    fn from(ids_set: ElementIdsSet) -> Self {
        let mut ids = ElementIds::new();
        for (et, indices_set) in ids_set.0 {
            let indices: Vec<usize> = indices_set.into_iter().sorted_unstable().collect();
            ids.0.insert(et, indices);
        }
        ids
    }
}

impl FromIterator<ElementId> for ElementIdsSet {
    fn from_iter<T: IntoIterator<Item = ElementId>>(iter: T) -> Self {
        let mut ids_set = ElementIdsSet::new();
        for element_id in iter {
            let entry = ids_set.0.entry(element_id.element_type()).or_default();
            entry.insert(element_id.index());
        }
        ids_set
    }
}

impl Default for ElementIdsSet {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ElementIds> for ElementIdsSet {
    fn from(ids: ElementIds) -> Self {
        let mut ids_set = ElementIdsSet::new();
        for (et, indices) in ids.0 {
            let indices_set: FxHashSet<usize> = indices.into_iter().collect();
            ids_set.0.insert(et, indices_set);
        }
        ids_set
    }
}

impl ElementIdsSet {
    /// Creates a new empty element ID set.
    pub fn new() -> Self {
        ElementIdsSet(BTreeMap::new())
    }

    /// Returns an iterator over all element types in the set.
    pub fn keys(&self) -> impl Iterator<Item = ElementType> + '_ {
        self.0.keys().copied()
    }

    /// Adds all element IDs from another set to this one.
    pub fn union(&mut self, other: &Self) {
        for (et, indices_set) in &other.0 {
            let entry = self.0.entry(*et).or_default();
            for index in indices_set {
                entry.insert(*index);
            }
        }
    }

    /// Retains only element IDs that are also present in another set.
    pub fn intersection(&mut self, other: &Self) {
        self.0.retain(|et, indices_set| {
            if let Some(other_indices_set) = other.0.get(et) {
                indices_set.retain(|index| other_indices_set.contains(index));
                !indices_set.is_empty()
            } else {
                false
            }
        });
    }

    /// Removes all element IDs that are present in another set.
    pub fn difference(&mut self, other: &Self) {
        for (et, other_indices_set) in &other.0 {
            if let Some(indices_set) = self.0.remove(et) {
                let diff: FxHashSet<usize> =
                    indices_set.difference(other_indices_set).cloned().collect();
                if !diff.is_empty() {
                    self.0.insert(*et, diff);
                }
            }
        }
    }

    /// Keeps element IDs that are in either set but not in both.
    pub fn symmetric_difference(&mut self, other: &Self) {
        for (et, other_indices_set) in &other.0 {
            if let Some(indices_set) = self.0.remove(et) {
                let diff: FxHashSet<usize> = indices_set
                    .symmetric_difference(other_indices_set)
                    .cloned()
                    .collect();
                if !diff.is_empty() {
                    self.0.insert(*et, diff);
                }
            }
        }
    }

    /// Consumes the set and returns an iterator over all element IDs.
    #[allow(clippy::should_implement_trait)]
    pub fn into_iter(self) -> impl Iterator<Item = ElementId> {
        self.0.into_iter().flat_map(|(et, indices_set)| {
            indices_set
                .into_iter()
                .map(move |index| ElementId::new(et, index))
        })
    }

    /// Returns an iterator over all element IDs without consuming the set.
    pub fn iter(&self) -> impl Iterator<Item = ElementId> {
        self.0.iter().flat_map(|(et, indices_set)| {
            indices_set
                .iter()
                .map(move |index| ElementId::new(*et, *index))
        })
    }

    /// Returns `true` if the set contains the given element ID.
    pub fn contains(&self, element_id: ElementId) -> bool {
        if let Some(indices_set) = self.0.get(&element_id.element_type()) {
            indices_set.contains(&element_id.index())
        } else {
            false
        }
    }

    /// Returns `true` if the set contains any elements of the given type.
    pub fn contains_type(&self, element_type: ElementType) -> bool {
        self.0.contains_key(&element_type)
    }

    /// Adds an element ID to the set.
    pub fn add(&mut self, element_id: ElementId) {
        let entry = self.0.entry(element_id.element_type()).or_default();
        entry.insert(element_id.index());
    }

    /// Ensures the set has an entry for the given element type.
    pub fn add_type(&mut self, element_type: ElementType) {
        self.0.entry(element_type).or_default();
    }

    /// Removes an element ID from the set. Returns `true` if the ID was present.
    pub fn remove(&mut self, element_id: ElementId) -> bool {
        if let Some(indices_set) = self.0.get_mut(&element_id.element_type()) {
            indices_set.remove(&element_id.index())
        } else {
            false
        }
    }

    /// Removes all element IDs of the given type from the set.
    pub fn remove_type(&mut self, element_type: ElementType) {
        self.0.remove(&element_type);
    }

    /// Returns the total number of element IDs in the set.
    pub fn len(&self) -> usize {
        self.0.values().map(|indices_set| indices_set.len()).sum()
    }

    /// Returns `true` if the set contains no element IDs.
    pub fn is_empty(&self) -> bool {
        self.0.values().all(|indices_set| indices_set.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::ElementType;

    #[test]
    fn test_union() {
        let mut set1 = ElementIdsSet::new();
        set1.0.entry(ElementType::TRI3).or_default().insert(1);
        set1.0.entry(ElementType::TRI3).or_default().insert(2);

        let mut set2 = ElementIdsSet::new();
        set2.0.entry(ElementType::TRI3).or_default().insert(2);
        set2.0.entry(ElementType::TRI3).or_default().insert(3);
        set2.0.entry(ElementType::QUAD4).or_default().insert(4);

        set1.union(&set2);

        assert_eq!(set1.0.get(&ElementType::TRI3).unwrap().len(), 3);
        assert_eq!(set1.0.get(&ElementType::QUAD4).unwrap().len(), 1);
    }

    #[test]
    fn test_intersection() {
        let mut set1 = ElementIdsSet::new();
        set1.0.entry(ElementType::TRI3).or_default().insert(1);
        set1.0.entry(ElementType::TRI3).or_default().insert(2);

        let mut set2 = ElementIdsSet::new();
        set2.0.entry(ElementType::TRI3).or_default().insert(2);
        set2.0.entry(ElementType::TRI3).or_default().insert(3);
        set2.0.entry(ElementType::QUAD4).or_default().insert(4);

        set1.intersection(&set2);

        assert_eq!(set1.0.get(&ElementType::TRI3).unwrap().len(), 1);
        assert!(!set1.0.contains_key(&ElementType::QUAD4));
    }
}
