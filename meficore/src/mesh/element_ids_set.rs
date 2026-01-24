use itertools::Itertools;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use rustc_hash::FxHashSet;
use std::collections::BTreeMap;

use super::element::{ElementId, ElementType};
use super::element_ids::ElementIds;

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
    pub fn new() -> Self {
        ElementIdsSet(BTreeMap::new())
    }
    pub fn keys(&self) -> impl Iterator<Item = ElementType> + '_ {
        self.0.keys().copied()
    }
    pub fn union(&mut self, other: &Self) {
        for (et, indices_set) in &other.0 {
            let entry = self.0.entry(*et).or_default();
            for index in indices_set {
                entry.insert(*index);
            }
        }
    }
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
    pub fn into_iter(self) -> impl Iterator<Item = ElementId> {
        self.0.into_iter().flat_map(|(et, indices_set)| {
            indices_set
                .into_iter()
                .map(move |index| ElementId::new(et, index))
        })
    }
    pub fn iter(&self) -> impl Iterator<Item = ElementId> {
        self.0.iter().flat_map(|(et, indices_set)| {
            indices_set
                .iter()
                .map(move |index| ElementId::new(*et, *index))
        })
    }
    pub fn contains(&self, element_id: ElementId) -> bool {
        if let Some(indices_set) = self.0.get(&element_id.element_type()) {
            indices_set.contains(&element_id.index())
        } else {
            false
        }
    }
    pub fn contains_type(&self, element_type: ElementType) -> bool {
        self.0.contains_key(&element_type)
    }
    pub fn add(&mut self, element_id: ElementId) {
        let entry = self.0.entry(element_id.element_type()).or_default();
        entry.insert(element_id.index());
    }
    pub fn add_type(&mut self, element_type: ElementType) {
        self.0.entry(element_type).or_default();
    }
    pub fn remove(&mut self, element_id: ElementId) -> bool {
        if let Some(indices_set) = self.0.get_mut(&element_id.element_type()) {
            indices_set.remove(&element_id.index())
        } else {
            false
        }
    }
    pub fn remove_type(&mut self, element_type: ElementType) {
        self.0.remove(&element_type);
    }
    pub fn len(&self) -> usize {
        self.0.values().map(|indices_set| indices_set.len()).sum()
    }
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
        assert!(set1.0.get(&ElementType::QUAD4).is_none());
    }
}
