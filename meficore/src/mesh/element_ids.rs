#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

use crate::prelude::ElementId;
use crate::prelude::ElementType;

#[derive(Debug, Clone)]
pub struct ElementIds(BTreeMap<ElementType, Vec<usize>>);

impl Default for ElementIds {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementIds {
    pub fn new() -> Self {
        ElementIds(BTreeMap::new())
    }

    pub fn add(&mut self, element_type: ElementType, index: usize) {
        self.0.entry(element_type).or_default().push(index);
    }

    pub fn remove(&mut self, element_type: ElementType, index: usize) -> Option<usize> {
        if let Some(indices) = self.0.get_mut(&element_type)
            && let Some(pos) = indices.iter().position(|&i| i == index)
        {
            return Some(indices.remove(pos));
        }
        None
    }

    pub fn get(&self, element_type: &ElementType) -> Option<&Vec<usize>> {
        self.0.get(element_type)
    }

    pub fn contains_type(&self, element_type: ElementType) -> bool {
        self.0.contains_key(&element_type)
    }
    pub fn contains(&self, element_id: ElementId) -> bool {
        if let Some(indices) = self.0.get(&element_id.element_type()) {
            indices.contains(&element_id.index())
        } else {
            false
        }
    }
    pub fn iter_blocks(&self) -> impl Iterator<Item = (&ElementType, &Vec<usize>)> {
        self.0.iter()
    }
    pub fn iter(&self) -> impl Iterator<Item = ElementId> {
        self.0
            .iter()
            .flat_map(|(et, indices)| indices.iter().map(|index| ElementId::new(*et, *index)))
    }

    #[cfg(feature = "rayon")]
    pub fn into_par_iter(self) -> impl ParallelIterator<Item = ElementId> {
        self.0.into_par_iter().flat_map(|(et, indices)| {
            indices
                .into_par_iter()
                .map(move |index| ElementId::new(et, index))
        })
    }

    pub fn into_iter(self) -> impl Iterator<Item = ElementId> {
        self.0.into_iter().flat_map(|(et, indices)| {
            indices
                .into_iter()
                .map(move |index| ElementId::new(et, index))
        })
    }
    #[cfg(not(feature = "rayon"))]
    pub fn into_par_iter(self) -> impl Iterator<Item = ElementId> {
        self.into_iter()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }
    pub fn element_types(&self) -> Vec<ElementType> {
        self.0.keys().cloned().collect()
    }
}

impl From<BTreeMap<ElementType, Vec<usize>>> for ElementIds {
    fn from(map: BTreeMap<ElementType, Vec<usize>>) -> Self {
        ElementIds(map)
    }
}

impl FromIterator<ElementId> for ElementIds {
    fn from_iter<T: IntoIterator<Item = ElementId>>(iter: T) -> Self {
        let mut ids = ElementIds::new();
        for id in iter {
            ids.add(id.element_type(), id.index());
        }
        ids
    }
}

#[cfg(feature = "rayon")]
impl FromParallelIterator<ElementId> for ElementIds {
    fn from_par_iter<T>(par_iter: T) -> Self
    where
        T: IntoParallelIterator<Item = ElementId>,
    {
        par_iter
            .into_par_iter()
            .fold(ElementIds::new, |mut acc, id| {
                acc.add(id.element_type(), id.index());
                acc
            })
            .reduce(ElementIds::new, |mut acc, other| {
                for (et, indices) in other.0 {
                    for index in indices {
                        acc.add(et, index);
                    }
                }
                acc
            })
    }
}
