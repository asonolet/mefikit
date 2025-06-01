use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::umesh::{ElementBlock, ElementId, ElementType, UMesh};

/// Here umesh should be replace with UMeshView, so that it can interact with non owned umesh
/// struct.

pub struct Selector<'a, State = ElementTypeSelector> {
    pub umesh: &'a UMesh,
    pub index: HashMap<ElementType, Vec<usize>>,
    pub state: State,
}

pub struct FieldBasedSelector {
    pub field_name: String,
}

pub struct ElementTypeSelector;

pub struct NodeBasedSelector {
    pub all_nodes: bool,
}

pub struct GroupBasedSelector {
    pub families: HashSet<usize>,
}

pub struct CentroidBasedSelector;

impl<'a> Selector<'a> {
    pub fn new(umesh: &'a UMesh) -> Self {
        let index = umesh
            .element_blocks()
            .iter()
            .map(|(k, v)| (k.clone(), (0..v.len()).collect()))
            .collect();
        let state = ElementTypeSelector {};
        Self {
            umesh,
            index,
            state,
        }
    }
    fn collect(self: Self) -> Self {
        self
    }

    pub fn groups(self: Self) -> Selector<'a, GroupBasedSelector> {
        let collected = self.collect();
        let state = GroupBasedSelector {
            families: HashSet::new(),
        };
        Selector {
            umesh: collected.umesh,
            index: collected.index,
            state,
        }
    }

    pub fn field(self: Self, name: &str) -> Selector<'a, FieldBasedSelector> {
        let collected = self.collect();
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: collected.umesh,
            index: collected.index,
            state,
        }
    }

    pub fn elements(self: Self) -> Selector<'a, ElementTypeSelector> {
        let collected = self.collect();
        let state = ElementTypeSelector {};
        Selector {
            umesh: collected.umesh,
            index: collected.index,
            state,
        }
    }

    pub fn centroids(self: Self) -> Selector<'a, CentroidBasedSelector> {
        let collected = self.collect();
        let state = CentroidBasedSelector {};
        Selector {
            umesh: collected.umesh,
            index: collected.index,
            state,
        }
    }

    pub fn nodes(self: Self, all: bool) -> Selector<'a, NodeBasedSelector> {
        let collected = self.collect();
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: collected.umesh,
            index: collected.index,
            state,
        }
    }
}

impl<'a> Selector<'a, FieldBasedSelector> {
    pub fn ge(self: Self, val: f64) -> Self {
        let index: HashMap<ElementType, Vec<usize>> = self
            .index
            .into_par_iter()
            .map(|(k, v)| {
                (
                    k,
                    v.into_iter()
                        .filter(|&i| {
                            self.umesh
                                .element_block(k)
                                .unwrap()
                                .fields
                                .get(self.state.field_name.as_str())
                                .unwrap()[[i]]
                                >= val
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            umesh: self.umesh,
            index,
            state: self.state,
        }
    }
}
