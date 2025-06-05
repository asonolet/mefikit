use todo;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet};

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
    pub families: HashMap<ElementType, BTreeSet<usize>>,
}

pub struct CentroidBasedSelector;


impl<'a, State> Selector<'a, State> {

    fn to_groups(self: Self) -> Selector<'a, GroupBasedSelector> {
        let state = GroupBasedSelector {
            families: HashMap::new(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_field(self: Self, name: &str) -> Selector<'a, FieldBasedSelector> {
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_elements(self: Self) -> Selector<'a, ElementTypeSelector> {
        let state = ElementTypeSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_centroids(self: Self) -> Selector<'a, CentroidBasedSelector> {
        let state = CentroidBasedSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_nodes(self: Self, all: bool) -> Selector<'a, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }
}

impl<'a> Selector<'a, ElementTypeSelector> {

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

    pub fn lt(self: Self, val: f64) -> Self {
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
                                < val
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

    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {self.to_groups()}
    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {self.to_elements()}
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {self.to_nodes(all)}
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {self.to_centroids()}
}

impl<'a> Selector<'a, GroupBasedSelector> {
    pub fn inside(self: Self, name: &str) -> Self {
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks()
            .par_iter()
            .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
            .collect();
        let intersection_fmies = self
            .state
            .families
            .into_iter()
            .map(|(et, fmies)| {
                let inter = &fmies & grp_fmies.get(&et).unwrap_or(&BTreeSet::new());
                (et, inter)
            })
            .collect();
        let state = GroupBasedSelector {
            families: intersection_fmies,
        };
        Self {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    pub fn outside(self: Self, name: &str) -> Self {
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks()
            .par_iter()
            .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
            .collect();
        let intersection_fmies = self
            .state
            .families
            .into_iter()
            .map(|(et, fmies)| {
                let inter = &fmies & grp_fmies.get(&et).unwrap_or(&BTreeSet::new());
                let exclu = fmies.difference(&inter).cloned().collect();
                (et, exclu)
            })
            .collect();
        let state = GroupBasedSelector {
            families: intersection_fmies,
        };
        Self {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    /// I have a set of families per element_type, I can now select the real elements
    fn collect(self: Self) -> Selector<'a, ElementTypeSelector> {
        todo!();
        // let index = self.umesh.families(et);
        // let state = ElementTypeSelector{};
        // Selector {
        //     umesh: self.umesh,
        //     index,
        //     state,
        // }
    }

    pub fn fields(self: Self, name: &str) -> Selector<'a, FieldBasedSelector> {self.to_field(name)}
    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {self.collect().to_elements()}
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {self.collect().to_nodes(all)}
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {self.collect().to_centroids()}
}

impl<'a> Selector<'a, NodeBasedSelector> {
    pub fn all(self: Self) -> Selector<'a, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: true };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    pub fn elements(self: Self) -> Selector<'a, ElementTypeSelector> {self.to_elements()}
    pub fn fields(self: Self, name: &str) -> Selector<'a, FieldBasedSelector> {self.to_field(name)}
    pub fn groups(self: Self) -> Selector<'a, GroupBasedSelector> {self.to_groups()}
    pub fn centroids(self: Self) -> Selector<'a, CentroidBasedSelector> {self.to_centroids()}
}

impl<'a> Selector<'a, CentroidBasedSelector> {
    pub fn elements(self: Self) -> Selector<'a, ElementTypeSelector> {self.to_elements()}
    pub fn fields(self: Self, name: &str) -> Selector<'a, FieldBasedSelector> {self.to_field(name)}
    pub fn groups(self: Self) -> Selector<'a, GroupBasedSelector> {self.to_groups()}
    pub fn nodes(self: Self, all: bool) -> Selector<'a, NodeBasedSelector> {self.to_nodes(all)}
}
