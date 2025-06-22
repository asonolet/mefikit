use ndarray as nd;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use todo;

use crate::umesh::umesh_core::UMeshBase;
use crate::umesh::ElementType;

/// Here umesh should be replace with UMeshView, so that it can interact with non owned umesh
/// struct.

pub struct Selector<'a, Coo, Conn, Field, Group, State = ElementTypeSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub umesh: &'a UMeshBase<Coo, Conn, Field, Group>,
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

impl<'a, Coo, Conn, Field, Group, State> Selector<'a, Coo, Conn, Field, Group, State>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    fn to_groups(self: Self) -> Selector<'a, Coo, Conn, Field, Group, GroupBasedSelector> {
        let state = GroupBasedSelector {
            families: HashMap::new(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_field(
        self: Self,
        name: &str,
    ) -> Selector<'a, Coo, Conn, Field, Group, FieldBasedSelector> {
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_elements(self: Self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        let state = ElementTypeSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_centroids(self: Self) -> Selector<'a, Coo, Conn, Field, Group, CentroidBasedSelector> {
        let state = CentroidBasedSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_nodes(self: Self, all: bool) -> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }
}

impl<'a, Coo, Conn, Field, Group> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn new(umesh: &'a UMeshBase<Coo, Conn, Field, Group>) -> Self {
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

impl<'a, Coo, Conn, Field, Group> Selector<'a, Coo, Conn, Field, Group, FieldBasedSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
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

    pub fn groups(self) -> Selector<'a, Coo, Conn, Field, Group, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn elements(self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector> {
        self.to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, Coo, Conn, Field, Group, CentroidBasedSelector> {
        self.to_centroids()
    }
}

impl<'a, Coo, Conn, Field, Group> Selector<'a, Coo, Conn, Field, Group, GroupBasedSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
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
    fn collect(self: Self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        todo!();
        // let index = self.umesh.families(et);
        // let state = ElementTypeSelector{};
        // Selector {
        //     umesh: self.umesh,
        //     index,
        //     state,
        // }
    }

    pub fn fields(
        self: Self,
        name: &str,
    ) -> Selector<'a, Coo, Conn, Field, Group, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn elements(self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        self.collect().to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector> {
        self.collect().to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, Coo, Conn, Field, Group, CentroidBasedSelector> {
        self.collect().to_centroids()
    }
}

impl<'a, Coo, Conn, Field, Group> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn all(self: Self) -> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: true };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    pub fn elements(self: Self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(
        self: Self,
        name: &str,
    ) -> Selector<'a, Coo, Conn, Field, Group, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self: Self) -> Selector<'a, Coo, Conn, Field, Group, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn centroids(self: Self) -> Selector<'a, Coo, Conn, Field, Group, CentroidBasedSelector> {
        self.to_centroids()
    }
}

impl<'a, Coo, Conn, Field, Group> Selector<'a, Coo, Conn, Field, Group, CentroidBasedSelector>
where
    Coo: nd::RawData<Elem = f64> + nd::Data + Sync,
    Conn: nd::RawData<Elem = usize> + nd::Data + Sync,
    Field: nd::RawData<Elem = f64> + nd::Data + Sync,
    Group: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn elements(self: Self) -> Selector<'a, Coo, Conn, Field, Group, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(
        self: Self,
        name: &str,
    ) -> Selector<'a, Coo, Conn, Field, Group, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self: Self) -> Selector<'a, Coo, Conn, Field, Group, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn nodes(
        self: Self,
        all: bool,
    ) -> Selector<'a, Coo, Conn, Field, Group, NodeBasedSelector> {
        self.to_nodes(all)
    }
}
