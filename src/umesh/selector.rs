use ndarray as nd;
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap};
use todo;

use crate::umesh::geometry as geo;
use crate::umesh::umesh_core::UMeshBase;
use crate::umesh::{ElementId, ElementLike, ElementType};

/// Here umesh should be replace with UMeshView, so that it can interact with non owned umesh
/// struct.

pub struct Selector<'a, N, C, F, G, State = ElementTypeSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub umesh: &'a UMeshBase<N, C, F, G>,
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

impl<'a, N, C, F, G, State> Selector<'a, N, C, F, G, State>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    fn to_groups(self) -> Selector<'a, N, C, F, G, GroupBasedSelector> {
        let state = GroupBasedSelector {
            families: HashMap::new(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_field(self, name: &str) -> Selector<'a, N, C, F, G, FieldBasedSelector> {
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_elements(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        let state = ElementTypeSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_centroids(self) -> Selector<'a, N, C, F, G, CentroidBasedSelector> {
        let state = CentroidBasedSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }
}

impl<'a, N, C, F, G> Selector<'a, N, C, F, G, ElementTypeSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn new(umesh: &'a UMeshBase<N, C, F, G>) -> Self {
        let index = umesh
            .element_blocks()
            .iter()
            .map(|(k, v)| (*k, (0..v.len()).collect()))
            .collect();
        let state = ElementTypeSelector {};
        Self {
            umesh,
            index,
            state,
        }
    }

    pub fn groups(self) -> Selector<'a, N, C, F, G, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        self.to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, N, C, F, G, CentroidBasedSelector> {
        self.to_centroids()
    }
    pub fn fields(self, name: &str) -> Selector<'a, N, C, F, G, FieldBasedSelector> {
        self.to_field(name)
    }
}

impl<'a, N, C, F, G> Selector<'a, N, C, F, G, FieldBasedSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn ge(self, val: f64) -> Self {
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

    pub fn lt(self, val: f64) -> Self {
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

    pub fn groups(self) -> Selector<'a, N, C, F, G, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn elements(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        self.to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, N, C, F, G, CentroidBasedSelector> {
        self.to_centroids()
    }
}

impl<'a, N, C, F, G> Selector<'a, N, C, F, G, GroupBasedSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn inside(self, name: &str) -> Self {
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

    pub fn outside(self, name: &str) -> Self {
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
    fn collect(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        todo!();
        // let index = self.umesh.families(et);
        // let state = ElementTypeSelector{};
        // Selector {
        //     umesh: self.umesh,
        //     index,
        //     state,
        // }
    }

    pub fn fields(self, name: &str) -> Selector<'a, N, C, F, G, FieldBasedSelector> {
        self.collect().to_field(name)
    }
    pub fn elements(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        self.collect().to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        self.collect().to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, N, C, F, G, CentroidBasedSelector> {
        self.collect().to_centroids()
    }
}

impl<'a, N, C, F, G> Selector<'a, N, C, F, G, NodeBasedSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn elements(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, N, C, F, G, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self) -> Selector<'a, N, C, F, G, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn centroids(self) -> Selector<'a, N, C, F, G, CentroidBasedSelector> {
        self.to_centroids()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        self.to_nodes(all)
    }
}

impl<'a, N, C, F, G> Selector<'a, N, C, F, G, CentroidBasedSelector>
where
    N: nd::RawData<Elem = f64> + nd::Data + Sync,
    C: nd::RawData<Elem = usize> + nd::Data + Sync,
    F: nd::RawData<Elem = f64> + nd::Data + Sync,
    G: nd::RawData<Elem = usize> + nd::Data + Sync,
{
    pub fn is_in<F0>(self, f: F0) -> Selector<'a, N, C, F, G, CentroidBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .map(|(et, ids)| {
                (
                    et,
                    ids.into_par_iter()
                        .filter(|&i| {
                            f(self
                                .umesh
                                .get_element(ElementId::new(et, i))
                                .centroid()
                                .as_slice()
                                .unwrap())
                        })
                        .collect(),
                )
            })
            .collect();

        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
        self.is_in(|x| {
            geo::in_sphere(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                r,
            )
        })
    }

    pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
        self.is_in(|x| {
            geo::in_aa_bbox(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                p1,
            )
        })
    }

    pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
        self.is_in(|x| {
            geo::in_aa_rectangle(
                x.try_into().expect("Coords should have 2 components."),
                p0,
                p1,
            )
        })
    }

    pub fn elements(self) -> Selector<'a, N, C, F, G, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, N, C, F, G, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self) -> Selector<'a, N, C, F, G, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, N, C, F, G, NodeBasedSelector> {
        self.to_nodes(all)
    }
}
