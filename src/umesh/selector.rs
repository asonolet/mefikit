use rayon::prelude::*;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use todo;

use super::UMeshView;
use super::element::{ElementIds, ElementLike, ElementType};
use super::geometry as geo;

/// Here umesh should be replace with UMeshView, so that it can interact with non owned umesh
/// struct.

pub struct Selector<'a, State = ElementTypeSelector> {
    umesh: UMeshView<'a>,
    pub index: ElementIds,
    state: State,
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
    fn to_groups(self) -> Selector<'a, GroupBasedSelector> {
        let state = GroupBasedSelector {
            families: HashMap::new(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_field(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_elements(self) -> Selector<'a, ElementTypeSelector> {
        let state = ElementTypeSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_centroids(self) -> Selector<'a, CentroidBasedSelector> {
        let state = CentroidBasedSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn to_nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }
}

impl<'a> Selector<'a, ElementTypeSelector> {
    pub fn new(umesh: UMeshView<'a>) -> Self {
        let index: BTreeMap<ElementType, Vec<usize>> = umesh
            .element_blocks
            .iter()
            .map(|(k, v)| (*k, (0..v.len()).collect()))
            .collect();
        let state = ElementTypeSelector {};
        Self {
            umesh,
            index: index.into(),
            state,
        }
    }

    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.to_centroids()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.to_field(name)
    }
}

impl<'a> Selector<'a, FieldBasedSelector> {
    pub fn ge(self, val: f64) -> Self {
        let index: ElementIds = self
            .index
            .into_iter()
            .filter(|&e_id| {
                self.umesh
                    .element_blocks
                    .get(&e_id.element_type())
                    .unwrap()
                    .fields
                    .get(self.state.field_name.as_str())
                    .unwrap()[[e_id.index()]]
                    >= val
            })
            .collect();
        Self {
            umesh: self.umesh,
            index,
            state: self.state,
        }
    }

    pub fn lt(self, val: f64) -> Self {
        let index: ElementIds = self
            .index
            .into_iter()
            .filter(|&e_id| {
                self.umesh
                    .element_blocks
                    .get(&e_id.element_type())
                    .unwrap()
                    .fields
                    .get(self.state.field_name.as_str())
                    .unwrap()[[e_id.index()]]
                    < val
            })
            .collect();
        Self {
            umesh: self.umesh,
            index,
            state: self.state,
        }
    }

    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.to_centroids()
    }
}

impl<'a> Selector<'a, GroupBasedSelector> {
    pub fn inside(self, name: &str) -> Self {
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks
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
            .element_blocks
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
    fn collect(self) -> Selector<'a, ElementTypeSelector> {
        todo!();
        // let index = self.umesh.families(et);
        // let state = ElementTypeSelector{};
        // Selector {
        //     umesh: self.umesh,
        //     index,
        //     state,
        // }
    }

    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.collect().to_field(name)
    }
    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {
        self.collect().to_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.collect().to_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.collect().to_centroids()
    }
}

impl<'a> Selector<'a, NodeBasedSelector> {
    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.to_centroids()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.to_nodes(all)
    }
}

impl<'a> Selector<'a, CentroidBasedSelector> {
    pub fn is_in<F0>(self, f: F0) -> Selector<'a, CentroidBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_iter()
            .filter(|&e_id| f(self.umesh.get_element(e_id).centroid().as_slice().unwrap()))
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
            debug_assert_eq!(x.len(), 3);
            geo::in_sphere(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                r,
            )
        })
    }

    pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
        self.is_in(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_aa_bbox(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                p1,
            )
        })
    }

    pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
        self.is_in(|x| {
            debug_assert_eq!(x.len(), 2);
            geo::in_aa_rectangle(
                x.try_into().expect("Coords should have 2 components."),
                p0,
                p1,
            )
        })
    }

    pub fn elements(self) -> Selector<'a, ElementTypeSelector> {
        self.to_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.to_field(name)
    }
    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.to_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.to_nodes(all)
    }
}
