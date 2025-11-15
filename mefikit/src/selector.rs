#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::geometry::ElementGeo;
use super::geometry::is_in as geo;
use super::umesh::{ElementIds, ElementType, UMesh};

pub struct Selector<'a, State = ElementSelector> {
    umesh: &'a UMesh,
    index: ElementIds,
    state: State,
}

pub struct FieldBasedSelector {
    pub field_name: String,
}

pub struct ElementSelector;

pub struct NodeBasedSelector {
    pub all_nodes: bool,
}

pub struct GroupBasedSelector {
    pub families: HashMap<ElementType, BTreeSet<usize>>,
}

pub struct CentroidBasedSelector;

impl<'a, State> Selector<'a, State> {
    pub fn index(&self) -> &ElementIds {
        &self.index
    }

    pub fn select(&self) -> UMesh {
        self.umesh.extract(&self.index)
    }

    fn into_groups(self) -> Selector<'a, GroupBasedSelector> {
        let state = GroupBasedSelector {
            families: HashMap::new(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn into_field(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        let state = FieldBasedSelector {
            field_name: name.to_owned(),
        };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn into_elements(self) -> Selector<'a, ElementSelector> {
        let state = ElementSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn into_centroids(self) -> Selector<'a, CentroidBasedSelector> {
        let state = CentroidBasedSelector {};
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }

    fn into_nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        let state = NodeBasedSelector { all_nodes: all };
        Selector {
            umesh: self.umesh,
            index: self.index,
            state,
        }
    }
}

impl<'a> Selector<'a, ElementSelector> {
    pub fn new(umesh: &'a UMesh) -> Self {
        let index: BTreeMap<ElementType, Vec<usize>> = umesh
            .element_blocks
            .iter()
            .map(|(k, v)| (*k, (0..v.len()).collect()))
            .collect();
        let state = ElementSelector {};
        Self {
            umesh,
            index: index.into(),
            state,
        }
    }

    pub fn of_types(self, ets: &[ElementType]) -> Self {
        let index: ElementIds = self
            .index
            .into_iter()
            .filter(|&e| ets.iter().any(|&et| e.element_type() == et))
            .collect();
        Self {
            umesh: self.umesh,
            index,
            state: self.state,
        }
    }

    pub fn in_index(self, ids: &ElementIds) -> Self {
        let index: ElementIds = self
            .index
            .into_par_iter()
            .filter(|&e_id| ids.contains(e_id))
            .collect();
        Self {
            umesh: self.umesh,
            index,
            state: self.state,
        }
    }

    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.into_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.into_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.into_centroids()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.into_field(name)
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
        self.into_groups()
    }
    pub fn elements(self) -> Selector<'a, ElementSelector> {
        self.into_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.into_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.into_centroids()
    }
}

impl<'a> Selector<'a, GroupBasedSelector> {
    pub fn inside(self, name: &str) -> Self {
        #[cfg(feature = "rayon")]
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks
            .par_iter()
            .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks
            .iter()
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
        #[cfg(feature = "rayon")]
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks
            .par_iter()
            .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
            .collect();
        #[cfg(not(feature = "rayon"))]
        let grp_fmies: HashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .element_blocks
            .iter()
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
    fn collect(self) -> Selector<'a, ElementSelector> {
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
        self.collect().into_field(name)
    }
    pub fn elements(self) -> Selector<'a, ElementSelector> {
        self.collect().into_elements()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.collect().into_nodes(all)
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.collect().into_centroids()
    }
}

impl<'a> Selector<'a, NodeBasedSelector> {
    pub fn all_in<F0>(self, f: F0) -> Selector<'a, NodeBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| self.umesh.get_element(e_id).coords().all(&f))
            .collect();

        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    pub fn any_in<F0>(self, f: F0) -> Selector<'a, NodeBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| self.umesh.get_element(e_id).coords().any(&f))
            .collect();

        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    pub fn all_in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
        self.all_in(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_sphere(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                r,
            )
        })
    }

    pub fn all_in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
        self.all_in(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_aa_bbox(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                p1,
            )
        })
    }

    pub fn all_in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
        self.all_in(|x| {
            debug_assert_eq!(x.len(), 2);
            geo::in_aa_rectangle(
                x.try_into().expect("Coords should have 2 components."),
                p0,
                p1,
            )
        })
    }

    pub fn any_in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
        self.any_in(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_sphere(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                r,
            )
        })
    }

    pub fn any_in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
        self.any_in(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_aa_bbox(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                p1,
            )
        })
    }

    pub fn any_in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
        self.any_in(|x| {
            debug_assert_eq!(x.len(), 2);
            geo::in_aa_rectangle(
                x.try_into().expect("Coords should have 2 components."),
                p0,
                p1,
            )
        })
    }

    pub fn elements(self) -> Selector<'a, ElementSelector> {
        self.into_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.into_field(name)
    }
    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.into_groups()
    }
    pub fn centroids(self) -> Selector<'a, CentroidBasedSelector> {
        self.into_centroids()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.into_nodes(all)
    }
}

impl<'a> Selector<'a, CentroidBasedSelector> {
    pub fn is_in<F0>(self, f: F0) -> Selector<'a, CentroidBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| match self.umesh.space_dimension() {
                2 => f(self.umesh.get_element(e_id).centroid2().as_slice()),
                3 => f(self.umesh.get_element(e_id).centroid3().as_slice()),
                _ => todo!(),
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

    pub fn elements(self) -> Selector<'a, ElementSelector> {
        self.into_elements()
    }
    pub fn fields(self, name: &str) -> Selector<'a, FieldBasedSelector> {
        self.into_field(name)
    }
    pub fn groups(self) -> Selector<'a, GroupBasedSelector> {
        self.into_groups()
    }
    pub fn nodes(self, all: bool) -> Selector<'a, NodeBasedSelector> {
        self.into_nodes(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ElementType;
    use crate::mesh_examples as me;

    #[test]
    fn test_umesh_element_selection() {
        let mesh = me::make_mesh_2d_quad();
        let selected_ids = Selector::new(&mesh)
            .centroids()
            .in_rectangle(&[0.0, 0.0], &[1.0, 1.0])
            .index;
        assert_eq!(selected_ids.len(), 1);
        assert_eq!(selected_ids.get(&ElementType::QUAD4).unwrap(), &vec![0]);
    }
}
