#[cfg(feature = "rayon")]
use rayon::prelude::*;

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;
use crate::mesh::{ElementIds, ElementLike, ElementType, UMesh};

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
    pub families: FxHashMap<ElementType, BTreeSet<usize>>,
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
            families: HashMap::default(),
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
            .blocks()
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
                    .block(e_id.element_type())
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
                    .block(e_id.element_type())
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
        let grp_fmies: FxHashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .par_blocks()
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
        let grp_fmies: FxHashMap<ElementType, BTreeSet<usize>> = self
            .umesh
            .par_blocks()
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
    fn all_in<F0>(self, f: F0) -> Selector<'a, NodeBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| self.umesh.element(e_id).coords().all(&f))
            .collect();

        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    fn any_in<F0>(self, f: F0) -> Selector<'a, NodeBasedSelector>
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| self.umesh.element(e_id).coords().any(&f))
            .collect();

        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    pub fn in_shape<F0>(self, f: F0) -> Self
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        if self.state.all_nodes {
            self.all_in(f)
        } else {
            self.any_in(f)
        }
    }

    pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
        self.in_shape(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_sphere(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                r,
            )
        })
    }

    pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
        self.in_shape(|x| {
            debug_assert_eq!(x.len(), 3);
            geo::in_aa_bbox(
                x.try_into().expect("Coords should have 3 components."),
                p0,
                p1,
            )
        })
    }

    pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
        self.in_shape(|x| {
            debug_assert_eq!(x.len(), 2);
            geo::in_aa_rectangle(
                x.try_into().expect("Coords should have 2 components."),
                p0,
                p1,
            )
        })
    }

    fn any_id_in(self, nodes_ids: &[usize]) -> Self {
        let index = if nodes_ids.len() < 50 {
            self.index
                .into_iter()
                .filter(|&e_id| {
                    nodes_ids
                        .iter()
                        .any(|n| self.umesh.element(e_id).connectivity().contains(n))
                })
                .collect()
        } else {
            let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
            nodes_ids.sort_unstable();

            self.index
                .into_iter()
                .filter(|&e_id| {
                    self.umesh
                        .element(e_id)
                        .connectivity()
                        .iter()
                        .any(|n| nodes_ids.binary_search(n).is_ok())
                })
                .collect()
        };
        Selector {
            umesh: self.umesh,
            index,
            state: self.state,
        }
        // let mut node_to_elem: FxHashMap<usize, ElementIds> =
        //     FxHashMap::with_capacity_and_hasher(self.umesh.used_nodes().len(), FxBuildHasher);
        // for e_id in self.index.into_iter() {
        //     for n in self.umesh.element(e_id).connectivity().iter() {
        //         if let Some(elem_ids) = node_to_elem.get_mut(n) {
        //             elem_ids.add(e_id.element_type(), e_id.index());
        //         } else {
        //             node_to_elem.insert(*n, std::iter::once(e_id).collect());
        //         }
        //     }
        // }
        // let index = nodes_ids
        //     .iter()
        //     .flat_map(|n| node_to_elem[n].iter())
        //     .unique()
        //     .collect();
        // let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();
    }

    fn all_id_in(self, nodes_ids: &[usize]) -> Self {
        let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();

        let index = self
            .index
            .into_par_iter()
            .filter(|&e_id| {
                self.umesh
                    .element(e_id)
                    .connectivity()
                    .iter()
                    .all(|n| nodes_ids.contains(n))
            })
            .collect();
        let state = self.state;

        Selector {
            umesh: self.umesh,
            index,
            state,
        }
    }

    pub fn id_in(self, nodes_ids: &[usize]) -> Self {
        let all = self.state.all_nodes;
        if all {
            self.all_id_in(nodes_ids)
        } else {
            self.any_id_in(nodes_ids)
        }
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
                2 => f(self.umesh.element(e_id).centroid2().as_slice()),
                3 => f(self.umesh.element(e_id).centroid3().as_slice()),
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
    use crate::mesh::ElementType;
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
