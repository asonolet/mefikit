use itertools::Itertools;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use rustc_hash::{FxHashSet};
use std::collections::{BTreeMap};
use std::ops::{BitAnd, BitOr, Not};
use std::sync::Arc;
use std::thread;

use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;
use crate::mesh::{Dimension, ElementId, ElementIds, ElementLike, ElementType, UMesh, UMeshView};

type ElementIdsSet = BTreeMap<ElementType, FxHashSet<usize>>;

/// This object is the one that will be evaluated by unitary selection_ops.
/// The UMeshView is always passed as the same, whereas the ElementsIds are updated. Each unitary
/// op takes a previous ElementsIds list and returns a new one (shorter).
#[derive(Clone, Debug)]
struct SelectedView<'a>(UMeshView<'a>, ElementIdsSet);

trait Select {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a>;
}

pub enum Selection<const N: usize> {
    ElementSelection(ElementSelection),
    GroupSelection(GroupSelection),
    FieldSelection(FieldSelection),
    CentroidSelection(CentroidSelection<N>),
    NodeSelection(NodeSelection<N>),
    BinarayExpr(BinarayExpr<N>),
    NotExpr(NotExpr<N>),
}

impl<const N: usize> Selection<N> {
    /// The lower, the simpler it is to compute and then should be computed first.
    /// 0: compute right now and blocks
    /// 1: to be computed in parallel
    /// 2: computed the latest
    fn weight(&self) -> u8 {
        match self {
            Self::ElementSelection(_) => 0,
            Self::GroupSelection(_) => 0,
            Self::FieldSelection(_) => 1,
            Self::CentroidSelection(_) => 1,
            Self::NodeSelection(_) => 1,
            Self::NotExpr(_) => 2,
            Self::BinarayExpr(_) => 2,
        }
    }
    fn is_leaf(&self) -> bool {
        match self {
            Self::BinarayExpr(_) => false,
            Self::NotExpr(_) => false,
            _ => true,
        }
    }
    /// Switch operations so that simpler/more selective operations are evaluated sooner
    fn optimize(&self) -> Self {
        todo!()
    }
    fn nodes_in_bbox(all: bool, min: [f64; N], max: [f64; N]) -> Self {
        Self::NodeSelection(NodeSelection::InBBox { all, min, max })
    }
    fn nodes_in_sphere(all: bool, center: [f64; N], r2: f64) -> Self {
        Self::NodeSelection(NodeSelection::InSphere { all, center, r2 })
    }
    fn nodes_in_ids(all: bool, ids: Vec<usize>) -> Self {
        Self::NodeSelection(NodeSelection::InIds { all, ids } )
    }
    fn in_bbox(min: [f64; N], max: [f64; N]) -> Self {
        Self::CentroidSelection(CentroidSelection::InBBox { min, max })
    }
    fn in_sphere(center: [f64; N], r2: f64) -> Self {
        Self::CentroidSelection(CentroidSelection::InSphere { center, r2 })
    }
    fn types(elems: Vec<ElementType>) -> Self {
        Self::ElementSelection(ElementSelection::Types(elems))
    }
    fn dimensions(dims: Vec<Dimension>) -> Self {
        Self::ElementSelection(ElementSelection::Dimensions(dims))
    }
    fn in_ids(eids: ElementIdsSet) -> Self {
        Self::ElementSelection(ElementSelection::InIds(eids))
    }
}

impl<const N: usize> Select for Selection<N> {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            Selection::ElementSelection(elemt_expr) => elemt_expr.select(selection),
            Selection::BinarayExpr(binary) => binary.select(selection),
            _ => todo!(),
        }
    }
}

impl<const N: usize> BitAnd for Selection<N> {
    type Output = Selection<N>;

    fn bitand(self, rhs: Self) -> Self::Output {
        Selection::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl<const N: usize> BitOr for Selection<N> {
    type Output = Selection<N>;

    fn bitor(self, rhs: Self) -> Self::Output {
        Selection::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Or,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl<const N: usize> Not for Selection<N> {
    type Output = Selection<N>;

    fn not(self) -> Self::Output {
        Selection::NotExpr(NotExpr(Arc::new(self)))
    }
}

// Not leaf operations

pub struct BinarayExpr<const N: usize> {
    operator: BooleanOp,
    left: Arc<Selection<N>>,
    right: Arc<Selection<N>>,
}
pub struct NotExpr<const N: usize>(Arc<Selection<N>>);

// Leaf operations

enum NodeSelection<const N: usize> {
    InBBox {
        all: bool,
        min: [f64; N],
        max: [f64; N],
    }, // Axis aligned BBox
    InSphere {
        all: bool,
        center: [f64; N],
        r2: f64,
    }, // center and rayon
    InIds {
        all: bool,
        ids: Vec<usize>,
    },
}

enum CentroidSelection<const N: usize> {
    InBBox { min: [f64; N], max: [f64; N] }, // Axis aligned BBox
    InSphere { center: [f64; N], r2: f64 },  // center and rayon
}

enum ElementSelection {
    Types(Vec<ElementType>),
    InIds(ElementIdsSet),
    Dimensions(Vec<Dimension>),
}

enum GroupSelection {
    IncludeGroups(Vec<String>),
    ExcludeGroups(Vec<String>),
    IncludeFamilies(Vec<usize>),
    ExcludeFamilies(Vec<usize>),
}

enum FieldSelection {
    Gt(Arc<FieldExpr>, f64),
    Geq(Arc<FieldExpr>, f64),
    Eq {
        field: Arc<FieldExpr>,
        val: f64,
        eps: f64,
    },
    Lt(Arc<FieldExpr>, f64),
    Leq(Arc<FieldExpr>, f64),
}

pub enum FieldExpr {
    Scalar(f64),
    Field(String),
    BinarayExpr {
        operator: FieldOp,
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
}

#[derive(Copy, Clone)]
pub enum FieldOp {
    Add,
    Mul,
    Sub,
    Div,
    Pow,
}

#[derive(Copy, Clone)]
pub enum BooleanOp {
    Eq,
    And,
    Or,
    Xor,
}

impl Select for ElementSelection {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            ElementSelection::Types(types) => {
                let SelectedView(view, mut sel) = selection;
                for k in types {
                    sel.remove(k);
                }
                SelectedView(view, sel)
            }
            ElementSelection::Dimensions(dims) => {
                let SelectedView(view, mut sel) = selection;
                let mut key_toremove = Vec::new();
                for k in sel.keys() {
                    if !dims.contains(&k.dimension()) {
                        key_toremove.push(*k);
                    }
                }
                for k in key_toremove {
                    sel.remove(&k);
                }
                SelectedView(view, sel)
            }
            _ => todo!(),
        }
    }
}

impl<const N: usize> NodeSelection<N> {
    // fn all_in<F0>(self, f: F0, selection: SelectedView) -> SelectedView
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //      let SelectedView(mview, index) = selection;
    //     let index = index
    //         .into_par_iter()
    //         .filter(|&e_id| mview.element(e_id).coords().all(&f))
    //         .collect();

    //      SelectedView(mview, index)
    // }

    // fn any_in<F0>(self, f: F0, selection: SelectedView) -> SelectedView
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //     let SelectedView(mview, index) = selection;
    //     let index = index
    //         .into_iter()
    //         .flat_map(|(t, eids)| std::iter::repeat(t).zip(eids))
    //         .filter(|(&t, &eid)| mview.element(ElementId::new(t, eid)).coords().any(&f))
    //         .collect();
    //     SelectedView(mview, index)


    // }

    // pub fn in_shape<F0>(self, f: F0) -> Self
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //     if self.state.all_nodes {
    //         self.all_in(f)
    //     } else {
    //         self.any_in(f)
    //     }
    // }

    // pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 3);
    //         geo::in_sphere(
    //             x.try_into().expect("Coords should have 3 components."),
    //             p0,
    //             r,
    //         )
    //     })
    // }

    // pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 3);
    //         geo::in_aa_bbox(
    //             x.try_into().expect("Coords should have 3 components."),
    //             p0,
    //             p1,
    //         )
    //     })
    // }

    // pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 2);
    //         geo::in_aa_rectangle(
    //             x.try_into().expect("Coords should have 2 components."),
    //             p0,
    //             p1,
    //         )
    //     })
    // }

    // fn any_id_in(self, nodes_ids: &[usize]) -> Self {
    //     let index = if nodes_ids.len() < 50 {
    //         self.index
    //             .into_iter()
    //             .filter(|&e_id| {
    //                 nodes_ids
    //                     .iter()
    //                     .any(|n| self.umesh.element(e_id).connectivity().contains(n))
    //             })
    //             .collect()
    //     } else {
    //         let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
    //         nodes_ids.sort_unstable();

    //         self.index
    //             .into_iter()
    //             .filter(|&e_id| {
    //                 self.umesh
    //                     .element(e_id)
    //                     .connectivity()
    //                     .iter()
    //                     .any(|n| nodes_ids.binary_search(n).is_ok())
    //             })
    //             .collect()
    //     };
    // }

    // fn all_id_in(self, nodes_ids: &[usize]) -> Self {
    //     let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();

    //     let index = self
    //         .index
    //         .into_par_iter()
    //         .filter(|&e_id| {
    //             self.umesh
    //                 .element(e_id)
    //                 .connectivity()
    //                 .iter()
    //                 .all(|n| nodes_ids.contains(n))
    //         })
    //         .collect();
    // }

    // pub fn id_in(self, nodes_ids: &[usize]) -> Self {
    //     let all = self.state.all_nodes;
    //     if all {
    //         self.all_id_in(nodes_ids)
    //     } else {
    //         self.any_id_in(nodes_ids)
    //     }
    // }
}

impl<const N: usize> Select for NodeSelection<N> {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            NodeSelection::InBBox { all, min, max } => selection,
            _ => todo!(),
        }
    }
}

impl<const N: usize> BinarayExpr<N> {
    fn and_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        if self.left.weight() < self.right.weight() {
            let selection = self.left.select(selection);
            self.right.select(selection)
        } else {
            let selection = self.right.select(selection);
            self.left.select(selection)
        }
    }
    fn or_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        let selection1 = self.left.select(selection.clone());
        let _selection2 = self.right.select(selection);
        //TODO: return the union of both
        selection1
    }
}

impl<const N: usize> Select for BinarayExpr<N> {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self.operator {
            BooleanOp::And => self.and_select(selection),
            BooleanOp::Or => self.or_select(selection),
            _ => todo!(),
        }
    }
}

// pub struct Selector<State = ElementSelector> {
//     selection: Box<dyn Fn(UMesh, ElementIds) -> (UMesh, ElementIds)>,
//     state: State,
// }
//
// pub struct FieldBasedSelector {
//     pub field_name: String,
// }
//
// pub struct ElementSelector;
//
// pub struct NodeBasedSelector {
//     pub all_nodes: bool,
// }
//
// pub struct GroupBasedSelector {
//     pub families: FxHashMap<ElementType, BTreeSet<usize>>,
// }
//
// pub struct CentroidBasedSelector;
//
// impl<State> Selector<State> {
//     pub fn index(&self) -> &ElementIds {
//         &self.index
//     }
//
//     pub fn select(&self) -> UMesh {
//         self.umesh.extract(&self.index)
//     }
//
//     fn into_groups(self) -> Selector<GroupBasedSelector> {
//         let state = GroupBasedSelector {
//             families: HashMap::default(),
//         };
//         Selector {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     fn into_field(self, name: &str) -> Selector<FieldBasedSelector> {
//         let state = FieldBasedSelector {
//             field_name: name.to_owned(),
//         };
//         Selector {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     fn into_elements(self) -> Selector<ElementSelector> {
//         let state = ElementSelector {};
//         Selector {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     fn into_centroids(self) -> Selector<CentroidBasedSelector> {
//         let state = CentroidBasedSelector {};
//         Selector {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     fn into_nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         let state = NodeBasedSelector { all_nodes: all };
//         Selector {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
// }
//
// impl<'a> Selector<ElementSelector> {
//     pub fn new(umesh: UMesh) -> Self {
//         let index: BTreeMap<ElementType, Vec<usize>> = umesh
//             .blocks()
//             .map(|(k, v)| (*k, (0..v.len()).collect()))
//             .collect();
//         let state = ElementSelector {};
//         Self {
//             umesh,
//             index: index.into(),
//             state,
//         }
//     }
//
//     pub fn of_types(self, ets: &[ElementType]) -> Self {
//         let index: ElementIds = self
//             .index
//             .into_iter()
//             .filter(|&e| ets.iter().any(|&et| e.element_type() == et))
//             .collect();
//         Self {
//             umesh: self.umesh,
//             index,
//             state: self.state,
//         }
//     }
//
//     pub fn in_index(self, ids: &ElementIds) -> Self {
//         let index: ElementIds = self
//             .index
//             .into_par_iter()
//             .filter(|&e_id| ids.contains(e_id))
//             .collect();
//         Self {
//             umesh: self.umesh,
//             index,
//             state: self.state,
//         }
//     }
//
//     pub fn groups(self) -> Selector<GroupBasedSelector> {
//         self.into_groups()
//     }
//     pub fn nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         self.into_nodes(all)
//     }
//     pub fn centroids(self) -> Selector<CentroidBasedSelector> {
//         self.into_centroids()
//     }
//     pub fn fields(self, name: &str) -> Selector<FieldBasedSelector> {
//         self.into_field(name)
//     }
// }
//
// impl Selector<FieldBasedSelector> {
//     pub fn ge(self, val: f64) -> Self {
//         let index: ElementIds = self
//             .index
//             .into_iter()
//             .filter(|&e_id| {
//                 self.umesh
//                     .block(e_id.element_type())
//                     .unwrap()
//                     .fields
//                     .get(self.state.field_name.as_str())
//                     .unwrap()[[e_id.index()]]
//                     >= val
//             })
//             .collect();
//         Self {
//             umesh: self.umesh,
//             index,
//             state: self.state,
//         }
//     }
//
//     pub fn lt(self, val: f64) -> Self {
//         let index: ElementIds = self
//             .index
//             .into_iter()
//             .filter(|&e_id| {
//                 self.umesh
//                     .block(e_id.element_type())
//                     .unwrap()
//                     .fields
//                     .get(self.state.field_name.as_str())
//                     .unwrap()[[e_id.index()]]
//                     < val
//             })
//             .collect();
//         Self {
//             umesh: self.umesh,
//             index,
//             state: self.state,
//         }
//     }
//
//     pub fn groups(self) -> Selector<GroupBasedSelector> {
//         self.into_groups()
//     }
//     pub fn elements(self) -> Selector<ElementSelector> {
//         self.into_elements()
//     }
//     pub fn nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         self.into_nodes(all)
//     }
//     pub fn centroids(self) -> Selector<CentroidBasedSelector> {
//         self.into_centroids()
//     }
// }
//
// impl Selector<GroupBasedSelector> {
//     pub fn inside(self, name: &str) -> Self {
//         let grp_fmies: FxHashMap<ElementType, BTreeSet<usize>> = self
//             .umesh
//             .par_blocks()
//             .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
//             .collect();
//         let intersection_fmies = self
//             .state
//             .families
//             .into_iter()
//             .map(|(et, fmies)| {
//                 let inter = &fmies & grp_fmies.get(&et).unwrap_or(&BTreeSet::new());
//                 (et, inter)
//             })
//             .collect();
//         let state = GroupBasedSelector {
//             families: intersection_fmies,
//         };
//         Self {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     pub fn outside(self, name: &str) -> Self {
//         let grp_fmies: FxHashMap<ElementType, BTreeSet<usize>> = self
//             .umesh
//             .par_blocks()
//             .map(|(&k, v)| (k, v.groups.get(name).unwrap_or(&BTreeSet::new()).clone()))
//             .collect();
//         let intersection_fmies = self
//             .state
//             .families
//             .into_iter()
//             .map(|(et, fmies)| {
//                 let inter = &fmies & grp_fmies.get(&et).unwrap_or(&BTreeSet::new());
//                 let exclu = fmies.difference(&inter).cloned().collect();
//                 (et, exclu)
//             })
//             .collect();
//         let state = GroupBasedSelector {
//             families: intersection_fmies,
//         };
//         Self {
//             umesh: self.umesh,
//             index: self.index,
//             state,
//         }
//     }
//
//     /// I have a set of families per element_type, I can now select the real elements
//     fn collect(self) -> Selector<ElementSelector> {
//         todo!();
//         // let index = self.umesh.families(et);
//         // let state = ElementTypeSelector{};
//         // Selector {
//         //     umesh: self.umesh,
//         //     index,
//         //     state,
//         // }
//     }
//
//     pub fn fields(self, name: &str) -> Selector<FieldBasedSelector> {
//         self.collect().into_field(name)
//     }
//     pub fn elements(self) -> Selector<ElementSelector> {
//         self.collect().into_elements()
//     }
//     pub fn nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         self.collect().into_nodes(all)
//     }
//     pub fn centroids(self) -> Selector<CentroidBasedSelector> {
//         self.collect().into_centroids()
//     }
// }
//
// impl Selector<NodeBasedSelector> {
//     fn all_in<F0>(self, f: F0) -> Selector<NodeBasedSelector>
//     where
//         F0: Fn(&[f64]) -> bool + Sync,
//     {
//         let index = self
//             .index
//             .into_par_iter()
//             .filter(|&e_id| self.umesh.element(e_id).coords().all(&f))
//             .collect();
//
//         let state = self.state;
//
//         Selector {
//             umesh: self.umesh,
//             index,
//             state,
//         }
//     }
//
//     fn any_in<F0>(self, f: F0) -> Selector<NodeBasedSelector>
//     where
//         F0: Fn(&[f64]) -> bool + Sync,
//     {
//         let index = self
//             .index
//             .into_par_iter()
//             .filter(|&e_id| self.umesh.element(e_id).coords().any(&f))
//             .collect();
//
//         let state = self.state;
//
//         Selector {
//             umesh: self.umesh,
//             index,
//             state,
//         }
//     }
//
//     pub fn in_shape<F0>(self, f: F0) -> Self
//     where
//         F0: Fn(&[f64]) -> bool + Sync,
//     {
//         if self.state.all_nodes {
//             self.all_in(f)
//         } else {
//             self.any_in(f)
//         }
//     }
//
//     pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
//         self.in_shape(|x| {
//             debug_assert_eq!(x.len(), 3);
//             geo::in_sphere(
//                 x.try_into().expect("Coords should have 3 components."),
//                 p0,
//                 r,
//             )
//         })
//     }
//
//     pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
//         self.in_shape(|x| {
//             debug_assert_eq!(x.len(), 3);
//             geo::in_aa_bbox(
//                 x.try_into().expect("Coords should have 3 components."),
//                 p0,
//                 p1,
//             )
//         })
//     }
//
//     pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
//         self.in_shape(|x| {
//             debug_assert_eq!(x.len(), 2);
//             geo::in_aa_rectangle(
//                 x.try_into().expect("Coords should have 2 components."),
//                 p0,
//                 p1,
//             )
//         })
//     }
//
//     fn any_id_in(self, nodes_ids: &[usize]) -> Self {
//         let index = if nodes_ids.len() < 50 {
//             self.index
//                 .into_iter()
//                 .filter(|&e_id| {
//                     nodes_ids
//                         .iter()
//                         .any(|n| self.umesh.element(e_id).connectivity().contains(n))
//                 })
//                 .collect()
//         } else {
//             let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
//             nodes_ids.sort_unstable();
//
//             self.index
//                 .into_iter()
//                 .filter(|&e_id| {
//                     self.umesh
//                         .element(e_id)
//                         .connectivity()
//                         .iter()
//                         .any(|n| nodes_ids.binary_search(n).is_ok())
//                 })
//                 .collect()
//         };
//         Selector {
//             umesh: self.umesh,
//             index,
//             state: self.state,
//         }
//         // let mut node_to_elem: FxHashMap<usize, ElementIds> =
//         //     FxHashMap::with_capacity_and_hasher(self.umesh.used_nodes().len(), FxBuildHasher);
//         // for e_id in self.index.into_iter() {
//         //     for n in self.umesh.element(e_id).connectivity().iter() {
//         //         if let Some(elem_ids) = node_to_elem.get_mut(n) {
//         //             elem_ids.add(e_id.element_type(), e_id.index());
//         //         } else {
//         //             node_to_elem.insert(*n, std::iter::once(e_id).collect());
//         //         }
//         //     }
//         // }
//         // let index = nodes_ids
//         //     .iter()
//         //     .flat_map(|n| node_to_elem[n].iter())
//         //     .unique()
//         //     .collect();
//         // let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();
//     }
//
//     fn all_id_in(self, nodes_ids: &[usize]) -> Self {
//         let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();
//
//         let index = self
//             .index
//             .into_par_iter()
//             .filter(|&e_id| {
//                 self.umesh
//                     .element(e_id)
//                     .connectivity()
//                     .iter()
//                     .all(|n| nodes_ids.contains(n))
//             })
//             .collect();
//         let state = self.state;
//
//         Selector {
//             umesh: self.umesh,
//             index,
//             state,
//         }
//     }
//
//     pub fn id_in(self, nodes_ids: &[usize]) -> Self {
//         let all = self.state.all_nodes;
//         if all {
//             self.all_id_in(nodes_ids)
//         } else {
//             self.any_id_in(nodes_ids)
//         }
//     }
//
//     pub fn elements(self) -> Selector<ElementSelector> {
//         self.into_elements()
//     }
//     pub fn fields(self, name: &str) -> Selector<FieldBasedSelector> {
//         self.into_field(name)
//     }
//     pub fn groups(self) -> Selector<GroupBasedSelector> {
//         self.into_groups()
//     }
//     pub fn centroids(self) -> Selector<CentroidBasedSelector> {
//         self.into_centroids()
//     }
//     pub fn nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         self.into_nodes(all)
//     }
// }
//
// impl Selector<CentroidBasedSelector> {
//     pub fn is_in<F0>(self, f: F0) -> Selector<CentroidBasedSelector>
//     where
//         F0: Fn(&[f64]) -> bool + Sync,
//     {
//         let index = self
//             .index
//             .into_par_iter()
//             .filter(|&e_id| match self.umesh.space_dimension() {
//                 2 => f(self.umesh.element(e_id).centroid2().as_slice()),
//                 3 => f(self.umesh.element(e_id).centroid3().as_slice()),
//                 _ => todo!(),
//             })
//             .collect();
//
//         let state = self.state;
//
//         Selector {
//             umesh: self.umesh,
//             index,
//             state,
//         }
//     }
//
//     pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
//         self.is_in(|x| {
//             debug_assert_eq!(x.len(), 3);
//             geo::in_sphere(
//                 x.try_into().expect("Coords should have 3 components."),
//                 p0,
//                 r,
//             )
//         })
//     }
//
//     pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
//         self.is_in(|x| {
//             debug_assert_eq!(x.len(), 3);
//             geo::in_aa_bbox(
//                 x.try_into().expect("Coords should have 3 components."),
//                 p0,
//                 p1,
//             )
//         })
//     }
//
//     pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
//         self.is_in(|x| {
//             debug_assert_eq!(x.len(), 2);
//             geo::in_aa_rectangle(
//                 x.try_into().expect("Coords should have 2 components."),
//                 p0,
//                 p1,
//             )
//         })
//     }
//
//     pub fn elements(self) -> Selector<ElementSelector> {
//         self.into_elements()
//     }
//     pub fn fields(self, name: &str) -> Selector<FieldBasedSelector> {
//         self.into_field(name)
//     }
//     pub fn groups(self) -> Selector<GroupBasedSelector> {
//         self.into_groups()
//     }
//     pub fn nodes(self, all: bool) -> Selector<NodeBasedSelector> {
//         self.into_nodes(all)
//     }
// }

trait MeshSelect<const N: usize> {
    fn select(&self, expr: Selection<N>) -> (ElementIds, Self);
}

impl<const N: usize> MeshSelect<N> for UMesh {
    fn select(&self, expr: Selection<N>) -> (ElementIds, Self) {
        let index: BTreeMap<ElementType, FxHashSet<usize>> = self
            .blocks()
            .map(|(k, v)| (*k, (0..v.len()).collect()))
            .collect();
        let SelectedView(_, index) = expr.select(SelectedView(self.view(), index));
        let index: BTreeMap<ElementType, Vec<usize>> = index
            .into_iter()
            .map(|(k, v)| (k, v.into_iter().sorted_unstable().collect()))
            .collect();
        let eids = ElementIds::from(index);
        let extracted = self.extract(&eids);
        (eids, extracted)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use crate::mesh_examples as me;

    // #[test]
    // fn test_umesh_element_selection() {
    //     let mesh = me::make_mesh_2d_quad();
    //     let selected_ids = Selector::new(mesh)
    //         .centroids()
    //         .in_rectangle(&[0.0, 0.0], &[1.0, 1.0])
    //         .index;
    //     assert_eq!(selected_ids.len(), 1);
    //     assert_eq!(selected_ids.get(&ElementType::QUAD4).unwrap(), &vec![0]);
    // }

    #[test]
    fn test_umesh_element_selection() {
        use Selection as Sl;
        let mesh = me::make_mesh_2d_multi();
        let expr = Sl::in_bbox([0.0, 0.0], [1., 1.]) & Sl::types(vec![ElementType::QUAD4]);
        let (eids, mesh_sel) = mesh.select(expr);
        assert_eq!(mesh_sel.num_elements(), 1)
        assert_eq!(eids.iter_blocks(), 1)
    }
}
