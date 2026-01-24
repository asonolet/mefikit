#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::ops::{BitAnd, BitOr, BitXor, Not};
use std::sync::Arc;

use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType, UMesh, UMeshView};

use super::binary::{BinarayExpr, BooleanOp, NotExpr};
use super::centroid::CentroidSelection;
use super::element::ElementSelection;
use super::field::FieldSelection;
use super::group::GroupSelection;
use super::node::NodeSelection;

/// This object is the one that will be evaluated by unitary selection_ops.
/// The UMeshView is always passed as the same, whereas the ElementsIds are updated. Each unitary
/// op takes a previous ElementsIds list and returns a new one (shorter).
#[derive(Clone, Debug)]
pub struct SelectedView<'a>(pub UMeshView<'a>, pub ElementIdsSet);

pub trait Select {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a>;
}

#[derive(Clone, Debug)]
pub enum Selection {
    ElementSelection(ElementSelection),
    GroupSelection(GroupSelection),
    FieldSelection(FieldSelection),
    CentroidSelection(CentroidSelection),
    NodeSelection(NodeSelection),
    BinarayExpr(BinarayExpr),
    NotExpr(NotExpr),
}

impl Selection {
    /// The lower, the simpler it is to compute and then should be computed first.
    /// 0: compute right now and blocks
    /// 1: to be computed in parallel
    /// 2: computed the latest
    pub fn weight(&self) -> u8 {
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
    pub fn is_leaf(&self) -> bool {
        !matches!(self, Self::BinarayExpr(_) | Self::NotExpr(_))
    }
    /// Switch operations so that simpler/more selective operations are evaluated sooner
    fn _optimize(&self) -> Self {
        todo!()
    }
    pub fn nbbox(self, min: [f64; 3], max: [f64; 3], all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::BBox { all, min, max });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn nrect(self, min: [f64; 2], max: [f64; 2], all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::Rect { all, min, max });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    /// This method filters upon nodes position.
    pub fn nsphere(self, center: [f64; 3], r2: f64, all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::Sphere { all, center, r2 });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn ncircle(self, center: [f64; 2], r2: f64, all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::Circle { all, center, r2 });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn nids(self, ids: Vec<usize>, all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::Ids { all, ids });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn bbox(self, min: [f64; 3], max: [f64; 3]) -> Self {
        let right = Self::CentroidSelection(CentroidSelection::BBox { min, max });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn rect(self, min: [f64; 2], max: [f64; 2]) -> Self {
        let right = Self::CentroidSelection(CentroidSelection::Rect { min, max });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn sphere(self, center: [f64; 3], r2: f64) -> Self {
        let right = Self::CentroidSelection(CentroidSelection::Sphere { center, r2 });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn circle(self, center: [f64; 2], r2: f64) -> Self {
        let right = Self::CentroidSelection(CentroidSelection::Circle { center, r2 });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn types(self, elems: Vec<ElementType>) -> Self {
        let right = Self::ElementSelection(ElementSelection::Types(elems));
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn dimensions(self, dims: Vec<Dimension>) -> Self {
        let right = Self::ElementSelection(ElementSelection::Dimensions(dims));
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn ids(self, eids: ElementIds) -> Self {
        let right = Self::ElementSelection(ElementSelection::InIds(eids));
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
}
pub fn nbbox(min: [f64; 3], max: [f64; 3], all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::BBox { all, min, max })
}
pub fn nrect(min: [f64; 2], max: [f64; 2], all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Rect { all, min, max })
}
/// This method filters upon nodes position.
pub fn nsphere(center: [f64; 3], r2: f64, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Sphere { all, center, r2 })
}
pub fn ncircle(center: [f64; 2], r2: f64, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Circle { all, center, r2 })
}
pub fn nids(ids: Vec<usize>, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Ids { all, ids })
}
pub fn bbox(min: [f64; 3], max: [f64; 3]) -> Selection {
    Selection::CentroidSelection(CentroidSelection::BBox { min, max })
}
pub fn rect(min: [f64; 2], max: [f64; 2]) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Rect { min, max })
}
pub fn sphere(center: [f64; 3], r2: f64) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Sphere { center, r2 })
}
pub fn circle(center: [f64; 2], r2: f64) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Circle { center, r2 })
}
pub fn types(elems: Vec<ElementType>) -> Selection {
    Selection::ElementSelection(ElementSelection::Types(elems))
}
pub fn dimensions(dims: Vec<Dimension>) -> Selection {
    Selection::ElementSelection(ElementSelection::Dimensions(dims))
}
pub fn ids(eids: ElementIds) -> Selection {
    Selection::ElementSelection(ElementSelection::InIds(eids))
}

impl Select for Selection {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            Selection::ElementSelection(elemt_expr) => elemt_expr.select(selection),
            Selection::NodeSelection(nodes_expr) => nodes_expr.select(selection),
            Selection::CentroidSelection(centroid) => centroid.select(selection),
            Selection::BinarayExpr(binary) => binary.select(selection),
            _ => todo!(),
        }
    }
}

impl BitAnd for Selection {
    type Output = Selection;

    fn bitand(self, rhs: Self) -> Self::Output {
        Selection::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl BitOr for Selection {
    type Output = Selection;

    fn bitor(self, rhs: Self) -> Self::Output {
        Selection::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Or,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl BitXor for Selection {
    type Output = Selection;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Selection::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Xor,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl Not for Selection {
    type Output = Selection;

    fn not(self) -> Self::Output {
        Selection::NotExpr(NotExpr(Arc::new(self)))
    }
}

// Leaf operations

impl Select for ElementSelection {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            ElementSelection::Types(types) => Self::select_types(types.as_slice(), selection),
            ElementSelection::Dimensions(dims) => {
                Self::select_dimensions(dims.as_slice(), selection)
            }
            ElementSelection::InIds(ids) => Self::select_ids(ids.clone(), selection),
        }
    }
}

impl Select for NodeSelection {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            NodeSelection::BBox { all, min, max } => {
                NodeSelection::in_bbox(*all, min, max, selection)
            }
            NodeSelection::Rect { all, min, max } => {
                NodeSelection::in_rectangle(*all, min, max, selection)
            }
            NodeSelection::Sphere { all, center, r2 } => {
                NodeSelection::in_sphere(*all, center, *r2, selection)
            }
            NodeSelection::Circle { all, center, r2 } => {
                NodeSelection::in_circle(*all, center, *r2, selection)
            }
            NodeSelection::Ids { all, ids } => Self::id_in(*all, ids.as_slice(), selection),
        }
    }
}

impl Select for BinarayExpr {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self.operator {
            BooleanOp::And => self.and_select(selection),
            BooleanOp::Or => self.or_select(selection),
            _ => todo!(),
        }
    }
}

impl Select for CentroidSelection {
    fn select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        match self {
            CentroidSelection::BBox { min, max } => CentroidSelection::in_bbox(min, max, selection),
            CentroidSelection::Rect { min, max } => {
                CentroidSelection::in_rectangle(min, max, selection)
            }
            CentroidSelection::Sphere { center, r2 } => {
                CentroidSelection::in_sphere(center, *r2, selection)
            }
            CentroidSelection::Circle { center, r2 } => {
                CentroidSelection::in_circle(center, *r2, selection)
            }
        }
    }
}

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

pub trait MeshSelect {
    fn select_ids(&self, expr: Selection) -> ElementIds;
    fn select(&self, expr: Selection) -> (ElementIds, Self);
}

impl MeshSelect for UMesh {
    fn select_ids(&self, expr: Selection) -> ElementIds {
        let index: ElementIdsSet = ElementIdsSet(
            self.blocks()
                .map(|(k, v)| (*k, (0..v.len()).collect()))
                .collect(),
        );
        let SelectedView(_, index) = expr.select(SelectedView(self.view(), index));
        index.into()
    }
    fn select(&self, expr: Selection) -> (ElementIds, Self) {
        let eids = self.select_ids(expr);
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
        use ElementType::*;
        let mesh = me::make_mesh_2d_multi();
        // Here is my cool expression !
        let (_eids, mesh_sel) = mesh.select(
            (rect([0.0, 0.0], [1., 1.]) | ncircle([0.0, 0.0], 1.0, false)) & types(vec![QUAD4]),
        );
        assert_eq!(mesh_sel.num_elements(), 1);
    }
}
