#[cfg(feature = "rayon")]
use rayon::prelude::*;

use std::ops::{BitAnd, BitOr, BitXor, Not, Sub};
use std::sync::Arc;

use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType, UMesh, UMeshView};
use crate::tools::fieldexpr::Evaluable;

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
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids: ElementIdsSet) -> ElementIdsSet;
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
        let right = Self::NodeSelection(NodeSelection::Sphere { all, center, r: r2 });
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn ncircle(self, center: [f64; 2], r2: f64, all: bool) -> Self {
        let right = Self::NodeSelection(NodeSelection::Circle { all, center, r: r2 });
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
    Selection::NodeSelection(NodeSelection::Sphere { all, center, r: r2 })
}
pub fn ncircle(center: [f64; 2], r2: f64, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Circle { all, center, r: r2 })
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
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        match self {
            Self::ElementSelection(elemt_expr) => elemt_expr.select(view, eids_in),
            Self::NodeSelection(nodes_expr) => nodes_expr.select(view, eids_in),
            Self::CentroidSelection(centroid) => centroid.select(view, eids_in),
            Self::GroupSelection(group) => group.select(view, eids_in),
            Self::FieldSelection(field) => field.select(view, eids_in),
            Self::NotExpr(not) => not.select(view, eids_in),
            Self::BinarayExpr(binary) => binary.select(view, eids_in),
        }
    }
}

impl BitAnd for Selection {
    type Output = Selection;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl BitOr for Selection {
    type Output = Selection;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Or,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl BitXor for Selection {
    type Output = Selection;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Xor,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl Sub for Selection {
    type Output = Selection;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::Diff,
            left: Arc::new(self),
            right: Arc::new(rhs),
        })
    }
}

impl Not for Selection {
    type Output = Selection;

    fn not(self) -> Self::Output {
        Self::NotExpr(NotExpr(Arc::new(self)))
    }
}

// Leaf operations

impl Select for ElementSelection {
    fn select<'a>(&'a self, _view: &'a UMeshView<'a>, mut eids_in: ElementIdsSet) -> ElementIdsSet {
        match self {
            Self::Types(types) => Self::select_types(types.as_slice(), eids_in),
            Self::Dimensions(dims) => Self::select_dimensions(dims.as_slice(), eids_in),
            Self::InIds(ids) => Self::select_ids(ids.clone(), eids_in),
        }
    }
}

impl Select for NodeSelection {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        match self {
            Self::BBox { all, min, max } => Self::in_bbox(*all, min, max, view, eids_in),
            Self::Rect { all, min, max } => Self::in_rectangle(*all, min, max, view, eids_in),
            Self::Sphere { all, center, r } => Self::in_sphere(*all, center, *r, view, eids_in),
            Self::Circle { all, center, r } => Self::in_circle(*all, center, *r, view, eids_in),
            Self::Ids { all, ids } => Self::id_in(*all, ids.as_slice(), view, eids_in),
        }
    }
}

impl Select for GroupSelection {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        match self {
            Self::IncludeGroup(name) => Self::include_group(name, view, eids_in),
            Self::ExcludeGroup(name) => Self::exclude_group(name, view, eids_in),
            Self::IncludeFamily(fid) => Self::include_family(*fid, view, eids_in),
            Self::ExcludeFamily(fid) => Self::exclude_family(*fid, view, eids_in),
        }
    }
}

impl Select for FieldSelection {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, mut eids_in: ElementIdsSet) -> ElementIdsSet {
        let eids = match self {
            Self::Gt(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.gt(f2)
            }
            Self::Geq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.ge(f2)
            }
            Self::Lt(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.lt(f2)
            }
            Self::Leq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.le(f2)
            }
            Self::Eq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.eq(f2)
            }
            Self::Neq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, None);
                let f2 = &expr2.evaluate(view, None);
                f1.neq(f2)
            }
        };
        eids_in.intersection(&eids.into());
        eids_in
    }
}

impl Select for NotExpr {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        self.not_select(view, eids_in)
    }
}

impl Select for BinarayExpr {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        match self.operator {
            BooleanOp::And => self.and_select(view, eids_in),
            BooleanOp::Or => self.or_select(view, eids_in),
            BooleanOp::Xor => self.xor_select(view, eids_in),
            BooleanOp::Diff => self.diff_select(view, eids_in),
        }
    }
}

impl Select for CentroidSelection {
    fn select<'a>(&'a self, view: &'a UMeshView<'a>, eids_in: ElementIdsSet) -> ElementIdsSet {
        match self {
            Self::BBox { min, max } => Self::in_bbox(min, max, view, eids_in),
            Self::Rect { min, max } => Self::in_rectangle(min, max, view, eids_in),
            Self::Sphere { center, r2 } => Self::in_sphere(center, *r2, view, eids_in),
            Self::Circle { center, r2 } => Self::in_circle(center, *r2, view, eids_in),
        }
    }
}

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
        expr.select(&self.view(), index).into()
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
        let mesh = me::make_mesh_2d_quad();
        // Here is my cool expression !
        let eps = -1e12;
        let (_eids, mesh_sel) = mesh.select(
            (rect([-eps, -eps], [1. + eps, 1. + eps]) | ncircle([0.0, 0.0], 1.0, false))
                & types(vec![QUAD4]),
        );
        assert_eq!(mesh_sel.num_elements(), 1);
    }
}
