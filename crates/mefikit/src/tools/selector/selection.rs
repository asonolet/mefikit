//! Selection types and operations for mesh queries.

use std::ops::{BitAnd, BitOr, BitXor, Not, Sub};
use std::sync::Arc;
use std::thread;

use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType, UMesh, UMeshView};
use crate::tools::fieldexpr::{Evaluable, FieldExpr, infer_dim};

use super::centroid::CentroidSelection;
use super::element::ElementSelection;
use super::field::FieldSelection;
use super::group::GroupSelection;
use super::node::NodeSelection;

pub use super::field::Comparable;

/// Trait for selection objects that can filter element IDs.
pub trait Select {
    /// Applies the selection to the given mesh view, filtering the element ID set.
    ///
    /// `dim` sets the dimension on which field-value comparisons are evaluated; `None`
    /// infers it from the expression (matching field-computation semantics).
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids: ElementIdsSet,
        dim: Option<Dimension>,
    ) -> ElementIdsSet;
}

/// A selection expression for querying mesh elements.
///
/// Selections can be combined using boolean operators (AND, OR, XOR, NOT)
/// to build complex queries.
#[derive(Clone, Debug)]
pub enum Selection {
    /// Selection based on element type or dimension.
    ElementSelection(ElementSelection),
    /// Selection based on element group membership.
    GroupSelection(GroupSelection),
    /// Selection based on field values.
    FieldSelection(FieldSelection),
    /// Selection based on element centroid positions.
    CentroidSelection(CentroidSelection),
    /// Selection based on node positions.
    NodeSelection(NodeSelection),
    /// Binary boolean expression combining two selections.
    BinarayExpr(BinarayExpr),
    /// Negation of a selection.
    NotExpr(NotExpr),
}

/// Boolean operators for combining selections.
#[derive(Copy, Clone, Debug)]
pub enum BooleanOp {
    /// Logical AND (intersection).
    And,
    /// Logical OR (union).
    Or,
    /// Logical XOR (symmetric difference).
    Xor,
    /// Set difference.
    Diff,
}

/// A binary boolean expression combining two selections.
#[derive(Clone, Debug)]
pub struct BinarayExpr {
    pub operator: BooleanOp,
    pub left: Arc<Selection>,
    pub right: Arc<Selection>,
}

/// A negation expression wrapping a selection.
#[derive(Clone, Debug)]
pub struct NotExpr(pub Arc<Selection>);

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
    pub fn group(self, name: &str) -> Self {
        let right = Self::GroupSelection(GroupSelection::IncludeGroup(name.to_string()));
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
    pub fn exclude_group(self, name: &str) -> Self {
        let right = Self::GroupSelection(GroupSelection::ExcludeGroup(name.to_string()));
        Self::BinarayExpr(BinarayExpr {
            operator: BooleanOp::And,
            left: Arc::new(self),
            right: Arc::new(right),
        })
    }
}

/// Creates a selection for nodes inside an axis-aligned 3D bounding box.
pub fn nbbox(min: [f64; 3], max: [f64; 3], all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::BBox { all, min, max })
}

/// Creates a selection for nodes inside an axis-aligned 2D rectangle.
pub fn nrect(min: [f64; 2], max: [f64; 2], all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Rect { all, min, max })
}

/// Creates a selection for nodes inside a 3D sphere.
pub fn nsphere(center: [f64; 3], r2: f64, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Sphere { all, center, r: r2 })
}

/// Creates a selection for nodes inside a 2D circle.
pub fn ncircle(center: [f64; 2], r2: f64, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Circle { all, center, r: r2 })
}

/// Creates a selection for nodes by their indices.
pub fn nids(ids: Vec<usize>, all: bool) -> Selection {
    Selection::NodeSelection(NodeSelection::Ids { all, ids })
}

/// Creates a selection for element centroids inside a 3D bounding box.
pub fn bbox(min: [f64; 3], max: [f64; 3]) -> Selection {
    Selection::CentroidSelection(CentroidSelection::BBox { min, max })
}

/// Creates a selection for element centroids inside a 2D rectangle.
pub fn rect(min: [f64; 2], max: [f64; 2]) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Rect { min, max })
}

/// Creates a selection for element centroids inside a 3D sphere.
pub fn sphere(center: [f64; 3], r2: f64) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Sphere { center, r2 })
}

/// Creates a selection for element centroids inside a 2D circle.
pub fn circle(center: [f64; 2], r2: f64) -> Selection {
    Selection::CentroidSelection(CentroidSelection::Circle { center, r2 })
}

/// Creates a selection matching every element of the mesh.
pub fn all() -> Selection {
    Selection::ElementSelection(ElementSelection::All)
}

/// Creates a selection for elements of specific types.
pub fn types(elems: Vec<ElementType>) -> Selection {
    Selection::ElementSelection(ElementSelection::Types(elems))
}

/// Creates a selection for elements of specific dimensions.
pub fn dimensions(dims: Vec<Dimension>) -> Selection {
    Selection::ElementSelection(ElementSelection::Dimensions(dims))
}

/// Creates a selection for elements by their IDs.
pub fn ids(eids: ElementIds) -> Selection {
    Selection::ElementSelection(ElementSelection::InIds(eids))
}

/// Creates a selection for elements belonging to a named group.
pub fn group(name: &str) -> Selection {
    Selection::GroupSelection(GroupSelection::IncludeGroup(name.to_string()))
}

/// Creates a selection for elements NOT belonging to a named group.
pub fn exclude_group(name: &str) -> Selection {
    Selection::GroupSelection(GroupSelection::ExcludeGroup(name.to_string()))
}

impl Select for Selection {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        dim: Option<Dimension>,
    ) -> ElementIdsSet {
        match self {
            Self::ElementSelection(elemt_expr) => elemt_expr.select(view, eids_in, dim),
            Self::NodeSelection(nodes_expr) => nodes_expr.select(view, eids_in, dim),
            Self::CentroidSelection(centroid) => centroid.select(view, eids_in, dim),
            Self::GroupSelection(group) => group.select(view, eids_in, dim),
            Self::FieldSelection(field) => field.select(view, eids_in, dim),
            Self::NotExpr(not) => not.select(view, eids_in, dim),
            Self::BinarayExpr(binary) => binary.select(view, eids_in, dim),
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
    fn select<'a>(
        &'a self,
        _view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        _dim: Option<Dimension>,
    ) -> ElementIdsSet {
        match self {
            Self::All => eids_in,
            Self::Types(types) => Self::select_types(types.as_slice(), eids_in),
            Self::Dimensions(dims) => Self::select_dimensions(dims.as_slice(), eids_in),
            Self::InIds(ids) => Self::select_ids(ids.clone(), eids_in),
        }
    }
}

impl Select for NodeSelection {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        _dim: Option<Dimension>,
    ) -> ElementIdsSet {
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
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        _dim: Option<Dimension>,
    ) -> ElementIdsSet {
        match self {
            Self::IncludeGroup(name) => Self::include_group(name, view, eids_in),
            Self::ExcludeGroup(name) => Self::exclude_group(name, view, eids_in),
        }
    }
}

impl Select for FieldSelection {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        mut eids_in: ElementIdsSet,
        dim: Option<Dimension>,
    ) -> ElementIdsSet {
        let (e1, e2): (&FieldExpr, &FieldExpr) = match self {
            Self::Gt(a, b) => (a, b),
            Self::Geq(a, b) => (a, b),
            Self::Lt(a, b) => (a, b),
            Self::Leq(a, b) => (a, b),
            Self::Eq(a, b) => (a, b),
            Self::Neq(a, b) => (a, b),
        };
        let dim = match dim {
            Some(d) => d,
            None => infer_dim(view, &[e1, e2]),
        };
        let eids = match self {
            Self::Gt(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.gt(f2)
            }
            Self::Geq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.ge(f2)
            }
            Self::Lt(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.lt(f2)
            }
            Self::Leq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.le(f2)
            }
            Self::Eq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.eq(f2)
            }
            Self::Neq(expr1, expr2) => {
                let f1 = expr1.evaluate(view, Some(dim));
                let f2 = &expr2.evaluate(view, Some(dim));
                f1.neq(f2)
            }
        };
        eids_in.intersection(&eids.into());
        eids_in
    }
}

impl Select for NotExpr {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        mut eids_in: ElementIdsSet,
        dim: Option<Dimension>,
    ) -> ElementIdsSet {
        let all_ids: ElementIdsSet = ElementIdsSet(
            view.blocks()
                .map(|(k, v)| (*k, (0..v.len()).collect()))
                .collect(),
        );
        let not_sel = self.0.select(view, all_ids, dim);
        // let mut not_sel = all_ids;
        // not_sel.difference(&sel);
        // sel0.intersection(&not_sel);
        eids_in.difference(&not_sel);
        eids_in
    }
}

impl Select for BinarayExpr {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        dim: Option<Dimension>,
    ) -> ElementIdsSet {
        match self.operator {
            BooleanOp::And => {
                if self.left.weight() < self.right.weight() {
                    let selection = self.left.select(view, eids_in, dim);
                    self.right.select(view, selection, dim)
                } else {
                    let selection = self.right.select(view, eids_in, dim);
                    self.left.select(view, selection, dim)
                }
            }
            BooleanOp::Or => {
                let (mut sel1, sel2) = thread::scope(move |s| {
                    let eids_clone = eids_in.clone();
                    let h1 = s.spawn(move || self.left.select(view, eids_clone, dim));
                    let h2 = s.spawn(move || self.right.select(view, eids_in, dim));
                    (h1.join().unwrap(), h2.join().unwrap())
                });
                sel1.union(&sel2);
                sel1
            }
            BooleanOp::Xor => {
                let mut sel1 = self.left.select(view, eids_in.clone(), dim);
                let sel2 = self.right.select(view, eids_in, dim);
                sel1.symmetric_difference(&sel2);
                sel1
            }
            BooleanOp::Diff => {
                let mut sel1 = self.left.select(view, eids_in.clone(), dim);
                let sel2 = self.right.select(view, eids_in, dim);
                sel1.difference(&sel2);
                sel1
            }
        }
    }
}

impl Select for CentroidSelection {
    fn select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        eids_in: ElementIdsSet,
        _dim: Option<Dimension>,
    ) -> ElementIdsSet {
        match self {
            Self::BBox { min, max } => Self::in_bbox(min, max, view, eids_in),
            Self::Rect { min, max } => Self::in_rectangle(min, max, view, eids_in),
            Self::Sphere { center, r2 } => Self::in_sphere(center, *r2, view, eids_in),
            Self::Circle { center, r2 } => Self::in_circle(center, *r2, view, eids_in),
        }
    }
}

/// Trait for applying selections to meshes.
pub trait MeshSelect {
    /// Returns the element IDs matching the selection expression.
    ///
    /// `dim` sets the dimension on which field-value comparisons are evaluated (`None`
    /// infers it). Element/centroid/node selections ignore it.
    fn select_ids(&self, expr: Selection, dim: Option<Dimension>) -> ElementIds;

    /// Returns matching element IDs and extracts a sub-mesh.
    fn select(
        &self,
        expr: Selection,
        with_fields: bool,
        dim: Option<Dimension>,
    ) -> (ElementIds, Self);
}

impl MeshSelect for UMesh {
    fn select_ids(&self, expr: Selection, dim: Option<Dimension>) -> ElementIds {
        let index: ElementIdsSet = ElementIdsSet(
            self.blocks()
                .map(|(k, v)| (*k, (0..v.len()).collect()))
                .collect(),
        );
        expr.select(&self.view(), index, dim).into()
    }
    fn select(
        &self,
        expr: Selection,
        with_fields: bool,
        dim: Option<Dimension>,
    ) -> (ElementIds, Self) {
        let eids = self.select_ids(expr, dim);
        let extracted = self.extract(&eids, with_fields);
        (eids, extracted)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr0;

    use super::*;
    use crate::mesh::ElementType;
    use crate::mesh_examples as me;
    use crate::prelude as mf;
    use crate::tools::fieldexpr::{arr, field, nz};
    use crate::tools::{Measurable, RegularUMeshBuilder};
    use ndarray as nd;

    #[test]
    fn test_umesh_element_selection() {
        use ElementType::*;
        let mesh = me::make_mesh_2d_quad();
        // Here is my cool expression !
        let eps = -1e12;
        let (_eids, mesh_sel) = mesh.select(
            (rect([-eps, -eps], [1. + eps, 1. + eps]) | ncircle([0.0, 0.0], 1.0, false))
                & types(vec![QUAD4]),
            false,
            None,
        );
        assert_eq!(mesh_sel.num_elements(), 1);
    }

    #[test]
    fn test_umesh_measure() {
        let mut mesh = RegularUMeshBuilder::new()
            .add_axis((0..=10).map(|k| ((k * k) as f64) / 100.0).collect())
            .add_axis((0..=10).map(|k| ((k * k) as f64) / 100.0).collect())
            .build();
        mesh.measure_update("M", None);
        let two_surf = field("M") * arr(arr0(2.0));
        let threshold = arr(arr0(0.01));
        let expr = two_surf.gt(threshold);
        let eids = mesh.select_ids(Selection::FieldSelection(expr), None);
        assert_eq!(eids.len(), 62)
    }

    fn hex8_quad4_mesh() -> mf::UMesh {
        let coords = nd::Array2::from_shape_vec(
            (9, 3),
            vec![
                0.0, 0.0, 0.0, // 0
                1.0, 0.0, 0.0, // 1
                1.0, 1.0, 0.0, // 2
                0.0, 1.0, 0.0, // 3
                0.0, 0.0, 1.0, // 4
                1.0, 0.0, 1.0, // 5
                1.0, 1.0, 1.0, // 6
                0.0, 1.0, 1.0, // 7
                0.0, 0.0, 2.0, // 8
            ],
        )
        .unwrap();
        let mut mesh = mf::UMesh::new(coords.into());
        mesh.add_regular_block(
            mf::ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh.add_regular_block(
            mf::ElementType::QUAD4,
            nd::arr2(&[[4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh
    }

    #[test]
    fn test_select_normal_infers_hypersurface_dim() {
        // On a mesh with HEX8 + QUAD4, `nz > 0.9` must infer the hypersurface dim
        // (space_dim - 1 = 2) and select only the QUAD4 boundary face, not HEX8.
        let mesh = hex8_quad4_mesh();
        let sel = nz().gt(arr(nd::arr0(0.9)));
        let eids = mesh.select_ids(Selection::FieldSelection(sel), None);
        assert!(eids.contains_type(ElementType::QUAD4));
        assert!(!eids.contains_type(ElementType::HEX8));
    }

    #[test]
    fn test_select_normal_explicit_dim_is_strict() {
        // An explicit dim=2 forces the QUAD4 evaluation regardless of inference.
        let mesh = hex8_quad4_mesh();
        let sel = nz().gt(arr(nd::arr0(0.9)));
        let eids = mesh.select_ids(Selection::FieldSelection(sel), Some(Dimension::D2));
        assert!(eids.contains_type(ElementType::QUAD4));
        assert!(!eids.contains_type(ElementType::HEX8));
    }
}
