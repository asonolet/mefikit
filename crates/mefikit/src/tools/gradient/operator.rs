//! A sparse gradient operator for scalar fields on meshes and point clouds.
//!
//! The gradient tool follows the same two-stage design as the transfer tool: a geometry-only
//! precompute turns the source points and the evaluation points into a sparse operator
//! ([`GradientOperator`]), and applying it to a scalar field is a sparse matrix-vector product
//! that can be reused for many fields (e.g. across time steps) as long as the point sets do not
//! change.
//!
//! Unlike a transfer, the result is a vector field with `d` trailing components (one per space
//! direction), so the per-target weights are stored as `d` weights per neighbour. The evaluation
//! points are either the cells of a target mesh, the cells of the source mesh itself, or an
//! arbitrary cloud of points ([`GradientTarget`]).

use std::collections::BTreeMap;

use ndarray as nd;
use ndarray::{Axis, concatenate};

use crate::mesh::{Dimension, ElementType, FieldArcD, FieldOwnedD, FieldViewD, UMesh};
use crate::tools::meshless::DistanceWeighting;

/// The reconstruction method used to estimate the gradient of a scalar field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GradientMethod {
    /// Weighted degree-1 moving least-squares fit over the `k` nearest source points.
    ///
    /// For each evaluation point the `k` nearest source points are gathered and a degree-1
    /// polynomial is fitted through them by weighted least squares, the weights coming from
    /// [`DistanceWeighting`]. The gradient of the field is the gradient of that polynomial, which
    /// reproduces affine fields exactly when the local system is full rank.
    MovingLeastSquares {
        /// Number of nearest source points used in each local fit.
        k: usize,
        /// Distance kernel of the weighted fit.
        weighting: DistanceWeighting,
    },
}

/// Sparse gradient rows for one target element type (or for a cloud of points).
///
/// Each target point occupies `d` consecutive rows of `weights` per neighbour entry, where `d` is
/// the space dimension. The gradient component `a` at target point `j` is
/// `sum_l weights[p(j,l)*d + a] * f(src_idx[p(j,l)])`.
#[derive(Clone, Debug)]
pub struct GradientRowSparse {
    pub(crate) row_ptr: Vec<usize>,
    pub(crate) src_idx: Vec<usize>,
    pub(crate) weights: Vec<f64>,
}

impl GradientRowSparse {
    /// Number of target points described by these rows.
    pub fn n_rows(&self) -> usize {
        self.row_ptr.len() - 1
    }
}

/// Where a [`GradientOperator`] evaluates the gradient of the scalar field.
#[derive(Clone, Debug)]
pub enum GradientTarget {
    /// The centroids of the cells of a target mesh, grouped per element type.
    Cells(Vec<(ElementType, GradientRowSparse)>),
    /// An arbitrary cloud of points, in the order they were given.
    Points(GradientRowSparse),
}

/// A reusable sparse gradient operator.
///
/// Build one with [`GradientOperator::new`] (evaluate on the cells of a target mesh),
/// [`GradientOperator::on_source`] (evaluate on the source mesh's own cells) or
/// [`GradientOperator::at_points`] (evaluate on an arbitrary point cloud).
#[derive(Clone, Debug)]
pub struct GradientOperator {
    method: GradientMethod,
    /// Topological dimension of the source cells the field is defined on.
    src_dim: Dimension,
    /// Space dimension, which is also the number of gradient components.
    space: usize,
    /// Number of source cells the operator was built from, flattened over element types in
    /// BTreeMap order.
    n_src: usize,
    /// Topological dimension of the target cells ([`GradientTarget::Cells`] only).
    tgt_dim: Option<Dimension>,
    target: GradientTarget,
}

impl GradientOperator {
    /// Assembles a [`GradientOperator`] from fully built sparse data; called by the method
    /// `prepare` functions in the sibling submodules.
    pub(crate) fn build(
        method: GradientMethod,
        src_dim: Dimension,
        space: usize,
        n_src: usize,
        tgt_dim: Option<Dimension>,
        target: GradientTarget,
    ) -> Self {
        Self {
            method,
            src_dim,
            space,
            n_src,
            tgt_dim,
            target,
        }
    }

    /// Number of gradient components (the space dimension, 2 or 3).
    pub fn n_components(&self) -> usize {
        self.space
    }

    /// The reconstruction method this operator was built with.
    pub fn method(&self) -> GradientMethod {
        self.method
    }

    /// Topological dimension of the source cells the field is defined on.
    pub fn src_dim(&self) -> Dimension {
        self.src_dim
    }

    /// Whether this operator evaluates on mesh cells (as opposed to an arbitrary point cloud).
    pub fn has_cell_target(&self) -> bool {
        matches!(self.target, GradientTarget::Cells(_))
    }

    /// Evaluates the gradient of a scalar source field at an arbitrary point cloud.
    ///
    /// Returns an `(n_points, d)` array, one row of `d` components per evaluation point.
    ///
    /// # Panics
    ///
    /// If this operator was not built with [`GradientOperator::at_points`], or if the field is not
    /// a scalar field defined on the source cells.
    pub fn apply_points(&self, field: &FieldViewD, default: f64) -> nd::Array2<f64> {
        let GradientTarget::Points(rows) = &self.target else {
            panic!("This gradient operator evaluates on mesh cells; use `apply` instead");
        };
        let src = self.concatenate_source(field);
        self.apply_rows(rows, &src.view(), default)
    }

    /// Concatenates the per-element-type arrays of `field` in BTreeMap order, asserting that the
    /// source has one entry per source cell and a single component.
    fn concatenate_source(&self, field: &FieldViewD) -> nd::ArrayD<f64> {
        assert_eq!(
            field.dimension(),
            Some(self.src_dim),
            "The field should be defined on the source topological dimension {src_dim:?}, got {got:?}",
            src_dim = self.src_dim,
            got = field.dimension()
        );
        // Concatenate the per-element-type source arrays in the same order used to build the
        // operator (BTreeMap order), so the source indices stored in the rows are valid whatever
        // the source element types are.
        let src_views: Vec<nd::ArrayViewD<f64>> = field.0.values().map(|a| a.view()).collect();
        concatenate(Axis(0), src_views.as_slice()).unwrap()
    }

    /// Evaluates the sparse rows on a flat source array `src` of shape `(n_src, ...)`, producing
    /// a `(n_tgt, d)` array.
    fn apply_rows(
        &self,
        rows: &GradientRowSparse,
        src: &nd::ArrayViewD<f64>,
        default: f64,
    ) -> nd::Array2<f64> {
        let n_src = src.shape()[0];
        assert_eq!(
            n_src, self.n_src,
            "The field should have one entry per source cell, got {n_src} for {}",
            self.n_src
        );
        let n_compo = src.len() / n_src;
        assert_eq!(
            n_compo, 1,
            "The gradient is only defined for a scalar source field, got {n_compo} components per cell"
        );
        let src = src.view().into_shape_with_order((n_src, n_compo)).unwrap();

        let d = self.space;
        let n_tgt = rows.n_rows();
        let mut out = nd::Array2::<f64>::zeros((n_tgt, d));
        for j in 0..n_tgt {
            let (lo, hi) = (rows.row_ptr[j], rows.row_ptr[j + 1]);
            if lo == hi {
                out.row_mut(j).fill(default);
                continue;
            }
            let mut row = out.row_mut(j);
            for p in lo..hi {
                let value = src[[rows.src_idx[p], 0]];
                let w = &rows.weights[p * d..(p + 1) * d];
                for a in 0..d {
                    row[a] += w[a] * value;
                }
            }
        }
        out
    }
}

/// The gradient of a scalar source field, as a vector field on the target.
pub trait Gradient {
    /// Evaluates `field` (a scalar field on the source cells) at the target cells.
    ///
    /// `default` is used for target cells whose sparse row is empty. The result is a vector field
    /// with the space dimension as trailing axis.
    fn apply(&self, field: &FieldViewD, default: f64) -> FieldOwnedD;

    /// Dimension of the target cells that receive the gradient.
    fn tgt_dim(&self) -> Dimension;

    /// Number of gradient components (the space dimension).
    fn n_components(&self) -> usize;

    /// Evaluates the field and stores it in `target` under `name`.
    ///
    /// Returns the previous field if it existed, or `None` if it did not.
    fn apply_update(
        &self,
        target: &mut UMesh,
        name: &str,
        field: &FieldViewD,
        default: f64,
    ) -> Option<FieldArcD>
    where
        Self: Sized,
    {
        let field = self.apply(field, default);
        target.update_field(name, field.into_shared())
    }
}

impl Gradient for GradientOperator {
    fn apply(&self, field: &FieldViewD, default: f64) -> FieldOwnedD {
        let GradientTarget::Cells(blocks) = &self.target else {
            panic!(
                "This gradient operator evaluates at arbitrary points; use `apply_points` instead"
            );
        };
        let src = self.concatenate_source(field);
        let mut res = BTreeMap::new();
        for (et, rows) in blocks {
            res.insert(*et, self.apply_rows(rows, &src.view(), default).into_dyn());
        }
        FieldOwnedD::new(res)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
            .expect("This gradient operator evaluates at arbitrary points and has no target cells")
    }

    fn n_components(&self) -> usize {
        self.space
    }
}
