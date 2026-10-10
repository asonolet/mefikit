//! The single entry point for building a gradient operator.
//!
//! Every evaluation target reduces to the same sparse operation, so the crate exposes one type,
//! [`super::operator::GradientOperator`], and three constructors: [`GradientOperator::new`]
//! (target mesh cells), [`GradientOperator::on_source`] (source mesh cells) and
//! [`GradientOperator::at_points`] (arbitrary point cloud). Each constructor dispatches to the
//! per-method `prepare` functions; this module depends on the method submodules and glues them
//! under a common call while the apply-time machinery ([`super::operator`]) has no dependency on
//! any specific method.

use ndarray as nd;

use super::moving_least_squares;
use super::operator::{GradientMethod, GradientOperator};
use crate::mesh::UMeshView;

impl GradientOperator {
    /// Builds a gradient operator that evaluates the field gradient on the cells of `mesh_tgt`.
    ///
    /// # Panics
    ///
    /// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or if it is not 2 or 3.
    /// - If the source mesh is empty, or if `k` is zero.
    pub fn new(mesh_src: &UMeshView, mesh_tgt: &UMeshView, method: GradientMethod) -> Self {
        match method {
            GradientMethod::MovingLeastSquares { k, weighting } => {
                moving_least_squares::prepare_cells(mesh_src, mesh_tgt, k, weighting)
            }
        }
    }

    /// Builds a gradient operator that evaluates the field gradient on the source mesh's own cells.
    ///
    /// # Panics
    ///
    /// - If the source mesh does not live in a 2D or 3D space.
    /// - If the source mesh is empty, or if `k` is zero.
    pub fn on_source(mesh_src: &UMeshView, method: GradientMethod) -> Self {
        match method {
            GradientMethod::MovingLeastSquares { k, weighting } => {
                moving_least_squares::prepare_on_source(mesh_src, k, weighting)
            }
        }
    }

    /// Builds a gradient operator that evaluates the field gradient at `points`.
    ///
    /// `points` is an `(n_points, d)` array of coordinates in the source mesh's space.
    ///
    /// # Panics
    ///
    /// - If the source mesh does not live in a 2D or 3D space.
    /// - If the source mesh is empty, if `k` is zero, or if `points` does not have the space
    ///   dimension as number of columns.
    pub fn at_points(
        mesh_src: &UMeshView,
        points: &nd::ArrayView2<f64>,
        method: GradientMethod,
    ) -> Self {
        match method {
            GradientMethod::MovingLeastSquares { k, weighting } => {
                moving_least_squares::prepare_points(mesh_src, points, k, weighting)
            }
        }
    }
}
