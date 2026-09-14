//! The single entry point for building a transfer operator.
//!
//! Every transfer method reduces to the same sparse operation, so the crate exposes one type,
//! [`super::operator::TransferOperator`], and one constructor, [`TransferOperator::new`], which
//! dispatches to the per-method `prepare` functions. This module depends on all four method
//! submodules (the "new method implementations") and glues them under a single call; it exists
//! apart from `operator` so that the construction logic stays here while the apply-time machinery
//! has no dependency on any specific method.

use super::conservative_p0;
use super::constant_piecewise;
use super::inverse_distance;
use super::moving_least_squares;
use super::operator::{TransferMethod, TransferOperator};
use crate::mesh::UMeshView;

impl TransferOperator {
    /// Builds a transfer operator from `mesh_src` to `mesh_tgt` with the given method.
    ///
    /// The method selects both the coefficient construction and the apply-time behaviour (an
    /// intensive [`TransferMethod::ConservativeP0`] field is normalized by the target cell
    /// measure). Each variant carries exactly the parameters it needs.
    ///
    /// # Panics
    ///
    /// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or if it is not 2 or 3.
    /// - Depending on the method: if a non-empty source/target requirement is not met, if `k` is
    ///   zero, or if an exponent/weighting parameter is not positive.
    pub fn new(mesh_src: &UMeshView, mesh_tgt: &UMeshView, method: TransferMethod) -> Self {
        match method {
            TransferMethod::ConstantPiecewise { point_location } => {
                constant_piecewise::prepare(mesh_src, mesh_tgt, &point_location)
            }
            TransferMethod::ConservativeP0 => conservative_p0::prepare(mesh_src, mesh_tgt),
            TransferMethod::InverseDistance { k, exponent } => {
                inverse_distance::prepare(mesh_src, mesh_tgt, k, exponent)
            }
            TransferMethod::MovingLeastSquares { k, weighting } => {
                moving_least_squares::prepare(mesh_src, mesh_tgt, k, weighting)
            }
        }
    }
}
