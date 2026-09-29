//! `TransferOperator`: the prepare-once / apply-many half of a transfer.

use mefikit::prelude as mf;
use mefikit::tools::Transfer as _;

use crate::ffi::bridge::{Dimension, FieldNature, TransferMethod};
use crate::ffi::{Error, TransferOperator, UMesh};

impl TransferOperator {
    /// Precomputes the transfer from `src` to `tgt`.
    ///
    /// This walks the geometry (k-nearest-neighbours, or cell overlap, depending
    /// on `method`) and is by far the expensive part of a transfer. The result
    /// holds no reference to either mesh and stays valid as long as their
    /// geometry does not change, so build it once and call
    /// [`apply_update`](Self::apply_update) for every field and time step.
    pub fn prepare(src: &UMesh, tgt: &UMesh, method: &TransferMethod) -> Result<Box<Self>, Error> {
        let operator = mf::TransferOperator::new(&src.0.view(), &tgt.0.view(), method.to_core()?);
        Ok(Box::new(Self(operator)))
    }

    /// Transfers field `field_name` of `src` onto `tgt` under `tgt_field_name`.
    ///
    /// Target cells not covered by the source mesh get `default_value`. The
    /// field is replaced if `tgt_field_name` already exists on `tgt`.
    pub fn apply_update(
        &self,
        src: &UMesh,
        field_name: &str,
        tgt: &mut UMesh,
        tgt_field_name: &str,
        default_value: f64,
        nature: FieldNature,
    ) -> Result<(), Error> {
        let field = src.0.field(field_name, None).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "source mesh has no field named '{field_name}' at its topological dimension"
            ))
        })?;
        self.0.apply_update(
            &mut tgt.0,
            tgt_field_name,
            &field,
            nature.to_core()?,
            default_value,
        );
        Ok(())
    }

    /// Topological dimension of the cells this operator writes to.
    #[must_use]
    pub fn target_dimension(&self) -> Dimension {
        Dimension::from(self.0.tgt_dim())
    }
}

/// Prepares a transfer and applies it in one call.
///
/// Convenient when a single field is transferred once; when you transfer several
/// fields (or several time steps) onto the same meshes, use
/// [`TransferOperator::prepare`] instead so the precompute is paid for once.
pub fn transfer_field(
    src: &UMesh,
    field_name: &str,
    tgt: &mut UMesh,
    tgt_field_name: &str,
    method: &TransferMethod,
    default_value: f64,
    nature: FieldNature,
) -> Result<(), Error> {
    let operator = TransferOperator::prepare(src, tgt, method)?;
    operator.apply_update(src, field_name, tgt, tgt_field_name, default_value, nature)
}
