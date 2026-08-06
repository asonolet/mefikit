//! Field transfer between meshes.
//!
//! This module provides the [`Transfer`] trait and its implementations for transferring
//! per-element fields from a source mesh to the cells of a target mesh. The geometry-only
//! precompute (locating each target cell in the source mesh) is performed once when a transfer
//! operator is constructed, so the same operator can be reused to evaluate many fields (for
//! example across time steps) as long as the meshes do not change.
//!
//! A transfer may downcast: a field defined on the cells of a full-dimensional source mesh can
//! be transferred onto the cells of a lower-dimensional target mesh embedded in the same space
//! (for example a 3D volume mesh onto a 2D manifold in 3D space).

use crate::mesh::{Dimension, FieldArcD, FieldOwnedD, FieldViewD, UMesh};

/// Nature of a field, governing how its values behave when the supporting cells change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldNature {
    /// Per-unit-measure field (temperature, density, pressure).
    Intensive,
    /// Total-quantity field (mass, energy).
    Extensive,
}

/// How each target cell's representative sampling point is chosen (fixed at precompute time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointLocation {
    /// Arithmetic mean of the cell's vertices.
    Centroid,
    /// True region centroid (center of mass).
    Barycenter,
    /// A strictly-interior point.
    StrictInterior,
}

/// A geometry-only precompute for transferring a field from source cells to target cells.
pub trait Transfer {
    /// Evaluates `field` (defined on the source cells) at the resolved locations.
    ///
    /// `default` is used for target cells that are not covered by the source mesh. The
    /// `field_nature` parameter lets schemes distinguish intensive and extensive fields.
    fn apply(&self, field: &FieldViewD, field_nature: FieldNature, default: f64) -> FieldOwnedD;

    /// Dimension of the target cells that receive values.
    fn tgt_dim(&self) -> Dimension;

    /// Evaluates the field and stores it in `target` under `name`.
    ///
    /// Returns the previous field if it existed, or `None` if it did not.
    fn apply_update(
        &self,
        target: &mut UMesh,
        name: &str,
        field: &FieldViewD,
        field_nature: FieldNature,
        default: f64,
    ) -> Option<FieldArcD>
    where
        Self: Sized,
    {
        let field = self.apply(field, field_nature, default);
        target.update_field(name, field.into_shared())
    }
}
