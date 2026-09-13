//! A single sparse interpolation/transfer operator shared by all four transfer methods.
//!
//! Every transfer method reduces to the same operation: each target cell is a weighted sum of
//! source cells, `t_j = sum_i w_ji s_i`. What differs from one method to the next is only how the
//! coefficients `w_ji` are obtained at construction time:
//!
//! - [`TransferMethod::ConstantPiecewise`]: one row entry with weight `1.0` pointing at the source
//!   cell containing the sampling point (no entry and the `default` value is kept when nothing is
//!   found).
//! - [`TransferMethod::ConservativeP0`]: one row entry per overlapping source cell, weighted by the
//!   intersection measure. An intensive field is normalized by the target cell measure after the
//!   row is accumulated, an extensive field keeps the raw sum.
//! - [`TransferMethod::InverseDistance`] and [`TransferMethod::MovingLeastSquares`]: exactly `k`
//!   row entries per target point, the interpolation weights over its `k` nearest source points.
//!
//! The coefficients are laid out once in CSR form ([`RowSparse`]) at construction time, so
//! [`Transfer::apply`] is a single sparse matrix-vector product that can be reused for any field
//! as long as the meshes do not change. Each method's `prepare` free function (in the
//! [`super::constant_piecewise`], [`super::conservative_p0`], [`super::inverse_distance`] and
//! [`super::moving_least_squares`] submodules) builds its own coefficients and hands them to
//! [`TransferOperator::build`]; the `k`-nearest-neighbours machinery shared by the two
//! interpolation methods lives in the [`super::solver`] submodule, which never depends on this one.
//! The mesh plumbing that turns raw `(indices, weights)` point data into [`RowSparse`] blocks
//! ([`point_interpolation`], [`build_row_ptr`]) lives here next to the CSR data they assemble.

use std::collections::BTreeMap;

use ndarray as nd;
use ndarray::{Axis, concatenate};

use super::transfer_trait::{FieldNature, Transfer};
use crate::element_traits::ElementGeo;
use crate::mesh::{Dimension, ElementType, FieldOwnedD, FieldViewD, UMeshView};

/// The four transfer methods sharing the [`TransferOperator`] machinery.
///
/// The method only shapes the construction of the coefficients; at apply time it only decides
/// whether an intensive [`TransferMethod::ConservativeP0`] field is normalized by the target cell
/// measure.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum TransferMethod {
    /// Piecewise-constant copy of the containing source cell.
    ConstantPiecewise,
    /// Measure-weighted overlap average (`ConservativeP0`).
    ConservativeP0,
    /// Inverse-distance (`k`-nearest-neighbours Shepard) interpolation.
    InverseDistance,
    /// Moving least-squares interpolation.
    MovingLeastSquares,
}

/// Sparse (CSR) interpolation coefficients from the flattened source cells to the target cells of
/// a single target element type.
#[derive(Clone, Debug)]
pub(crate) struct RowSparse {
    /// Measure (area or volume) of each target cell, kept at `1.0` by the interpolation methods
    /// and filled by `ConservativeP0`; used to normalize intensive fields.
    pub(crate) target_measure: nd::Array1<f64>,
    /// Row pointer: the neighbours of target cell `j` live in `src_idx[row_ptr[j]..row_ptr[j + 1]]`.
    pub(crate) row_ptr: Vec<usize>,
    /// Global index (in the flattened source array) of each contributing source cell.
    pub(crate) src_idx: Vec<usize>,
    /// Interpolation weight / overlap measure of each contribution.
    pub(crate) weights: Vec<f64>,
}

/// Validates the space dimension of a source/target mesh pair for a transfer and returns the
/// common space dimension together with the two topological dimensions.
///
/// All transfer methods work on pairs of meshes living in the same 2D or 3D space, both non-empty.
/// Returns the common space dimension and the topological dimensions of the source and target,
/// which the callers compare against their own full-dimensionality requirements.
pub(crate) fn validated_dims(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    method: &str,
) -> (Dimension, Dimension, usize) {
    let src_space = mesh_src.space_dimension();
    let tgt_space = mesh_tgt.space_dimension();
    assert_eq!(
        src_space, tgt_space,
        "Source and target meshes should share the same space dimension, got source = {src_space}D and target = {tgt_space}D"
    );
    assert!(
        (2..=3).contains(&src_space),
        "{method} transfer is only supported in 2D and 3D space, got {src_space}D"
    );
    let src_dim = mesh_src
        .topological_dimension()
        .expect("Source mesh should not be empty");
    let tgt_dim = mesh_tgt
        .topological_dimension()
        .expect("Target mesh should not be empty");
    (src_dim, tgt_dim, src_space)
}

/// A unified sparse transfer operator, shared by all four transfer methods.
///
/// The source cells are flattened over the per-element-type field arrays in `BTreeMap` order,
/// exactly like the concatenation of the field arrays done at apply time, so the global index
/// stored in [`RowSparse::src_idx`] is valid whatever the source element types are.
#[derive(Clone, Debug)]
pub(crate) struct TransferOperator {
    method: TransferMethod,
    /// Topological dimension of the full-dimensional source cells.
    src_dim: Dimension,
    tgt_dim: Dimension,
    /// Number of source cells the operator was built from, flattened over element types in
    /// BTreeMap order.
    n_src: usize,
    /// Sparse interpolation data, grouped by target element type.
    data: Vec<(ElementType, RowSparse)>,
}

impl TransferOperator {
    /// Assembles a [`TransferOperator`] from fully built sparse data; called by the per-method
    /// `prepare` functions in the respective submodules.
    pub(crate) fn build(
        method: TransferMethod,
        src_dim: Dimension,
        tgt_dim: Dimension,
        n_src: usize,
        data: Vec<(ElementType, RowSparse)>,
    ) -> Self {
        Self {
            method,
            src_dim,
            tgt_dim,
            n_src,
            data,
        }
    }

    /// Wraps raw `(indices, weights)` interpolation data into a full single-element-type operator.
    ///
    /// This is the inverse of the raw point-to-point path: it lets tests build an operator from
    /// coordinates only and evaluate it on flat arrays.
    #[cfg(test)]
    pub(crate) fn from_neighbours(
        src_dim: Dimension,
        tgt_dim: Dimension,
        n_src: usize,
        et: ElementType,
        indices: nd::Array2<usize>,
        weights: nd::Array2<f64>,
    ) -> Self {
        let n_tgt = indices.nrows();
        let k = indices.ncols();
        let mut row_ptr = Vec::with_capacity(n_tgt + 1);
        row_ptr.push(0);
        let mut src_idx = Vec::with_capacity(n_tgt * k);
        let mut flat = Vec::with_capacity(n_tgt * k);
        for j in 0..n_tgt {
            for l in 0..k {
                src_idx.push(indices[[j, l]]);
                flat.push(weights[[j, l]]);
            }
            row_ptr.push(src_idx.len());
        }
        Self {
            method: TransferMethod::MovingLeastSquares,
            src_dim,
            tgt_dim,
            n_src,
            data: vec![(
                et,
                RowSparse {
                    target_measure: nd::Array1::ones(n_tgt),
                    row_ptr,
                    src_idx,
                    weights: flat,
                },
            )],
        }
    }

    /// Evaluates the operator on a flat source array for a single target element type.
    #[cfg(test)]
    pub(crate) fn apply_et(&self, et: ElementType, src: &nd::ArrayViewD<f64>) -> nd::ArrayD<f64> {
        let (_, rows) = self.data.iter().find(|(e, _)| *e == et).unwrap();
        self.apply_rows(rows, src, FieldNature::Extensive, 0.0)
    }

    /// Evaluates the operator on a flat source array `src` of shape `(n_src, ...)` against the
    /// sparse rows of one target element type, producing a `(n_tgt, ...)` array.
    fn apply_rows(
        &self,
        rows: &RowSparse,
        src: &nd::ArrayViewD<f64>,
        field_nature: FieldNature,
        default: f64,
    ) -> nd::ArrayD<f64> {
        let n_src = src.shape()[0];
        assert_eq!(
            n_src, self.n_src,
            "The field should have one entry per source cell, got {n_src} for {}",
            self.n_src
        );
        let n_compo = src.len() / n_src;
        let n_tgt = rows.target_measure.len();

        let mut out_shape = src.raw_dim();
        out_shape[0] = n_tgt;
        let src_view = src.view().into_shape_with_order((n_src, n_compo)).unwrap();

        let normalize =
            self.method == TransferMethod::ConservativeP0 && field_nature == FieldNature::Intensive;

        let mut tgt = nd::Array::zeros((n_tgt, n_compo));
        for j in 0..n_tgt {
            let (lo, hi) = (rows.row_ptr[j], rows.row_ptr[j + 1]);
            if lo == hi {
                tgt.row_mut(j).fill(default);
                continue;
            }
            let mut row = tgt.row_mut(j);
            for p in lo..hi {
                let w = rows.weights[p];
                nd::Zip::from(&mut row)
                    .and(src_view.row(rows.src_idx[p]))
                    .for_each(|d, &s| {
                        *d += w * s;
                    });
            }
            if normalize {
                let inv = 1.0 / rows.target_measure[j];
                row *= inv;
            }
        }

        tgt.into_shape_with_order(out_shape).unwrap()
    }
}

impl Transfer for TransferOperator {
    fn apply(&self, field: &FieldViewD, field_nature: FieldNature, default: f64) -> FieldOwnedD {
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
        let src = concatenate(Axis(0), src_views.as_slice()).unwrap();

        let mut res = BTreeMap::new();
        for (et, rows) in &self.data {
            res.insert(
                *et,
                self.apply_rows(rows, &src.view(), field_nature, default),
            );
        }
        FieldOwnedD::new(res)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
    }
}

/// Per-element-type centroids of the target mesh, flattened into a `Vec` of `(indices, weights)`
/// over the `solve` closure, one CSR block per target element type.
pub(crate) fn point_interpolation(
    mesh_tgt: &UMeshView,
    space: usize,
    solve: impl Fn(nd::ArrayView2<f64>) -> (nd::Array2<usize>, nd::Array2<f64>),
) -> Vec<(ElementType, RowSparse)> {
    let mut data = Vec::new();
    for et in mesh_tgt.element_types() {
        let tgt_coords = match space {
            2 => {
                let v: Vec<f64> = mesh_tgt
                    .elements_of_type(*et)
                    .flat_map(|e| e.centroid2().into_iter())
                    .collect();
                nd::Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 2), v).unwrap()
            }
            3 => {
                let v: Vec<f64> = mesh_tgt
                    .elements_of_type(*et)
                    .flat_map(|e| e.centroid3().into_iter())
                    .collect();
                nd::Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 3), v).unwrap()
            }
            _ => unreachable!(),
        };
        let (ind, wei) = solve(tgt_coords.view());
        data.push((
            *et,
            RowSparse {
                target_measure: nd::Array1::ones(ind.nrows()),
                row_ptr: build_row_ptr(&ind),
                src_idx: ind.as_slice().unwrap().to_vec(),
                weights: wei.as_slice().unwrap().to_vec(),
            },
        ));
    }
    data
}

/// Row pointers of a dense `(n_tgt × k)` neighbour array stored in row-major order.
pub(crate) fn build_row_ptr(indices: &nd::Array2<usize>) -> Vec<usize> {
    let n_tgt = indices.nrows();
    let k = indices.ncols();
    let mut row_ptr = Vec::with_capacity(n_tgt + 1);
    row_ptr.push(0);
    for j in 0..n_tgt {
        row_ptr.push((j + 1) * k);
    }
    row_ptr
}
