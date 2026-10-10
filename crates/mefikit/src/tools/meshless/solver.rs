//! Point-level solvers shared by the meshless reconstruction operators.
//!
//! Every meshless operator reduces to the same raw computation: a `k`-nearest-neighbours search
//! over the source points, followed by a per-target-point weighted local fit. This module gathers
//! the shared machinery:
//!
//! - [`nearest_neighbours`]: the `k`-nearest-neighbours search returning `(indices, r²)`.
//! - [`kernel_weights`]: the [`DistanceWeighting`] kernel evaluated at the `h`-scaled distances.
//! - [`affine_fit`] (macro-generated): the weighted degree-1 least-squares normal equations.
//! - [`solve_neighbours`]: the value-interpolation weights (transfer).
//! - [`solve_gradient_neighbours`]: the gradient weights (gradient tool).
//!
//! The raw solvers work on plain `n × D` coordinate arrays and never depend on the mesh API or on
//! any operator; future meshless operators (divergence, laplacian, ...) can add their own weight
//! derivation here and reuse the same search and fit.

use std::num::NonZero;

use kiddo::{ImmutableKdTree, dist::SquaredEuclidean};
use nalgebra as na;
use ndarray as nd;

use super::kernel::DistanceWeighting;

/// The flavour of per-target-point interpolation weights computed by [`solve_neighbours`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum NeighbourScheme {
    /// Inverse-distance weights `w_i ∝ 1 / r_i^exponent`.
    InverseDistance { exponent: f64 },
    /// Weighted degree-1 least-squares fit with a [`DistanceWeighting`] kernel.
    MovingLeastSquares { weighting: DistanceWeighting },
}

/// Computes the `k` nearest source points of every target point.
///
/// Returns `(indices, r2)`, both of shape `(n_tgt, k)`: the global source indices and the
/// **squared** Euclidean distances (the metric used internally by the kd-tree). A target point
/// coinciding with a source point has a zero squared distance in its row.
///
/// # Panics
///
/// - If `src_coords` and `tgt_coords` do not share the same space dimension, or if it is not 2
///   or 3.
/// - If `k` is zero or larger than the number of source points.
pub(crate) fn nearest_neighbours(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
) -> (nd::Array2<usize>, nd::Array2<f64>) {
    let dim = src_coords.ncols();
    assert_eq!(
        dim,
        tgt_coords.ncols(),
        "Source and target coordinates should have the same number of columns, got source = {dim} and target = {}",
        tgt_coords.ncols()
    );
    assert!(
        (2..=3).contains(&dim),
        "Meshless reconstruction is only supported in 2D and 3D space, got {dim}D"
    );
    assert!(k > 0, "k should be at least 1");

    let n_tgt = tgt_coords.nrows();
    let mut indices = nd::Array2::<usize>::zeros((n_tgt, k));
    let mut r2 = nd::Array2::<f64>::zeros((n_tgt, k));

    match dim {
        2 => nearest_points::<2>(src_coords, tgt_coords, k, &mut indices, &mut r2),
        3 => nearest_points::<3>(src_coords, tgt_coords, k, &mut indices, &mut r2),
        _ => unreachable!(),
    }

    (indices, r2)
}

/// Fills the `(n_tgt, k)` index and squared-distance arrays for one space dimension.
fn nearest_points<const D: usize>(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    indices: &mut nd::Array2<usize>,
    r2: &mut nd::Array2<f64>,
) {
    let tree = ImmutableKdTree::new_from_slice(as_points::<D>(src_coords)).unwrap();

    for (j, p) in tgt_coords.outer_iter().enumerate() {
        let query: [f64; D] = std::array::from_fn(|c| p[c]);
        let nn = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).unwrap())
            .execute();

        for (l, r) in nn.iter().enumerate() {
            indices[[j, l]] = r.item as usize;
            r2[[j, l]] = r.distance;
        }
    }
}

/// Evaluates a [`DistanceWeighting`] kernel on the squared distances of one target point.
///
/// Returns the normalized kernel weights and the characteristic length `h` (the distance to the
/// farthest selected neighbour). The global scale of the weights cancels in the least-squares
/// solve, so the largest weight is brought back to `1` to keep the system well conditioned. When
/// the kernel is not finite (e.g. an inverse-distance kernel evaluated at zero) the weights are
/// returned untouched so the caller can fall back.
pub(crate) fn kernel_weights(r2: &[f64], weighting: DistanceWeighting) -> (Vec<f64>, f64) {
    let h = r2.iter().copied().fold(0.0_f64, f64::max).sqrt();
    let h2 = h * h;
    let mut w_kernel: Vec<f64> = r2.iter().map(|&r2| weighting.kernel(r2 / h2)).collect();

    if w_kernel.iter().all(|w| w.is_finite()) {
        let w_max = w_kernel.iter().copied().fold(0.0_f64, f64::max);
        if w_max > 0.0 {
            for w in &mut w_kernel {
                *w /= w_max;
            }
        }
    }

    (w_kernel, h)
}

/// Computes the interpolation weights for every target point of an `n × D` pair of point sets.
///
/// This is the value-interpolation solver used by the inverse-distance and moving least-squares
/// transfers: each target point is a weighted sum `sum_i w_i f(x_i)` of its `k` nearest source
/// points. A target point coinciding with a source point reproduces that source value exactly.
///
/// # Panics
///
/// - If `src_coords` and `tgt_coords` do not share the same space dimension, or if it is not 2
///   or 3.
/// - If `k` is zero or larger than the number of source points.
pub(crate) fn solve_neighbours(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    scheme: NeighbourScheme,
) -> (nd::Array2<usize>, nd::Array2<f64>) {
    let (indices, r2) = nearest_neighbours(src_coords, tgt_coords, k);
    let n_tgt = tgt_coords.nrows();
    let dim = src_coords.ncols();
    let mut weights = nd::Array2::<f64>::zeros((n_tgt, k));

    for j in 0..n_tgt {
        let nb_idx: Vec<usize> = indices.row(j).to_vec();
        let r2_row: Vec<f64> = r2.row(j).to_vec();

        // A target point coinciding with a source point interpolates it exactly.
        if let Some(l) = r2_row.iter().position(|&r2| r2 == 0.0) {
            weights[[j, l]] = 1.0;
            continue;
        }

        match scheme {
            NeighbourScheme::InverseDistance { exponent } => {
                let sum: f64 = r2_row.iter().map(|&r2| r2.powf(-0.5 * exponent)).sum();
                for (l, &r2) in r2_row.iter().enumerate() {
                    weights[[j, l]] = r2.powf(-0.5 * exponent) / sum;
                }
            }
            NeighbourScheme::MovingLeastSquares { weighting } => {
                let (w_kernel, h) = kernel_weights(&r2_row, weighting);
                let w_interp = local_fit_value_weights(
                    src_coords,
                    &nb_idx,
                    &tgt_coords.row(j),
                    &w_kernel,
                    h,
                    dim,
                );
                let w_interp = match w_interp {
                    Some(w) => w,
                    None => {
                        // Degenerate local system: fall back to normalized kernel weights
                        // (Shepard's method), then to a plain average.
                        let sum: f64 = w_kernel.iter().filter(|w| w.is_finite()).sum();
                        if sum > 0.0 {
                            w_kernel
                                .iter()
                                .map(|&w| if w.is_finite() { w / sum } else { 0.0 })
                                .collect()
                        } else {
                            vec![1.0 / k as f64; k]
                        }
                    }
                };
                for (l, w) in w_interp.into_iter().enumerate() {
                    weights[[j, l]] = w;
                }
            }
        }
    }

    (indices, weights)
}

/// Computes the gradient weights for every target point of an `n × D` pair of point sets.
///
/// Returns `(indices, grads)` with shapes `(n_tgt, k)` and `(n_tgt, k, D)`: the gradient component
/// `a` at target point `j` is `sum_l grads[j, l, a] * f(src_indices[j, l])`. The weights are those
/// of the derivative of the weighted degree-1 least-squares fit, i.e. the linear part of the fit
/// divided by the characteristic length `h`.
///
/// A row stays zero when the local system is degenerate (rank-deficient geometry or all `k`
/// neighbours coinciding with the target point), so building never fails.
///
/// # Panics
///
/// - If `src_coords` and `tgt_coords` do not share the same space dimension, or if it is not 2
///   or 3.
/// - If `k` is zero or larger than the number of source points.
pub(crate) fn solve_gradient_neighbours(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    weighting: DistanceWeighting,
) -> (nd::Array2<usize>, nd::Array3<f64>) {
    let (indices, r2) = nearest_neighbours(src_coords, tgt_coords, k);
    let n_tgt = tgt_coords.nrows();
    let dim = src_coords.ncols();
    let mut grads = nd::Array3::<f64>::zeros((n_tgt, k, dim));

    for j in 0..n_tgt {
        let nb_idx: Vec<usize> = indices.row(j).to_vec();
        let r2_row: Vec<f64> = r2.row(j).to_vec();
        let (mut w_kernel, h) = kernel_weights(&r2_row, weighting);

        // An evaluation point coinciding with a source point makes an inverse-distance kernel
        // singular (`r = 0`). The affine fit is exact for any positive weights, so replace the
        // singular weights by the largest finite one to keep the normal equations finite while
        // still reproducing affine fields exactly.
        if !w_kernel.iter().all(|w| w.is_finite()) {
            let finite_max = w_kernel
                .iter()
                .copied()
                .filter(|w| w.is_finite())
                .fold(0.0_f64, f64::max);
            let fallback = if finite_max > 0.0 { finite_max } else { 1.0 };
            for w in &mut w_kernel {
                if !w.is_finite() {
                    *w = fallback;
                }
            }
        }

        // Zero gradient for a degenerate local geometry.
        if h <= 0.0 {
            continue;
        }

        let p = tgt_coords.row(j);
        for a in 0..dim {
            let Some(coeff) = affine_fit(src_coords, &nb_idx, &p, &w_kernel, h, a + 1, dim) else {
                continue;
            };
            if !coeff.iter().all(|c| c.is_finite()) {
                continue;
            }
            for (l, &i) in nb_idx.iter().enumerate() {
                grads[[j, l, a]] = w_kernel[l] * shape_value(&coeff, src_coords, i, &p, h, dim) / h;
            }
        }
    }

    (indices, grads)
}

/// Builds the value-interpolation weights from the affine fit of one target point.
///
/// Returns `None` when the local system is rank-deficient or non-finite, so the caller can fall
/// back. The returned weights are `w_l = w_kernel[l] · (c0 + c·u_l)` where `c` is the fit of the
/// value at the evaluation point.
fn local_fit_value_weights(
    src_coords: &nd::ArrayView2<f64>,
    nb_idx: &[usize],
    p: &nd::ArrayView1<f64>,
    w_kernel: &[f64],
    h: f64,
    dim: usize,
) -> Option<Vec<f64>> {
    if !w_kernel.iter().all(|w| w.is_finite()) || h <= 0.0 {
        return None;
    }
    let coeff = affine_fit(src_coords, nb_idx, p, w_kernel, h, 0, dim)?;
    if !coeff.iter().all(|c| c.is_finite()) {
        return None;
    }
    Some(
        nb_idx
            .iter()
            .enumerate()
            .map(|(l, &i)| w_kernel[l] * shape_value(&coeff, src_coords, i, p, h, dim))
            .collect(),
    )
}

/// Evaluates the fitted affine polynomial `c0 + sum_a c[a+1] · (x_a - p_a) / h` at source point `i`.
fn shape_value(
    coeff: &[f64],
    src_coords: &nd::ArrayView2<f64>,
    i: usize,
    p: &nd::ArrayView1<f64>,
    h: f64,
    dim: usize,
) -> f64 {
    let mut value = coeff[0];
    for a in 0..dim {
        value += (src_coords[[i, a]] - p[a]) / h * coeff[a + 1];
    }
    value
}

/// Solves the weighted affine normal equations for one unit right-hand side.
///
/// Dispatches to the fixed-size 2D/3D solvers (generated by [`affine_fit!`]) and returns the
/// `dim + 1` fit coefficients `[c0, c1, .., c_dim]`, or `None` when the system is rank-deficient.
fn affine_fit(
    src_coords: &nd::ArrayView2<f64>,
    nb_idx: &[usize],
    p: &nd::ArrayView1<f64>,
    w_kernel: &[f64],
    h: f64,
    rhs: usize,
    dim: usize,
) -> Option<Vec<f64>> {
    match dim {
        2 => affine_fit_2d(src_coords, nb_idx, p, w_kernel, h, rhs).map(|c| c.to_vec()),
        3 => affine_fit_3d(src_coords, nb_idx, p, w_kernel, h, rhs).map(|c| c.to_vec()),
        _ => unreachable!(),
    }
}

/// Generates the weighted degree-1 least-squares normal-equations solver for a space dimension.
///
/// The fit is computed on centered, `h`-normalized coordinates `u_i = (x_i - p) / h`, which put
/// the evaluation point at the origin. With `A` the `k × (d+1)` design matrix whose rows are
/// `[1, u]` and `W` the diagonal kernel-weight matrix, the `(d+1) × (d+1)` normal-equations system
/// `G y = e_rhs` with `G = Aᵀ W A` is solved by the SVD-based least-squares solver of the `lstsq`
/// crate. Rank deficiency (`rank < d + 1`) returns `None`. The 2D and 3D cases are generated
/// separately so the system is a fixed-size static matrix.
macro_rules! affine_fit {
    ($name:ident, $mat:ident, $spatial:expr) => {
        fn $name(
            src_coords: &nd::ArrayView2<f64>,
            nb_idx: &[usize],
            p: &nd::ArrayView1<f64>,
            w_kernel: &[f64],
            h: f64,
            rhs: usize,
        ) -> Option<[f64; $spatial + 1]> {
            let d = $spatial;
            let mut u = [0.0; $spatial];
            let mut gram = na::$mat::zeros();
            for (l, &i) in nb_idx.iter().enumerate() {
                let w = w_kernel[l];
                for a in 0..d {
                    u[a] = (src_coords[[i, a]] - p[a]) / h;
                }
                gram[(0, 0)] += w;
                for a in 0..d {
                    gram[(0, a + 1)] += w * u[a];
                    gram[(a + 1, 0)] += w * u[a];
                }
                for a in 0..d {
                    for b in 0..d {
                        gram[(a + 1, b + 1)] += w * u[a] * u[b];
                    }
                }
            }

            let mut b = na::SVector::<f64, { $spatial + 1 }>::zeros();
            b[rhs] = 1.0;

            let solve = lstsq::lstsq(&gram, &b, 1e-12).ok()?;
            if solve.rank < $spatial + 1 {
                return None;
            }
            Some(std::array::from_fn(|i| solve.solution[i]))
        }
    };
}

affine_fit!(affine_fit_2d, Matrix3, 2);
affine_fit!(affine_fit_3d, Matrix4, 3);

/// Views an `n × D` contiguous coordinate array as a slice of `[f64; D]` points.
fn as_points<'a, const D: usize>(coords: &nd::ArrayView2<'a, f64>) -> &'a [[f64; D]] {
    assert_eq!(coords.ncols(), D);
    let slice = coords.as_slice().expect("coordinates should be contiguous");

    // Safety:
    // - the slice length is a multiple of D
    // - f64 is properly aligned
    // - `[f64; D]` is a contiguous run of D f64 values
    let len = slice.len() / D;
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [f64; D], len) }
}
