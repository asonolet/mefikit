//! Shared solver utilities for the point-to-point transfer precomputations.
//!
//! The inverse-distance and moving least-squares transfer methods both reduce to the same raw
//! computation: a `k`-nearest-neighbours search over the source points followed by a per-target
//! point weight computation. This module gathers the helpers reused by both (and by some tests):
//!
//! - [`DistanceWeighting`], the kernel converting a distance into a fit weight.
//! - [`NeighbourScheme`], the selector between inverse-distance and moving least-squares weights.
//! - [`solve_neighbours`], the raw `k`-NN solver producing `(indices, weights)`.
//!
//! The raw solver works on plain `n × D` coordinate arrays and returns dense `(n_tgt × k)`
//! `(indices, weights)`; the per-method `prepare` functions feed it through the
//! [`crate::tools::transfer::operator::point_interpolation`] glue, which turns that output into the
//! per-element-type [`crate::tools::transfer::operator::RowSparse`] blocks of the
//! [`crate::tools::transfer::operator::TransferOperator`]. This module never depends on the
//! operator module: the two are only joined by the method modules.

use std::num::NonZero;

use kiddo::{ImmutableKdTree, dist::SquaredEuclidean};
use nalgebra as na;
use ndarray as nd;
use ndarray::{Axis, concatenate};

use crate::mesh::{Dimension, UMeshView};
use crate::tools::centroids::centroids;

/// How the distance from a target interpolation point to its `k` nearest source points is turned
/// into a weight of the local (moving) least-squares fit.
///
/// All kernels are written as a function of `s^2 = (r / h)^2`, where `r` is the distance from the
/// target point to a source point and `h` is a characteristic length chosen per target point (the
/// distance to its farthest selected neighbour).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceWeighting {
    /// All selected neighbours get the same weight: a plain linear least-squares fit.
    Constant,
    /// Inverse-distance weight `w = (h / r)^exponent` (`2.0` gives the usual inverse squared
    /// distance).
    InverseDistance { exponent: f64 },
    /// Compact-support moving least-squares kernel `w = (1 - s^2)^exponent` for `s < 1`, zero
    /// otherwise (`exponent = 2` or `3` are the usual choices).
    CompactSupport { exponent: f64 },
    /// Gaussian kernel `w = exp(-s^2)`.
    Gaussian,
}

impl DistanceWeighting {
    /// Evaluates the kernel on the squared scaled distance `s2 = (r / h)^2`.
    pub(crate) fn kernel(self, s2: f64) -> f64 {
        match self {
            Self::Constant => 1.0,
            Self::InverseDistance { exponent } => s2.powf(-0.5 * exponent),
            Self::CompactSupport { exponent } => {
                if s2 >= 1.0 {
                    0.0
                } else {
                    (1.0 - s2).powf(exponent)
                }
            }
            Self::Gaussian => (-s2).exp(),
        }
    }
}

/// The flavour of per-target-point interpolation weights computed by [`solve_neighbours`].
#[derive(Clone, Copy, Debug)]
pub(crate) enum NeighbourScheme {
    /// Inverse-distance weights `w_i ∝ 1 / r_i^exponent`.
    InverseDistance { exponent: f64 },
    /// Weighted degree-1 least-squares fit with a [`DistanceWeighting`] kernel.
    MovingLeastSquares { weighting: DistanceWeighting },
}

/// Centroids of the source cells of topological dimension `src_dim`, flattened in the same
/// `BTreeMap` order used to flatten the source field arrays, together with the total point count.
///
/// # Panics
///
/// If `src_dim` is neither 2 nor 3, or if the source mesh has no cell of that dimension.
pub(crate) fn source_centroids(
    mesh_src: &UMeshView,
    src_dim: Dimension,
) -> (nd::Array2<f64>, usize) {
    let src_coords = centroids(mesh_src, Some(src_dim));
    let src_views: Vec<_> = src_coords.values().map(|a| a.view()).collect();
    let src_coords = concatenate(Axis(0), src_views.as_slice()).unwrap();
    let n_src = src_coords.nrows();
    (src_coords, n_src)
}

/// Computes the interpolation coefficients for every target point of an `n × D` pair of
/// point sets, together with the interpolation scheme.
///
/// This is the raw solver over plain coordinate arrays, without any mesh: it is used by the
/// [`crate::tools::transfer::inverse_distance::prepare`] and
/// [`crate::tools::transfer::moving_least_squares::prepare`] constructors and by the tests.
///
/// # Panics
///
/// - If `src_coords` and `tgt_coords` do not share the same space dimension, or if it is not 2
///   or 3.
/// - If `k` is zero.
pub(crate) fn solve_neighbours(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    scheme: NeighbourScheme,
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
        "Moving least-squares interpolation is only supported in 2D and 3D space, got {dim}D"
    );
    assert!(k > 0, "k should be at least 1");

    let n_tgt = tgt_coords.nrows();
    let mut indices = nd::Array2::<usize>::zeros((n_tgt, k));
    let mut weights = nd::Array2::<f64>::zeros((n_tgt, k));

    match dim {
        2 => solve_points::<2>(
            src_coords,
            tgt_coords,
            k,
            scheme,
            &mut indices,
            &mut weights,
        ),
        3 => solve_points::<3>(
            src_coords,
            tgt_coords,
            k,
            scheme,
            &mut indices,
            &mut weights,
        ),
        _ => unreachable!(),
    }

    (indices, weights)
}

/// Computes the interpolation weights for every target point of an `n × D` coordinate pair.
fn solve_points<const D: usize>(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    scheme: NeighbourScheme,
    indices: &mut nd::Array2<usize>,
    weights: &mut nd::Array2<f64>,
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
        }

        // A target point coinciding with a source point interpolates it exactly.
        if let Some(l) = nn.iter().position(|r| r.distance == 0.0) {
            weights[[j, l]] = 1.0;
            continue;
        }

        match scheme {
            NeighbourScheme::InverseDistance { exponent } => {
                let mut sum = 0.0;
                for r in &nn {
                    sum += r.distance.powf(-0.5 * exponent);
                }
                for (l, r) in nn.iter().enumerate() {
                    weights[[j, l]] = r.distance.powf(-0.5 * exponent) / sum;
                }
            }
            NeighbourScheme::MovingLeastSquares { weighting } => {
                let nb_idx: Vec<usize> = nn.iter().map(|r| r.item as usize).collect();
                let r2: Vec<f64> = nn.iter().map(|r| r.distance).collect();

                let r_max = r2.iter().copied().fold(0.0_f64, f64::max).sqrt();

                // `h` is the distance to the farthest selected neighbour, computed once: no attempt
                // is made to inflate it until the local system becomes full rank. If the weighted
                // fit fails (degenerate geometry, or a compact kernel dropping a neighbour out of
                // its support), the solution degrades immediately to the Shepard fallback below.
                let h = r_max;
                let h2 = h * h;
                let mut w_kernel: Vec<f64> =
                    r2.iter().map(|&r2| weighting.kernel(r2 / h2)).collect();
                let finite = w_kernel.iter().all(|w| w.is_finite());
                if finite {
                    // The global scale of the kernel weights cancels in the least-squares solve;
                    // bringing the largest weight to 1 keeps the system well conditioned.
                    let w_max = w_kernel.iter().copied().fold(0.0_f64, f64::max);
                    if w_max > 0.0 {
                        for w in &mut w_kernel {
                            *w /= w_max;
                        }
                    }
                }

                let w_interp = if finite {
                    let result = match D {
                        2 => solve_normal_2d(src_coords, &nb_idx, &p, &w_kernel, h),
                        3 => solve_normal_3d(src_coords, &nb_idx, &p, &w_kernel, h),
                        _ => unreachable!(),
                    };
                    match result {
                        Some(w) if w.iter().all(|w| w.is_finite()) => Some(w),
                        _ => None,
                    }
                } else {
                    None
                };

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
}

/// Fits `a + b·x` to the selected source points, weighted by `w_kernel`, and returns the
/// interpolation weights such that the value at `p` is a weighted sum of the neighbour values.
///
/// The fit is computed on centered, h-normalized coordinates `u_i = (x_i - p) / h`, which put the
/// evaluation point at the origin. With `A` the `k × (d+1)` design matrix whose rows are `[1, u]`
/// and `W` the diagonal kernel-weight matrix, the interpolation weights are `w = W A y` where `y`
/// solves the weighted least-squares problem `W^0.5 A y = W^0.5 e_0` (here `e_0 = [1, 0, ...]`
/// since the evaluation point is the origin). The `(d+1) × (d+1)` normal-equations system `G y =
/// e_0` with `G = Aᵀ W A = (W^0.5 A)ᵀ (W^0.5 A)` is solved by the SVD-based least-squares solver of
/// the `lstsq` crate, which is allocation-free on the static matrix and signals rank deficiency
/// through `rank < d + 1`; `None` is then returned so the caller can fall back. The 2D and 3D
/// cases are generated separately so that the system is a fixed-size static matrix.
macro_rules! solve_normal {
    ($name:ident, $mat:ident, $spatial:expr) => {
        fn $name(
            src_coords: &nd::ArrayView2<f64>,
            nb_idx: &[usize],
            p: &nd::ArrayView1<f64>,
            w_kernel: &[f64],
            h: f64,
        ) -> Option<Vec<f64>> {
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

            let mut rhs = na::SVector::<f64, { $spatial + 1 }>::zeros();
            rhs[0] = 1.0;

            let solve = lstsq::lstsq(&gram, &rhs, 1e-12).ok()?;
            if solve.rank < $spatial + 1 {
                return None;
            }
            let coeff = solve.solution;

            let mut w = vec![0.0; nb_idx.len()];
            for (l, &i) in nb_idx.iter().enumerate() {
                let mut ac = coeff[0];
                for a in 0..d {
                    ac += ((src_coords[[i, a]] - p[a]) / h) * coeff[a + 1];
                }
                w[l] = w_kernel[l] * ac;
            }
            Some(w)
        }
    };
}

solve_normal!(solve_normal_2d, Matrix3, 2);
solve_normal!(solve_normal_3d, Matrix4, 3);

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
