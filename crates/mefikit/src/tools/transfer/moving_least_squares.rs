use std::collections::BTreeMap;
use std::num::NonZero;

use kiddo::{ImmutableKdTree, dist::SquaredEuclidean};
use nalgebra as na;
use ndarray::{Array2, ArrayD, ArrayView1, ArrayView2, ArrayViewD, Axis, Zip, concatenate};

use crate::element_traits::ElementGeo;
use crate::mesh::{Dimension, ElementType, FieldOwnedD, FieldViewD, UMeshView};
use crate::tools::centroids::centroids;
use crate::tools::{FieldNature, Transfer};

/// How the distance from a target interpolation point to its `k` nearest source points is turned
/// into a weight of the local (moving) least-squares fit.
///
/// All kernels are written as a function of `s^2 = (r / h)^2`, where `r` is the distance from the
/// target point to a source point and `h` is a characteristic length chosen per target point (the
/// distance to its farthest selected neighbour, inflated as needed so that every neighbour takes
/// part in the fit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceWeighting {
    /// All selected neighbours get the same weight: a plain linear least-squares fit.
    None,
    /// Inverse-distance weight `w = (h / r)^exponent` (`2.0` gives the usual inverse squared
    /// distance).
    InverseDistance { exponent: f64 },
    /// Compact-support moving least squares kernel `w = (1 - s^2)^exponent` for `s < 1`, zero
    /// otherwise (`exponent = 2` or `3` are the usual choices).
    CompactSupport { exponent: f64 },
    /// Gaussian kernel `w = exp(-s^2)`.
    Gaussian,
}

impl DistanceWeighting {
    /// Evaluates the kernel on the squared scaled distance `s2 = (r / h)^2`.
    fn kernel(self, s2: f64) -> f64 {
        match self {
            Self::None => 1.0,
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

/// A moving least-squares interpolation transfer between point clouds.
///
/// The operator is built from the coordinates only, so it can be reused to evaluate many fields
/// (e.g. across time steps) as long as the point sets do not change.
#[derive(Clone, Debug)]
pub struct MovingLeastSquaresTransfer {
    tgt_dim: Dimension,
    /// Number of source points the operator was built from.
    n_src: usize,
    /// Indices of the `k` source points used for each target point. Shape `(n_tgt × k)`.
    /// They are grouped by element type.
    indices: Vec<(ElementType, Array2<usize>)>,
    /// Interpolation weights, one per selected source point. Shape `(n_tgt × k)`.
    /// They are grouped by element type.
    weights: Vec<(ElementType, Array2<f64>)>,
}

impl MovingLeastSquaresTransfer {
    /// Builds a moving least-squares interpolation operator from source points to target points.
    ///
    /// For each target point the `k` nearest source points are gathered and a degree-1 polynomial
    /// is fitted through them by weighted least squares, the weights coming from
    /// [`DistanceWeighting`]. The fit is evaluated at the target point, which yields `k`
    /// interpolation weights `w_i` such that the transferred value is `sum_i w_i f(x_i)`. Affine
    /// fields are reproduced exactly when the local system is full rank; degenerate local systems
    /// fall back to the normalized kernel weights (Shepard's method), then to a plain average, so
    /// building the operator never fails. The characteristic length `h` of the weighting kernel is
    /// the distance from the target point to its farthest selected neighbour, inflated until every
    /// neighbour takes part in the fit.
    ///
    /// # Panics
    ///
    /// - If `src_coords` and `tgt_coords` do not share the same space dimension, or if it is not 2
    ///   or 3.
    /// - If `k` is zero.
    fn from_coords(
        src_coords: &ArrayView2<f64>,
        tgt_coords: &ArrayView2<f64>,
        k: usize,
        weighting: DistanceWeighting,
    ) -> (Array2<usize>, Array2<f64>) {
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
        let mut indices = Array2::<usize>::zeros((n_tgt, k));
        let mut weights = Array2::<f64>::zeros((n_tgt, k));

        match dim {
            2 => solve_points::<2>(
                src_coords,
                tgt_coords,
                k,
                weighting,
                &mut indices,
                &mut weights,
            ),
            3 => solve_points::<3>(
                src_coords,
                tgt_coords,
                k,
                weighting,
                &mut indices,
                &mut weights,
            ),
            _ => unreachable!(),
        }

        (indices, weights)
    }

    pub fn new(
        mesh_src: &UMeshView,
        mesh_tgt: &UMeshView,
        k: usize,
        weighting: DistanceWeighting,
    ) -> Self {
        let src_coords = centroids(mesh_src, None);
        let src_view: Vec<_> = src_coords.values().map(|a| a.view()).collect();
        let src_coords = concatenate(Axis(0), src_view.as_slice()).unwrap();
        let tgt_dim = mesh_tgt
            .topological_dimension()
            .expect("Target mesh should not be empty");

        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for et in mesh_tgt.element_types() {
            let tgt_coords = match mesh_tgt.space_dimension() {
                2 => {
                    let v: Vec<f64> = mesh_tgt
                        .elements_of_type(*et)
                        .flat_map(|e| e.centroid2().into_iter())
                        .collect();
                    Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 2), v).unwrap()
                }
                _ => todo!(),
            };
            let (ind, wei) =
                Self::from_coords(&src_coords.view(), &tgt_coords.view(), k, weighting);
            indices.push((*et, ind));
            weights.push((*et, wei));
        }
        Self {
            tgt_dim,
            n_src: mesh_src.num_elements(),
            indices,
            weights,
        }
    }

    /// Evaluates the operator on a field defined on the source points.
    ///
    /// `src_field` has shape `(n_src, ...)`; the result has the same trailing shape with the first
    /// dimension replaced by the number of target points.
    fn apply_on_array(&self, et: ElementType, src_field: &ArrayViewD<f64>) -> ArrayD<f64> {
        let n_src = src_field.shape()[0];
        assert_eq!(
            n_src, self.n_src,
            "The field should have one entry per source point, got {n_src} for {}",
            self.n_src
        );

        let n_compo = src_field.len() / n_src;

        let i = self
            .indices
            .iter()
            .enumerate()
            .find(|(_, (e, _))| *e == et)
            .unwrap()
            .0;

        let indices = &self.indices[i].1;
        let weights = &self.weights[i].1;

        let n_tgt = indices.nrows();
        let k = indices.ncols();

        let mut tgt = Array2::<f64>::zeros((n_tgt, n_compo));
        let src_view = src_field
            .view()
            .into_shape_with_order((n_src, n_compo))
            .unwrap();

        for j in 0..n_tgt {
            for l in 0..k {
                let i = indices[[j, l]];
                let w = weights[[j, l]];
                // accumulate the whole row
                Zip::from(tgt.row_mut(j))
                    .and(src_view.row(i))
                    .for_each(|d, &s| {
                        *d += w * s;
                    });
            }
        }

        let mut tgt_shape = src_field.raw_dim();
        tgt_shape[0] = n_tgt;
        tgt.into_shape_with_order(tgt_shape).unwrap()
    }
}

/// Computes the interpolation weights for every target point.
fn solve_points<const D: usize>(
    src_coords: &ArrayView2<f64>,
    tgt_coords: &ArrayView2<f64>,
    k: usize,
    weighting: DistanceWeighting,
    indices: &mut Array2<usize>,
    weights: &mut Array2<f64>,
) {
    let tree = ImmutableKdTree::new_from_slice(as_points::<D>(src_coords)).unwrap();

    for (j, p) in tgt_coords.outer_iter().enumerate() {
        let query: [f64; D] = std::array::from_fn(|c| p[c]);
        let nn = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).unwrap())
            .execute();

        let nb_idx: Vec<usize> = nn.iter().map(|r| r.item as usize).collect();
        let r2: Vec<f64> = nn.iter().map(|r| r.distance).collect();

        for (l, &i) in nb_idx.iter().enumerate() {
            indices[[j, l]] = i;
        }

        // A target point coinciding with a source point interpolates it exactly.
        if let Some(l) = r2.iter().position(|&r2| r2 == 0.0) {
            weights.row_mut(j).fill(0.0);
            weights[[j, l]] = 1.0;
            continue;
        }

        let r_max = r2.iter().copied().fold(0.0_f64, f64::max).sqrt();

        // Start with h = max(r) and grow it until the local weighted system is full rank. This
        // keeps the farthest neighbour inside the support of compact kernels (where it would
        // otherwise carry zero weight and drop out of the fit) and never fails: a genuinely
        // degenerate geometry still ends up on the Shepard fallback below.
        let mut h = r_max;
        let mut w_interp = None;
        let mut w_kernel = Vec::new();
        for _ in 0..12 {
            let h2 = h * h;
            w_kernel = r2.iter().map(|&r2| weighting.kernel(r2 / h2)).collect();
            if !w_kernel.iter().all(|w| w.is_finite()) {
                break;
            }
            // The global scale of the kernel weights cancels in the least-squares solve; bringing
            // the largest weight to 1 keeps the system well conditioned.
            let w_max = w_kernel.iter().copied().fold(0.0_f64, f64::max);
            if w_max > 0.0 {
                for w in &mut w_kernel {
                    *w /= w_max;
                }
            }
            let result = match D {
                2 => solve_normal_2d(src_coords, &nb_idx, &p, &w_kernel, h),
                3 => solve_normal_3d(src_coords, &nb_idx, &p, &w_kernel, h),
                _ => unreachable!(),
            };
            match result {
                Some(w) if w.iter().all(|w| w.is_finite()) => {
                    w_interp = Some(w);
                    break;
                }
                _ => h *= 1.1,
            }
        }

        let w_interp = match w_interp {
            Some(w) => w,
            None => {
                // Degenerate local system: fall back to normalized kernel weights (Shepard's
                // method), then to a plain average.
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
            src_coords: &ArrayView2<f64>,
            nb_idx: &[usize],
            p: &ArrayView1<f64>,
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
fn as_points<'a, const D: usize>(coords: &ArrayView2<'a, f64>) -> &'a [[f64; D]] {
    assert_eq!(coords.ncols(), D);
    let slice = coords.as_slice().expect("coordinates should be contiguous");

    // Safety:
    // - the slice length is a multiple of D
    // - f64 is properly aligned
    // - `[f64; D]` is a contiguous run of D f64 values
    let len = slice.len() / D;
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [f64; D], len) }
}

impl Transfer for MovingLeastSquaresTransfer {
    fn apply(&self, field: &FieldViewD, _field_nature: FieldNature, _default: f64) -> FieldOwnedD {
        let mut res = BTreeMap::new();
        for (et, a) in &field.0 {
            res.insert(*et, self.apply_on_array(*et, a));
        }
        FieldOwnedD::new(res)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    // use ndarray as nd;

    // fn src_grid_3d() -> nd::Array2<f64> {
    //     let mut pts = Vec::new();
    //     for z in 0..=4 {
    //         for y in 0..=4 {
    //             for x in 0..=4 {
    //                 pts.extend([x as f64 * 0.25, y as f64 * 0.25, z as f64 * 0.25]);
    //             }
    //         }
    //     }
    //     nd::Array2::from_shape_vec((125, 3), pts).unwrap()
    // }

    #[test]
    fn kernel_values() {
        assert_eq!(DistanceWeighting::None.kernel(0.25), 1.0);
        let inv = DistanceWeighting::InverseDistance { exponent: 2.0 };
        assert_relative_eq!(inv.kernel(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(inv.kernel(4.0), 0.25, epsilon = 1e-12);
        let compact = DistanceWeighting::CompactSupport { exponent: 2.0 };
        assert_relative_eq!(compact.kernel(0.25), 0.5625, epsilon = 1e-12);
        assert_eq!(compact.kernel(1.0), 0.0);
        assert_eq!(compact.kernel(2.0), 0.0);
        let gaussian = DistanceWeighting::Gaussian;
        assert_relative_eq!(gaussian.kernel(1.0), (-1.0_f64).exp(), epsilon = 1e-12);
    }

    // #[test]
    // fn reproduces_affine_field_exactly_3d() {
    //     let src = nd::array![
    //         [0.0, 0.0, 0.0],
    //         [1.0, 0.0, 0.0],
    //         [0.0, 1.0, 0.0],
    //         [0.0, 0.0, 1.0]
    //     ];
    //     let tgt = nd::array![[0.25, 0.25, 0.25], [0.1, 0.2, 0.3]];
    //     // f = 1 + 2x - 3y + 4z
    //     let field = nd::array![1.0, 3.0, -2.0, 5.0].into_dyn();
    //     let expected = nd::array![1.75, 1.8].into_dyn();
    //     for weighting in [
    //         DistanceWeighting::None,
    //         DistanceWeighting::Gaussian,
    //         DistanceWeighting::CompactSupport { exponent: 2.0 },
    //     ] {
    //         let op =
    //             MovingLeastSquaresTransfer::from_coords(&src.view(), &tgt.view(), 4, weighting);
    //         let out = op.apply_on_array(&field.view());
    //         for (o, e) in out.iter().zip(expected.iter()) {
    //             assert!(
    //                 (o - e).abs() < 1e-9,
    //                 "weighting = {weighting:?}: got {o}, expected {e}"
    //             );
    //         }
    //     }
    // }

    // #[test]
    // fn reproduces_affine_field_exactly_2d() {
    //     let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    //     let tgt = nd::array![[1.0 / 3.0, 1.0 / 3.0], [0.2, 0.4]];
    //     // f = 1 + 2x - 3y
    //     let field = nd::array![1.0, 3.0, -2.0].into_dyn();
    //     let expected = nd::array![2.0 / 3.0, 0.2].into_dyn();
    //     let op = MovingLeastSquaresTransfer::from_coords(
    //         &src.view(),
    //         &tgt.view(),
    //         3,
    //         DistanceWeighting::None,
    //     );
    //     let out = op.apply_on_array(&field.view());
    //     for (o, e) in out.iter().zip(expected.iter()) {
    //         assert_relative_eq!(o, e, epsilon = 1e-9);
    //     }
    // }

    // #[test]
    // fn reproduces_constant_field_on_grid() {
    //     let src = src_grid_3d();
    //     let tgt = nd::array![[0.32, 0.17, 0.71], [0.9, 0.1, 0.9]];
    //     let field = nd::Array::from_elem(nd::IxDyn(&[125]), 7.0);
    //     for weighting in [
    //         DistanceWeighting::None,
    //         DistanceWeighting::InverseDistance { exponent: 2.0 },
    //         DistanceWeighting::CompactSupport { exponent: 3.0 },
    //         DistanceWeighting::Gaussian,
    //     ] {
    //         let op =
    //             MovingLeastSquaresTransfer::from_coords(&src.view(), &tgt.view(), 8, weighting);
    //         let out = op.apply_on_array(&field.view());
    //         assert!(
    //             out.iter().all(|&v| (v - 7.0).abs() < 1e-9),
    //             "constant not reproduced with weighting = {weighting:?}"
    //         );
    //     }
    // }

    // #[test]
    // fn coincident_target_is_exact() {
    //     let src = nd::array![
    //         [0.0, 0.0, 0.0],
    //         [1.0, 0.0, 0.0],
    //         [0.0, 1.0, 0.0],
    //         [0.0, 0.0, 1.0],
    //         [0.5, 0.5, 0.5]
    //     ];
    //     let tgt = nd::array![[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]];
    //     let field = nd::array![1.0, 2.0, 3.0, 4.0, 9.0].into_dyn();
    //     let op = MovingLeastSquaresTransfer::from_coords(
    //         &src.view(),
    //         &tgt.view(),
    //         4,
    //         DistanceWeighting::Gaussian,
    //     );
    //     let out = op.apply_on_array(&field.view());
    //     assert_eq!(out[0], 9.0);
    //     assert!(out[1].is_finite());
    // }

    // #[test]
    // fn singular_cloud_falls_back() {
    //     // Coplanar source points: the 3D linear system is rank-deficient and the solve fails,
    //     // but the transfer must not panic and a constant field is still reproduced.
    //     let src = nd::array![
    //         [0.0, 0.0, 0.0],
    //         [1.0, 0.0, 0.0],
    //         [0.0, 1.0, 0.0],
    //         [1.0, 1.0, 0.0]
    //     ];
    //     let tgt = nd::array![[0.4, 0.3, 0.0]];
    //     let field = nd::Array::from_elem(nd::IxDyn(&[4]), 5.0);
    //     for weighting in [
    //         DistanceWeighting::None,
    //         DistanceWeighting::Gaussian,
    //         DistanceWeighting::InverseDistance { exponent: 2.0 },
    //         DistanceWeighting::CompactSupport { exponent: 2.0 },
    //     ] {
    //         let op =
    //             MovingLeastSquaresTransfer::from_coords(&src.view(), &tgt.view(), 4, weighting);
    //         let out = op.apply_on_array(&field.view());
    //         assert_relative_eq!(out[0], 5.0, epsilon = 1e-9);
    //     }
    // }

    // #[test]
    // fn vector_valued_field() {
    //     let src = src_grid_3d();
    //     let tgt = nd::array![[0.32, 0.17, 0.71], [0.5, 0.5, 0.5]];
    //     let mut field = nd::Array2::<f64>::zeros((125, 3));
    //     for i in 0..125 {
    //         field[[i, 0]] = 1.0;
    //         field[[i, 1]] = i as f64;
    //         field[[i, 2]] = 2.0 * i as f64;
    //     }
    //     let op = MovingLeastSquaresTransfer::from_coords(
    //         &src.view(),
    //         &tgt.view(),
    //         8,
    //         DistanceWeighting::InverseDistance { exponent: 2.0 },
    //     );
    //     let out = op.apply_on_array(&field.view().into_dyn());
    //     assert_eq!(out.shape(), &[2, 3]);
    //     // constant component reproduced
    //     assert_relative_eq!(out[[0, 0]], 1.0, epsilon = 1e-9);
    //     // coincident target at (0.5,0.5,0.5) is exact
    //     let i50 = 2 * 25 + 2 * 5 + 2;
    //     assert_eq!(out[[1, 1]], i50 as f64);
    //     assert_eq!(out[[1, 2]], 2.0 * i50 as f64);
    // }

    // #[test]
    // fn weighted_differs_from_unweighted() {
    //     // With k > d + 1 the weighting scheme changes the interpolation weights.
    //     let src = nd::array![
    //         [0.0, 0.0, 0.0],
    //         [1.0, 0.0, 0.0],
    //         [0.0, 1.0, 0.0],
    //         [0.0, 0.0, 1.0],
    //         [1.0, 1.0, 0.0],
    //         [1.0, 0.0, 1.0],
    //         [0.0, 1.0, 1.0],
    //         [1.0, 1.0, 1.0]
    //     ];
    //     let tgt = nd::array![[0.4, 0.3, 0.2]];
    //     let plain = MovingLeastSquaresTransfer::from_coords(
    //         &src.view(),
    //         &tgt.view(),
    //         8,
    //         DistanceWeighting::None,
    //     );
    //     let inverse = MovingLeastSquaresTransfer::from_coords(
    //         &src.view(),
    //         &tgt.view(),
    //         8,
    //         DistanceWeighting::InverseDistance { exponent: 4.0 },
    //     );
    //     let field = nd::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_dyn();
    //     let a = plain.apply_on_array(&field.view());
    //     let b = inverse.apply_on_array(&field.view());
    //     assert!(
    //         (a[0] - b[0]).abs() > 1e-6,
    //         "weighted and unweighted interpolation should differ, got {a:?} vs {b:?}"
    //     );
    // }
}
