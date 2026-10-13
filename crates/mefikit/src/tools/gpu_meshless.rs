//! GPU-accelerated meshless reconstruction via CubeCL.
//!
//! This is the device counterpart of the `meshless` module: it speeds up the per-target-point
//! weighted local fit that dominates building the moving least-squares gradient and transfer
//! operators. The `k`-nearest-neighbours search is *not* recomputed on the device; it stays on the
//! host (the accelerated `kiddo` kd-tree of [`crate::tools::meshless`]) and only the fit runs on
//! the device, so both paths start from the same neighbour lists.
//!
//! Two entry points mirror the CPU solvers one-for-one:
//! - [`gpu_gradient_weights`] mirrors `solve_gradient_neighbours`: `k` weights per space direction
//!   per target point (the derivative of the weighted degree-1 fit).
//! - [`gpu_interpolation_weights`] mirrors `solve_neighbours`: `k` value-interpolation weights per
//!   target point, either the moving least-squares fit or the inverse-distance (Shepard) scheme.
//!
//! The device fit mirrors the CPU semantics, including the degenerate cases (rank deficiency,
//! coincident points, non-finite inverse-distance weights). Two precision details differ from the
//! `f64` CPU reference:
//! - The kernel works in `f32` and widens back to `f64` at the host boundary. The `wgpu` backend
//!   cannot evaluate the `f64` transcendental kernels (`exp`, `powf`) correctly, so `f32` is the
//!   device precision.
//! - The rank threshold is `1e-6` (just above `f32::EPSILON`); the CPU `lstsq` uses `1e-12`.
//!
//! The linear solve also differs: the CPU reference uses the generic SVD-based `lstsq`, while the
//! kernel inverts the fixed-size symmetric normal-equations matrix with pivoted Gauss-Jordan.
//! Requires the `gpu` feature; add `gpu-cpu` to enable the CPU runtime fallback.

use cubecl::Device;
use cubecl::prelude::*;
use ndarray as nd;

use crate::tools::meshless::{DistanceWeighting, NeighbourScheme, nearest_neighbours};

/// Maximum number of lanes used by the fit kernel.
const MAX_UNITS: usize = 1024;

/// Relative pivoting threshold of the fixed-size linear solve. The CPU reference `lstsq` uses
/// `1e-12` in `f64`; the device solve runs in `f32`, where that threshold is below the resolution
/// of the arithmetic, so a value just above `f32::EPSILON` separates genuine pivots from the
/// rounding noise of algebraically dependent rows.
const RANK_TOL: f32 = 1e-6;

/// Code of a [`DistanceWeighting`] variant, selecting the kernel formula on the device.
fn weighting_variant(weighting: DistanceWeighting) -> u32 {
    match weighting {
        DistanceWeighting::Constant => 0,
        DistanceWeighting::InverseDistance { .. } => 1,
        DistanceWeighting::CompactSupport { .. } => 2,
        DistanceWeighting::Gaussian => 3,
    }
}

/// Exponent carried by a [`DistanceWeighting`] variant (`0.0` when the kernel ignores it).
fn weighting_exponent(weighting: DistanceWeighting) -> f64 {
    match weighting {
        DistanceWeighting::InverseDistance { exponent }
        | DistanceWeighting::CompactSupport { exponent } => exponent,
        DistanceWeighting::Constant | DistanceWeighting::Gaussian => 0.0,
    }
}

/// Evaluates the [`DistanceWeighting`] kernel on the squared scaled distance `s2 = (r / h)²`.
#[cube]
fn distance_kernel<F: Float>(variant: u32, exponent: F, s2: F) -> F {
    if variant == 0u32 {
        F::new(1.0f32)
    } else if variant == 1u32 {
        s2.powf(F::new(-0.5f32) * exponent)
    } else if variant == 2u32 {
        if s2 >= F::new(1.0f32) {
            F::new(0.0f32)
        } else {
            (F::new(1.0f32) - s2).powf(exponent)
        }
    } else {
        (F::new(0.0f32) - s2).exp()
    }
}

/// Whether `x` is finite. Both the `NaN` and the infinities fail the comparisons, which is used
/// instead of `is_nan`/`is_inf` because those do not keep a plain `bool` type for a generic float.
#[cube]
fn finite<F: Float>(x: F) -> bool {
    x <= F::new(f32::MAX) && x >= F::new(f32::MIN)
}

/// Computes gradient or value-interpolation weights for every target point.
///
/// One lane handles several targets with a strided walk. `d` is the space dimension and `k` the
/// number of neighbours (both comptime, as they size the local arrays). `gradient` selects the
/// output between the gradient weights (`n_tgt · k · d`, the derivative of the affine fit, `1`) and
/// the value-interpolation weights (`n_tgt · k`, `0`). `variant`/`params[0]` select the
/// [`DistanceWeighting`] kernel.
#[cube(launch_unchecked)]
fn mls_fit<F: Float>(
    src: &[F],
    tgt: &[F],
    nb: &[u32],
    r2: &[F],
    params: &[F],
    out: &mut [F],
    variant: u32,
    gradient: u32,
    #[comptime] d: usize,
    #[comptime] k: usize,
) {
    let exponent = params[0];
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let step = CUBE_DIM as usize;
    let n_tgt = tgt.len() / d;
    let dim1 = comptime![d + 1];
    let ow = if gradient == 0u32 { 1usize } else { d };

    let mut j = UNIT_POS as usize;
    while j < n_tgt {
        let row = j * k;
        let orow = j * k * ow;

        // Zero the output row: the CPU reference pre-fills its result with zeros and only writes
        // the entries that survive the (possibly rank-deficient) fit.
        let mut z = 0usize;
        while z < k * ow {
            out[orow + z] = zero;
            z += 1;
        }

        // Characteristic length: the distance to the farthest selected neighbour.
        let mut h2 = zero;
        let mut l = 0usize;
        while l < k {
            let rr = r2[row + l];
            if rr > h2 {
                h2 = rr;
            }
            l += 1;
        }
        let h = h2.sqrt();

        // Raw kernel weights, then the same normalization as the host `kernel_weights`: bring the
        // largest weight to one (skipped when any weight is non-finite).
        let mut raw = Array::<F>::new(k);
        let mut any_nonfinite = false;
        let mut wmax = zero;
        let mut l = 0usize;
        while l < k {
            let s2 = if h2 > zero { r2[row + l] / h2 } else { zero };
            let ww = distance_kernel::<F>(variant, exponent, s2);
            raw[l] = ww;
            if finite::<F>(ww) {
                if ww > wmax {
                    wmax = ww;
                }
            } else {
                any_nonfinite = true;
            }
            l += 1;
        }
        let norm = if !any_nonfinite && wmax > zero {
            wmax
        } else {
            one
        };
        let mut w = Array::<F>::new(k);
        let mut l = 0usize;
        while l < k {
            w[l] = if any_nonfinite { raw[l] } else { raw[l] / norm };
            l += 1;
        }

        // An evaluation point coinciding with a source point makes an inverse-distance kernel
        // singular (`r = 0`). The affine fit is exact for any positive weights, so replace the
        // singular weights by the largest finite one to keep the normal equations finite.
        if gradient != 0u32 && any_nonfinite {
            let mut fmax = zero;
            let mut l = 0usize;
            while l < k {
                let ww = w[l];
                if finite::<F>(ww) && ww > fmax {
                    fmax = ww;
                }
                l += 1;
            }
            let fb = if fmax > zero { fmax } else { one };
            let mut l = 0usize;
            while l < k {
                if !finite::<F>(w[l]) {
                    w[l] = fb;
                }
                l += 1;
            }
        }

        // Weighted degree-1 normal equations `G = AᵀWA` on centered, `h`-normalized coordinates.
        // Only built when the characteristic length is positive (otherwise the row stays zero, as
        // the CPU solver `continue`s).
        let mut inv = Array::<F>::new(dim1 * dim1);
        let mut rank = 0u32;
        if h > zero {
            let mut gram = Array::<F>::new(dim1 * dim1);
            let mut l = 0usize;
            while l < k {
                let si = nb[row + l] as usize;
                let wl = w[l];
                gram[0] += wl;
                let mut b = 0usize;
                while b < d {
                    let ub = (src[si * d + b] - tgt[j * d + b]) / h;
                    let ib = b + 1;
                    gram[ib] += wl * ub;
                    gram[ib * dim1] += wl * ub;
                    let mut c = 0usize;
                    while c < d {
                        let uc = (src[si * d + c] - tgt[j * d + c]) / h;
                        gram[ib * dim1 + (c + 1)] += wl * ub * uc;
                        c += 1;
                    }
                    b += 1;
                }
                l += 1;
            }

            // Scale of the rank test, relative to the largest entry of the system.
            let mut amax = zero;
            let mut p = 0usize;
            while p < dim1 * dim1 {
                let v = if gram[p] < zero {
                    zero - gram[p]
                } else {
                    gram[p]
                };
                if v > amax {
                    amax = v;
                }
                p += 1;
            }
            let tol = F::new(RANK_TOL) * amax;

            // Pivoted Gauss-Jordan on `[gram | I]` -> `inv`; `rank` counts the accepted pivots.
            let mut p = 0usize;
            while p < dim1 {
                inv[p * dim1 + p] = one;
                p += 1;
            }
            let mut col = 0usize;
            while col < dim1 {
                let mut piv = col;
                let bc = gram[col * dim1 + col];
                let mut pivval = if bc < zero { zero - bc } else { bc };
                let mut rr = col + 1;
                while rr < dim1 {
                    let v = gram[rr * dim1 + col];
                    let av = if v < zero { zero - v } else { v };
                    if av > pivval {
                        pivval = av;
                        piv = rr;
                    }
                    rr += 1;
                }
                if pivval > tol {
                    if piv != col {
                        let mut c = 0usize;
                        while c < dim1 {
                            let tmp = gram[col * dim1 + c];
                            gram[col * dim1 + c] = gram[piv * dim1 + c];
                            gram[piv * dim1 + c] = tmp;
                            let tmp = inv[col * dim1 + c];
                            inv[col * dim1 + c] = inv[piv * dim1 + c];
                            inv[piv * dim1 + c] = tmp;
                            c += 1;
                        }
                    }
                    let pv = gram[col * dim1 + col];
                    let mut c = 0usize;
                    while c < dim1 {
                        gram[col * dim1 + c] /= pv;
                        inv[col * dim1 + c] /= pv;
                        c += 1;
                    }
                    let mut rr = 0usize;
                    while rr < dim1 {
                        if rr != col {
                            let f = gram[rr * dim1 + col];
                            let mut c = 0usize;
                            while c < dim1 {
                                gram[rr * dim1 + c] =
                                    gram[rr * dim1 + c] - f * gram[col * dim1 + c];
                                inv[rr * dim1 + c] = inv[rr * dim1 + c] - f * inv[col * dim1 + c];
                                c += 1;
                            }
                        }
                        rr += 1;
                    }
                    rank += 1;
                }
                col += 1;
            }
        }

        let full_rank = rank == dim1 as u32;

        if gradient != 0u32 {
            // Gradient weights: `w_l · shape(coeff_a) / h`, one set per space direction.
            if full_rank {
                let mut a = 0usize;
                while a < d {
                    let col = a + 1;
                    let mut finite_coeff = true;
                    let mut c = 0usize;
                    while c < dim1 {
                        if !finite::<F>(inv[c * dim1 + col]) {
                            finite_coeff = false;
                        }
                        c += 1;
                    }
                    if finite_coeff {
                        let mut l = 0usize;
                        while l < k {
                            let si = nb[row + l] as usize;
                            let mut sv = inv[col];
                            let mut b = 0usize;
                            while b < d {
                                let ub = (src[si * d + b] - tgt[j * d + b]) / h;
                                sv += ub * inv[(b + 1) * dim1 + col];
                                b += 1;
                            }
                            out[orow + l * d + a] = w[l] * sv / h;
                            l += 1;
                        }
                    }
                    a += 1;
                }
            }
        } else {
            // Value-interpolation weights: `w_l · shape(coeff_0)`. A target point coinciding with a
            // source point interpolates it exactly.
            let mut any_zero = false;
            let mut first = 0usize;
            let mut l = 0usize;
            while l < k {
                if r2[row + l] == zero && !any_zero {
                    first = l;
                    any_zero = true;
                }
                l += 1;
            }
            if any_zero {
                out[orow + first] = one;
            } else {
                let mut fit = full_rank;
                if fit {
                    let mut c = 0usize;
                    while c < dim1 {
                        if !finite::<F>(inv[c * dim1]) {
                            fit = false;
                        }
                        c += 1;
                    }
                }
                if fit {
                    let mut l = 0usize;
                    while l < k {
                        let si = nb[row + l] as usize;
                        let mut sv = inv[0];
                        let mut b = 0usize;
                        while b < d {
                            let ub = (src[si * d + b] - tgt[j * d + b]) / h;
                            sv += ub * inv[(b + 1) * dim1];
                            b += 1;
                        }
                        out[orow + l] = w[l] * sv;
                        l += 1;
                    }
                } else {
                    // Degenerate local system: normalized kernel weights (Shepard's method), then
                    // a plain average.
                    let mut sum = zero;
                    let mut l = 0usize;
                    while l < k {
                        if finite::<F>(w[l]) {
                            sum += w[l];
                        }
                        l += 1;
                    }
                    let mut l = 0usize;
                    while l < k {
                        let w_finite = finite::<F>(w[l]);
                        out[orow + l] = if sum > zero {
                            if w_finite { w[l] / sum } else { zero }
                        } else {
                            one / F::new(k as f32)
                        };
                        l += 1;
                    }
                }
            }
        }

        j += step;
    }
}

/// Computes the inverse-distance (Shepard) weights for every target point.
#[cube(launch_unchecked)]
fn idw_fit<F: Float>(
    tgt: &[F],
    r2: &[F],
    params: &[F],
    out: &mut [F],
    #[comptime] d: usize,
    #[comptime] k: usize,
) {
    let exponent = params[0];
    let zero = F::new(0.0f32);
    let one = F::new(1.0f32);
    let half = F::new(0.5f32);
    let step = CUBE_DIM as usize;
    let n_tgt = tgt.len() / d;

    let mut j = UNIT_POS as usize;
    while j < n_tgt {
        let row = j * k;
        let mut any_zero = false;
        let mut first = 0usize;
        let mut l = 0usize;
        while l < k {
            if r2[row + l] == zero && !any_zero {
                first = l;
                any_zero = true;
            }
            l += 1;
        }
        if any_zero {
            let mut l = 0usize;
            while l < k {
                out[row + l] = if l == first { one } else { zero };
                l += 1;
            }
        } else {
            let mut sum = zero;
            let mut l = 0usize;
            while l < k {
                sum += r2[row + l].powf(zero - half * exponent);
                l += 1;
            }
            let mut l = 0usize;
            while l < k {
                out[row + l] = r2[row + l].powf(zero - half * exponent) / sum;
                l += 1;
            }
        }
        j += step;
    }
}

/// Launches [`mls_fit`] over `n_tgt` strided lanes and returns the device output.
#[allow(clippy::too_many_arguments)]
fn launch_mls(
    device: &Device,
    src: &[f64],
    d: usize,
    tgt: &[f64],
    n_tgt: usize,
    nb: &[u32],
    r2: &[f64],
    k: usize,
    gradient: u32,
    variant: u32,
    exponent: f64,
) -> Vec<f64> {
    let client = device.client();
    let units = u32::try_from(n_tgt.clamp(1, MAX_UNITS)).expect("gpu point count exceeds `u32`");
    let out_elems = n_tgt * k * if gradient == 0 { 1 } else { d };
    let out = client.empty(out_elems.max(1) * size_of::<f32>());
    let params = client.create_from_slice(f32::as_bytes(&[exponent as f32]));
    let src_h = client.create_from_slice(f32::as_bytes(&to_f32(src)));
    let tgt_h = client.create_from_slice(f32::as_bytes(&to_f32(tgt)));
    let nb_h = client.create_from_slice(u32::as_bytes(nb));
    let r2_h = client.create_from_slice(f32::as_bytes(&to_f32(r2)));
    unsafe {
        mls_fit::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(src_h, src.len()),
            BufferArg::from_raw_parts(tgt_h, tgt.len()),
            BufferArg::from_raw_parts(nb_h, nb.len()),
            BufferArg::from_raw_parts(r2_h, r2.len()),
            BufferArg::from_raw_parts(params, 1),
            BufferArg::from_raw_parts(out.clone(), out_elems.max(1)),
            variant,
            gradient,
            d,
            k,
        );
    }
    f32s(&client.read_one_unchecked(out))
}

/// Launches [`idw_fit`] over `n_tgt` strided lanes and returns the device output.
fn launch_idw(
    device: &Device,
    d: usize,
    tgt: &[f64],
    n_tgt: usize,
    r2: &[f64],
    k: usize,
    exponent: f64,
) -> Vec<f64> {
    let client = device.client();
    let units = u32::try_from(n_tgt.clamp(1, MAX_UNITS)).expect("gpu point count exceeds `u32`");
    let out_elems = n_tgt * k;
    let out = client.empty(out_elems.max(1) * size_of::<f32>());
    let params = client.create_from_slice(f32::as_bytes(&[exponent as f32]));
    let tgt_h = client.create_from_slice(f32::as_bytes(&to_f32(tgt)));
    let r2_h = client.create_from_slice(f32::as_bytes(&to_f32(r2)));
    unsafe {
        idw_fit::launch_unchecked::<f32>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(units),
            BufferArg::from_raw_parts(tgt_h, tgt.len()),
            BufferArg::from_raw_parts(r2_h, r2.len()),
            BufferArg::from_raw_parts(params, 1),
            BufferArg::from_raw_parts(out.clone(), out_elems.max(1)),
            d,
            k,
        );
    }
    f32s(&client.read_one_unchecked(out))
}

/// Narrows a `f64` slice to the `f32` device precision.
fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|&v| v as f32).collect()
}

/// Widens a byte buffer returned by `Client::read_one_unchecked` into `f64` values.
fn f32s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<{ size_of::<f32>() }>()
        .0
        .iter()
        .map(|chunk| f32::from_ne_bytes(*chunk) as f64)
        .collect()
}

/// Validates the shared source/target coordinates and runs the host k-NN search.
fn neighbours(
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
) -> (usize, nd::Array2<usize>, nd::Array2<f64>) {
    let d = src_coords.ncols();
    assert!(
        (2..=3).contains(&d),
        "Meshless reconstruction is only supported in 2D and 3D space, got {d}D"
    );
    assert_eq!(
        d,
        tgt_coords.ncols(),
        "Source and target coordinates should have the same number of columns"
    );
    assert!(k > 0, "k should be at least 1");
    let (indices, r2) = nearest_neighbours(src_coords, tgt_coords, k);
    (d, indices, r2)
}

/// Flattens a coordinate array to a contiguous row-major `Vec`.
fn flatten(coords: &nd::ArrayView2<f64>) -> Vec<f64> {
    match coords.as_slice() {
        Some(s) => s.to_vec(),
        None => coords.iter().copied().collect(),
    }
}

/// Global neighbour indices as `u32` (the device index type).
fn indices_u32(indices: &nd::Array2<usize>) -> Vec<u32> {
    indices
        .iter()
        .map(|&i| u32::try_from(i).expect("source index exceeds `u32`"))
        .collect()
}

/// Computes the moving least-squares gradient weights on `device`.
///
/// Mirrors `solve_gradient_neighbours`: returns the `(n_tgt, k)` source indices (from the host
/// kd-tree) and the `(n_tgt, k, d)` gradient weights.
pub(crate) fn gpu_gradient_weights(
    device: &Device,
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    weighting: DistanceWeighting,
) -> (nd::Array2<usize>, nd::Array3<f64>) {
    let (d, indices, r2) = neighbours(src_coords, tgt_coords, k);
    let n_tgt = tgt_coords.nrows();
    if n_tgt == 0 {
        return (indices, nd::Array3::zeros((0, k, d)));
    }
    let out = launch_mls(
        device,
        &flatten(src_coords),
        d,
        &flatten(tgt_coords),
        n_tgt,
        &indices_u32(&indices),
        r2.as_slice().unwrap(),
        k,
        1,
        weighting_variant(weighting),
        weighting_exponent(weighting),
    );
    let grads = nd::Array3::from_shape_vec((n_tgt, k, d), out).expect("device output length");
    (indices, grads)
}

/// Computes the interpolation weights on `device` for a [`NeighbourScheme`].
///
/// Mirrors `solve_neighbours`: returns the `(n_tgt, k)` source indices (from the host kd-tree) and
/// the `(n_tgt, k)` interpolation weights.
pub(crate) fn gpu_interpolation_weights(
    device: &Device,
    src_coords: &nd::ArrayView2<f64>,
    tgt_coords: &nd::ArrayView2<f64>,
    k: usize,
    scheme: NeighbourScheme,
) -> (nd::Array2<usize>, nd::Array2<f64>) {
    let (d, indices, r2) = neighbours(src_coords, tgt_coords, k);
    let n_tgt = tgt_coords.nrows();
    if n_tgt == 0 {
        return (indices, nd::Array2::zeros((0, k)));
    }
    let r2 = r2.as_slice().unwrap();
    let out = match scheme {
        NeighbourScheme::InverseDistance { exponent } => {
            launch_idw(device, d, &flatten(tgt_coords), n_tgt, r2, k, exponent)
        }
        NeighbourScheme::MovingLeastSquares { weighting } => launch_mls(
            device,
            &flatten(src_coords),
            d,
            &flatten(tgt_coords),
            n_tgt,
            &indices_u32(&indices),
            r2,
            k,
            0,
            weighting_variant(weighting),
            weighting_exponent(weighting),
        ),
    };
    let weights = nd::Array2::from_shape_vec((n_tgt, k), out).expect("device output length");
    (indices, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use cubecl::device::WgpuDeviceKind;

    use crate::tools::meshless::{solve_gradient_neighbours, solve_neighbours};

    /// Device used by the tests; `None` when no compatible adapter exists, so the checks are
    /// skipped instead of failing the suite.
    fn device() -> Option<Device> {
        Device::wgpu(WgpuDeviceKind::DefaultDevice).ok()
    }

    /// Asserts that two arrays agree elementwise within a relative tolerance.
    fn assert_close(got: &nd::ArrayD<f64>, want: &nd::ArrayD<f64>, tol: f64) {
        assert_eq!(got.shape(), want.shape());
        for (g, w) in got.iter().zip(want.iter()) {
            assert!(
                (g - w).abs() <= tol * w.abs().max(1.0),
                "got {g}, expected {w} (tolerance {tol})"
            );
        }
    }

    fn cloud_2d() -> nd::Array2<f64> {
        let mut pts = Vec::new();
        for y in 0..=6 {
            for x in 0..=6 {
                pts.extend([x as f64 * 0.3, y as f64 * 0.3]);
            }
        }
        nd::Array2::from_shape_vec((49, 2), pts).unwrap()
    }

    fn cloud_3d() -> nd::Array2<f64> {
        let mut pts = Vec::new();
        for z in 0..=3 {
            for y in 0..=3 {
                for x in 0..=3 {
                    pts.extend([x as f64 * 0.5, y as f64 * 0.5, z as f64 * 0.5]);
                }
            }
        }
        nd::Array2::from_shape_vec((64, 3), pts).unwrap()
    }

    const WEIGHTINGS: [DistanceWeighting; 4] = [
        DistanceWeighting::Constant,
        DistanceWeighting::InverseDistance { exponent: 2.0 },
        DistanceWeighting::CompactSupport { exponent: 3.0 },
        DistanceWeighting::Gaussian,
    ];

    #[test]
    fn gradient_weights_match_cpu_2d() {
        let Some(dev) = device() else { return };
        let src = cloud_2d();
        let tgt = nd::array![[0.37, 0.41], [1.2, 0.7], [0.05, 0.05]];
        for weighting in WEIGHTINGS {
            let (i_cpu, g_cpu) = solve_gradient_neighbours(&src.view(), &tgt.view(), 6, weighting);
            let (i_gpu, g_gpu) = gpu_gradient_weights(&dev, &src.view(), &tgt.view(), 6, weighting);
            assert_eq!(i_cpu, i_gpu);
            assert_close(&g_gpu.into_dyn(), &g_cpu.into_dyn(), 1e-5);
        }
    }

    #[test]
    fn gradient_weights_match_cpu_3d() {
        let Some(dev) = device() else { return };
        let src = cloud_3d();
        let tgt = nd::array![[0.2, 0.6, 0.9], [1.1, 1.2, 0.4]];
        for weighting in WEIGHTINGS {
            let (i_cpu, g_cpu) = solve_gradient_neighbours(&src.view(), &tgt.view(), 8, weighting);
            let (i_gpu, g_gpu) = gpu_gradient_weights(&dev, &src.view(), &tgt.view(), 8, weighting);
            assert_eq!(i_cpu, i_gpu);
            assert_close(&g_gpu.into_dyn(), &g_cpu.into_dyn(), 1e-5);
        }
    }

    #[test]
    fn affine_gradient_reproduced_2d_and_3d() {
        let Some(dev) = device() else { return };

        // 2D: f = 1 + 2x - 3y.
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tgt = nd::array![[0.25, 0.25], [0.6, 0.3]];
        let values = [1.0, 3.0, -2.0, 0.0];
        let (indices, grads) = gpu_gradient_weights(
            &dev,
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::Constant,
        );
        for j in 0..tgt.nrows() {
            let mut gx = 0.0;
            let mut gy = 0.0;
            for l in 0..4 {
                gx += grads[[j, l, 0]] * values[indices[[j, l]]];
                gy += grads[[j, l, 1]] * values[indices[[j, l]]];
            }
            assert_relative_eq!(gx, 2.0, epsilon = 1e-5);
            assert_relative_eq!(gy, -3.0, epsilon = 1e-5);
        }

        // 3D: f = 1 + 2x - 3y + 4z.
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let tgt = nd::array![[0.2, 0.2, 0.2]];
        let values = [1.0, 3.0, -2.0, 5.0];
        let (indices, grads) = gpu_gradient_weights(
            &dev,
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::Constant,
        );
        let mut g = [0.0; 3];
        for l in 0..4 {
            for a in 0..3 {
                g[a] += grads[[0, l, a]] * values[indices[[0, l]]];
            }
        }
        assert_relative_eq!(g[0], 2.0, epsilon = 1e-5);
        assert_relative_eq!(g[1], -3.0, epsilon = 1e-5);
        assert_relative_eq!(g[2], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn coincident_inverse_distance_gradient_matches_cpu() {
        let Some(dev) = device() else { return };
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tgt = nd::array![[0.0, 0.0], [1.0, 1.0]];
        let weighting = DistanceWeighting::InverseDistance { exponent: 2.0 };
        let (i_cpu, g_cpu) = solve_gradient_neighbours(&src.view(), &tgt.view(), 4, weighting);
        let (i_gpu, g_gpu) = gpu_gradient_weights(&dev, &src.view(), &tgt.view(), 4, weighting);
        assert_eq!(i_cpu, i_gpu);
        assert_close(&g_gpu.into_dyn(), &g_cpu.into_dyn(), 1e-5);
    }

    #[test]
    fn rank_deficient_gradient_is_zero() {
        let Some(dev) = device() else { return };
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let tgt = nd::array![[0.5, 0.5]];
        let (_, grads) = gpu_gradient_weights(
            &dev,
            &src.view(),
            &tgt.view(),
            3,
            DistanceWeighting::Constant,
        );
        assert!(grads.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn interpolation_weights_match_cpu() {
        let Some(dev) = device() else { return };
        let src = cloud_2d();
        let tgt = nd::array![[0.37, 0.41], [1.2, 0.7], [0.05, 0.05]];

        let schemes = [
            NeighbourScheme::InverseDistance { exponent: 2.0 },
            NeighbourScheme::InverseDistance { exponent: 3.0 },
            NeighbourScheme::MovingLeastSquares {
                weighting: DistanceWeighting::Constant,
            },
            NeighbourScheme::MovingLeastSquares {
                weighting: DistanceWeighting::InverseDistance { exponent: 2.0 },
            },
            NeighbourScheme::MovingLeastSquares {
                weighting: DistanceWeighting::CompactSupport { exponent: 3.0 },
            },
            NeighbourScheme::MovingLeastSquares {
                weighting: DistanceWeighting::Gaussian,
            },
        ];
        for scheme in schemes {
            let (i_cpu, w_cpu) = solve_neighbours(&src.view(), &tgt.view(), 6, scheme);
            let (i_gpu, w_gpu) =
                gpu_interpolation_weights(&dev, &src.view(), &tgt.view(), 6, scheme);
            assert_eq!(i_cpu, i_gpu);
            assert_close(&w_gpu.into_dyn(), &w_cpu.into_dyn(), 1e-5);
        }
    }

    #[test]
    fn coincident_interpolation_is_exact() {
        let Some(dev) = device() else { return };
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5]
        ];
        let tgt = nd::array![[0.5, 0.5, 0.5]];
        let scheme = NeighbourScheme::MovingLeastSquares {
            weighting: DistanceWeighting::Gaussian,
        };
        let (indices, weights) =
            gpu_interpolation_weights(&dev, &src.view(), &tgt.view(), 4, scheme);
        // The coincident source is the only one with a unit weight.
        let mut sum = 0.0;
        let mut weighted = 0.0;
        let values = [1.0, 2.0, 3.0, 4.0, 9.0];
        for l in 0..4 {
            sum += weights[[0, l]];
            weighted += weights[[0, l]] * values[indices[[0, l]]];
        }
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);
        assert_relative_eq!(weighted, 9.0, epsilon = 1e-5);
    }

    #[test]
    fn k_exceeding_source_points_matches_cpu() {
        let Some(dev) = device() else { return };

        // Fewer source points than `k` (as in a coarse 3D-to-2D downcast): the host kd-tree pads
        // the neighbour row with the first source at zero distance, and the device must reproduce
        // that instead of rejecting the request.
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tgt = nd::array![[0.2, 0.2]];
        let weighting = DistanceWeighting::Constant;

        let (i_cpu, g_cpu) = solve_gradient_neighbours(&src.view(), &tgt.view(), 5, weighting);
        let (i_gpu, g_gpu) = gpu_gradient_weights(&dev, &src.view(), &tgt.view(), 5, weighting);
        assert_eq!(i_cpu, i_gpu);
        assert_close(&g_gpu.into_dyn(), &g_cpu.into_dyn(), 1e-5);

        let scheme = NeighbourScheme::MovingLeastSquares { weighting };
        let (i_cpu, w_cpu) = solve_neighbours(&src.view(), &tgt.view(), 5, scheme);
        let (i_gpu, w_gpu) = gpu_interpolation_weights(&dev, &src.view(), &tgt.view(), 5, scheme);
        assert_eq!(i_cpu, i_gpu);
        assert_close(&w_gpu.into_dyn(), &w_cpu.into_dyn(), 1e-5);
    }

    #[test]
    fn empty_target_is_empty() {
        let Some(dev) = device() else { return };
        let src = cloud_2d();
        let tgt = nd::Array2::<f64>::zeros((0, 2));
        let (indices, grads) = gpu_gradient_weights(
            &dev,
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::Constant,
        );
        assert_eq!(indices.shape(), &[0, 4]);
        assert_eq!(grads.shape(), &[0, 4, 2]);
    }
}
