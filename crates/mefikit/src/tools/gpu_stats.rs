//! GPU-accelerated unreduced field statistics via CubeCL.
//!
//! Runs a single-pass reduction (sum, min, max) of a `f64` field on a
//! CubeCL [`Device`], then folds the partial results back on the host.
//! Requires the `gpu` feature; add `gpu-cpu` to enable the CPU runtime
//! fallback. All arithmetic stays in `f64`.

use cubecl::Device;
use cubecl::prelude::*;

/// Maximum number of lanes (cube units) used by the reduction kernel.
const MAX_UNITS: usize = 1024;

/// Basic statistics of a field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    /// Number of values.
    pub count: usize,
    /// Sum of all values.
    pub sum: f64,
    /// Arithmetic mean of all values.
    pub mean: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
}

/// One lane per output slot; each lane reduces its strided chunk of the input.
#[cube(launch_unchecked)]
fn partial_stats<F: Float>(input: &[F], sum: &mut [F], min: &mut [F], max: &mut [F]) {
    let mut acc = F::new(0.0f32);
    let mut acc_min = F::new(0.0f32);
    let mut acc_max = F::new(0.0f32);
    let mut started = false;
    let step = CUBE_DIM as usize;
    let mut i = UNIT_POS as usize;
    while i < input.len() {
        let x = input[i];
        acc += x;
        if !started {
            acc_min = x;
            acc_max = x;
            started = true;
        } else {
            if x < acc_min {
                acc_min = x;
            }
            if x > acc_max {
                acc_max = x;
            }
        }
        i += step;
    }
    sum[UNIT_POS as usize] = acc;
    min[UNIT_POS as usize] = acc_min;
    max[UNIT_POS as usize] = acc_max;
}

/// Computes the [`Stats`] of `values` as a `f64` reduction on `device`.
///
/// Panics if the device cannot run the kernel (for example, when the runtime
/// is not linked or the backend rejects `f64`). An empty slice yields
/// `count == 0` with `NaN` mean/min/max.
pub fn field_stats(device: &Device, values: &[f64]) -> Stats {
    if values.is_empty() {
        return Stats {
            count: 0,
            sum: 0.0,
            mean: f64::NAN,
            min: f64::NAN,
            max: f64::NAN,
        };
    }

    let client = device.client();
    let count = values.len();
    let units = count.min(MAX_UNITS);
    let sum = client.empty(units * size_of::<f64>());
    let min = client.empty(units * size_of::<f64>());
    let max = client.empty(units * size_of::<f64>());
    unsafe {
        partial_stats::launch_unchecked::<f64>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(u32::try_from(units).expect("gpu field length exceeds `u32`")),
            BufferArg::from_raw_parts(client.create_from_slice(f64::as_bytes(values)), count),
            BufferArg::from_raw_parts(sum.clone(), units),
            BufferArg::from_raw_parts(min.clone(), units),
            BufferArg::from_raw_parts(max.clone(), units),
        );
    }
    let sum = f64s(&client.read_one_unchecked(sum));
    let min = f64s(&client.read_one_unchecked(min));
    let max = f64s(&client.read_one_unchecked(max));
    fold_partials(values.len(), &sum, &min, &max)
}

/// Decodes a byte buffer returned by [`Client::read_one_unchecked`] into `f64`
/// values, independently of allocation alignment.
fn f64s(bytes: &[u8]) -> Vec<f64> {
    bytes
        .as_chunks::<{ size_of::<f64>() }>()
        .0
        .iter()
        .map(|chunk| f64::from_ne_bytes(*chunk))
        .collect()
}

/// Folds the per-lane partial results into a single [`Stats`].
fn fold_partials(count: usize, sum: &[f64], min: &[f64], max: &[f64]) -> Stats {
    let total: f64 = sum.iter().sum();
    let min = min.iter().copied().fold(f64::INFINITY, f64::min);
    let max = max.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Stats {
        count,
        sum: total,
        mean: total / count as f64,
        min,
        max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::device::WgpuDeviceKind;

    #[test]
    fn fold_matches_naive_reduction() {
        let values: Vec<f64> = vec![1.5, -2.0, 3.25, 0.0, -0.5, 7.0];
        // Mimic the strided per-lane reduction performed by the kernel.
        let (sum, min, max) = host_partials(&values, values.len());
        let stats = fold_partials(values.len(), &sum, &min, &max);
        let naive_sum: f64 = values.iter().sum();
        let naive_min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let naive_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(stats.count, values.len());
        assert_eq!(stats.sum, naive_sum);
        assert_eq!(stats.min, naive_min);
        assert_eq!(stats.max, naive_max);
        assert_eq!(stats.mean, naive_sum / values.len() as f64);
    }

    #[test]
    fn fold_uses_two_lanes() {
        let values: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let (sum, min, max) = host_partials(&values, 2);
        let stats = fold_partials(values.len(), &sum, &min, &max);
        assert_eq!(stats.sum, values.iter().sum::<f64>());
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 9.0);
    }

    fn host_partials(values: &[f64], units: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut sum = vec![0.0; units];
        let mut min = vec![f64::INFINITY; units];
        let mut max = vec![f64::NEG_INFINITY; units];
        for (i, &value) in values.iter().enumerate() {
            let lane = i % units;
            sum[lane] += value;
            min[lane] = min[lane].min(value);
            max[lane] = max[lane].max(value);
        }
        (sum, min, max)
    }

    #[test]
    fn field_stats_matches_reference() {
        let Ok(device) = Device::wgpu(WgpuDeviceKind::DefaultDevice) else {
            // No compatible adapter in this environment: nothing to check.
            return;
        };
        let values: Vec<f64> = (0..5000).map(|i| f64::from(i).sin() * 3.0 - 1.0).collect();
        let got = field_stats(&device, &values);
        let want_sum: f64 = values.iter().sum();
        let want_min = values.iter().copied().fold(f64::INFINITY, f64::min);
        let want_max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(got.count, values.len());
        assert!((got.sum - want_sum).abs() <= 1e-9 * want_sum.abs().max(1.0));
        assert!((got.min - want_min).abs() <= 1e-9 * want_min.abs().max(1.0));
        assert!((got.max - want_max).abs() <= 1e-9 * want_max.abs().max(1.0));
        assert!((got.mean - want_sum / values.len() as f64).abs() <= 1e-9);
    }
}
