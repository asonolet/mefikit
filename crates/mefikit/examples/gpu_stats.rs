//! GPU-accelerated field statistics: sum / mean / min / max in `f64`.
//!
//! Demonstrates the `mf::field_stats` tool backed by CubeCL and gives a small,
//! rough time comparison against the classical single-threaded CPU reduction
//! (`Iterator::sum` + `fold` min/max) on the same values. The device is picked
//! with the canonical [`Device::default`] walk (wgpu first, then the CPU
//! runtime when the `gpu-cpu` feature is enabled). Set
//! `MEFIKIT_GPU_BACKEND=wgpu` or `=cpu` to pin a specific backend.
//!
//! Run with: `cargo run -p mefikit --features gpu,gpu-cpu --example gpu_stats`

use cubecl::device::{Device, WgpuDeviceKind};
use mefikit::prelude as mf;
use mefikit::tools::fieldexpr::{FieldExpr, MeshEvalUpdatable};

/// Mesh resolution per axis: `N_AXIS²` cell values are reduced.
const N_AXIS: usize = 601;
/// Number of timed repetitions (after one warm-up pass).
const ITERATIONS: usize = 50;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis(
            (0..=N_AXIS)
                .map(|i| i as f64 / N_AXIS as f64 * 3.0)
                .collect::<Vec<f64>>(),
        )
        .add_axis(
            (0..=N_AXIS)
                .map(|i| i as f64 / N_AXIS as f64 * 2.0)
                .collect::<Vec<f64>>(),
        )
        .build();

    let mut mesh = mesh;
    mesh.eval_update_field("temperature", None, FieldExpr::X * FieldExpr::Y);

    let field = mesh
        .field("temperature", Some(mf::Dimension::D2))
        .expect("temperature field on 2D cells");
    let values = field.0[&mf::ElementType::QUAD4]
        .as_slice()
        .expect("contiguous field");
    println!("values: {}", values.len());

    let device = select_device();
    let client = device.client();
    println!("device: {}", client.name());

    let now = std::time::Instant::now();
    let gpu = mf::field_stats(&device, values);
    println!("first gpu stats in {:.3?}", now.elapsed());
    println!("count: {}", gpu.count);
    println!("sum:   {:.6}", gpu.sum);
    println!("mean:  {:.6}", gpu.mean);
    println!("min:   {:.6}", gpu.min);
    println!("max:   {:.6}", gpu.max);

    let want = Stats {
        sum: values.iter().sum(),
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    };
    let mean = want.sum / values.len() as f64;
    println!("reference mean: {mean:.6}");

    let ok = (gpu.sum - want.sum).abs() <= 1e-9 * want.sum.abs().max(1.0)
        && (gpu.mean - mean).abs() <= 1e-9
        && (gpu.min - want.min).abs() <= 1e-9 * want.min.abs().max(1.0)
        && (gpu.max - want.max).abs() <= 1e-9 * want.max.abs().max(1.0);
    println!("match: {ok}");

    // Timed comparison: warm up, then time `ITERATIONS` runs of each path.
    std::hint::black_box(mf::field_stats(&device, values));
    let gpu_total = timed(|| {
        std::hint::black_box(mf::field_stats(&device, values));
    });
    let cpu_total = timed(|| {
        let s = values.iter().sum::<f64>();
        let mn = values.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        std::hint::black_box((s, mn, mx));
    });

    let gpu_per = gpu_total / ITERATIONS as u32;
    let cpu_per = cpu_total / ITERATIONS as u32;
    println!(
        "mean per call over {ITERATIONS} runs: gpu {gpu_per:.3?}, cpu {cpu_per:.3?} (speedup {:.2}x)",
        cpu_per.as_secs_f64() / gpu_per.as_secs_f64()
    );
    Ok(())
}

fn timed(mut f: impl FnMut()) -> std::time::Duration {
    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    start.elapsed()
}

struct Stats {
    sum: f64,
    min: f64,
    max: f64,
}

/// Pick the cubecl device: `MEFIKIT_GPU_BACKEND=cpu|wgpu`, else the canonical
/// auto-detection walk (wgpu first, then the CPU runtime when linked).
fn select_device() -> Device {
    match std::env::var("MEFIKIT_GPU_BACKEND").as_deref() {
        Ok("wgpu") => Device::wgpu(WgpuDeviceKind::DefaultDevice).expect("no wgpu adapter found"),
        Ok("cpu") => Device::cpu().expect("CPU runtime not linked: enable the `gpu-cpu` feature"),
        Ok(other) => panic!("MEFIKIT_GPU_BACKEND must be `wgpu` or `cpu`, got `{other}`"),
        Err(_) => Device::default(),
    }
}
