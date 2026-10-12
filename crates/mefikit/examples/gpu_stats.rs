//! GPU-accelerated field statistics: sum / mean / min / max in `f64`.
//!
//! Demonstrates the `mf::field_stats` tool backed by CubeCL. The device is
//! picked with the canonical [`Device::default`] walk (wgpu first, then the
//! CPU runtime when the `gpu-cpu` feature is enabled). Set
//! `MEFIKIT_GPU_BACKEND=wgpu` or `=cpu` to pin a specific backend.
//!
//! Run with: `cargo run -p mefikit --features gpu gpu-cpu --example gpu_stats`

use cubecl::device::{Device, WgpuDeviceKind};
use mefikit::prelude as mf;
use mefikit::tools::fieldexpr::{FieldExpr, MeshEvalUpdatable};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis(
            (0..=50)
                .map(|i| i as f64 / 50.0 * 3.0)
                .collect::<Vec<f64>>(),
        )
        .add_axis(
            (0..=40)
                .map(|i| i as f64 / 40.0 * 2.0)
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
    println!("gpu stats in {:.3?}", now.elapsed());
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
    Ok(())
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
