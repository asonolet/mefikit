//! GPU-accelerated meshless gradient and transfer in 2D and 3D.
//!
//! Builds the moving least-squares gradient operator and the inverse-distance / moving
//! least-squares transfer operators on a CubeCL device (the `*_gpu` constructors) and checks them
//! against the host-CPU references on the same meshes. The device fit runs in `f32` (the `wgpu`
//! backend cannot evaluate the `f64` transcendental kernels) while the CPU reference stays in
//! `f64`, so the agreement is measured and reported rather than assumed.
//!
//! Run with: `cargo run -p mefikit --features gpu,gpu-cpu --example gpu_meshless`

use std::time::{Duration, Instant};

use cubecl::device::{Device, WgpuDeviceKind};
use mefikit::mesh::ElementType;
use mefikit::prelude as mf;
use mefikit::prelude::{Gradient, Transfer};
use mefikit::tools::fieldexpr::{FieldExpr, MeshEvalUpdatable};
use mefikit::tools::meshless::DistanceWeighting;

/// Relative agreement tolerance of the GPU path with the CPU reference.
const TOL: f64 = 1e-4;
/// Number of timed operator builds per case.
const ITERATIONS: u32 = 10;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::wgpu(WgpuDeviceKind::DefaultDevice).unwrap_or_else(|_| Device::default());
    println!("device: {}", device.client().name());
    println!();

    run_2d(&device);
    run_3d(&device);
    Ok(())
}

/// Reports how close the `gpu` result is to the `cpu` reference and the build-time ratio.
fn assess(name: &str, cpu: &mf::FieldOwnedD, gpu: &mf::FieldOwnedD, et: mf::ElementType) {
    let cpu = &cpu.0[&et];
    let gpu = &gpu.0[&et];
    let mut max_abs = 0.0_f64;
    let mut mismatches = 0;
    for (&g, &r) in gpu.iter().zip(cpu.iter()) {
        let err = (g - r).abs();
        max_abs = max_abs.max(err);
        if err > r.abs().max(1.0) * TOL {
            mismatches += 1;
        }
    }
    let status = if mismatches > 0 {
        format!("MISMATCH ({mismatches})")
    } else {
        "ok".to_string()
    };
    println!("{name}: [{status}] max abs diff {max_abs:.3e}");
}

fn bench(mut f: impl FnMut()) -> Duration {
    f();
    let now = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    now.elapsed() / ITERATIONS
}

fn case(
    name: &str,
    et: ElementType,
    build_cpu: &mut dyn FnMut() -> (mf::FieldOwnedD, Duration),
    build_gpu: &mut dyn FnMut() -> (mf::FieldOwnedD, Duration),
) {
    let (cpu, cpu_t) = build_cpu();
    let (gpu, gpu_t) = build_gpu();
    assess(name, &cpu, &gpu, et);
    println!(
        "  build per call: cpu {:.2?}, gpu {:.2?} (speedup {:.2}x)",
        cpu_t,
        gpu_t,
        cpu_t.as_secs_f64() / gpu_t.as_secs_f64()
    );
}

fn run_2d(device: &Device) {
    println!("--- 2D ---");
    let mut src_mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=40).map(|i| i as f64 / 40.0).collect())
        .add_axis((0..=40).map(|i| i as f64 / 40.0).collect())
        .build();
    src_mesh.eval_update_field("T", None, FieldExpr::X - FieldExpr::Y);
    let src = src_mesh.view();
    let field = src_mesh
        .field("T", Some(mf::Dimension::D2))
        .expect("2D temperature field");

    let tgt_mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=24).map(|i| i as f64 / 24.0).collect())
        .add_axis((0..=24).map(|i| i as f64 / 24.0).collect())
        .build();
    let tgt = tgt_mesh.view();

    let method = mf::GradientMethod::MovingLeastSquares {
        k: 12,
        weighting: DistanceWeighting::CompactSupport { exponent: 3.0 },
    };
    case(
        "gradient ML S (2D)",
        ElementType::QUAD4,
        &mut || {
            let cpu_t = bench(|| {
                std::hint::black_box(
                    mf::GradientOperator::new(&src, &tgt, method).apply(&field.view(), 0.0),
                );
            });
            let op = mf::GradientOperator::new(&src, &tgt, method);
            (op.apply(&field.view(), 0.0), cpu_t)
        },
        &mut || {
            let gpu_t = bench(|| {
                std::hint::black_box(
                    mf::GradientOperator::new_gpu(device, &src, &tgt, method)
                        .apply(&field.view(), 0.0),
                );
            });
            let op = mf::GradientOperator::new_gpu(device, &src, &tgt, method);
            (op.apply(&field.view(), 0.0), gpu_t)
        },
    );

    let idw = mf::TransferMethod::InverseDistance {
        k: 8,
        exponent: 2.0,
    };
    case(
        "transfer inverse-distance (2D)",
        ElementType::QUAD4,
        &mut || {
            let cpu_t = bench(|| {
                std::hint::black_box(mf::TransferOperator::new(&src, &tgt, idw).apply(
                    &field.view(),
                    mf::FieldNature::Intensive,
                    0.0,
                ));
            });
            let op = mf::TransferOperator::new(&src, &tgt, idw);
            (
                op.apply(&field.view(), mf::FieldNature::Intensive, 0.0),
                cpu_t,
            )
        },
        &mut || {
            let gpu_t = bench(|| {
                std::hint::black_box(
                    mf::TransferOperator::new_gpu(device, &src, &tgt, idw).apply(
                        &field.view(),
                        mf::FieldNature::Intensive,
                        0.0,
                    ),
                );
            });
            let op = mf::TransferOperator::new_gpu(device, &src, &tgt, idw);
            (
                op.apply(&field.view(), mf::FieldNature::Intensive, 0.0),
                gpu_t,
            )
        },
    );

    let mls = mf::TransferMethod::MovingLeastSquares {
        k: 12,
        weighting: DistanceWeighting::Constant,
    };
    case(
        "transfer ML S (2D)",
        ElementType::QUAD4,
        &mut || {
            let cpu_t = bench(|| {
                std::hint::black_box(mf::TransferOperator::new(&src, &tgt, mls).apply(
                    &field.view(),
                    mf::FieldNature::Intensive,
                    0.0,
                ));
            });
            let op = mf::TransferOperator::new(&src, &tgt, mls);
            (
                op.apply(&field.view(), mf::FieldNature::Intensive, 0.0),
                cpu_t,
            )
        },
        &mut || {
            let gpu_t = bench(|| {
                std::hint::black_box(
                    mf::TransferOperator::new_gpu(device, &src, &tgt, mls).apply(
                        &field.view(),
                        mf::FieldNature::Intensive,
                        0.0,
                    ),
                );
            });
            let op = mf::TransferOperator::new_gpu(device, &src, &tgt, mls);
            (
                op.apply(&field.view(), mf::FieldNature::Intensive, 0.0),
                gpu_t,
            )
        },
    );
}

fn run_3d(device: &Device) {
    println!("--- 3D ---");
    let mut src_mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=16).map(|i| i as f64 / 16.0).collect())
        .add_axis((0..=16).map(|i| i as f64 / 16.0).collect())
        .add_axis((0..=16).map(|i| i as f64 / 16.0).collect())
        .build();
    src_mesh.eval_update_field("T", None, FieldExpr::X - FieldExpr::Y + FieldExpr::Z);
    let src = src_mesh.view();
    let field = src_mesh
        .field("T", Some(mf::Dimension::D3))
        .expect("3D temperature field");

    let tgt_mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect())
        .build();
    let tgt = tgt_mesh.view();

    let method = mf::GradientMethod::MovingLeastSquares {
        k: 16,
        weighting: DistanceWeighting::Gaussian,
    };

    case(
        "gradient ML S (3D)",
        ElementType::HEX8,
        &mut || {
            let cpu_t = bench(|| {
                std::hint::black_box(
                    mf::GradientOperator::new(&src, &tgt, method).apply(&field.view(), 0.0),
                );
            });
            let op = mf::GradientOperator::new(&src, &tgt, method);
            (op.apply(&field.view(), 0.0), cpu_t)
        },
        &mut || {
            let gpu_t = bench(|| {
                std::hint::black_box(
                    mf::GradientOperator::new_gpu(device, &src, &tgt, method)
                        .apply(&field.view(), 0.0),
                );
            });
            let op = mf::GradientOperator::new_gpu(device, &src, &tgt, method);
            (op.apply(&field.view(), 0.0), gpu_t)
        },
    );

    let idw = mf::TransferMethod::InverseDistance {
        k: 16,
        exponent: 3.0,
    };
    let cpu = {
        let op = mf::TransferOperator::new(&src, &tgt, idw);
        op.apply(&field.view(), mf::FieldNature::Intensive, 0.0)
    };
    let gpu = {
        let op = mf::TransferOperator::new_gpu(device, &src, &tgt, idw);
        op.apply(&field.view(), mf::FieldNature::Intensive, 0.0)
    };
    assess(
        "transfer inverse-distance (3D)",
        &cpu,
        &gpu,
        mf::ElementType::HEX8,
    );
}
