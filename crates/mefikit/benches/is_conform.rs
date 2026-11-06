//! Benchmarks for [`mefikit::tools::is_conform`].
//!
//! Two families are measured: a conforming hexahedral grid, and two stacked grids with
//! different interface tessellations. Each is run with the full checks and with the fast
//! checks, so the cost of the two cell-pair searches is visible.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use std::time::Duration;

use mefikit::mesh::UMesh;
use mefikit::prelude as mf;

const TOL: f64 = 1e-9;

fn axis(n: usize) -> Vec<f64> {
    (0..=n).map(|i| i as f64 / n as f64).collect()
}

fn conforming_hex(n: usize) -> UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis(axis(n))
        .add_axis(axis(n))
        .add_axis(axis(n))
        .build()
}

/// Two stacked grids whose interface is tessellated differently. The interface nodes are
/// duplicated by `concat`, so only the geometric face search can see the mismatch.
fn nonconforming_interface(n: usize, m: usize) -> UMesh {
    let bottom = mf::RegularUMeshBuilder::new()
        .add_axis(axis(n))
        .add_axis(axis(n))
        .add_axis(vec![0.0, 1.0])
        .build();
    let top = mf::RegularUMeshBuilder::new()
        .add_axis(axis(m))
        .add_axis(axis(m))
        .add_axis(vec![1.0, 2.0])
        .build();
    mf::concat(&bottom.view(), &top.view()).unwrap()
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("is_conform");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(2));

    for n in [4, 8, 12] {
        let mesh = conforming_hex(n);
        let cells = mesh.num_elements();
        group.bench_with_input(
            BenchmarkId::new("full/conforming", cells),
            &mesh,
            |b, mesh| {
                b.iter(|| mf::is_conform(black_box(&mesh.view()), TOL, None));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fast/conforming", cells),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    mf::is_conform_with(
                        black_box(&mesh.view()),
                        TOL,
                        mf::ConformanceOptions::fast(),
                    )
                });
            },
        );
    }

    for (n, m) in [(4, 6), (8, 12)] {
        let mesh = nonconforming_interface(n, m);
        let cells = mesh.num_elements();
        group.throughput(Throughput::Elements(cells as u64));
        group.bench_with_input(
            BenchmarkId::new("full/nonconforming_interface", cells),
            &mesh,
            |b, mesh| {
                b.iter(|| mf::is_conform(black_box(&mesh.view()), TOL, None));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("fast/nonconforming_interface", cells),
            &mesh,
            |b, mesh| {
                b.iter(|| {
                    mf::is_conform_with(
                        black_box(&mesh.view()),
                        TOL,
                        mf::ConformanceOptions::fast(),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
