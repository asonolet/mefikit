use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use mefikit::compute_submesh;
use mefikit::RegularUMeshBuilder;

fn submesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("submesh");

    for i in [2, 10, 100] {
        let mesh = RegularUMeshBuilder::new()
            .add_axis((0..i).map(|i| i as f64).collect::<Vec<f64>>())
            .add_axis((0..i).map(|i| i as f64).collect::<Vec<f64>>())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(compute_submesh(&mesh, None));
            })
        });
    }
}

criterion_group!(bench, submesh);
criterion_main!(bench);
