use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

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
                std::hint::black_box(mesh.compute_submesh(None));
            })
        });
    }
}

fn selection_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection");

    for i in [2, 10, 40] {
        let mesh = RegularUMeshBuilder::new()
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    mesh.select_ids()
                        .centroids()
                        .in_sphere(&[0.5, 0.5, 0.5], 0.25),
                );
            })
        });
    }
}

fn measure2(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure2");

    for i in [2, 10, 100] {
        let mesh = RegularUMeshBuilder::new()
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(mesh.measure());
            })
        });
    }
}

criterion_group!(bench, submesh, selection_sphere, measure2);
criterion_main!(bench);
