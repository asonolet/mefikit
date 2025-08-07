use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit as mf;

fn submesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("submesh");

    for i in [2, 10, 100] {
        let mesh = mf::RegularUMeshBuilder::new()
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
        let mesh = mf::RegularUMeshBuilder::new()
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    mf::Selector::new(mesh.view())
                        .centroids()
                        .in_sphere(&[0.5, 0.5, 0.5], 0.25),
                );
            })
        });
    }
}

fn take_view(c: &mut Criterion) {
    let mut group = c.benchmark_group("take_view");

    for i in [4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("mesh_size", i), &i, |b, _| {
            b.iter_batched(
                || {
                    RegularUMeshBuilder::new()
                        .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
                        .add_axis((0..i).map(|k| (k as f64) / (i as f64)).collect())
                        .build()
                },
                |data| {
                    std::hint::black_box(data.view());
                },
                BatchSize::LargeInput,
            )
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

criterion_group!(bench, submesh, selection_sphere, measure2, take_view);
criterion_main!(bench);
