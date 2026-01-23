use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::prelude as mf;

fn selection_sphere(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection");

    for i in [2, 20, 30] {
        let mesh = mf::RegularUMeshBuilder::new()
            .add_axis((0..=i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..=i).map(|k| (k as f64) / (i as f64)).collect())
            .add_axis((0..=i).map(|k| (k as f64) / (i as f64)).collect())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i * i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(
                    mf::Selector::new(&mesh)
                        .centroids()
                        .in_sphere(&[0.5, 0.5, 0.5], 0.25),
                );
            })
        });
    }
}

criterion_group!(bench, selection_sphere,);
criterion_main!(bench);
