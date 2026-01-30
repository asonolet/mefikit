use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::prelude as mf;

fn snap(c: &mut Criterion) {
    let mut group = c.benchmark_group("snap");

    for i in [4, 60, 100] {
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter_batched(
                || {
                    let m1 = mf::RegularUMeshBuilder::new()
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .build();
                    let m2 = m1.clone();
                    (m1, m2)
                },
                |(m1, m2)| {
                    let mut m1 = m1.clone();
                    let m2_view = m2.view();
                    std::hint::black_box(mf::snap(&mut m1, m2_view, 1e-12));
                },
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(bench, snap);
criterion_main!(bench);
