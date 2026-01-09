use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::prelude as mf;

fn crack(c: &mut Criterion) {
    let mut group = c.benchmark_group("crack");

    for i in [4, 60, 100] {
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter_batched(
                || {
                    let m1 = mf::RegularUMeshBuilder::new()
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .build();
                    let cut = mf::compute_submesh(&m1, None, None);
                    (m1, cut)
                },
                |(m1, cut)| {
                    let cut_view = cut.view();
                    std::hint::black_box(mf::crack(m1, cut_view));
                },
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(bench, crack);
criterion_main!(bench);
