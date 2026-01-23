use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::prelude as mf;

fn measure2(c: &mut Criterion) {
    let mut group = c.benchmark_group("measure2");

    for i in [4, 60, 100] {
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter_batched(
                || {
                    mf::RegularUMeshBuilder::new()
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
                        .build()
                },
                |mesh| {
                    let view = mesh.view();
                    std::hint::black_box(mf::measure(view));
                },
                BatchSize::LargeInput,
            )
        });
    }
}

criterion_group!(bench, measure2,);
criterion_main!(bench);
