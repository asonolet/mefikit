use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::prelude as mf;

fn submesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("submesh");

    for i in [4, 60, 100] {
        let mesh = mf::RegularUMeshBuilder::new()
            .add_axis((0..(i + 1)).map(|i| i as f64).collect::<Vec<f64>>())
            .add_axis((0..(i + 1)).map(|i| i as f64).collect::<Vec<f64>>())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(mf::compute_submesh(&mesh, None, None));
            })
        });
    }
}

fn neighbours(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbours");

    for i in [4, 60, 100] {
        let mesh = mf::RegularUMeshBuilder::new()
            .add_axis((0..=i).map(|i| i as f64).collect::<Vec<f64>>())
            .add_axis((0..=i).map(|i| i as f64).collect::<Vec<f64>>())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(mf::compute_neighbours(&mesh, None, None));
            })
        });
    }
}

#[cfg(feature = "rayon")]
fn par_neighbours(c: &mut Criterion) {
    let mut group = c.benchmark_group("par_neighbours");

    for i in [4, 60, 100] {
        let mesh = mf::RegularUMeshBuilder::new()
            .add_axis((0..=i).map(|i| i as f64).collect::<Vec<f64>>())
            .add_axis((0..=i).map(|i| i as f64).collect::<Vec<f64>>())
            .build();
        group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
            b.iter(|| {
                std::hint::black_box(mf::par_compute_neighbours(&mesh, None, None));
            })
        });
    }
}

criterion_group!(bench, submesh, neighbours,);
criterion_main!(bench);
