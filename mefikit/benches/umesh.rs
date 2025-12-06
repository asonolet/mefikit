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

// fn take_view(c: &mut Criterion) {
//     let mut group = c.benchmark_group("take_view");
//
//     for i in [4, 12, 20] {
//         group.bench_with_input(BenchmarkId::new("mesh_size", i * i), &i, |b, _| {
//             b.iter_batched(
//                 || {
//                     mf::RegularUMeshBuilder::new()
//                         .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
//                         .add_axis((0..(i + 1)).map(|k| (k as f64) / (i as f64)).collect())
//                         .build()
//                 },
//                 |data| {
//                     std::hint::black_box(data.view());
//                 },
//                 BatchSize::LargeInput,
//             )
//         });
//     }
// }

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

criterion_group!(
    bench,
    submesh,
    neighbours,
    // par_neighbours,
    selection_sphere,
    measure2
);
criterion_main!(bench);
