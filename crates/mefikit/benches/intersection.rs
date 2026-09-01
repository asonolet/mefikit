//! Benchmarks for `Polyhedron::convex_intersection_volume` on hexa-mesh polyhedra.
//!
//! Two overlapping `n x n x n` HEX8 cartesian meshes (`RegularUMeshBuilder`) are converted to
//! per-cell convex polyhedra via `ElementGeo::to_polyhedron` (the `polyze` pipeline), and the
//! intersector is measured over the resulting cell pairs: per-call latency on overlapping cells,
//! aggregate throughput over the pair product, and the AABB fast path on disjoint cells.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::element_traits::ElementGeo;
use mefikit::geometry::Polyhedron;
use mefikit::mesh::UMesh;
use mefikit::tools::grid::RegularUMeshBuilder;

/// Structured `n^3` HEX8 mesh on `[x0, x0 + width] x [0, 1] x [0, 1]`.
fn hex_mesh(n: usize, x0: f64, width: f64) -> UMesh {
    let x = (0..=n).map(|k| x0 + width * (k as f64) / (n as f64));
    let y = (0..=n).map(|k| (k as f64) / (n as f64));
    let z = (0..=n).map(|k| (k as f64) / (n as f64));
    RegularUMeshBuilder::new()
        .add_axis(x.collect())
        .add_axis(y.collect())
        .add_axis(z.collect())
        .build()
}

fn polyhedra(mesh: &UMesh) -> Vec<Polyhedron> {
    mesh.elements().map(|e| e.to_polyhedron()).collect()
}

fn intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("convex_intersection_volume");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    // Mesh B overlaps A with a half-cell shift in x, y and z, so most cells overlap several
    // neighbours: this mirrors two unstructured hexa meshes in a transfer/contact use case.
    for i in [4, 8, 12] {
        let mesh_a = hex_mesh(i, 0.0, 1.0);
        let shift = 0.5 / (i as f64);
        let mesh_b = hex_mesh(i, shift, 1.0);
        let pa = polyhedra(&mesh_a);
        let pb = polyhedra(&mesh_b);
        assert_eq!(pa.len(), i * i * i);

        // Per-call latency on a representative overlapping pair (interior cells of both meshes).
        let mid = i * i * i / 2;
        group.bench_with_input(
            BenchmarkId::new("single_overlapping_pair", i * i * i),
            &(mid, &pa, &pb),
            |b, (mid, pa, pb)| {
                b.iter(|| std::hint::black_box(pa[*mid].convex_intersection_volume(&pb[*mid])));
            },
        );

        // AABB fast path: disjoint cells must return immediately.
        let p_far = polyhedra(&hex_mesh(i, 5.0, 1.0)); // far away in x
        let a0 = &pa[0];
        let far0 = &p_far[0];
        group.bench_with_input(
            BenchmarkId::new("disjoint_pair", i * i * i),
            &(a0, far0),
            |b, (a, far)| {
                b.iter(|| {
                    std::hint::black_box(a.convex_intersection_volume(far));
                });
            },
        );
    }

    // Throughput over the full pair product, bounded to a size where the count stays reasonable
    // (n=8 → 8^6 = 262k calls per iteration).
    for i in [4, 6, 8] {
        let mesh_a = hex_mesh(i, 0.0, 1.0);
        let mesh_b = hex_mesh(i, 0.5 / (i as f64), 1.0);
        let pa = polyhedra(&mesh_a);
        let pb = polyhedra(&mesh_b);

        group.bench_with_input(
            BenchmarkId::new("all_pairs", i * i * i),
            &(i, &pa, &pb),
            |b, (_n, pa, pb)| {
                b.iter(|| {
                    let mut total = 0.0;
                    for a in pa.iter() {
                        for bb in pb.iter() {
                            total += std::hint::black_box(a.convex_intersection_volume(bb));
                        }
                    }
                    std::hint::black_box(total);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(bench, intersection);
criterion_main!(bench);
