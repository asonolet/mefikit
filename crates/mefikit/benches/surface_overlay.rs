use criterion::{BatchSize, BenchmarkId, Criterion};
use ndarray as nd;
use scaling::bench_scaling_gen;
use std::hint::black_box;

use mefikit::mesh::{ElementType, UMesh};
use mefikit::tools::overlay_surfaces;

fn grid_coords(n: usize) -> nd::ArcArray2<f64> {
    let mut c = nd::Array2::<f64>::zeros(((n + 1) * (n + 1), 3));
    let mut k = 0;
    for j in 0..=n {
        for i in 0..=n {
            c[(k, 0)] = i as f64 / n as f64;
            c[(k, 1)] = j as f64 / n as f64;
            k += 1;
        }
    }
    c.into_shared()
}

fn nid(i: usize, j: usize, n: usize) -> usize {
    j * (n + 1) + i
}

/// Structured `n x n` quad surface over the unit square at z = 0.
fn quad_surface(n: usize) -> UMesh {
    let mut m = UMesh::new(grid_coords(n));
    let mut flat = Vec::new();
    for j in 0..n {
        for i in 0..n {
            flat.extend([
                nid(i, j, n),
                nid(i + 1, j, n),
                nid(i + 1, j + 1, n),
                nid(i, j + 1, n),
            ]);
        }
    }
    let conn = nd::Array2::from_shape_vec((n * n, 4), flat).unwrap();
    m.add_regular_block(ElementType::QUAD4, conn.into_shared(), None);
    m
}

/// Structured triangulated surface over the same footprint.
fn tri_surface(n: usize) -> UMesh {
    let mut m = UMesh::new(grid_coords(n));
    let mut flat = Vec::new();
    for j in 0..n {
        for i in 0..n {
            flat.extend([nid(i, j, n), nid(i + 1, j, n), nid(i + 1, j + 1, n)]);
            flat.extend([nid(i, j, n), nid(i + 1, j + 1, n), nid(i, j + 1, n)]);
        }
    }
    let conn = nd::Array2::from_shape_vec((2 * n * n, 3), flat).unwrap();
    m.add_regular_block(ElementType::TRI3, conn.into_shared(), None);
    m
}

fn surface_overlay(c: &mut Criterion) {
    let mut group = c.benchmark_group("surface_overlay");

    for n in [8, 16, 32, 64] {
        group.bench_with_input(BenchmarkId::new("tri_vs_quad", n * n), &n, |b, &n| {
            b.iter_batched(
                || (quad_surface(n), tri_surface(n)),
                |(s1, s2)| {
                    let out = overlay_surfaces(black_box(&s1.view()), black_box(&s2.view()), 1e-9)
                        .unwrap();
                    black_box(out)
                },
                BatchSize::LargeInput,
            )
        });
    }
    group.finish();
}

fn main() {
    let mut c = Criterion::default();
    surface_overlay(&mut c);

    let benched = bench_scaling_gen(
        |n| {
            let m = (n as f64).sqrt() as usize;
            (quad_surface(m), tri_surface(m))
        },
        |(s1, s2)| {
            overlay_surfaces(&s1.view(), &s2.view(), 1e-9).unwrap();
        },
        10000,
    );
    println!("surface overlay scaling bench: {benched}");
}
