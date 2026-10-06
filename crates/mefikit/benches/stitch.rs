//! Benchmarks for [`mefikit::tools::stitch`].
//!
//! The fixtures are deliberately small so that a full run stays in the second range: the point
//! is to compare revisions of the algorithm, not to measure the absolute throughput of a solver.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use mefikit::mesh::{ElementLike, UMesh};
use mefikit::tools::{RegularUMeshBuilder, stitch};

const TOL: f64 = 1e-9;

/// Uniform `nx * ny * nz` grid of hexahedra over the box of the given extents.
fn block(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis(xs.to_vec())
        .add_axis(ys.to_vec())
        .add_axis(zs.to_vec())
        .build()
}

/// `n` equal intervals on `[lo, hi]`.
fn axis(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

fn hex_count(meshes: &[UMesh]) -> usize {
    meshes.iter().map(|m| m.num_elements()).sum()
}

/// Two grids stacked along z whose faces match node for node: no imprint, pure overhead.
fn matched(n: usize) -> Vec<UMesh> {
    vec![
        block(&axis(0.0, 1.0, n), &axis(0.0, 1.0, n), &axis(0.0, 1.0, n)),
        block(&axis(0.0, 1.0, n), &axis(0.0, 1.0, n), &axis(1.0, 2.0, n)),
    ]
}

/// Two grids stacked along z whose in-plane lines are offset by half a cell.
fn offset_lines(n: usize) -> Vec<UMesh> {
    let h = 0.5 / n as f64;
    vec![
        block(&axis(0.0, 1.0, n), &axis(0.0, 1.0, n), &axis(0.0, 1.0, n)),
        block(
            &axis(h, 1.0 + h, n),
            &axis(h, 1.0 + h, n),
            &axis(1.0, 2.0, n),
        ),
    ]
}

/// Two grids stacked along z with the upper one rotated in plane: slanted imprint lines.
fn rotated_in_plane(n: usize) -> Vec<UMesh> {
    let bottom = block(&axis(0.0, 1.0, n), &axis(0.0, 1.0, n), &axis(0.0, 1.0, n));
    let top = block(&axis(0.0, 1.0, n), &axis(0.0, 1.0, n), &axis(1.0, 2.0, n));
    let mut coords = top.coords().to_owned();
    let (s, c) = (
        std::f64::consts::FRAC_PI_8.sin(),
        std::f64::consts::FRAC_PI_8.cos(),
    );
    for mut row in coords.outer_iter_mut() {
        let (x, y) = (row[0] - 0.5, row[1] - 0.5);
        row[0] = 0.5 + c * x - s * y;
        row[1] = 0.5 + s * x + c * y;
    }
    let mut rotated = UMesh::new(coords.into_shared());
    for cell in top.elements() {
        rotated.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
    }
    vec![bottom, rotated]
}

/// A wall of `k` blocks side by side along x, each subdivided along z: every block is imprinted
/// on its two neighbours, so there are many small regions.
fn wall(k: usize, n: usize) -> Vec<UMesh> {
    (0..k)
        .map(|i| {
            block(
                &axis(i as f64, i as f64 + 1.0, n),
                &axis(0.0, 1.0, n),
                &axis(0.0, 1.0, 2 * n),
            )
        })
        .collect()
}

/// Several blocks meeting along two different directions, the case of the third block touching
/// two others on distinct faces.
fn three_way(n: usize) -> Vec<UMesh> {
    vec![
        block(&axis(0.0, 2.0, n), &axis(0.0, 2.0, n), &axis(0.0, 1.0, n)),
        block(&axis(0.0, 2.0, n), &axis(0.0, 2.0, n), &axis(1.0, 2.0, n)),
        block(&axis(2.0, 3.0, n), &axis(0.0, 2.0, n), &axis(0.0, 2.0, n)),
    ]
}

/// A coarse volume split into eight blocks, each refined differently, then stitched back.
fn eight_blocks(n: usize) -> Vec<UMesh> {
    let mut meshes = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                // Refine every other block so that the interfaces need real imprinting.
                let m = if (i + j + k) % 2 == 0 { n } else { 2 * n };
                meshes.push(block(
                    &axis(i as f64, i as f64 + 1.0, m),
                    &axis(j as f64, j as f64 + 1.0, m),
                    &axis(k as f64, k as f64 + 1.0, m),
                ));
            }
        }
    }
    meshes
}

/// Builds the meshes of one case for a grid of `n x n` cells.
type MeshFactory = fn(usize) -> Vec<UMesh>;

fn run(meshes: &[UMesh]) {
    let views: Vec<_> = meshes.iter().map(|m| m.view()).collect();
    black_box(stitch(black_box(&views), TOL).unwrap());
}

fn stitch_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("stitch");

    let cases: [(&str, MeshFactory); 6] = [
        ("matched", matched),
        ("offset_lines", offset_lines),
        ("rotated_in_plane", rotated_in_plane),
        ("wall_4", |n| wall(4, n)),
        ("three_way", three_way),
        ("eight_blocks", eight_blocks),
    ];
    for n in [4, 8] {
        for (name, make) in cases {
            // The eight-block case is by far the heaviest, so it only runs on the coarse grid.
            if name == "eight_blocks" && n > 4 {
                continue;
            }
            let meshes = make(n);
            group.throughput(Throughput::Elements(hex_count(&meshes) as u64));
            group.bench_with_input(BenchmarkId::new(name, n), &meshes, |b, meshes| {
                b.iter(|| run(meshes))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, stitch_bench);
criterion_main!(benches);
