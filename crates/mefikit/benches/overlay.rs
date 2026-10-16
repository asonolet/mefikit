//! Benchmarks for [`mefikit::tools::overlay`] on 2D meshes.
//!
//! The fixtures are deliberately small so that a full run stays in the second range: the point
//! is to compare revisions of the algorithm, not to measure the absolute throughput of a solver.
//!
//! Only [`OverlayOperation::Imprint`] is measured: it is the operation that runs the whole
//! `weld -> compute_intersections -> to_sorted_intersections -> cut_cells_all` pipeline, and the
//! only one that will ever apply to quadratic elements.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;

use mefikit::mesh::{Dimension, UMesh};
use mefikit::tools::{OverlayOperation, Overlayable, RegularUMeshBuilder};

/// Regular `n x n` grid of QUAD4 elements over the rectangle `[x0, x1] x [y0, y1]`.
fn regular(n: usize, x0: f64, x1: f64, y0: f64, y1: f64) -> UMesh {
    let axis = |lo: f64, hi: f64| {
        (0..=n)
            .map(|i| lo + (hi - lo) * i as f64 / n as f64)
            .collect::<Vec<f64>>()
    };
    RegularUMeshBuilder::new()
        .add_axis(axis(x0, x1))
        .add_axis(axis(y0, y1))
        .build()
}

/// Rotates the coordinates of a mesh by `angle` around `(cx, cy)`, keeping the element types.
fn rotate(mut mesh: UMesh, angle: f64, cx: f64, cy: f64) -> UMesh {
    let (s, c) = (angle.sin(), angle.cos());
    for mut row in mesh.coords_mut().outer_iter_mut() {
        let (x, y) = (row[0] - cx, row[1] - cy);
        row[0] = cx + c * x - s * y;
        row[1] = cy + s * x + c * y;
    }
    mesh
}

fn subject_cells(mesh: &UMesh) -> u64 {
    mesh.num_elements_of_dim(Dimension::D2) as u64
}

/// Builds one `(subject, cutting)` pair for a grid of `n x n` cells.
type CaseFactory = fn(usize) -> (UMesh, UMesh);

/// The cutting mesh is nested entirely inside the subject: a clean interior cut.
fn nested(n: usize) -> (UMesh, UMesh) {
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        regular(n, 0.25, 0.75, 0.25, 0.75),
    )
}

/// The cutting mesh is shifted by half a cell, so every subject edge is crossed and the
/// cutting mesh straddles two sides of the subject domain.
fn half_shift(n: usize) -> (UMesh, UMesh) {
    let h = 0.5 / n as f64;
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        regular(n, h, 1.0 + h, h, 1.0 + h),
    )
}

/// The cell centres of the cutting mesh land on the subject nodes: the maximum number of
/// intersections, and by far the heaviest case.
fn staggered(n: usize) -> (UMesh, UMesh) {
    let h = 0.5 / n as f64;
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        regular(n + 1, -h, 1.0 + h, -h, 1.0 + h),
    )
}

/// The cutting mesh is scaled up and rotated in plane: slanted cut lines that never align
/// with the subject ones.
fn rotated(n: usize) -> (UMesh, UMesh) {
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        rotate(
            regular(n, -0.1, 1.1, -0.1, 1.1),
            std::f64::consts::FRAC_PI_8,
            0.5,
            0.5,
        ),
    )
}

/// A finer cutting mesh over an interior patch: many small pieces.
fn fine_in_coarse(n: usize) -> (UMesh, UMesh) {
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        regular(2 * n, 0.2, 0.8, 0.2, 0.8),
    )
}

/// The two meshes do not overlap at all: pure weld and bounding-volume-tree overhead.
fn disjoint(n: usize) -> (UMesh, UMesh) {
    (
        regular(n, 0.0, 1.0, 0.0, 1.0),
        regular(n, 1.25, 2.25, 1.25, 2.25),
    )
}

fn run(subject: &UMesh, cutting: &UMesh) {
    // `overlay` consumes `mesh2`, so the clone is part of the measured end-to-end cost, exactly
    // as in `examples/overlay.rs`.
    black_box(subject.overlay(black_box(cutting.clone()), OverlayOperation::Imprint));
}

fn overlay_bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlay");

    let cases: [(&str, CaseFactory); 6] = [
        ("nested", nested),
        ("half_shift", half_shift),
        ("staggered", staggered),
        ("rotated", rotated),
        ("fine_in_coarse", fine_in_coarse),
        ("disjoint", disjoint),
    ];
    for n in [16, 64] {
        for (name, make) in cases {
            // The staggered case is by far the heaviest, so it only runs on the coarse grid.
            if name == "staggered" && n > 16 {
                continue;
            }
            let (subject, cutting) = make(n);
            group.throughput(Throughput::Elements(subject_cells(&subject)));
            group.bench_with_input(
                BenchmarkId::new("imprint", format!("{name}/{n}")),
                &(&subject, &cutting),
                |b, (subject, cutting)| b.iter(|| run(subject, cutting)),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, overlay_bench);
criterion_main!(benches);
