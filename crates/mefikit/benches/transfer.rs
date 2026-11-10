//! Benchmarks for the conservative P0 transfer (`TransferMethod::ConservativeP0`) on 3D hexa
//! meshes.
//!
//! Measures the two phases separately:
//!
//! - `build`: the one-time overlap precompute (BVH broad phase + geometric narrow phase on each
//!   candidate pair), which grows with the product of the meshes.
//! - `apply`: the sparse matrix-vector product that a single field evaluation costs, independent
//!   of the mesh sizes once the operator is built.
//!
//! Both the same-dimensional `3D → 3D` transfer and the cross-dimensional `3D → 2D` / `3D → 1D`
//! transfers are benchmarked.

use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::mesh::{Dimension, ElementType, FieldOwnedD, UMesh};
use mefikit::tools::grid::RegularUMeshBuilder;
use mefikit::tools::transfer::{FieldNature, Transfer, TransferMethod, TransferOperator};

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

/// Attaches a constant scalar field of `value` to the HEX8 cells.
fn with_constant_field(mesh: &mut UMesh, value: f64) {
    let n = mesh
        .block(ElementType::HEX8)
        .expect("expected a HEX8 mesh")
        .len();
    let field = FieldOwnedD::new(BTreeMap::from([(
        ElementType::HEX8,
        ndarray::Array::from_elem(ndarray::IxDyn(&[n]), value),
    )]));
    mesh.update_field("f", field.into_shared());
}

/// Structured `n^2` QUAD4 surface mesh in the plane `z`, shifted by a fraction of a cell in `x`
/// and `y` so that it is not conforming with a regular `n^3` hexa source mesh.
fn plane_quad_mesh(n: usize, z: f64) -> UMesh {
    let shift_x = 0.37 / n as f64;
    let shift_y = 0.19 / n as f64;
    let xs: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64 + shift_x).collect();
    let ys: Vec<f64> = (0..=n).map(|i| i as f64 / n as f64 + shift_y).collect();
    let coords: Vec<f64> = ys
        .iter()
        .flat_map(|&y| xs.iter().map(move |&x| [x, y, z]))
        .flatten()
        .collect();
    let mut mesh =
        UMesh::new(ndarray::ArcArray2::from_shape_vec(((n + 1) * (n + 1), 3), coords).unwrap());
    let mut conn = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let n0 = j * (n + 1) + i;
            conn.push([n0, n0 + 1, n0 + n + 2, n0 + n + 1]);
        }
    }
    mesh.add_regular_block(
        ElementType::QUAD4,
        ndarray::Array2::from_shape_vec((n * n, 4), conn.into_iter().flatten().collect())
            .unwrap()
            .to_shared(),
        None,
    );
    mesh
}

/// Structured `n`-segment SEG2 polyline through a regular `n^3` hexa source mesh, shifted so
/// that its endpoints are not aligned with the source grid planes.
fn line_seg_mesh(n: usize) -> UMesh {
    let shift = 0.37 / n as f64;
    let coords: Vec<f64> = (0..=n)
        .flat_map(|i| [i as f64 / n as f64 + shift, 0.43, 0.37])
        .collect();
    let mut mesh = UMesh::new(ndarray::ArcArray2::from_shape_vec((n + 1, 3), coords).unwrap());
    let conn: Vec<[usize; 2]> = (0..n).map(|i| [i, i + 1]).collect();
    mesh.add_regular_block(
        ElementType::SEG2,
        ndarray::Array2::from_shape_vec((n, 2), conn.into_iter().flatten().collect())
            .unwrap()
            .to_shared(),
        None,
    );
    mesh
}

fn conservative_transfer_3d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conservative_transfer_3d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    // The target mesh overlaps the source with a half-cell shift, like two unstructured meshes
    // in a coupling use case. Each target cell then overlaps up to eight source cells.
    for i in [4, 8, 12, 16] {
        let mut src = hex_mesh(i, 0.0, 1.0);
        let shift = 0.5 / (i as f64);
        let tgt = hex_mesh(i, shift, 1.0);
        with_constant_field(&mut src, 7.0);

        group.bench_with_input(
            BenchmarkId::new("build", i * i * i),
            &(&src, &tgt),
            |b, (src, tgt)| {
                b.iter(|| {
                    std::hint::black_box(mefikit::tools::transfer::TransferOperator::new(
                        &src.view(),
                        &tgt.view(),
                        mefikit::tools::transfer::TransferMethod::ConservativeP0,
                    ))
                });
            },
        );

        let op = TransferOperator::new(&src.view(), &tgt.view(), TransferMethod::ConservativeP0);
        let field = src.field("f", Some(Dimension::D3)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("apply", i * i * i),
            &(&op, &field),
            |b, (op, field)| {
                b.iter(|| {
                    std::hint::black_box(op.apply(field, FieldNature::Intensive, 0.0));
                });
            },
        );
    }

    group.finish();
}

fn conservative_transfer_3d_to_2d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conservative_transfer_3d_to_2d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for i in [4, 8, 12, 16] {
        let mut src = hex_mesh(i, 0.0, 1.0);
        let tgt = plane_quad_mesh(i, 0.4);
        with_constant_field(&mut src, 7.0);

        group.bench_with_input(
            BenchmarkId::new("build", i * i),
            &(&src, &tgt),
            |b, (src, tgt)| {
                b.iter(|| {
                    std::hint::black_box(TransferOperator::new(
                        &src.view(),
                        &tgt.view(),
                        TransferMethod::ConservativeP0,
                    ))
                });
            },
        );

        let op = TransferOperator::new(&src.view(), &tgt.view(), TransferMethod::ConservativeP0);
        let field = src.field("f", Some(Dimension::D3)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("apply", i * i),
            &(&op, &field),
            |b, (op, field)| {
                b.iter(|| {
                    std::hint::black_box(op.apply(field, FieldNature::Intensive, 0.0));
                });
            },
        );
    }

    group.finish();
}

fn conservative_transfer_3d_to_1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conservative_transfer_3d_to_1d");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));

    for i in [4, 8, 12, 16] {
        let mut src = hex_mesh(i, 0.0, 1.0);
        let tgt = line_seg_mesh(i);
        with_constant_field(&mut src, 7.0);

        group.bench_with_input(
            BenchmarkId::new("build", i),
            &(&src, &tgt),
            |b, (src, tgt)| {
                b.iter(|| {
                    std::hint::black_box(TransferOperator::new(
                        &src.view(),
                        &tgt.view(),
                        TransferMethod::ConservativeP0,
                    ))
                });
            },
        );

        let op = TransferOperator::new(&src.view(), &tgt.view(), TransferMethod::ConservativeP0);
        let field = src.field("f", Some(Dimension::D3)).unwrap();
        group.bench_with_input(
            BenchmarkId::new("apply", i),
            &(&op, &field),
            |b, (op, field)| {
                b.iter(|| {
                    std::hint::black_box(op.apply(field, FieldNature::Intensive, 0.0));
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    bench,
    conservative_transfer_3d,
    conservative_transfer_3d_to_2d,
    conservative_transfer_3d_to_1d
);
criterion_main!(bench);
