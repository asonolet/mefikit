//! Benchmarks for the conservative P0 transfer (`ConservativeP0Transfer`) on 3D hexa meshes.
//!
//! Measures the two phases separately:
//!
//! - `build`: the one-time overlap precompute (BVH broad phase + `convex_intersection_volume`
//!   narrow phase on each candidate pair), which grows with the product of the meshes.
//! - `apply`: the sparse matrix-vector product that a single field evaluation costs, independent
//!   of the mesh sizes once the operator is built.

use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use mefikit::mesh::{Dimension, ElementType, FieldOwnedD, UMesh};
use mefikit::tools::grid::RegularUMeshBuilder;
use mefikit::tools::transfer::{ConservativeP0Transfer, FieldNature, Transfer};

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
                    std::hint::black_box(mefikit::tools::transfer::ConservativeP0Transfer::new(
                        &src.view(),
                        &tgt.view(),
                    ))
                });
            },
        );

        let op = ConservativeP0Transfer::new(&src.view(), &tgt.view());
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

criterion_group!(bench, conservative_transfer_3d);
criterion_main!(bench);
