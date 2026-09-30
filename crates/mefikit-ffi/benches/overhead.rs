//! What the bindings cost on top of mefikit itself.
//!
//! Each benchmark runs the core operation and the binding that wraps it, on
//! identical data in the same process, so the difference is the overhead of the
//! binding and not a difference in the work being done. The point is not the
//! absolute numbers but the ratio, at sizes where a user would care.

use std::collections::BTreeMap;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mefikit::mesh::{ElementType as CoreElementType, FieldOwnedD, UMesh as CoreUMesh};
use mefikit::tools::transfer::{Transfer, TransferOperator as CoreOperator};

use mefikit_ffi::bridge::{Checks, ElementType, FieldBlock, FieldNature, TransferMethod};
use mefikit_ffi::{TransferOperator, UMesh, set_checks};

/// Structured `n^3` HEX8 mesh, built once and shared by every benchmark.
fn hex_mesh_data(n: usize) -> (Vec<f64>, Vec<usize>, usize) {
    let mut coords = Vec::with_capacity((n + 1).pow(3) * 3);
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                coords.push(i as f64 / n as f64);
                coords.push(j as f64 / n as f64);
                coords.push(k as f64 / n as f64);
            }
        }
    }
    let stride = (n + 1) * (n + 1);
    let plane = n + 1;
    let mut conn = Vec::with_capacity(n * n * n * 8);
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                let b = k * stride + j * plane + i;
                conn.extend_from_slice(&[
                    b,
                    b + 1,
                    b + plane + 1,
                    b + plane,
                    b + stride,
                    b + stride + 1,
                    b + stride + plane + 1,
                    b + stride + plane,
                ]);
            }
        }
    }
    (coords, conn, n * n * n)
}

fn ffi_coords(coords: &[f64]) -> Box<UMesh> {
    UMesh::from_coords(coords, coords.len() / 3, 3).unwrap()
}

fn core_coords(coords: &[f64]) -> CoreUMesh {
    let array = ndarray::Array2::from_shape_vec((coords.len() / 3, 3), coords.to_vec()).unwrap();
    CoreUMesh::new(array.into_shared())
}

fn ffi_mesh(coords: &[f64], conn: &[usize], n_elems: usize) -> Box<UMesh> {
    let mut mesh = ffi_coords(coords);
    mesh.add_regular_block(ElementType::HEX8, conn, n_elems)
        .unwrap();
    mesh
}

fn core_mesh(coords: &[f64], conn: &[usize], n_elems: usize) -> CoreUMesh {
    let coords = ndarray::Array2::from_shape_vec((coords.len() / 3, 3), coords.to_vec()).unwrap();
    let conn = ndarray::Array2::from_shape_vec((n_elems, 8), conn.to_vec()).unwrap();
    let mut mesh = CoreUMesh::new(coords.into_shared());
    mesh.add_regular_block(CoreElementType::HEX8, conn.into_shared(), None);
    mesh
}

fn constant_field(n_elems: usize, value: f64) -> FieldOwnedD {
    FieldOwnedD::new(BTreeMap::from([(
        CoreElementType::HEX8,
        ndarray::Array::from_elem(ndarray::IxDyn(&[n_elems]), value),
    )]))
}

fn build_mesh(c: &mut Criterion) {
    let mut group = c.benchmark_group("build_mesh");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for n in [8, 16, 24] {
        let (coords, conn, n_elems) = hex_mesh_data(n);
        let n_nodes = coords.len() / 3;

        group.throughput(Throughput::Bytes((coords.len() + conn.len() * 8) as u64));
        // Both sides do the same thing: take a C++-side buffer of coordinates
        // and hand an owned array to the core. The only difference is the check
        // the binding adds.
        group.bench_with_input(
            BenchmarkId::new("core_from_coords", n_nodes),
            &n_nodes,
            |b, &n_nodes| {
                b.iter(|| {
                    let array =
                        ndarray::Array2::from_shape_vec((n_nodes, 3), coords.to_vec()).unwrap();
                    CoreUMesh::new(array.into_shared())
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ffi_from_coords_full", n_nodes),
            &n_nodes,
            |b, &n_nodes| {
                set_checks(Checks::Full).unwrap();
                b.iter(|| UMesh::from_coords(&coords, n_nodes, 3).unwrap());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ffi_from_coords_fast", n_nodes),
            &n_nodes,
            |b, &n_nodes| {
                set_checks(Checks::Fast).unwrap();
                b.iter(|| UMesh::from_coords(&coords, n_nodes, 3).unwrap());
            },
        );

        group.throughput(Throughput::Bytes((conn.len() * 8) as u64));
        group.bench_with_input(
            BenchmarkId::new("core_add_block", n_elems),
            &n_elems,
            |b, &ne| {
                b.iter(|| {
                    let array = ndarray::Array2::from_shape_vec((ne, 8), conn.to_vec()).unwrap();
                    let mut mesh = core_coords(&coords);
                    mesh.add_regular_block(CoreElementType::HEX8, array.into_shared(), None);
                    std::hint::black_box(mesh);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ffi_add_block_full", n_elems),
            &n_elems,
            |b, &ne| {
                set_checks(Checks::Full).unwrap();
                b.iter(|| {
                    let mut mesh = ffi_coords(&coords);
                    mesh.add_regular_block(ElementType::HEX8, &conn, ne)
                        .unwrap();
                    std::hint::black_box(mesh);
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("ffi_add_block_fast", n_elems),
            &n_elems,
            |b, &ne| {
                set_checks(Checks::Fast).unwrap();
                b.iter(|| {
                    let mut mesh = ffi_coords(&coords);
                    mesh.add_regular_block(ElementType::HEX8, &conn, ne)
                        .unwrap();
                    std::hint::black_box(mesh);
                });
            },
        );
    }
    group.finish();
}

fn set_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("set_field");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    for n in [8, 16] {
        let (coords, conn, n_elems) = hex_mesh_data(n);
        let values = vec![1.0; n_elems];
        let blocks = [FieldBlock {
            element_type: ElementType::HEX8,
            n_components: 1,
            offset: 0,
            len: n_elems,
        }];

        group.throughput(Throughput::Bytes((n_elems * 8) as u64));
        group.bench_with_input(BenchmarkId::new("core", n_elems), &n_elems, |b, _| {
            b.iter_batched(
                || core_mesh(&coords, &conn, n_elems),
                |mut mesh| {
                    mesh.update_field("f", constant_field(n_elems, 1.0).into_shared());
                },
                criterion::BatchSize::SmallInput,
            );
        });
        group.bench_with_input(BenchmarkId::new("ffi", n_elems), &n_elems, |b, _| {
            b.iter_batched(
                || ffi_mesh(&coords, &conn, n_elems),
                |mut mesh| {
                    mesh.set_field("f", &blocks, &values).unwrap();
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

/// The transfer is the operation a user spends their time in, so the ratio here
/// is the one that decides whether the bindings are worth their cost.
fn transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("transfer");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));

    for n in [4, 8, 12] {
        let (coords, conn, n_elems) = hex_mesh_data(n);
        let shift = 0.5 / (n as f64);
        let (tcoords, tconn, tn_elems) = hex_mesh_data(n);
        let mut tgt_coords = tcoords;
        // Half a cell along x, so the target overlaps up to eight source cells.
        for node in 0..tgt_coords.len() / 3 {
            tgt_coords[3 * node] += shift;
        }

        let mut core_src = core_mesh(&coords, &conn, n_elems);
        core_src.update_field("f", constant_field(n_elems, 7.0).into_shared());
        let core_tgt = core_mesh(&tgt_coords, &tconn, tn_elems);
        let core_method = mefikit::tools::transfer::TransferMethod::ConservativeP0;

        let mut ffi_src = ffi_mesh(&coords, &conn, n_elems);
        ffi_src
            .set_field_uniform("f", ElementType::HEX8, 1, &vec![7.0; n_elems])
            .unwrap();
        let ffi_tgt = ffi_mesh(&tgt_coords, &tconn, tn_elems);
        let ffi_method = TransferMethod::conservative_p0();

        group.bench_with_input(BenchmarkId::new("core_build", n_elems), &n_elems, |b, _| {
            b.iter(|| CoreOperator::new(&core_src.view(), &core_tgt.view(), core_method));
        });
        group.bench_with_input(BenchmarkId::new("ffi_build", n_elems), &n_elems, |b, _| {
            b.iter(|| TransferOperator::prepare(&ffi_src, &ffi_tgt, &ffi_method).unwrap());
        });

        // apply: the per-field cost once the operator exists. A time-stepping
        // loop is many of these, so this is the number that has to be small.
        let core_op = CoreOperator::new(&core_src.view(), &core_tgt.view(), core_method);
        let field = core_src.field("f", None).unwrap();
        let mut core_out = core_tgt.clone();
        group.bench_with_input(BenchmarkId::new("core_apply", n_elems), &n_elems, |b, _| {
            b.iter(|| {
                core_op.apply_update(
                    &mut core_out,
                    "g",
                    &field,
                    mefikit::tools::transfer::FieldNature::Intensive,
                    0.0,
                );
            });
        });

        let ffi_op = TransferOperator::prepare(&ffi_src, &ffi_tgt, &ffi_method).unwrap();
        // No field, so that the target starts out in the same state as
        // `core_out` and both sides create the written field on their first
        // iteration.
        let mut ffi_out = ffi_mesh(&tgt_coords, &tconn, tn_elems);
        group.bench_with_input(BenchmarkId::new("ffi_apply", n_elems), &n_elems, |b, _| {
            b.iter(|| {
                ffi_op
                    .apply_update(
                        &ffi_src,
                        "f",
                        &mut ffi_out,
                        "g",
                        0.0,
                        FieldNature::Intensive,
                    )
                    .unwrap();
            });
        });
    }
    group.finish();
}

/// Reading a field back is the one place a copy would be hardest to justify:
/// the whole point of a slice return is that C++ gets a view of mefikit's own
/// storage.
fn field_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("field_read");
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let n = 16;
    let (coords, conn, n_elems) = hex_mesh_data(n);
    let core = core_mesh(&coords, &conn, n_elems);

    group.throughput(Throughput::Bytes((n_elems * 8) as u64));
    // The C++ path hands back a slice of mefikit's own storage, so this should
    // cost about as much as the core's own field lookup and no more. Anything
    // copying here would show up as a per-element cost.
    group.bench_function("core_field_lookup", |b| {
        b.iter(|| std::hint::black_box(core.field("f", None)));
    });
    let mut with_field = ffi_mesh(&coords, &conn, n_elems);
    with_field
        .set_field_uniform("f", ElementType::HEX8, 1, &vec![3.0; n_elems])
        .unwrap();
    group.bench_function("ffi_slice_present", |b| {
        b.iter(|| {
            // SAFETY: no `&mut self` method runs while the borrow is alive.
            let values = unsafe { with_field.field_values("f", ElementType::HEX8) }.unwrap();
            std::hint::black_box(values);
        });
    });
    group.finish();
}

criterion_group!(benches, build_mesh, set_field, transfer, field_read);
criterion_main!(benches);
