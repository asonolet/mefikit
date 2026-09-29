//! Tests for the Rust side of the bindings.
//!
//! The C++ suite in `../examples/tests` is the one that exercises the API the way
//! users will; these cover the pieces whose edge cases are awkward to reach from
//! C++ (field-shape validation, unknown discriminants, poly offset bookkeeping).

use mefikit_ffi::Error;
use mefikit_ffi::bridge::{
    Dimension, DistanceWeighting, ElementType, FieldBlock, FieldNature, PointLocation,
    TransferMethod, TransferMethodKind,
};
use mefikit_ffi::{UMesh, transfer_field};

fn slice<T>(v: &[T]) -> &[T] {
    v
}

/// `Result::unwrap_err` needs a `Debug` on the success type, and the cxx shared
/// structs deliberately do not have one.
fn expect_error<T>(result: Result<T, Error>) -> Error {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(err) => err,
    }
}

/// A 2 x 2 grid of QUAD4 cells over [0, 1]^2: 9 nodes, 4 elements.
fn quad_mesh() -> Box<UMesh> {
    let mut coords = Vec::new();
    for j in 0..3 {
        for i in 0..3 {
            coords.push(i as f64 / 2.0);
            coords.push(j as f64 / 2.0);
        }
    }
    let mut conn = Vec::new();
    for j in 0..2 {
        for i in 0..2 {
            let base = j * 3 + i;
            conn.extend_from_slice(&[base, base + 1, base + 4, base + 3]);
        }
    }
    let mut mesh = UMesh::from_coords(slice(&coords), 9, 2).unwrap();
    mesh.add_regular_block(ElementType::QUAD4, slice(&conn), 4)
        .unwrap();
    mesh
}

fn mixed_mesh() -> Box<UMesh> {
    let coords = [
        0.0, 0.0, 1.0, 0.0, 1.0, 1.0, // 0 1 2
        0.0, 1.0, 0.0, 2.0, 1.0, 2.0, // 3 4 5
    ];
    let mut mesh = UMesh::from_coords(slice(&coords), 6, 2).unwrap();
    mesh.add_regular_block(ElementType::QUAD4, slice(&[0usize, 1, 2, 3]), 1)
        .unwrap();
    mesh.add_regular_block(ElementType::TRI3, slice(&[2usize, 3, 4, 4, 5, 2]), 2)
        .unwrap();
    mesh
}

#[test]
fn topology_reports_the_mesh_shape() {
    let mesh = quad_mesh();
    mesh.validate_structure().unwrap();
    assert_eq!(mesh.n_nodes(), 9);
    assert_eq!(mesh.n_elements(), 4);
    assert_eq!(mesh.n_elements_of(ElementType::QUAD4), 4);
    assert_eq!(mesh.n_elements_of(ElementType::TET4), 0);
    assert_eq!(mesh.space_dimension(), 2);
    assert_eq!(mesh.topological_dimension(), Dimension::D2);
    assert!(!mesh.is_empty());
    assert_eq!(mesh.element_types(), vec![ElementType::QUAD4]);
}

#[test]
fn a_mesh_without_blocks_is_empty() {
    let coords = [0.0, 0.0, 1.0, 0.0];
    let mesh = UMesh::from_coords(slice(&coords), 2, 2).unwrap();
    assert!(mesh.is_empty());
    assert_eq!(mesh.topological_dimension(), Dimension::D0);
    assert_eq!(mesh.n_elements(), 0);
    assert_eq!(mesh.n_nodes(), 2);
}

#[test]
fn wrong_coordinate_count_is_rejected() {
    let coords = [0.0, 0.0, 1.0];
    assert!(matches!(
        UMesh::from_coords(slice(&coords), 2, 2),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn a_variable_node_count_element_needs_the_poly_entry_point() {
    let mut mesh = quad_mesh();
    let err = mesh
        .add_regular_block(ElementType::PGON, slice(&[0usize, 1, 2, 3]), 1)
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("variable node count")),
        "unexpected error: {err}"
    );

    // The poly entry point takes it, with cumulative end offsets.
    mesh.add_poly_block(ElementType::PGON, slice(&[0usize, 1, 2]), slice(&[3usize]))
        .unwrap();
    mesh.validate_structure().unwrap();
    assert_eq!(mesh.n_elements_of(ElementType::PGON), 1);
}

#[test]
fn fields_round_trip_through_the_uniform_shortcut() {
    let mut mesh = quad_mesh();
    let values = [1.0, 2.0, 3.0, 4.0];
    mesh.set_field_uniform("T", ElementType::QUAD4, 1, slice(&values))
        .unwrap();

    let info = mesh.field_info("T", ElementType::QUAD4).unwrap();
    assert_eq!(info.n_elements, 4);
    assert_eq!(info.n_components, 1);
    assert_eq!(mesh.field_names(), vec!["T".to_owned()]);

    // SAFETY: no method taking &mut self runs while the borrow is alive.
    assert_eq!(
        unsafe { mesh.field_values("T", ElementType::QUAD4) }.unwrap(),
        values
    );
}

#[test]
fn a_field_must_match_the_mesh_block_layout() {
    let mut mesh = mixed_mesh();
    let blocks = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
        FieldBlock {
            element_type: ElementType::TRI3,
            n_components: 1,
            offset: 1,
            len: 2,
        },
    ];
    mesh.set_field("T", slice(&blocks), slice(&[1.0, 2.0, 3.0]))
        .unwrap();
    // SAFETY: no method taking &mut self runs while the borrow is alive.
    let tri = unsafe { mesh.field_values("T", ElementType::TRI3) }.unwrap();
    assert_eq!(tri, [2.0, 3.0]);

    // Only one of the two blocks: the field would silently be half a field.
    let err = mesh
        .set_field("U", slice(&blocks[..1]), slice(&[1.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("QUAD4") && m.contains("TRI3")),
        "unexpected error: {err}"
    );

    // Declared shape does not match how many values were supplied: QUAD4 has one
    // element, so it wants one value, not three.
    let over = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 3,
        },
        FieldBlock {
            element_type: ElementType::TRI3,
            n_components: 1,
            offset: 1,
            len: 2,
        },
    ];
    let err = mesh
        .set_field("U", slice(&over), slice(&[1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("1 elements x 1 components")),
        "unexpected error: {err}"
    );

    // A block range that runs off the end of `values`.
    let err = mesh
        .set_field("U", slice(&blocks), slice(&[1.0, 2.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("values[1..3]")),
        "unexpected error: {err}"
    );
}

#[test]
fn the_uniform_shortcut_needs_a_single_block() {
    let mut mesh = mixed_mesh();
    let values = [1.0, 2.0, 3.0];
    assert!(matches!(
        mesh.set_field_uniform("T", ElementType::QUAD4, 1, slice(&values)),
        Err(Error::InvalidArgument(_))
    ));
    // A single-block mesh is fine.
    let mut single = quad_mesh();
    single
        .set_field_uniform("T", ElementType::QUAD4, 1, slice(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    assert_eq!(single.field_names(), vec!["T".to_owned()]);
}

#[test]
fn missing_fields_and_blocks_are_reported() {
    let mesh = quad_mesh();
    assert!(matches!(
        mesh.field_info("nope", ElementType::QUAD4),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        mesh.field_info("nope", ElementType::TRI3),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(mesh.field_names(), Vec::<String>::new());
}

#[test]
fn an_unknown_element_type_from_cxx_is_an_error_not_a_panic() {
    let mut mesh = quad_mesh();
    // A C++ caller can hand us any discriminant; cxx shared enums are open structs
    // around an integer, so this has to be caught rather than matched.
    let bogus = ElementType { repr: 250 };
    let err = mesh
        .add_regular_block(bogus, slice(&[0usize, 1, 2, 3]), 1)
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("250")),
        "unexpected error: {err}"
    );
    // An element type we do not know about simply has no block.
    assert_eq!(mesh.n_elements_of(bogus), 0);
    assert!(format!("{bogus:?}").contains("250"));
}

#[test]
fn transfer_methods_reject_what_mefikit_cannot_do() {
    for point in [PointLocation::Barycenter, PointLocation::StrictInterior] {
        let err = TransferMethod::constant_piecewise(point)
            .to_core()
            .unwrap_err();
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("not implemented")),
            "unexpected error: {err}"
        );
    }
    let err = TransferMethod {
        kind: TransferMethodKind { repr: 99 },
        point_location: PointLocation::Centroid,
        k: 0,
        exponent: 0.0,
        weighting: DistanceWeighting::Constant,
    }
    .to_core()
    .unwrap_err();
    assert!(matches!(&err, Error::InvalidArgument(m) if m.contains("99")));
}

#[test]
fn transfer_method_factories_leave_unused_parameters_zeroed() {
    let m = TransferMethod::conservative_p0();
    assert_eq!(m.kind.repr, TransferMethodKind::ConservativeP0.repr);
    assert_eq!(m.k, 0);
    assert_eq!(m.exponent, 0.0);

    let m = TransferMethod::inverse_distance(3, 1.5);
    assert_eq!(m.k, 3);
    assert_eq!(m.exponent, 1.5);

    let m = TransferMethod::moving_least_squares(2, DistanceWeighting::Gaussian);
    assert_eq!(m.k, 2);
    assert_eq!(m.weighting.repr, DistanceWeighting::Gaussian.repr);
}

#[test]
fn a_prepared_operator_can_be_applied_to_several_fields() {
    let mut src = quad_mesh();
    src.set_field_uniform("a", ElementType::QUAD4, 1, slice(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    src.set_field_uniform("b", ElementType::QUAD4, 1, slice(&[5.0, 6.0, 7.0, 8.0]))
        .unwrap();
    let mut tgt = quad_mesh();

    let op = mefikit_ffi::TransferOperator::prepare(
        &src,
        &tgt,
        &TransferMethod::constant_piecewise(PointLocation::Centroid),
    )
    .unwrap();
    op.apply_update(&src, "a", &mut tgt, "a", 0.0, FieldNature::Intensive)
        .unwrap();
    op.apply_update(&src, "b", &mut tgt, "b", 0.0, FieldNature::Intensive)
        .unwrap();

    // SAFETY: no method taking &mut self runs while the borrow is alive.
    assert_eq!(
        unsafe { tgt.field_values("a", ElementType::QUAD4) }.unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        unsafe { tgt.field_values("b", ElementType::QUAD4) }.unwrap(),
        [5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn the_one_shot_transfer_agrees_with_the_prepared_one() {
    let mut src = quad_mesh();
    src.set_field_uniform("T", ElementType::QUAD4, 1, slice(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let method = TransferMethod::conservative_p0();

    let mut one_shot = quad_mesh();
    transfer_field(
        &src,
        "T",
        &mut one_shot,
        "T",
        &method,
        0.0,
        FieldNature::Intensive,
    )
    .unwrap();

    let mut prepared = quad_mesh();
    let op = mefikit_ffi::TransferOperator::prepare(&src, &prepared, &method).unwrap();
    op.apply_update(&src, "T", &mut prepared, "T", 0.0, FieldNature::Intensive)
        .unwrap();

    // SAFETY: no method taking &mut self runs while the borrow is alive.
    assert_eq!(
        unsafe { one_shot.field_values("T", ElementType::QUAD4) }.unwrap(),
        unsafe { prepared.field_values("T", ElementType::QUAD4) }.unwrap()
    );
}

#[test]
fn transferring_a_field_the_source_does_not_have_is_an_error() {
    let src = quad_mesh();
    let mut tgt = quad_mesh();
    let err = transfer_field(
        &src,
        "missing",
        &mut tgt,
        "T",
        &TransferMethod::conservative_p0(),
        0.0,
        FieldNature::Intensive,
    )
    .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("missing")),
        "unexpected error: {err}"
    );
}

#[test]
fn a_uniform_field_survives_a_transfer_between_two_med_files() {
    // The two reference meshes in the repository's test data: 2000 polyhedra
    // each, two different discretizations of the unit cube. Reading a real .med
    // file and moving a field between them is the workflow the bindings exist
    // for, so it is worth doing against real geometry and not only against the
    // small structured meshes above.
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../tests/data");
    let mut src = UMesh::read(&format!("{data}/mesh_27.med")).unwrap();
    let mut tgt = UMesh::read(&format!("{data}/mesh_36.med")).unwrap();

    // Both files are 3D polyhedral blocks, so the field is per-PHED, not per-node.
    assert_eq!(src.n_elements(), 2000);
    assert_eq!(tgt.n_elements(), 2000);
    assert_eq!(src.n_elements_of(ElementType::PHED), 2000);
    assert_eq!(src.space_dimension(), 3);
    assert_eq!(src.topological_dimension(), Dimension::D3);
    assert_eq!(src.element_types(), vec![ElementType::PHED]);

    // Deliberately not calling validate_structure(): it fails on these two files,
    // and not because of the bindings. The core's MED reader marks the face
    // boundaries inside a polyhedron with usize::MAX sentinels
    // (crates/mefikit/src/io/med_io.rs, "mefikit convention") while the core's own
    // validate_structure() rejects any node index >= n_nodes
    // (crates/mefikit/src/mesh/umesh.rs). The transfer below is what shows the
    // mesh is usable regardless.

    // A field of 1.0 on every cell. Every method here is an average or a
    // least-squares fit, so a constant field has to come back constant: the value
    // cannot depend on how the two meshes happen to be cut up.
    let uniform = vec![1.0; 2000];
    let negative = vec![-1.0; 2000];
    src.set_field_uniform("u", ElementType::PHED, 1, slice(&uniform))
        .unwrap();
    src.set_field_uniform("v", ElementType::PHED, 1, slice(&negative))
        .unwrap();

    let methods = [
        TransferMethod::conservative_p0(),
        TransferMethod::constant_piecewise(PointLocation::Centroid),
        TransferMethod::inverse_distance(8, 2.0),
        TransferMethod::moving_least_squares(8, DistanceWeighting::Gaussian),
    ];
    for method in &methods {
        // One prepare, two fields: the shape of a time-step loop, and the reason
        // prepare() is separate from apply_update().
        let op = mefikit_ffi::TransferOperator::prepare(&src, &tgt, method).unwrap();
        op.apply_update(&src, "u", &mut tgt, "u", 0.0, FieldNature::Intensive)
            .unwrap();
        op.apply_update(&src, "v", &mut tgt, "v", 0.0, FieldNature::Intensive)
            .unwrap();

        // The meshes discretize the same unit cube, so no target cell falls back
        // to the default value and every one of them must come back at 1.0.
        // SAFETY: no method taking &mut self runs while the borrow is alive.
        let got = unsafe { tgt.field_values("u", ElementType::PHED) }.unwrap();
        let got_negative = unsafe { tgt.field_values("v", ElementType::PHED) }.unwrap();
        assert_eq!(got.len(), 2000);
        assert_eq!(got_negative.len(), 2000);
        let worst = got
            .iter()
            .chain(got_negative.iter())
            .map(|&x| (x.abs() - 1.0).abs())
            .fold(0.0f64, f64::max);
        assert!(worst < 1e-5, "a uniform field came back as {worst} off 1.0");
        assert!(
            !got.contains(&0.0),
            "some target cell was left uncovered by the source"
        );
    }

    // The other direction, on the same two files: 36 -> 27.
    let back =
        mefikit_ffi::TransferOperator::prepare(&tgt, &src, &TransferMethod::conservative_p0())
            .unwrap();
    back.apply_update(&tgt, "u", &mut src, "u_back", 0.0, FieldNature::Intensive)
        .unwrap();
    // SAFETY: no method taking &mut self runs while the borrow is alive.
    let reversed = unsafe { src.field_values("u_back", ElementType::PHED) }.unwrap();
    assert_eq!(reversed.len(), 2000);
    let worst_back = reversed
        .iter()
        .map(|&x| (x - 1.0).abs())
        .fold(0.0f64, f64::max);
    assert!(
        worst_back < 1e-5,
        "the reverse transfer came back as {worst_back} off 1.0"
    );
}

#[test]
fn io_round_trips_through_a_file() {
    let dir = std::env::temp_dir().join("mefikit_ffi_rust_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("mesh.json");
    let path = path.to_str().unwrap();

    let mut mesh = mixed_mesh();
    let blocks = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
        FieldBlock {
            element_type: ElementType::TRI3,
            n_components: 1,
            offset: 1,
            len: 2,
        },
    ];
    mesh.set_field("T", slice(&blocks), slice(&[1.5, 2.5, 3.5]))
        .unwrap();
    mesh.write(path).unwrap();

    let reloaded = UMesh::read(path).unwrap();
    assert_eq!(reloaded.n_nodes(), 6);
    assert_eq!(reloaded.n_elements(), 3);
    // SAFETY: no method taking &mut self runs while the borrow is alive.
    assert_eq!(
        unsafe { reloaded.field_values("T", ElementType::TRI3) }.unwrap(),
        [2.5, 3.5]
    );

    std::fs::remove_file(path).unwrap();
    assert!(matches!(UMesh::read(path), Err(Error::Io(_))));
    assert!(matches!(
        mesh.write("/no/such/dir/mesh.json"),
        Err(Error::Io(_))
    ));
}

/// A mesh whose blocks are checked against the one just added. The core's
/// `add_*_block` silently keeps the first block when handed a second of the
/// same type, which looks like a successful call to a caller that then reads
/// back the values it just wrote and gets the old ones.
#[test]
fn an_element_type_may_only_have_one_block() {
    let mut mesh = quad_mesh();
    mesh.set_field_uniform("T", ElementType::QUAD4, 1, slice(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();

    let err = mesh
        .add_regular_block(ElementType::QUAD4, slice(&[8, 7, 6, 5]), 1)
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("already has a QUAD4 block")),
        "unexpected error: {err}"
    );

    // A block of a fixed-size type is refused by the poly entry point too, and
    // that is worth knowing before the duplicate check, since the type is
    // wrong for this call whichever way round it is reported.
    let err = mesh
        .add_poly_block(ElementType::QUAD4, slice(&[0usize]), slice(&[1]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("fixed node count")),
        "unexpected error: {err}"
    );

    let mut pgon = quad_mesh();
    pgon.add_poly_block(
        ElementType::PGON,
        slice(&[0usize, 1, 2, 3]),
        slice(&[4usize]),
    )
    .unwrap();
    let err = pgon
        .add_poly_block(
            ElementType::PGON,
            slice(&[3usize, 2, 1, 0]),
            slice(&[4usize]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("already has a PGON block")),
        "unexpected error: {err}"
    );

    // The rejected call left the mesh exactly as it was.
    assert_eq!(mesh.n_elements(), 4);
    // SAFETY: no method taking &mut self runs while the borrow is alive.
    assert_eq!(
        unsafe { mesh.field_values("T", ElementType::QUAD4) }.unwrap(),
        [1.0, 2.0, 3.0, 4.0]
    );
}

/// The poly offset table is what tells the core how many elements a block has
/// and where each one starts. Offsets that do not run from 0 to the number of
/// given nodes make it read a different number of elements than the caller
/// intended, which then shows up much later as a transfer of the wrong size or
/// as a panic.
#[test]
fn poly_offsets_must_describe_the_nodes_they_are_given() {
    let mut mesh = mixed_mesh();
    // Offsets that go backwards.
    let err = mesh
        .add_poly_block(
            ElementType::PGON,
            slice(&[0usize, 1, 2, 3]),
            slice(&[5usize, 3]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("only 4 nodes were given")),
        "unexpected error: {err}"
    );
    // Two elements that start at the same offset, so the first is empty.
    let err = mesh
        .add_poly_block(
            ElementType::PGON,
            slice(&[0usize, 1, 2, 3]),
            slice(&[0usize, 3]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("empty element at index 0")),
        "unexpected error: {err}"
    );
    // The last offset does not reach the end of the connectivity.
    let err = mesh
        .add_poly_block(
            ElementType::PGON,
            slice(&[0usize, 1, 2, 3]),
            slice(&[2usize]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("ends at node 2")),
        "unexpected error: {err}"
    );
    // The poly entry point only takes variable-size elements.
    let err = mesh
        .add_poly_block(
            ElementType::QUAD4,
            slice(&[0usize, 1, 2, 3]),
            slice(&[4usize]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("fixed node count")),
        "unexpected error: {err}"
    );
    assert_eq!(mesh.n_elements(), 3);
}

/// The core indexes nodes with these numbers, so a bad one is a panic rather
/// than a rejected mesh.
#[test]
fn connectivity_may_only_reference_existing_nodes() {
    let mut mesh = UMesh::from_coords(slice(&[0.0, 0.0, 1.0, 0.0, 1.0, 1.0]), 3, 2).unwrap();
    let err = mesh
        .add_regular_block(ElementType::TRI3, slice(&[0usize, 1, 99]), 1)
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("node 99 but the mesh has 3 nodes")),
        "unexpected error: {err}"
    );
    let err = mesh
        .add_poly_block(
            ElementType::PGON,
            slice(&[0usize, 1, 2, 42]),
            slice(&[4usize]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("node 42 but the mesh has 3 nodes")),
        "unexpected error: {err}"
    );
    assert!(mesh.is_empty());
}

/// mefikit works in 1D, 2D and 3D space, and a transfer of non-finite
/// coordinates produces meaningless weights rather than a diagnosable failure.
#[test]
fn coordinates_must_be_finite_and_of_a_usable_dimension() {
    for space_dim in [0usize, 4] {
        let err = expect_error(UMesh::from_coords(slice(&[0.0; 3]), 1, space_dim));
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("space_dim must be")),
            "unexpected error: {err}"
        );
    }
    let err = expect_error(UMesh::from_coords(slice(&[0.0, 0.0, f64::NAN, 0.0]), 2, 2));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("is NaN, which is not finite")),
        "unexpected error: {err}"
    );
    let err = expect_error(UMesh::from_coords(
        slice(&[0.0, 0.0, f64::INFINITY, 0.0]),
        2,
        2,
    ));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("is inf, which is not finite")),
        "unexpected error: {err}"
    );
}

/// A transfer reads a field as one array by gluing the per-element-type parts
/// together, which only makes sense if they agree on the trailing dimensions.
/// Two blocks of one field with different component counts used to get all the
/// way into the core before it asserted on the shapes.
#[test]
fn a_field_has_one_shape_across_all_of_the_mesh_element_types() {
    let mut mesh = mixed_mesh();
    let blocks = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
        FieldBlock {
            element_type: ElementType::TRI3,
            n_components: 3,
            offset: 1,
            len: 6,
        },
    ];
    let err = mesh
        .set_field(
            "T",
            slice(&blocks),
            slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
        )
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("1 components on the first block but 3 on TRI3")),
        "unexpected error: {err}"
    );
    assert!(mesh.field_names().is_empty());
}

#[test]
fn field_names_may_not_be_empty() {
    let mut mesh = quad_mesh();
    let err = mesh
        .set_field_uniform("", ElementType::QUAD4, 1, slice(&[1.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("may not be empty")),
        "unexpected error: {err}"
    );
}

#[test]
fn a_field_block_may_not_be_listed_twice() {
    let mut mesh = quad_mesh();
    let blocks = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
    ];
    let err = mesh
        .set_field("T", slice(&blocks), slice(&[1.0, 2.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("lists QUAD4 more than once")),
        "unexpected error: {err}"
    );
}

/// The element count times the component count has to fit in a `usize` before
/// it is used to slice the caller's array. In release builds the core's
/// arithmetic used to wrap around and index somewhere else in memory.
#[test]
fn an_impossible_field_size_is_reported_rather_than_wrapping() {
    let mut mesh = quad_mesh();
    let blocks = [FieldBlock {
        element_type: ElementType::QUAD4,
        n_components: usize::MAX / 2 + 1,
        offset: 0,
        len: 0,
    }];
    let err = mesh.set_field("T", slice(&blocks), slice(&[])).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("which overflows")),
        "unexpected error: {err}"
    );
}

/// A `um` built over a hexahedral block, plus a triangle lying on one of its
/// faces: enough to try a transfer between meshes whose space dimensions or
/// cell dimensions disagree.
fn hex_mesh() -> Box<UMesh> {
    let coords = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
    ];
    let mut mesh = UMesh::from_coords(slice(&coords), 8, 3).unwrap();
    mesh.add_regular_block(ElementType::HEX8, slice(&[0usize, 1, 2, 3, 4, 5, 6, 7]), 1)
        .unwrap();
    mesh
}

fn quad_surface() -> Box<UMesh> {
    let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
    let mut mesh = UMesh::from_coords(slice(&coords), 3, 3).unwrap();
    mesh.add_regular_block(ElementType::TRI3, slice(&[0usize, 1, 2]), 1)
        .unwrap();
    mesh
}

/// mefikit's transfer methods are documented with preconditions that the core
/// enforces by asserting. Reaching the core with them violated is a panic, so
/// the bindings check first and return the reason instead.
#[test]
fn a_transfer_that_mefikit_cannot_do_reports_the_reason() {
    let coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let bare = UMesh::from_coords(slice(&coords), 4, 2).unwrap();
    let a = quad_mesh();
    let b = quad_mesh();

    let p0 = TransferMethod::conservative_p0();
    let cases: [(&str, &UMesh, &UMesh, &TransferMethod, &str); 5] = [
        ("no source cells", &bare, &a, &p0, "no elements"),
        ("no target cells", &a, &bare, &p0, "no elements"),
        (
            "k of zero",
            &a,
            &b,
            &TransferMethod::inverse_distance(0, 2.0),
            "k of at least 1",
        ),
        (
            "zero exponent",
            &a,
            &b,
            &TransferMethod::inverse_distance(3, 0.0),
            "positive exponent",
        ),
        (
            "mls k of zero",
            &a,
            &b,
            &TransferMethod::moving_least_squares(0, DistanceWeighting::Gaussian),
            "k of at least 1",
        ),
    ];
    for (label, src, tgt, method, expected) in cases {
        let err = expect_error(mefikit_ffi::TransferOperator::prepare(src, tgt, method));
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains(expected)),
            "{label}: unexpected error: {err}"
        );
    }

    // A method that cannot work on cells that do not fill their space dimension
    // is rejected, but the methods that are defined on a lower-dimensional
    // source keep working.
    let hex = hex_mesh();
    let surface = quad_surface();
    for method in [
        TransferMethod::conservative_p0(),
        TransferMethod::constant_piecewise(PointLocation::Centroid),
    ] {
        // A surface target: no cell to integrate over.
        let err = expect_error(mefikit_ffi::TransferOperator::prepare(
            &hex, &surface, &method,
        ));
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("full-dimensional target cells")),
            "unexpected error: {err}"
        );
        // A surface source: no cell to take the value from.
        let err = expect_error(mefikit_ffi::TransferOperator::prepare(
            &surface, &hex, &method,
        ));
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("full-dimensional source cells")),
            "unexpected error: {err}"
        );
    }
    mefikit_ffi::TransferOperator::prepare(
        &hex,
        &surface,
        &TransferMethod::inverse_distance(1, 2.0),
    )
    .unwrap();

    // Both meshes have to live in the same space for an operator to be built.
    let hex2 = hex_mesh();
    let err = expect_error(mefikit_ffi::TransferOperator::prepare(
        &a,
        &hex2,
        &TransferMethod::conservative_p0(),
    ));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("same space dimension")),
        "unexpected error: {err}"
    );
}

/// A 1 x 1 grid of QUAD4 cells: same kinds as `quad_mesh`, half the size.
fn coarse_quad_mesh() -> Box<UMesh> {
    let coords = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let mut mesh = UMesh::from_coords(slice(&coords), 4, 2).unwrap();
    mesh.add_regular_block(ElementType::QUAD4, slice(&[0usize, 1, 2, 3]), 1)
        .unwrap();
    mesh
}

/// An operator holds matrices sized by the meshes it was built from. Applying
/// it to a different mesh used to be an assertion failure inside the core, and
/// applying it to a mesh that had been modified since had no check at all.
#[test]
fn an_operator_refuses_meshes_it_was_not_prepared_for() {
    let mut src = quad_mesh();
    src.set_field_uniform("T", ElementType::QUAD4, 1, slice(&[1.0, 2.0, 3.0, 4.0]))
        .unwrap();
    let mut tgt = quad_mesh();
    let op = mefikit_ffi::TransferOperator::prepare(&src, &tgt, &TransferMethod::conservative_p0())
        .unwrap();

    // A different source of the same kind: the operator's matrices would no
    // longer line up with it.
    let mut other = coarse_quad_mesh();
    other
        .set_field_uniform("T", ElementType::QUAD4, 1, slice(&[1.0]))
        .unwrap();
    let err =
        expect_error(op.apply_update(&other, "T", &mut tgt, "T", 0.0, FieldNature::Intensive));
    assert!(
        matches!(&err, Error::InvalidArgument(m)
            if m.contains("source mesh given here has 1; the geometry must not change")),
        "unexpected error: {err}"
    );

    // A source in a different space is a different kind of mistake.
    let mut other_space = hex_mesh();
    other_space
        .set_field_uniform("T", ElementType::HEX8, 1, slice(&[1.0]))
        .unwrap();
    let err = expect_error(op.apply_update(
        &other_space,
        "T",
        &mut tgt,
        "T",
        0.0,
        FieldNature::Intensive,
    ));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("at dimension D3")),
        "unexpected error: {err}"
    );

    // A target that gained a block since the operator was built.
    tgt.add_regular_block(ElementType::TRI3, slice(&[0usize, 1, 4]), 1)
        .unwrap();
    let err = expect_error(op.apply_update(&src, "T", &mut tgt, "T", 0.0, FieldNature::Intensive));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("prepared for a target mesh")),
        "unexpected error: {err}"
    );
}

/// A field whose blocks disagree on the component count is something the
/// bindings cannot build, but a hand-written or third-party file can contain it,
/// and the core panics when it concatenates such a field. Kept as a literal
/// rather than produced through the API, because producing it is exactly what is
/// being tested against.
const RAGGED_FIELD_MESH: &str = r#"{"coords":{"v":1,"dim":[6,2],"data":[0.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,2.0,1.0,2.0]},"element_blocks":{"TRI3":{"cell_type":"TRI3","connectivity":{"Regular":{"v":1,"dim":[2,3],"data":[2,3,4,4,5,2]}},"fields":{"T":{"v":1,"dim":[2,3],"data":[2.5,3.5,4.5,5.5,6.5,7.5]}},"families":{"v":1,"dim":[2],"data":[0,0]},"groups":{}},"QUAD4":{"cell_type":"QUAD4","connectivity":{"Regular":{"v":1,"dim":[1,4],"data":[0,1,2,3]}},"fields":{"T":{"v":1,"dim":[1,1],"data":[1.5]}},"families":{"v":1,"dim":[1],"data":[0]},"groups":{}}}}"#;

#[test]
fn a_field_from_a_file_still_needs_one_shape() {
    let dir = std::env::temp_dir().join("mefikit_ffi_ragged_test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("ragged.json");
    std::fs::write(&path, RAGGED_FIELD_MESH).unwrap();
    let path = path.to_str().unwrap();

    let src = UMesh::read(path).unwrap();
    let mut tgt = quad_mesh();
    let method = TransferMethod::conservative_p0();

    let err = expect_error(transfer_field(
        &src,
        "T",
        &mut tgt,
        "T",
        &method,
        0.0,
        FieldNature::Intensive,
    ));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("3 components on TRI3 and 1 on QUAD4")),
        "unexpected error: {err}"
    );

    let op = mefikit_ffi::TransferOperator::prepare(&src, &tgt, &method).unwrap();
    let err = expect_error(op.apply_update(&src, "T", &mut tgt, "T", 0.0, FieldNature::Intensive));
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("3 components on TRI3 and 1 on QUAD4")),
        "unexpected error: {err}"
    );

    std::fs::remove_file(path).unwrap();
}
