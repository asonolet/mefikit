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

    // Declared shape does not match how many values were supplied.
    let wide = [
        FieldBlock {
            element_type: ElementType::QUAD4,
            n_components: 1,
            offset: 0,
            len: 1,
        },
        FieldBlock {
            element_type: ElementType::TRI3,
            n_components: 2,
            offset: 1,
            len: 2,
        },
    ];
    let err = mesh
        .set_field("U", slice(&wide), slice(&[1.0, 2.0, 3.0]))
        .unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("2 elements x 2 components")),
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
