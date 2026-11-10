use std::collections::BTreeMap;

use approx::assert_abs_diff_eq;
use ndarray as nd;

use crate::mesh::{Dimension, ElementType, FieldOwnedD, UMesh};
use crate::tools::grid::RegularUMeshBuilder;
use crate::tools::transfer::{FieldNature, Transfer, TransferMethod, TransferOperator};

fn build_op(src: &UMesh, tgt: &UMesh) -> TransferOperator {
    TransferOperator::new(&src.view(), &tgt.view(), TransferMethod::ConservativeP0)
}

fn flat3(coords: &[[f64; 3]]) -> Vec<f64> {
    coords.iter().copied().flatten().collect()
}

fn mesh3(coords: &[[f64; 3]], cells: &[(ElementType, Vec<usize>)]) -> UMesh {
    let mut mesh =
        UMesh::new(nd::ArcArray2::from_shape_vec((coords.len(), 3), flat3(coords)).unwrap());
    for (et, conn) in cells {
        mesh.add_element(*et, conn, None);
    }
    mesh
}

fn mesh2(coords: &[[f64; 2]], cells: &[(ElementType, Vec<usize>)]) -> UMesh {
    let flat: Vec<f64> = coords.iter().copied().flatten().collect();
    let mut mesh = UMesh::new(nd::ArcArray2::from_shape_vec((coords.len(), 2), flat).unwrap());
    for (et, conn) in cells {
        mesh.add_element(*et, conn, None);
    }
    mesh
}

fn hex_cube() -> UMesh {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    mesh3(&coords, &[(ElementType::HEX8, (0..8).collect())])
}

fn quad3(coords: [[f64; 3]; 4]) -> UMesh {
    mesh3(&coords, &[(ElementType::QUAD4, vec![0, 1, 2, 3])])
}

fn seg3(a: [f64; 3], b: [f64; 3]) -> UMesh {
    mesh3(&[a, b], &[(ElementType::SEG2, vec![0, 1])])
}

fn points3(points: &[[f64; 3]]) -> UMesh {
    let cells: Vec<(ElementType, Vec<usize>)> = (0..points.len())
        .map(|i| (ElementType::VERTEX, vec![i]))
        .collect();
    mesh3(points, &cells)
}

fn scalar_field(et: ElementType, values: Vec<f64>) -> FieldOwnedD {
    FieldOwnedD::new(BTreeMap::from([(
        et,
        nd::Array1::from_vec(values).into_dyn(),
    )]))
}

fn multi_field(entries: Vec<(ElementType, Vec<f64>)>) -> FieldOwnedD {
    FieldOwnedD::new(
        entries
            .into_iter()
            .map(|(et, values)| (et, nd::Array1::from_vec(values).into_dyn()))
            .collect(),
    )
}

fn result_for(
    op: &TransferOperator,
    field: &FieldOwnedD,
    tgt_et: ElementType,
    nature: FieldNature,
    default: f64,
) -> Vec<f64> {
    let out = op.apply(&field.view(), nature, default);
    out.0[&tgt_et].iter().copied().collect()
}

/// Result of an operator with a single target element type.
fn single_result(
    op: &TransferOperator,
    field: &FieldOwnedD,
    nature: FieldNature,
    default: f64,
) -> Vec<f64> {
    let out = op.apply(&field.view(), nature, default);
    assert_eq!(out.0.len(), 1, "expected a single target element type");
    out.0.values().next().unwrap().iter().copied().collect()
}

fn apply_scalar(
    op: &TransferOperator,
    et: ElementType,
    values: Vec<f64>,
    nature: FieldNature,
    default: f64,
) -> Vec<f64> {
    let field = scalar_field(et, values);
    single_result(op, &field, nature, default)
}

// ---------------------------------------------------------------------------
// 3D -> 2D
// ---------------------------------------------------------------------------

#[test]
fn transfer_3d_to_2d_cube_face() {
    let src = hex_cube();
    let tgt = quad3([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_half_face() {
    let src = hex_cube();
    let tgt = quad3([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_inner_square() {
    let src = hex_cube();
    let tgt = quad3([
        [0.25, 0.25, 0.5],
        [0.75, 0.25, 0.5],
        [0.75, 0.75, 0.5],
        [0.25, 0.75, 0.5],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 1.75, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_partial_cover_and_default() {
    let src = hex_cube();
    let tgt = mesh3(
        &[
            [0.5, 0.25, 0.5],
            [1.5, 0.25, 0.5],
            [1.5, 0.75, 0.5],
            [0.5, 0.75, 0.5],
            [2.0, 0.0, 0.5],
            [3.0, 0.0, 0.5],
            [3.0, 1.0, 0.5],
            [2.0, 1.0, 0.5],
        ],
        &[
            (ElementType::QUAD4, vec![0, 1, 2, 3]),
            (ElementType::QUAD4, vec![4, 5, 6, 7]),
        ],
    );
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        99.0,
    );
    // The first target is half covered: its value is homogenized over the whole target.
    assert_abs_diff_eq!(intensive[0], 3.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 1.75, epsilon = 1e-12);
    assert_eq!(intensive[1], 99.0);
    assert_eq!(extensive[1], 99.0);
}

#[test]
fn transfer_3d_to_2d_crossing_four_cells() {
    let src = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 0.5, 1.0])
        .add_axis(vec![0.0, 0.5, 1.0])
        .add_axis(vec![0.0, 0.5, 1.0])
        .build();
    // Bottom layer values, in x-fastest order: 1, 2, 3, 4.
    let field = scalar_field(
        ElementType::HEX8,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let tgt = quad3([
        [0.25, 0.25, 0.25],
        [0.75, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.25, 0.75, 0.25],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = single_result(&op, &field, FieldNature::Intensive, 0.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 0.0);
    assert_abs_diff_eq!(intensive[0], 2.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 0.625, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_oblique_plane() {
    let src = hex_cube();
    // Unit-area square? Edge lengths sqrt(2) and 1, so area = sqrt(2).
    let tgt = quad3([
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0 * 2.0_f64.sqrt(), epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_shared_face_sums_both_sides() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::HEX8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (ElementType::HEX8, vec![1, 8, 9, 2, 5, 10, 11, 6]),
        ],
    );
    let tgt = quad3([
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
    ]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::HEX8, vec![10.0, 20.0]);

    let intensive = single_result(&op, &field, FieldNature::Intensive, 0.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 0.0);
    // Like MEDCoupling: both adjacent cells contribute to a surface on their shared face.
    assert_abs_diff_eq!(intensive[0], 30.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 30.0, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_polyhedral_source_and_polygon_target() {
    let src = crate::tools::polyze::polyze(&hex_cube().view());
    let tgt = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        &[(ElementType::PGON, vec![0, 1, 2, 3])],
    );
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::PHED,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::PHED,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_warped_source_face() {
    // A hexahedron whose top face is slightly warped. A target quad built from the very same
    // four top nodes must not be clipped by the best-fit plane of the source face.
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.01],
        [0.0, 1.0, 1.0],
    ];
    let src = mesh3(&coords, &[(ElementType::HEX8, (0..8).collect())]);
    let top = [coords[4], coords[5], coords[6], coords[7]];
    let tgt = quad3(top);
    let expected_area = crate::geometry::Polygon::<3>::unknown(top.iter().copied()).area();
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0 * expected_area, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_wedge_source() {
    let src = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        &[(ElementType::WEDGE6, (0..6).collect())],
    );
    let tgt = mesh3(
        &[[0.0, 0.0, 0.5], [0.3, 0.0, 0.5], [0.0, 0.3, 0.5]],
        &[(ElementType::TRI3, vec![0, 1, 2])],
    );
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::WEDGE6,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::WEDGE6,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0 * 0.045, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_tetra_cross_section() {
    let src = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        &[(ElementType::TET4, vec![0, 1, 2, 3])],
    );
    let tgt = quad3([
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.5],
        [0.0, 0.5, 0.5],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::TET4,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::TET4,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 3.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 0.875, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_mixed_source_and_target_types() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::HEX8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (ElementType::TET4, vec![8, 9, 10, 11]),
        ],
    );
    let tgt = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        &[
            (ElementType::QUAD4, vec![0, 1, 2, 3]),
            (ElementType::TRI3, vec![4, 5, 6]),
        ],
    );
    let field = multi_field(vec![
        (ElementType::HEX8, vec![10.0]),
        (ElementType::TET4, vec![20.0]),
    ]);
    let op = build_op(&src, &tgt);

    let quad = result_for(&op, &field, ElementType::QUAD4, FieldNature::Intensive, 0.0);
    let tri = result_for(&op, &field, ElementType::TRI3, FieldNature::Intensive, 0.0);
    assert_abs_diff_eq!(quad[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tri[0], 20.0, epsilon = 1e-12);

    let quad = result_for(&op, &field, ElementType::QUAD4, FieldNature::Extensive, 0.0);
    let tri = result_for(&op, &field, ElementType::TRI3, FieldNature::Extensive, 0.0);
    assert_abs_diff_eq!(quad[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(tri[0], 10.0, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_vector_field() {
    let src = hex_cube();
    let tgt = quad3([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let field = FieldOwnedD::new(BTreeMap::from([(
        ElementType::HEX8,
        nd::array![[7.0, 3.0]].into_dyn(),
    )]));
    let op = build_op(&src, &tgt);

    let intensive = op.apply(&field.view(), FieldNature::Intensive, 0.0);
    let extensive = op.apply(&field.view(), FieldNature::Extensive, 0.0);
    assert_eq!(intensive.0[&ElementType::QUAD4].shape(), &[1, 2]);
    assert_eq!(extensive.0[&ElementType::QUAD4].shape(), &[1, 2]);
    let intensive_values = intensive.0[&ElementType::QUAD4]
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(intensive_values[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive_values[1], 3.0, epsilon = 1e-12);
    let extensive_values = extensive.0[&ElementType::QUAD4]
        .iter()
        .copied()
        .collect::<Vec<_>>();
    assert_abs_diff_eq!(extensive_values[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive_values[1], 3.0, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_2d_reuse_and_apply_update() {
    let mut src = hex_cube();
    let tgt = quad3([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    src.update_field(
        "a",
        scalar_field(ElementType::HEX8, vec![7.0]).into_shared(),
    );
    src.update_field(
        "b",
        scalar_field(ElementType::HEX8, vec![3.0]).into_shared(),
    );
    let op = build_op(&src, &tgt);

    let a = src.field("a", Some(Dimension::D3)).unwrap();
    let b = src.field("b", Some(Dimension::D3)).unwrap();
    let out_a = op.apply(&a, FieldNature::Intensive, 0.0);
    let out_b = op.apply(&b, FieldNature::Intensive, 0.0);
    assert_abs_diff_eq!(out_a.0[&ElementType::QUAD4][0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(out_b.0[&ElementType::QUAD4][0], 3.0, epsilon = 1e-12);

    let mut target = quad3([
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let old = op.apply_update(
        &mut target,
        "T",
        &src.field("a", Some(Dimension::D3)).unwrap(),
        FieldNature::Intensive,
        0.0,
    );
    assert!(old.is_none());
    let transferred = target.field("T", Some(Dimension::D2)).unwrap();
    assert_abs_diff_eq!(transferred.0[&ElementType::QUAD4][0], 7.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// 3D -> 1D
// ---------------------------------------------------------------------------

#[test]
fn transfer_3d_to_1d_inside_and_partial() {
    let src = hex_cube();
    let tgt = seg3([0.25, 0.5, 0.5], [0.75, 0.5, 0.5]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);

    let partial = seg3([-0.5, 0.5, 0.5], [0.5, 0.5, 0.5]);
    let op = build_op(&src, &partial);
    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 3.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_1d_diagonal() {
    let src = hex_cube();
    let tgt = seg3([0.1, 0.1, 0.1], [0.9, 0.9, 0.9]);
    let op = build_op(&src, &tgt);
    let expected = 0.8 * 3.0_f64.sqrt();

    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0 * expected, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_1d_crossing_two_cells() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::HEX8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (ElementType::HEX8, vec![1, 8, 9, 2, 5, 10, 11, 6]),
        ],
    );
    let tgt = seg3([0.25, 0.5, 0.5], [1.5, 0.5, 0.5]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::HEX8, vec![10.0, 20.0]);

    let intensive = single_result(&op, &field, FieldNature::Intensive, 0.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 0.0);
    // Lengths 0.75 in the first cell and 0.5 in the second, over a total length of 1.25.
    assert_abs_diff_eq!(intensive[0], 14.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 17.5, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_1d_shared_face_sums_both_sides() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::HEX8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (ElementType::HEX8, vec![1, 8, 9, 2, 5, 10, 11, 6]),
        ],
    );
    let tgt = seg3([1.0, 0.0, 0.5], [1.0, 1.0, 0.5]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::HEX8, vec![10.0, 20.0]);

    let intensive = single_result(&op, &field, FieldNature::Intensive, 0.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 0.0);
    assert_abs_diff_eq!(intensive[0], 30.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 30.0, epsilon = 1e-12);
}

#[test]
fn transfer_3d_to_1d_outside_default() {
    let src = hex_cube();
    let tgt = seg3([2.0, 0.5, 0.5], [3.0, 0.5, 0.5]);
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::HEX8,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    assert_eq!(intensive[0], 99.0);
}

// ---------------------------------------------------------------------------
// 0D targets
// ---------------------------------------------------------------------------

#[test]
fn transfer_3d_to_0d_inside_outside_and_interface() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
        [2.0, 1.0, 1.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::HEX8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (ElementType::HEX8, vec![1, 8, 9, 2, 5, 10, 11, 6]),
        ],
    );
    let tgt = points3(&[
        [0.5, 0.5, 0.5],
        [1.5, 0.5, 0.5],
        // Shared face: exactly one cell owns the point, the lowest global index.
        [1.0, 0.5, 0.5],
        // Shared vertex: the lowest global index again.
        [1.0, 0.0, 0.0],
        [3.0, 0.5, 0.5],
    ]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::HEX8, vec![10.0, 20.0]);

    let intensive = single_result(&op, &field, FieldNature::Intensive, 99.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 99.0);
    assert_abs_diff_eq!(intensive[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[1], 20.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[2], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[3], 10.0, epsilon = 1e-12);
    assert_eq!(intensive[4], 99.0);
    assert_eq!(extensive, intensive);
}

#[test]
fn transfer_2d_surface_to_0d_in_3d_shared_edge_has_single_owner() {
    let coords = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ];
    let src = mesh3(
        &coords,
        &[
            (ElementType::QUAD4, vec![0, 1, 2, 3]),
            (ElementType::QUAD4, vec![1, 4, 5, 2]),
        ],
    );
    let tgt = points3(&[[0.5, 0.25, 0.0], [0.75, 0.25, 0.0]]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::QUAD4, vec![10.0, 20.0]);
    let intensive = single_result(&op, &field, FieldNature::Intensive, 99.0);
    assert_abs_diff_eq!(intensive[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[1], 20.0, epsilon = 1e-12);
}

#[test]
fn transfer_1d_to_0d_in_3d_shared_endpoint_has_single_owner() {
    let src = mesh3(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        &[
            (ElementType::SEG2, vec![0, 1]),
            (ElementType::SEG2, vec![1, 2]),
        ],
    );
    let tgt = points3(&[[1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]);
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::SEG2, vec![10.0, 20.0]);
    let intensive = single_result(&op, &field, FieldNature::Intensive, 99.0);
    assert_abs_diff_eq!(intensive[0], 10.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[1], 20.0, epsilon = 1e-12);
}

#[test]
fn transfer_2d_surface_to_0d_in_3d() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let src = mesh3(&coords, &[(ElementType::QUAD4, (0..4).collect())]);
    let tgt = points3(&[[0.5, 0.5, 0.0], [0.5, 0.5, 0.1], [2.0, 0.5, 0.0]]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_eq!(intensive[1], 99.0);
    assert_eq!(intensive[2], 99.0);
}

#[test]
fn transfer_1d_to_0d_in_3d() {
    let src = seg3([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]);
    let tgt = points3(&[
        [0.5, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.5, 0.1, 0.0],
        [3.0, 0.0, 0.0],
    ]);
    let op = build_op(&src, &tgt);

    let intensive = apply_scalar(
        &op,
        ElementType::SEG2,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(intensive[1], 7.0, epsilon = 1e-12);
    assert_eq!(intensive[2], 99.0);
    assert_eq!(intensive[3], 99.0);
}

// ---------------------------------------------------------------------------
// 2D space
// ---------------------------------------------------------------------------

#[test]
fn transfer_2d_to_1d_inside_and_partial() {
    let src = mesh2(
        &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        &[(ElementType::QUAD4, (0..4).collect())],
    );
    let tgt = mesh2(
        &[[0.25, 0.5], [0.75, 0.5]],
        &[(ElementType::SEG2, vec![0, 1])],
    );
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);

    let partial = mesh2(
        &[[-0.5, 0.5], [0.5, 0.5]],
        &[(ElementType::SEG2, vec![0, 1])],
    );
    let op = build_op(&src, &partial);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 3.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);
}

#[test]
fn transfer_2d_to_1d_polygon_source() {
    let src = mesh2(
        &[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        &[(ElementType::PGON, (0..4).collect())],
    );
    let tgt = mesh2(
        &[[0.25, 0.5], [0.75, 0.5]],
        &[(ElementType::SEG2, vec![0, 1])],
    );
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::PGON,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::PGON,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);
}

#[test]
fn transfer_2d_to_1d_crossing_two_cells() {
    let src = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 0.5, 1.0])
        .add_axis(vec![0.0, 0.5, 1.0])
        .build();
    let tgt = mesh2(
        &[[0.25, 0.25], [0.75, 0.25]],
        &[(ElementType::SEG2, vec![0, 1])],
    );
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::QUAD4, vec![1.0, 2.0, 3.0, 4.0]);
    let intensive = single_result(&op, &field, FieldNature::Intensive, 0.0);
    let extensive = single_result(&op, &field, FieldNature::Extensive, 0.0);
    assert_abs_diff_eq!(intensive[0], 1.5, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 0.75, epsilon = 1e-12);
}

#[test]
fn transfer_2d_to_0d_in_2d() {
    let src = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 0.5, 1.0])
        .add_axis(vec![0.0, 0.5, 1.0])
        .build();
    let tgt = mesh2(
        &[[0.25, 0.25], [0.5, 0.25], [1.5, 0.25]],
        &[
            (ElementType::VERTEX, vec![0]),
            (ElementType::VERTEX, vec![1]),
            (ElementType::VERTEX, vec![2]),
        ],
    );
    let op = build_op(&src, &tgt);
    let field = scalar_field(ElementType::QUAD4, vec![1.0, 2.0, 3.0, 4.0]);
    let intensive = single_result(&op, &field, FieldNature::Intensive, 99.0);
    assert_abs_diff_eq!(intensive[0], 1.0, epsilon = 1e-12);
    // Point on the shared edge x = 0.5: the lowest global index owns it.
    assert_abs_diff_eq!(intensive[1], 1.0, epsilon = 1e-12);
    assert_eq!(intensive[2], 99.0);
}

// ---------------------------------------------------------------------------
// 2D surfaces embedded in 3D space
// ---------------------------------------------------------------------------

#[test]
fn transfer_2d_to_1d_in_3d_planar() {
    let src = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        &[(ElementType::QUAD4, (0..4).collect())],
    );
    let tgt = seg3([0.25, 0.5, 0.0], [0.75, 0.5, 0.0]);
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 3.5, epsilon = 1e-12);

    let off_plane = seg3([0.25, 0.5, 0.1], [0.75, 0.5, 0.1]);
    let op = build_op(&src, &off_plane);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    assert_eq!(intensive[0], 99.0);
}

#[test]
fn transfer_2d_to_1d_in_3d_oblique() {
    let src = mesh3(
        &[
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        &[(ElementType::QUAD4, (0..4).collect())],
    );
    let tgt = seg3([0.2, 0.8, 0.2], [0.8, 0.2, 0.5]);
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        0.0,
    );
    let extensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Extensive,
        0.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_abs_diff_eq!(extensive[0], 7.0 * 0.9, epsilon = 1e-12);
}

#[test]
fn transfer_2d_to_0d_in_3d() {
    let src = mesh3(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        &[(ElementType::QUAD4, (0..4).collect())],
    );
    let tgt = points3(&[[0.5, 0.5, 0.0], [0.5, 0.5, 0.1], [1.5, 0.5, 0.0]]);
    let op = build_op(&src, &tgt);
    let intensive = apply_scalar(
        &op,
        ElementType::QUAD4,
        vec![7.0],
        FieldNature::Intensive,
        99.0,
    );
    assert_abs_diff_eq!(intensive[0], 7.0, epsilon = 1e-12);
    assert_eq!(intensive[1], 99.0);
    assert_eq!(intensive[2], 99.0);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "not supported")]
fn transfer_unsupported_source_type_panics() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
    ];
    let src = mesh3(&coords, &[(ElementType::QUAD8, (0..8).collect())]);
    let tgt = seg3([0.5, 0.5, 0.0], [0.75, 0.75, 0.0]);
    let _ = build_op(&src, &tgt);
}

#[test]
#[should_panic(expected = "not supported")]
fn transfer_unsupported_target_type_panics() {
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [0.5, 1.0, 0.0],
        [0.0, 0.5, 0.0],
    ];
    let tgt = mesh3(&coords, &[(ElementType::QUAD8, (0..8).collect())]);
    let src = hex_cube();
    let _ = build_op(&src, &tgt);
}
