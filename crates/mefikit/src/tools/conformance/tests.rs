//! Tests for the conformance diagnostics.

use super::*;
use crate::mesh::{ElementType, UMesh};
use crate::tools::test_helpers::{box_mesh, hex_block};
use ndarray as nd;
use proptest::prelude::*;
use std::f64::consts::PI;

/// Builds a `PHED` mesh from full face rings, each `Vec<usize>` a face in winding order.
fn phed_mesh(coords: Vec<[f64; 3]>, cells: &[Vec<Vec<usize>>]) -> UMesh {
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec((coords.len(), 3), flat)
            .unwrap()
            .into_shared(),
    );
    let mut conn: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::new();
    for (ci, cell) in cells.iter().enumerate() {
        if ci > 0 {
            conn.push(usize::MAX);
        }
        for face in cell {
            conn.extend_from_slice(face);
            conn.push(usize::MAX);
        }
        offsets.push(conn.len());
    }
    if let Some(last) = conn.last_mut() {
        *last = usize::MAX;
    }
    // The last separator is not part of the data: `offsets` already ends the last cell.
    conn.pop();
    if let Some(last) = offsets.last_mut() {
        *last = conn.len();
    }
    mesh.add_poly_block(
        ElementType::PHED,
        nd::ArcArray1::from_vec(conn),
        nd::ArcArray1::from_vec(offsets),
        None,
    );
    mesh
}

/// The six faces, in winding order, of the unit cube whose eight nodes are numbered
/// `n(i, j, k) = i + 2 j + 4 k`.
fn unit_cube_faces() -> Vec<Vec<usize>> {
    vec![
        vec![0, 2, 3, 1],
        vec![4, 5, 7, 6],
        vec![0, 1, 5, 4],
        vec![1, 3, 7, 5],
        vec![3, 2, 6, 7],
        vec![2, 0, 4, 6],
    ]
}

fn unit_cube_coords() -> Vec<[f64; 3]> {
    let mut c = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                c.push([i as f64, j as f64, k as f64]);
            }
        }
    }
    c
}

/// A single conformal `HEX8` block is conformal.
#[test]
fn is_conform_accepts_a_hex_block() {
    let mesh = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// Two `HEX8` blocks sharing a face through the same node ids is conformal.
#[test]
fn is_conform_accepts_two_stacked_hexes() {
    let mesh = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0, 2.0]);
    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// Two cells touching only at a vertex (a shared node id) is conformal: a shared node does not
/// need a shared face.
#[test]
fn is_conform_accepts_vertex_touching_cells() {
    // Node 0 is the shared corner at the origin. Cell A occupies [0, 1]^3, cell B [-1, 0]^3.
    // A's nodes are 0..8, B's are 8..16, but B reuses node 0 as its corner at the origin.
    let a_faces = unit_cube_faces();
    let mut coords = unit_cube_coords();
    let b_local: Vec<[f64; 3]> = (0..8)
        .map(|n| {
            let (i, j, k) = (n % 2, (n / 2) % 2, n / 4);
            [i as f64 - 1.0, j as f64 - 1.0, k as f64 - 1.0]
        })
        .collect();
    // b_local[7] = (0, 0, 0) is the shared corner: map it to node 0, the others to fresh ids.
    let mut local_to_global = [usize::MAX; 8];
    local_to_global[7] = 0;
    for (n, slot) in local_to_global.iter_mut().enumerate() {
        if *slot == usize::MAX {
            *slot = coords.len();
            coords.push(b_local[n]);
        }
    }
    let b_faces: Vec<Vec<usize>> = unit_cube_faces()
        .iter()
        .map(|f| f.iter().map(|&n| local_to_global[n]).collect())
        .collect();

    let mesh = phed_mesh(coords, &[a_faces, b_faces]);
    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// Two tetrahedra linked by one node, each with a face on the same plane. The faces share only
/// that node and do not overlap, so the mesh is conformal.
#[test]
fn is_conform_accepts_tets_sharing_one_node_with_coplanar_faces() {
    let coords: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0], // shared node
        [1.0, 0.0, 0.0], // first face
        [0.0, 1.0, 0.0],
        [0.2, 0.2, 1.0],
        [-1.0, 0.0, 0.0], // second face
        [0.0, -1.0, 0.0],
        [-0.2, -0.2, 1.0],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::TET4, &[0, 1, 2, 3], None);
    mesh.add_element(ElementType::TET4, &[0, 4, 5, 6], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// The reported false negative: one cell that spells part of its boundary with coincident
/// duplicate nodes, so it folds onto itself. This mesh is *not* conformal.
#[test]
fn is_conform_rejects_folded_cell() {
    // Unit cube, plus duplicates of nodes 5=(1,0,1) and 7=(1,1,1) at ids 8 and 9.
    let mut coords = unit_cube_coords();
    coords.push(coords[5]);
    coords.push(coords[7]);
    let mut faces = unit_cube_faces();
    // The x = 1 face uses the duplicates 8, 9 instead of 5, 7 (which the top face still uses).
    faces[3] = vec![1, 3, 9, 8];
    let mesh = phed_mesh(coords, &[faces]);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform());
    let merged = report
        .issues
        .iter()
        .filter(|i| matches!(i, ConformanceIssue::MergedNodes { .. }))
        .count();
    assert!(merged >= 2, "{report:?}");
}

/// Two stacked `HEX8` cells whose complete interface is spelled with separate node ids are
/// conformal. This is the case the canonical-node map must reduce back to actual node ids, not
/// dense cluster ids.
#[test]
fn is_conform_accepts_stacked_hexes_with_duplicate_interface_nodes() {
    let coords: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 2.0],
        [1.0, 0.0, 2.0],
        [1.0, 1.0, 2.0],
        [0.0, 1.0, 2.0],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::HEX8, &[0, 1, 2, 3, 4, 5, 6, 7], None);
    mesh.add_element(ElementType::HEX8, &[8, 9, 10, 11, 12, 13, 14, 15], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// Two cells whose interface is only *partially* shared (some ids in common, some duplicated)
/// is not conformal.
#[test]
fn is_conform_rejects_partial_interface() {
    let a_coords = unit_cube_coords();
    let a_faces = unit_cube_faces();

    // Cell B to the right (x in [1, 2]), with fresh node ids 8..16 and its own coords.
    let mut coords = a_coords.clone();
    let mut b_coords = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                b_coords.push([1.0 + i as f64, j as f64, k as f64]);
            }
        }
    }
    let base = coords.len();
    coords.extend(b_coords.iter().copied());
    let remap = |n: usize| base + n;
    let mut b_faces: Vec<Vec<usize>> = unit_cube_faces()
        .iter()
        .map(|f| f.iter().map(|&n| remap(n)).collect())
        .collect();
    // B's x = 1 face keeps A's ids 1 and 5 but uses its own ids for the other two nodes: a
    // partial interface.
    b_faces[2] = vec![1, 5, remap(7), remap(4)];

    let mesh = phed_mesh(coords, &[a_faces, b_faces]);
    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform(), "{report:?}");
}

/// Two cells whose faces are coplanar but meet only along an edge are conformal. A shared edge
/// is not a positive-area overlap, even though its nodes lie on both face boundaries.
#[test]
fn is_conform_accepts_coplanar_faces_touching_along_an_edge() {
    let mesh = hex_block(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0],
        &[(0, 0, 0), (1, 1, 0)],
    );
    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

/// Two hexahedra sharing a quad but split into tetrahedra with opposite diagonals have a
/// non-conforming interface: the triangles overlap without sharing whole faces.
#[test]
fn is_conform_rejects_mismatched_tet_interface() {
    let coords: Vec<[f64; 3]> = vec![
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 1.0],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    // Left side, diagonal 0-2 on the shared quad x = 1.
    mesh.add_element(ElementType::TET4, &[0, 1, 2, 4], None);
    mesh.add_element(ElementType::TET4, &[0, 2, 3, 5], None);
    // Right side, diagonal 1-3 on the same quad.
    mesh.add_element(ElementType::TET4, &[0, 1, 3, 6], None);
    mesh.add_element(ElementType::TET4, &[1, 2, 3, 7], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform(), "{report:?}");
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, ConformanceIssue::NonConformingFaces { .. })),
        "{report:?}"
    );
}

/// Two tetrahedra on opposite sides of a plane whose interface triangles share an edge but not
/// their third vertex have a non-conforming interface: the triangles overlap over a positive
/// area without being the same face.
#[test]
fn is_conform_rejects_two_tets_with_mismatched_interface_triangles() {
    let coords: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0], // shared interface edge
        [1.0, 0.0, 0.0], // shared interface edge
        [0.5, 1.0, 0.0], // first interface apex
        [0.5, 0.5, 0.0], // second interface apex, inside the first triangle
        [0.5, 0.5, -1.0],
        [0.5, 0.5, 1.0],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::TET4, &[0, 2, 1, 4], None);
    mesh.add_element(ElementType::TET4, &[0, 1, 3, 5], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform(), "{report:?}");
    assert_eq!(report.issues.len(), 1, "{report:?}");
    assert!(
        matches!(
            report.issues[0],
            ConformanceIssue::NonConformingFaces { .. }
        ),
        "{report:?}"
    );
}

/// The fast checks skip the expensive cell-pair intersection search.
#[test]
fn is_conform_checks_can_skip_cell_intersections() {
    let coords: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.1, 0.2, 0.1],
        [0.1, 0.1, 0.2],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::TET4, &[0, 1, 2, 3], None);
    mesh.add_element(ElementType::TET4, &[4, 5, 6, 7], None);

    let fast = is_conform_with(&mesh.view(), 1e-9, ConformanceOptions::fast()).unwrap();
    assert!(fast.is_conform(), "{fast:?}");

    let intersections = is_conform_with(
        &mesh.view(),
        1e-9,
        ConformanceOptions::new(ConformanceChecks::INTERSECTIONS),
    )
    .unwrap();
    assert!(
        intersections
            .issues
            .iter()
            .any(|i| matches!(i, ConformanceIssue::IntersectingCells { .. })),
        "{intersections:?}"
    );
}

/// The fast checks skip the geometric face-overlap search.
#[test]
fn is_conform_checks_can_skip_geometric_face_overlap() {
    let coords: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, -1.0],
        [0.5, 0.5, 1.0],
    ];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::TET4, &[0, 2, 1, 4], None);
    mesh.add_element(ElementType::TET4, &[0, 1, 3, 5], None);

    let fast = is_conform_with(&mesh.view(), 1e-9, ConformanceOptions::fast()).unwrap();
    assert!(fast.is_conform(), "{fast:?}");

    let faces = is_conform_with(
        &mesh.view(),
        1e-9,
        ConformanceOptions::new(ConformanceChecks::FACE_OVERLAP),
    )
    .unwrap();
    assert_eq!(faces.issues.len(), 1, "{faces:?}");
    assert!(
        matches!(faces.issues[0], ConformanceIssue::NonConformingFaces { .. }),
        "{faces:?}"
    );
}

/// The butterfly check is selectable on its own.
#[test]
fn is_conform_butterfly_check_is_selectable() {
    let mut coords = unit_cube_coords();
    coords[6] = [1.0, 1.0, 1.0];
    coords[7] = [0.0, 1.0, 1.0];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::HEX8, &[0, 1, 3, 2, 4, 5, 7, 6], None);

    let topology = is_conform_with(
        &mesh.view(),
        1e-9,
        ConformanceOptions::new(ConformanceChecks::TOPOLOGY),
    )
    .unwrap();
    assert!(topology.is_conform(), "{topology:?}");

    let butterfly = is_conform_with(
        &mesh.view(),
        1e-9,
        ConformanceOptions::new(ConformanceChecks::BUTTERFLY),
    )
    .unwrap();
    assert_eq!(butterfly.issues.len(), 1, "{butterfly:?}");
    assert!(
        matches!(butterfly.issues[0], ConformanceIssue::ButterflyCell { .. }),
        "{butterfly:?}"
    );
}

/// Two cells overlapping in volume are reported as intersecting.
#[test]
fn is_conform_rejects_volume_overlap() {
    let a = unit_cube_faces();
    // Cell B is the same cube translated by half a cell: it overlaps A in a positive volume.
    let coords_a = unit_cube_coords();
    let mut coords = coords_a.clone();
    let base = coords.len();
    for p in &coords_a {
        coords.push([p[0] + 0.5, p[1], p[2]]);
    }
    let b: Vec<Vec<usize>> = a
        .iter()
        .map(|f| f.iter().map(|&n| base + n).collect())
        .collect();
    let mesh = phed_mesh(coords, &[a, b]);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform());
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, ConformanceIssue::IntersectingCells { .. })),
        "{report:?}"
    );
}

/// A `HEX8` whose top face is spelled as a bow tie is reported as a butterfly cell.
#[test]
fn is_conform_rejects_butterfly_hex() {
    let mut coords = unit_cube_coords();
    // The top ring of a `HEX8` is nodes `4, 5, 7, 6`; swapping 6 and 7 makes it cross itself.
    coords[6] = [1.0, 1.0, 1.0];
    coords[7] = [0.0, 1.0, 1.0];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::HEX8, &[0, 1, 3, 2, 4, 5, 7, 6], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform());
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, ConformanceIssue::ButterflyCell { .. })),
        "{report:?}"
    );
}

/// A `PHED` whose top face is spelled as a bow tie is reported as a butterfly cell.
#[test]
fn is_conform_rejects_butterfly_phed() {
    let mut faces = unit_cube_faces();
    // The top face is `4, 5, 7, 6`; swapping the last two nodes makes it cross itself.
    faces[1] = vec![4, 5, 6, 7];
    let mesh = phed_mesh(unit_cube_coords(), &[faces]);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(!report.is_conform());
    assert!(
        report
            .issues
            .iter()
            .any(|i| matches!(i, ConformanceIssue::ButterflyCell { .. })),
        "{report:?}"
    );
}

/// A mildly warped but non-self-intersecting `HEX8` stays conformal.
#[test]
fn is_conform_accepts_warped_hex() {
    let mut coords = unit_cube_coords();
    coords[6] = [0.0, 1.0, 1.1];
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap()
        .into_shared(),
    );
    mesh.add_element(ElementType::HEX8, &[0, 1, 3, 2, 4, 5, 7, 6], None);

    let report = is_conform(&mesh.view(), 1e-9, None).unwrap();
    assert!(report.is_conform(), "{report:?}");
}

fn coords(points: &[[f64; 3]]) -> nd::Array2<f64> {
    nd::Array2::from_shape_vec(
        (points.len(), 3),
        points.iter().flatten().copied().collect(),
    )
    .unwrap()
}

#[test]
fn quad4_bow_tie_is_butterfly() {
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    assert!(cell_is_butterfly(&c, &[vec![0, 1, 2, 3]], 1e-9));
}

#[test]
fn quad4_convex_is_not_butterfly() {
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    assert!(!cell_is_butterfly(&c, &[vec![0, 1, 2, 3]], 1e-9));
}

#[test]
fn quad4_concave_is_not_butterfly() {
    // A simple, non-convex quad: one vertex is pulled inside the triangle of the other three.
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.3, 0.3, 0.0],
    ]);
    assert!(!cell_is_butterfly(&c, &[vec![0, 1, 2, 3]], 1e-9));
}

#[test]
fn collinear_edges_overlapping_in_the_middle_are_non_conforming() {
    let center = segments_overlap_center(
        &[0.0, 0.0, 0.0],
        &[1.0, 0.0, 0.0],
        &[0.0, 0.0, 0.0],
        &[0.5, 0.0, 0.0],
        1e-9,
    )
    .expect("a half-edge is a T-junction");
    assert!((center[0] - 0.25).abs() < 1e-12);
}

#[test]
fn edges_touching_at_a_shared_vertex_are_conforming() {
    assert!(
        segments_overlap_center(
            &[0.0, 0.0, 0.0],
            &[1.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            &[0.0, 1.0, 0.0],
            1e-9,
        )
        .is_none()
    );
}

#[test]
fn wedge6_valid_is_not_butterfly() {
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]);
    let rings = [
        vec![0, 2, 1],
        vec![3, 4, 5],
        vec![0, 1, 4, 3],
        vec![1, 2, 5, 4],
        vec![2, 0, 3, 5],
    ];
    assert!(!cell_is_butterfly(&c, &rings, 1e-9));
}

#[test]
fn wedge6_with_a_bow_tie_face_is_butterfly() {
    // The side face [0, 1, 4, 3] is a bow tie in the z = 0 plane; the two remaining nodes
    // give the wedge a non-zero height.
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.5, -1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 1.0],
    ]);
    let rings = [
        vec![0, 2, 1],
        vec![3, 4, 5],
        vec![0, 1, 4, 3],
        vec![1, 2, 5, 4],
        vec![2, 0, 3, 5],
    ];
    assert!(cell_is_butterfly(&c, &rings, 1e-9));
}

#[test]
fn edge_through_face_is_detected() {
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let face = FaceGeom::new(&c, &[0, 1, 2, 3]);
    assert!(edge_pierces_face(
        &[0.5, 0.5, -1.0],
        &[0.5, 0.5, 1.0],
        &face,
        1e-9
    ));
    assert!(!edge_pierces_face(
        &[2.0, 0.5, -1.0],
        &[2.0, 0.5, 1.0],
        &face,
        1e-9
    ));
}
#[test]
fn coplanar_t_junction_is_not_butterfly() {
    let c = coords(&[
        [0.0, -0.3, 0.0],
        [1.0, -0.3, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.3, 0.0, 0.0],
        [0.7, 0.0, 0.0],
        [0.7, 0.3, 0.0],
        [0.3, 0.3, 0.0],
    ]);
    let rings = [vec![0, 1, 2, 3], vec![4, 5, 6, 7]];
    assert!(!cell_is_butterfly(&c, &rings, 1e-9));
}

#[test]
fn coplanar_edge_crossing_face_is_detected() {
    let c = coords(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [-0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [0.5, 2.0, 0.0],
    ]);
    let rings = [vec![0, 1, 2, 3], vec![4, 5, 6]];
    assert!(cell_is_butterfly(&c, &rings, 1e-9));
}
fn flat_face(points: &[[f64; 3]]) -> FaceData {
    let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
    for p in points {
        for k in 0..3 {
            bounds[0][k] = bounds[0][k].min(p[k]);
            bounds[1][k] = bounds[1][k].max(p[k]);
        }
    }
    FaceData {
        id: ElementId::new(ElementType::QUAD4, 0),
        et: ElementType::QUAD4,
        ring: (0..points.len()).collect(),
        pts: points.to_vec(),
        normal: [0.0, 0.0, 1.0],
        bounds,
    }
}

#[test]
fn shared_edge_is_not_a_coplanar_overlap() {
    let a = flat_face(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let b = flat_face(&[
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]);
    assert!(faces_overlap(&a, &b, 1e-9).is_none());
}

#[test]
fn triangle_half_of_a_quad_is_a_coplanar_overlap() {
    let quad = flat_face(&[
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]);
    let triangle = flat_face(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]);
    assert!(faces_overlap(&quad, &triangle, 1e-9).is_some());
}

// ---------------------------------------------------------------------------
// Property-based checks
// ---------------------------------------------------------------------------

struct TwoCellFixture {
    coords: Vec<[f64; 3]>,
    conns: Vec<Vec<usize>>,
    second_start: usize,
    offset_axis: usize,
    et: ElementType,
}

/// Two regular cells sharing a complete interface, with the interface nodes duplicated. The
/// second cell starts at `second_start`; moving it along `offset_axis` makes the interface
/// partial.
fn two_cell_fixture(kind: usize) -> TwoCellFixture {
    match kind {
        0 => TwoCellFixture {
            coords: vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.2, 0.2, -1.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.2, 0.2, 1.0],
            ],
            conns: vec![vec![0, 2, 1, 3], vec![4, 5, 6, 7]],
            second_start: 4,
            offset_axis: 0,
            et: ElementType::TET4,
        },
        1 => TwoCellFixture {
            coords: vec![
                [0.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
                [0.0, 2.0, -1.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
            ],
            conns: vec![vec![0, 1, 2, 3, 4, 5], vec![6, 7, 8, 9, 10, 11]],
            second_start: 6,
            offset_axis: 0,
            et: ElementType::WEDGE6,
        },
        2 => TwoCellFixture {
            coords: vec![
                [0.0, 0.0, -1.0],
                [1.0, 0.0, -1.0],
                [1.0, 1.0, -1.0],
                [0.0, 1.0, -1.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            conns: vec![
                vec![0, 1, 2, 3, 4, 5, 6, 7],
                vec![8, 9, 10, 11, 12, 13, 14, 15],
            ],
            second_start: 8,
            offset_axis: 0,
            et: ElementType::HEX8,
        },
        3 => TwoCellFixture {
            coords: vec![
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [-2.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            conns: vec![vec![0, 1, 2], vec![3, 5, 4]],
            second_start: 3,
            offset_axis: 1,
            et: ElementType::TRI3,
        },
        _ => TwoCellFixture {
            coords: vec![
                [-1.0, 0.0, 0.0],
                [-1.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            conns: vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7]],
            second_start: 4,
            offset_axis: 1,
            et: ElementType::QUAD4,
        },
    }
}

/// Applies a node-id permutation to both the coordinates and the connectivity, so the mesh is
/// only relabeled and its geometry is unchanged.
fn shuffled_mesh(
    coords: &[[f64; 3]],
    conns: &[Vec<usize>],
    seed: u64,
) -> (Vec<[f64; 3]>, Vec<Vec<usize>>) {
    let n = coords.len();
    let mut perm: Vec<usize> = (0..n).collect();
    let mut state = seed | 1;
    for i in (1..n).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state as usize) % (i + 1);
        perm.swap(i, j);
    }
    let mut relabeled = vec![[0.0; 3]; n];
    for (old, &new) in perm.iter().enumerate() {
        relabeled[new] = coords[old];
    }
    let conns = conns
        .iter()
        .map(|conn| conn.iter().map(|&node| perm[node]).collect())
        .collect();
    (relabeled, conns)
}

fn fixture_mesh(
    fixture: &TwoCellFixture,
    offset: f64,
    angle: f64,
    scale: f64,
    translation: [f64; 3],
    seed: Option<u64>,
) -> UMesh {
    let mut coords = fixture.coords.clone();
    for point in &mut coords[fixture.second_start..] {
        point[fixture.offset_axis] += offset;
    }
    let (sin, cos) = angle.sin_cos();
    for point in &mut coords {
        let x = scale * (cos * point[0] - sin * point[1]) + translation[0];
        let y = scale * (sin * point[0] + cos * point[1]) + translation[1];
        let z = scale * point[2] + translation[2];
        *point = [x, y, z];
    }
    let conns = match seed {
        Some(seed) => {
            let (relabeled, conns) = shuffled_mesh(&coords, &fixture.conns, seed);
            coords = relabeled;
            conns
        }
        None => fixture.conns.clone(),
    };
    let flat: Vec<f64> = coords.iter().flatten().copied().collect();
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec((coords.len(), 3), flat)
            .unwrap()
            .into_shared(),
    );
    for conn in conns {
        mesh.add_element(fixture.et, &conn, None);
    }
    mesh
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// A complete duplicated interface must stay conformal under rigid motion, scaling and node
    /// relabeling. This covers every regular 2D/3D cell kind and the canonical-node mapping.
    #[test]
    fn regular_duplicate_interface_is_conformal_under_motion(
        kind in 0usize..5,
        angle in -PI..PI,
        scale in 0.1f64..10.0,
        tx in -10.0f64..10.0,
        ty in -10.0f64..10.0,
        tz in -10.0f64..10.0,
        seed in any::<u64>(),
    ) {
        let fixture = two_cell_fixture(kind);
        let mesh = fixture_mesh(&fixture, 0.0, angle, scale, [tx, ty, tz], Some(seed));
        let tol = 1e-9 * scale.max(1.0);
        let report = is_conform(&mesh.view(), tol, None).unwrap();
        prop_assert!(
            report.is_conform(),
            "kind={kind} angle={angle} scale={scale} report={report:?}"
        );
    }

    /// Moving one side of a duplicated interface makes it partial; every regular 2D/3D cell
    /// kind must be rejected.
    #[test]
    fn regular_partial_interface_is_rejected(
        kind in 0usize..5,
        offset in 0.1f64..0.8,
        angle in -PI..PI,
        scale in 0.5f64..2.0,
        tx in -5.0f64..5.0,
        ty in -5.0f64..5.0,
        tz in -5.0f64..5.0,
    ) {
        let fixture = two_cell_fixture(kind);
        let mesh = fixture_mesh(&fixture, offset, angle, scale, [tx, ty, tz], None);
        let tol = 1e-9 * scale.max(1.0);
        let report = is_conform(&mesh.view(), tol, None).unwrap();
        prop_assert!(
            !report.is_conform(),
            "kind={kind} offset={offset} angle={angle} scale={scale} report={report:?}"
        );
    }
}

#[test]
fn coplanar_triangles_sharing_vertex_do_not_overlap() {
    // Two TET4 faces from `conform_tet_false_positive.med`. They are coplanar, share one vertex,
    // and have nearly collinear edges leaving it. Their edge intersection is numerically near the
    // shared endpoint and must not be mistaken for a proper crossing.
    let a = flat_face(&[
        [
            -0.01187691982332946,
            -0.0034285714285714024,
            0.0033743593808771843,
        ],
        [
            -0.011678971159607306,
            -0.0037714285714285384,
            0.0033743593808772104,
        ],
        [
            -0.011777945491468371,
            -0.0035999999999999904,
            0.0030279492260965064,
        ],
    ]);
    let b = flat_face(&[
        [
            -0.011579996827746231,
            -0.003942857142857102,
            0.0037207695356578988,
        ],
        [
            -0.011678971159607306,
            -0.0037714285714285384,
            0.0033743593808772104,
        ],
        [
            -0.011481022495885156,
            -0.004114285714285671,
            0.0033743593808771843,
        ],
    ]);
    assert!(faces_overlap(&a, &b, 1e-9).is_none());
    assert_eq!(coplanar_overlap_area(&a, &b), 0.0);
}
