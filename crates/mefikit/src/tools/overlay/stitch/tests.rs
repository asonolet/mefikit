use super::*;
use crate::element_traits::ElementGeo;
use crate::geometry::{cross2, signed_area2};
use crate::mesh::ElementType;
use crate::tools::RegularUMeshBuilder;
use std::collections::BTreeSet;

fn box_mesh(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis(xs.to_vec())
        .add_axis(ys.to_vec())
        .add_axis(zs.to_vec())
        .build()
}

/// Structured hex mesh over the node grid `xs × ys × zs` keeping only the cells listed in
/// `cells` as `(i, j, k)` grid indices. Nodes are shared between neighbouring cells, and
/// the `HEX8` winding matches [`RegularUMeshBuilder`].
fn hex_block(xs: &[f64], ys: &[f64], zs: &[f64], cells: &[(usize, usize, usize)]) -> UMesh {
    let (nx, ny) = (xs.len(), ys.len());
    let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let flat: Vec<f64> = (0..zs.len())
        .flat_map(|k| {
            ys.iter()
                .flat_map(move |y| xs.iter().map(move |x| [*x, *y, zs[k]]))
        })
        .flatten()
        .collect();
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec((nx * ny * zs.len(), 3), flat)
            .unwrap()
            .into_shared(),
    );
    for &(i, j, k) in cells {
        mesh.add_element(
            ElementType::HEX8,
            &[
                node(i, j, k),
                node(i + 1, j, k),
                node(i + 1, j + 1, k),
                node(i, j + 1, k),
                node(i, j, k + 1),
                node(i + 1, j, k + 1),
                node(i + 1, j + 1, k + 1),
                node(i, j + 1, k + 1),
            ],
            None,
        );
    }
    mesh
}

/// Rotates the mesh coordinates around `axis = value` by `angle` radians.
fn rotate_about_axis(mesh: &UMesh, axis: usize, value: f64, angle: f64) -> UMesh {
    let (c, s) = (angle.cos(), angle.sin());
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut coords = mesh.coords().to_owned();
    for mut row in coords.outer_iter_mut() {
        let (du, dv) = (row[u] - value, row[v] - value);
        row[u] = value + c * du - s * dv;
        row[v] = value + s * du + c * dv;
    }
    let mut rotated = UMesh::new(coords.into_shared());
    for cell in mesh.elements() {
        rotated.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
    }
    rotated
}

fn total_volume(mesh: &UMesh) -> f64 {
    mesh.elements_of_dim(Dimension::D3)
        .map(|c| c.measure3())
        .sum()
}

/// Every face of the mesh in its stored winding, together with the cells using it, keyed by
/// the sorted node ids so that a face shared by two cells appears only once.
fn face_table(mesh: &UMesh) -> BTreeMap<Vec<usize>, (Vec<usize>, Vec<usize>)> {
    let mut table: BTreeMap<Vec<usize>, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (_, conn) = cell.to_poly();
        for ring in split_phed(&conn) {
            let mut key = ring.to_vec();
            key.sort_unstable();
            let entry = table.entry(key).or_default();
            entry.0 = ring.to_vec();
            entry.1.push(cell.id().index());
        }
    }
    table
}

/// Points of a face ring, in its stored winding.
fn face_points(mesh: &UMesh, ring: &[usize]) -> Vec<[f64; 3]> {
    let coords = mesh.coords();
    ring.iter()
        .map(|&g| [coords[(g, 0)], coords[(g, 1)], coords[(g, 2)]])
        .collect()
}

/// Summary of a validated `stitch` result.
#[derive(Debug)]
struct Report {
    cells: usize,
    interface_faces: usize,
    interface_area: f64,
    /// Interface area covered by each pair of input meshes, keyed by the pair of families.
    by_pair: BTreeMap<(usize, usize), f64>,
    volume: f64,
}

/// Validates a `stitch` result and returns a summary of its interface.
///
/// Asserts that, within `tol`:
/// - every cell is a `PHED`;
/// - the total volume is conserved, since imprinting only subdivides matter;
/// - every face is used by at most two cells, so no two cells overlap;
/// - every face is planar and has at least three nodes;
/// - faces shared by two cells are consistently wound, whatever the convention;
/// - the interface covers exactly the area that the input meshes really share, pair by pair.
///   A face left unshared inside a contact region (a missing imprint) or a face shared that
///   should not be (a spurious one) moves that area, so this is what pins down conformity;
/// - no two output nodes are within `tol` of each other, so welding did not leave
///   duplicates behind.
///
/// Families are assumed to be relabeled in input order, which is what every fixture here
/// feeds in (all input cells have family 0).
fn check_result(out: &UMesh, inputs: &[&UMesh], tol: f64) -> Report {
    const PLANAR_TOL: f64 = 1e-12;
    for cell in out.elements() {
        assert_eq!(
            cell.element_type(),
            ElementType::PHED,
            "every output cell must be a PHED"
        );
    }
    let volume = total_volume(out);
    let expected: f64 = inputs.iter().map(|m| total_volume(m)).sum();
    assert!(
        (volume - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "volume is not conserved: got {volume}, expected {expected}"
    );

    let coords = out.coords();
    for i in 0..coords.nrows() {
        for j in i + 1..coords.nrows() {
            let d: f64 = (0..3)
                .map(|k| (coords[(i, k)] - coords[(j, k)]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(d > tol, "nodes {i} and {j} are {d} apart, within {tol}");
        }
    }

    let mut interface_faces = 0;
    let mut interface_area = 0.0;
    let mut by_pair: BTreeMap<(usize, usize), f64> = BTreeMap::new();
    let mut winding = 0usize;
    for (key, (ring, cells)) in face_table(out) {
        assert!(
            cells.len() <= 2,
            "face {key:?} is used by {} cells, so two cells overlap",
            cells.len()
        );
        assert!(ring.len() >= 3, "face {key:?} is degenerate");
        let pts = face_points(out, &ring);
        let frame = PlaneFrame::from_points(&pts);
        assert!(
            frame.max_deviation(&pts) <= PLANAR_TOL,
            "face {key:?} is not planar (deviation {})",
            frame.max_deviation(&pts)
        );
        if cells.len() != 2 {
            continue;
        }
        let [c0, c1] =
            [cells[0], cells[1]].map(|i| out.element(ElementId::new(ElementType::PHED, i)));
        let normal = Polygon::unknown(pts.iter().copied()).normal();
        let d: f64 = (0..3)
            .map(|k| normal[k] * (c1.centroid3()[k] - c0.centroid3()[k]))
            .sum();
        let side = usize::from(d > 0.0);
        assert!(
            winding == 0 || winding == side,
            "face {key:?} is wound inconsistently with the other shared faces"
        );
        winding = side;
        if *c0.family == *c1.family {
            continue;
        }
        let area = Polygon::unknown(pts.iter().copied()).area();
        interface_faces += 1;
        interface_area += area;
        *by_pair
            .entry(((*c0.family).min(*c1.family), (*c0.family).max(*c1.family)))
            .or_default() += area;
    }
    for i in 0..inputs.len() {
        for j in i + 1..inputs.len() {
            let got = by_pair.get(&(i, j)).copied().unwrap_or(0.0);
            let want = contact_area(inputs[i], inputs[j], tol);
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "the interface between meshes {i} and {j} covers {got}, \
                 but the two meshes really share {want}"
            );
        }
    }
    Report {
        cells: out.num_elements(),
        interface_faces,
        interface_area,
        by_pair,
        volume,
    }
}

/// Area of the region covered by the boundary faces that `a` and `b` share, computed without
/// `stitch`: every pair of coplanar boundary faces is clipped against each other. Only valid
/// for meshes whose faces are all convex, which is the case for every fixture below.
fn contact_area(a: &UMesh, b: &UMesh, tol: f64) -> f64 {
    fn boundary_faces(mesh: &UMesh) -> Vec<Vec<[f64; 3]>> {
        face_table(mesh)
            .into_values()
            .filter(|(_, cells)| cells.len() == 1)
            .map(|(ring, _)| face_points(mesh, &ring))
            .collect()
    }
    /// `true` if the two faces lie on the same plane, whatever side of it they are on.
    fn coplanar(pa: &[[f64; 3]], pb: &[[f64; 3]], tol: f64) -> bool {
        let na = Polygon::unknown(pa.iter().copied()).normal();
        let nb = Polygon::unknown(pb.iter().copied()).normal();
        if (0..3).map(|k| na[k] * nb[k]).sum::<f64>().abs() < 1.0 - 1e-9 {
            return false;
        }
        pb.iter().all(|q| {
            let d: f64 = (0..3).map(|k| na[k] * (q[k] - pa[0][k])).sum();
            d.abs() <= tol
        })
    }
    /// Area of the intersection of two convex 2D polygons given in any winding.
    fn overlap_2d(p: &[[f64; 2]], q: &[[f64; 2]]) -> f64 {
        let mut clip = q.to_vec();
        if signed_area2(&clip) < 0.0 {
            clip.reverse();
        }
        let mut out = p.to_vec();
        for i in 0..clip.len() {
            if out.len() < 3 {
                return 0.0;
            }
            let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
            let (input, mut next) = (out.clone(), Vec::new());
            for k in 0..input.len() {
                let (p, q) = (input[k], input[(k + 1) % input.len()]);
                let (p_in, q_in) = (cross2(a, b, p) >= 0.0, cross2(a, b, q) >= 0.0);
                let cut = || {
                    let (dax, day) = (b[0] - a[0], b[1] - a[1]);
                    let (dpx, dpy) = (q[0] - p[0], q[1] - p[1]);
                    let t = ((p[0] - a[0]) * dpy - (p[1] - a[1]) * dpx) / (dax * dpy - day * dpx);
                    [a[0] + t * dax, a[1] + t * day]
                };
                match (p_in, q_in) {
                    (true, true) => next.push(q),
                    (true, false) => next.push(cut()),
                    (false, true) => {
                        next.push(cut());
                        next.push(q);
                    }
                    (false, false) => {}
                }
            }
            out = next;
        }
        signed_area2(&out).abs()
    }
    let mut total = 0.0;
    for pa in boundary_faces(a) {
        for pb in boundary_faces(b) {
            if !coplanar(&pa, &pb, tol) {
                continue;
            }
            let frame = PlaneFrame::from_points(&pa);
            let p: Vec<[f64; 2]> = pa.iter().map(|q| frame.project(q)).collect();
            let q: Vec<[f64; 2]> = pb.iter().map(|r| frame.project(r)).collect();
            total += overlap_2d(&p, &q);
        }
    }
    total
}

/// Faces lying on the plane `coords[axis] = value`, mapped to how many cells use them.
fn faces_on_plane(mesh: &UMesh, axis: usize, value: f64, tol: f64) -> BTreeMap<Vec<usize>, usize> {
    let mut counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (_, conn) = cell.to_poly();
        for face in split_phed(&conn) {
            if face
                .iter()
                .all(|&g| (mesh.coords()[(g, axis)] - value).abs() <= tol)
            {
                let mut key = face.to_vec();
                key.sort_unstable();
                *counts.entry(key).or_default() += 1;
            }
        }
    }
    counts
}

/// Number of interface faces on the plane `axis = value`, asserting that every face there is
/// shared by exactly two cells (i.e. the interface is fully internal and conforming).
fn count_interface_faces(mesh: &UMesh, axis: usize, value: f64, tol: f64) -> usize {
    let counts = faces_on_plane(mesh, axis, value, tol);
    assert!(
        counts.values().all(|&c| c == 2),
        "every interface face must be shared by exactly two cells, got {counts:?}"
    );
    counts.len()
}

#[test]
fn test_stitch_requires_two_meshes() {
    let m = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let views = vec![m.view()];
    let e = stitch(&views, 1e-9).unwrap_err();
    assert_eq!(e, StitchError::NotEnoughMeshes { found: 1 });
}

#[test]
fn test_stitch_rejects_bad_tolerance() {
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    assert_eq!(
        stitch(&views, -1.0).unwrap_err(),
        StitchError::InvalidTolerance { tol: -1.0 }
    );
    assert!(matches!(
        stitch(&views, f64::NAN).unwrap_err(),
        StitchError::InvalidTolerance { .. }
    ));
}

#[test]
fn test_stitch_rejects_2d_mesh() {
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0])
        .add_axis(vec![0.0, 1.0])
        .build();
    let views = vec![a.view(), b.view()];
    assert_eq!(
        stitch(&views, 1e-9).unwrap_err(),
        StitchError::InvalidSpaceDimension { mesh: 1, found: 2 }
    );
}

#[test]
fn test_stitch_rejects_surface_mesh_in_3d() {
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let mut b = UMesh::new(
        nd::arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        .to_shared(),
    );
    b.add_element(ElementType::QUAD4, &[0, 1, 2, 3], None);
    let views = vec![a.view(), b.view()];
    assert_eq!(
        stitch(&views, 1e-9).unwrap_err(),
        StitchError::NonVolumeElement {
            mesh: 1,
            element_type: ElementType::QUAD4
        }
    );
}

#[test]
fn test_stitch_matching_interfaces() {
    // Two boxes with identical interface discretization.
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 2);
    assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 1);
    assert_eq!(out.coords().nrows(), 12);
    assert!(
        out.elements()
            .all(|e| e.element_type() == ElementType::PHED)
    );
}

#[test]
fn test_stitch_mismatched_interfaces() {
    // A: 1x1x1 hex, B: 2x2x1 hexes sitting on top, interface is 1 quad vs 4 quads.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 5);
    // 8 + 18 nodes, 4 of which are shared interface corners.
    assert_eq!(out.coords().nrows(), 22);
    // The single quad of A became 4 quads shared with the 4 quads of B.
    assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 4);
}

#[test]
fn test_stitch_partial_overlap() {
    // B's bottom face sticks out of A's top face on the +x side.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 3.0], &[0.0, 1.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 2);
    // A's top quad is split by B's edge x = 2 and B's bottom quad is split by A's edge y = 1,
    // yielding 3 faces on z = 1: one shared interface and one exclusive face per mesh.
    let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
    assert_eq!(counts.len(), 3);
    assert_eq!(counts.values().filter(|&&c| c == 2).count(), 1);
    assert_eq!(counts.values().filter(|&&c| c == 1).count(), 2);
}

#[test]
fn test_stitch_three_meshes() {
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
    let c = box_mesh(&[0.0, 2.0], &[0.0, 1.0, 2.0], &[2.0, 3.0]);
    let views = vec![a.view(), b.view(), c.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 5);
    // A|B: A's 2x2 top is split in 2 by B's x = 1 edge; both pieces are shared with B.
    assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 2);
    // B|C: B's top is 2x1 (x 0..2) and C's bottom is 2x2, so B's top is split in 2 by C's
    // y = 1 edge and C's bottom in 2 by B's x = 1 edge: all 4 pieces are shared.
    assert_eq!(count_interface_faces(&out, 2, 2.0, 1e-12), 4);
}

#[test]
fn test_stitch_accepts_poly_input() {
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
    let a_poly = crate::tools::polyze::polyze(&a.view());
    let b_poly = crate::tools::polyze::polyze(&b.view());
    let views = vec![a_poly.view(), b_poly.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 5);
    assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 4);
}

#[test]
fn test_stitch_families_are_relabeled() {
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();
    let fams: Vec<usize> = out.elements().map(|e| *e.family).collect();
    assert!(fams.contains(&0));
    assert!(fams.contains(&1));
}

#[test]
fn test_stitch_disjoint_meshes() {
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = box_mesh(&[5.0, 6.0], &[0.0, 1.0], &[0.0, 1.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();
    assert_eq!(out.num_elements(), 2);
    // The two boxes only touch the plane z = 1 on their own tops, which stay boundary faces.
    let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
    assert_eq!(counts.len(), 2);
    assert!(counts.values().all(|&c| c == 1));
}

/// Signed volume of the tetrahedron `t`, six times its actual volume.
fn tet6v(coords: &nd::Array2<f64>, t: [usize; 4]) -> f64 {
    let p = |i: usize| [coords[(i, 0)], coords[(i, 1)], coords[(i, 2)]];
    let (a, b, c, d) = (p(t[0]), p(t[1]), p(t[2]), p(t[3]));
    let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

/// A hexahedral grid of `xs` x `ys` x `zs` split into 6 tetrahedra per hex (Kuhn's
/// decomposition around the `0-6` diagonal), oriented consistently.
fn tet_box(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
    let kuhn = [
        [0, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    let mut coords: Vec<[f64; 3]> = Vec::new();
    let mut tets: Vec<[usize; 4]> = Vec::new();
    let (nx, ny, nz) = (xs.len(), ys.len(), zs.len());
    for k in 0..nz - 1 {
        for j in 0..ny - 1 {
            for i in 0..nx - 1 {
                let base = coords.len();
                let local: Vec<[f64; 3]> = [
                    (0, 0, 0),
                    (1, 0, 0),
                    (1, 1, 0),
                    (0, 1, 0),
                    (0, 0, 1),
                    (1, 0, 1),
                    (1, 1, 1),
                    (0, 1, 1),
                ]
                .iter()
                .map(|(di, dj, dk)| [xs[i + di], ys[j + dj], zs[k + dk]])
                .collect();
                coords.extend_from_slice(&local);
                let local_arr =
                    nd::Array2::from_shape_vec((8, 3), local.iter().flatten().copied().collect())
                        .unwrap();
                for t in kuhn {
                    let t = [t[0] + base, t[1] + base, t[2] + base, t[3] + base];
                    let t = if tet6v(
                        &local_arr,
                        [t[0] - base, t[1] - base, t[2] - base, t[3] - base],
                    ) >= 0.0
                    {
                        t
                    } else {
                        [t[0], t[2], t[1], t[3]]
                    };
                    tets.push(t);
                }
            }
        }
    }
    let arr = nd::Array2::from_shape_vec(
        (coords.len(), 3),
        coords.iter().flatten().copied().collect(),
    )
    .unwrap();
    let mut mesh = UMesh::new(arr.into_shared());
    for t in tets {
        mesh.add_element(ElementType::TET4, &t, None);
    }
    mesh
}

/// Total area of the faces lying on the plane `axis = value`.
fn total_area_on_plane(mesh: &UMesh, axis: usize, value: f64, tol: f64) -> f64 {
    faces_on_plane(mesh, axis, value, tol)
        .keys()
        .map(|k| {
            let pts: Vec<[f64; 3]> = k
                .iter()
                .map(|&g| {
                    [
                        mesh.coords()[(g, 0)],
                        mesh.coords()[(g, 1)],
                        mesh.coords()[(g, 2)],
                    ]
                })
                .collect();
            Polygon::unknown(pts).area()
        })
        .sum()
}

/// Known limitation: see `walk_dart_map` in `crates/mefikit/src/element_traits/cut.rs`, which
/// still `todo!()`s when a cutting segment enters a cell without cutting it. Triangular
/// interfaces hit that path, so this test is disabled until the cutter is fixed. The same bug
/// is reachable from the public `overlay_surfaces` API with plain triangles.
#[test]
#[ignore = "blocked on the unimplemented rewind in walk_dart_map"]
fn test_stitch_tetrahedra() {
    // One 2x2x1 hex split into 6 tets (2 triangles on the interface) against a 2x2x1 grid of
    // hexes split into tets (8 triangles on the interface).
    let a = tet_box(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = tet_box(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
    let views = vec![a.view(), b.view()];
    let out = stitch(&views, 1e-9).unwrap();

    assert_eq!(out.num_elements(), 6 + 24);
    // Every triangle of the common refinement is shared by exactly two cells.
    assert!(count_interface_faces(&out, 2, 1.0, 1e-12) > 8);
    // The interface is covered exactly twice: once by A, once by B.
    assert!((total_area_on_plane(&out, 2, 1.0, 1e-12) - 8.0).abs() < 1e-12);
}

#[test]
fn test_stitch_reports_non_planar_region() {
    // Two quads from distinct meshes whose nodes are not coplanar.
    let mk = |z_last: f64| FaceData {
        id: ElementId::new(ElementType::QUAD4, 0),
        et: ElementType::QUAD4,
        ring: vec![0, 1, 2, 3],
        pts: vec![
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, z_last],
            [0.0, 1.0, z_last],
        ],
        normal: [0.0, 0.0, 1.0],
        bounds: [[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]],
    };
    let faces = vec![mk(1.0), mk(1.5)];
    let e = process_region(0, &faces, &[0, 1], &[0, 1], 1e-9).unwrap_err();
    match e {
        StitchError::NonPlanarRegion {
            region, deviation, ..
        } => {
            assert_eq!(region, 0);
            assert!(deviation > 1e-9);
        }
        other => panic!("expected NonPlanarRegion, got {other:?}"),
    }
}

#[test]
fn test_weld_points() {
    let pts = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1e-13],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let (uniq, ids) = weld_points(&pts, 1e-9);
    assert_eq!(uniq.len(), 3);
    assert_eq!(ids[1], ids[3]);
    assert_eq!(ids[0], ids[2]);
    assert_ne!(ids[0], ids[4]);
}

#[test]
fn test_weld_points_paired_against_a_far_neighbour() {
    // Points 0 and 1 differ by a rounding error in x, but point 2 sorts between them and is a
    // whole cell away in z, so neither an anchored nor a chained scan over the sorted order
    // pairs 0 with 1. Point 3 is the same story in y. This is what a rotated interface leaves
    // behind: an imprint node computed by projecting a 2D intersection back to 3D next to the
    // input mesh's own node for the same corner.
    let eps = f64::EPSILON / 2.0;
    let pts = vec![
        [0.0, 0.0, 0.0],
        [0.0, 0.0, eps],
        [0.0, 0.0, 1.0],
        [eps, 0.0, 0.0],
    ];
    let (uniq, ids) = weld_points(&pts, 1e-9);
    assert_eq!(uniq.len(), 2, "0 and 1, and 0 and 3, are all coincident");
    assert_eq!(ids[0], ids[1]);
    assert_eq!(ids[0], ids[3]);
    assert_ne!(ids[0], ids[2]);
    // The lowest input index represents the cluster.
    assert_eq!(uniq[ids[0]], [0.0, 0.0, 0.0]);
}

#[test]
fn test_weld_points_snap_imprints_onto_input_nodes() {
    // An imprint node that lands within `tol` of an input node must be welded onto it, and the
    // input node's own coordinates must win so the output keeps them exactly. `stitch` feeds
    // the input mesh nodes first, so the lower index must represent the cluster even when the
    // imprint sorts before it.
    let eps = f64::EPSILON / 2.0;
    let pts = vec![[0.0, 0.0, 0.0], [0.0, -eps, 0.0]];
    let (uniq, ids) = weld_points(&pts, 1e-9);
    assert_eq!(uniq.len(), 1);
    assert_eq!(uniq[0], [0.0, 0.0, 0.0]);
    assert_eq!(ids[0], ids[1]);
}

// ------------------------------------------------------------------------------------------
// Hex-dominant fixtures. These are the shapes `stitch` is meant for: structured hex blocks and
// already-conformized hex blocks (i.e. `PHED` cells whose faces are no longer all quads).
// ------------------------------------------------------------------------------------------

#[test]
fn test_stitch_hex_grids_offset_in_plane() {
    // A is a single 2x2 hex; B is a 3x2 hex grid whose in-plane lines share no node with A.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(
        &[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0],
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    // A's top quad becomes the 3x2 grid of B, so A stays a single PHED with 12 nodes.
    assert_eq!(rep.cells, 7);
    assert_eq!(rep.interface_faces, 6);
    assert!((rep.interface_area - 4.0).abs() < 1e-12);
    assert!((rep.volume - 8.0).abs() < 1e-9);
}

#[test]
fn test_stitch_hex_grids_rotated_in_plane() {
    // A is a 2x2 hex, B is a 2x2 hex block rotated 45 degrees about the interface centre, so
    // the contact region is an octagon and both sides are cut into non-quadrilateral faces.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = rotate_about_axis(
        &box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]),
        2,
        1.0,
        std::f64::consts::FRAC_PI_4,
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    // Square minus the four corner triangles cut by the rotated square's edges x + y = 2 +- r.
    let r = 2.0 - std::f64::consts::SQRT_2;
    let expected = 4.0 - 2.0 * r * r;
    assert!((rep.interface_area - expected).abs() < 1e-12);
    assert!((rep.volume - 8.0).abs() < 1e-12);
    // The two in-plane lines of the rotated grid cross at the centre, so they cut the
    // octagon into four pentagons: three of its nodes plus the centre and one crossing.
    assert_eq!(rep.interface_faces, 4);
    let face_sizes: BTreeSet<usize> = face_table(&out).into_keys().map(|key| key.len()).collect();
    assert!(
        face_sizes.contains(&5),
        "expected pentagonal faces, got {face_sizes:?}"
    );
}

#[test]
fn test_stitch_hex_blocks_on_a_slanted_plane() {
    // The interface is the plane x + y + z = 3 rather than a coordinate plane.
    let a = rotate_about_axis(
        &box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]),
        0,
        0.0,
        std::f64::consts::FRAC_PI_4,
    );
    let b = rotate_about_axis(
        &box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]),
        0,
        0.0,
        std::f64::consts::FRAC_PI_4,
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert_eq!(rep.cells, 2);
    assert_eq!(rep.interface_faces, 1);
    assert!((rep.interface_area - 4.0).abs() < 1e-12);
}

#[test]
fn test_stitch_hex_blocks_interlocking_on_two_planes() {
    // A is a stepped block: a 2x1x1 hex next to a 1x1x2 hex. B fills the notch, so the two
    // meshes share a z = 1 face *and* an x = 2 face. A2's x = 2 face is a quad that B's
    // x = 2 quad only covers halfway, so it gains a node and becomes a PHED face pair.
    let a = hex_block(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0],
        &[0.0, 1.0, 2.0],
        &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
    );
    let b = hex_block(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0],
        &[1.0, 2.0],
        &[(0, 0, 0), (1, 0, 0)],
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    // z = 1 over x in [0, 2] (area 2) plus x = 2 over z in [1, 2] (area 1).
    assert!((rep.interface_area - 3.0).abs() < 1e-12);
    assert!((rep.volume - 6.0).abs() < 1e-12);
    // Two interface faces on z = 1 (x = [0, 1] and [1, 2]) and one on x = 2 (z = [1, 2]).
    assert_eq!(rep.interface_faces, 3);
    assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 3);
    assert_eq!(count_interface_faces(&out, 0, 2.0, 1e-12), 2);
}

#[test]
fn test_stitch_hex_block_on_l_shaped_footprint() {
    // A is a 2x2 grid of hexes, B an L-shaped block covering three of its four top faces.
    let a = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[0.0, 1.0]);
    let b = hex_block(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[1.0, 2.0],
        &[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert_eq!(rep.cells, 7);
    assert_eq!(rep.interface_faces, 3);
    assert!((rep.interface_area - 3.0).abs() < 1e-12);
    assert!((rep.interface_area - contact_area(&a, &b, 1e-9)).abs() < 1e-12);
    // The uncovered top face of A stays a boundary face.
    let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
    assert_eq!(counts.len(), 4);
    assert_eq!(counts.values().filter(|&&c| c == 1).count(), 1);
    assert_eq!(counts.values().filter(|&&c| c == 2).count(), 3);
}

#[test]
fn test_stitch_two_interface_regions_between_the_same_pair() {
    // One pair of meshes, two disjoint contact patches: region grouping must not merge them.
    let a = box_mesh(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = hex_block(
        &[0.0, 1.0, 2.0, 3.0],
        &[0.0, 1.0],
        &[1.0, 2.0],
        &[(0, 0, 0), (2, 0, 0)],
    );
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert_eq!(rep.cells, 5);
    assert_eq!(rep.interface_faces, 2);
    assert!((rep.interface_area - 2.0).abs() < 1e-12);
    // A's middle top face is left alone: it belongs to no contact region.
    let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
    assert_eq!(counts.values().filter(|&&c| c == 1).count(), 1);
    assert_eq!(counts.values().filter(|&&c| c == 2).count(), 2);
}

#[test]
fn test_stitch_l_shaped_hex_block_against_a_grid() {
    // The "hex-like" case: a concave PHED-shaped block (three hexes in an L) against a grid
    // whose in-plane lines cut its exposed faces into non-quadrilateral pieces.
    let a = hex_block(
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[0.0, 1.0],
        &[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
    );
    let b = box_mesh(&[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert!((rep.interface_area - 3.0).abs() < 1e-12);
    assert!((rep.volume - 7.0).abs() < 1e-12);
    // A's three unit-square top faces are each cut once by B's x lines at 2/3 and 4/3, so
    // the interface is six rectangles of area 1/3 rather than three unit squares.
    assert_eq!(rep.interface_faces, 6);
    assert_eq!(rep.by_pair.get(&(0, 1)), Some(&3.0));
}

#[test]
fn test_stitch_four_hex_blocks_in_a_chain() {
    // Four blocks stacked in a chain, every pair mismatched in-plane.
    let ys_list: [Vec<f64>; 4] = [
        vec![0.0, 1.0, 2.0],
        vec![0.0, 0.5, 1.0, 1.5, 2.0],
        vec![0.0, 1.0, 2.0],
        vec![0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
    ];
    let blocks: Vec<UMesh> = ys_list
        .iter()
        .enumerate()
        .map(|(k, ys)| {
            let k = k as f64;
            box_mesh(&[0.0, 1.0, 2.0], ys, &[k, k + 1.0])
        })
        .collect();
    let views: Vec<_> = blocks.iter().map(UMesh::view).collect();
    let out = stitch(&views, 1e-9).unwrap();
    let rep = check_result(&out, &blocks.iter().collect::<Vec<_>>(), 1e-9);

    // Three interfaces, each covering the full 2x2 footprint, so twelve in total.
    assert!((rep.interface_area - 12.0).abs() < 1e-9);
    assert!((rep.volume - 16.0).abs() < 1e-12);
    assert!(rep.interface_faces >= 6);
}

#[test]
fn test_stitch_hex_blocks_touching_only_along_an_edge() {
    // Zero-area contact: the blocks share a line segment, so nothing must be imprinted.
    let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let b = box_mesh(&[1.0, 2.0], &[0.0, 1.0], &[1.0, 2.0]);
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert_eq!(rep.cells, 2);
    assert_eq!(rep.interface_faces, 0);
    assert_eq!(rep.interface_area, 0.0);
    assert!((rep.volume - 2.0).abs() < 1e-12);
}

#[test]
fn test_stitch_result_can_be_stitched_again() {
    // Stitching an already-conformized result (PHED cells with refined faces) against a new
    // hex block must behave like any other PHED input.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
    let first = stitch(&[a.view(), b.view()], 1e-9).unwrap();

    // Stack a third block on top of the conformed result, with its own in-plane mismatch.
    let top = box_mesh(
        &[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
        &[0.0, 1.0, 2.0],
        &[2.0, 3.0],
    );
    for cell in top.elements() {
        assert_eq!(cell.element_type(), ElementType::HEX8);
    }
    let second = stitch(&[first.view(), top.view()], 1e-9).unwrap();
    let rep = check_result(&second, &[&first, &top], 1e-9);

    assert!((rep.interface_area - 8.0).abs() < 1e-9);
    assert!((rep.volume - 12.0).abs() < 1e-9);
}

#[test]
fn test_stitch_conformized_hex_against_finer_hex() {
    // `polyze` turns the hexes into PHED cells; stitching a polyze'd block against a finer
    // hex grid is the closest stand-in for "already conformized hexa blocks".
    let a = crate::tools::polyze::polyze(&box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]).view());
    let b = box_mesh(&[0.0, 0.5, 1.0, 1.5, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
    let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
    let rep = check_result(&out, &[&a, &b], 1e-9);

    assert!((rep.interface_area - 4.0).abs() < 1e-12);
    assert!((rep.volume - 8.0).abs() < 1e-12);
    // A's single top quad is split 4x2 by B, so it becomes a PHED with 4 + 8 + 4 nodes.
    assert_eq!(rep.cells, 9);
    assert_eq!(rep.interface_faces, 8);
}

#[test]
fn test_stitch_third_block_touching_both_others_with_partial_overlap() {
    // A plate under a wider, refined block, with a third block butting against both of them
    // along x. The contact patch on x = 0 therefore carries three meshes, and each part of it
    // is covered by the third block and one of the two others only.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 1.0, 2.2], &[0.0, 1.0, 2.5], &[1.0, 2.0]);
    let c = box_mesh(&[-1.0, -0.5, 0.0], &[0.0, 2.0], &[0.0, 2.0]);
    let views = vec![a.view(), b.view(), c.view()];
    let out = stitch(&views, 1e-9).unwrap();

    let rep = check_result(&out, &[&a, &b, &c], 1e-9);

    // 1 + 4 + 2 input cells, none of them split since C's faces already match B's lines.
    assert_eq!(rep.cells, 7);
    assert!((rep.interface_area - 8.0).abs() < 1e-12);
    assert_eq!(rep.by_pair.get(&(0, 1)), Some(&4.0));
    assert_eq!(rep.by_pair.get(&(0, 2)), Some(&2.0));
    assert_eq!(rep.by_pair.get(&(1, 2)), Some(&2.0));
}

#[test]
fn test_stitch_three_blocks_where_the_third_touches_both_others() {
    // A and B stack along z while C touches them both along x, so the interface region of
    // C carries three meshes and no part of it is covered by all of them at once.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
    let c = box_mesh(&[2.0, 3.0], &[0.0, 2.0], &[0.0, 2.0]);
    let views = vec![a.view(), b.view(), c.view()];
    let out = stitch(&views, 1e-9).unwrap();

    let rep = check_result(&out, &[&a, &b, &c], 1e-9);

    assert_eq!(rep.cells, 3);
    assert_eq!(rep.interface_faces, 3);
    // A|B on z = 1, A|C on x = 2 over z 0..1, B|C on x = 2 over z 1..2.
    assert!((rep.interface_area - 8.0).abs() < 1e-12);
    assert!((rep.volume - 12.0).abs() < 1e-12);
}

#[test]
fn test_stitch_three_blocks_where_the_third_is_imprinted_on_both() {
    // Same assembly, but C is refined along y and z, so both of its faces have to imprint
    // their lines into the faces of A and B.
    let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
    let b = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
    let c = box_mesh(&[2.0, 3.0], &[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
    let views = vec![a.view(), b.view(), c.view()];
    let out = stitch(&views, 1e-9).unwrap();

    let rep = check_result(&out, &[&a, &b, &c], 1e-9);

    // C is refined 2x2, so the total cell count and volume are those of the three inputs.
    assert_eq!(rep.cells, 6);
    // One face for A|B, then A's and B's x = 2 faces are each split in 2 by C's y = 1 line.
    assert_eq!(rep.interface_faces, 5);
    assert!((rep.interface_area - 8.0).abs() < 1e-12);
    assert_eq!(rep.by_pair.get(&(0, 1)), Some(&4.0));
    assert_eq!(rep.by_pair.get(&(0, 2)), Some(&2.0));
    assert_eq!(rep.by_pair.get(&(1, 2)), Some(&2.0));
    assert!((rep.volume - 12.0).abs() < 1e-12);
}

#[test]
fn test_stitch_without_boundary_welds_and_polyizes() {
    // Degenerate input: every face of the mesh is used by two coincident cells, so there is no
    // boundary skin at all and `stitch` falls back to welding the input and polyizing it.
    let single = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
    let mut m = UMesh::new(single.coords().to_shared());
    for _ in 0..2 {
        for cell in single.elements() {
            m.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
        }
    }
    let out = stitch(&[m.view(), m.view()], 1e-9).unwrap();

    assert_eq!(out.num_elements(), 4);
    assert_eq!(out.coords().nrows(), 8);
    assert!(
        out.elements()
            .all(|e| e.element_type() == ElementType::PHED)
    );
    assert!((total_volume(&out) - 4.0).abs() < 1e-12);
}
