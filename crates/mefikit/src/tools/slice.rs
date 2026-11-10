//! Slice a mesh with a plane or a line, producing a mesh of one dimension lower.
//!
//! - A 3D mesh cut by a plane produces a 2D mesh made of `PGON` cells embedded in 3D.
//! - A 3D mesh cut by a line produces a 1D mesh made of `SEG2` cells embedded in 3D.
//! - A 2D mesh cut by a line produces a 1D mesh made of `SEG2` cells embedded in 2D.
//!
//! [`slice`] defines the cut by a point `origin` and a `normal`: in 3D this is the plane
//! through `origin` with normal `normal`, and in 2D the line through `origin` normal to
//! `normal`. [`slice_line`] defines the cut by a point and a direction, in 2D or 3D.
//!
//! Only cells of the highest dimension are sliced. Lower-dimensional elements are ignored.
//! Fields are not carried over: use the transfer operators to map source fields onto the
//! produced mesh (see the slicing example in the Python documentation).

use ndarray as nd;
use rustc_hash::FxHashSet;

use crate::mesh::{Dimension, Element, ElementLike, ElementType, UMesh, UMeshView};
use crate::tools::snap::merge_nodes;

/// Slice a mesh with the hyperplane through `origin` with normal `normal`.
///
/// The space dimension of the mesh (2 or 3) selects the behaviour:
/// - 2D mesh -> `SEG2` segments along the line through `origin` normal to `normal`;
/// - 3D mesh -> `PGON` polygons on the plane through `origin` with normal `normal`.
///
/// The produced mesh keeps the space dimension of the input, so a slice of a 3D mesh is a 2D
/// mesh embedded in 3D space and can be visualized together with the source mesh.
///
/// # Panics
/// Panics if the mesh is not in 2D or 3D space, if `origin`/`normal` do not match the space
/// dimension, or if `normal` is zero.
pub fn slice(mesh: &UMeshView, origin: &[f64], normal: &[f64]) -> UMesh {
    let space_dim = mesh.coords().ncols();
    assert!(
        space_dim == 2 || space_dim == 3,
        "Slicing requires a 2D or 3D mesh, got a {space_dim}D mesh"
    );
    assert_eq!(
        origin.len(),
        space_dim,
        "The slice origin must have the same dimension as the mesh"
    );
    assert_eq!(
        normal.len(),
        space_dim,
        "The slice normal must have the same dimension as the mesh"
    );

    match space_dim {
        2 => slice_2d(mesh, origin.try_into().unwrap(), normal.try_into().unwrap()),
        3 => slice_3d(mesh, origin.try_into().unwrap(), normal.try_into().unwrap()),
        _ => unreachable!(),
    }
}

/// Slice a mesh with the line through `origin` with direction `direction`.
///
/// - A 3D mesh produces a 1D `SEG2` mesh embedded in 3D.
/// - A 2D mesh produces a 1D `SEG2` mesh embedded in 2D.
///
/// The line direction does not need to be normalized.
///
/// # Panics
/// Panics if the mesh is not in 2D or 3D space, if `origin`/`direction` do not match the
/// space dimension, or if `direction` is zero.
pub fn slice_line(mesh: &UMeshView, origin: &[f64], direction: &[f64]) -> UMesh {
    let space_dim = mesh.coords().ncols();
    assert!(
        space_dim == 2 || space_dim == 3,
        "Line slicing requires a 2D or 3D mesh, got a {space_dim}D mesh"
    );
    assert_eq!(
        origin.len(),
        space_dim,
        "The slice origin must have the same dimension as the mesh"
    );
    assert_eq!(
        direction.len(),
        space_dim,
        "The slice direction must have the same dimension as the mesh"
    );

    match space_dim {
        2 => {
            let origin: [f64; 2] = origin.try_into().unwrap();
            let direction: [f64; 2] = unit(&direction.try_into().unwrap());
            slice_2d(mesh, origin, [-direction[1], direction[0]])
        }
        3 => slice_line_3d(
            mesh,
            origin.try_into().unwrap(),
            direction.try_into().unwrap(),
        ),
        _ => unreachable!(),
    }
}

/// Trait for meshes that can be sliced by a plane/line or by a line direction.
pub trait Sliceable {
    /// Slices the mesh with the hyperplane through `origin` with normal `normal`.
    ///
    /// See [`slice`] for the dimensional conventions.
    fn slice(&self, origin: &[f64], normal: &[f64]) -> UMesh;

    /// Slices the mesh with the line through `origin` with direction `direction`.
    ///
    /// See [`slice_line`] for the dimensional conventions.
    fn slice_line(&self, origin: &[f64], direction: &[f64]) -> UMesh;
}

impl Sliceable for UMeshView<'_> {
    fn slice(&self, origin: &[f64], normal: &[f64]) -> UMesh {
        slice(self, origin, normal)
    }

    fn slice_line(&self, origin: &[f64], direction: &[f64]) -> UMesh {
        slice_line(self, origin, direction)
    }
}

impl Sliceable for UMesh {
    fn slice(&self, origin: &[f64], normal: &[f64]) -> UMesh {
        slice(&self.view(), origin, normal)
    }

    fn slice_line(&self, origin: &[f64], direction: &[f64]) -> UMesh {
        slice_line(&self.view(), origin, direction)
    }
}

fn empty_mesh(space_dim: usize) -> UMesh {
    UMesh::new(nd::Array2::zeros((0, space_dim)).into_shared())
}

fn slice_3d(mesh: &UMeshView, origin: [f64; 3], normal: [f64; 3]) -> UMesh {
    let normal = unit(&normal);
    let (u, v) = plane_basis(&normal);
    let tol = slice_tolerance(mesh);

    let mut coords: Vec<[f64; 3]> = Vec::new();
    let mut polygons: Vec<(Vec<usize>, usize)> = Vec::new();

    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (points, loops) = section_cell_by_plane(&cell, &origin, &normal, &u, &v, tol);
        if loops.is_empty() {
            continue;
        }

        let base = coords.len();
        coords.extend(points.iter().map(|p| deproject(p, &origin, &u, &v)));
        for nodes in loops {
            let connectivity: Vec<usize> = nodes.iter().map(|&i| base + i).collect();
            polygons.push((connectivity, *cell.family));
        }
    }

    if polygons.is_empty() {
        return empty_mesh(3);
    }

    let coords =
        nd::Array2::from_shape_vec((coords.len(), 3), coords.into_iter().flatten().collect())
            .unwrap();
    let mut out = UMesh::new(coords.into_shared());
    for (connectivity, family) in polygons {
        out.add_element(ElementType::PGON, &connectivity, Some(family));
    }
    merge_nodes(&mut out, tol);
    let out = deduplicate_cells(out);
    out.remove_unused_coords()
}

/// Cuts a single 3D cell with a plane and returns the section loops in the plane local basis.
///
/// `u`/`v` are an orthonormal in-plane basis with `u x v = normal`.
fn section_cell_by_plane(
    cell: &Element<'_>,
    origin: &[f64; 3],
    normal: &[f64; 3],
    u: &[f64; 3],
    v: &[f64; 3],
    tol: f64,
) -> (Vec<[f64; 2]>, Vec<Vec<usize>>) {
    let mut segments = Vec::new();
    for face in cell_faces(cell) {
        let polygon: Vec<[f64; 3]> = face.iter().map(|&node| coord3(cell, node)).collect();
        for segment in plane_section_segments(&polygon, origin, normal, tol) {
            segments.push(segment.map(|p| project(&p, origin, u, v)));
        }
    }
    stitch_segments(&segments, tol)
}

fn slice_line_3d(mesh: &UMeshView, origin: [f64; 3], direction: [f64; 3]) -> UMesh {
    let direction = unit(&direction);
    let tol = slice_tolerance(mesh);

    // Build a plane that contains the line. Its in-plane basis is (direction, e2), so the
    // line itself is simply the local line y = 0.
    let seed = least_aligned_axis(&direction);
    let e2 = unit(&cross(&direction, &seed));
    let plane_normal = cross(&direction, &e2);

    let mut coords: Vec<[f64; 3]> = Vec::new();
    let mut segments: Vec<[usize; 2]> = Vec::new();

    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (points, loops) =
            section_cell_by_plane(&cell, &origin, &plane_normal, &direction, &e2, tol);
        for nodes in loops {
            let polygon: Vec<[f64; 2]> = nodes.iter().map(|&i| points[i]).collect();
            for [a, b] in plane_section_segments(&polygon, &[0.0, 0.0], &[0.0, 1.0], tol) {
                let ia = coords.len();
                coords.push([
                    origin[0] + a[0] * direction[0],
                    origin[1] + a[0] * direction[1],
                    origin[2] + a[0] * direction[2],
                ]);
                let ib = coords.len();
                coords.push([
                    origin[0] + b[0] * direction[0],
                    origin[1] + b[0] * direction[1],
                    origin[2] + b[0] * direction[2],
                ]);
                segments.push([ia, ib]);
            }
        }
    }

    build_segment_mesh(coords, segments, tol)
}

fn slice_2d(mesh: &UMeshView, origin: [f64; 2], normal: [f64; 2]) -> UMesh {
    let normal = unit(&normal);
    let tol = slice_tolerance(mesh);

    let mut coords: Vec<[f64; 2]> = Vec::new();
    let mut segments: Vec<[usize; 2]> = Vec::new();

    for cell in mesh.elements_of_dim(Dimension::D2) {
        let polygon = cell_polygon(&cell);
        for [a, b] in plane_section_segments(&polygon, &origin, &normal, tol) {
            let ia = coords.len();
            coords.push(a);
            let ib = coords.len();
            coords.push(b);
            segments.push([ia, ib]);
        }
    }

    build_segment_mesh(coords, segments, tol)
}

/// Builds a `SEG2` mesh from line segments embedded in a 2D or 3D space.
fn build_segment_mesh<const D: usize>(
    coords: Vec<[f64; D]>,
    segments: Vec<[usize; 2]>,
    tol: f64,
) -> UMesh {
    if segments.is_empty() {
        return empty_mesh(D);
    }

    let coords =
        nd::Array2::from_shape_vec((coords.len(), D), coords.into_iter().flatten().collect())
            .unwrap();
    let connectivity = nd::Array2::from_shape_vec(
        (segments.len(), 2),
        segments.into_iter().flatten().collect(),
    )
    .unwrap();
    let mut out = UMesh::new(coords.into_shared());
    out.add_regular_block(ElementType::SEG2, connectivity.into_shared(), None);
    merge_nodes(&mut out, tol);
    let out = deduplicate_cells(out);
    out.remove_unused_coords()
}

/// Removes cells with the same node set. This happens when the cutting hyperplane coincides
/// with an internal interface: both neighbouring cells then produce the same section cell.
fn deduplicate_cells(mesh: UMesh) -> UMesh {
    let mut out = UMesh::new(mesh.coords().to_shared());
    let mut seen = FxHashSet::default();
    for cell in mesh.elements() {
        let mut key = cell.connectivity().to_vec();
        key.sort_unstable();
        if seen.insert((cell.element_type(), key)) {
            out.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
        }
    }
    out
}

/// Tolerance of the geometric predicates, scaled to the mesh extent and coordinates.
fn slice_tolerance(mesh: &UMeshView) -> f64 {
    let coords = mesh.coords();
    if coords.nrows() == 0 {
        return 1e-12;
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    let mut magnitude = 0.0_f64;
    for row in coords.rows() {
        for (k, &value) in row.iter().enumerate() {
            lo[k] = lo[k].min(value);
            hi[k] = hi[k].max(value);
            magnitude = magnitude.max(value.abs());
        }
    }
    let extent = (0..coords.ncols())
        .map(|k| hi[k] - lo[k])
        .fold(0.0_f64, f64::max);
    1e-12 * extent + 64.0 * f64::EPSILON * magnitude
}

fn unit<const D: usize>(v: &[f64; D]) -> [f64; D] {
    let n = dot(v, v).sqrt();
    assert!(n > 0.0, "The slice normal must not be zero");
    v.map(|x| x / n)
}

fn dot<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn dist2<const D: usize>(a: &[f64; D], b: &[f64; D]) -> f64 {
    (0..D).map(|k| (a[k] - b[k]).powi(2)).sum()
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Coordinate axis least aligned with `vector`, used to build a well-conditioned basis.
fn least_aligned_axis(vector: &[f64; 3]) -> [f64; 3] {
    if vector[0].abs() <= vector[1].abs() && vector[0].abs() <= vector[2].abs() {
        [1.0, 0.0, 0.0]
    } else if vector[1].abs() <= vector[2].abs() {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// Orthonormal in-plane basis `(u, v)` with `u x v = normal`.
fn plane_basis(normal: &[f64; 3]) -> ([f64; 3], [f64; 3]) {
    let seed = least_aligned_axis(normal);
    let d = dot(&seed, normal);
    let mut u = [0.0; 3];
    for k in 0..3 {
        u[k] = seed[k] - d * normal[k];
    }
    u = unit(&u);
    let v = cross(normal, &u);
    (u, v)
}

fn project(p: &[f64; 3], origin: &[f64; 3], u: &[f64; 3], v: &[f64; 3]) -> [f64; 2] {
    let d = [p[0] - origin[0], p[1] - origin[1], p[2] - origin[2]];
    [dot(&d, u), dot(&d, v)]
}

fn deproject(p: &[f64; 2], origin: &[f64; 3], u: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        origin[0] + p[0] * u[0] + p[1] * v[0],
        origin[1] + p[0] * u[1] + p[1] * v[1],
        origin[2] + p[0] * u[2] + p[1] * v[2],
    ]
}

/// Coordinates of the local connectivity entry `local` of a cell.
fn coord3(cell: &Element<'_>, local: usize) -> [f64; 3] {
    let c = cell.coord(local);
    [c[0], c[1], c[2]]
}

/// Returns the corner vertices of a 2D cell in ring order.
fn cell_polygon(cell: &Element<'_>) -> Vec<[f64; 2]> {
    use ElementType::*;
    let n = match cell.element_type() {
        TRI3 | TRI6 | TRI7 => 3,
        QUAD4 | QUAD8 | QUAD9 => 4,
        PGON => cell.connectivity().len(),
        _ => panic!(
            "Cannot slice a {:?} cell as a 2D polygon",
            cell.element_type()
        ),
    };
    (0..n)
        .map(|local| {
            let c = cell.coord(local);
            [c[0], c[1]]
        })
        .collect()
}

/// Returns the faces of a 3D cell as rings of local connectivity indices.
fn cell_faces(cell: &Element<'_>) -> Vec<Vec<usize>> {
    use ElementType::*;
    let mut faces = Vec::new();
    match cell.element_type() {
        TET4 | TET10 => {
            faces.push(vec![0, 1, 3]);
            faces.push(vec![1, 2, 3]);
            faces.push(vec![2, 0, 3]);
            faces.push(vec![0, 2, 1]);
        }
        WEDGE6 | WEDGE15 => {
            faces.push(vec![0, 2, 1]);
            faces.push(vec![3, 4, 5]);
            faces.push(vec![0, 1, 4, 3]);
            faces.push(vec![1, 2, 5, 4]);
            faces.push(vec![2, 0, 3, 5]);
        }
        HEX8 | HEX21 => {
            faces.push(vec![0, 3, 2, 1]);
            faces.push(vec![4, 5, 6, 7]);
            faces.push(vec![0, 1, 5, 4]);
            faces.push(vec![2, 3, 7, 6]);
            faces.push(vec![1, 2, 6, 5]);
            faces.push(vec![3, 0, 4, 7]);
        }
        PHED => {
            let mut face = Vec::new();
            for (local, &node) in cell.connectivity().iter().enumerate() {
                if node == usize::MAX {
                    if face.len() >= 3 {
                        faces.push(std::mem::take(&mut face));
                    } else {
                        face.clear();
                    }
                } else {
                    face.push(local);
                }
            }
            if face.len() >= 3 {
                faces.push(face);
            }
        }
        _ => panic!(
            "Cannot slice a {:?} cell as a 3D volume",
            cell.element_type()
        ),
    }
    faces
}

/// Adds `p` to `coords` unless an equal coordinate is already present. Returns its index.
fn push_coord<const D: usize>(coords: &mut Vec<[f64; D]>, p: [f64; D], tol: f64) -> usize {
    let tol2 = tol * tol;
    if let Some(index) = coords.iter().position(|q| dist2(q, &p) <= tol2) {
        index
    } else {
        coords.push(p);
        coords.len() - 1
    }
}

/// Intersection of a polygon with the hyperplane through `origin` with unit `normal`.
///
/// The intersection is computed by clipping the polygon to one side of the hyperplane and
/// keeping the boundary edges that lie on the hyperplane. This handles arbitrary (possibly
/// concave) polygons and yields one segment per crossing interval.
fn plane_section_segments<const D: usize>(
    polygon: &[[f64; D]],
    origin: &[f64; D],
    normal: &[f64; D],
    tol: f64,
) -> Vec<[[f64; D]; 2]> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }

    let distances: Vec<f64> = polygon
        .iter()
        .map(|p| (0..D).map(|k| (p[k] - origin[k]) * normal[k]).sum::<f64>())
        .collect();
    let has_positive = distances.iter().any(|&d| d > tol);
    let has_negative = distances.iter().any(|&d| d < -tol);
    if !has_positive && !has_negative {
        // The polygon lies in the hyperplane: it has no proper section.
        return Vec::new();
    }

    // Keep the side that contains the polygon interior; plane values are included in both.
    let sign = if has_positive { 1.0 } else { -1.0 };
    let signed: Vec<f64> = distances.iter().map(|&d| sign * d).collect();
    let inside = |d: f64| d >= -tol;

    let mut clipped = Vec::with_capacity(n + 2);
    for i in 0..n {
        let j = (i + 1) % n;
        let (da, db) = (signed[i], signed[j]);
        let (inside_a, inside_b) = (inside(da), inside(db));
        match (inside_a, inside_b) {
            (true, true) => push_unique(&mut clipped, polygon[j], tol),
            (true, false) => {
                let p = plane_intersection(polygon[i], da, polygon[j], db, origin, normal);
                push_unique(&mut clipped, p, tol);
            }
            (false, true) => {
                let p = plane_intersection(polygon[i], da, polygon[j], db, origin, normal);
                push_unique(&mut clipped, p, tol);
                push_unique(&mut clipped, polygon[j], tol);
            }
            (false, false) => {}
        }
    }

    // Remove the closing duplicate left by the cyclic walk.
    if clipped.len() > 1 && dist2(&clipped[0], &clipped[clipped.len() - 1]) <= tol * tol {
        clipped.pop();
    }
    if clipped.len() < 2 {
        return Vec::new();
    }

    let mut segments: Vec<[[f64; D]; 2]> = Vec::new();
    for i in 0..clipped.len() {
        let a = clipped[i];
        let b = clipped[(i + 1) % clipped.len()];
        let da = (0..D).map(|k| (a[k] - origin[k]) * normal[k]).sum::<f64>();
        let db = (0..D).map(|k| (b[k] - origin[k]) * normal[k]).sum::<f64>();
        if da.abs() <= tol && db.abs() <= tol && dist2(&a, &b) > tol * tol {
            let duplicate = segments.iter().any(|[p, q]| {
                (dist2(p, &a) <= tol * tol && dist2(q, &b) <= tol * tol)
                    || (dist2(p, &b) <= tol * tol && dist2(q, &a) <= tol * tol)
            });
            if !duplicate {
                segments.push([a, b]);
            }
        }
    }
    segments
}

fn push_unique<const D: usize>(points: &mut Vec<[f64; D]>, p: [f64; D], tol: f64) {
    if points.last().is_none_or(|q| dist2(q, &p) > tol * tol) {
        points.push(p);
    }
}

fn plane_intersection<const D: usize>(
    a: [f64; D],
    da: f64,
    b: [f64; D],
    db: f64,
    origin: &[f64; D],
    normal: &[f64; D],
) -> [f64; D] {
    let denom = da - db;
    let t = if denom.abs() > f64::MIN_POSITIVE {
        da / denom
    } else {
        0.5
    };
    let mut p = [0.0; D];
    for k in 0..D {
        p[k] = a[k] + t * (b[k] - a[k]);
    }
    // Snap the point onto the hyperplane to remove the interpolation error.
    let d: f64 = (0..D).map(|k| (p[k] - origin[k]) * normal[k]).sum();
    for k in 0..D {
        p[k] -= d * normal[k];
    }
    p
}

/// Stitches plane segments of a single cell into closed loops.
///
/// Returns the (deduplicated) 2D points and loops of point indices. Loops are oriented
/// counter-clockwise.
fn stitch_segments(segments: &[[[f64; 2]; 2]], tol: f64) -> (Vec<[f64; 2]>, Vec<Vec<usize>>) {
    let mut points: Vec<[f64; 2]> = Vec::new();
    let mut edges: Vec<[usize; 2]> = Vec::new();

    for segment in segments {
        let a = push_coord(&mut points, segment[0], tol);
        let b = push_coord(&mut points, segment[1], tol);
        if a == b {
            continue;
        }
        let edge = if a < b { [a, b] } else { [b, a] };
        if !edges.contains(&edge) {
            edges.push(edge);
        }
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); points.len()];
    for (edge_id, edge) in edges.iter().enumerate() {
        adjacency[edge[0]].push(edge_id);
        adjacency[edge[1]].push(edge_id);
    }

    let mut used = vec![false; edges.len()];
    let mut loops = Vec::new();
    for start_edge in 0..edges.len() {
        if used[start_edge] {
            continue;
        }
        let start_node = edges[start_edge][0];
        let mut previous = start_node;
        let mut current = edges[start_edge][1];
        used[start_edge] = true;
        let mut nodes = vec![start_node];
        let mut valid = true;

        while current != start_node {
            nodes.push(current);
            let next_edge = select_next_edge(
                current,
                previous,
                &adjacency[current],
                &edges,
                &used,
                &points,
            );
            let Some(next_edge) = next_edge else {
                valid = false;
                break;
            };
            used[next_edge] = true;
            let next = if edges[next_edge][0] == current {
                edges[next_edge][1]
            } else {
                edges[next_edge][0]
            };
            previous = current;
            current = next;
        }

        if valid && nodes.len() >= 3 {
            let area = signed_area(&nodes, &points);
            if area.abs() > tol * tol {
                if area < 0.0 {
                    nodes.reverse();
                }
                loops.push(nodes);
            }
        }
    }

    (points, loops)
}

/// Chooses the next edge of a boundary walk. In the usual degree-2 case the choice is unique.
/// At a degenerate node, the straightest continuation is used.
fn select_next_edge(
    current: usize,
    previous: usize,
    candidates: &[usize],
    edges: &[[usize; 2]],
    used: &[bool],
    points: &[[f64; 2]],
) -> Option<usize> {
    let available: Vec<usize> = candidates
        .iter()
        .copied()
        .filter(|&edge| !used[edge])
        .collect();
    let first = *available.first()?;
    if available.len() == 1 {
        return Some(first);
    }

    let incoming = [
        points[current][0] - points[previous][0],
        points[current][1] - points[previous][1],
    ];
    let incoming_norm = (incoming[0] * incoming[0] + incoming[1] * incoming[1]).sqrt();
    let turn = |edge: usize| {
        let next = if edges[edge][0] == current {
            edges[edge][1]
        } else {
            edges[edge][0]
        };
        let outgoing = [
            points[next][0] - points[current][0],
            points[next][1] - points[current][1],
        ];
        let outgoing_norm = (outgoing[0] * outgoing[0] + outgoing[1] * outgoing[1]).sqrt();
        let cos = (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (incoming_norm * outgoing_norm);
        cos.clamp(-1.0, 1.0).acos().abs()
    };

    let mut best = first;
    let mut best_turn = turn(first);
    for &edge in &available[1..] {
        let candidate_turn = turn(edge);
        if candidate_turn < best_turn {
            best = edge;
            best_turn = candidate_turn;
        }
    }
    Some(best)
}

fn signed_area(nodes: &[usize], points: &[[f64; 2]]) -> f64 {
    let n = nodes.len();
    let mut area2 = 0.0;
    for i in 0..n {
        let a = points[nodes[i]];
        let b = points[nodes[(i + 1) % n]];
        area2 += a[0] * b[1] - b[0] * a[1];
    }
    area2 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray as nd;

    use crate::element_traits::ElementGeo;
    use crate::mesh::{ElementType, FieldArcD, UMesh};
    use crate::mesh_examples::make_imesh_3d;
    use crate::tools::transfer::{FieldNature, Transfer, TransferMethod, TransferOperator};

    fn polygon_area(cell: &Element<'_>) -> f64 {
        let points: Vec<[f64; 3]> = (0..cell.connectivity().len())
            .map(|i| cell.coord3(i).into())
            .collect();
        crate::geometry::Polygon::unknown(points).area()
    }

    fn slice_area(mesh: &UMesh) -> f64 {
        mesh.elements_of_dim(Dimension::D2)
            .map(|cell| polygon_area(&cell))
            .sum()
    }

    #[test]
    fn slice_hex_cube_midplane() {
        let mesh = make_imesh_3d(1);
        let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D2), 1);
        assert_eq!(sliced.coords().nrows(), 4);
        assert_abs_diff_eq!(slice_area(&sliced), 1.0, epsilon = 1e-12);
        for row in sliced.coords().rows() {
            assert_abs_diff_eq!(row[2], 0.5, epsilon = 1e-12);
        }
        assert!(sliced.element_types().all(|&et| et == ElementType::PGON));
    }

    #[test]
    fn slice_hex_grid_midplane() {
        let mesh = make_imesh_3d(2);
        let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D2), 4);
        assert_eq!(sliced.coords().nrows(), 9);
        assert_abs_diff_eq!(slice_area(&sliced), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn slice_tet_midplane() {
        let mesh = crate::mesh_examples::make_mesh_3d_tet();
        let sliced = mesh.slice(&[0.5, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert_eq!(sliced.num_elements_of_dim(Dimension::D2), 1);
        let cell = sliced.elements_of_dim(Dimension::D2).next().unwrap();
        assert_eq!(cell.connectivity().len(), 3);
        assert_abs_diff_eq!(polygon_area(&cell), 3f64.sqrt() / 8.0, epsilon = 1e-12);
    }

    #[test]
    fn slice_cube_on_boundary_plane() {
        let mesh = make_imesh_3d(1);
        let sliced = mesh.slice(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D2), 1);
        assert_abs_diff_eq!(slice_area(&sliced), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn slice_line_3d_through_hex() {
        let mesh = make_imesh_3d(1);
        let sliced = mesh.slice_line(&[0.5, 0.5, 0.0], &[0.0, 0.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 1);
        assert_eq!(sliced.coords().nrows(), 2);
        assert!(sliced.element_types().all(|&et| et == ElementType::SEG2));
        let length = sliced
            .elements_of_dim(Dimension::D1)
            .next()
            .unwrap()
            .measure3();
        assert_abs_diff_eq!(length, 1.0, epsilon = 1e-12);
        for row in sliced.coords().rows() {
            assert_abs_diff_eq!(row[0], 0.5, epsilon = 1e-12);
            assert_abs_diff_eq!(row[1], 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn slice_line_3d_oblique() {
        let mesh = make_imesh_3d(1);
        let sliced = mesh.slice_line(&[0.5, 0.5, 0.5], &[1.0, 1.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 1);
        let length = sliced
            .elements_of_dim(Dimension::D1)
            .next()
            .unwrap()
            .measure3();
        assert_abs_diff_eq!(length, 3f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn slice_line_3d_on_internal_interface_deduplicates_cells() {
        let mesh = make_imesh_3d(2);
        let sliced = mesh.slice_line(&[0.5, 0.5, 0.0], &[0.0, 0.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 2);
        assert_eq!(sliced.coords().nrows(), 3);
        let length: f64 = sliced
            .elements_of_dim(Dimension::D1)
            .map(|cell| cell.measure3())
            .sum();
        assert_abs_diff_eq!(length, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn slice_line_2d_matches_hyperplane_slice() {
        let mesh = crate::mesh_examples::make_imesh_2d(1);
        let by_normal = mesh.slice(&[0.0, 1.0], &[1.0, 1.0]);
        let by_direction = mesh.slice_line(&[0.0, 1.0], &[1.0, -1.0]);

        assert_eq!(
            by_normal.num_elements_of_dim(Dimension::D1),
            by_direction.num_elements_of_dim(Dimension::D1)
        );
        let length = by_direction
            .elements_of_dim(Dimension::D1)
            .next()
            .unwrap()
            .measure2();
        assert_abs_diff_eq!(length, 2f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn slice_line_then_transfer_volume_field() {
        let mut mesh = make_imesh_3d(1);
        let field = FieldArcD::new(std::collections::BTreeMap::from([(
            ElementType::HEX8,
            nd::array![5.0].into_dyn().into_shared(),
        )]));
        mesh.update_field("T", field);
        let sliced = mesh.slice_line(&[0.5, 0.5, 0.0], &[0.0, 0.0, 1.0]);
        let op = TransferOperator::new(
            &mesh.view(),
            &sliced.view(),
            TransferMethod::ConstantPiecewise {
                point_location: crate::tools::transfer::PointLocation::Centroid,
            },
        );
        let field = mesh.field("T", Some(Dimension::D3)).unwrap();
        let transferred = op.apply(&field, FieldNature::Intensive, 0.0);
        assert_eq!(transferred.0[&ElementType::SEG2][0], 5.0);
    }

    #[test]
    fn slice_2d_quad_line() {
        let mesh = crate::mesh_examples::make_imesh_2d(1);
        let sliced = mesh.slice(&[0.5, 0.0], &[1.0, 0.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 1);
        assert_eq!(sliced.coords().nrows(), 2);
        assert!(sliced.element_types().all(|&et| et == ElementType::SEG2));
        let length: f64 = sliced
            .elements_of_dim(Dimension::D1)
            .map(|cell| cell.measure2())
            .sum();
        assert_abs_diff_eq!(length, 1.0, epsilon = 1e-12);
        for row in sliced.coords().rows() {
            assert_abs_diff_eq!(row[0], 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn slice_2d_grid_line() {
        let mesh = crate::mesh_examples::make_imesh_2d(2);
        let sliced = mesh.slice(&[0.5, 0.0], &[1.0, 0.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 2);
        assert_eq!(sliced.coords().nrows(), 3);
        let length: f64 = sliced
            .elements_of_dim(Dimension::D1)
            .map(|cell| cell.measure2())
            .sum();
        assert_abs_diff_eq!(length, 1.0, epsilon = 1e-12);
        for row in sliced.coords().rows() {
            assert_abs_diff_eq!(row[0], 0.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn slice_2d_diagonal_line() {
        let mesh = crate::mesh_examples::make_imesh_2d(1);
        let sliced = mesh.slice(&[0.0, 1.0], &[1.0, 1.0]);

        assert_eq!(sliced.num_elements_of_dim(Dimension::D1), 1);
        let length = sliced
            .elements_of_dim(Dimension::D1)
            .next()
            .unwrap()
            .measure2();
        assert_abs_diff_eq!(length, 2f64.sqrt(), epsilon = 1e-12);
    }

    #[test]
    fn slice_empty_mesh() {
        let mesh = UMesh::new(nd::Array2::zeros((0, 3)).into_shared());
        let sliced = mesh.slice(&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0]);
        assert_eq!(sliced.num_elements(), 0);
    }

    #[test]
    fn slice_all_volume_element_types() {
        use ElementType::*;
        let corners = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];

        for (et, n_nodes, conn) in [
            (TET4, 4, vec![0, 1, 3, 4]),
            (TET10, 10, vec![0, 1, 3, 4, 8, 8, 8, 8, 8, 8]),
            (HEX8, 8, vec![0, 1, 2, 3, 4, 5, 6, 7]),
            (
                HEX21,
                21,
                vec![
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                ],
            ),
            (WEDGE6, 6, vec![0, 1, 3, 4, 5, 7]),
            (
                WEDGE15,
                15,
                vec![0, 1, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8],
            ),
        ] {
            let mut coords = nd::Array2::zeros((n_nodes.max(corners.len()), 3));
            for (i, corner) in corners.iter().enumerate() {
                for k in 0..3 {
                    coords[[i, k]] = corner[k];
                }
            }
            let mut mesh = UMesh::new(coords.into_shared());
            mesh.add_element(et, &conn, None);
            let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);
            assert_eq!(
                sliced.num_elements_of_dim(Dimension::D2),
                1,
                "slicing {et:?} should produce one polygon"
            );
        }

        // PHED cube.
        let mut coords = nd::Array2::zeros((8, 3));
        for (i, corner) in corners.iter().enumerate() {
            for k in 0..3 {
                coords[[i, k]] = corner[k];
            }
        }
        let mut mesh = UMesh::new(coords.into_shared());
        let m = usize::MAX;
        mesh.add_element(
            PHED,
            &[
                0, 3, 2, 1, m, 4, 5, 6, 7, m, 0, 1, 5, 4, m, 1, 2, 6, 5, m, 2, 3, 7, 6, m, 3, 0, 4,
                7,
            ],
            None,
        );
        let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);
        assert_eq!(sliced.num_elements_of_dim(Dimension::D2), 1);
        assert_abs_diff_eq!(slice_area(&sliced), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn slice_then_transfer_volume_field() {
        let mut mesh = make_imesh_3d(2);
        let field = FieldArcD::new(std::collections::BTreeMap::from([(
            ElementType::HEX8,
            nd::array![3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
                .into_dyn()
                .into_shared(),
        )]));
        mesh.update_field("T", field);
        let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);
        let op = TransferOperator::new(
            &mesh.view(),
            &sliced.view(),
            TransferMethod::ConstantPiecewise {
                point_location: crate::tools::transfer::PointLocation::Centroid,
            },
        );
        let field = mesh.field("T", Some(Dimension::D3)).unwrap();
        let transferred = op.apply(&field, FieldNature::Intensive, 0.0);
        let values = &transferred.0[&ElementType::PGON];
        assert_eq!(values.len(), 4);
        assert!(values.iter().all(|&value| value == 3.0));
    }

    #[test]
    fn slice_of_conforming_grid_is_conformal() {
        // The slice of a conforming volume mesh must not create duplicate nodes along shared
        // cell boundaries.
        let mesh = make_imesh_3d(2);
        let sliced = mesh.slice(&[0.0, 0.0, 0.5], &[0.0, 0.0, 1.0]);
        let report = crate::tools::is_conform(&sliced.view(), 1e-10, None).unwrap();
        assert!(report.is_conform());
    }
}
