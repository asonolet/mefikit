use super::umesh::Dimension::*;
use super::umesh::{Element, ElementId, ElementLike};
use super::umesh::{UMesh, UMeshView};

use std::collections::HashMap;

use arrayvec::ArrayVec;
use ndarray::prelude::*;
use rstar::primitives::{GeomWithData, Line};
use rstar::{AABB, RTree};

const EPSLION_L: f64 = 1e-12;
const EPSLION_THETA: f64 = 1e-4;
const EPSILON_NN: f64 = 2.0 * EPSLION_L / EPSLION_THETA;
const EPSILON_NN2: f64 = EPSILON_NN * EPSILON_NN;

/// A wrapper struct representing a geometric line segment with associated element ID data.
///
/// The segment is defined by a `Line<[f64; 2]>` geometry and an `ElementId` for identification.
struct Segment(GeomWithData<Line<[f64; 2]>, ElementId>);

impl Segment {
    pub fn new(p1: [f64; 2], p2: [f64; 2], eid: ElementId) -> Self {
        Self(GeomWithData::new(Line { from: p1, to: p2 }, eid))
    }
    fn from(el: &Element) -> Self {
        let p1: &[f64] = el.coords().index_axis(Axis(0), 0).to_slice().unwrap();
        let p2: &[f64] = el.coords().index_axis(Axis(0), 1).to_slice().unwrap();
        Self::new([p1[0], p1[1]], [p2[0], p2[1]], el.id())
    }
}

fn to_aabb(el: &Element) -> AABB<[f64; 2]> {
    AABB::from_points(
        el.coords()
            .axis_iter(Axis(0))
            .map(|e| e.to_slice().unwrap()[..2].try_into().unwrap()),
    )
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct DirectedEdge(usize, usize);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct UndirectedEdge(usize, usize);

impl UndirectedEdge {
    fn new(a: usize, b: usize) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

impl From<DirectedEdge> for UndirectedEdge {
    fn from(e: DirectedEdge) -> Self {
        Self::new(e.0, e.1)
    }
}

// impl From<UndirectedEdge> for DirectedEdge {
//     fn from(e: UndirectedEdge) -> Self {
//         Self(e.0, e.1)
//     }
// }

fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    let cross = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])).abs();
    if cross > eps {
        return false;
    }
    let dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1]);
    if dot < -eps {
        return false;
    }
    let len_sq = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
    if dot > len_sq + eps {
        return false;
    }
    true
}

/// Computes the intersection between two line segments, applying fusion rules to avoid degenerate elements.
///
/// # Fusion Rules
/// - Segment 1 (`seg1`) is the reference; its nodes are preferred for fusion.
/// - Nodes are merged if they are closer than `2 * EPSILON_NN`.
/// - Tangent nodes are detected using orthogonal projection and `EPSILON_L`.
/// - Handles colinear, tangent, classic intersection, and no intersection cases.
///
/// # Arguments
/// * `seg1` - The first segment (reference).
/// * `seg2` - The second segment.
///
/// # Returns
/// * ArrayVec<usize, 4> - New connectivity for `seg1`.
/// * ArrayVec<usize, 4> - New connectivity for `seg2`.
/// * Option<([f64; 2], usize)> - Intersection point and its index if a new node is created.
///
/// # Note
/// The function assumes that both segments are of type SEG2 (2-node segments).
pub fn intersect_seg_seg(
    seg1: &Element,
    seg2: &Element,
    next_node_index: usize,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    let seg1_nodes = [seg1.connectivity()[0], seg1.connectivity()[1]];
    let seg2_nodes = [seg2.connectivity()[0], seg2.connectivity()[1]];
    let p1 = [seg1.coords()[[0, 0]], seg1.coords()[[0, 1]]];
    let p2 = [seg1.coords()[[1, 0]], seg1.coords()[[1, 1]]];
    let p3 = [seg2.coords()[[0, 0]], seg2.coords()[[0, 1]]];
    let p4 = [seg2.coords()[[1, 0]], seg2.coords()[[1, 1]]];

    let d1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]];

    let denom = d1[0] * d2[1] - d1[1] * d2[0];
    let epsilon = 1e-12;

    if denom.abs() < epsilon {
        return handle_colinear_or_parallel(
            seg1_nodes, seg2_nodes, p1, p2, p3, p4, d1, d2, epsilon,
        );
    }

    let (t, u, intersection) = compute_parametric_intersection(p1, d1, p3, d2, denom);

    if is_intersection_within_segments(t, u, epsilon) {
        if is_existing_node(&intersection, &[p1, p2, p3, p4], epsilon) {
            return (
                [seg1_nodes[0], seg1_nodes[1]][..].try_into().unwrap(),
                [seg2_nodes[0], seg2_nodes[1]][..].try_into().unwrap(),
                None,
            );
        }
        let seg1_new = insert_intersection_node(seg1_nodes, t, next_node_index);
        let seg2_new = insert_intersection_node(seg2_nodes, u, next_node_index);
        return (seg1_new, seg2_new, Some(intersection));
    }

    (
        [seg1_nodes[0], seg1_nodes[1]][..].try_into().unwrap(),
        [seg2_nodes[0], seg2_nodes[1]][..].try_into().unwrap(),
        None,
    )
}

/// Checks if two points are nearly equal within a given epsilon.
fn nearly_equal(a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    // TODO: use sqared distance comparison instead
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps
}

/// Projects a point onto a segment defined by origin and direction.
/// Returns the parametric coordinate along the segment.
fn project_onto_segment(origin: [f64; 2], dir: [f64; 2], pt: [f64; 2]) -> f64 {
    let dx = pt[0] - origin[0];
    let dy = pt[1] - origin[1];
    let norm = dir[0] * dir[0] + dir[1] * dir[1];
    if norm.abs() < 1e-20 {
        return 0.0;
    }
    (dx * dir[0] + dy * dir[1]) / norm
}

/// Handles the case where segments are colinear or parallel.
/// Returns the new connectivities for each segment, with no intersection node.
fn handle_colinear_or_parallel(
    seg1_nodes: [usize; 2],
    seg2_nodes: [usize; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    p3: [f64; 2],
    p4: [f64; 2],
    d1: [f64; 2],
    d2: [f64; 2],
    epsilon: f64,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    let nodes = [
        (seg1_nodes[0], p1),
        (seg1_nodes[1], p2),
        (seg2_nodes[0], p3),
        (seg2_nodes[1], p4),
    ];

    // Remove duplicates (shared nodes)
    let mut unique_nodes: Vec<(usize, [f64; 2])> = Vec::new();
    for (idx, pt) in nodes {
        if !unique_nodes
            .iter()
            .any(|(_, p)| nearly_equal(*p, pt, epsilon))
        {
            unique_nodes.push((idx, pt));
        }
    }

    // Sort nodes along seg1
    let origin = p1;
    let dir = d1;
    unique_nodes.sort_by(|a, b| {
        let pa = project_onto_segment(origin, dir, a.1);
        let pb = project_onto_segment(origin, dir, b.1);
        pa.partial_cmp(&pb).unwrap()
    });

    let mut seg1_new = ArrayVec::new();
    let mut seg2_new = ArrayVec::new();
    for &(idx, pt) in &unique_nodes {
        if nearly_equal(pt, p1, epsilon) || nearly_equal(pt, p2, epsilon) {
            seg1_new.push(idx);
        }
        if nearly_equal(pt, p3, epsilon) || nearly_equal(pt, p4, epsilon) {
            seg2_new.push(idx);
        }
    }
    // Insert interior nodes from the other segment if they are between endpoints
    for &(idx, pt) in &unique_nodes {
        // For seg1: if node belongs to seg2 and is between seg1 endpoints
        if (idx == seg2_nodes[0] || idx == seg2_nodes[1])
            && !nearly_equal(pt, p1, epsilon)
            && !nearly_equal(pt, p2, epsilon)
        {
            let t = project_onto_segment(origin, dir, pt);
            let mut inserted = false;
            for i in 0..seg1_new.len() - 1 {
                let t0 = project_onto_segment(
                    origin,
                    dir,
                    if seg1_new[i] == seg1_nodes[0] { p1 } else { p2 },
                );
                let t1 = project_onto_segment(
                    origin,
                    dir,
                    if seg1_new[i + 1] == seg1_nodes[0] {
                        p1
                    } else {
                        p2
                    },
                );
                if t > t0 && t < t1 {
                    seg1_new.insert(i + 1, idx);
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                seg1_new.push(idx);
            }
        }
        // For seg2: if node belongs to seg1 and is between seg2 endpoints
        if (idx == seg1_nodes[0] || idx == seg1_nodes[1])
            && !nearly_equal(pt, p3, epsilon)
            && !nearly_equal(pt, p4, epsilon)
        {
            let t = project_onto_segment(p3, d2, pt);
            let mut inserted = false;
            for i in 0..seg2_new.len() - 1 {
                let t0 = project_onto_segment(
                    p3,
                    d2,
                    if seg2_new[i] == seg2_nodes[0] { p3 } else { p4 },
                );
                let t1 = project_onto_segment(
                    p3,
                    d2,
                    if seg2_new[i + 1] == seg2_nodes[0] {
                        p3
                    } else {
                        p4
                    },
                );
                if t > t0 && t < t1 {
                    seg2_new.insert(i + 1, idx);
                    inserted = true;
                    break;
                }
            }
            if !inserted {
                seg2_new.push(idx);
            }
        }
    }

    (seg1_new, seg2_new, None)
}

/// Computes the parametric intersection values t and u, and intersection point.
fn compute_parametric_intersection(
    p1: [f64; 2],
    d1: [f64; 2],
    p3: [f64; 2],
    d2: [f64; 2],
    denom: f64,
) -> (f64, f64, [f64; 2]) {
    let dx = p3[0] - p1[0];
    let dy = p3[1] - p1[1];
    let t = (dx * d2[1] - dy * d2[0]) / denom;
    let u = (dx * d1[1] - dy * d1[0]) / denom;
    let intersection = [p1[0] + t * d1[0], p1[1] + t * d1[1]];
    (t, u, intersection)
}

/// Checks if the intersection is within the bounds of both segments.
fn is_intersection_within_segments(t: f64, u: f64, epsilon: f64) -> bool {
    t >= -epsilon && t <= 1.0 + epsilon && u >= -epsilon && u <= 1.0 + epsilon
}

/// Checks if the intersection point coincides with any of the segment endpoints.
fn is_existing_node(intersection: &[f64; 2], endpoints: &[[f64; 2]; 4], epsilon: f64) -> bool {
    endpoints
        .iter()
        .any(|pt| nearly_equal(*intersection, *pt, epsilon))
}

/// Inserts the intersection node into the connectivity in the correct order.
fn insert_intersection_node(
    nodes: [usize; 2],
    param: f64,
    intersection_idx: usize,
) -> ArrayVec<usize, 4> {
    let t0 = 0.0;
    let t1 = 1.0;
    if param < t0 {
        [intersection_idx, nodes[0], nodes[1]][..]
            .try_into()
            .unwrap()
    } else if param > t1 {
        [nodes[0], nodes[1], intersection_idx][..]
            .try_into()
            .unwrap()
    } else {
        [nodes[0], intersection_idx, nodes[1]][..]
            .try_into()
            .unwrap()
    }
}

/// In the general case, there could be multiple intersections beween 1d elements (ex SEG2 and
/// SEG3).
///
/// The implementation should be completly symmetric for it to be correct.
fn intersect_1d_elems(
    e1: &Element,
    e2: &Element,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    todo!()
}

/// Cette méthode permet de découper un maillage 2d potentiellement non conforme avec un maillage
/// de segments propres (sans noeuds non fusionnés).
pub fn cut_2d_mesh_with_1d_mesh(mesh: UMeshView, tool_mesh: UMesh) -> Result<UMesh, String> {
    // tool_mesh.merge_nodes();
    let tool_mesh = tool_mesh.prepend_coords(mesh.coords());

    let (m1d, mgraph) = mesh.compute_submesh(Some(D2), None);

    // build R*-tree with 1d tool mesh
    let segs: Vec<_> = tool_mesh
        .elements()
        .map(|seg| Segment::from(&seg).0)
        .collect();
    let cutter_tree = RTree::bulk_load(segs);

    // Maintenant je travaille uniquement en terme de connectivité, j'utilise la table de
    // coordonnées suivante pour les identifiants de noeuds:
    // Table de mesh + table de tool_mesh après merge + nouveaux noeuds différents
    // Je créé la liste des segments de tool mesh découpés (avec les noeuds d'intersection)
    let mut e2int: HashMap<ElementId, Vec<[usize; 2]>> = HashMap::new();
    for el in mesh.elements_of_dim(D2) {
        let segs_in_elem: Vec<_> = cutter_tree
            .locate_in_envelope_intersecting(&to_aabb(&el))
            .collect();
        for (_, _, &eid) in mgraph.edges(el.id()) {
            // TODO: edge could be SEG3!
            let edge = m1d.get_element(eid);
            let intersections: &_ = e2int.entry(eid).or_insert_with(|| {
                let mut intersections = Vec::new();
                for seg in &segs_in_elem {
                    let seg_elem = tool_mesh.get_element(seg.data);
                    // Calcul des intersections avec edge
                    // Une intersection est soit un Point, soit un Segment
                    let intersection_coords = intersect_1d_elems(&edge, &seg_elem);
                    todo!()
                }
                intersections
            });
        }
    }
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock Element with two nodes and coordinates
    fn mock_seg(coords: [[f64; 2]; 2]) -> UMesh {
        // Replace with your actual Element constructor
        let mut umesh = UMesh::new(
            ArcArray::from_shape_vec(
                [2, 2],
                vec![coords[0][0], coords[0][1], coords[1][0], coords[1][1]],
            )
            .unwrap(),
        );
        umesh.add_element(crate::ElementType::SEG2, &[0, 1], None, None);
        umesh
    }

    #[test]
    fn test_classic_intersection() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 1.0]]);
        let seg2 = mock_seg([[0.0, 1.0], [1.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 4, 1]);
        assert_eq!(&conn2[..], [0, 4, 1]);
        assert!(nearly_equal(intersection.unwrap(), [0.5, 0.5], 1e-12));
    }

    #[test]
    fn test_tangency_at_endpoint() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [1.0, 1.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [1, 1]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_colinear_overlap() {
        let seg1 = mock_seg([[0.0, 0.0], [2.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [3.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 0, 1]);
        assert_eq!(&conn2[..], [0, 1, 1]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_colinear_included_overlap() {
        let seg1 = mock_seg([[0.0, 0.0], [4.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [3.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 0, 1, 1]);
        assert_eq!(&conn2[..], [0, 1]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_no_intersection() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[2.0, 1.0], [2.0, 2.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [0, 1]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_endpoint_merging() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [2.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let (conn1, conn2, intersection) =
            intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0), 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [1, 1]);
        assert!(intersection.is_none());
    }
}
