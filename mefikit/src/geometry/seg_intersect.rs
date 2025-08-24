use arrayvec::ArrayVec;

const EPSLION_L: f64 = 1e-12;
const EPSLION_THETA: f64 = 1e-4;
const EPSILON_NN: f64 = 2.0 * EPSLION_L / EPSLION_THETA;
const EPSILON_NN2: f64 = EPSILON_NN * EPSILON_NN;

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
    seg1_nodes: [usize; 2],
    seg2_nodes: [usize; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    p3: [f64; 2],
    p4: [f64; 2],
    next_node_index: usize,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    let d1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]];

    let denom = d1[0] * d2[1] - d1[1] * d2[0];
    let epsilon = 1e-12;

    if denom.abs() < epsilon {
        return handle_colinear_or_parallel(seg1_nodes, seg2_nodes, p1, p2, p3, p4, epsilon);
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
    epsilon: f64,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    let nodes = [
        (seg1_nodes[0], p1),
        (seg1_nodes[1], p2),
        (seg2_nodes[0], p3),
        (seg2_nodes[1], p4),
    ];
    // Sort nodes along seg1
    let origin = p1;
    let dir = [p2[0] - p1[0], p2[1] - p1[1]];

    let p3_dist = compute_distance2_to_line(origin, dir, p3);
    let p4_dist = compute_distance2_to_line(origin, dir, p4);
    if p3_dist > EPSILON_NN2 && p4_dist > EPSILON_NN2 {
        // seg2 is parallel and disjoint from seg1
        return (
            [seg1_nodes[0], seg1_nodes[1]][..].try_into().unwrap(),
            [seg2_nodes[0], seg2_nodes[1]][..].try_into().unwrap(),
            None,
        );
    }

    // Detect duplicates (shared nodes)
    let mut replace_nodes: Vec<(usize, usize)> = Vec::new();
    for (idx2, pt) in nodes[2..].iter() {
        for (idx1, p) in nodes[..2].iter() {
            if nearly_equal(*p, *pt, epsilon) {
                replace_nodes.push((*idx2, *idx1));
                break;
            }
        }
    }

    // Sort nodes along seg1
    let mut sorted_nodes = nodes.clone();
    sorted_nodes.sort_by(|a, b| {
        let pa = project_onto_segment(origin, dir, a.1);
        let pb = project_onto_segment(origin, dir, b.1);
        pa.partial_cmp(&pb).unwrap()
    });

    let mut seg1_new = ArrayVec::new();
    let mut seg2_new = ArrayVec::new();

    let mut in_seg1 = false;
    let mut in_seg2 = false;
    // Iterate over sorted nodes and build new connectivities
    for (idx, _) in sorted_nodes {
        // If idx==seg1_nodes[0], start inserting into seg1_new until idx==seg1_nodes[1]
        if idx == seg1_nodes[0] {
            in_seg1 = true;
        }
        if in_seg1 {
            seg1_new.push(idx);
        }
        // If idx==seg2_nodes[0] or idx==seg2_nodes[1], start inserting into seg2_new until
        // idx==seg2_nodes[1] or idx==seg2_nodes[0] (included)
        let on_seg2 = if idx == seg2_nodes[0] || idx == seg2_nodes[1] {
            in_seg2 = !in_seg2;
            true
        } else {
            false
        };
        if on_seg2 || in_seg2 {
            // If idx is in replace_nodes, replace it with the corresponding new node
            let mut idx = idx;
            for (old, new) in &replace_nodes {
                if idx == *old {
                    idx = *new;
                    break;
                }
            }
            seg2_new.push(idx);
        }

        if idx == seg1_nodes[1] {
            in_seg1 = false;
        }
    }

    (seg1_new, seg2_new, None)
}

/// Computes the sqared distance of point p3 to the line defined by origin and dir.
fn compute_distance2_to_line(origin: [f64; 2], dir: [f64; 2], p3: [f64; 2]) -> f64 {
    let dx = dir[0];
    let dy = dir[1];
    let p1 = origin;
    let len2 = dx * dx + dy * dy;
    if len2.abs() < 1e-20 {
        return (p3[0] - p1[0]).powi(2) + (p3[1] - p1[1]).powi(2);
    }
    let t = ((p3[0] - p1[0]) * dx + (p3[1] - p1[1]) * dy) / len2;
    let proj = [p1[0] + t * dx, p1[1] + t * dy];
    (p3[0] - proj[0]).powi(2) + (p3[1] - proj[1]).powi(2)
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

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock Element with two nodes and coordinates
    fn mock_seg(coords: &[f64]) -> ([usize; 2], [usize; 2], [[f64; 2]; 4]) {
        // Replace with your actual Element constructor
        let seg1_nodes = [0, 1];
        let seg2_nodes = [2, 3];
        let p1 = [coords[0], coords[1]];
        let p2 = [coords[2], coords[3]];
        let p3 = [coords[4], coords[5]];
        let p4 = [coords[6], coords[7]];
        (seg1_nodes, seg2_nodes, [p1, p2, p3, p4])
    }

    #[test]
    fn test_classic_intersection() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 4, 1]);
        assert_eq!(&conn2[..], [0, 4, 1]);
        assert!(nearly_equal(intersection.unwrap(), [0.5, 0.5], 1e-12));
    }

    #[test]
    fn test_tangency_at_endpoint() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [1, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_intersection_colinear_overlap() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 2.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 2, 1]);
        assert_eq!(&conn2[..], [2, 1, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_intersection_colinear_included_overlap() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 4.0, 0.0, 1.0, 0.0, 3.0, 0.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 2, 3, 1]);
        assert_eq!(&conn2[..], [2, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_no_intersection_colinear() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [2, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_no_intersection_colinear_parallel() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [2, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_no_intersection() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 2.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [2, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_intersection_endpoint_merging() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.5]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [1, 3]);
        assert!(intersection.is_none());
    }

    #[test]
    fn test_intersection_colinear_endpoint_merging() {
        let (seg1, seg2, [p1, p2, p3, p4]) = mock_seg(&[0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0]);
        let (conn1, conn2, intersection) = intersect_seg_seg(seg1, seg2, p1, p2, p3, p4, 4);
        assert_eq!(&conn1[..], [0, 1]);
        assert_eq!(&conn2[..], [1, 3]);
        assert!(intersection.is_none());
    }
}
