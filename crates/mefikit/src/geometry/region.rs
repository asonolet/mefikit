//! Point-in-region tests for simple geometric regions.

use robust as ro;

/// Returns `true` if point `x` is inside a 3D sphere.
pub fn in_sphere(x: &[f64; 3], center: &[f64; 3], r: f64) -> bool {
    let x = ro::Coord3D {
        x: x[0],
        y: x[1],
        z: x[2],
    };
    let pa = ro::Coord3D {
        x: center[0] + r,
        y: center[1],
        z: center[2],
    };
    let pb = ro::Coord3D {
        x: center[0],
        y: center[1] + r,
        z: center[2],
    };
    let pd = ro::Coord3D {
        x: center[0] - r,
        y: center[1],
        z: center[2],
    };
    let pc = ro::Coord3D {
        x: center[0],
        y: center[1],
        z: center[2] + r,
    };
    ro::insphere(pa, pb, pc, pd, x) > 0.0
}

/// Returns `true` if point `x` is inside a 2D circle.
pub fn in_circle(x: &[f64; 2], center: &[f64; 2], r: f64) -> bool {
    let x = ro::Coord { x: x[0], y: x[1] };
    let pa = ro::Coord {
        x: center[0] + r,
        y: center[1],
    };
    let pb = ro::Coord {
        x: center[0],
        y: center[1] + r,
    };
    let pc = ro::Coord {
        x: center[0] - r,
        y: center[1],
    };
    ro::incircle(pa, pb, pc, x) > 0.0
}

/// Returns `true` if point `x` is inside an axis-aligned 3D bounding box.
///
/// The box is defined by corner points `p0` (min) and `p1` (max).
///
/// The comparison is exact; points on the faces are classified with the half-open convention
/// `[p0, p1)`. If a tolerance is needed, the caller must extend the box bounds themselves.
pub fn in_aa_bbox(x: &[f64; 3], p0: &[f64; 3], p1: &[f64; 3]) -> bool {
    !((x[0] < p0[0])
        || (x[0] >= p1[0])
        || (x[1] < p0[1])
        || (x[1] >= p1[1])
        || (x[2] < p0[2])
        || (x[2] >= p1[2]))
}

/// Returns `true` if point `x` is inside an axis-aligned 2D rectangle.
///
/// The rectangle is defined by corner points `p0` (min) and `p1` (max).
///
/// The comparison is exact; points on the edges are classified with the half-open convention
/// `[p0, p1)`. If a tolerance is needed, the caller must extend the rectangle bounds themselves.
pub fn in_aa_rectangle(x: &[f64; 2], p0: &[f64; 2], p1: &[f64; 2]) -> bool {
    !((x[0] < p0[0]) || (x[0] >= p1[0]) || (x[1] < p0[1]) || (x[1] >= p1[1]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere() {
        assert!(in_sphere(&[0.5, 0.5, 0.5], &[0.0, 0.0, 0.0], 1.0));
        assert!(!in_sphere(&[1.5, 0.0, 0.0], &[0.0, 0.0, 0.0], 1.0));
    }

    #[test]
    fn circle() {
        assert!(in_circle(&[0.5, 0.0], &[0.0, 0.0], 1.0));
        assert!(!in_circle(&[1.5, 0.0], &[0.0, 0.0], 1.0));
    }

    #[test]
    fn aa_bbox() {
        assert!(in_aa_bbox(
            &[1.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[2.0, 2.0, 2.0]
        ));
        assert!(!in_aa_bbox(
            &[2.0, 1.0, 1.0],
            &[0.0, 0.0, 0.0],
            &[2.0, 2.0, 2.0]
        ));
    }

    /// Adjacent hexahedra tangent to a face of a bounding box must all be classified
    /// consistently by the exact bbox test when their centroid is the vertex centroid.
    ///
    /// The vertex centroid of a regular hexa is exactly the arithmetic mean of its nodes, so a
    /// cell symmetric about a box face has a centroid lying exactly on that face; the half-open
    /// convention `[p0, p1)` then classifies every such tangent cell the same way. No epsilon is
    /// baked into [`in_aa_bbox`]: a user wanting a tolerance extends the box bounds themselves.
    #[test]
    fn aa_bbox_tangent_hexa_vertex_centroids() {
        use crate::geometry::Polyhedron;
        let hexa = |a: f64, b: f64| {
            Polyhedron::unknown(
                [
                    [a, 0.0, 0.0],
                    [b, 0.0, 0.0],
                    [b, 1.0, 0.0],
                    [a, 1.0, 0.0],
                    [a, 0.0, 1.0],
                    [b, 0.0, 1.0],
                    [b, 1.0, 1.0],
                    [a, 1.0, 1.0],
                ],
                vec![
                    vec![0, 1, 2, 3],
                    vec![0, 3, 7, 4],
                    vec![0, 4, 5, 1],
                    vec![1, 5, 6, 2],
                    vec![2, 6, 7, 3],
                    vec![4, 7, 6, 5],
                ],
            )
        };
        let p0 = [0.0, 0.0, 0.0];
        let p1 = [1.0, 1.0, 1.0];

        // Tangent cells: their vertex centroid lies exactly on the max-x face (x == p1[0]).
        // All must be consistently classified by the exact half-open bbox test.
        let tangent = [
            (0.1, 1.9),
            (0.2, 1.8),
            (0.3, 1.7),
            (0.4, 1.6),
            (0.6, 1.4),
            (0.7, 1.3),
            (0.8, 1.2),
            (0.9, 1.1),
        ];
        let mut tangent_results: Vec<bool> = Vec::new();
        for (a, b) in tangent {
            let c = hexa(a, b).centroid();
            assert_eq!(
                c[0],
                (a + b) / 2.0,
                "vertex centroid must be exact for regular hexa [{a}, {b}]"
            );
            assert_eq!(
                c[0], p1[0],
                "vertex centroid of tangent hexa [{a}, {b}] must lie exactly on the max-x face"
            );
            tangent_results.push(in_aa_bbox(&c, &p0, &p1));
        }
        assert!(
            tangent_results.iter().all(|&r| r == tangent_results[0]),
            "tangent hexas are classified inconsistently: {tangent_results:?}"
        );

        // Controls: a clearly inside cell and a clearly outside cell.
        assert!(in_aa_bbox(&hexa(0.0, 0.5).centroid(), &p0, &p1));
        assert!(!in_aa_bbox(&hexa(1.05, 1.55).centroid(), &p0, &p1));
    }

    #[test]
    fn aa_rectangle() {
        assert!(in_aa_rectangle(&[1.0, 1.0], &[0.0, 0.0], &[2.0, 2.0]));
        assert!(!in_aa_rectangle(&[2.0, 1.0], &[0.0, 0.0], &[2.0, 2.0]));
    }
}
