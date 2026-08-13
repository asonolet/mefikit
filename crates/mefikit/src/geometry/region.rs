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

    #[test]
    fn aa_rectangle() {
        assert!(in_aa_rectangle(&[1.0, 1.0], &[0.0, 0.0], &[2.0, 2.0]));
        assert!(!in_aa_rectangle(&[2.0, 1.0], &[0.0, 0.0], &[2.0, 2.0]));
    }
}
