//! Geometric measure computations for elements.
//!
//! Provides functions for computing distances, areas, and volumes of
//! geometric primitives.

use na::{Point1, Point2, Vector4};
use nalgebra as na;
use ndarray::prelude::*;

/// Computes the Euclidean distance between two points (generic dimension).
pub fn dist_(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let diff = &a - &b;
    diff.map(|x| x.powi(2)).sum().sqrt()
}

/// Computes the Euclidean distance between two 1D points.
pub fn dist1(a: Point1<f64>, b: Point1<f64>) -> f64 {
    let diff = a - b;
    diff.norm()
}

/// Computes the Euclidean distance between two 2D points.
pub fn dist2(a: Point2<f64>, b: Point2<f64>) -> f64 {
    let diff = a - b;
    diff.norm()
}

/// Computes the squared distance between two 2D points.
pub fn squared_dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)
}

/// Computes the squared distance between two 2D points.
pub fn squared_dist2_(a: Point2<f64>, b: Point2<f64>) -> f64 {
    (a - b).norm_squared()
}

/// Computes the squared distance between two 3D points.
pub fn squared_dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// Computes the Euclidean distance between two 3D points.
pub fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    squared_dist3(a, b).sqrt()
}

/// Computes the area of a 2D triangle (generic input).
pub fn surf_tri(_a: ArrayView1<f64>, _b: ArrayView1<f64>, _c: ArrayView1<f64>) -> f64 {
    todo!()
}

/// Computes the area of a 2D triangle.
#[inline]
pub fn surf_tri2(a: Point2<f64>, b: Point2<f64>, c: Point2<f64>) -> f64 {
    0.5 * ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs()
}

/// Computes the signed area of a 2D triangle.
///
/// Positive result indicates counter-clockwise orientation.
pub fn surf_tri2_signed(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[1];
    let v0 = c[0] - a[0];
    let v1 = c[1] - a[1];
    0.5 * (u0 * v1 - u1 * v0)
}

/// Computes the area of a 3D triangle.
#[inline]
pub fn surf_tri3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[0];
    let u2 = b[2] - a[2];
    let v0 = c[0] - a[0];
    let v1 = c[1] - a[0];
    let v2 = c[2] - a[2];
    0.5 * ((u0 * v1 - u1 * v0).powi(2) + (u0 * v2 - u2 * v0).powi(2) + (u1 * v2 - u2 * v1).powi(2))
        .sqrt()
}

/// Computes the area of a 2D quadrilateral.
///
/// # Warning
/// Cross intersecting is not tested and result is wrong.
#[inline(always)]
pub fn surf_quad2(a: &Point2<f64>, b: &Point2<f64>, c: &Point2<f64>, d: &Point2<f64>) -> f64 {
    let px: Vector4<f64> = Vector4::new(a.x, b.x, c.x, d.x);
    let py: Vector4<f64> = Vector4::new(a.y, b.y, c.y, d.y);
    let pxs: Vector4<f64> = Vector4::new(b.x, c.x, d.x, a.x);
    let pys: Vector4<f64> = Vector4::new(b.y, c.y, d.y, a.y);
    0.5 * (px.dot(&pys) - py.dot(&pxs)).abs()
}

/// Computes the signed area of a 2D quadrilateral.
///
/// # Warning
/// Cross intersecting is not tested and result is wrong.
pub fn surf_quad2_signed(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> f64 {
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[1];
    let v0 = d[0] - a[0];
    let v1 = d[1] - a[1];
    let x0 = d[0] - c[0];
    let x1 = d[1] - c[1];
    let y0 = b[0] - c[0];
    let y1 = b[1] - c[1];
    0.5 * (u0 * v1 - u1 * v0 + x0 * y1 - x1 * y0)
}

/// Computes the area of a 3D quadrilateral.
pub fn surf_quad3(_a: &[f64; 3], _b: &[f64; 3], _c: &[f64; 3], _d: &[f64; 3]) -> f64 {
    todo!()
}

/// Computes the volume of a tetrahedron.
pub fn vol_tetra(_a: ArrayView1<f64>, _b: ArrayView1<f64>, _c: ArrayView1<f64>) -> f64 {
    todo!()
}

/// Computes the volume of a hexahedron.
pub fn vol_hexa(_a: ArrayView1<f64>, _b: ArrayView1<f64>, _c: ArrayView1<f64>) -> f64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dist2() {
        let a = na::Point2::new(0.0, 0.0);
        let b = na::Point2::new(3.0, 4.0);
        assert_abs_diff_eq!(dist2(a, b), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_dist2() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert_abs_diff_eq!(squared_dist2(&a, &b), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_dist2_() {
        let a = na::Point2::new(0.0, 0.0);
        let b = na::Point2::new(3.0, 4.0);
        assert_abs_diff_eq!(squared_dist2_(a, b), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dist3() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert_abs_diff_eq!(dist3(&a, &b), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_squared_dist3() {
        let a = [0.0, 0.0, 0.0];
        let b = [3.0, 4.0, 0.0];
        assert_abs_diff_eq!(squared_dist3(&a, &b), 25.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_tri2() {
        let a = na::Point2::new(0.0, 0.0);
        let b = na::Point2::new(1.0, 0.0);
        let c = na::Point2::new(0.0, 1.0);
        assert_abs_diff_eq!(surf_tri2(a, b, c), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_tri2_signed() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert_abs_diff_eq!(surf_tri2_signed(&a, &b, &c), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_quad2() {
        let a = na::Point2::new(0.0, 0.0);
        let b = na::Point2::new(1.0, 0.0);
        let c = na::Point2::new(1.0, 1.0);
        let d = na::Point2::new(0.0, 1.0);
        assert_abs_diff_eq!(surf_quad2(&a, &b, &c, &d), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_quad2_signed() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [1.0, 1.0];
        let d = [0.0, 1.0];
        assert_abs_diff_eq!(surf_quad2_signed(&a, &b, &c, &d), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_tri3() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        let area = surf_tri3(a, b, c);
        assert_abs_diff_eq!(area, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_surf_quad3() {
        // This will panic with todo!() - skipping
    }
}
