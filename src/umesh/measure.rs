use ndarray::prelude::*;

pub fn dist(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    let diff = &a - &b;
    diff.map(|x| x.powi(2)).sum().sqrt()
}

pub fn dist2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

pub fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

pub fn surf_tri(a: ArrayView1<f64>, b: ArrayView1<f64>, c: ArrayView1<f64>) -> f64 {
    todo!()
}

pub fn surf_tri2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    // ad - bc
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

pub fn surf_tri2_signed(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2]) -> f64 {
    // ad - bc
    // positive is counter clockwise
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[1];
    let v0 = c[0] - a[0];
    let v1 = c[1] - a[1];
    0.5 * (u0 * v1 - u1 * v0)
}

pub fn surf_tri3(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> f64 {
    // 1/2 || u ^ v ||
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[0];
    let u2 = b[2] - a[2];
    let v0 = c[0] - a[0];
    let v1 = c[1] - a[0];
    let v2 = c[2] - a[2];
    0.5 * ((u0 * v1 - u1 * v0).powi(2) + (u0 * v2 - u2 * v0).powi(2) + (u1 * v2 - u2 * v1).powi(2))
        .sqrt()
}

/// Cross intersecting is not tested and result is wrong
pub fn surf_quad2(a: &[f64; 2], b: &[f64; 2], c: &[f64; 2], d: &[f64; 2]) -> f64 {
    let u0 = b[0] - a[0];
    let u1 = b[1] - a[1];
    let v0 = d[0] - a[0];
    let v1 = d[1] - a[1];
    let x0 = d[0] - c[0];
    let x1 = d[1] - c[1];
    let y0 = b[0] - c[0];
    let y1 = b[1] - c[1];
    0.5 * (u0 * v1 - u1 * v0 + x0 * y1 - x1 * y0).abs()
}

/// Cross intersecting is not tested and result is wrong
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

pub fn surf_quad3(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    // 1/2 || u ^ v ||
    todo!()
}

pub fn vol_tetra(a: ArrayView1<f64>, b: ArrayView1<f64>, c: ArrayView1<f64>) -> f64 {
    todo!()
}

pub fn vol_hexa(a: ArrayView1<f64>, b: ArrayView1<f64>, c: ArrayView1<f64>) -> f64 {
    todo!()
}
