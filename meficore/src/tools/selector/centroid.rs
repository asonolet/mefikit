pub enum CentroidSelection<const N: usize> {
    InBBox { min: [f64; N], max: [f64; N] }, // Axis aligned BBox
    InSphere { center: [f64; N], r2: f64 },  // center and rayon
}
