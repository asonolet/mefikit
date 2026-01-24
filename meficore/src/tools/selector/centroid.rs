pub enum CentroidSelection {
    InBBox { min: [f64; 3], max: [f64; 3] }, // Axis aligned BBox
    InRect { min: [f64; 2], max: [f64; 2] }, // Axis aligned BBox
    InSphere { center: [f64; 3], r2: f64 },  // center and rayon
    InCircle { center: [f64; 2], r2: f64 },  // center and rayon
}
