use ndarray::{ArrayView1, ArrayView2, ArrayViewD};
use std::collections::HashMap;
use std::collections::HashSet;


pub struct MeshElementView<'a> {
    pub coords: ArrayView2<'a, f64>,
    pub fields: HashMap<String, ArrayViewD<'a, f64>>,
    pub family: usize,
    pub groups: HashSet<String>,
    pub connectivity: ArrayView1<'a, usize>,
    pub compo_type: ElementType,
}

#[derive(Debug, Eq, Hash, Copy, Clone, PartialEq)]
pub enum ElementType {
    // 0d
    VERTEX,

    // 1d
    SEG2,
    SEG3,
    SEG4,
    SPLINE,

    // 2d
    TRI3,
    TRI6,
    TRI7,
    QUAD4,
    QUAD8,
    QUAD9,
    PGON,

    // 3d
    TET4,
    TET10,
    HEX8,
    HEX21,
    PHED,
}