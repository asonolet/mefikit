use ndarray::{ArrayView1, ArrayView2, ArrayViewD, ArrayViewMut1, ArrayViewMutD};
use std::collections::HashMap;
use std::collections::HashSet;


pub struct Element<'a> {
    pub global_index: usize,
    pub coords: &'a Array2<f64>,
    pub fields: HashMap<&'a str, ArrayViewD<'a, f64>>,
    pub family: &'a usize,
    pub groups: &'a HashMap<String, HashSet<String>>,
    pub connectivity: ArrayView1<'a, usize>,
    pub compo_type: ElementType,
}

pub struct ElementMut<'a> {
    pub global_index: usize,
    pub coords: &'a Array2<f64>,
    pub connectivity: ArrayViewMut1<'a, usize>,
    pub family: &'a mut usize,
    pub fields: HashMap<&'a str, ArrayViewMutD<'a, f64>>,
    pub groups: &'a HashMap<String, HashSet<usize>>, // safely shared across threads
    pub element_type: ElementType,
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


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayView1, ArrayViewD};

    #[test]
    fn test_element_struct_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = array![0, 1, 2];
        let fields = HashMap::new();
        let groups = HashMap::new();
        let family = 0;

        let element = Element {
            global_index: 0,
            coords: &coords,
            fields,
            family: &family,
            groups: &groups,
            connectivity: conn.view(),
            compo_type: ElementType::TRI3,
        };

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.global_index, 0);
    }
}
