use ndarray::{Array2, ArrayView1, ArrayViewD, ArrayViewMut1, ArrayViewMutD};
use std::collections::HashMap;
use std::collections::HashSet;

#[derive(Copy, Clone)]
pub enum EdgeType {
    SEG2,
    SEG3,
    SEG4,
    SPLINES,
}

#[derive(Copy, Clone)]
pub enum FaceType {
    TRI3,
    TRI6,
    TRI7,
    QUAD4,
    QUAD8,
    QUAD9,
    PGON,
}

#[derive(Copy, Clone)]
pub enum VolumeType {
    TET4,
    TET10,
    HEX8,
    HEX21,
    PHDRON,
}

#[derive(Copy, Clone)]
pub enum PolyCellType {
    SPLINE,
    PGON,
    PHED,
}

#[derive(Copy, Clone)]
pub enum RegularCellType {
    VERTEX,
    SEG2,
    SEG3,
    SEG4,
    TRI3,
    TRI6,
    TRI7,
    QUAD4,
    QUAD8,
    QUAD9,
    TET4,
    TET10,
    HEX8,
    HEX21,
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


impl From<PolyCellType> for ElementType {
    fn from(cell: PolyCellType) -> Self {
        match cell {
            PolyCellType::SPLINE => ElementType::SPLINE,
            PolyCellType::PGON => ElementType::PGON,
            PolyCellType::PHED => ElementType::PHED,
        }
    }
}

impl From<RegularCellType> for ElementType {
    fn from(cell: RegularCellType) -> Self {
        match cell {
            RegularCellType::VERTEX => ElementType::VERTEX,
            RegularCellType::SEG2 => ElementType::SEG2,
            RegularCellType::SEG3 => ElementType::SEG3,
            RegularCellType::SEG4 => ElementType::SEG4,
            RegularCellType::TRI3 => ElementType::TRI3,
            RegularCellType::TRI6 => ElementType::TRI6,
            RegularCellType::TRI7 => ElementType::TRI7,
            RegularCellType::QUAD4 => ElementType::QUAD4,
            RegularCellType::QUAD8 => ElementType::QUAD8,
            RegularCellType::QUAD9 => ElementType::QUAD9,
            RegularCellType::TET4 => ElementType::TET4,
            RegularCellType::TET10 => ElementType::TET10,
            RegularCellType::HEX8 => ElementType::HEX8,
            RegularCellType::HEX21 => ElementType::HEX21,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TopologicalDimension {
    D0,
    D1,
    D2,
    D3,
}


impl ElementType {
    pub fn topological_dimension(&self) -> TopologicalDimension {
        use ElementType::*;
        match self {
            // 0D
            VERTEX => TopologicalDimension::D0,

            // 1D
            SEG2 | SEG3 | SEG4 | SPLINE => TopologicalDimension::D1,

            // 2D
            TRI3 | TRI6 | TRI7 | QUAD4 | QUAD8 | QUAD9 | PGON => TopologicalDimension::D2,

            // 3D
            TET4 | TET10 | HEX8 | HEX21 | PHED => TopologicalDimension::D3,
        }
    }
}

pub struct Element<'a> {
    pub global_index: usize,
    pub coords: &'a Array2<f64>,
    pub fields: HashMap<&'a str, ArrayViewD<'a, f64>>,
    pub family: &'a usize,
    pub groups: &'a HashMap<String, HashSet<usize>>,
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


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
