use ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;
use std::collections::HashSet;


use crate::mesh_element::{ElementType, ElementView};
use crate::element_block_like::ElementBlockLike;


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

pub enum Connectivity {
    Regular(Array2<usize>),
    Poly {
        data: Array1<usize>,
        offsets: Array1<usize>,
    },
}

impl Connectivity {
    pub fn len(&self) -> usize {
        match self {
            Connectivity::Regular(conn) => conn.nrows(),
            Connectivity::Poly {offsets, ..} => offsets.len() - 1,
        }
    }

    pub fn element_connectivity(&self, index: usize) -> ArrayView1<'_, usize> {
        match self {
            Connectivity::Regular(conn) => conn.row(index),
            Connectivity::Poly { data, offsets } => {
                let start = offsets[index];
                let end = offsets[index + 1];
                data.slice(s![start..end])
            }
        }
    }
}

pub struct ElementBlock {
    cell_type: ElementType,
    connectivity: Connectivity,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
}

impl ElementBlock {
    fn len(&self) -> usize {
        self.connectivity.len()
    }
    fn params(&self) -> &HashMap<String, f64> {
        &self.params
    }
    fn fields(&self) -> &HashMap<String, ArrayD<f64>> {
        &self.fields
    }
    fn families(&self) -> &Array1<usize> {
        &self.families
    }
    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        &self.groups
    }
    fn compo_type(&self) -> ElementType {
        self.cell_type.into()
    }
    fn element_connectivity<'a>(&'a self, index: usize) -> ArrayView1<'a, usize> {
        self.connectivity.element_connectivity(index)
    }
    fn iter<'a>(&'a self, coords: &'a Array2<f64>) -> Box<dyn Iterator<Item = Element<'a>> + 'a> {
        Box::new((0..self.len()).map(move |i| {
            let connectivity = self.element_connectivity(i);
            let family = &self.families()[i];
            let fields = self.fields().iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();
            Element {
                global_index: i,
                coords,
                fields,
                family,
                groups: self.groups(),
                connectivity,
                compo_type: self.compo_type(),
            }
        }))
    }
    fn par_iter<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> impl ParallelIterator<Item = Element<'a>> + 'a {
        (0..self.len()).into_par_iter().map(move |i| {
            let connectivity = self.element_connectivity(i);
            let fields = self.fields()
                .iter()
                .map(|(k, v)| (k.as_str(), v.index_axis(Axis(0), i)))
                .collect();

            Element {
                global_index: i,
                coords,
                fields,
                family: &self.families()[i],
                groups: self.groups(),
                connectivity,
                compo_type: self.compo_type(),
            }
        })
    }
}

pub trait IntoElementBlockEntry {
    fn into_entry(self) -> (ElementType, ElementBlock);
}

impl IntoElementBlockEntry for RegularCells {
    fn into_entry(self) -> (ElementType, ElementBlock) {
        (self.cell_type.into(), ElementBlock::RegularCells(self))
    }
}

impl IntoElementBlockEntry for PolyCells {
    fn into_entry(self) -> (ElementType, ElementBlock) {
        (self.cell_type.into(), ElementBlock::PolyCells(self))
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


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1, Array2};
    use std::collections::{HashMap, HashSet};

    fn dummy_regular_cells() -> RegularCells {
        let connectivity = array![[0, 1, 2], [2, 3, 0]];
        let families = Array1::from(vec![0, 1]);

        let mut groups = HashMap::new();
        groups.insert("groupA".into(), HashSet::from([0]));
        groups.insert("groupB".into(), HashSet::from([1]));

        RegularCells {
            cell_type: ElementType::TRI3,
            connectivity,
            params: HashMap::new(),
            fields: HashMap::new(),
            families,
            groups,
        }
    }

    #[test]
    fn test_regular_cells_len() {
        let rc = dummy_regular_cells();
        assert_eq!(rc.len(), 2);
    }

    #[test]
    fn test_element_block_variant() {
        let rc = dummy_regular_cells();
        let eb = ElementBlock::RegularCells(rc);
        if let ElementBlock::RegularCells(inner) = eb {
            assert_eq!(inner.connectivity.nrows(), 2);
        } else {
            panic!("Expected RegularCells variant");
        }
    }
}
