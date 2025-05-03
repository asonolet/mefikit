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

pub struct RegularCells {
    cell_type: RegularCellType,
    connectivity: Array2<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
}

pub struct PolyCells {
    cell_type: PolyCellType,
    connectivity: Array1<usize>,
    offsets: Array1<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
}

pub enum ElementBlock {
    RegularCells(RegularCells),
    PolyCells(PolyCells),
}


impl RegularCells {
    pub fn new(
        cell_type: RegularCellType,
        connectivity: Array2<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, HashSet<usize>>,
    ) -> Self {
        Self {
            cell_type,
            connectivity,
            params,
            fields,
            families,
            groups,
        }
    }
}

impl PolyCells {
    pub fn new(
        cell_type: PolyCellType,
        connectivity: Array1<usize>,
        offsets: Array1<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, HashSet<usize>>,
    ) -> Self {
        Self {
            cell_type,
            connectivity,
            offsets,
            params,
            fields,
            families,
            groups,
        }
    }
}

impl ElementBlockLike for RegularCells {
    fn len(&self) -> usize {
        self.families.len()
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
    fn iter_elements<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> Box<dyn Iterator<Item = ElementView<'a>> + 'a> {
        let iter = (0..self.len()).map(move |i| {
            // 1. Extract element fields
            let fields = self.element_fields(i);

            // 2. 3. Extract and copy family and groups
            let (family, groups) = self.element_family_and_groups(i);

            // 4. Extract connectivity
            let connectivity = self.connectivity.row(i);

            ElementView {
                fields,
                family,
                groups,
                connectivity,
                compo_type: self.cell_type.into(),
                coords: coords.view(), // pass full view
            }
        });

        Box::new(iter)
    }
    fn compo_type(&self) -> ElementType {
        self.cell_type.into()
    }
}

impl ElementBlockLike for PolyCells {
    fn len(&self) -> usize {
        self.families.len()
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
    fn iter_elements<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> Box<dyn Iterator<Item = ElementView<'a>> + 'a> {
        let iter = (0..self.len()).map(move |i| {
            // 1. Extract element fields
            let fields = self.element_fields(i);

            // 2. 3. Extract and copy family and groups
            let (family, groups) = self.element_family_and_groups(i);

            // 4. Get the connectivity slice for element `i`
            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let connectivity = self.connectivity.slice(ndarray::s![start..end]);

            ElementView {
                fields,
                family,
                groups,
                connectivity,
                compo_type: self.cell_type.into(),
                coords: coords.view(), // full view of the coordinates
            }
        });

        Box::new(iter)
    }
    fn compo_type(&self) -> ElementType {
        self.cell_type.into()
    }
}

impl ElementBlockLike for ElementBlock {
    fn len(&self) -> usize {
        match self {
            ElementBlock::RegularCells(c) => c.len(),
            ElementBlock::PolyCells(p) => p.len(),
        }
    }

    fn params(&self) -> &HashMap<String, f64> {
        match self {
            ElementBlock::RegularCells(c) => c.params(),
            ElementBlock::PolyCells(p) => p.params(),
        }
    }

    fn fields(&self) -> &HashMap<String, ArrayD<f64>> {
        match self {
            ElementBlock::RegularCells(c) => c.fields(),
            ElementBlock::PolyCells(p) => p.fields(),
        }
    }

    fn families(&self) -> &Array1<usize> {
        match self {
            ElementBlock::RegularCells(c) => c.families(),
            ElementBlock::PolyCells(p) => p.families(),
        }
    }

    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        match self {
            ElementBlock::RegularCells(c) => c.groups(),
            ElementBlock::PolyCells(p) => p.groups(),
        }
    }
    fn iter_elements<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> Box<dyn Iterator<Item = ElementView<'a>> + 'a> {
        match self {
            ElementBlock::RegularCells(c) => c.iter_elements(coords),
            ElementBlock::PolyCells(p) => p.iter_elements(coords),
        }
    }
    fn compo_type(&self) -> ElementType {
        match self {
            ElementBlock::RegularCells(c) => c.compo_type(),
            ElementBlock::PolyCells(p) => p.compo_type(),
        }
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
