use ndarray::{concatenate, s, Array1, Array2, ArrayD, Axis};
use std::collections::HashMap;

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

#[derive(Copy, Clone)]
pub enum MeshCompoType {
    // 0d
    VERTICES,

    // 1d
    SEG2,
    SEG3,
    SEG4,
    SPLINES,

    // 2d
    TRI3,
    TRI6,
    TRI7,
    QUAD4,
    QUAD8,
    QUAD9,
    PGONS,

    // 3d
    TET4,
    TET10,
    HEX8,
    HEX21,
    PHEDS,
}

impl From<PolyCellType> for MeshCompoType {
    fn from(cell: PolyCellType) -> Self {
        match cell {
            PolyCellType::SPLINE => MeshCompoType::SPLINES,
            PolyCellType::PGON => MeshCompoType::PGONS,
            PolyCellType::PHED => MeshCompoType::PHEDS,
        }
    }
}

impl From<RegularCellType> for MeshCompoType {
    fn from(cell: RegularCellType) -> Self {
        match cell {
            RegularCellType::SEG2 => MeshCompoType::SEG2,
            RegularCellType::SEG3 => MeshCompoType::SEG3,
            RegularCellType::SEG4 => MeshCompoType::SEG4,
            RegularCellType::TRI3 => MeshCompoType::TRI3,
            RegularCellType::TRI6 => MeshCompoType::TRI6,
            RegularCellType::TRI7 => MeshCompoType::TRI7,
            RegularCellType::QUAD4 => MeshCompoType::QUAD4,
            RegularCellType::QUAD8 => MeshCompoType::QUAD8,
            RegularCellType::QUAD9 => MeshCompoType::QUAD9,
            RegularCellType::TET4 => MeshCompoType::TET4,
            RegularCellType::TET10 => MeshCompoType::TET10,
            RegularCellType::HEX8 => MeshCompoType::HEX8,
            RegularCellType::HEX21 => MeshCompoType::HEX21,
        }
    }
}

struct Vertices {
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, Array1<usize>>,
    len: usize,
}

struct RegularCells {
    element_type: RegularCellType,
    connectivity: Array2<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, Array1<usize>>,
    len: usize,
}

struct PolyCells {
    elem_type: PolyCellType,
    connectivity: Array1<usize>,
    offsets: Array1<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, Array1<usize>>,
    len: usize,
}

enum MeshCompo {
    Vertices(Vertices),
    RegularCells(RegularCells),
    PolyCells(PolyCells),
}

struct UMesh {
    coords: Array2<f64>,
    components: HashMap<MeshCompoType, MeshCompo>,
}

trait MeshComponent {
    fn len(&self) -> usize;
    fn params(&self) -> &HashMap<String, f64>;
    fn fields(&self) -> &HashMap<String, ArrayD<f64>>;
    fn families(&self) -> &Array1<usize>;
    fn groups(&self) -> &HashMap<String, Array1<usize>>;
}

impl Vertices {
    pub fn new(
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, Array1<usize>>,
    ) -> Self {
        let len = families.len();
        Self {
            params,
            fields,
            families,
            groups,
            len,
        }
    }
}

impl RegularCells {
    pub fn new(
        element_type: RegularCellType,
        connectivity: Array2<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, Array1<usize>>,
    ) -> Self {
        let len = families.len();
        Self {
            element_type,
            connectivity,
            params,
            fields,
            families,
            groups,
            len,
        }
    }
}

impl PolyCells {
    pub fn new(
        elem_type: PolyCellType,
        connectivity: Array1<usize>,
        offsets: Array1<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, Array1<usize>>,
    ) -> Self {
        let len = families.len();
        Self {
            elem_type,
            connectivity,
            offsets,
            params,
            fields,
            families,
            groups,
            len,
        }
    }
}

impl MeshComponent for Vertices {
    fn len(&self) -> usize {
        self.len
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
    fn groups(&self) -> &HashMap<String, Array1<usize>> {
        &self.groups
    }
}

impl MeshComponent for RegularCells {
    fn len(&self) -> usize {
        self.len
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
    fn groups(&self) -> &HashMap<String, Array1<usize>> {
        &self.groups
    }
}

impl MeshComponent for PolyCells {
    fn len(&self) -> usize {
        self.len
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
    fn groups(&self) -> &HashMap<String, Array1<usize>> {
        &self.groups
    }
}

impl MeshComponent for MeshCompo {
    fn len(&self) -> usize {
        match self {
            MeshCompo::Vertices(v) => v.len(),
            MeshCompo::RegularCells(c) => c.len(),
            MeshCompo::PolyCells(p) => p.len(),
        }
    }

    fn params(&self) -> &HashMap<String, f64> {
        match self {
            MeshCompo::Vertices(v) => v.params(),
            MeshCompo::RegularCells(c) => c.params(),
            MeshCompo::PolyCells(p) => p.params(),
        }
    }

    fn fields(&self) -> &HashMap<String, ArrayD<f64>> {
        match self {
            MeshCompo::Vertices(v) => v.fields(),
            MeshCompo::RegularCells(c) => c.fields(),
            MeshCompo::PolyCells(p) => p.fields(),
        }
    }

    fn families(&self) -> &Array1<usize> {
        match self {
            MeshCompo::Vertices(v) => v.families(),
            MeshCompo::RegularCells(c) => c.families(),
            MeshCompo::PolyCells(p) => p.families(),
        }
    }

    fn groups(&self) -> &HashMap<String, Array1<usize>> {
        match self {
            MeshCompo::Vertices(v) => v.groups(),
            MeshCompo::RegularCells(c) => c.groups(),
            MeshCompo::PolyCells(p) => p.groups(),
        }
    }
}


pub trait IntoMeshCompo {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo);
}

impl IntoMeshCompo for Vertices {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (MeshCompoType::VERTICES, MeshCompo::Vertices(self))
    }
}

impl IntoMeshCompo for RegularCells {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (self.element_type.into(), MeshCompo::RegularCells(self))
    }
}

impl IntoMeshCompo for PolyCells {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (self.elem_type.into(), MeshCompo::PolyCells(self))
    }
}

impl UMesh {
    pub fn add_compo<T: IntoMeshCompo>(&mut self, compo: T) {
        let (key, wrapped) = compo.into_mesh_compo();
        self.components.insert(key, wrapped);
    }
}

