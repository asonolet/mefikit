use ndarray::{s, Array1, Array2, ArrayD, ArrayView1, ArrayViewD, Axis};
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

#[derive(Eq, Hash, Copy, Clone, PartialEq)]
pub enum MeshCompoType {
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

impl From<PolyCellType> for MeshCompoType {
    fn from(cell: PolyCellType) -> Self {
        match cell {
            PolyCellType::SPLINE => MeshCompoType::SPLINE,
            PolyCellType::PGON => MeshCompoType::PGON,
            PolyCellType::PHED => MeshCompoType::PHED,
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

pub struct Vertices {
    connectivity: Array1<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
    len: usize,
}

pub struct RegularCells {
    cell_type: RegularCellType,
    connectivity: Array2<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
    len: usize,
}

pub struct PolyCells {
    cell_type: PolyCellType,
    connectivity: Array1<usize>,
    offsets: Array1<usize>,
    params: HashMap<String, f64>,
    fields: HashMap<String, ArrayD<f64>>,
    families: Array1<usize>,
    groups: HashMap<String, HashSet<usize>>,
    len: usize,
}

pub struct MeshElementView<'a> {
    pub fields: HashMap<String, ArrayViewD<'a, f64>>,
    pub family: usize,
    pub groups: HashSet<String>,
    pub connectivity: ArrayView1<'a, usize>,
    pub compo_type: MeshCompoType,
}

pub enum MeshCompo {
    Vertices(Vertices),
    RegularCells(RegularCells),
    PolyCells(PolyCells),
}

pub struct UMesh {
    coords: Array2<f64>,
    components: HashMap<MeshCompoType, MeshCompo>,
}

trait MeshComponent {
    fn len(&self) -> usize;
    fn params(&self) -> &HashMap<String, f64>;
    fn fields(&self) -> &HashMap<String, ArrayD<f64>>;
    fn families(&self) -> &Array1<usize>;
    fn groups(&self) -> &HashMap<String, HashSet<usize>>;
    fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_>;
    fn compo_type(&self) -> MeshCompoType;
    fn element_fields(&self, i: usize) -> HashMap<String, ArrayViewD<f64>> {
        self.fields()
            .iter()
            .map(|(k, v)| (k.clone(), v.index_axis(Axis(0), i)))
            .collect()
    }
    fn element_family_and_groups(&self, i: usize) -> (usize, HashSet<String>) {
        let families = self.families();
        let groups = self.groups();

        let family = families[i];
        let group_names = groups
            .iter()
            .filter_map(|(name, fset)| {
                if fset.contains(&family) {
                    Some(name.clone())
                } else {
                    None
                }
            })
        .collect();

        (family, group_names)
    }
}

impl Vertices {
    pub fn new(
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, HashSet<usize>>,
    ) -> Self {
        let len = families.len();
        let connectivity = Array1::from_shape_fn(len, |i| i);
        Self {
            connectivity,
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
        cell_type: RegularCellType,
        connectivity: Array2<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, HashSet<usize>>,
    ) -> Self {
        let len = families.len();
        Self {
            cell_type,
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
        cell_type: PolyCellType,
        connectivity: Array1<usize>,
        offsets: Array1<usize>,
        params: HashMap<String, f64>,
        fields: HashMap<String, ArrayD<f64>>,
        families: Array1<usize>,
        groups: HashMap<String, HashSet<usize>>,
    ) -> Self {
        let len = families.len();
        Self {
            cell_type,
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
    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        &self.groups
    }
    fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
        Box::new((0..self.len).map(move |i| {
            let fields = self.element_fields(i);

            let (family, groups) = self.element_family_and_groups(i);

            let connectivity = self.connectivity.slice(s![i..=i]);

            MeshElementView {
                fields,
                family,
                groups,
                connectivity,
                compo_type: MeshCompoType::VERTEX,
            }
        }))
    }
    fn compo_type(&self) -> MeshCompoType {
        MeshCompoType::VERTEX
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
    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        &self.groups
    }
    fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
        Box::new((0..self.len).map(move |i| {
            // 1. Extract element fields
            let fields = self.element_fields(i);

            let (family, groups) = self.element_family_and_groups(i);

            // 4. Extract connectivity
            let connectivity = self.connectivity.row(i);

            // 5. Compose MeshElementView
            MeshElementView {
                fields,
                family,
                groups,
                connectivity,
                compo_type: self.cell_type.into(), // assuming From<RegularCellType> for MeshCompoType
            }
        }))
    }
    fn compo_type(&self) -> MeshCompoType {
        self.cell_type.into()
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
    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        &self.groups
    }
    fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
        Box::new((0..self.len).map(move |i| {
            let fields = self.element_fields(i);

            let (family, groups) = self.element_family_and_groups(i);

            let start = self.offsets[i];
            let end = self.offsets[i + 1];
            let connectivity = self.connectivity.slice(ndarray::s![start..end]);

            MeshElementView {
                fields,
                family,
                groups,
                connectivity,
                compo_type: self.cell_type.into(), // assumes From<PolyCellType> for MeshCompoType
            }
        }))
    }
    fn compo_type(&self) -> MeshCompoType {
        self.cell_type.into()
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

    fn groups(&self) -> &HashMap<String, HashSet<usize>> {
        match self {
            MeshCompo::Vertices(v) => v.groups(),
            MeshCompo::RegularCells(c) => c.groups(),
            MeshCompo::PolyCells(p) => p.groups(),
        }
    }
    fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
        match self {
            MeshCompo::Vertices(v) => v.iter_elements(),
            MeshCompo::RegularCells(c) => c.iter_elements(),
            MeshCompo::PolyCells(p) => p.iter_elements(),
        }
    }
    fn compo_type(&self) -> MeshCompoType {
        match self {
            MeshCompo::Vertices(v) => v.compo_type(),
            MeshCompo::RegularCells(c) => c.compo_type(),
            MeshCompo::PolyCells(p) => p.compo_type(),
        }
    }
}



pub trait IntoMeshCompo {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo);
}

impl IntoMeshCompo for Vertices {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (MeshCompoType::VERTEX, MeshCompo::Vertices(self))
    }
}

impl IntoMeshCompo for RegularCells {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (self.cell_type.into(), MeshCompo::RegularCells(self))
    }
}

impl IntoMeshCompo for PolyCells {
    fn into_mesh_compo(self) -> (MeshCompoType, MeshCompo) {
        (self.cell_type.into(), MeshCompo::PolyCells(self))
    }
}

impl UMesh {
    pub fn add_compo<T: IntoMeshCompo>(&mut self, compo: T) {
        let (key, wrapped) = compo.into_mesh_compo();
        self.components.insert(key, wrapped);
    }
}
