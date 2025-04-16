mod umesh {

    use ndarray::{Array1, Array2, ArrayD, ArrayView1, ArrayViewD, Axis};
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

    #[derive(Eq, Hash, Copy, Clone, PartialEq)]
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

    pub struct MeshElementView<'a> {
        pub fields: HashMap<String, ArrayViewD<'a, f64>>,
        pub family: usize,
        pub groups: HashSet<String>,
        pub connectivity: ArrayView1<'a, usize>,
        pub compo_type: ElementType,
    }

    pub enum ElementBlock {
        RegularCells(RegularCells),
        PolyCells(PolyCells),
    }

    pub struct UMesh {
        coords: Array2<f64>,
        element_blocks: HashMap<ElementType, ElementBlock>,
    }

    trait ElementBlockLike {
        fn len(&self) -> usize;
        fn params(&self) -> &HashMap<String, f64>;
        fn fields(&self) -> &HashMap<String, ArrayD<f64>>;
        fn families(&self) -> &Array1<usize>;
        fn groups(&self) -> &HashMap<String, HashSet<usize>>;
        fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_>;
        fn compo_type(&self) -> ElementType;
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
        fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
            Box::new((0..self.len()).map(move |i| {
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
                    compo_type: self.cell_type.into(), // assuming From<RegularCellType> for ElementType
                }
            }))
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
        fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
            Box::new((0..self.len()).map(move |i| {
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
                    compo_type: self.cell_type.into(), // assumes From<PolyCellType> for ElementType
                }
            }))
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
        fn iter_elements(&self) -> Box<dyn Iterator<Item = MeshElementView> + '_> {
            match self {
                ElementBlock::RegularCells(c) => c.iter_elements(),
                ElementBlock::PolyCells(p) => p.iter_elements(),
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

    impl UMesh {
        pub fn add_compo<T: IntoElementBlockEntry>(&mut self, compo: T) {
            let (key, wrapped) = compo.into_entry();
            self.element_blocks.entry(key).or_insert(wrapped);
        }
    }
}
