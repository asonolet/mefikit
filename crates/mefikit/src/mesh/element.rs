use ndarray::prelude as nd;
use once_cell::sync::OnceCell;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::Dimension;

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Regularity {
    Regular,
    Poly,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum PolyElemType {
    Spline,
    Pgon,
    Phed,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, Ord, PartialOrd)]
pub enum RegularElemType {
    Vertex,
    Seg2,
    Seg3,
    Seg4,
    Tri3,
    Tri6,
    Tri7,
    Quad4,
    Quad8,
    Quad9,
    Tet4,
    Tet10,
    Hex8,
    Hex21,
}

/// All kinds of elements supported in mefikit.
///
/// An element consists of a list of nodes (indices refering to a coordinates table) and can hold
/// metadata (fields, family). Those elements can be 0D, 1D, 2D or 3D. Points (VERTEX) are
/// considered as elements. A mesh will hold VERTEX elements if it needs to store node groups or
/// node fields for example. This can also be used to store nodes order, or duplicated nodes, etc.
/// Some elements are not linear but of higher order such as SEG3, HEX21. The elements node
/// connecivity follows a convention. Three kinds of elements can hold an abitrary number of nodes
/// and are specials: SPLINE, PGON (Polygon), and PHED (Polyhedron).
#[repr(u8)]
#[derive(Debug, Eq, Hash, Copy, Clone, PartialEq, PartialOrd, Ord, Serialize, Deserialize)]
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

impl From<PolyElemType> for ElementType {
    fn from(cell: PolyElemType) -> Self {
        use PolyElemType::*;
        match cell {
            Spline => Self::SPLINE,
            Pgon => Self::PGON,
            Phed => Self::PHED,
        }
    }
}

impl From<RegularElemType> for ElementType {
    fn from(cell: RegularElemType) -> Self {
        use RegularElemType::*;
        match cell {
            Vertex => Self::VERTEX,
            Seg2 => Self::SEG2,
            Seg3 => Self::SEG3,
            Seg4 => Self::SEG4,
            Tri3 => Self::TRI3,
            Tri6 => Self::TRI6,
            Tri7 => Self::TRI7,
            Quad4 => Self::QUAD4,
            Quad8 => Self::QUAD8,
            Quad9 => Self::QUAD9,
            Tet4 => Self::TET4,
            Tet10 => Self::TET10,
            Hex8 => Self::HEX8,
            Hex21 => Self::HEX21,
        }
    }
}

impl ElementType {
    pub fn dimension(&self) -> Dimension {
        use ElementType::*;
        match self {
            // 0D
            VERTEX => Dimension::D0,

            // 1D
            SEG2 | SEG3 | SEG4 | SPLINE => Dimension::D1,

            // 2D
            TRI3 | TRI6 | TRI7 | QUAD4 | QUAD8 | QUAD9 | PGON => Dimension::D2,

            // 3D
            TET4 | TET10 | HEX8 | HEX21 | PHED => Dimension::D3,
        }
    }

    pub fn regularity(&self) -> Regularity {
        use ElementType::*;
        match self {
            // poly
            SPLINE | PGON | PHED => Regularity::Poly,
            // regular
            _ => Regularity::Regular,
        }
    }

    pub fn num_nodes(&self) -> Option<usize> {
        use ElementType::*;
        match self {
            VERTEX => Some(1),
            SEG2 => Some(2),
            SEG3 => Some(3),
            SEG4 => Some(4),
            SPLINE => None, // Spline can have arbitrary number of nodes
            TRI3 => Some(3),
            TRI6 => Some(6),
            TRI7 => Some(7),
            QUAD4 => Some(4),
            QUAD8 => Some(8),
            QUAD9 => Some(9),
            PGON => None, // Polygon can have arbitrary number of nodes
            TET4 => Some(4),
            TET10 => Some(10),
            HEX8 => Some(8),
            HEX21 => Some(21),
            PHED => None, // Polyhedron can have arbitrary number of nodes
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ElementId(ElementType, usize);

impl ElementId {
    pub fn new(element_type: ElementType, index: usize) -> Self {
        ElementId(element_type, index)
    }

    pub fn element_type(&self) -> ElementType {
        self.0
    }

    pub fn index(&self) -> usize {
        self.1
    }
}

/// Imutable Item of an ElementBlock.
///
/// This struct is used to read data on an element in an element block. Note that is is only a
/// view. It holds references to this element family, fields and connectivity (local data). This
/// view still has access to the whole coordinates array and the whole groups hashmap (but not
/// publicly).
#[derive(Debug)]
pub struct Element<'a> {
    pub index: usize,
    coords: nd::ArrayView2<'a, f64>,
    pub fields: Option<BTreeMap<&'a str, nd::ArrayViewD<'a, f64>>>,
    pub family: &'a usize,
    groups: &'a BTreeMap<String, BTreeSet<usize>>,
    pub connectivity: &'a [usize],
    pub element_type: ElementType,
    // element_coords_cache: OnceCell<Array2<f64>>,
    element_groups_cache: OnceCell<Vec<String>>,
}

/// Panics if the coords array is empty or if the connectivity array is empty.
pub trait ElementLike<'a> {
    // Topology queries

    fn element_type(&self) -> ElementType;
    fn index(&self) -> usize;
    /// Returns the global index of the element.
    fn id(&self) -> ElementId {
        ElementId::new(self.element_type(), self.index())
    }
    fn connectivity(&self) -> &[usize];

    /// Returns the regularity of the element.
    ///
    /// This is used to determine if the element is a regular element or a polyhedral element.
    fn regularity(&self) -> Regularity {
        self.element_type().regularity()
    }

    /// Returns the number of nodes in the element.
    ///
    /// This is used to determine the number of nodes in the element.
    fn num_nodes(&self) -> usize {
        self.connectivity().len()
    }

    /// Returns the topological dimension of the element.
    fn dimension(&self) -> Dimension {
        self.element_type().dimension()
    }

    fn connectivity_equals(&self, other: &Self) -> bool {
        // Check if the connectivity arrays are equal
        self.connectivity().len() == other.connectivity().len()
            && self.connectivity().iter().eq(other.connectivity().iter())
    }

    // Geometric queries

    /// Returns a reference
    fn coord(&self, i: usize) -> &[f64];

    /// Returns the space dimension of the element
    fn space_dimension(&self) -> usize;

    // Groups queries

    fn groups(&self) -> &Vec<String>;
    fn in_group(&self, group: &str) -> bool;

    // TODO: fields queries
    // fn fields(&self) -> BTreeMap<String, ArrayViewD<'a, f64>>;
    // fn field(&self, field: &str) -> ArrayViewD<'a, f64>;
    // fn fields_mut(&mut self) -> BTreeMap<String, ArrayViewMutD<'a, f64>>;
    // fn field_mut(&mut self, field: &str) -> ArrayViewMutD<'a, f64>;
    // fn field_names(&self) -> Vec<String>;
    // fn has_field(&self, field: &str) -> bool;
}

impl<'a> Element<'a> {
    pub fn new(
        index: usize,
        coords: nd::ArrayView2<'a, f64>,
        fields: Option<BTreeMap<&'a str, nd::ArrayViewD<'a, f64>>>,
        family: &'a usize,
        groups: &'a BTreeMap<String, BTreeSet<usize>>,
        connectivity: &'a [usize],
        element_type: ElementType,
    ) -> Element<'a> {
        Element {
            index,
            coords,
            fields,
            family,
            groups,
            connectivity,
            element_type,
            // element_coords_cache: OnceCell::new(),
            element_groups_cache: OnceCell::new(),
        }
    }
}

impl<'a> ElementLike<'a> for Element<'a> {
    fn element_type(&self) -> ElementType {
        self.element_type
    }
    fn index(&self) -> usize {
        self.index
    }
    fn connectivity(&self) -> &[usize] {
        self.connectivity
    }

    #[inline(always)]
    fn coord(&self, i: usize) -> &[f64] {
        let co = self.connectivity;
        self.coords.row(co[i]).to_slice().unwrap()
    }

    #[cfg(feature = "rayon")]
    fn groups(&self) -> &Vec<String> {
        self.element_groups_cache.get_or_init(|| {
            self.groups
                .par_iter()
                .filter(|(_, v)| v.contains(self.family))
                .map(|(k, _)| k)
                .cloned()
                .collect()
        })
    }
    #[cfg(not(feature = "rayon"))]
    fn groups(&self) -> &Vec<String> {
        self.element_groups_cache.get_or_init(|| {
            self.groups
                .iter()
                .filter(|(_, v)| v.contains(self.family))
                .map(|(k, _)| k)
                .cloned()
                .collect()
        })
    }
    fn in_group(&self, group: &str) -> bool {
        self.groups.contains_key(group) && self.groups[group].contains(self.family)
    }
    fn space_dimension(&self) -> usize {
        self.coords.shape()[1]
    }
}

/// Mutable Item of an ElementBlock.
///
/// This struct is used to read and write data on an element in an element block. Note that is is
/// only a view. It holds mut references to this element family, fields and connectivity (local
/// data). This view still has read access to the whole coordinates array and the whole groups
/// hashmap (but not publicly).
/// This iterator is thread safe and does not allow to change an element nature or the number of
/// nodes in this element.
pub struct ElementMut<'a> {
    pub index: usize,
    coords: nd::ArrayView2<'a, f64>,
    pub fields: Option<BTreeMap<&'a str, nd::ArrayViewMutD<'a, f64>>>,
    pub family: &'a mut usize,
    groups: &'a BTreeMap<String, BTreeSet<usize>>, // safely shared across threads
    pub connectivity: &'a mut [usize],
    pub element_type: ElementType,
    // element_coords_cache: OnceCell<Array2<f64>>,
    element_groups_cache: OnceCell<Vec<String>>,
}

impl<'a> ElementLike<'a> for ElementMut<'a> {
    fn element_type(&self) -> ElementType {
        self.element_type
    }
    fn index(&self) -> usize {
        self.index
    }
    fn connectivity(&self) -> &[usize] {
        self.connectivity
    }

    #[inline(always)]
    fn coord(&self, i: usize) -> &[f64] {
        let co = &self.connectivity;
        self.coords.row(co[i]).to_slice().unwrap()
    }

    #[cfg(feature = "rayon")]
    fn groups(&self) -> &Vec<String> {
        self.element_groups_cache.get_or_init(|| {
            self.groups
                .par_iter()
                .filter(|(_, v)| v.contains(self.family))
                .map(|(k, _)| k)
                .cloned()
                .collect()
        })
    }

    #[cfg(not(feature = "rayon"))]
    fn groups(&self) -> &Vec<String> {
        self.element_groups_cache.get_or_init(|| {
            self.groups
                .iter()
                .filter(|(_, v)| v.contains(self.family))
                .map(|(k, _)| k)
                .cloned()
                .collect()
        })
    }
    fn in_group(&self, group: &str) -> bool {
        self.groups.contains_key(group) && self.groups[group].contains(self.family)
    }
    fn space_dimension(&self) -> usize {
        self.coords.shape()[1]
    }
}

impl<'a> ElementMut<'a> {
    pub fn new(
        index: usize,
        coords: nd::ArrayView2<'a, f64>,
        fields: Option<BTreeMap<&'a str, nd::ArrayViewMutD<'a, f64>>>,
        family: &'a mut usize,
        groups: &'a BTreeMap<String, BTreeSet<usize>>,
        connectivity: &'a mut [usize],
        element_type: ElementType,
    ) -> ElementMut<'a> {
        ElementMut {
            index,
            coords,
            fields,
            family,
            groups,
            connectivity,
            element_type,
            // element_coords_cache: OnceCell::new(),
            element_groups_cache: OnceCell::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_element_tri3_2d_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = array![0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn.as_slice().unwrap(),
            ElementType::TRI3,
        );

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::TRI3);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 3);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::TRI3, 0));
        // assert_abs_diff_eq!(element.measure2(), 0.5);
        assert!(element.groups().is_empty());
        assert!(!element.in_group("nonexistent_group"));
    }

    #[test]
    fn test_element_tri3_3d_basics() {
        let coords = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ];
        let conn = array![0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn.as_slice().unwrap(),
            ElementType::TRI3,
        );

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::TRI3);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 3);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::TRI3, 0));
        // assert_abs_diff_eq!(element.measure3(), 0.5);
        assert!(element.groups().is_empty());
        assert!(!element.in_group("nonexistent_group"));
    }

    #[test]
    fn test_element_quad4_2d_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = array![0, 1, 3, 2];
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn.as_slice().unwrap(),
            ElementType::QUAD4,
        );

        assert_eq!(element.connectivity.len(), 4);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::QUAD4);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 4);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::QUAD4, 0));
        // assert_abs_diff_eq!(element.measure2(), 1.0);
        assert!(element.groups().is_empty());
        assert!(!element.in_group("nonexistent_group"));
    }
}
