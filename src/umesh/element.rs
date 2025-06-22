use ndarray::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::collections::BTreeSet;

// #[derive(Copy, Clone)]
// pub enum EdgeType {
//     SEG2,
//     SEG3,
//     SEG4,
//     SPLINES,
// }
//
// #[derive(Copy, Clone)]
// pub enum FaceType {
//     TRI3,
//     TRI6,
//     TRI7,
//     QUAD4,
//     QUAD8,
//     QUAD9,
//     PGON,
// }
//
// #[derive(Copy, Clone)]
// pub enum VolumeType {
//     TET4,
//     TET10,
//     HEX8,
//     HEX21,
//     PHDRON,
// }

#[derive(Copy, Clone)]
pub enum Regularity {
    Regular,
    Poly,
}

#[derive(Copy, Clone)]
pub enum PolyElemType {
    SPLINE,
    PGON,
    PHED,
}

#[derive(Copy, Clone)]
pub enum RegularElemType {
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

/// All kinds of elements supported in mefikit.
///
/// An element consists of a list of nodes (indices refering to a coordinates table) and can hold
/// metadata (fields, family). Those elements can be 0D, 1D, 2D or 3D. Points (VERTEX) are
/// considered as elements. A mesh will hold VERTEX elements if it needs to store node groups or
/// node fields for example. This can also be used to store nodes order, or duplicated nodes, etc.
/// Some elements are not linear but of higher order such as SEG3, HEX21. The elements node
/// connecivity follows a convention. Three kinds of elements can hold an abitrary number of nodes
/// and are specials: SPLINE, PGON (Polygon), and PHED (Polyhedron).
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
        match cell {
            PolyElemType::SPLINE => ElementType::SPLINE,
            PolyElemType::PGON => ElementType::PGON,
            PolyElemType::PHED => ElementType::PHED,
        }
    }
}

impl From<RegularElemType> for ElementType {
    fn from(cell: RegularElemType) -> Self {
        match cell {
            RegularElemType::VERTEX => ElementType::VERTEX,
            RegularElemType::SEG2 => ElementType::SEG2,
            RegularElemType::SEG3 => ElementType::SEG3,
            RegularElemType::SEG4 => ElementType::SEG4,
            RegularElemType::TRI3 => ElementType::TRI3,
            RegularElemType::TRI6 => ElementType::TRI6,
            RegularElemType::TRI7 => ElementType::TRI7,
            RegularElemType::QUAD4 => ElementType::QUAD4,
            RegularElemType::QUAD8 => ElementType::QUAD8,
            RegularElemType::QUAD9 => ElementType::QUAD9,
            RegularElemType::TET4 => ElementType::TET4,
            RegularElemType::TET10 => ElementType::TET10,
            RegularElemType::HEX8 => ElementType::HEX8,
            RegularElemType::HEX21 => ElementType::HEX21,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    D0,
    D1,
    D2,
    D3,
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
pub struct Element<'a> {
    pub index: usize,
    coords: ArrayView2<'a, f64>,
    pub fields: BTreeMap<&'a str, ArrayViewD<'a, f64>>,
    pub family: &'a usize,
    groups: &'a BTreeMap<String, BTreeSet<usize>>,
    pub connectivity: ArrayView1<'a, usize>,
    pub element_type: ElementType,
}

/// Panics if the coords array is empty or if the connectivity array is empty.
pub trait ElementLike<'a> {
    /// Topology queries

    fn element_type(&self) -> ElementType;
    fn index(&self) -> usize;
    /// Returns the global index of the element.
    fn id(&self) -> ElementId {
        ElementId::new(self.element_type(), self.index())
    }
    fn connectivity<'b>(&'b self) -> ArrayView1<'b, usize>;

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
        self.connectivity().shape()[0]
    }

    /// Returns the topological dimension of the element.
    fn dimension(&self) -> Dimension {
        self.element_type().dimension()
    }

    fn connectivity_equals(&self, other: &Self) -> bool {
        // Check if the connectivity arrays are equal
        // self.connectivity().shape() == other.connectivity().shape()
        //     && self.connectivity().iter().eq(other.connectivity().iter())
        todo!()
    }

    /// This function returns the subentities of the element based on the codimension.
    fn subentities(&self, codim: Option<Dimension>) -> Option<Vec<(ElementType, Vec<usize>)>> {
        use ElementType::*;
        let codim = match codim {
            None => Dimension::D1,
            Some(c) => c,
        };
        let co = self.connectivity();
        match self.element_type() {
            SEG2 | SEG3 | SEG4 => {
                // 1D elements have edges as subentities
                if codim == Dimension::D1 {
                    Some(vec![(VERTEX, vec![co[0]]), (VERTEX, vec![co[1]])])
                } else {
                    None
                }
            }
            TRI3 => {
                // 2D elements have edges as subentities
                if codim == Dimension::D1 {
                    Some(vec![
                        (SEG2, vec![co[0], co[1]]),
                        (SEG2, vec![co[1], co[2]]),
                        (SEG2, vec![co[2], co[0]]),
                    ])
                } else {
                    None
                }
            }
            TRI6 | TRI7 => {
                // 2D Quad elements have edges3 as subentities
                if codim == Dimension::D1 {
                    Some(vec![
                        (SEG3, vec![co[0], co[1], co[3]]),
                        (SEG3, vec![co[1], co[2], co[4]]),
                        (SEG3, vec![co[2], co[0], co[5]]),
                    ])
                } else {
                    None
                }
            }
            QUAD4 => {
                // 2D elements have edges as subentities
                if codim == Dimension::D1 {
                    Some(vec![
                        (SEG2, vec![co[0], co[1]]),
                        (SEG2, vec![co[1], co[2]]),
                        (SEG2, vec![co[2], co[3]]),
                        (SEG2, vec![co[3], co[0]]),
                    ])
                } else {
                    None
                }
            }
            TET4 => {
                // 3D elements have faces as subentities
                if codim == Dimension::D1 {
                    Some(vec![
                        (TRI3, vec![co[0], co[1], co[2]]),
                        (TRI3, vec![co[1], co[2], co[3]]),
                        (TRI3, vec![co[2], co[3], co[0]]),
                        (TRI3, vec![co[3], co[0], co[1]]),
                    ])
                } else if codim == Dimension::D2 {
                    todo!()
                } else {
                    None
                }
            }
            _ => todo!(), // For other types, return empty vector
        }
    }

    /// Geometric queries

    fn coords(&self) -> Array2<f64>;

    /// Returns the space dimension of the element
    fn space_dimension(&self) -> usize;

    fn bounding_box(&self) -> (Array1<f64>, Array1<f64>) {
        // Returns the bounding box of the element
        // let coords = self.coords();
        // let min = coords.fold(Array1::from_elem(coords.shape()[1], f64::INFINITY), |a, b| a.zip_map(&b, f64::min));
        // let max = coords.fold(Array1::from_elem(coords.shape()[1], f64::NEG_INFINITY), |a, b| a.zip_map(&b, f64::max));
        // (min, max)
        todo!()
    }

    fn centroid(&self) -> Array1<f64> {
        self.coords().mean_axis(Axis(0)).unwrap()
    }

    fn measure(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
        // For 3D elements, return the volume
        todo!()
    }

    fn is_point_inside(&self, point: &[f64]) -> bool {
        // Returns true if the point is inside the element
        // For 0D elements, return true if the point is equal to the element's coordinates
        // For 1D elements, return true if the point is between the two nodes
        // For 2D elements, return true if the point is inside the polygon
        // For 3D elements, return true if the point is inside the polyhedron
        todo!()
    }

    /// Groups queries

    fn groups(&self) -> Vec<String>;
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
        coords: ArrayView2<'a, f64>,
        fields: BTreeMap<&'a str, ArrayViewD<'a, f64>>,
        family: &'a usize,
        groups: &'a BTreeMap<String, BTreeSet<usize>>,
        connectivity: ArrayView1<'a, usize>,
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
    fn connectivity<'b>(&'b self) -> ArrayView1<'b, usize> {
        self.connectivity.view()
    }
    fn coords(&self) -> Array2<f64> {
        // TODO: implement cache mechanism for this using once_cell or similar
        self.coords
            .select(Axis(0), self.connectivity.as_slice().unwrap())
    }
    fn groups(&self) -> Vec<String> {
        // TODO: implement cache mechanism for this using once_cell or similar
        self.groups
            .par_iter()
            .filter(|(_, v)| v.contains(self.family))
            .map(|(k, _)| k)
            .cloned()
            .collect()
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
    coords: ArrayView2<'a, f64>,
    pub connectivity: ArrayViewMut1<'a, usize>,
    pub family: &'a mut usize,
    pub fields: BTreeMap<&'a str, ArrayViewMutD<'a, f64>>,
    groups: &'a BTreeMap<String, BTreeSet<usize>>, // safely shared across threads
    pub element_type: ElementType,
}

impl<'a> ElementLike<'a> for ElementMut<'a> {
    fn element_type(&self) -> ElementType {
        self.element_type
    }
    fn index(&self) -> usize {
        self.index
    }
    fn connectivity<'b>(&'b self) -> ArrayView1<'b, usize> {
        self.connectivity.view()
    }
    fn coords(&self) -> Array2<f64> {
        self.coords
            .select(Axis(0), self.connectivity.as_slice().unwrap())
    }
    fn groups(&self) -> Vec<String> {
        self.groups
            .iter()
            .filter(|(_, v)| v.contains(self.family))
            .map(|(k, _)| k)
            .cloned()
            .collect()
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
        coords: ArrayView2<'a, f64>,
        connectivity: ArrayViewMut1<'a, usize>,
        family: &'a mut usize,
        fields: BTreeMap<&'a str, ArrayViewMutD<'a, f64>>,
        groups: &'a BTreeMap<String, BTreeSet<usize>>,
        element_type: ElementType,
    ) -> ElementMut<'a> {
        ElementMut {
            index,
            coords,
            connectivity,
            family,
            fields,
            groups,
            element_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_element_struct_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = array![0, 1, 2];
        let fields = BTreeMap::new();
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element {
            index: 0,
            coords: coords.view(),
            fields,
            family: &family,
            groups: &groups,
            connectivity: conn.view(),
            element_type: ElementType::TRI3,
        };

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.index, 0);
    }
}
