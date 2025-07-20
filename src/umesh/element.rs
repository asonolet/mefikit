use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use crate::umesh::measure as mes;

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

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum Regularity {
    Regular,
    Poly,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq)]
pub enum PolyElemType {
    SPLINE,
    PGON,
    PHED,
}

#[derive(Copy, Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, Ord, PartialOrd)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Dimension {
    D0,
    D1,
    D2,
    D3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub struct ElementId(ElementType, usize);

#[derive(Debug, Clone)]
pub struct ElementIds(BTreeMap<ElementType, Vec<usize>>);

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

impl Default for ElementIds {
    fn default() -> Self {
        Self::new()
    }
}

impl ElementIds {
    pub fn new() -> Self {
        ElementIds(BTreeMap::new())
    }

    pub fn add(&mut self, element_type: ElementType, index: usize) {
        self.0.entry(element_type).or_default().push(index);
    }

    pub fn remove(&mut self, element_type: ElementType, index: usize) -> Option<usize> {
        if let Some(indices) = self.0.get_mut(&element_type) {
            if let Some(pos) = indices.iter().position(|&i| i == index) {
                return Some(indices.remove(pos));
            }
        }
        None
    }

    pub fn get(&self, element_type: &ElementType) -> Option<&Vec<usize>> {
        self.0.get(element_type)
    }

    pub fn contains(&self, element_type: ElementType) -> bool {
        self.0.contains_key(&element_type)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&ElementType, &Vec<usize>)> {
        self.0.iter()
    }
    pub fn into_iter(self) -> impl Iterator<Item = ElementId> {
        self.0
            .into_iter()
            .flat_map(|(et, indices)| indices.into_iter().map(move |index| ElementId(et, index)))
    }
    pub fn into_par_iter(self) -> impl ParallelIterator<Item = ElementId> {
        self.0.into_par_iter().flat_map(|(et, indices)| {
            indices
                .into_par_iter()
                .map(move |index| ElementId(et, index))
        })
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn len(&self) -> usize {
        self.0.values().map(|v| v.len()).sum()
    }
    pub fn element_types(&self) -> Vec<ElementType> {
        self.0.keys().cloned().collect()
    }
}

impl From<BTreeMap<ElementType, Vec<usize>>> for ElementIds {
    fn from(map: BTreeMap<ElementType, Vec<usize>>) -> Self {
        ElementIds(map)
    }
}

impl FromIterator<ElementId> for ElementIds {
    fn from_iter<T: IntoIterator<Item = ElementId>>(iter: T) -> Self {
        let mut ids = ElementIds::new();
        for id in iter {
            ids.add(id.element_type(), id.index());
        }
        ids
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
    element_coords_cache: OnceCell<Array2<f64>>,
    element_groups_cache: OnceCell<Vec<String>>,
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

    fn to_simplexes(&self) -> Vec<(ElementType, Vec<usize>)> {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX => vec![(VERTEX, vec![co[0]])],
            SEG2 | SEG3 | SEG4 => vec![(SEG2, vec![co[0], co[1]])],
            TRI3 | TRI6 | TRI7 => vec![(TRI3, vec![co[0], co[1], co[2]])],
            QUAD4 | QUAD8 | QUAD9 => vec![
                (TRI3, vec![co[0], co[1], co[3]]),
                (TRI3, vec![co[2], co[3], co[1]]),
            ],
            TET4 | TET10 => vec![(TET4, vec![co[0], co[1], co[2], co[3]])],
            HEX8 | HEX21 => vec![
                (TET4, vec![co[0], co[1], co[3], co[4]]),
                (TET4, vec![co[2], co[3], co[1], co[6]]),
                (TET4, vec![co[7], co[6], co[4], co[3]]),
                (TET4, vec![co[5], co[4], co[6], co[1]]),
                (TET4, vec![co[4], co[6], co[3], co[1]]),
            ],
            _ => todo!(),
        }
    }

    /// Geometric queries

    /// Returns a reference to an owned
    fn coords(&self) -> &Array2<f64>;

    /// Returns the space dimension of the element
    fn space_dimension(&self) -> usize;

    fn bounding_box(&self) -> (Array1<f64>, Array1<f64>) {
        // Returns the bounding box of the element
        let coords = self.coords();
        let min = coords.fold_axis(Axis(0), f64::INFINITY, |x: &f64, y: &f64| x.min(*y));
        let max = coords.fold_axis(Axis(0), f64::NEG_INFINITY, |x: &f64, y: &f64| x.max(*y));
        (min, max)
    }

    fn centroid(&self) -> Array1<f64> {
        self.coords().mean_axis(Axis(0)).unwrap()
    }

    fn measure2(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => mes::dist(self.coords().row(0), self.coords().row(1)),
            TRI3 => mes::surf_tri2(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
            ),
            QUAD4 => mes::surf_quad2(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(3).as_slice().unwrap().try_into().unwrap(),
            ),
            _ => todo!(),
        }
    }

    fn measure3(&self) -> f64 {
        // Returns the measure of the element
        // For 0D elements, return 0.0
        // For 1D elements, return the length
        // For 2D elements, return the area
        use ElementType::*;
        match self.element_type() {
            VERTEX => 0.0,
            SEG2 => todo!(),
            TRI3 => mes::surf_tri3(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
            ),
            QUAD4 => mes::surf_quad3(
                self.coords().row(0).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(1).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(2).as_slice().unwrap().try_into().unwrap(),
                self.coords().row(3).as_slice().unwrap().try_into().unwrap(),
            ),
            _ => todo!(),
        }
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
            element_coords_cache: OnceCell::new(),
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
    fn connectivity<'b>(&'b self) -> ArrayView1<'b, usize> {
        self.connectivity.view()
    }
    fn coords(&self) -> &Array2<f64> {
        // TODO: implement cache mechanism for this using once_cell or similar
        self.element_coords_cache.get_or_init(|| {
            self.coords
                .select(Axis(0), self.connectivity.as_slice().unwrap())
        })
    }
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
    element_coords_cache: OnceCell<Array2<f64>>,
    element_groups_cache: OnceCell<Vec<String>>,
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
    fn coords(&self) -> &Array2<f64> {
        // TODO: implement cache mechanism for this using once_cell or similar
        self.element_coords_cache.get_or_init(|| {
            self.coords
                .select(Axis(0), self.connectivity.as_slice().unwrap())
        })
    }
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
            element_coords_cache: OnceCell::new(),
            element_groups_cache: OnceCell::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;
    use ndarray::array;

    #[test]
    fn test_element_tri3_2d_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = array![0, 1, 2];
        let fields = BTreeMap::new();
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            fields,
            &family,
            &groups,
            conn.view(),
            ElementType::TRI3,
        );

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::TRI3);
        assert_eq!(element.coords().shape(), [3, 2]);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 3);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::TRI3, 0));
        assert_abs_diff_eq!(element.measure2(), 0.5);
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
        let fields = BTreeMap::new();
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            fields,
            &family,
            &groups,
            conn.view(),
            ElementType::TRI3,
        );

        assert_eq!(element.connectivity.len(), 3);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::TRI3);
        assert_eq!(element.coords().shape(), [3, 3]);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 3);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::TRI3, 0));
        assert_abs_diff_eq!(element.measure3(), 0.5);
        assert!(element.groups().is_empty());
        assert!(!element.in_group("nonexistent_group"));
    }

    #[test]
    fn test_element_quad4_2d_basics() {
        let coords = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let conn = array![0, 1, 3, 2];
        let fields = BTreeMap::new();
        let groups = BTreeMap::new();
        let family = 0;

        let element = Element::new(
            0,
            coords.view(),
            fields,
            &family,
            &groups,
            conn.view(),
            ElementType::QUAD4,
        );

        assert_eq!(element.connectivity.len(), 4);
        assert_eq!(element.index, 0);
        assert_eq!(element.element_type, ElementType::QUAD4);
        assert_eq!(element.coords().shape(), [4, 2]);
        assert_eq!(element.dimension(), Dimension::D2);
        assert_eq!(element.num_nodes(), 4);
        assert_eq!(element.regularity(), Regularity::Regular);
        assert_eq!(element.id(), ElementId::new(ElementType::QUAD4, 0));
        assert_abs_diff_eq!(element.measure2(), 1.0);
        assert!(element.groups().is_empty());
        assert!(!element.in_group("nonexistent_group"));
    }
}
