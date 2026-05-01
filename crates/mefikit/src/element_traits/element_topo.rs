//! Topological operations for mesh elements.
//!
//! Provides the [`ElementTopo`] trait for extracting subentities (faces, edges, vertices)
//! and decomposing elements into simplexes.

use ndarray::prelude::*;

use crate::mesh::Connectivity;
use crate::mesh::{Dimension, ElementLike, ElementType};

/// Topological operations for mesh elements.
///
/// Extends [`ElementLike`] with methods for extracting subentities at various
/// codimensions and decomposing elements into simplex components.
pub trait ElementTopo<'a>: ElementLike<'a> {
    /// Returns the subentities of the element at the given codimension.
    ///
    /// For example, for a QUAD4 element:
    /// - `codim = D1` returns the 4 edges (SEG2)
    /// - `codim = D2` returns the 4 vertices (VERTEX)
    ///
    /// If `codim` is `None`, defaults to `D1`.
    fn subentities(&self, codim: Option<Dimension>) -> Vec<(ElementType, Connectivity)> {
        use ElementType::*;
        let codim = match codim {
            None => Dimension::D1,
            Some(c) => c,
        };
        let co = self.connectivity();
        let mut res = Vec::new();
        match self.element_type() {
            SEG2 | SEG3 | SEG4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[[co[0]], [co[1]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim different from D1 on SEG"),
            },
            TRI3 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[[co[0], co[1]], [co[1], co[2]], [co[2], co[0]]]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on TRI3"),
            },
            TRI6 | TRI7 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[3]],
                        [co[1], co[2], co[4]],
                        [co[2], co[0], co[5]],
                    ]);
                    res.push((SEG3, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on TRI3"),
            },
            QUAD4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[1], co[2]],
                        [co[2], co[3]],
                        [co[3], co[0]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]], [co[3]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 and D2 on QUAD"),
            },
            TET4 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[2]],
                        [co[1], co[2], co[3]],
                        [co[2], co[3], co[0]],
                        [co[3], co[0], co[1]],
                    ]);
                    res.push((TRI3, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[0], co[2]],
                        [co[0], co[3]],
                        [co[1], co[2]],
                        [co[1], co[3]],
                        [co[2], co[3]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D3 => {
                    let conn = arr2(&[[co[0]], [co[1]], [co[2]], [co[3]]]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => {
                    panic!("It is not possible to ask for codim diff from D1, D2 or D3 on TET")
                }
            },
            HEX8 => match codim {
                Dimension::D1 => {
                    let conn = arr2(&[
                        [co[0], co[1], co[2], co[3]],
                        [co[0], co[3], co[7], co[4]],
                        [co[0], co[4], co[5], co[1]],
                        [co[1], co[5], co[6], co[2]],
                        [co[2], co[6], co[7], co[3]],
                        [co[4], co[7], co[6], co[5]],
                    ]);
                    res.push((QUAD4, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = arr2(&[
                        [co[0], co[1]],
                        [co[0], co[3]],
                        [co[0], co[4]],
                        [co[1], co[2]],
                        [co[1], co[5]],
                        [co[2], co[3]],
                        [co[2], co[6]],
                        [co[3], co[7]],
                        [co[4], co[5]],
                        [co[4], co[7]],
                        [co[5], co[6]],
                        [co[6], co[7]],
                    ]);
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D3 => {
                    let conn = arr2(&[
                        [co[0]],
                        [co[1]],
                        [co[2]],
                        [co[3]],
                        [co[4]],
                        [co[5]],
                        [co[6]],
                        [co[7]],
                    ]);
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => {
                    panic!("It is not possible to ask for codim diff from D1, D2 or D3 on HEX")
                }
            },
            PGON => match codim {
                Dimension::D1 => {
                    let mut conn: Vec<_> = co.windows(2).flatten().cloned().collect();
                    conn.push(co[co.len() - 1]);
                    conn.push(co[0]);
                    let conn = Array2::from_shape_vec([conn.len() / 2, 2], conn).unwrap();
                    res.push((SEG2, Connectivity::new_regular(conn.to_shared())));
                }
                Dimension::D2 => {
                    let conn = Array2::from_shape_vec([co.len(), 1], co.to_vec()).unwrap();
                    res.push((VERTEX, Connectivity::new_regular(conn.to_shared())));
                }
                _ => panic!("It is not possible to ask for codim diff from D1 or D2 on PGON"),
            },
            PHED => match codim {
                Dimension::D1 => {
                    let mut conn = Vec::new();
                    let mut offsets = Vec::new();
                    let mut offset = 0;
                    co.split_inclusive(|&e| e == usize::MAX).for_each(|a| {
                        let len = a.len() - 1;
                        offset += len;
                        offsets.push(offset);
                        conn.append(&mut a[..len].to_vec())
                    });
                    let offsets = Array1::from_vec(offsets);
                    let conn = Array::from_vec(conn);
                    res.push((
                        PGON,
                        Connectivity::new_poly(conn.to_shared(), offsets.to_shared()),
                    ));
                }
                _ => {
                    todo!()
                }
            },
            _ => todo!(), // For other types, return empty vector
        };
        res
    }

    /// Decomposes the element into simplex elements.
    ///
    /// Returns a list of (element type, connectivity) tuples representing
    /// the simplex decomposition. For example, a QUAD4 is decomposed into
    /// two TRI3 elements.
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
}

impl<'a, T> ElementTopo<'a> for T where T: ElementLike<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Element, ElementType};
    use ndarray as nd;
    use std::collections::BTreeMap;

    #[test]
    fn test_subentities_quad4_codim1() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::QUAD4);
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D1));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 4 edges
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::SEG2);
        // Check that connectivity contains 4 edges (4 x 2 nodes = 8 values)
        assert_eq!(connectivity.len(), 4);
    }

    #[test]
    fn test_subentities_quad4_codim2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::QUAD4);
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D2));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 4 vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 4);
    }

    #[test]
    fn test_subentities_tri3_codim1() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::TRI3);
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D1));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 3 edges
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::SEG2);
        assert_eq!(connectivity.len(), 3);
    }

    #[test]
    fn test_subentities_tri3_codim2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::TRI3);
        let subentities = elem.subentities(Some(crate::mesh::Dimension::D2));
        assert_eq!(subentities.len(), 1); // One Connectivity containing all 3 vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 3);
    }

    #[test]
    fn test_subentities_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::SEG2);
        let subentities = elem.subentities(None); // defaults to D1
        assert_eq!(subentities.len(), 1); // One Connectivity containing both vertices
        let (et, connectivity) = &subentities[0];
        assert_eq!(*et, ElementType::VERTEX);
        assert_eq!(connectivity.len(), 2);
    }

    #[test]
    fn test_to_simplexes_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::QUAD4);
        let simplexes = elem.to_simplexes();
        assert_eq!(simplexes.len(), 2); // QUAD4 -> 2 TRI3
        for (et, _) in &simplexes {
            assert_eq!(*et, ElementType::TRI3);
        }
    }

    #[test]
    fn test_to_simplexes_tri3() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(0, coords.view(), None, &family, &groups, conn, ElementType::TRI3);
        let simplexes = elem.to_simplexes();
        assert_eq!(simplexes.len(), 1); // TRI3 -> 1 TRI3
        assert_eq!(simplexes[0].0, ElementType::TRI3);
    }
}
