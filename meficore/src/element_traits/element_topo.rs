use ndarray::prelude::*;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::mesh::Connectivity;
use crate::mesh::{Dimension, ElementLike, ElementType};

pub trait ElementTopo<'a>: ElementLike<'a> {
    /// This function returns the subentities of the element based on the codimension.
    /// The returned type is a vec because in case of prism elements, faces have different element
    /// types.
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
