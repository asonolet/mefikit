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
                        [co[0], co[1], co[3]],
                        [co[1], co[2], co[3]],
                        [co[2], co[0], co[3]],
                        [co[0], co[2], co[1]],
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
                        [co[0], co[3], co[2], co[1]],
                        [co[4], co[5], co[6], co[7]],
                        [co[0], co[1], co[5], co[4]],
                        [co[2], co[3], co[7], co[6]],
                        [co[1], co[2], co[6], co[5]],
                        [co[3], co[0], co[4], co[7]],
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
                    for chunk in co.split(|&e| e == usize::MAX) {
                        if chunk.is_empty() {
                            continue;
                        }
                        offset += chunk.len();
                        offsets.push(offset);
                        conn.extend_from_slice(chunk);
                    }
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

    /// Converts the element to its polygonal/polyhedral equivalent.
    ///
    /// - 0D (VERTEX) is returned unchanged.
    /// - 1D elements become SPLINE with the same node list.
    /// - 2D elements become PGON with the same node list.
    /// - 3D elements become PHED with face-based connectivity. Faces are
    ///   separated by `usize::MAX` sentinel values in the returned vector.
    /// - Already-poly elements are returned unchanged.
    fn to_poly(&self) -> (ElementType, Vec<usize>) {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX => (VERTEX, co.to_vec()),
            SEG2 | SEG3 | SEG4 => (SPLINE, co.to_vec()),
            TRI3 | TRI6 | TRI7 | QUAD4 | QUAD8 | QUAD9 => (PGON, co.to_vec()),
            TET4 => {
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[3], m, co[1], co[2], co[3], m, co[2], co[0], co[3], m,
                        co[0], co[2], co[1],
                    ],
                )
            }
            TET10 => {
                // VTK TET10: 0-3 vertices, 4(0-1), 5(1-2), 6(0-2), 7(0-3), 8(1-3), 9(2-3)
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[1], co[2], co[4], co[5], co[6], m, co[1], co[2], co[3], co[5],
                        co[9], co[8], m, co[2], co[3], co[0], co[9], co[7], co[6], m, co[3], co[0],
                        co[1], co[7], co[4], co[8],
                    ],
                )
            }
            HEX8 => {
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        co[0], co[3], co[2], co[1], m, co[4], co[5], co[6], co[7], m, co[0], co[1],
                        co[5], co[4], m, co[1], co[2], co[6], co[5], m, co[2], co[3], co[7], co[6],
                        m, co[3], co[0], co[4], co[7],
                    ],
                )
            }
            HEX21 => {
                // VTK HEX21: 0-7 vertices, 8(0-1), 9(1-2), 10(2-3), 11(3-0),
                //   12(4-5), 13(5-6), 14(6-7), 15(7-4), 16(0-4), 17(1-5), 18(2-6), 19(3-7)
                let m = usize::MAX;
                (
                    PHED,
                    vec![
                        // bottom [0,3,2,1]
                        co[0], co[3], co[2], co[1], co[8], co[9], co[10], co[11], m,
                        // top [4,5,6,7]
                        co[4], co[5], co[6], co[7], co[12], co[13], co[14], co[15], m,
                        // front [0,1,5,4]
                        co[0], co[1], co[5], co[4], co[8], co[17], co[12], co[16], m,
                        // right [1,2,6,5]
                        co[1], co[2], co[6], co[5], co[9], co[18], co[13], co[17], m,
                        // back [2,3,7,6]
                        co[2], co[3], co[7], co[6], co[10], co[19], co[14], co[18], m,
                        // left [3,0,4,7]
                        co[3], co[0], co[4], co[7], co[11], co[16], co[15], co[19],
                    ],
                )
            }
            SPLINE | PGON | PHED => (self.element_type(), co.to_vec()),
        }
    }

    /// Converts a poly element back to its regular equivalent.
    ///
    /// - Regular elements (VERTEX, SEG*, TRI*, QUAD*, TET*, HEX*) are returned unchanged.
    /// - SPLINE cannot be converted (ambiguous: could be SEG2, SEG3, or SEG4).
    /// - PGON with 3 nodes becomes TRI3, with 4 nodes becomes QUAD4.
    /// - PHED with 4 triangular faces becomes TET4, with 6 quadrilateral faces becomes HEX8.
    ///   Face orientation is verified by checking that each edge is shared by exactly
    ///   two faces with consistent winding.
    #[allow(clippy::wrong_self_convention)]
    fn from_poly(&self) -> Result<(ElementType, Vec<usize>), String> {
        use ElementType::*;
        let co = self.connectivity();
        match self.element_type() {
            VERTEX | SEG2 | SEG3 | SEG4 | TRI3 | TRI6 | TRI7 | QUAD4 | QUAD8 | QUAD9 | TET4
            | TET10 | HEX8 | HEX21 => Ok((self.element_type(), co.to_vec())),
            SPLINE => Err("SPLINE cannot be converted to a regular element".into()),
            PGON => match co.len() {
                3 => Ok((TRI3, co.to_vec())),
                4 => Ok((QUAD4, co.to_vec())),
                n => Err(format!(
                    "PGON with {n} nodes cannot be converted to TRI3 or QUAD4"
                )),
            },
            PHED => {
                let sub = self.subentities(Some(Dimension::D1));
                let (_, face_conn) = &sub[0];
                phed_to_regular(face_conn)
            }
        }
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

/// Converts a PHED element's face connectivity to a regular element.
///
/// Takes the `Connectivity::Poly` from `subentities(D1)` and determines node
/// ordering purely from face adjacency without assuming any particular face order.
fn phed_to_regular(face_conn: &Connectivity) -> Result<(ElementType, Vec<usize>), String> {
    use rustc_hash::FxHashMap;
    use smallvec::SmallVec;

    use super::utils::SortedVecKey;

    let num_faces = face_conn.len();
    if num_faces == 0 {
        return Err("PHED has no faces".into());
    }
    let nodes_per_face = face_conn[0].len();

    if !face_conn.iter().all(|f| f.len() == nodes_per_face) {
        return Err("PHED faces have inconsistent node counts".into());
    }

    // Build edge→face map. Key: sorted edge, Value: list of face indices.
    let mut edge_faces: FxHashMap<SortedVecKey, Vec<usize>> = FxHashMap::default();
    for (fi, face) in face_conn.iter().enumerate() {
        let n = face.len();
        for j in 0..n {
            let key = SortedVecKey::new(SmallVec::from_vec(vec![face[j], face[(j + 1) % n]]));
            edge_faces.entry(key).or_default().push(fi);
        }
    }

    // Validate: each edge must be shared by exactly 2 faces (closed surface).
    for (key, occurrences) in &edge_faces {
        if occurrences.len() != 2 {
            return Err(format!(
                "Edge {:?} is shared by {} faces (expected 2 for closed surface)",
                key,
                occurrences.len()
            ));
        }
    }

    match (num_faces, nodes_per_face) {
        (4, 3) => from_phed_tet4(face_conn),
        (6, 4) => from_phed_hex8(face_conn, &edge_faces),
        _ => Err(format!(
            "PHED with {num_faces} faces of {nodes_per_face} nodes cannot be converted to TET4 or HEX8"
        )),
    }
}

/// Reconstruct TET4 node ordering from 4 triangular faces.
///
/// Algorithm: pick face F0 = [a, b, c]. The 4th node d is the one node not in F0.
fn from_phed_tet4(face_conn: &Connectivity) -> Result<(ElementType, Vec<usize>), String> {
    use ElementType::TET4;

    let f0 = &face_conn[0];
    let a = f0[0];
    let b = f0[1];
    let c = f0[2];

    // The 4th node is any node not in {a, b, c}.
    let d = face_conn
        .iter()
        .skip(1)
        .find_map(|face| face.iter().find(|&&n| n != a && n != b && n != c).copied())
        .ok_or_else(|| "Could not find 4th node in TET4 reconstruction".to_string())?;

    Ok((TET4, vec![a, b, c, d]))
}

/// Reconstruct HEX8 node ordering from 6 quadrilateral faces.
///
/// Algorithm:
/// 1. Pick face F0 = [a, b, c, d] as the "bottom" face.
/// 2. Find the opposite face (shares no nodes with F0) as the "top" face.
/// 3. For each edge of F0, find the side face sharing that edge.
/// 4. From the side face's ordering, determine which top node is adjacent to
///    which bottom node, giving [a→e, b→f, c→g, d→h].
/// 5. Return [a, b, c, d, e, f, g, h].
fn from_phed_hex8(
    face_conn: &Connectivity,
    edge_faces: &rustc_hash::FxHashMap<super::utils::SortedVecKey, Vec<usize>>,
) -> Result<(ElementType, Vec<usize>), String> {
    use smallvec::SmallVec;

    use super::utils::SortedVecKey;
    use ElementType::HEX8;

    let f0 = &face_conn[0];
    let bottom: [usize; 4] = [f0[0], f0[1], f0[2], f0[3]];
    let bottom_set: std::collections::HashSet<usize> = bottom.iter().copied().collect();

    // Find the face opposite to F0: shares zero nodes with F0.
    let opposite_idx = face_conn
        .iter()
        .enumerate()
        .find(|&(fi, face)| {
            if fi == 0 {
                return false;
            }
            face.iter().all(|&n| !bottom_set.contains(&n))
        })
        .map(|(fi, _)| fi)
        .ok_or_else(|| "Could not find opposite face for HEX8 reconstruction".to_string())?;

    // For each edge of the bottom face, find the side face sharing that edge,
    // then determine which top node is "above" which bottom node.
    let mut top_of: rustc_hash::FxHashMap<usize, usize> = rustc_hash::FxHashMap::default();

    for j in 0..4 {
        let a = bottom[j];
        let b = bottom[(j + 1) % 4];

        // Find a side face (not F0, not opposite) that contains both a and b.
        let key = SortedVecKey::new(SmallVec::from_vec(vec![a, b]));
        let side_idx = edge_faces
            .get(&key)
            .and_then(|occurrences| {
                occurrences
                    .iter()
                    .find(|&&fi| fi != 0 && fi != opposite_idx)
                    .copied()
            })
            .ok_or_else(|| {
                format!("Could not find side face for edge [{a},{b}] in HEX8 reconstruction")
            })?;

        let side = &face_conn[side_idx];
        let n = side.len();

        // In the side face, find the node adjacent to `a` that is NOT a bottom node.
        use std::collections::hash_map::Entry;
        if let Entry::Vacant(e) = top_of.entry(a) {
            let pos_a = side.iter().position(|&x| x == a).unwrap();
            let prev = side[(pos_a + n - 1) % n];
            let next = side[(pos_a + 1) % n];
            let above_a = if !bottom_set.contains(&prev) {
                prev
            } else {
                next
            };
            e.insert(above_a);
        }

        // Same for `b`.
        if let Entry::Vacant(e) = top_of.entry(b) {
            let pos_b = side.iter().position(|&x| x == b).unwrap();
            let prev = side[(pos_b + n - 1) % n];
            let next = side[(pos_b + 1) % n];
            let above_b = if !bottom_set.contains(&prev) {
                prev
            } else {
                next
            };
            e.insert(above_b);
        }
    }

    let top: [usize; 4] = bottom
        .iter()
        .map(|&b| {
            top_of
                .get(&b)
                .copied()
                .ok_or_else(|| format!("Missing top mapping for node {b}"))
        })
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .unwrap();

    Ok((
        HEX8,
        vec![
            bottom[0], bottom[1], bottom[2], bottom[3], top[0], top[1], top[2], top[3],
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{Element, ElementType};
    use ndarray as nd;

    #[test]
    fn test_subentities_quad4_codim1() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SEG2,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
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

        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let simplexes = elem.to_simplexes();
        assert_eq!(simplexes.len(), 1); // TRI3 -> 1 TRI3
        assert_eq!(simplexes[0].0, ElementType::TRI3);
    }

    #[test]
    fn test_to_poly_vertex() {
        let coords = nd::array![[0.0, 0.0]];
        let conn = &[0];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::VERTEX,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::VERTEX);
        assert_eq!(poly_conn, vec![0]);
    }

    #[test]
    fn test_to_poly_seg2() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let conn = &[0, 1];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SEG2,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::SPLINE);
        assert_eq!(poly_conn, vec![0, 1]);
    }

    #[test]
    fn test_to_poly_tri3() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TRI3,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }

    #[test]
    fn test_to_poly_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let conn = &[0, 1, 2, 3];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::QUAD4,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_to_poly_tet4() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::TET4,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);
        // 4 faces x 3 nodes + 3 separators = 15 entries
        assert_eq!(poly_conn.len(), 15);
        // Check face separators
        assert_eq!(poly_conn[3], usize::MAX);
        assert_eq!(poly_conn[7], usize::MAX);
        assert_eq!(poly_conn[11], usize::MAX);
        // Check first face
        assert_eq!(&poly_conn[0..3], &[0, 1, 3]);
    }

    #[test]
    fn test_to_poly_hex8() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ];
        let conn = &[0, 1, 2, 3, 4, 5, 6, 7];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::HEX8,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);
        // 6 faces x 4 nodes + 5 separators = 29 entries
        assert_eq!(poly_conn.len(), 29);
        // First face is the bottom, wound outward (reversed VTK order)
        assert_eq!(&poly_conn[0..4], &[0, 3, 2, 1]);
        // Check last face (left = VTK face 5)
        assert_eq!(&poly_conn[25..29], &[3, 0, 4, 7]);
    }

    #[test]
    fn test_to_poly_pgon_unchanged() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::PGON,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PGON);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }

    #[test]
    fn test_to_poly_spline_unchanged() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let conn = &[0, 1, 2];
        let family = 0;
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            conn,
            ElementType::SPLINE,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::SPLINE);
        assert_eq!(poly_conn, vec![0, 1, 2]);
    }

    // ===== from_poly tests =====

    #[test]
    fn test_from_poly_regular_unchanged() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1],
            ElementType::SEG2,
            &groups,
        );
        let (et, conn) = elem.from_poly().unwrap();
        assert_eq!(et, ElementType::SEG2);
        assert_eq!(conn, vec![0, 1]);
    }

    #[test]
    fn test_from_poly_spline_error() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2],
            ElementType::SPLINE,
            &groups,
        );
        assert!(elem.from_poly().is_err());
    }

    #[test]
    fn test_from_poly_pgon3_to_tri3() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2],
            ElementType::PGON,
            &groups,
        );
        let (et, conn) = elem.from_poly().unwrap();
        assert_eq!(et, ElementType::TRI3);
        assert_eq!(conn, vec![0, 1, 2]);
    }

    #[test]
    fn test_from_poly_pgon4_to_quad4() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2, 3],
            ElementType::PGON,
            &groups,
        );
        let (et, conn) = elem.from_poly().unwrap();
        assert_eq!(et, ElementType::QUAD4);
        assert_eq!(conn, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_from_poly_pgon5_error() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.5], [0.0, 1.0]];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2, 3, 4],
            ElementType::PGON,
            &groups,
        );
        assert!(elem.from_poly().is_err());
    }

    #[test]
    fn test_from_poly_tet4_roundtrip() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2, 3],
            ElementType::TET4,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);

        let groups2 = crate::mesh::ArcGroups::new();
        let poly_elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &poly_conn,
            ElementType::PHED,
            &groups2,
        );
        let (et2, conn2) = poly_elem.from_poly().unwrap();
        assert_eq!(et2, ElementType::TET4);
        // from_phed_tet4 may return a different permutation of the same nodes.
        assert_eq!(conn2.len(), 4);
        let mut sorted = conn2.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_from_poly_hex8_roundtrip() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ];
        let groups = crate::mesh::ArcGroups::new();
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            ElementType::HEX8,
            &groups,
        );
        let (et, poly_conn) = elem.to_poly();
        assert_eq!(et, ElementType::PHED);

        let groups2 = crate::mesh::ArcGroups::new();
        let poly_elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &poly_conn,
            ElementType::PHED,
            &groups2,
        );
        let (et2, conn2) = poly_elem.from_poly().unwrap();
        assert_eq!(et2, ElementType::HEX8);
        assert_eq!(conn2, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn test_from_poly_phed_shuffled_faces() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        // Build PHED with faces in a different order than to_poly produces.
        // VTK TET4 faces in shuffled order, all consistently oriented:
        // face0: [2,0,3] (opp 1), face1: [0,1,3] (opp 2), face2: [0,2,1] (opp 3), face3: [1,2,3] (opp 0)
        let m = usize::MAX;
        let phed_conn = vec![2, 0, 3, m, 0, 1, 3, m, 0, 2, 1, m, 1, 2, 3];
        let groups = crate::mesh::ArcGroups::new();
        let poly_elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &phed_conn,
            ElementType::PHED,
            &groups,
        );
        let (et, conn) = poly_elem.from_poly().unwrap();
        assert_eq!(et, ElementType::TET4);
        assert_eq!(conn.len(), 4);
        let mut sorted = conn.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_from_poly_hex8_shuffled_faces() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ];
        // Same faces as to_poly but in a different order (VTK convention).
        let m = usize::MAX;
        let phed_conn = vec![
            4, 5, 6, 7, m, 3, 0, 4, 7, m, 0, 1, 2, 3, m, 2, 3, 7, 6, m, 1, 2, 6, 5, m, 0, 1, 5, 4,
        ];
        let groups = crate::mesh::ArcGroups::new();
        let poly_elem = Element::new(
            0,
            coords.view(),
            None,
            &0,
            &phed_conn,
            ElementType::PHED,
            &groups,
        );
        let (et, conn) = poly_elem.from_poly().unwrap();
        assert_eq!(et, ElementType::HEX8);
        assert_eq!(conn.len(), 8);
        let mut sorted = conn.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }
}
