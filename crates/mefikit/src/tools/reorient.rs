//! Mesh reorientation: repair every element's winding to the canonical convention.
//!
//! Cell-level repair is delegated to [`ElementGeo::oriented_positive_connectivity`] for the
//! linear cell types it understands:
//! - 2D cells (TRI3, QUAD4, PGON) are rewound counter-clockwise in the xy plane
//!   (`signed_area2` tells the sign, so the fix is translation/rotation independent);
//! - TET4 and HEX8 are given positive signed volume.
//!
//! Polyhedral (PHED) cells cannot be handled by that linear fixer (their node lists do not map
//! one-to-one onto a regular base shape), so they are reoriented face by face with the same
//! test as [`Polyhedron::into_ccw`]: each face whose Newell normal points towards the cell's
//! vertex centroid is reversed, so every face is CCW when viewed from outside. Faces whose
//! plane contains the centroid reference point are left untouched (degenerate guard) instead of
//! panicking, keeping the repair total on pathological input.
//!
//! Cells with no well-defined winding in the mesh's space dimension are copied unchanged
//! (higher-order cells, 1D/0D cells, and 2D cells embedded in a 3D mesh).
//!
//! The operation is out-of-place and block-preserving: for every input element block a block
//! with the same element type, count and element order is produced, so families, fields and
//! groups can be carried over as-is — `Arc`-cheap when the input is already a [`UMesh`].

use crate::element_traits::{ElementGeo, ElementTopo};
use crate::geometry::face_plane;
use crate::mesh::{
    Connectivity, ConnectivityView, Dimension, Element, ElementBlock, ElementLike, ElementType,
    UMesh, UMeshView,
};
use ndarray as nd;
use rustc_hash::FxHashMap;

/// Whether `et` is a linear cell with a well-defined winding in `space_dim`.
fn fixable(et: ElementType, space_dim: usize) -> bool {
    match et {
        ElementType::TRI3 | ElementType::QUAD4 | ElementType::PGON => space_dim == 2,
        ElementType::TET4 | ElementType::HEX8 => space_dim == 3,
        _ => false,
    }
}

/// The reoriented node list for one cell, or the original list when there is no well-defined
/// winding in `space_dim`.
fn reoriented_row(elem: &Element, space_dim: usize) -> Vec<usize> {
    if fixable(elem.element_type(), space_dim) {
        elem.oriented_positive_connectivity().1
    } else {
        elem.connectivity().to_vec()
    }
}

/// Returns the flat PHED cell connectivity with all faces CCW from outside, sentinels included.
fn reoriented_phed(elem: &Element, coords: nd::ArrayView2<f64>) -> Vec<usize> {
    let (_, face_conn) = &elem.subentities(Some(Dimension::D1))[0];
    let faces: Vec<Vec<usize>> = face_conn.iter().map(|f| f.to_vec()).collect();

    // The cell's distinct nodes, mapped to consecutive point indices (`into_ccw` works on a
    // depacked point list, so the same compaction is applied here).
    let mut nodes: Vec<usize> = faces.iter().flatten().copied().collect();
    nodes.sort_unstable();
    nodes.dedup();
    let mut local_of: FxHashMap<usize, usize> = FxHashMap::default();
    for (i, &n) in nodes.iter().enumerate() {
        local_of.insert(n, i);
    }
    let pts: Vec<[f64; 3]> = nodes
        .iter()
        .map(|&n| coords.row(n).to_slice().unwrap().try_into().unwrap())
        .collect();

    let refp: [f64; 3] = {
        let mut c = [0.0; 3];
        for p in &pts {
            for k in 0..3 {
                c[k] += p[k];
            }
        }
        let n = pts.len() as f64;
        [c[0] / n, c[1] / n, c[2] / n]
    };

    let mut out: Vec<usize> = Vec::with_capacity(face_conn.len() + faces.len());
    for face in &faces {
        let local: Vec<usize> = face.iter().map(|n| local_of[n]).collect();
        let (nvec, _d) = face_plane(&pts, &local);
        let p0 = pts[local[0]];
        let dot =
            nvec[0] * (p0[0] - refp[0]) + nvec[1] * (p0[1] - refp[1]) + nvec[2] * (p0[2] - refp[2]);
        let scale = nvec[0].abs().max(nvec[1].abs()).max(nvec[2].abs());
        // Degenerate faces (centroid lying on the face plane) are left untouched rather than
        // guessed at; see `Polyhedron::into_ccw` for the non-degenerate form of this test.
        if dot < -1e-12 * scale {
            out.extend(face.iter().rev());
        } else {
            out.extend_from_slice(face);
        }
        out.push(usize::MAX);
    }
    out.pop();
    out
}

/// Rebuilt connectivity for a poly block (PGON face lists, or PHED face lists with sentinels).
fn poly_parts<'a>(
    elements: impl Iterator<Item = Element<'a>>,
    coords: nd::ArrayView2<f64>,
    space_dim: usize,
    len: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut data: Vec<usize> = Vec::new();
    let mut offsets: Vec<usize> = Vec::with_capacity(len);
    for elem in elements {
        if elem.element_type() == ElementType::PHED && space_dim == 3 {
            data.extend(reoriented_phed(&elem, coords));
        } else {
            data.extend(reoriented_row(&elem, space_dim));
        }
        offsets.push(data.len());
    }
    (data, offsets)
}

/// Rebuilt connectivity for a regular block (each cell a flat node list).
fn regular_conn<'a>(
    elements: impl Iterator<Item = Element<'a>>,
    space_dim: usize,
    len: usize,
    ncols: usize,
) -> nd::ArcArray2<usize> {
    let mut rows: Vec<usize> = Vec::with_capacity(len * ncols);
    for elem in elements {
        rows.extend(reoriented_row(&elem, space_dim));
    }
    nd::Array2::from_shape_vec((len, ncols), rows)
        .expect("shape mismatch while rebuilding regular connectivity")
        .into_shared()
}

fn reorient_owned(mesh: &UMesh) -> UMesh {
    let coords = mesh.coords.clone();
    let space_dim = mesh.space_dimension();
    let mut out = UMesh::new(coords.clone());
    for (&et, block) in mesh.blocks() {
        let conn = match &block.connectivity {
            Connectivity::Regular(arr) => Connectivity::Regular(regular_conn(
                block.iter(coords.view()),
                space_dim,
                block.len(),
                arr.ncols(),
            )),
            Connectivity::Poly(_) => {
                let (data, offsets) = poly_parts(
                    block.iter(coords.view()),
                    coords.view(),
                    space_dim,
                    block.len(),
                );
                Connectivity::new_poly(
                    nd::Array1::from_vec(data).into_shared(),
                    nd::Array1::from_vec(offsets).into_shared(),
                )
            }
        };
        out.insert_block(ElementBlock::new_with_metadata(
            et,
            conn,
            block.families_owned(),
            block.fields.clone(),
            block.arc_groups().clone(),
        ));
    }
    out
}

fn reorient_view(mesh: &UMeshView) -> UMesh {
    let coords = mesh.coords().to_shared();
    let space_dim = mesh.space_dimension();
    let mut out = UMesh::new(coords.clone());
    for (&et, block) in mesh.blocks() {
        let conn = match &block.connectivity {
            ConnectivityView::Regular(arr) => Connectivity::Regular(regular_conn(
                block.iter(coords.view()),
                space_dim,
                block.len(),
                arr.ncols(),
            )),
            ConnectivityView::Poly(_) => {
                let (data, offsets) = poly_parts(
                    block.iter(coords.view()),
                    coords.view(),
                    space_dim,
                    block.len(),
                );
                Connectivity::new_poly(
                    nd::Array1::from_vec(data).into_shared(),
                    nd::Array1::from_vec(offsets).into_shared(),
                )
            }
        };
        let fields = block
            .fields
            .iter()
            .map(|(n, f)| (n.clone(), f.to_owned().into_shared()))
            .collect();
        out.insert_block(ElementBlock::new_with_metadata(
            et,
            conn,
            block.families().to_owned().into_shared(),
            fields,
            block.arc_groups().clone(),
        ));
    }
    out
}

/// Reorients every element of the mesh, returning a new mesh.
///
/// Out of place and block-preserving: each input block yields a block with the same element
/// type, count and order, so families, fields and groups are carried over unchanged.
///
/// This free function works on a borrowed view, so it cannot share the underlying storage and
/// necessarily copies the coordinates, families and field data into the returned mesh. Callers
/// that own a [`UMesh`] should use [`Reorientable::reorient`] instead, which carries that
/// metadata over with cheap `Arc` clones (only the connectivity arrays are rebuilt).
pub fn reorient(mesh: &UMeshView) -> UMesh {
    reorient_view(mesh)
}

/// High-level mesh reorientation.
pub trait Reorientable {
    /// Returns a new mesh whose elements all follow the canonical winding
    /// (positive signed volume for 3D, counter-clockwise for 2D), with blocks, fields and
    /// groups preserved.
    fn reorient(&self) -> UMesh;
}

impl Reorientable for UMesh {
    fn reorient(&self) -> UMesh {
        reorient_owned(self)
    }
}

impl Reorientable for UMeshView<'_> {
    fn reorient(&self) -> UMesh {
        reorient_view(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_traits::{ElementGeo, ElementTopo};
    use crate::mesh::{ArcGroups, ElementId, ElementLike};
    use approx::assert_abs_diff_eq;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    fn grid(axes: Vec<Vec<f64>>) -> UMesh {
        crate::tools::grid::RegularUMeshBuilder::new()
            .add_axis(axes[0].clone())
            .add_axis(axes[1].clone())
            .add_axis(axes[2].clone())
            .build()
    }

    fn hexa_total_measure3(mesh: &UMesh) -> f64 {
        let mut total = 0.0;
        for (&et, block) in mesh.blocks() {
            if et == ElementType::HEX8 {
                for elem in block.iter(mesh.coords().view()) {
                    total += elem.measure3();
                }
            }
        }
        total
    }

    fn connectivity_snapshot(mesh: &UMesh) -> Vec<(ElementType, Vec<Vec<usize>>)> {
        mesh.blocks()
            .map(|(&et, b)| {
                (
                    et,
                    b.iter(mesh.coords().view())
                        .map(|e| e.connectivity().to_vec())
                        .collect(),
                )
            })
            .collect()
    }

    #[test]
    fn mirrored_axis_mesh_is_fixed_to_canonical() {
        let mut axes: Vec<Vec<f64>> = vec![
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
        ];
        axes[1].reverse();
        let mesh = grid(axes);
        assert!(
            hexa_total_measure3(&mesh) < 0.0,
            "reversed axis mirrors the mesh"
        );
        assert_abs_diff_eq!(hexa_total_measure3(&mesh.reorient()), 1.0, epsilon = 1e-12);
        // Idempotent: a second pass must be a no-op.
        let once = mesh.reorient();
        assert_eq!(
            connectivity_snapshot(&once),
            connectivity_snapshot(&once.reorient())
        );
    }

    #[test]
    fn ascending_grid_is_unchanged() {
        let axes: Vec<Vec<f64>> = vec![
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
        ];
        let mesh = grid(axes);
        assert_abs_diff_eq!(hexa_total_measure3(&mesh), 1.0, epsilon = 1e-12);
        assert_eq!(
            connectivity_snapshot(&mesh),
            connectivity_snapshot(&mesh.reorient())
        );
    }

    #[test]
    fn reversed_grid_through_polyze_unpolyze_is_repaired() {
        // polyze + (topological) unpolyze preserves the (mirrored) winding, so the resulting
        // HEX8 cells are left-handed; reorient must repair them without knowing the origin.
        let mut axes: Vec<Vec<f64>> = vec![
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
            (0..=1).map(|i| i as f64).collect(),
        ];
        axes[2].reverse();
        let mesh = grid(axes);
        let lefty = crate::tools::polyze::unpolyze(&mesh.view()).unwrap();
        assert!(hexa_total_measure3(&lefty) < 0.0);
        assert_abs_diff_eq!(hexa_total_measure3(&lefty.reorient()), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn cw_2d_cells_are_rewound_ccw() {
        let coords =
            nd::array![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]].into_shared();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 3, 2, 1]]).to_shared(),
            None,
        );
        mesh.add_regular_block(ElementType::TRI3, nd::arr2(&[[1, 4, 2]]).to_shared(), None);
        mesh.add_element(ElementType::PGON, &[0, 3, 2, 1], None, None);

        let fixed = Reorientable::reorient(&mesh);
        let mut node_sets: Vec<(ElementType, Vec<usize>)> = Vec::new();
        for (&et, block) in mesh.blocks() {
            for elem in block.iter(mesh.coords().view()) {
                let mut nodes = elem.connectivity().to_vec();
                nodes.sort_unstable();
                node_sets.push((et, nodes));
            }
        }
        for (&et, block) in fixed.blocks() {
            for elem in block.iter(fixed.coords().view()) {
                // The fixer must consider every output cell already positively wound.
                assert_eq!(
                    elem.oriented_positive_connectivity().1,
                    elem.connectivity().to_vec(),
                    "{et:?} cell not ccw after reorient"
                );
                let mut nodes = elem.connectivity().to_vec();
                nodes.sort_unstable();
                let pos = node_sets.iter().position(|(e, n)| *e == et && *n == nodes);
                assert!(pos.is_some(), "{et:?} lost a cell node set");
                node_sets.remove(pos.unwrap());
            }
        }
        assert!(node_sets.is_empty(), "reorient changed the cell count");
    }

    #[test]
    fn phed_inward_face_is_repaired() {
        // Translated unit cube with the front face wound inward; the signed volume is
        // then not the canonical +1 (the divergent-theorem sum is orientation-sensitive).
        let coords = nd::array![
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
            [2.0, 1.0, 2.0],
            [2.0, 2.0, 2.0],
            [1.0, 2.0, 2.0]
        ]
        .into_shared();
        let mut mesh = UMesh::new(coords.clone());
        mesh.add_element(
            ElementType::PHED,
            &[
                0,
                3,
                2,
                1,
                usize::MAX,
                4,
                5,
                6,
                7,
                usize::MAX,
                4,
                5,
                1,
                0,
                usize::MAX,
                1,
                2,
                6,
                5,
                usize::MAX,
                2,
                3,
                7,
                6,
                usize::MAX,
                3,
                0,
                4,
                7,
            ],
            None,
            None,
        );

        let fixed = reorient(&mesh.view());
        let (_, out_block) = fixed.blocks().next().unwrap();
        let c = fixed.coords();
        let elem = out_block.iter(c).next().unwrap();
        assert_abs_diff_eq!(elem.measure3(), 1.0, epsilon = 1e-12);

        // Face node sets are preserved; only their orientation may change.
        let faces: Vec<Vec<usize>> = elem.subentities(Some(Dimension::D1))[0]
            .1
            .iter()
            .map(|f| {
                let mut s = f.to_vec();
                s.sort_unstable();
                s
            })
            .collect();
        let expected: Vec<Vec<usize>> = [
            vec![0, 1, 2, 3],
            vec![4, 5, 6, 7],
            vec![0, 1, 4, 5],
            vec![1, 2, 5, 6],
            vec![2, 3, 6, 7],
            vec![0, 3, 4, 7],
        ]
        .into_iter()
        .map(|mut f| {
            f.sort_unstable();
            f
        })
        .collect();
        assert_eq!(faces, expected);
    }

    #[test]
    fn owned_reorient_shares_metadata() {
        // `Reorientable for UMesh` must not reallocate anything but the connectivity: the
        // coords, families, fields and groups of the result point at the same data.
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ]
        .into_shared();
        let mut mesh = UMesh::new(coords);
        let fields = {
            let mut f: BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>> = BTreeMap::new();
            f.insert(
                "temperature".to_string(),
                nd::arr1(&[21.5, 22.0, 23.0]).into_dyn().into_shared(),
            );
            f
        };
        mesh.insert_block(ElementBlock::new_with_metadata(
            ElementType::HEX8,
            Connectivity::Regular(nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared()),
            nd::arr1(&[3usize, 4, 5]).into_shared(),
            fields,
            ArcGroups::default(),
        ));
        let mut ids = crate::mesh::ElementIds::new();
        ids.add(ElementType::HEX8, 0);
        mesh.add_to_group("g", &ids);

        let (_, src_block) = mesh.blocks().next().unwrap();
        let src_coords = mesh.coords().as_ptr();
        let src_fams = src_block.families().as_ptr();
        let src_field = src_block.fields["temperature"].as_ptr();
        let src_groups = src_block.arc_groups();

        let fixed = mesh.reorient();
        let (_, out_block) = fixed.blocks().next().unwrap();
        assert_eq!(fixed.coords().as_ptr(), src_coords);
        assert_eq!(out_block.families().as_ptr(), src_fams);
        assert_eq!(out_block.fields["temperature"].as_ptr(), src_field);
        assert_eq!(
            Arc::as_ptr(&out_block.arc_groups().0),
            Arc::as_ptr(&src_groups.0)
        );
    }

    #[test]
    fn metadata_is_preserved() {
        let coords = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0]
        ]
        .into_shared();
        let fields = {
            let mut f: BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>> = BTreeMap::new();
            f.insert(
                "temperature".to_string(),
                nd::arr1(&[21.5]).into_dyn().into_shared(),
            );
            f
        };
        let mut mesh = UMesh::new(coords);
        mesh.insert_block(ElementBlock::new_with_metadata(
            ElementType::HEX8,
            Connectivity::Regular(nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared()),
            nd::arr1(&[0usize]).into_shared(),
            fields,
            ArcGroups::default(),
        ));
        let mut ids = crate::mesh::ElementIds::new();
        ids.add(ElementType::HEX8, 0);
        mesh.add_to_group("heated", &ids);

        let fixed = mesh.reorient();
        assert!(fixed.in_group(ElementId::new(ElementType::HEX8, 0), "heated"));
        let f = fixed.field("temperature", None).unwrap();
        assert_abs_diff_eq!(f.0[&ElementType::HEX8][[0]], 21.5);
    }
}
