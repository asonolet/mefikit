//! Piecewise-constant transfer: each target cell copies the value of the source cell containing
//! its sampling point.
//!
//! This is a pure sample of the containing source cell: `FieldNature` does not affect the
//! transferred value (extensive conservation requires a measure-weighted scheme). The cell location
//! precompute lives in this module ([`prepare`] and its helpers) and the apply-time sparse product
//! is handled by the shared [`super::operator::TransferOperator`].

use std::collections::BTreeMap;

use ndarray as nd;

use super::operator::{RowSparse, TransferMethod, TransferOperator, validated_dims};
use super::transfer_trait::PointLocation;
use crate::element_traits::{ElementGeo, ElementTopo};
use crate::geometry::{Polygon, point_in_phed};
use crate::mesh::{Dimension, Element, ElementId, ElementIds, ElementLike, ElementType, UMeshView};
use crate::tools::spatial_index::{SpIdx2, SpIdx3, SpatiallyIndexable};

/// Builds the piecewise-constant operator: each target cell is located in the source mesh
/// (of the source's topological dimension) through its sampling point, chosen with `point`.
///
/// # Panics
///
/// - If `source` and `target` do not live in the same space dimension.
/// - If `source` is not full-dimensional (its topological dimension must match its space
///   dimension so that its cells define regions).
/// - If the space dimension is neither 2 nor 3.
pub(crate) fn prepare(
    source: &UMeshView,
    target: &UMeshView,
    point: &PointLocation,
) -> TransferOperator {
    let (src_dim, tgt_dim, src_space) = validated_dims(source, target, "Constant piecewise");
    let src_dim_usize = u8::from(src_dim) as usize;
    assert_eq!(
        src_dim_usize, src_space,
        "Source mesh should be full-dimensional (topological dimension = space dimension), got topological {src_dim:?} in a {src_space}D space"
    );

    let index = match src_space {
        2 => SpIndex::D2(source.bvh2()),
        3 => SpIndex::D3(source.bvh3()),
        _ => unreachable!(),
    };

    let mut located: BTreeMap<ElementType, Vec<Option<ElementId>>> = BTreeMap::new();
    for elem in target.elements_of_dim(tgt_dim) {
        let sample = sampling_point(&elem, *point, src_space);
        located
            .entry(elem.element_type())
            .or_default()
            .push(locate(source, &index, src_dim, src_space, sample));
    }

    // Flatten the source cells in BTreeMap element-type order, so the global index stored in
    // the rows matches the concatenation of the field arrays done at apply time.
    let mut src_offsets: BTreeMap<ElementType, usize> = BTreeMap::new();
    let mut n_src = 0;
    for (et, block) in source.blocks() {
        if u8::from(et.dimension()) as usize == src_space {
            src_offsets.insert(*et, n_src);
            n_src += block.len();
        }
    }

    let mut data = Vec::new();
    for (tgt_et, ids) in located {
        let n = ids.len();
        let mut row_ptr = vec![0];
        let mut src_idx = Vec::new();
        let mut weights = Vec::new();
        for id in &ids {
            if let Some(id) = id {
                src_idx.push(src_offsets[&id.element_type()] + id.index());
                weights.push(1.0);
            }
            row_ptr.push(src_idx.len());
        }
        data.push((
            tgt_et,
            RowSparse {
                target_measure: nd::Array1::ones(n),
                row_ptr,
                src_idx,
                weights,
            },
        ));
    }

    TransferOperator::build(
        TransferMethod::ConstantPiecewise {
            point_location: *point,
        },
        src_dim,
        tgt_dim,
        n_src,
        data,
    )
}

/// Computes the sampling point of an element according to `point`.
fn sampling_point(elem: &Element, point: PointLocation, space_dim: usize) -> [f64; 3] {
    match point {
        PointLocation::Centroid => match space_dim {
            2 => {
                let c = elem.centroid2();
                [c[0], c[1], 0.0]
            }
            3 => elem.centroid3(),
            _ => unreachable!(),
        },
        PointLocation::Barycenter => todo!("PointLocation::Barycenter is not implemented yet"),
        PointLocation::StrictInterior => {
            todo!("PointLocation::StrictInterior is not implemented yet")
        }
    }
}

/// A spatial index over the source elements, kept alive for the whole precompute so the
/// underlying BVH is built exactly once instead of once per target element.
enum SpIndex {
    D2(SpIdx2),
    D3(SpIdx3),
}

impl SpIndex {
    fn intersects(&self, sample: [f64; 3]) -> ElementIds {
        match self {
            Self::D2(idx) => idx.intersects([sample[0], sample[1]]),
            Self::D3(idx) => idx.intersects(sample),
        }
    }
}

/// Locates the source cell containing `sample`, using a spatial index refined by an exact
/// containment test. Ties (a point exactly on a shared boundary) are broken by the smallest
/// element id.
fn locate(
    src: &UMeshView,
    index: &SpIndex,
    src_dim: Dimension,
    space_dim: usize,
    sample: [f64; 3],
) -> Option<ElementId> {
    let candidates = index.intersects(sample);
    let mut best: Option<ElementId> = None;
    for (et, indices) in candidates.0 {
        if et.dimension() != src_dim {
            continue;
        }
        for &index in &indices {
            let id = ElementId::new(et, index);
            let elem = src.element(id);
            if contains_point(&elem, sample, space_dim) {
                let replace = match best {
                    None => true,
                    Some(best_id) => id < best_id,
                };
                if replace {
                    best = Some(id);
                }
            }
        }
    }
    best
}

/// Returns `true` if `sample` lies inside the element.
fn contains_point(elem: &Element, sample: [f64; 3], space_dim: usize) -> bool {
    match space_dim {
        2 => {
            let pgon =
                Polygon::unknown((0..elem.connectivity().len()).map(|i| elem.coord2(i).into()))
                    .into_ccw();
            pgon.contains_stable(&[sample[0], sample[1]])
        }
        // TODO: this does not work for polyhedra as coords size does not match connectivity size.
        3 => {
            let coords: Vec<[f64; 3]> = elem.coords3().copied().collect();
            // `coords3()` skips the `usize::MAX` face separators, so the local index is the
            // position among the non-separator connectivity entries -- NOT the raw flat index.
            let local: BTreeMap<usize, usize> = elem
                .connectivity()
                .iter()
                .filter(|&&n| n != usize::MAX)
                .enumerate()
                .map(|(i, &n)| (n, i))
                .collect();
            let mut faces: Vec<usize> = Vec::new();
            for (_, face_conn) in elem.subentities(Some(Dimension::D1)) {
                for face in face_conn.iter() {
                    faces.extend(face.iter().map(|&node| local[&node]));
                    faces.push(usize::MAX);
                }
            }
            point_in_phed(&sample, &coords, &faces)
        }
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use ndarray as nd;

    use crate::element_traits::ElementGeo;
    use crate::mesh::{ElementType, FieldOwnedD, FieldViewD, UMesh};
    use crate::mesh_examples as me;
    use crate::tools::transfer::transfer_trait::{FieldNature, Transfer};

    fn source_with_field(values: nd::Array<f64, nd::IxDyn>) -> UMesh {
        let mut source = me::make_imesh_2d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(ElementType::QUAD4, values)]));
        source.update_field("f", field.into_shared());
        source
    }

    fn field_view(source: &UMesh) -> FieldViewD<'_> {
        source.field("f", Some(Dimension::D2)).unwrap()
    }

    /// Builds a centroid-sampling piecewise-constant operator for the common 2D test meshes.
    fn cpw(source: &UMeshView, target: &UMeshView) -> TransferOperator {
        TransferOperator::new(
            source,
            target,
            TransferMethod::ConstantPiecewise {
                point_location: PointLocation::Centroid,
            },
        )
    }

    /// A constant intensive field is sampled on each target cell.
    #[test]
    fn transfer_constant_intensive() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(4);
        let op = cpw(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr.shape(), &[16]);
        assert!(arr.iter().all(|&v| v == 7.0));
    }

    /// Each target cell receives the value of the source cell containing its centroid.
    #[test]
    fn transfer_two_cells() {
        let coords = nd::ArcArray2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let mut source = UMesh::new(coords);
        source.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 4, 3], [1, 2, 5, 4]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![1.0, 2.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let target = me::make_imesh_2d(4);
        let op = cpw(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, -1.0);
        let arr = &field.0[&ElementType::QUAD4];
        for (i, elem) in target.elements_of_dim(Dimension::D2).enumerate() {
            let centroid = elem.centroid2();
            let expected = if centroid[0] < 0.5 { 1.0 } else { 2.0 };
            assert_eq!(
                arr[i], expected,
                "target element {i} with centroid {centroid:?}"
            );
        }
    }

    /// Target cells not covered by the source mesh get the default value.
    #[test]
    fn transfer_default_uncovered() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let coords = nd::ArcArray2::from_shape_vec(
            (9, 2),
            vec![
                0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 2.0,
                2.0, 2.0,
            ],
        )
        .unwrap();
        let mut target = UMesh::new(coords);
        target.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6], [4, 5, 8, 7]]).to_shared(),
            None,
        );
        let op = cpw(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 99.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr[0], 7.0);
        assert_eq!(arr[1], 99.0);
        assert_eq!(arr[2], 99.0);
        assert_eq!(arr[3], 99.0);
    }

    /// The same operator can evaluate several fields without rebuilding the precompute.
    #[test]
    fn transfer_reuse() {
        let mut source = me::make_imesh_2d(1);
        let f1 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![7.0].into_dyn(),
        )]));
        let f2 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![3.0].into_dyn(),
        )]));
        source.update_field("f1", f1.into_shared());
        source.update_field("f2", f2.into_shared());
        let target = me::make_imesh_2d(2);
        let op = cpw(&source.view(), &target.view());
        let r1 = op.apply(
            &source.field("f1", Some(Dimension::D2)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        let r2 = op.apply(
            &source.field("f2", Some(Dimension::D2)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert!(r1.0[&ElementType::QUAD4].iter().all(|&v| v == 7.0));
        assert!(r2.0[&ElementType::QUAD4].iter().all(|&v| v == 3.0));
    }

    /// `apply_update` stores the transferred field on the target mesh.
    #[test]
    fn transfer_apply_update() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let mut target = me::make_imesh_2d(2);
        let op = cpw(&source.view(), &target.view());
        let old = op.apply_update(
            &mut target,
            "transferred",
            &field_view(&source),
            FieldNature::Intensive,
            0.0,
        );
        assert!(old.is_none());
        let field = target.field("transferred", Some(Dimension::D2)).unwrap();
        assert!(field.0[&ElementType::QUAD4].iter().all(|&v| v == 7.0));
    }

    /// A field on a 3D volume mesh is transferred onto a 2D manifold in 3D space.
    #[test]
    fn transfer_3d_downcast() {
        let coords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut source = UMesh::new(coords);
        source.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::TET4,
            nd::array![5.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let tcoords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
        )
        .unwrap();
        let mut target = UMesh::new(tcoords);
        target.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let op = cpw(&source.view(), &target.view());
        let field = op.apply(
            &source.field("f", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr.iter().copied().collect::<Vec<f64>>(), vec![5.0]);
    }

    /// Feeding meshes with different space dimensions fails with a clear message.
    #[test]
    #[should_panic(expected = "same space dimension")]
    fn transfer_space_dim_mismatch_panics() {
        let source = me::make_imesh_2d(1);
        let target = me::make_imesh_3d(1);
        let _ = cpw(&source.view(), &target.view());
    }

    /// A PHED source whose connectivity repeats vertices across faces is sampled correctly.
    /// (Regression: the face-local index table used the raw flat-connectivity position, which
    /// exceeds the compressed vertex list for late repeats and index-panics.)
    #[test]
    fn transfer_phed_hex_seed_source() {
        let coords = nd::ArcArray2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut source = UMesh::new(coords);
        // Face connectivity of the unit cube [0,1]^3, polyhedron convention (MED faces):
        let m = usize::MAX;
        source.add_element(
            ElementType::PHED,
            &[
                0, 3, 2, 1, m, 4, 5, 6, 7, m, 0, 1, 5, 4, m, 1, 2, 6, 5, m, 2, 3, 7, 6, m, 3, 0, 4,
                7,
            ],
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::PHED,
            nd::array![5.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        // Target: the unit cube split into 8 HEX8 cells.
        let target = me::make_imesh_3d(2);
        let op = cpw(&source.view(), &target.view());
        let field = op.apply(
            &source.field("f", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        let arr = &field.0[&ElementType::HEX8];
        assert_eq!(arr.len(), target.num_elements());
        assert!(arr.iter().all(|&v| v == 5.0));
    }
}
