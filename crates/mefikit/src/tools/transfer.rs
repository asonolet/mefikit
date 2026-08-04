//! Field transfer between meshes.
//!
//! This module provides the [`Transfer`] trait and its implementations for transferring
//! per-element fields from a source mesh to the cells of a target mesh. The geometry-only
//! precompute (locating each target cell in the source mesh) is performed once when a transfer
//! operator is constructed, so the same operator can be reused to evaluate many fields (for
//! example across time steps) as long as the meshes do not change.
//!
//! A transfer may downcast: a field defined on the cells of a full-dimensional source mesh can
//! be transferred onto the cells of a lower-dimensional target mesh embedded in the same space
//! (for example a 3D volume mesh onto a 2D manifold in 3D space).

use std::collections::BTreeMap;

use ndarray as nd;

use crate::element_traits::{ElementGeo, ElementTopo, is_in};
use crate::mesh::{
    Dimension, Element, ElementId, ElementIds, ElementLike, ElementType, FieldArcD, FieldOwnedD,
    FieldViewD, UMesh, UMeshView,
};
use crate::tools::spatial_index::{SpIdx2, SpIdx3, SpatiallyIndexable};

/// Nature of a field, governing how its values behave when the supporting cells change.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldNature {
    /// Per-unit-measure field (temperature, density, pressure).
    Intensive,
    /// Total-quantity field (mass, energy).
    Extensive,
}

/// How each target cell's representative sampling point is chosen (fixed at precompute time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointLocation {
    /// Arithmetic mean of the cell's vertices.
    Centroid,
    /// True region centroid (center of mass).
    Barycenter,
    /// A strictly-interior point.
    StrictInterior,
}

/// A geometry-only precompute for transferring a field from source cells to target cells.
pub trait Transfer {
    /// Evaluates `field` (defined on the source cells) at the resolved locations.
    ///
    /// `default` is used for target cells that are not covered by the source mesh. The
    /// `field_nature` parameter lets schemes distinguish intensive and extensive fields.
    fn apply(&self, field: &FieldViewD, field_nature: FieldNature, default: f64) -> FieldOwnedD;

    /// Dimension of the target cells that receive values.
    fn tgt_dim(&self) -> Dimension;

    /// Evaluates the field and stores it in `target` under `name`.
    ///
    /// Returns the previous field if it existed, or `None` if it did not.
    fn apply_update(
        &self,
        target: &mut UMesh,
        name: &str,
        field: &FieldViewD,
        field_nature: FieldNature,
        default: f64,
    ) -> Option<FieldArcD>
    where
        Self: Sized,
    {
        let field = self.apply(field, field_nature, default);
        target.update_field(name, field.into_shared(), Some(self.tgt_dim()))
    }
}

/// Piecewise-constant transfer: each target cell copies the value of the source cell containing
/// its sampling point.
///
/// This is a pure sample of the containing source cell: `FieldNature` does not affect the
/// transferred value (extensive conservation requires a measure-weighted scheme).
#[derive(Debug)]
pub struct ConstantPiecewiseTransfer {
    src_dim: Dimension,
    tgt_dim: Dimension,
    mapping: BTreeMap<ElementType, Vec<Option<ElementId>>>,
}

impl ConstantPiecewiseTransfer {
    /// Precomputes the piecewise-constant transfer operator from `source` to `target`.
    ///
    /// Each target cell (of its topological dimension) is located in the source mesh (of the
    /// source's topological dimension) through its sampling point, chosen with `point`.
    ///
    /// # Panics
    ///
    /// - If `source` and `target` do not live in the same space dimension.
    /// - If `source` is not full-dimensional (its topological dimension must match its space
    ///   dimension so that its cells define regions).
    /// - If the space dimension is neither 2 nor 3.
    pub fn new(source: &UMeshView, target: &UMeshView, point: PointLocation) -> Self {
        let src_space = source.space_dimension();
        let tgt_space = target.space_dimension();
        assert_eq!(
            src_space, tgt_space,
            "Source and target meshes should share the same space dimension, got source = {src_space}D and target = {tgt_space}D"
        );
        assert!(
            (2..=3).contains(&src_space),
            "Transfer is only supported in 2D and 3D space, got {src_space}D"
        );
        let src_dim = source
            .topological_dimension()
            .expect("Source mesh should not be empty");
        let tgt_dim = target
            .topological_dimension()
            .expect("Target mesh should not be empty");
        let src_dim_usize = u8::from(src_dim) as usize;
        assert_eq!(
            src_dim_usize, src_space,
            "Source mesh should be full-dimensional (topological dimension = space dimension), got topological {src_dim:?} in a {src_space}D space"
        );

        let index = match src_space {
            2 => SpIndex::D2(source.bvh2()),
            3 => SpIndex::D3(target.bvh3()),
            _ => unreachable!(),
        };

        let mut mapping: BTreeMap<ElementType, Vec<Option<ElementId>>> = BTreeMap::new();
        for elem in target.elements_of_dim(tgt_dim) {
            let sample = sampling_point(&elem, point, src_space);
            let located = locate(source, &index, src_dim, src_space, sample);
            mapping
                .entry(elem.element_type())
                .or_default()
                .push(located);
        }
        Self {
            src_dim,
            tgt_dim,
            mapping,
        }
    }
}

impl Transfer for ConstantPiecewiseTransfer {
    fn apply(&self, field: &FieldViewD, _field_nature: FieldNature, default: f64) -> FieldOwnedD {
        assert_eq!(
            field.dimension(),
            Some(self.src_dim),
            "The field should be defined on the source topological dimension {src_dim:?}, got {got:?}",
            src_dim = self.src_dim,
            got = field.dimension()
        );
        let trailing: Vec<usize> = field
            .0
            .values()
            .next()
            .expect("The field should not be empty")
            .shape()[1..]
            .to_vec();
        let mut result: BTreeMap<ElementType, nd::Array<f64, nd::IxDyn>> = BTreeMap::new();
        for (&tgt_et, located) in &self.mapping {
            let n = located.len();
            let shape: Vec<usize> = std::iter::once(n).chain(trailing.iter().copied()).collect();
            let mut arr = nd::Array::from_elem(nd::IxDyn(&shape), default);
            for (i, located) in located.iter().enumerate() {
                if let Some(id) = located {
                    let src_arr = field.0.get(&id.element_type()).unwrap_or_else(|| {
                        panic!(
                            "The field is missing the source element type {:?} required by the transfer",
                            id.element_type()
                        )
                    });
                    let src_row = src_arr.index_axis(nd::Axis(0), id.index());
                    arr.index_axis_mut(nd::Axis(0), i).assign(&src_row);
                }
            }
            result.insert(tgt_et, arr);
        }
        FieldOwnedD::new(result)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
    }
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
            let mut pgon: Vec<[f64; 2]> = elem.coords2().copied().collect();
            if signed_area2(&pgon) < 0.0 {
                pgon.reverse();
            }
            is_in::in_polygon_stable(&[sample[0], sample[1]], &pgon)
        }
        3 => {
            let coords: Vec<[f64; 3]> = elem.coords3().copied().collect();
            let local: BTreeMap<usize, usize> = elem
                .connectivity()
                .iter()
                .enumerate()
                .map(|(i, &node)| (node, i))
                .collect();
            let mut faces: Vec<usize> = Vec::new();
            for (_, face_conn) in elem.subentities(Some(Dimension::D1)) {
                for face in face_conn.iter() {
                    faces.extend(face.iter().map(|&node| local[&node]));
                    faces.push(usize::MAX);
                }
            }
            is_in::point_in_phed(&sample, &coords, &faces)
        }
        _ => unreachable!(),
    }
}

/// Signed area of a polygon using the shoelace formula.
fn signed_area2(pgon: &[[f64; 2]]) -> f64 {
    let n = pgon.len();
    let mut area2 = 0.0;
    for i in 0..n {
        let [x0, y0] = pgon[i];
        let [x1, y1] = pgon[(i + 1) % n];
        area2 += x0 * y1 - x1 * y0;
    }
    area2 / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, FieldOwnedD};
    use crate::mesh_examples as me;
    use ndarray as nd;

    fn source_with_field(values: nd::Array<f64, nd::IxDyn>) -> UMesh {
        let mut source = me::make_imesh_2d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(ElementType::QUAD4, values)]));
        source.update_field("f", field.into_shared(), Some(Dimension::D2));
        source
    }

    fn field_view(source: &UMesh) -> FieldViewD<'_> {
        source.field("f", Some(Dimension::D2)).unwrap()
    }

    /// A constant intensive field is sampled on each target cell.
    #[test]
    fn transfer_constant_intensive() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(4);
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        source.update_field("f", field.into_shared(), Some(Dimension::D2));
        let target = me::make_imesh_2d(4);
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        source.update_field("f1", f1.into_shared(), Some(Dimension::D2));
        source.update_field("f2", f2.into_shared(), Some(Dimension::D2));
        let target = me::make_imesh_2d(2);
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        source.update_field("f", field.into_shared(), Some(Dimension::D3));
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
        let op =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
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
        let _ =
            ConstantPiecewiseTransfer::new(&source.view(), &target.view(), PointLocation::Centroid);
    }
}
