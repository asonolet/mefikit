//! Inverse-distance (Shepard) interpolation transfer between point clouds.
//!
//! Each target point samples its `k` nearest source points and the value is the weighted average
//! `sum_i w_i f(x_i)` with weights `w_i ∝ 1/r_i^p`, normalized so that `sum_i w_i = 1`. Since the
//! weights are non-negative and sum to one, the transferred value is a convex combination of the
//! source values and can never overshoot or undershoot the range spanned by the selected
//! neighbours; a target point coinciding with a source point reproduces that source value exactly.
//!
//! The validation and the plumbing between the `k`-nearest-neighbours search (in the shared
//! [`super::solver`] module) and the [`super::operator::TransferOperator`] live in this module.

use super::operator::{TransferMethod, TransferOperator, point_interpolation, validated_dims};
use super::solver::{NeighbourScheme, solve_neighbours, source_centroids};
use super::transfer_trait::{FieldNature, Transfer};
use crate::mesh::{Dimension, FieldOwnedD, FieldViewD, UMeshView};

/// Builds the inverse-distance operator over the `k` nearest source points of every target point.
///
/// For each target point the `k` nearest source points are gathered and weighed by
/// `1 / r^exponent`, where `r` is the distance to the target point. Weights are normalized so
/// each row sums to one, which makes the interpolation a convex combination of the source
/// values (no overshoot or undershoot). A target point coinciding with a source point
/// interpolates it exactly.
///
/// # Panics
///
/// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or if it is not 2 or 3.
/// - If `k` is zero or `exponent` is not positive.
pub(crate) fn prepare(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    k: usize,
    exponent: f64,
) -> TransferOperator {
    let (src_dim, tgt_dim, src_space) = validated_dims(mesh_src, mesh_tgt, "Inverse-distance");
    assert!(k > 0, "k should be at least 1");
    assert!(
        exponent > 0.0,
        "exponent should be positive, got {exponent}"
    );
    assert!(
        mesh_src.num_elements() > 0,
        "Source mesh should not be empty"
    );

    let (src_coords, n_src) = source_centroids(mesh_src, src_dim);

    TransferOperator::build(
        TransferMethod::InverseDistance,
        src_dim,
        tgt_dim,
        n_src,
        point_interpolation(mesh_tgt, src_space, |coords| {
            solve_neighbours(
                &src_coords.view(),
                &coords,
                k,
                NeighbourScheme::InverseDistance { exponent },
            )
        }),
    )
}

/// An inverse-distance (Shepard) interpolation transfer between point clouds.
///
/// Each target point samples its `k` nearest source points and the value is the
/// weighted average `sum_i w_i f(x_i)` with weights `w_i ∝ 1/r_i^p`, normalized so
/// that `sum_i w_i = 1`. Since the weights are non-negative and sum to one, the
/// transferred value is a convex combination of the source values and can never
/// overshoot or undershoot the range spanned by the selected neighbours; a target
/// point coinciding with a source point reproduces that source value exactly.
///
/// No local polynomial is fitted (unlike [`super::MovingLeastSquaresTransfer`]),
/// which makes evaluation a plain weighted sum: the whole interpolation is
/// precomputed at construction time, so [`Transfer::apply`] costs a fixed `k`
/// fused multiply-adds per target component. The operator is built from the
/// coordinates only, so it can be reused to evaluate many fields (e.g. across
/// time steps) as long as the point sets do not change.
#[derive(Clone, Debug)]
pub struct InverseDistanceTransfer(pub(crate) TransferOperator);

impl InverseDistanceTransfer {
    /// Builds an inverse-distance interpolation operator from source points to
    /// target points.
    ///
    /// For each target point the `k` nearest source points are gathered and
    /// weighed by `1 / r^exponent`, where `r` is the distance to the target
    /// point. Weights are normalized so each row sums to one, which makes the
    /// interpolation a convex combination of the source values (no overshoot or
    /// undershoot). A target point coinciding with a source point interpolates
    /// it exactly.
    ///
    /// # Panics
    ///
    /// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or
    ///   if it is not 2 or 3.
    /// - If `k` is zero or `exponent` is not positive.
    pub fn new(mesh_src: &UMeshView, mesh_tgt: &UMeshView, k: usize, exponent: f64) -> Self {
        Self(prepare(mesh_src, mesh_tgt, k, exponent))
    }
}

impl Transfer for InverseDistanceTransfer {
    fn apply(&self, field: &FieldViewD, field_nature: FieldNature, default: f64) -> FieldOwnedD {
        self.0.apply(field, field_nature, default)
    }

    fn tgt_dim(&self) -> Dimension {
        self.0.tgt_dim()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use ndarray as nd;

    use crate::element_traits::ElementGeo;
    use crate::mesh::{ElementType, UMesh};
    use crate::mesh_examples as me;

    fn source_with_field(values: nd::Array<f64, nd::IxDyn>) -> UMesh {
        let mut source = me::make_imesh_2d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(ElementType::QUAD4, values)]));
        source.update_field("f", field.into_shared());
        source
    }

    fn field_view(source: &UMesh) -> FieldViewD<'_> {
        source.field("f", Some(Dimension::D2)).unwrap()
    }

    /// A constant intensive field is reproduced exactly, without overshoot.
    #[test]
    fn transfer_constant_intensive() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(4);
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr.shape(), &[16]);
        assert!(arr.iter().all(|&v| v == 7.0));
    }

    /// The interpolation is a convex combination: it never leaves the range of the
    /// source values (no overshoot/undershoot).
    #[test]
    fn transfer_no_overshoot() {
        let mut source = me::make_imesh_2d(2);
        let values: Vec<f64> = source
            .elements_of_dim(Dimension::D2)
            .map(|e| {
                let c = e.centroid2();
                c[0] + c[1]
            })
            .collect();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::Array::from_iter(values).into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        let target = me::make_imesh_2d(8);
        for k in [1, 2, 4, 8] {
            for exponent in [1.0, 2.0, 4.0] {
                let op = InverseDistanceTransfer::new(&source.view(), &target.view(), k, exponent);
                let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
                let arr = &out.0[&ElementType::QUAD4];
                for &v in arr {
                    assert!(
                        (0.0..=2.0).contains(&v),
                        "k = {k}, exponent = {exponent}: value {v} overshoots [0, 2]"
                    );
                }
            }
        }
    }

    /// A target point coinciding with a source point reproduces that source value.
    #[test]
    fn transfer_coincident_is_exact() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(1);
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
        let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        assert_eq!(out.0[&ElementType::QUAD4][0], 7.0);
    }

    /// `k = 1` degenerates to exact nearest-neighbour sampling.
    #[test]
    fn transfer_nearest_neighbour() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(4);
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 1, 2.0);
        let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        assert!(out.0[&ElementType::QUAD4].iter().all(|&v| v == 7.0));
    }

    /// Interpolated values between distinct source values stay strictly between them.
    #[test]
    fn transfer_bounded_by_neighbours() {
        let mut source = me::make_imesh_2d(2);
        let values: Vec<f64> = (0..4).map(|i| (i % 2) as f64).collect();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::Array::from_iter(values).into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        let target = me::make_imesh_2d(8);
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
        let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        for &v in out.0[&ElementType::QUAD4].iter() {
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
        }
    }

    /// An empty source fails with a clear message.
    #[test]
    #[should_panic(expected = "Source mesh should not be empty")]
    fn transfer_empty_source_panics() {
        let source = UMesh::new(nd::ArcArray2::from_shape_vec((0, 2), vec![]).unwrap());
        let target = me::make_imesh_2d(2);
        let _ = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
    }

    /// A source whose topological cells span several element types (regression: the transfer
    /// used to compare each flattened block to the whole-source point count and panic).
    #[test]
    fn transfer_mixed_element_type_source() {
        let coords = nd::Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 2.0, 1.0, 2.0],
        )
        .unwrap();
        let mut source = UMesh::new(coords.into());
        source.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        source.add_regular_block(
            ElementType::TRI3,
            nd::arr2(&[[2, 3, 4], [4, 5, 2]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([
            (ElementType::QUAD4, nd::array![1.0].into_dyn()),
            (ElementType::TRI3, nd::array![2.0, 3.0].into_dyn()),
        ]));
        source.update_field("f", field.into_shared());

        let target = me::make_imesh_2d(3);
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
        let out = op.apply(
            &source.field("f", Some(Dimension::D2)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert_eq!(out.0[&ElementType::QUAD4].len(), target.num_elements());
        assert!(
            out.0[&ElementType::QUAD4].iter().all(|&v| v > 0.0),
            "every target point should sample the mixed source"
        );
    }

    /// A field on a 3D volume mesh is transferred onto a 2D manifold in 3D space.
    #[test]
    fn transfer_3d_downcast() {
        let source = me::make_imesh_3d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::array![5.0].into_dyn(),
        )]));
        let mut source = source;
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

        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
        let out = op.apply(
            &source.field("f", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert_eq!(out.0[&ElementType::QUAD4][0], 5.0);
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
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
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
        let op = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
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

    /// Feeding meshes with different space dimensions fails with a clear message.
    #[test]
    #[should_panic(expected = "same space dimension")]
    fn transfer_space_dim_mismatch_panics() {
        let source = me::make_imesh_2d(1);
        let target = me::make_imesh_3d(1);
        let _ = InverseDistanceTransfer::new(&source.view(), &target.view(), 4, 2.0);
    }

    /// Zero neighbours or a non-positive exponent fail with a clear message.
    #[test]
    #[should_panic(expected = "k should be at least 1")]
    fn transfer_zero_k_panics() {
        let source = me::make_imesh_2d(1);
        let target = me::make_imesh_2d(2);
        let _ = InverseDistanceTransfer::new(&source.view(), &target.view(), 0, 2.0);
    }
}
