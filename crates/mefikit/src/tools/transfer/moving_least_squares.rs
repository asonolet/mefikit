//! Moving least-squares interpolation transfer between point clouds.
//!
//! For each target point the `k` nearest source points are gathered and a degree-1 polynomial is
//! fitted through them by weighted least squares, the weights coming from [`DistanceWeighting`].
//! The fit is evaluated at the target point, which yields `k` interpolation weights `w_i` such
//! that the transferred value is `sum_i w_i f(x_i)`. The `k`-nearest-neighbours precompute (in the
//! shared [`super::solver`] module) and the apply-time sparse product (in the shared
//! [`super::operator::TransferOperator`]) are shared with the other transfer methods; this module
//! owns the fitting itself and the method's validation.

use super::operator::{TransferMethod, TransferOperator, point_interpolation, validated_dims};
use super::solver::{DistanceWeighting, NeighbourScheme, solve_neighbours, source_centroids};
use crate::mesh::UMeshView;

/// Builds the moving least-squares operator over the `k` nearest source points of every target point.
///
/// For each target point the `k` nearest source points are gathered and a degree-1 polynomial is
/// fitted through them by weighted least squares, the weights coming from [`DistanceWeighting`]. The
/// fit is evaluated at the target point, which yields `k` interpolation weights `w_i` such that the
/// transferred value is `sum_i w_i f(x_i)`. Affine fields are reproduced exactly when the local
/// system is full rank; degenerate local systems fall back to the normalized kernel weights
/// (Shepard's method), then to a plain average, so building the operator never fails. The
/// characteristic length `h` of the weighting kernel is the distance from the target point to its
/// farthest selected neighbour, computed once with no inflation: a compact-support kernel gives the
/// neighbour at distance `h` a zero weight, which can drop it out of the fit and trigger the
/// fallback when `k` barely overdetermines the local system (increase `k` to keep the neighbour
/// inside the fit).
///
/// # Panics
///
/// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or if it is not 2 or 3.
/// - If `k` is zero.
pub(crate) fn prepare(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    k: usize,
    weighting: DistanceWeighting,
) -> TransferOperator {
    let (src_dim, tgt_dim, src_space) = validated_dims(mesh_src, mesh_tgt, "Moving least-squares");
    assert!(k > 0, "k should be at least 1");
    assert!(
        mesh_src.num_elements() > 0,
        "Source mesh should not be empty"
    );

    let (src_coords, n_src) = source_centroids(mesh_src, src_dim);

    TransferOperator::build(
        TransferMethod::MovingLeastSquares { k, weighting },
        src_dim,
        tgt_dim,
        n_src,
        point_interpolation(mesh_tgt, src_space, |coords| {
            solve_neighbours(
                &src_coords.view(),
                &coords,
                k,
                NeighbourScheme::MovingLeastSquares { weighting },
            )
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use approx::assert_relative_eq;

    use crate::mesh::{Dimension, ElementType, FieldOwnedD, UMesh};
    use crate::mesh_examples as me;
    use crate::tools::transfer::transfer_trait::{FieldNature, Transfer};
    use ndarray as nd;

    /// Builds a moving least-squares operator for the common `(k, weighting)` test pairs.
    fn mls(
        source: &UMeshView,
        target: &UMeshView,
        k: usize,
        weighting: DistanceWeighting,
    ) -> TransferOperator {
        TransferOperator::new(
            source,
            target,
            TransferMethod::MovingLeastSquares { k, weighting },
        )
    }

    fn src_grid_3d() -> nd::Array2<f64> {
        let mut pts = Vec::new();
        for z in 0..=4 {
            for y in 0..=4 {
                for x in 0..=4 {
                    pts.extend([x as f64 * 0.25, y as f64 * 0.25, z as f64 * 0.25]);
                }
            }
        }
        nd::Array2::from_shape_vec((125, 3), pts).unwrap()
    }

    /// Wraps the raw `(indices, weights)` returned by `solve_neighbours` into a full
    /// operator so the tests can exercise `apply_et`.
    fn build_op(
        src: &nd::ArrayView2<f64>,
        tgt: &nd::ArrayView2<f64>,
        k: usize,
        weighting: DistanceWeighting,
    ) -> TransferOperator {
        let (indices, weights) = solve_neighbours(
            src,
            tgt,
            k,
            NeighbourScheme::MovingLeastSquares { weighting },
        );
        let tgt_dim = if src.ncols() == 2 {
            Dimension::D2
        } else {
            Dimension::D3
        };
        TransferOperator::from_neighbours(
            tgt_dim,
            tgt_dim,
            src.nrows(),
            ElementType::QUAD4,
            indices,
            weights,
        )
    }

    /// Applies the operator to a flat source array.
    fn apply(op: &TransferOperator, field: &nd::ArrayViewD<f64>) -> nd::ArrayD<f64> {
        op.apply_et(ElementType::QUAD4, field)
    }

    #[test]
    fn kernel_values() {
        assert_eq!(DistanceWeighting::Constant.kernel(0.25), 1.0);
        let inv = DistanceWeighting::InverseDistance { exponent: 2.0 };
        assert_relative_eq!(inv.kernel(1.0), 1.0, epsilon = 1e-12);
        assert_relative_eq!(inv.kernel(4.0), 0.25, epsilon = 1e-12);
        let compact = DistanceWeighting::CompactSupport { exponent: 2.0 };
        assert_relative_eq!(compact.kernel(0.25), 0.5625, epsilon = 1e-12);
        assert_eq!(compact.kernel(1.0), 0.0);
        assert_eq!(compact.kernel(2.0), 0.0);
        let gaussian = DistanceWeighting::Gaussian;
        assert_relative_eq!(gaussian.kernel(1.0), (-1.0_f64).exp(), epsilon = 1e-12);
    }

    #[test]
    fn reproduces_affine_field_exactly_3d() {
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let tgt = nd::array![[0.25, 0.25, 0.25], [0.1, 0.2, 0.3]];
        // f = 1 + 2x - 3y + 4z
        let field = nd::array![1.0, 3.0, -2.0, 5.0].into_dyn();
        let expected = nd::array![1.75, 1.8].into_dyn();
        // Compact support is excluded: with `k = d + 1` the neighbour at distance `h` gets a zero
        // weight and drops out of the fit, so the local system is rank-deficient and the transfer
        // degrades to the Shepard fallback instead of reproducing the affine field exactly.
        for weighting in [DistanceWeighting::Constant, DistanceWeighting::Gaussian] {
            let op = build_op(&src.view(), &tgt.view(), 4, weighting);
            let out = apply(&op, &field.view());
            for (o, e) in out.iter().zip(expected.iter()) {
                assert!(
                    (o - e).abs() < 1e-9,
                    "weighting = {weighting:?}: got {o}, expected {e}"
                );
            }
        }
    }

    #[test]
    fn reproduces_affine_field_exactly_2d() {
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let tgt = nd::array![[1.0 / 3.0, 1.0 / 3.0], [0.2, 0.4]];
        // f = 1 + 2x - 3y
        let field = nd::array![1.0, 3.0, -2.0].into_dyn();
        let expected = nd::array![2.0 / 3.0, 0.2].into_dyn();
        let op = build_op(&src.view(), &tgt.view(), 3, DistanceWeighting::Constant);
        let out = apply(&op, &field.view());
        for (o, e) in out.iter().zip(expected.iter()) {
            assert_relative_eq!(o, e, epsilon = 1e-9);
        }
    }

    #[test]
    fn reproduces_constant_field_on_grid() {
        let src = src_grid_3d();
        let tgt = nd::array![[0.32, 0.17, 0.71], [0.9, 0.1, 0.9]];
        let field = nd::Array::from_elem(nd::IxDyn(&[125]), 7.0);
        for weighting in [
            DistanceWeighting::Constant,
            DistanceWeighting::InverseDistance { exponent: 2.0 },
            DistanceWeighting::CompactSupport { exponent: 3.0 },
            DistanceWeighting::Gaussian,
        ] {
            let op = build_op(&src.view(), &tgt.view(), 8, weighting);
            let out = apply(&op, &field.view());
            assert!(
                out.iter().all(|&v| (v - 7.0).abs() < 1e-9),
                "constant not reproduced with weighting = {weighting:?}"
            );
        }
    }

    #[test]
    fn coincident_target_is_exact() {
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.5]
        ];
        let tgt = nd::array![[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]];
        let field = nd::array![1.0, 2.0, 3.0, 4.0, 9.0].into_dyn();
        let op = build_op(&src.view(), &tgt.view(), 4, DistanceWeighting::Gaussian);
        let out = apply(&op, &field.view());
        assert_eq!(out[0], 9.0);
        assert!(out[1].is_finite());
    }

    #[test]
    fn singular_cloud_falls_back() {
        // Coplanar source points: the 3D linear system is rank-deficient and the solve fails,
        // but the transfer must not panic and a constant field is still reproduced.
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0]
        ];
        let tgt = nd::array![[0.4, 0.3, 0.0]];
        let field = nd::Array::from_elem(nd::IxDyn(&[4]), 5.0);
        for weighting in [
            DistanceWeighting::Constant,
            DistanceWeighting::Gaussian,
            DistanceWeighting::InverseDistance { exponent: 2.0 },
            DistanceWeighting::CompactSupport { exponent: 2.0 },
        ] {
            let op = build_op(&src.view(), &tgt.view(), 4, weighting);
            let out = apply(&op, &field.view());
            assert_relative_eq!(out[0], 5.0, epsilon = 1e-9);
        }
    }

    /// With `k = d + 1` a compact-support kernel zeroes the neighbour at distance `h`, so the fit
    /// degrades immediately to the Shepard fallback: the result stays a convex combination of the
    /// in-support source values (never overshooting the field range) and never panics, but the
    /// affine field is no longer reproduced exactly.
    #[test]
    fn compact_support_degrades_gracefully() {
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let tgt = nd::array![[0.25, 0.25, 0.25], [0.1, 0.2, 0.3]];
        // f = 1 + 2x - 3y + 4z, so the source values span [-2.0, 5.0].
        let field = nd::array![1.0, 3.0, -2.0, 5.0].into_dyn();
        for exponent in [1.0, 2.0, 3.0] {
            let op = build_op(
                &src.view(),
                &tgt.view(),
                4,
                DistanceWeighting::CompactSupport { exponent },
            );
            let out = apply(&op, &field.view());
            for (i, &v) in out.iter().enumerate() {
                assert!(v.is_finite(), "exponent = {exponent}: NaN at {i}");
                assert!(
                    (-2.0..=5.0).contains(&v),
                    "exponent = {exponent}: value {v} overshoots the field range at {i}"
                );
            }
        }
    }

    #[test]
    fn vector_valued_field() {
        let src = src_grid_3d();
        let tgt = nd::array![[0.32, 0.17, 0.71], [0.5, 0.5, 0.5]];
        let mut field = nd::Array2::<f64>::zeros((125, 3));
        for i in 0..125 {
            field[[i, 0]] = 1.0;
            field[[i, 1]] = i as f64;
            field[[i, 2]] = 2.0 * i as f64;
        }
        let op = build_op(
            &src.view(),
            &tgt.view(),
            8,
            DistanceWeighting::InverseDistance { exponent: 2.0 },
        );
        let out = apply(&op, &field.view().into_dyn());
        assert_eq!(out.shape(), &[2, 3]);
        // constant component reproduced
        assert_relative_eq!(out[[0, 0]], 1.0, epsilon = 1e-9);
        // coincident target at (0.5,0.5,0.5) is exact
        let i50 = 2 * 25 + 2 * 5 + 2;
        assert_eq!(out[[1, 1]], i50 as f64);
        assert_eq!(out[[1, 2]], 2.0 * i50 as f64);
    }

    #[test]
    fn weighted_differs_from_unweighted() {
        // With k > d + 1 the weighting scheme changes the interpolation weights.
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ];
        let tgt = nd::array![[0.4, 0.3, 0.2]];
        let plain = build_op(&src.view(), &tgt.view(), 8, DistanceWeighting::Constant);
        let inverse = build_op(
            &src.view(),
            &tgt.view(),
            8,
            DistanceWeighting::InverseDistance { exponent: 4.0 },
        );
        let field = nd::array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].into_dyn();
        let a = apply(&plain, &field.view());
        let b = apply(&inverse, &field.view());
        assert!(
            (a[0] - b[0]).abs() > 1e-6,
            "weighted and unweighted interpolation should differ, got {a:?} vs {b:?}"
        );
    }

    /// A source whose topological cells span several element types is handled by flattening the
    /// per-element-type arrays in the same order used to build the operator (regression: the
    /// operator used to feed each block separately and compare it to the whole-source point
    /// count, which panicked).
    #[test]
    fn mixed_element_type_source() {
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
        let op = mls(
            &source.view(),
            &target.view(),
            4,
            DistanceWeighting::Constant,
        );
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

    /// A 3D target mesh is supported, as is the downcast from volume source cells onto a surface
    /// target (regression: only 2D targets were implemented, the 3D arm was a `todo!()`).
    #[test]
    fn supports_3d_target_and_downcast() {
        let mut source = me::make_imesh_3d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::array![5.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        // 3D -> 3D
        let target = me::make_imesh_3d(2);
        let op = mls(
            &source.view(),
            &target.view(),
            4,
            DistanceWeighting::Constant,
        );
        let out = op.apply(
            &source.field("f", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert_eq!(out.0[&ElementType::HEX8].len(), target.num_elements());
        assert!(out.0[&ElementType::HEX8].iter().all(|&v| v == 5.0));

        // 3D -> 2D downcast (a single QUAD4 surface at z = 0)
        let tcoords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
        )
        .unwrap();
        let mut surface = UMesh::new(tcoords);
        surface.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let op = mls(
            &source.view(),
            &surface.view(),
            4,
            DistanceWeighting::Constant,
        );
        let out = op.apply(
            &source.field("f", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert_eq!(out.0[&ElementType::QUAD4][0], 5.0);
    }
}
