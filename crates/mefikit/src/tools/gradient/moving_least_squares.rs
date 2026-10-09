//! Moving least-squares gradient reconstruction on point clouds.
//!
//! For each evaluation point the `k` nearest source points are gathered and a degree-1 polynomial
//! is fitted through them by weighted least squares, the weights coming from [`DistanceWeighting`].
//! The gradient of the field is the gradient of that polynomial, whose coefficients are solved
//! directly from the weighted normal equations (see the shared meshless solver). Affine fields are
//! reproduced exactly when the local system is full rank; degenerate local systems (rank-deficient
//! geometry or a non-finite kernel) yield a zero gradient, so building the operator never fails.

use ndarray as nd;

use crate::mesh::UMeshView;
use crate::tools::meshless::{
    DistanceWeighting, solve_gradient_neighbours, source_centroids, target_centroids,
    validated_dims,
};

use super::operator::{GradientMethod, GradientOperator, GradientRowSparse, GradientTarget};

/// Builds a moving least-squares gradient operator that evaluates on the cells of `mesh_tgt`.
pub(crate) fn prepare_cells(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    k: usize,
    weighting: DistanceWeighting,
) -> GradientOperator {
    let (src_dim, tgt_dim, space) = validated_dims(mesh_src, mesh_tgt, "Gradient");
    assert!(k > 0, "k should be at least 1");
    assert!(
        mesh_src.num_elements() > 0,
        "Source mesh should not be empty"
    );

    let (src_coords, n_src) = source_centroids(mesh_src, src_dim);
    let blocks = target_centroids(mesh_tgt, space)
        .into_iter()
        .map(|(et, coords)| (et, gradient_block(&src_coords, &coords, k, weighting)))
        .collect();

    GradientOperator::build(
        GradientMethod::MovingLeastSquares { k, weighting },
        src_dim,
        space,
        n_src,
        Some(tgt_dim),
        GradientTarget::Cells(blocks),
    )
}

/// Builds a moving least-squares gradient operator that evaluates on the source mesh's own cells.
pub(crate) fn prepare_on_source(
    mesh_src: &UMeshView,
    k: usize,
    weighting: DistanceWeighting,
) -> GradientOperator {
    let (src_dim, _, space) = validated_dims(mesh_src, mesh_src, "Gradient");
    assert!(k > 0, "k should be at least 1");
    assert!(
        mesh_src.num_elements() > 0,
        "Source mesh should not be empty"
    );

    let (src_coords, n_src) = source_centroids(mesh_src, src_dim);
    // The source field is defined on the source cells of `src_dim`, so the gradient field is too.
    let blocks = target_centroids(mesh_src, space)
        .into_iter()
        .filter(|(et, _)| et.dimension() == src_dim)
        .map(|(et, coords)| (et, gradient_block(&src_coords, &coords, k, weighting)))
        .collect();

    GradientOperator::build(
        GradientMethod::MovingLeastSquares { k, weighting },
        src_dim,
        space,
        n_src,
        Some(src_dim),
        GradientTarget::Cells(blocks),
    )
}

/// Builds a moving least-squares gradient operator that evaluates at an arbitrary point cloud.
pub(crate) fn prepare_points(
    mesh_src: &UMeshView,
    points: &nd::ArrayView2<f64>,
    k: usize,
    weighting: DistanceWeighting,
) -> GradientOperator {
    let (src_dim, _, space) = validated_dims(mesh_src, mesh_src, "Gradient");
    assert!(k > 0, "k should be at least 1");
    assert!(
        mesh_src.num_elements() > 0,
        "Source mesh should not be empty"
    );
    assert_eq!(
        points.ncols(),
        space,
        "Evaluation points should have {space} columns, got {}",
        points.ncols()
    );

    let (src_coords, n_src) = source_centroids(mesh_src, src_dim);
    let rows = gradient_block(&src_coords, &points.to_owned(), k, weighting);

    GradientOperator::build(
        GradientMethod::MovingLeastSquares { k, weighting },
        src_dim,
        space,
        n_src,
        None,
        GradientTarget::Points(rows),
    )
}

/// Solves the gradient rows for one set of evaluation points and packs them into CSR-like form.
fn gradient_block(
    src_coords: &nd::Array2<f64>,
    tgt_coords: &nd::Array2<f64>,
    k: usize,
    weighting: DistanceWeighting,
) -> GradientRowSparse {
    let (indices, grads) =
        solve_gradient_neighbours(&src_coords.view(), &tgt_coords.view(), k, weighting);

    let n_tgt = indices.nrows();
    let k_nb = indices.ncols();
    let d = grads.shape()[2];

    let mut row_ptr = Vec::with_capacity(n_tgt + 1);
    row_ptr.push(0);
    let mut src_idx = Vec::with_capacity(n_tgt * k_nb);
    let mut weights = Vec::with_capacity(n_tgt * k_nb * d);
    for j in 0..n_tgt {
        for l in 0..k_nb {
            src_idx.push(indices[[j, l]]);
            for a in 0..d {
                weights.push(grads[[j, l, a]]);
            }
        }
        row_ptr.push(src_idx.len());
    }

    GradientRowSparse {
        row_ptr,
        src_idx,
        weights,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use approx::assert_relative_eq;
    use ndarray as nd;

    use crate::element_traits::ElementGeo;
    use crate::mesh::{ElementType, FieldOwnedD, UMesh};
    use crate::mesh_examples as me;
    use crate::tools::fieldexpr::field;
    use crate::tools::meshless::{DistanceWeighting, solve_gradient_neighbours};

    /// Recombines raw gradient weights with scalar source values, as the operator does at apply time.
    fn point_gradients(
        src: &nd::ArrayView2<f64>,
        tgt: &nd::ArrayView2<f64>,
        k: usize,
        weighting: DistanceWeighting,
        values: &[f64],
    ) -> nd::Array2<f64> {
        let (indices, grads) = solve_gradient_neighbours(src, tgt, k, weighting);
        let (n_tgt, k_nb, d) = grads.dim();
        let mut out = nd::Array2::<f64>::zeros((n_tgt, d));
        for j in 0..n_tgt {
            for l in 0..k_nb {
                let value = values[indices[[j, l]]];
                for a in 0..d {
                    out[[j, a]] += grads[[j, l, a]] * value;
                }
            }
        }
        out
    }

    #[test]
    fn reproduces_affine_gradient_2d() {
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tgt = nd::array![[0.25, 0.25], [0.6, 0.3]];
        // f = 1 + 2x - 3y
        let values = vec![1.0, 3.0, -2.0, 0.0];
        let g = point_gradients(
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::Constant,
            &values,
        );
        for j in 0..tgt.nrows() {
            assert_relative_eq!(g[[j, 0]], 2.0, epsilon = 1e-9);
            assert_relative_eq!(g[[j, 1]], -3.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn reproduces_affine_gradient_3d() {
        let src = nd::array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ];
        let tgt = nd::array![[0.2, 0.2, 0.2], [0.1, 0.3, 0.4]];
        // f = 1 + 2x - 3y + 4z
        let values = vec![1.0, 3.0, -2.0, 5.0];
        let g = point_gradients(
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::Constant,
            &values,
        );
        for j in 0..tgt.nrows() {
            assert_relative_eq!(g[[j, 0]], 2.0, epsilon = 1e-9);
            assert_relative_eq!(g[[j, 1]], -3.0, epsilon = 1e-9);
            assert_relative_eq!(g[[j, 2]], 4.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn coincident_point_with_inverse_distance_reproduces_affine() {
        // An evaluation point coinciding with a source point makes the inverse-distance kernel
        // singular; the affine fit must stay exact instead of degenerating to a zero gradient.
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let tgt = nd::array![[0.0, 0.0], [1.0, 1.0]];
        // f = 1 + 2x - 3y
        let values = vec![1.0, 3.0, -2.0, 0.0];
        let g = point_gradients(
            &src.view(),
            &tgt.view(),
            4,
            DistanceWeighting::InverseDistance { exponent: 2.0 },
            &values,
        );
        for j in 0..tgt.nrows() {
            assert_relative_eq!(g[[j, 0]], 2.0, epsilon = 1e-9);
            assert_relative_eq!(g[[j, 1]], -3.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn rank_deficient_geometry_gives_zero_gradient() {
        // Collinear source points cannot resolve the normal direction: the local fit is
        // rank-deficient and the gradient falls back to zero instead of failing.
        let src = nd::array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let tgt = nd::array![[0.5, 0.5]];
        let values = vec![0.0, 1.0, 2.0];
        let g = point_gradients(
            &src.view(),
            &tgt.view(),
            3,
            DistanceWeighting::Constant,
            &values,
        );
        assert_relative_eq!(g[[0, 0]], 0.0, epsilon = 1e-12);
        assert_relative_eq!(g[[0, 1]], 0.0, epsilon = 1e-12);
    }

    /// Stores `f = 1 + 2x - 3y` at the QUAD4 centroids of `mesh`.
    fn set_affine_field_2d(mesh: &mut UMesh) {
        let values: Vec<f64> = mesh
            .elements_of_type(ElementType::QUAD4)
            .map(|e| {
                let c = e.centroid2();
                1.0 + 2.0 * c[0] - 3.0 * c[1]
            })
            .collect();
        let n = values.len();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::Array::from_shape_vec((n,), values).unwrap().into_dyn(),
        )]));
        mesh.update_field("f", field.into_shared());
    }

    #[test]
    fn on_source_gradient_of_affine_field() {
        let mut mesh = me::make_imesh_2d(4);
        set_affine_field_2d(&mut mesh);
        let op = GradientOperator::on_source(
            &mesh.view(),
            GradientMethod::MovingLeastSquares {
                k: 9,
                weighting: DistanceWeighting::Constant,
            },
        );
        let grad = op.eval(&mesh, field("f"), 0.0);
        let block = grad.0.get(&ElementType::QUAD4).unwrap();
        assert_eq!(block.shape()[0], 16);
        assert_eq!(block.shape()[1], 2);
        for row in block.rows() {
            assert_relative_eq!(row[0], 2.0, epsilon = 1e-7);
            assert_relative_eq!(row[1], -3.0, epsilon = 1e-7);
        }
    }

    #[test]
    fn at_points_gradient_of_affine_field() {
        let mut mesh = me::make_imesh_2d(4);
        set_affine_field_2d(&mut mesh);
        let points = nd::array![[0.1, 0.1], [0.5, 0.5], [0.9, 0.2]];
        let op = GradientOperator::at_points(
            &mesh.view(),
            &points.view(),
            GradientMethod::MovingLeastSquares {
                k: 9,
                weighting: DistanceWeighting::Constant,
            },
        );
        assert!(!op.has_cell_target());
        let g = op.eval_points(&mesh, field("f"), 0.0);
        assert_eq!(g.shape(), &[3, 2]);
        for row in g.rows() {
            assert_relative_eq!(row[0], 2.0, epsilon = 1e-7);
            assert_relative_eq!(row[1], -3.0, epsilon = 1e-7);
        }
    }
}
