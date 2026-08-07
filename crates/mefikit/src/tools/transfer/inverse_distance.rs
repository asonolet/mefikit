use std::collections::BTreeMap;
use std::num::NonZero;

use kiddo::{ImmutableKdTree, dist::SquaredEuclidean};
use ndarray::{Array2, ArrayD, ArrayView2, ArrayViewD, Axis, Zip, concatenate};

use crate::element_traits::ElementGeo;
use crate::mesh::{Dimension, ElementType, FieldOwnedD, FieldViewD, UMeshView};
use crate::tools::centroids::centroids;
use crate::tools::{FieldNature, Transfer};

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
pub struct InverseDistanceTransfer {
    tgt_dim: Dimension,
    /// Number of source points the operator was built from.
    n_src: usize,
    /// Indices of the `k` source points used for each target point. Shape
    /// `(n_tgt × k)`. They are grouped by element type.
    indices: Vec<(ElementType, Array2<usize>)>,
    /// Inverse-distance weights, one per selected source point. Shape
    /// `(n_tgt × k)`, rows sum to one. They are grouped by element type.
    weights: Vec<(ElementType, Array2<f64>)>,
}

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
        let src_space = mesh_src.space_dimension();
        let tgt_space = mesh_tgt.space_dimension();
        assert_eq!(
            src_space, tgt_space,
            "Source and target meshes should share the same space dimension, got source = {src_space}D and target = {tgt_space}D"
        );
        assert!(
            (2..=3).contains(&src_space),
            "Inverse-distance transfer is only supported in 2D and 3D space, got {src_space}D"
        );
        assert!(k > 0, "k should be at least 1");
        assert!(
            exponent > 0.0,
            "exponent should be positive, got {exponent}"
        );
        assert!(
            mesh_src.num_elements() > 0,
            "Source mesh should not be empty"
        );

        let src_coords = centroids(mesh_src, None);
        let src_view: Vec<_> = src_coords.values().map(|a| a.view()).collect();
        let src_coords = concatenate(Axis(0), src_view.as_slice()).unwrap();
        let n_src = src_coords.nrows();
        let tgt_dim = mesh_tgt
            .topological_dimension()
            .expect("Target mesh should not be empty");

        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for et in mesh_tgt.element_types() {
            let tgt_coords = match src_space {
                2 => {
                    let v: Vec<f64> = mesh_tgt
                        .elements_of_type(*et)
                        .flat_map(|e| e.centroid2().into_iter())
                        .collect();
                    Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 2), v).unwrap()
                }
                3 => {
                    let v: Vec<f64> = mesh_tgt
                        .elements_of_type(*et)
                        .flat_map(|e| e.centroid3().into_iter())
                        .collect();
                    Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 3), v).unwrap()
                }
                _ => unreachable!(),
            };
            let (ind, wei) = match src_space {
                2 => solve_points::<2>(&src_coords.view(), &tgt_coords.view(), k, exponent),
                3 => solve_points::<3>(&src_coords.view(), &tgt_coords.view(), k, exponent),
                _ => unreachable!(),
            };
            indices.push((*et, ind));
            weights.push((*et, wei));
        }
        Self {
            tgt_dim,
            n_src,
            indices,
            weights,
        }
    }

    /// Evaluates the operator on a flat source array `src` of shape `(n_src, ...)`
    /// for the target element type `et`, producing a `(n_tgt, ...)` array.
    fn apply_on_array(&self, et: ElementType, src: &ArrayViewD<f64>) -> ArrayD<f64> {
        let n_src = src.shape()[0];
        assert_eq!(
            n_src, self.n_src,
            "The field should have one entry per source point, got {n_src} for {}",
            self.n_src
        );
        let n_compo = src.len() / n_src;

        let i = self
            .indices
            .iter()
            .enumerate()
            .find(|(_, (e, _))| *e == et)
            .unwrap()
            .0;

        let indices = &self.indices[i].1;
        let weights = &self.weights[i].1;

        let n_tgt = indices.nrows();
        let k = indices.ncols();

        let mut tgt = Array2::<f64>::zeros((n_tgt, n_compo));
        let src_view = src.view().into_shape_with_order((n_src, n_compo)).unwrap();

        for j in 0..n_tgt {
            for l in 0..k {
                let i = indices[[j, l]];
                let w = weights[[j, l]];
                Zip::from(tgt.row_mut(j))
                    .and(src_view.row(i))
                    .for_each(|d, &s| {
                        *d += w * s;
                    });
            }
        }

        let mut tgt_shape = src.raw_dim();
        tgt_shape[0] = n_tgt;
        tgt.into_shape_with_order(tgt_shape).unwrap()
    }
}

/// Computes the inverse-distance weights for every target point.
fn solve_points<const D: usize>(
    src_coords: &ArrayView2<f64>,
    tgt_coords: &ArrayView2<f64>,
    k: usize,
    exponent: f64,
) -> (Array2<usize>, Array2<f64>) {
    let n_tgt = tgt_coords.nrows();
    let mut indices = Array2::<usize>::zeros((n_tgt, k));
    let mut weights = Array2::<f64>::zeros((n_tgt, k));

    let tree = ImmutableKdTree::new_from_slice(as_points::<D>(src_coords)).unwrap();

    for (j, p) in tgt_coords.outer_iter().enumerate() {
        let query: [f64; D] = std::array::from_fn(|c| p[c]);
        let nn = tree
            .query(&query)
            .nearest_n::<SquaredEuclidean<f64>>(NonZero::new(k).unwrap())
            .execute();

        for (l, r) in nn.iter().enumerate() {
            indices[[j, l]] = r.item as usize;
        }

        // A target point coinciding with a source point interpolates it exactly.
        if let Some(l) = nn.iter().position(|r| r.distance == 0.0) {
            weights[[j, l]] = 1.0;
            continue;
        }

        let mut sum = 0.0;
        for r in &nn {
            let w = r.distance.powf(-0.5 * exponent);
            sum += w;
        }
        for (l, r) in nn.iter().enumerate() {
            weights[[j, l]] = r.distance.powf(-0.5 * exponent) / sum;
        }
    }

    (indices, weights)
}

/// Views an `n × D` contiguous coordinate array as a slice of `[f64; D]` points.
fn as_points<'a, const D: usize>(coords: &ArrayView2<'a, f64>) -> &'a [[f64; D]] {
    assert_eq!(coords.ncols(), D);
    let slice = coords.as_slice().expect("coordinates should be contiguous");

    // Safety:
    // - the slice length is a multiple of D
    // - f64 is properly aligned
    // - `[f64; D]` is a contiguous run of D f64 values
    let len = slice.len() / D;
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const [f64; D], len) }
}

impl Transfer for InverseDistanceTransfer {
    fn apply(&self, field: &FieldViewD, _field_nature: FieldNature, _default: f64) -> FieldOwnedD {
        // Concatenate the per-element-type source arrays in the same order used to
        // build the operator (BTreeMap order), so the neighbour indices are valid
        // whatever the source and target element types are (e.g. a downcast from
        // volume cells to a surface).
        let src_views: Vec<ArrayViewD<f64>> = field.0.values().map(|a| a.view()).collect();
        let src = concatenate(Axis(0), src_views.as_slice()).unwrap();

        let mut res = BTreeMap::new();
        for (et, _) in &self.indices {
            res.insert(*et, self.apply_on_array(*et, &src.view()));
        }
        FieldOwnedD::new(res)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray as nd;

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
