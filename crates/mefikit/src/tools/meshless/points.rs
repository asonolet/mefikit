//! Mesh-to-point plumbing for the meshless reconstruction operators.
//!
//! The meshless solvers work on plain `n × D` coordinate arrays; this module turns meshes into
//! those arrays. It is the only place where the meshless operators touch the mesh API:
//!
//! - [`validated_dims`] checks that a source/target mesh pair lives in the same 2D or 3D space
//!   and returns the two topological dimensions.
//! - [`source_centroids`] flattens the source cell centroids in the `BTreeMap` element-type order
//!   used to flatten the source field arrays at apply time.
//! - [`target_centroids`] gathers the target cell centroids, grouped per element type, so that a
//!   per-element-type sparse block can be assembled for each of them.

use ndarray as nd;

use crate::element_traits::ElementGeo;
use crate::mesh::{Dimension, ElementType, UMeshView};
use crate::tools::centroids::centroids;

/// Validates the space dimension of a source/target mesh pair and returns the common space
/// dimension together with the two topological dimensions.
///
/// All meshless operators work on pairs of meshes living in the same 2D or 3D space, both
/// non-empty. Returns `(src_dim, tgt_dim, space)`.
///
/// # Panics
///
/// If the two meshes do not share the same space dimension, if it is not 2 or 3, or if either
/// mesh is empty.
pub(crate) fn validated_dims(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    method: &str,
) -> (Dimension, Dimension, usize) {
    let src_space = mesh_src.space_dimension();
    let tgt_space = mesh_tgt.space_dimension();
    assert_eq!(
        src_space, tgt_space,
        "Source and target meshes should share the same space dimension, got source = {src_space}D and target = {tgt_space}D"
    );
    assert!(
        (2..=3).contains(&src_space),
        "{method} is only supported in 2D and 3D space, got {src_space}D"
    );
    let src_dim = mesh_src
        .topological_dimension()
        .expect("Source mesh should not be empty");
    let tgt_dim = mesh_tgt
        .topological_dimension()
        .expect("Target mesh should not be empty");
    (src_dim, tgt_dim, src_space)
}

/// Centroids of the source cells of topological dimension `src_dim`, flattened in the same
/// `BTreeMap` order used to flatten the source field arrays, together with the total point count.
///
/// # Panics
///
/// If `src_dim` is neither 2 nor 3, or if the source mesh has no cell of that dimension.
pub(crate) fn source_centroids(
    mesh_src: &UMeshView,
    src_dim: Dimension,
) -> (nd::Array2<f64>, usize) {
    let src_coords = centroids(mesh_src, Some(src_dim));
    let src_views: Vec<_> = src_coords.values().map(|a| a.view()).collect();
    let src_coords = nd::concatenate(nd::Axis(0), src_views.as_slice()).unwrap();
    let n_src = src_coords.nrows();
    (src_coords, n_src)
}

/// Centroids of each target cell, grouped per element type.
///
/// The returned blocks are in the target mesh's `element_types()` order, so an operator built
/// from them produces per-element-type result blocks without any further grouping.
///
/// # Panics
///
/// If `space` is neither 2 nor 3.
pub(crate) fn target_centroids(
    mesh_tgt: &UMeshView,
    space: usize,
) -> Vec<(ElementType, nd::Array2<f64>)> {
    let mut blocks = Vec::new();
    for et in mesh_tgt.element_types() {
        let coords = match space {
            2 => {
                let v: Vec<f64> = mesh_tgt
                    .elements_of_type(*et)
                    .flat_map(|e| e.centroid2().into_iter())
                    .collect();
                nd::Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 2), v).unwrap()
            }
            3 => {
                let v: Vec<f64> = mesh_tgt
                    .elements_of_type(*et)
                    .flat_map(|e| e.centroid3().into_iter())
                    .collect();
                nd::Array2::from_shape_vec((mesh_tgt.block(*et).unwrap().len(), 3), v).unwrap()
            }
            _ => unreachable!(),
        };
        blocks.push((*et, coords));
    }
    blocks
}
