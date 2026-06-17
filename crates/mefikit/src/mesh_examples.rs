//! Example mesh generators for testing and demonstration.

use crate::prelude as mf;
use ndarray as nd;

/// Creates a simple 2D mesh with a single QUAD4 element.
pub fn make_mesh_2d_quad() -> mf::UMesh {
    let coords =
        nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
    let mut mesh = mf::UMesh::new(coords);
    mesh.add_regular_block(
        mf::ElementType::QUAD4,
        nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
        None,
    );
    mesh
}

/// Creates a simple 3D mesh with two SEG2 elements.
pub fn make_mesh_3d_seg2() -> mf::UMesh {
    let coords = nd::Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let mut mesh = mf::UMesh::new(coords.into());
    mesh.add_regular_block(
        mf::ElementType::SEG2,
        nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
        None,
    );
    mesh
}

/// Creates a 2D mesh with multiple element types:
/// - Two SEG2 elements
/// - One QUAD4 element
/// - One PGON element
pub fn make_mesh_2d_multi() -> mf::UMesh {
    let coords = nd::Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.5],
    )
    .unwrap();
    let mut mesh = mf::UMesh::new(coords.into());
    mesh.add_regular_block(
        mf::ElementType::SEG2,
        nd::arr2(&[[0, 1], [1, 3]]).to_shared(),
        None,
    );
    mesh.add_regular_block(
        mf::ElementType::QUAD4,
        nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
        None,
    );
    mesh.add_element(mf::ElementType::PGON, &[0, 1, 4, 3, 2], None, None);
    mesh
}

/// Creates a structured 2D mesh with `n x n` QUAD4 elements.
pub fn make_imesh_2d(n: usize) -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build()
}

/// Creates a structured 3D mesh with `n x n x n` HEX8 elements.
pub fn make_imesh_3d(n: usize) -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build()
}
