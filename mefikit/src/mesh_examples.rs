use crate as mf;
use ndarray as nd;

pub fn make_mesh_2d_quad() -> mf::UMesh {
    let coords =
        nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
    let mut mesh = mf::UMesh::new(coords);
    mesh.add_regular_block(
        mf::ElementType::QUAD4,
        nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
    );
    mesh
}

pub fn make_mesh_3d_seg2() -> mf::UMesh {
    let coords = nd::Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let mut mesh = mf::UMesh::new(coords.into());
    mesh.add_regular_block(
        mf::ElementType::SEG2,
        nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
    );
    mesh
}

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
    );
    mesh.add_regular_block(
        mf::ElementType::QUAD4,
        nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
    );
    mesh.add_element(mf::ElementType::PGON, &[0, 1, 5, 3, 2], None, None);
    mesh
}

pub fn make_imesh_2d(n: usize) -> mf::UMesh {
    let mesh = crate::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build();
    mesh
}

pub fn make_imesh_3d(n: usize) -> mf::UMesh {
    let mesh = crate::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build();
    mesh
}
