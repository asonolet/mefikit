use crate::mesh::{ConnectivityView, ElementType, UMesh, UMeshView};
use crate::prelude::snap::merge_nodes;
use ndarray as nd;

pub fn split(mesh: UMeshView) -> UMesh {
    let element_blocks: Vec<_> = mesh.blocks().collect();
    let mut new_mesh = UMesh::new(mesh.coords().to_shared());

    for (&element_type, block) in element_blocks {
        match element_type {
            ElementType::SEG2 => {
                split_seg2(&mesh, block, &mut new_mesh);
            }
            ElementType::TRI3 => {
                split_tri3(&mesh, block, &mut new_mesh);
            }
            ElementType::QUAD4 => {
                split_quad4(&mesh, block, &mut new_mesh);
            }
            ElementType::TET4 => {
                split_tet4(&mesh, block, &mut new_mesh);
            }
            ElementType::HEX8 => {
                split_hex8(&mesh, block, &mut new_mesh);
            }
            _ => {
                // For unsupported element types, copy them as-is
                if let ConnectivityView::Regular(conn) = &block.connectivity {
                    new_mesh.add_regular_block(element_type, conn.to_shared(), None);
                }
            }
        }
    }

    merge_nodes(&mut new_mesh, f64::EPSILON);
    new_mesh
}

fn split_seg2(mesh: &UMeshView, block: &crate::mesh::ElementBlockView, new_mesh: &mut UMesh) {
    let conn = match &block.connectivity {
        ConnectivityView::Regular(c) => c.view(),
        _ => return,
    };
    let new_conn_size = conn.nrows() * 2;
    let mut new_conn: Vec<usize> = Vec::with_capacity(new_conn_size * 2);
    for element_conn in conn.rows() {
        let coords_n1 = mesh.coords.row(element_conn[0]);
        let coords_n2 = mesh.coords.row(element_conn[1]);

        let midpoint = (coords_n1.to_owned() + coords_n2.to_owned()) / 2.0;

        let new_node_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint.view())
            .expect("Shape error when adding coordinates to new mesh");

        new_conn.push(element_conn[0]);
        new_conn.push(new_node_index);
        new_conn.push(new_node_index);
        new_conn.push(element_conn[1]);
    }
    let new_conn_array = nd::Array2::from_shape_vec((new_conn_size, 2), new_conn)
        .expect("Shape error when building new connectivity array");

    new_mesh.add_regular_block(ElementType::SEG2, new_conn_array.into_shared(), None);
}

fn split_tri3(mesh: &UMeshView, block: &crate::mesh::ElementBlockView, new_mesh: &mut UMesh) {
    let conn = match &block.connectivity {
        ConnectivityView::Regular(c) => c.view(),
        _ => return,
    };
    let new_conn_size = conn.nrows() * 3;
    let mut new_conn: Vec<usize> = Vec::with_capacity(new_conn_size * 3);
    for element_conn in conn.rows() {
        let coords_n1 = mesh.coords.row(element_conn[0]);
        let coords_n2 = mesh.coords.row(element_conn[1]);
        let coords_n3 = mesh.coords.row(element_conn[2]);

        let midpoint = (coords_n1.to_owned() + coords_n2.to_owned() + coords_n3.to_owned()) / 3.0;

        let new_node_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint.view())
            .expect("Shape error when adding coordinates to new mesh");

        new_conn.push(element_conn[0]);
        new_conn.push(element_conn[1]);
        new_conn.push(new_node_index);

        new_conn.push(element_conn[1]);
        new_conn.push(element_conn[2]);
        new_conn.push(new_node_index);

        new_conn.push(element_conn[2]);
        new_conn.push(element_conn[0]);
        new_conn.push(new_node_index);
    }

    let new_conn_array = nd::Array2::from_shape_vec((new_conn_size, 3), new_conn)
        .expect("Shape error when building new connectivity array");

    new_mesh.add_regular_block(ElementType::TRI3, new_conn_array.into_shared(), None);
}

fn split_quad4(mesh: &UMeshView, block: &crate::mesh::ElementBlockView, new_mesh: &mut UMesh) {
    let conn = match &block.connectivity {
        ConnectivityView::Regular(c) => c.view(),
        _ => return,
    };
    let new_conn_size = conn.nrows() * 4;
    let mut new_conn: Vec<usize> = Vec::with_capacity(new_conn_size * 4);

    for element_conn in conn.rows() {
        let coords_n0 = mesh.coords.row(element_conn[0]);
        let coords_n1 = mesh.coords.row(element_conn[1]);
        let coords_n2 = mesh.coords.row(element_conn[2]);
        let coords_n3 = mesh.coords.row(element_conn[3]);

        let midpoint_01 = (coords_n0.to_owned() + coords_n1.to_owned()) / 2.0;
        let midpoint_12 = (coords_n1.to_owned() + coords_n2.to_owned()) / 2.0;
        let midpoint_23 = (coords_n2.to_owned() + coords_n3.to_owned()) / 2.0;
        let midpoint_30 = (coords_n3.to_owned() + coords_n0.to_owned()) / 2.0;

        let center = (coords_n0.to_owned()
            + coords_n1.to_owned()
            + coords_n2.to_owned()
            + coords_n3.to_owned())
            / 4.0;

        let mid_01_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint_01.view())
            .expect("Shape error when adding coordinates");

        let mid_12_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint_12.view())
            .expect("Shape error when adding coordinates");

        let mid_23_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint_23.view())
            .expect("Shape error when adding coordinates");

        let mid_30_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(midpoint_30.view())
            .expect("Shape error when adding coordinates");

        let center_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(center.view())
            .expect("Shape error when adding coordinates");

        new_conn.push(element_conn[0]);
        new_conn.push(mid_01_index);
        new_conn.push(center_index);
        new_conn.push(mid_30_index);

        new_conn.push(mid_01_index);
        new_conn.push(element_conn[1]);
        new_conn.push(mid_12_index);
        new_conn.push(center_index);

        new_conn.push(center_index);
        new_conn.push(mid_12_index);
        new_conn.push(element_conn[2]);
        new_conn.push(mid_23_index);

        new_conn.push(mid_30_index);
        new_conn.push(center_index);
        new_conn.push(mid_23_index);
        new_conn.push(element_conn[3]);
    }
    let new_conn_array = nd::Array2::from_shape_vec((new_conn_size, 4), new_conn)
        .expect("Shape error when building new connectivity array");

    new_mesh.add_regular_block(ElementType::QUAD4, new_conn_array.into_shared(), None);
}

fn split_tet4(mesh: &UMeshView, block: &crate::mesh::ElementBlockView, new_mesh: &mut UMesh) {
    let conn = match &block.connectivity {
        ConnectivityView::Regular(c) => c.view(),
        _ => return,
    };
    let new_conn_size = conn.nrows() * 4;
    let mut new_conn: Vec<usize> = Vec::with_capacity(new_conn_size * 4);

    for element_conn in conn.rows() {
        let coords_n0 = mesh.coords.row(element_conn[0]);
        let coords_n1 = mesh.coords.row(element_conn[1]);
        let coords_n2 = mesh.coords.row(element_conn[2]);
        let coords_n3 = mesh.coords.row(element_conn[3]);

        let centroid = (coords_n0.to_owned()
            + coords_n1.to_owned()
            + coords_n2.to_owned()
            + coords_n3.to_owned())
            / 4.0;

        let centroid_index = new_mesh.coords().nrows();
        new_mesh
            .append_coord(centroid.view())
            .expect("Shape error when adding coordinates");

        new_conn.push(element_conn[1]);
        new_conn.push(element_conn[2]);
        new_conn.push(element_conn[3]);
        new_conn.push(centroid_index);

        new_conn.push(element_conn[0]);
        new_conn.push(element_conn[3]);
        new_conn.push(element_conn[2]);
        new_conn.push(centroid_index);

        new_conn.push(element_conn[0]);
        new_conn.push(element_conn[1]);
        new_conn.push(element_conn[3]);
        new_conn.push(centroid_index);

        new_conn.push(element_conn[0]);
        new_conn.push(element_conn[2]);
        new_conn.push(element_conn[1]);
        new_conn.push(centroid_index);
    }
    let new_conn_array = nd::Array2::from_shape_vec((new_conn_size, 4), new_conn)
        .expect("Shape error when building new connectivity array");

    new_mesh.add_regular_block(ElementType::TET4, new_conn_array.into_shared(), None);
}

fn split_hex8(mesh: &UMeshView, block: &crate::mesh::ElementBlockView, new_mesh: &mut UMesh) {
    let conn = match &block.connectivity {
        ConnectivityView::Regular(c) => c.view(),
        _ => return,
    };
    let new_conn_size = conn.nrows() * 8;
    let mut new_conn: Vec<usize> = Vec::with_capacity(new_conn_size * 8);

    for element_conn in conn.rows() {
        let c0 = mesh.coords.row(element_conn[0]);
        let c1 = mesh.coords.row(element_conn[1]);
        let c2 = mesh.coords.row(element_conn[2]);
        let c3 = mesh.coords.row(element_conn[3]);
        let c4 = mesh.coords.row(element_conn[4]);
        let c5 = mesh.coords.row(element_conn[5]);
        let c6 = mesh.coords.row(element_conn[6]);
        let c7 = mesh.coords.row(element_conn[7]);

        let m01 = (c0.to_owned() + c1.to_owned()) / 2.0;
        let m12 = (c1.to_owned() + c2.to_owned()) / 2.0;
        let m23 = (c2.to_owned() + c3.to_owned()) / 2.0;
        let m30 = (c3.to_owned() + c0.to_owned()) / 2.0;
        let m45 = (c4.to_owned() + c5.to_owned()) / 2.0;
        let m56 = (c5.to_owned() + c6.to_owned()) / 2.0;
        let m67 = (c6.to_owned() + c7.to_owned()) / 2.0;
        let m74 = (c7.to_owned() + c4.to_owned()) / 2.0;
        let m04 = (c0.to_owned() + c4.to_owned()) / 2.0;
        let m15 = (c1.to_owned() + c5.to_owned()) / 2.0;
        let m26 = (c2.to_owned() + c6.to_owned()) / 2.0;
        let m37 = (c3.to_owned() + c7.to_owned()) / 2.0;

        let f0123 = (c0.to_owned() + c1.to_owned() + c2.to_owned() + c3.to_owned()) / 4.0;
        let f4567 = (c4.to_owned() + c5.to_owned() + c6.to_owned() + c7.to_owned()) / 4.0;
        let f0154 = (c0.to_owned() + c1.to_owned() + c5.to_owned() + c4.to_owned()) / 4.0;
        let f1265 = (c1.to_owned() + c2.to_owned() + c6.to_owned() + c5.to_owned()) / 4.0;
        let f2376 = (c2.to_owned() + c3.to_owned() + c7.to_owned() + c6.to_owned()) / 4.0;
        let f3074 = (c3.to_owned() + c0.to_owned() + c4.to_owned() + c7.to_owned()) / 4.0;

        let cv = (c0.to_owned()
            + c1.to_owned()
            + c2.to_owned()
            + c3.to_owned()
            + c4.to_owned()
            + c5.to_owned()
            + c6.to_owned()
            + c7.to_owned())
            / 8.0;

        let mut idx = Vec::with_capacity(19);
        let new_nodes = [
            m01, m12, m23, m30, m45, m56, m67, m74, m04, m15, m26, m37, f0123, f4567, f0154, f1265,
            f2376, f3074, cv,
        ];
        for node in new_nodes.iter() {
            let node_idx = new_mesh.coords().nrows();
            new_mesh.append_coord(node.view()).expect("Shape error");
            idx.push(node_idx);
        }

        let (im01, im12, im23, im30, im45, im56, im67, im74, im04, im15, im26, im37) = (
            idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7], idx[8], idx[9],
            idx[10], idx[11],
        );
        let (if0123, if4567, if0154, if1265, if2376, if3074, icv) = (
            idx[12], idx[13], idx[14], idx[15], idx[16], idx[17], idx[18],
        );

        new_conn.extend_from_slice(&[
            element_conn[0],
            im01,
            if0123,
            im30,
            im04,
            if0154,
            icv,
            if3074,
        ]);
        new_conn.extend_from_slice(&[
            im01,
            element_conn[1],
            im12,
            if0123,
            if0154,
            im15,
            icv,
            if1265,
        ]);
        new_conn.extend_from_slice(&[
            im12,
            element_conn[2],
            im23,
            if0123,
            if1265,
            im26,
            icv,
            if2376,
        ]);
        new_conn.extend_from_slice(&[
            im23,
            element_conn[3],
            im30,
            if0123,
            if2376,
            im37,
            icv,
            if3074,
        ]);
        new_conn.extend_from_slice(&[
            element_conn[4],
            im45,
            if4567,
            im74,
            im04,
            if0154,
            icv,
            if3074,
        ]);
        new_conn.extend_from_slice(&[
            im45,
            element_conn[5],
            im56,
            if4567,
            if0154,
            im15,
            icv,
            if1265,
        ]);
        new_conn.extend_from_slice(&[
            im56,
            element_conn[6],
            im67,
            if4567,
            if1265,
            im26,
            icv,
            if2376,
        ]);
        new_conn.extend_from_slice(&[
            im67,
            element_conn[7],
            im74,
            if4567,
            if2376,
            im37,
            icv,
            if3074,
        ]);
    }

    let new_conn_array = nd::Array2::from_shape_vec((new_conn_size, 8), new_conn)
        .expect("Shape error when building new connectivity array");

    new_mesh.add_regular_block(ElementType::HEX8, new_conn_array.into_shared(), None);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, UMesh};
    use ndarray as nd;

    #[test]
    fn test_split_empty_mesh() {
        let coords = nd::ArcArray2::zeros((0, 2));
        let mesh = UMesh::new(coords);
        let split_mesh = split(mesh.view());

        assert_eq!(split_mesh.num_elements(), 0);
        assert_eq!(split_mesh.coords.nrows(), 0);
    }

    #[test]
    fn test_split_single_seg2() {
        let space_dimension = 2;
        let coords =
            nd::ArcArray2::from_shape_vec((2, space_dimension), vec![0., 0., 1., 0.]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::Array2::from_shape_vec((1, 2), vec![0, 1]) // One element with two nodes (node 0 and node 1)
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        assert_eq!(splitted_mesh.num_elements(), 2);
        assert_eq!(splitted_mesh.coords.nrows(), 3);

        let tol = 1e-10;
        assert!(splitted_mesh.coords.row(0)[0] < tol);
        assert!(splitted_mesh.coords.row(0)[1] < tol);
        assert!((splitted_mesh.coords.row(1)[0] - 1.).abs() < tol);
        assert!(splitted_mesh.coords.row(1)[1] < tol);
        assert!((splitted_mesh.coords.row(2)[0] - 0.5).abs() < tol);
        assert!(splitted_mesh.coords.row(2)[1] < tol);
    }

    #[test]
    fn test_split_single_tri3() {
        let space_dimension = 2;
        let coords =
            nd::ArcArray2::from_shape_vec((3, space_dimension), vec![0., 0., 3., 0., 0., 3.])
                .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::TRI3,
            nd::Array2::from_shape_vec((1, 3), vec![0, 1, 2])
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        assert_eq!(splitted_mesh.num_elements(), 3);
        assert_eq!(splitted_mesh.coords.nrows(), 4);

        let tol = 1e-10;
        assert!(splitted_mesh.coords.row(0)[0] < tol);
        assert!(splitted_mesh.coords.row(0)[1] < tol);
        assert!((splitted_mesh.coords.row(1)[0] - 3.).abs() < tol);
        assert!(splitted_mesh.coords.row(1)[1] < tol);
        assert!(splitted_mesh.coords.row(2)[0] < tol);
        assert!((splitted_mesh.coords.row(2)[1] - 3.).abs() < tol);
        assert!((splitted_mesh.coords.row(3)[0] - 1.).abs() < tol);
        assert!((splitted_mesh.coords.row(3)[1] - 1.).abs() < tol);
    }

    #[test]
    fn test_split_single_quad4() {
        let space_dimension = 2;
        // Create a simple square with 4 nodes
        let coords = nd::ArcArray2::from_shape_vec(
            (4, space_dimension),
            vec![0., 0., 1., 0., 1., 1., 0., 1.],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::Array2::from_shape_vec((1, 4), vec![0, 1, 2, 3])
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        assert_eq!(splitted_mesh.num_elements(), 4);
        assert_eq!(splitted_mesh.coords.nrows(), 9);

        let tol = 1e-10;
        assert!(splitted_mesh.coords.row(0)[0] < tol);
        assert!(splitted_mesh.coords.row(0)[1] < tol);
        assert!(splitted_mesh.coords.row(1)[0] - 1. < tol);
        assert!(splitted_mesh.coords.row(1)[1] < tol);
        assert!(splitted_mesh.coords.row(2)[0] - 1. < tol);
        assert!(splitted_mesh.coords.row(2)[1] - 1. < tol);
        assert!(splitted_mesh.coords.row(3)[0] < tol);
        assert!(splitted_mesh.coords.row(3)[1] - 1. < tol);

        assert!((splitted_mesh.coords.row(4)[0] - 0.5).abs() < tol);
        assert!(splitted_mesh.coords.row(4)[1] < tol);

        assert!(splitted_mesh.coords.row(5)[0] - 1. < tol);
        assert!((splitted_mesh.coords.row(5)[1] - 0.5).abs() < tol);

        assert!((splitted_mesh.coords.row(6)[0] - 0.5).abs() < tol);
        assert!(splitted_mesh.coords.row(6)[1] - 1. < tol);

        assert!(splitted_mesh.coords.row(7)[0] < tol);
        assert!((splitted_mesh.coords.row(7)[1] - 0.5).abs() < tol);

        assert!((splitted_mesh.coords.row(8)[0] - 0.5).abs() < tol);
        assert!((splitted_mesh.coords.row(8)[1] - 0.5).abs() < tol);
    }

    #[test]
    fn test_split_single_tet4() {
        let space_dimension = 3;
        let coords = nd::ArcArray2::from_shape_vec(
            (4, space_dimension),
            vec![0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::TET4,
            nd::Array2::from_shape_vec((1, 4), vec![0, 1, 2, 3])
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        assert_eq!(splitted_mesh.num_elements(), 4);
        assert_eq!(splitted_mesh.coords.nrows(), 5);

        let tol = 1e-10;
        // Check centroid (should be at index 4)
        assert!((splitted_mesh.coords.row(4)[0] - 0.25).abs() < tol);
        assert!((splitted_mesh.coords.row(4)[1] - 0.25).abs() < tol);
        assert!((splitted_mesh.coords.row(4)[2] - 0.25).abs() < tol);
    }

    #[test]
    fn test_split_single_hex8() {
        let space_dimension = 3;
        let coords = nd::ArcArray2::from_shape_vec(
            (8, space_dimension),
            vec![
                0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1., 1., 1.,
                0., 1., 1.,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::HEX8,
            nd::Array2::from_shape_vec((1, 8), vec![0, 1, 2, 3, 4, 5, 6, 7])
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        assert_eq!(splitted_mesh.num_elements(), 8);
        // Original 8 nodes + 19 new nodes (6 faces + 12 edges + center) = 27 nodes
        assert_eq!(splitted_mesh.coords.nrows(), 27);

        let tol = 1e-10;
        // Check centroid (should be at index 26)
        assert!((splitted_mesh.coords.row(26)[0] - 0.5).abs() < tol);
        assert!((splitted_mesh.coords.row(26)[1] - 0.5).abs() < tol);
        assert!((splitted_mesh.coords.row(26)[2] - 0.5).abs() < tol);
    }
}
