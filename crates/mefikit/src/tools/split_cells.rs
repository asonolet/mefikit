use crate::mesh::{ConnectivityView, ElementType, UMesh, UMeshView};
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
            _ => {
                // For unsupported element types, copy them as-is
                if let ConnectivityView::Regular(conn) = &block.connectivity {
                    new_mesh.add_regular_block(element_type, conn.to_shared(), None);
                }
            }
        }
    }

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

        // Calculate midpoints for each edge
        let midpoint_01 = (coords_n0.to_owned() + coords_n1.to_owned()) / 2.0;
        let midpoint_12 = (coords_n1.to_owned() + coords_n2.to_owned()) / 2.0;
        let midpoint_23 = (coords_n2.to_owned() + coords_n3.to_owned()) / 2.0;
        let midpoint_30 = (coords_n3.to_owned() + coords_n0.to_owned()) / 2.0;

        // Calculate center point
        let center = (coords_n0.to_owned()
            + coords_n1.to_owned()
            + coords_n2.to_owned()
            + coords_n3.to_owned())
            / 4.0;

        // Add new nodes to the mesh
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

        // Create 4 new QUAD4 elements
        // Bottom-left quad: n0, mid_01, center, mid_30
        new_conn.push(element_conn[0]);
        new_conn.push(mid_01_index);
        new_conn.push(center_index);
        new_conn.push(mid_30_index);

        // Bottom-right quad: mid_01, n1, mid_12, center
        new_conn.push(mid_01_index);
        new_conn.push(element_conn[1]);
        new_conn.push(mid_12_index);
        new_conn.push(center_index);

        // Top-right quad: center, mid_12, n2, mid_23
        new_conn.push(center_index);
        new_conn.push(mid_12_index);
        new_conn.push(element_conn[2]);
        new_conn.push(mid_23_index);

        // Top-left quad: mid_30, center, mid_23, n3
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
            nd::Array2::from_shape_vec((1, 4), vec![0, 1, 2, 3]) // One quad element
                .unwrap()
                .to_shared(),
            None,
        );

        let splitted_mesh = split(mesh.view());

        // Should create 4 new QUAD4 elements
        assert_eq!(splitted_mesh.num_elements(), 4);
        assert_eq!(splitted_mesh.coords.nrows(), 9);

        let tol = 1e-10;
        // Check original nodes are preserved
        assert!(splitted_mesh.coords.row(0)[0] < tol);
        assert!(splitted_mesh.coords.row(0)[1] < tol);
        assert!(splitted_mesh.coords.row(1)[0] - 1. < tol);
        assert!(splitted_mesh.coords.row(1)[1] < tol);
        assert!(splitted_mesh.coords.row(2)[0] - 1. < tol);
        assert!(splitted_mesh.coords.row(2)[1] - 1. < tol);
        assert!(splitted_mesh.coords.row(3)[0] < tol);
        assert!(splitted_mesh.coords.row(3)[1] - 1. < tol);

        // Check midpoint between node 0 and 1 (should be at index 4)
        assert!((splitted_mesh.coords.row(4)[0] - 0.5).abs() < tol);
        assert!(splitted_mesh.coords.row(4)[1] < tol);

        // Check midpoint between node 1 and 2 (should be at index 5)
        assert!(splitted_mesh.coords.row(5)[0] - 1. < tol);
        assert!((splitted_mesh.coords.row(5)[1] - 0.5).abs() < tol);

        // Check midpoint between node 2 and 3 (should be at index 6)
        assert!((splitted_mesh.coords.row(6)[0] - 0.5).abs() < tol);
        assert!(splitted_mesh.coords.row(6)[1] - 1. < tol);

        // Check midpoint between node 3 and 0 (should be at index 7)
        assert!(splitted_mesh.coords.row(7)[0] < tol);
        assert!((splitted_mesh.coords.row(7)[1] - 0.5).abs() < tol);

        // Check center point (should be at index 8)
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
}
