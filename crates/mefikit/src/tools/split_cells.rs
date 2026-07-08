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
        assert!(splitted_mesh.coords.row(1)[0] - 1. < tol);
        assert!(splitted_mesh.coords.row(1)[1] < tol);
        assert!(splitted_mesh.coords.row(2)[0] - 0.5 < tol);
        assert!(splitted_mesh.coords.row(2)[1] < tol);
    }
}
