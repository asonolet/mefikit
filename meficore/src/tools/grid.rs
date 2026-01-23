use crate::mesh::{ElementType, UMesh};
use ndarray::{ArcArray2, Array2};

/// Regular umesh builder (1d, 2d or 3d).
///
/// This is a convenience struct to build a UMesh with regular coordinates along the axes.
/// The order of the axes is important, as it determines the shape of the mesh.
/// For example, if you add two axes, the first one will be the x-axis and the second one will be
/// the y-axis.
/// The nodes will be created by taking the Cartesian product of the axes.
/// The nodes index augment first along the first axis, then along the second axis, etc.
/// That way, adding another axis does not change the order of the first nodes.
///
/// For example, if you add the axes [0.0, 1.0, 2.0] and [0.0, 1.0], the resulting mesh will have
/// the nodes:
/// ```text
/// (0.0, 0.0)
/// (1.0, 0.0)
/// (2.0, 0.0)
/// (0.0, 1.0)
/// (1.0, 1.0)
/// (2.0, 1.0)
/// ```
/// And the elements will be the quadrilaterals formed by these nodes.
/// The connectivity will be the following:
/// ```text
/// (0, 3, 4, 1)
/// (1, 4, 5, 2)
/// ```
///
/// ```text
///  3           4           5
///  +-----------+-----------+
///  |           |           |
///  |           |           |
///  |     0     |     1     |
///  |           |           |
///  |           |           |
///  +-----------+-----------+
///  0           1           2
/// ```
pub struct RegularUMeshBuilder {
    coords_grid: Vec<Vec<f64>>,
}

impl Default for RegularUMeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl RegularUMeshBuilder {
    pub fn new() -> Self {
        Self {
            coords_grid: Vec::new(),
        }
    }

    pub fn add_axis(mut self, axis: Vec<f64>) -> Self {
        if self.coords_grid.len() < 3 {
            self.coords_grid.push(axis);
        } else {
            // If we already have two axes, we cannot add more
            panic!("Cannot add more than three axes to a regular mesh builder");
        }
        self
    }

    fn compute_connectivity(&self) -> Array2<usize> {
        let num_axes = self.coords_grid.len();
        let num_nodes: usize = self.coords_grid.iter().map(|axis| axis.len()).product();

        match num_axes {
            1 => {
                let connectivity = (0..num_nodes - 1).flat_map(|i| vec![i, i + 1]).collect();
                Array2::from_shape_vec((num_nodes - 1, 2), connectivity)
                    .expect("Failed to create 1D mesh connectivity")
            }
            2 => {
                let x_len = self.coords_grid[0].len();
                let y_len = self.coords_grid[1].len();
                let connectivity = (0..(x_len - 1) * (y_len - 1))
                    .flat_map(|i| {
                        let y_index = i / (x_len - 1);
                        let x_index = i % (x_len - 1);
                        vec![
                            y_index * x_len + x_index,
                            y_index * x_len + x_index + 1,
                            (y_index + 1) * x_len + x_index + 1,
                            (y_index + 1) * x_len + x_index,
                        ]
                    })
                    .collect();
                Array2::from_shape_vec(((x_len - 1) * (y_len - 1), 4), connectivity)
                    .expect("Failed to create 2D mesh connectivity")
            }
            3 => {
                let x_len = self.coords_grid[0].len();
                let y_len = self.coords_grid[1].len();
                let z_len = self.coords_grid[2].len();
                let connectivity = (0..(x_len - 1) * (y_len - 1) * (z_len - 1))
                    .flat_map(|i| {
                        let xy_plane_size = (x_len - 1) * (y_len - 1);
                        let z_index = i / xy_plane_size;
                        let xy_index = i % xy_plane_size;
                        let y_index = xy_index / (x_len - 1);
                        let x_index = xy_index % (x_len - 1);
                        vec![
                            z_index * (x_len * y_len) + y_index * x_len + x_index,
                            z_index * (x_len * y_len) + y_index * x_len + x_index + 1,
                            z_index * (x_len * y_len) + (y_index + 1) * x_len + x_index + 1,
                            z_index * (x_len * y_len) + (y_index + 1) * x_len + x_index,
                            (z_index + 1) * (x_len * y_len) + y_index * x_len + x_index,
                            (z_index + 1) * (x_len * y_len) + y_index * x_len + x_index + 1,
                            (z_index + 1) * (x_len * y_len) + (y_index + 1) * x_len + x_index + 1,
                            (z_index + 1) * (x_len * y_len) + (y_index + 1) * x_len + x_index,
                        ]
                    })
                    .collect();
                Array2::from_shape_vec(((x_len - 1) * (y_len - 1) * (z_len - 1), 8), connectivity)
                    .expect("Failed to create 3D mesh connectivity")
            }
            _ => panic!("Cannot create connectivity for more than 3 axes"),
        }
    }

    fn compute_coords(&self) -> Array2<f64> {
        let num_axes: usize = self.coords_grid.len();

        match num_axes {
            1 => {
                Array2::from_shape_vec((self.coords_grid[0].len(), 1), self.coords_grid[0].clone())
                    .expect("Failed to create 1D mesh coordinates")
            }
            2 => {
                let coords = self.coords_grid[1]
                    .iter()
                    .flat_map(|y| self.coords_grid[0].iter().flat_map(move |x| vec![*x, *y]))
                    .collect();
                Array2::from_shape_vec(
                    (self.coords_grid[0].len() * self.coords_grid[1].len(), 2),
                    coords,
                )
                .expect("Failed to create 2D mesh coordinates")
            }
            3 => Array2::from_shape_vec(
                (
                    self.coords_grid[0].len()
                        * self.coords_grid[1].len()
                        * self.coords_grid[2].len(),
                    3,
                ),
                self.coords_grid[2]
                    .iter()
                    .flat_map(|z| {
                        self.coords_grid[1].iter().flat_map(move |y| {
                            self.coords_grid[0]
                                .iter()
                                .flat_map(move |x| vec![*x, *y, *z])
                        })
                    })
                    .collect(),
            )
            .expect("Failed to create 3D mesh coordinates"),
            _ => panic!("Cannot create coordinates for more than 3 axes"),
        }
    }

    pub fn build(self) -> UMesh {
        let coords = self.compute_coords();
        let coords_dim = coords.shape()[1];
        let connectivity = self.compute_connectivity();

        let mut umesh = UMesh::new(ArcArray2::from(coords));
        if coords_dim == 1 {
            // 1D mesh
            umesh.add_regular_block(ElementType::SEG2, connectivity.to_shared());
        } else if coords_dim == 2 {
            // 2D mesh
            umesh.add_regular_block(ElementType::QUAD4, connectivity.to_shared());
        } else if coords_dim == 3 {
            // 3D mesh
            umesh.add_regular_block(ElementType::HEX8, connectivity.to_shared());
        } else {
            panic!("Unsupported number of dimensions for regular mesh");
        }
        umesh
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Connectivity;
    use crate::mesh::ElementType;

    #[test]
    fn test_regular_mesh_builder_1d() {
        let builder = RegularUMeshBuilder::new().add_axis(vec![0.0, 1.0, 2.0]);
        let mesh = builder.build();
        assert_eq!(mesh.coords().shape(), &[3, 1]);
        assert!(mesh.block(ElementType::SEG2).is_some());
    }

    #[test]
    fn test_regular_mesh_builder_2d() {
        let builder = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0, 2.0])
            .add_axis(vec![0.0, 1.0]);
        let mesh = builder.build();
        assert_eq!(mesh.coords().shape(), &[6, 2]);
        assert!(mesh.block(ElementType::QUAD4).is_some());
        assert_eq!(
            match &mesh.block(ElementType::QUAD4).unwrap().connectivity {
                Connectivity::Regular(conn) => conn.shape(),
                _ => panic!("Expected regular connectivity"),
            },
            &[2, 4]
        );
    }

    #[test]
    fn test_regular_mesh_builder_3d() {
        let builder = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0, 2.0])
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0, 2.0]);
        let mesh = builder.build();
        assert_eq!(mesh.coords().shape(), &[18, 3]);
        assert!(mesh.block(ElementType::HEX8).is_some());
        assert_eq!(
            match &mesh.block(ElementType::HEX8).unwrap().connectivity {
                Connectivity::Regular(conn) => conn.shape(),
                _ => panic!("Expected regular connectivity"),
            },
            &[4, 8]
        );
    }
}
