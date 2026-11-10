//! Shared test-only mesh builders.

use ndarray as nd;

use crate::mesh::{ElementType, UMesh};
use crate::tools::RegularUMeshBuilder;

pub(crate) fn box_mesh(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis(xs.to_vec())
        .add_axis(ys.to_vec())
        .add_axis(zs.to_vec())
        .build()
}

/// Structured hex mesh over the node grid `xs × ys × zs` keeping only the cells listed in
/// `cells` as `(i, j, k)` grid indices. Nodes are shared between neighbouring cells, and
/// the `HEX8` winding matches [`RegularUMeshBuilder`].
pub(crate) fn hex_block(
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    cells: &[(usize, usize, usize)],
) -> UMesh {
    let (nx, ny) = (xs.len(), ys.len());
    let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    let flat: Vec<f64> = (0..zs.len())
        .flat_map(|k| {
            ys.iter()
                .flat_map(move |y| xs.iter().map(move |x| [*x, *y, zs[k]]))
        })
        .flatten()
        .collect();
    let mut mesh = UMesh::new(
        nd::Array2::from_shape_vec((nx * ny * zs.len(), 3), flat)
            .unwrap()
            .into_shared(),
    );
    for &(i, j, k) in cells {
        mesh.add_element(
            ElementType::HEX8,
            &[
                node(i, j, k),
                node(i + 1, j, k),
                node(i + 1, j + 1, k),
                node(i, j + 1, k),
                node(i, j, k + 1),
                node(i + 1, j, k + 1),
                node(i + 1, j + 1, k + 1),
                node(i, j + 1, k + 1),
            ],
            None,
        );
    }
    mesh
}
