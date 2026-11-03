//! This module converts all cells from one UMesh to simplex into a new mesh.
use crate::element_traits::ElementTopo;
use crate::mesh::{Dimension, UMesh, UMeshView};

pub fn simplexize(mesh: &UMeshView, _dim: Option<Dimension>) -> UMesh {
    let mut new_mesh = UMesh::new(mesh.coords().to_shared());

    for elem in mesh.elements() {
        let ets = elem.to_simplexes();
        for (et, co) in ets {
            new_mesh.add_element(et, &co, None);
        }
    }
    new_mesh
}
