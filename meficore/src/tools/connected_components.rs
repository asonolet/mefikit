use petgraph::algo::kosaraju_scc;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::mesh::{Dimension, UMesh};
use crate::tools::compute_neighbours_graph;

pub fn compute_connected_components(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    link_dim: Option<Dimension>,
) -> Vec<UMesh> {
    let graph = compute_neighbours_graph(mesh, src_dim, link_dim);
    let compos = kosaraju_scc(&graph);
    #[cfg(feature = "rayon")]
    let res = compos
        .into_par_iter()
        .map(|compo| mesh.extract(&compo.into_iter().collect()))
        .collect();
    #[cfg(not(feature = "rayon"))]
    let res = compos
        .into_iter()
        .map(|compo| mesh.extract(&compo.into_iter().collect()))
        .collect();
    res
}

#[cfg(test)]
mod tests {
    use crate::mesh_examples::make_imesh_3d;
    use crate::prelude as mf;
    use crate::tools::connected_components::compute_connected_components;
    use crate::tools::{Descendable, MeshSelect, sel};

    #[test]
    fn test_connected_components() {
        let mesh = make_imesh_3d(20);
        let sphere1 = sel::sphere([0.35, 0.5, 0.5], 0.2);
        let sphere2 = sel::sphere([0.65, 0.5, 0.5], 0.2);
        let sphere3 = sel::sphere([0.5, 0.2, 0.2], 0.15);
        let (_, spheres) = mesh.select(sphere1 | sphere2 | sphere3);
        let bounds = spheres.boundaries(None, None);
        let cracked = mf::crack(mesh, bounds.view());
        let components = compute_connected_components(&cracked, None, None);
        assert_eq!(components.len(), 3);
    }
}
