use petgraph::algo::tarjan_scc;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use crate::mesh::{Dimension, UMesh};
use crate::tools::neighbours::compute_neighbours;

pub fn compute_connected_components(
    mesh: &UMesh,
    src_dim: Option<Dimension>,
    link_dim: Option<Dimension>,
) -> Vec<UMesh> {
    let (_, graph) = compute_neighbours(mesh, src_dim, link_dim);
    let compos = tarjan_scc(&graph);
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
