// This module builds a mesh of one dimension higher than the input mesh by extuding it.
// Duplicated nodes are allowed, both in the original mesh and the 1d mesh.

use crate::{ElementType, UMesh, UMeshView};
use ndarray::{ArcArray2, Array2};

trait MeshTransform {
    fn extrude_along(&self, path: &UMesh) -> UMesh;
    fn duplicate_with(&self, vectors: &[[f64;3]]) -> UMesh;
}

impl MeshTransform for UMesh {
    todo!();
}
