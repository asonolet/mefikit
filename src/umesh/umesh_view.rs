use ndarray::prelude::*;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
/// An unstrustured mesh.
///
/// The most general mesh format in mefikit. Can describe any kind on mesh, with multiple elements kinds and fields associated.
pub struct UMeshView<'a> {
    coords: ArrayView2<'a, f64>,
    // element_blocks: BTreeMap<ElementType, ElementBlockView<'a>>,
}
