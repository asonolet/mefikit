use ndarray::prelude::*;
use ndarray::ArcArray2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use todo;

use crate::umesh::element::{Element, ElementId, Regularity};
use crate::umesh::element_block::{ElementBlock, IntoElementBlockEntry};
use crate::umesh::selector::Selector;
use crate::umesh::umesh_core::UMesh;
use crate::umesh::ElementType;

/// An unstructured mesh.
///
/// This is the most general mesh format in mefikit, capable of describing any kind of mesh with
/// multiple element types and associated fields.
/// It is designed to be flexible and extensible, allowing for various element types and
/// properties.
/// This trait provides read-only access to the mesh data, allowing for efficient operations and
/// FFI compatibility.
pub trait UMeshAccess<'a> {
    type EBlock: 'a;
    type Coords: 'a;

    fn coords(&self) -> ArrayView2<'a, f64>;
    fn element_blocks(&self) -> &BTreeMap<ElementType, Self::EBlock>;
    fn element_block(&self, element_type: ElementType) -> Option<&Self::EBlock>;
    fn select_ids(&self) -> Selector;
    fn extract_mesh(&self, ids: &[ElementId]) -> UMesh;
}
