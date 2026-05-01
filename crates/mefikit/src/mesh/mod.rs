//! Core mesh data structures for unstructured mesh manipulation.
//!
//! This module provides the fundamental types for representing unstructured meshes,
//! including connectivity, element blocks, fields, and the main [`UMesh`] type.

mod connectivity;
mod dimension;
mod element;
mod element_block;
mod element_ids;
mod element_ids_set;
mod fields;
mod indirect_index;
mod umesh;

pub use connectivity::Connectivity;
pub use dimension::Dimension;
pub use element::{Element, ElementId, ElementLike, ElementMut, ElementType, Regularity};
pub use element_ids::ElementIds;
pub use element_ids_set::ElementIdsSet;
pub use fields::{
    FieldArc, FieldArcD, FieldBase, FieldCow, FieldCowD, FieldOwned, FieldOwnedD, FieldView,
    FieldViewD,
};
pub use indirect_index::{
    IndirectIndexIntoIter, IndirectIndexIter, IndirectIndexIterMut, IndirectIndexOwned,
    IndirectIndexShared, IndirectIndexView,
};
pub use umesh::{UMesh, UMeshBase, UMeshView};
