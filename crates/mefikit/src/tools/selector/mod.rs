//! Element and node selection utilities.
//!
//! Provides a domain-specific language for selecting mesh elements and nodes
//! based on geometric, topological, and field-based criteria.

mod centroid;
mod element;
mod field;
mod group;
mod node;
pub mod selection;

pub use selection as sel;
pub use selection::{MeshSelect, Selection};
