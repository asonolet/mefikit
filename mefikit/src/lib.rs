/// This module groups all algorithms operating on one or more meshes.
///
/// Most of the algorithms take a &UMesh when using optimizations (sharing coordinates) or a
/// UMeshView when not needed and produce a new owned UMesh.
pub mod algorithms;
/// This module defines mesh builders, ie creational methods which does not modify or extend an
/// existing mesh.
pub mod builders;
/// This module defines geometrical operations on elements.
///
/// The operations are provided through the `ElementGeo` trait.
pub mod geometry;
/// This module defines a `read` and a `write` functions that can use various mesh formats
mod io;
/// This module serves as the **central container** for all mesh-related data and logic in the
/// library.
///
/// ---
///
/// # `umesh` — Core Unstructured Mesh Representation
///
/// The `umesh` module defines the core mesh structure used for all operations in
/// this crate. It is responsible for organizing heterogeneous collections of
/// finite elements, topological data (connectivity), and associated fields (e.g.,
/// physical quantities).
///
/// This module is the foundation for more advanced geometric, topological, and
/// numerical operations built elsewhere in the library.
///
/// ---
///
/// ## 🧱 Structure: `UMesh`
///
/// Schematically, a `UMesh` consists of:
///
/// ```rust
/// pub struct UMesh {
///     coords: ArcArray2<f64>,
///     element_blocks: BTreeMap<ElementType, ElementBlock>
/// }
/// ```
///
/// Each `UMesh` contains a set of `ElementBlock`s and a coordinates array. An
/// `ElementBlock` groups together mesh elements of the same type
/// (e.g., all TRI3 or all TETRA4 elements).
///
/// This design enables:
/// - Heterogeneous element types (mixing 0D, 1D, 2D, 3D elements)
/// - Homogeneous coordinate storage (1D, 2D, or 3D points)
/// - Per-block field and group storage
/// - Efficient traversal and filtering based on element type or dimension
///
/// ---
///
/// ## 🔗 `ElementBlock`
///
/// Each block contains:
/// - Element **type** (e.g., QUAD4, TETRA4, etc.)
/// - **Topological connectivity** (indexing into a node list)
/// - **Fields** associated with the elements (e.g., temperature, velocity)
/// - **Groups** associated with the elements (through families)
///
/// ```rust
/// pub struct ElementBlock {
///     pub element_type: ElementType,
///     pub connectivity: Connectivity,
///     pub fields: BTreeMap<String, ArrayD<f64>>,
///     families: Vec<usize>,
///     pub groups: BTreeMap<String, BTreeSet<usize>>,
/// }
/// ```
///
/// Field data is stored as `ndarray::ArrayD<f64>`, and identified by a naming
/// convention supporting time-dependent fields, e.g.:
///
/// ```
/// "temperature_iter_3_time_0.01"
/// ```
///
/// Group data is stored as families groups. Families always form a partition of
/// the elements based on the groups (ie two element not pertaining to the same
/// groups have a different family id).
///
/// ---
///
/// ## 🔁 Iteration & Access
///
/// The `umesh` module provides building blocks for:
/// - Iterating over elements by block
/// - Filtering blocks by dimension or type
/// - Querying and extracting field data
/// - Querying, extracting, updating group related data
/// - Operating on mesh views
///
/// ---
///
/// ## 🎯 Design Goals
///
/// - Clean separation between **geometry**, **topology**, **field data** and **group data**
/// - Support for real-world meshes with mixed element types and poly element types
/// - Efficient per-block and global access patterns
/// - Compatibility with operations like:
///   - Interpolation
///   - Remapping
///   - Merging / conformizing
///   - File I/O (MED, CGNS, etc.)
///
/// ---
///
/// ## 🔄 Mesh Ownership and Views: `UMesh`, `UMeshView`, and `UMeshBase`
///
/// MeFiKit uses a very flexible ownership data model based on ndarray ownership
/// model. Each array can be OwnedRepr or ViewRepr (for connectivity array,
/// coordinates array, all fields and all groups). This makes integration with
/// other systems (e.g., C or Python) simple and safe, while also enabling
/// high-performance zero-copy operations.
/// This memory model brings a bit of complexity but allows for great flexibility.
///
/// ### 📦 `UMesh` – Owned Mesh
///
/// `UMesh` owns its data:
/// - Coordinate array (`ArcArray2<f64>`)
/// - Element blocks with connectivities, fields, families, groups are composed of
///   owned arrays (`Array2<usize>`, `Array1<usize>`, `ArrayD<f64>`, etc.)
///
/// This type is used for:
/// - Internal mesh manipulation in Rust
/// - Persistent mesh storage
/// - File I/O
/// - Long-term computation or modification
///
/// It is constructed by cloning or transferring ownership of arrays.
/// It has its own constructors and methods for adding/removing elements,
/// fields, and groups.
///
/// ---
///
/// ### 👁️ `UMeshView<'a>` – Read-Only Mesh View
///
/// `UMeshView` is a **zero-copy, non-owning** view over existing mesh data. It
/// holds references (slices or `ndarray::ArrayView`) to continuous memory block
/// that is managed elsewhere (e.g., passed from a C/C++/Python array or owned
/// by a Rust UMesh).
///
/// This is ideal for:
/// - Foreign function interface (FFI)
/// - Avoiding unnecessary copies when no modification is needed
///
/// ⚠️ Lifetimes must be respected — `UMeshView` is only valid as long as the
/// referenced data lives.
///
/// All out-of-place operations are implemented on `UMeshView`, allowing them
/// to be applied to both owned meshes and foreign/borrowed data.
///
/// ---
///
/// ### 🔄 Summary
///
/// | Type           | Ownership | Mutable | Use Case                                 | Copies |
/// |----------------|-----------|---------|------------------------------------------|--------|
/// | `UMesh`        | Yes       | Yes     | Full ownership, long-term usage          | Yes    |
/// | `UMeshView`    | No        | No      | Read-only access to foreign/borrowed data| No     |
///
/// This model ensures performance, safety, and clear interoperability boundaries.
///
/// ---
///
/// ## Copy-on-write Coordinate Storage
///
/// In unstructured mesh representations, it's common for multiple mesh objects or
/// views to share the same set of node coordinates. However, some operations (like
/// connectivity modification or group and field manipulation) can be done in-place,
/// while others (like node pruning, reordering or coordinate transformations)
/// require the array to diverge.
/// To ensure safety and performance, coordinates data are wrapped into an ArcArray2
/// when owned.
/// A more fine-grained copy-on-write mechanism could be implemented in the future if
/// needed. It could for example allow in-place shared mutability of the coordinates when the user
/// really knows what he is doing.
///
/// ---
///
/// ## 🛠️ In-Place vs. Out-of-Place Operations in `UMesh`
///
/// This document categorizes common mesh operations into two groups:
///
/// - **Out-of-Place Operations**: These typically rewrite the mesh's core
///   structures (e.g., connectivity) and are better implemented as
///   producing a new mesh.
/// - **In-Place Operations**: These can be applied directly to an existing `UMesh`
///   or its view without reallocating major structures. They are usued for initial manual mesh
///   creation and are lower level (potentially not safe to user actions).
///
///
/// ### Out-of-Place Operations (produce new mesh, valid on UMeshBase)
/// These operations fundamentally change the mesh's topology, usually requiring
/// reallocation of connectivity tables. As they operate on views, they can be applied to both
/// `UMesh` and `UMeshView`. They return a new `UMesh`. This approach is the safest and clearest.
///
/// | Operation                  | Description |
/// |----------------------------|-------------|
/// | `renumber_cells()`         | In-place reordering of cells. Out-of-place because of Poly |
/// | `compute_submesh()`          | Returns a new mesh composed of subentities depending on codim |
/// | `conformize(mesh)`         | Cleans internal inconsistencies, requires deep topology rewrite |
/// | `split_by(mesh_a, mesh_b)` | Cuts mesh A using B's topology, creates new elements |
/// | `fuse_meshes(a, b)`        | Boolean union with topological merging, produces a new mesh |
/// | `intersect_meshes(a, b)`   | Keeps overlapping parts of two meshes, new geometry required |
/// | `substract_with(a, b)`     | Subtracts domain of B from A, with new elements generated |
///
///
/// ### In-Place Operations
/// These operations can safely modify the owned mesh structure in-place. They might be more
/// fine-grained, powerfull and efficient. BUT they are only valid on `UMesh` (not `UMeshView`),
/// and they should only be used when the out-of-place approach was not possible or not performant
/// enough.
///
/// | Operation                 | Description |
/// |---------------------------|-------------|
/// | `assign_field`            | Adds or replaces field data on a block or selection |
/// | `set_family`              | Sets the family (zone/subdomain tag) of elements |
/// | `set_group`               | Modifies or defines a group of element IDs under a name |
/// | `renumber_nodes`          | In-place reordering of coordinates and node indices |
/// | `merge_close_nodes`       | Mutates coordinates and connectivity to merge nearby points |
/// | `set_coordinates`         | Mutate existing geometry without changing topology |
/// | `transform_coordinates`   | Apply affine transformation to node coordinates |
///
/// ---
///
/// ## 📚 Related Modules
///
/// - `geometry`, `topology`, `intersect` — operation-specific logic
/// - `io` — file import/export (serde_json, serde_yaml, MED, CGNS, etc.)
pub mod mesh;
#[cfg(test)]
pub mod mesh_examples;
/// This module defines topological operations on elements.
///
/// The operations are provided through the `ElementTopo` trait.
pub mod topology;

pub mod prelude {
    pub use crate::algorithms::*;
    pub use crate::builders::RegularUMeshBuilder;
    pub use crate::geometry::ElementGeo;
    pub use crate::io::{read, write};
    pub use crate::mesh::{
        Dimension, Element, ElementId, ElementIds, ElementLike, ElementMut, ElementType,
        Regularity, UMesh, UMeshBase, UMeshView,
    };
    pub use crate::topology::ElementTopo;
}
