# `umesh` — Core Unstructured Mesh Representation

The `umesh` module defines the core mesh structure used for all operations in
this crate. It is responsible for organizing heterogeneous collections of
finite elements, topological data (connectivity), and associated fields (e.g.,
physical quantities).

This module is the foundation for more advanced geometric, topological, and
numerical operations built elsewhere in the library.

---

## 🧱 Structure: `UMesh`

```rust
pub struct UMesh {
    coords: ArcArray2<f64>,
    element_blocks: BTreeMap<ElementType, ElementBlock>
}
```

Each `UMesh` contains a list of `ElementBlock`s and a connectivity array. An
`ElementBlock` groups together mesh elements of the same type and dimension
(e.g., all TRI3 or all TETRA4 elements).

This design enables:
- Heterogeneous meshes (mixing 0D, 1D, 2D, 3D elements)
- Per-block field storage
- Efficient traversal and filtering based on element type or dimension
- Note that it is cheap to create another mesh based on the same coords

---

## 🔗 `ElementBlock`

Each block contains:
- Element **type** (e.g., QUAD4, TETRA4, etc.)
- **Topological connectivity** (indexing into a node list)
- **Fields** associated with the elements (e.g., temperature, velocity)
- **Groups** associated with the elements (through families)

```rust
pub struct ElementBlock {
    pub element_type: ElementType,
    pub connectivity: Connectivity,
    pub fields: BTreeMap<String, ArcArrayD<f64>>,
    families: Vec<usize>,
    pub groups: BTreeMap<String, BTreeSet<usize>>,
}
```

Field data is stored as `ndarray::ArcArrayD<f64>`, and identified by a naming
convention supporting time-dependent fields, e.g.:

```
"temperature_iter_3_time_0.01"
```

Group data is stored as families groups. Families always form a partition of
the elements based on the groups (ie two element not pertaining to the same
groups have a different family id).

---

## 🔁 Iteration & Access

The `umesh` module provides building blocks for:
- Iterating over elements by block
- Filtering blocks by dimension or type
- Querying and extracting field data
- Querying, extracting, updating group related data
- Operating on mesh views (sub meshes)

---

## 🎯 Design Goals

- Clean separation between **topology**, **geometry**, **field data** and **group data**
- Support for real-world meshes with mixed element types and poly element types
- Efficient per-block and global access patterns
- Compatibility with operations like:
  - Interpolation
  - Remapping
  - Merging / conformizing
  - File I/O (MED, CGNS, etc.)

---

## 🔄 Mesh Ownership and Views: `UMesh`, `UMeshView`, and `UMeshBase`

Mefikit uses an internal very flexible ownership data model based on ndarray ownership
model. Each array can be OwnedArcRepred, OwnedRepr or ViewRepr (for connectivity array,
coordinates array, all fields and all groups). The following two variants are
exposed : UMeshView, which only contains views, and UMesh, which only contains
ArcArrays. UMesh is the rust owned struct that is managed by all "transformer"
algorithm. It allows for cheap copy of the coordinates tables, or of the
fields. UMeshView is the view only variant. This makes integration with other
systems (e.g., C or Python) efficient and safe, while also enabling
high-performance zero-copy operations.
This memory model brings a bit of complexity but allows for great flexibility.

### 📦 `UMesh` – Owned Mesh

`UMesh` owns and shares its data:
- Coordinate array (`ArcArray2<f64>`)
- Element blocks with connectivities (`ArcArray<usize>`), fields
  (`ArcArrayD<f64>`), families, groups

This type is used for:
- Internal mesh manipulation in Rust
- Persistent mesh storage
- File I/O
- Long-term computation or modification

It is constructed by cloning or transferring ownership of arrays.  When used in
PyO3, input numpy buffers must be cloned for then to be owned by Rust. This
favors a usage where mefikit is the core producer of the mesh, or make some
heavy tranformations (computionally intensive).

---

### 👁️ `UMeshView<'a>` – Read-Only Mesh View

`UMeshView` is a **zero-copy, non-owning** view over existing mesh data. It
holds references (slices or `ndarray::ArrayView`) to continuous memory block
that is managed elsewhere (e.g., passed from a C/C++/Python array).

This is ideal for:
- Foreign function interface (FFI)
- Avoiding unnecessary copies when no modification is needed

⚠️ Lifetimes must be respected — `UMeshView` is only valid as long as the
referenced data lives.

---

### 🔄 Summary

| Type           | Ownership | Mutable | Use Case                                 | Copies |
|----------------|-----------|---------|------------------------------------------|--------|
| `UMesh`        | Yes       | Yes     | Full ownership, long-term usage          | No -> Shared |
| `UMeshView`    | No        | No      | Read-only access to foreign/borrowed data| No |

This model ensures performance, safety, and clear interoperability boundaries.

---

## 🛠️ In-Place vs. Out-of-Place Operations in `UMesh`

This document categorizes common mesh operations into two groups:

- **In-Place Operations**: These can be applied directly to an existing `UMesh`
  or its view without reallocating major structures.
- **Out-of-Place Operations**: These typically rewrite the mesh's core
  structures (e.g., connectivity, coordinates) and are better implemented as
  producing a new mesh.


### ✅ In-Place Operations
These operations can safely modify the owned mesh structure in-place.

| Operation                 | Description |
|---------------------------|-------------|
| `assign_field`            | Adds or replaces field data on a block or selection |
| `set_family`              | Sets the family (zone/subdomain tag) of elements |
| `set_group`               | Modifies or defines a group of element IDs under a name |
| `renumber_nodes`          | In-place reordering of coordinates and node indices |
| `set_coordinates`         | Mutate existing geometry without changing topology |
| `transform_coordinates`   | Apply affine transformation to node coordinates |


### 🧱 Out-of-Place Operations (produce new mesh, valid on UMeshBase)
These operations fundamentally change the mesh's topology, usually requiring
reallocation of connectivity tables or geometry arrays.

| Operation                  | Description |
|----------------------------|-------------|
| `merge_close_nodes()`        | Mutates connectivity to merge nearby points |
| `zip_coords()`               | Mutates coods to remove useless coords |
| `renumber_cells()`         | In-place reordering of cells. Out-of-place because of Poly |
| `build_descend()`          | Returns a new mesh composed of subentities depending on codim |
| `conformize(mesh)`         | Cleans internal inconsistencies, requires deep topology rewrite |
| `split_by(mesh_a, mesh_b)` | Cuts mesh A using B's topology, creates new elements |
| `fuse_meshes(a, b)`        | Boolean union with topological merging, produces a new mesh |
| `intersect_meshes(a, b)`   | Keeps overlapping parts of two meshes, new geometry required |
| `substract_with(a, b)`     | Subtracts domain of B from A, with new elements generated |

---

## 🔧 Core API Overview

### `aggregate_meshes(meshes: &[UMeshView]) -> UMesh`
Concatenates meshes without modifying their topology or geometry.
- May result in **overlaps** or **duplicates**
- Fast, non-conforming operation

### `fuse_meshes(a: UMeshView, b: UMeshView) -> UMesh`
Computes the **boolean union** of two meshes, producing a **conforming**
result.
- Intersects overlapping elements faces
- Inserts new faces/nodes

### `intersect_meshes(a: UMeshView, b: UMeshView) -> UMesh`
Computes the **boolean intersection** of the two spatial domains.
- Returns only the overlapping region
- UMeshes are intersected topologically and geometrically

### `substract_with(a: UMeshView, b: UMeshView) -> UMesh`
Subtracts mesh B from A (`A \ B`), computing topological intersections where
needed.
- Useful for holes, notches, or subtractive modeling

### `split_by(a: UMeshView, b: UMeshView) -> UMesh`
Splits mesh A into sub-elements along the boundaries defined by mesh B.
- UMesh B acts as a "cutter"
- Preserves A’s domain while increasing resolution/conformity

### `conformize(mesh: UMeshView) -> UMesh`
Cleans and re-meshes a single mesh to make it internally **conforming**.
- Merges internal duplicates
- Optionally splits internal faces to improve element consistency


## 📚 Related Modules

- `geometry`, `topology` — operation-specific logic
- `io` — file import/export (serde_json, serde_yaml, MED, CGNS, etc.)

---

This module serves as the **central container** for all mesh-related data and logic in the library.
