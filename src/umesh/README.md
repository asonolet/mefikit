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
    pub name: Option<String>,
    pub description: Option<String>,
    pub coords: ArcArray2<f64>,
    pub element_blocks: BTreeMap<ElementType, ElementBlock>
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
    pub fields: BTreeMap<String, ArrayD<f64>>,
    families: Vec<usize>,
    pub groups: BTreeMap<String, BTreeSet<usize>>,
}
```

Field data is stored as `ndarray::ArrayD<f64>`, and identified by a naming
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

## 🛠️ In-Place vs. Out-of-Place Operations in `UMesh`

This document categorizes common mesh operations into two groups:

- **In-Place Operations**: These can be applied directly to an existing `UMesh` or its view without reallocating major structures.
- **Out-of-Place Operations**: These typically rewrite the mesh's core structures (e.g., connectivity, coordinates) and are better implemented as producing a new mesh.


### ✅ In-Place Operations (mutate existing mesh, applied on UMeshViewMut)
These operations can safely modify the mesh structure in-place, especially when
done through a `UMeshViewMut`.

| Operation                 | Description |
|---------------------------|-------------|
| `assign_field`            | Adds or replaces field data on a block or selection |
| `set_family`              | Sets the family (zone/subdomain tag) of elements |
| `set_group`               | Modifies or defines a group of element IDs under a name |
| `renumber_nodes`          | In-place reordering of coordinates and node indices |
| `merge_close_nodes`       | Mutates coordinates and connectivity to merge nearby points |
| `set_coordinates`         | Mutate existing geometry without changing topology |
| `transform_coordinates`   | Apply affine transformation to node coordinates |


### 🧱 Out-of-Place Operations (produce new mesh, valid on UMeshView)
These operations fundamentally change the mesh's topology, usually requiring
reallocation of connectivity tables or geometry arrays.

| Operation                  | Description |
|----------------------------|-------------|
| `renumber_cells()`         | In-place reordering of cells. Out-of-place because of Poly |
| `build_submesh()`          | Returns a new mesh composed of subentities depending on codim |
| `conformize(mesh)`         | Cleans internal inconsistencies, requires deep topology rewrite |
| `split_by(mesh_a, mesh_b)` | Cuts mesh A using B's topology, creates new elements |
| `fuse_meshes(a, b)`        | Boolean union with topological merging, produces a new mesh |
| `intersect_meshes(a, b)`   | Keeps overlapping parts of two meshes, new geometry required |
| `substract_with(a, b)`     | Subtracts domain of B from A, with new elements generated |

---

## 📚 Related Modules

- `geometry`, `topology` — operation-specific logic
- `io` — file import/export (serde_json, serde_yaml, MED, CGNS, etc.)

---

This module serves as the **central container** for all mesh-related data and logic in the library.
