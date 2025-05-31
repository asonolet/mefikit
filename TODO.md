# 🛠️ MeFiKit – Development Plan

> Focused on mesh usability, Serde I/O, connectivity comparison, face/edge extraction, and mesh gluing.

---

## 📅 Day 1 – Core Mesh Usability + Serde I/O

**🎯 Goal:** Make the `Mesh` format iterable and serializable for easy inspection.

### ✅ Tasks:
- Implement basic iterators:
  - `mesh.nodes() -> impl Iterator<Item = &Point3>`
  - `mesh.elements() -> impl Iterator<Item = &Element>`
- Derive `Serialize`/`Deserialize` for:
  - `Mesh`, `Element`, `ElementBlock`, `Connectivity`
- Add simple I/O helpers:
  - `save_to_json(path: &str)`
  - `load_from_json(path: &str)`
- Create a test or example:
  - Manually build a mesh
  - Save/load from file
  - Iterate and print element types and node counts

---

## 📅 Day 2 – Equivalent Connectivity + Descending Mesh

**🎯 Goal:** Enable comparing connectivities and building face/edge meshes.

### ✅ Tasks:
- Implement:
  - `Connectivity::is_equivalent(&self, other: &Self) -> bool`
    - Supports permutation/rotation-insensitive comparison
- Add:
  - `descending_mesh(mesh: &Mesh) -> Mesh` for faces/edges
- Use `HashSet<CanonicalConnectivity>` to deduplicate
- Write test/example:
  - Simple quad/tetra mesh
  - Print unique faces or edges
  - Validate face/edge counts

---

## 📅 Day 3 – Consoliding UMesh ownership model

### ✅ Tasks:
- Implement:
  - `UMeshView`
  - `UMeshViewMut`
  - `.view()`, `.view_mut()`, `to_owned()`
  - `SharedCoords`

---
