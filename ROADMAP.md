# 🧭 Roadmap & MEDCoupling Comparison


## ✅ Stage 1 — Core Mesh Representation

| Functionality                 | UMesh                        | MEDCoupling             | Notes                                                                         |
| ----------------------------- | ---------------------------- | ----------------------- | ----------------------------------------------------------------------------- |
| Basic mesh types (1D, 2D, 3D) | ✔️ via `RegularUMeshBuilder` | ✔️                      | UMesh handles structured grids; unstructured input likely next.               |
| Multi element type mesh       | ✔️ (`ElementBlock`)          | ✔️ (`MEDCouplingUMesh`) | UMesh uses `BTreeMap<ElementType, ElementBlock>` to cleanly group by type.    |
| Connectivity storage          | ✔️ (`Connectivity`)          | ✔️                      | Direct access in both; UMesh more idiomatic for Rust.                         |
| Coordinates                   | ✔️ (`SharedCoords`)          | ✔️                      | MEDCoupling supports deep/shallow copies; `ArcArray2` achieves similar goals. |
| Fields on elements            | ✔️                           | ✔️                      | Fields stored per `ElementBlock`.                                             |
| Cell groups & families        | ✔️                           | ✔️ (MEDLoader)          | UMesh stores group labels in `BTreeMap<String, BTreeSet<usize>>`.             |

---

### 🧠 Stage 2 — Selection & Filtering (similar to ParaView)

| Functionality                       | UMesh                     | MEDCoupling    | Notes                                                        |
| ----------------------------------- | ------------------------- | -------------- | ------------------------------------------------------------ |
| Selection by ID                     | ✔️ (`filter()`)            | ✔️             |                                                              |
| Field-based selection               | 🚧 (`SelectorBuilder`)    | ✔️             | UMesh selection DSL is more ergonomic and idiomatic in Rust. |
| Group-based selection               | 🚧                        | ✔️ (MEDLoader) |                                                              |
| Selection by position (bbox, plane) | 🚧                        | ✔️             | UMesh needs pointwise or centroid-based spatial filtering.   |
| Selection by connectivity patterns  | 🚧                        | ❌            |                                                              |
| Combine selection criteria          | ✔️ (AND)                   | ❌            | Simplifed by using AND only, which is reasonable.         |

---

## 🧩 Stage 3 — Topological Tools

| Functionality                     | UMesh      | MEDCoupling | Notes                                                           |
| --------------------------------- | ---------- | ----------- | --------------------------------------------------------------- |
| Cell–node adjacency               | ✔️         | ✔️          | Will be needed for topological algorithms and neighbor queries. |
| Cell–cell neighbors               | 🚧        | ❌         |                                                                 |
| face–cell neighbors               | 🚧        | ✔️          |                                                                 |
| Node–element inverse connectivity | ⏳         | ✔️          |                                                                 |
| Boundary extraction               | ⏳         | ✔️          |                                                                 |

➡ **Roadmap additions**:

* [ ] Topological region-growing / connectivity queries
* [ ] Connex components computation

---

## 🧬 Stage 4 — Python Bindings & FFI

| Feature                             | UMesh  | MEDCoupling           | Notes |
|-------------------------------------|--------|-----------------------|-------|
| Python Bindings via PyO3/maturin    | ⏳     | ✔️                    | Rust-native API with PyO3/maturin |
| Field Selection API in Python       | ⏳     | ✔️                    | Fluent `.field("temp").gt(...)` |
| Conversion to NumPy Arrays          | ⏳     | ✔️                    | For coords, connectivity, fields |
| Pythonic Mesh Access (coords, conn) | ⏳     | ✔️                    | Rust-style getter wrappers |
| C/C++ FFI Interface via `cbindgen`  | ⏳     | ✔️                    | Exported symbols with C ABI |
| Rust in C/C++ via `extern "C"`      | ⏳     | ✔️                    | Allows calling UMesh from legacy code |
| Python Submesh Creation             | ⏳     | ✔️                    | `mesh.filter().to_submesh()` |
| PyPI Distribution                   | ⏳     | ✔️                    | Simple install with `pip install` |

---

## 📐 Stage 5 — Mesh Tools & Geometry Processing

| Feature                                  | UMesh             | MEDCoupling | Notes |
|------------------------------------------|--------------------|------------|-------|
| Cell Volume Computation                  | ⏳ Planned         | ✔️         | Per-element volume (SEG2, TRI3, QUAD4, TET4, HEX8, etc.) |
| Cell Centroid Computation                | ✔️ Basic            | ✔️         | Already used for selection API |
| Bounding Box Computation                 | ✔️ (Element-level)  | ✔️         | Useful for acceleration structures |
| Mesh Bounding Box                        | ⏳                 | ✔️         | Global extent for visualization, filtering, etc. |
| 2D Mesh-Mesh Intersection                | ⏳                 | ✔️         | Boolean ops (cut, intersect) between 2D meshes |
| 3D Cell Slicing with Plane               | ⏳                 | ✔️         | Used for visualization and post-processing |
| Cell-to-Cell Intersection Volume         | ⏳                 | ✔️         | Useful for remapping and comparison |
| Distance to Point / Nearest Cell         | ⏳                 | ✔️         | Point location and snapping logic |
| Cell Normals (2D/3D)                     | ⏳                 | ✔️         | Important for post-processing and boundary conditions |
| Intersections with Line, Plane, Volume   | ⏳                 | ✔️         | Critical for cutting/sampling geometry |
| Parallel Geometry Computation            | ⏳                 | ❌        | UMesh aims to eventually support Rayon/parallelism here |

---

## 🧪 Stage 6 — Field Tools and Math

| Functionality             | UMesh | MEDCoupling | Notes                                         |
| ------------------------- | ----- | ----------- | --------------------------------------------- |
| Scalar & vector fields    | ✔️     | ✔️           |                                               |
| Field interpolation       | ⏳    | ✔️           | Important for remapping and mesh-to-mesh ops. |
| Field reduction / stats   | ⏳    | ❌          |                                               |
| Norms, extrema, threshold | ⏳    | ❌          |                                               |

---

## 🔁 Stage 6 — I/O

| Functionality   | UMesh      | MEDCoupling | Notes |
| --------------- | ---------- | ----------- | ----- |
| Serialization   | ✔️ (serde)  | ❌          |       |
| I/O from VTK    | ⏳         | ❌/✔️        | write only |
| I/O from MED    | ⏳         | ✔️           |       |
| I/O from CGNS   | ⏳         | ❌          |       |
| I/O from meshio | ⏳         | ❌          |       |

---

## 🚀 Stage 7 — Performance, Parallelism, and WASM

| Functionality      | UMesh              | MEDCoupling | Notes                                            |
| ------------------ | ------------------ | ----------- | ------------------------------------------------ |
| Thread-safe ops    | ✔️ (`Arc`, `Send`)  | ❌           | MEDCoupling not thread-safe in Python            |
| WASM support       | ⏳                 | ❌           | MEDCoupling depends on HDF5, not WASM-compatible |
| Parallel iteration | ⏳                 | ❌           |                                                  |
