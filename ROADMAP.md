# 🧭 Roadmap & MEDCoupling Comparison

## ✅ Stage 1 — Core Mesh Representation

| Functionality                 | UMesh                        | MEDCoupling             | Notes                                                                             |
| ----------------------------- | ---------------------------- | ----------------------- | --------------------------------------------------------------------------------- |
| Basic mesh types (1D, 2D, 3D) | ✔️ via `build_cmesh`         | ✔️                      | Structured grids (SEG2/QUAD4/HEX8) via `RegularUMeshBuilder`.                      |
| Multi element type mesh       | ✔️ (`ElementBlock`)          | ✔️ (`MEDCouplingUMesh`) | UMesh uses `BTreeMap<ElementType, ElementBlock>` to cleanly group by type.        |
| Connectivity storage          | ✔️ (`Connectivity`)          | ✔️                      | Regular and poly (data/offsets) connectivities.                                   |
| Coordinates                   | ✔️                           | ✔️                      | `ArcArray2` with copy-on-write sharing.                                           |
| Fields on elements            | ✔️                           | ✔️                      | Fields stored per `ElementBlock`.                                                 |
| Cell groups & families        | ✔️ (rust)                    | ✔️ (MEDLoader)          | Stored in `BTreeMap<String, BTreeSet<usize>>`; python binding not exposed yet.    |

---

### 🧠 Stage 2 — Selection & Filtering (similar to ParaView/Polars)

| Functionality                       | UMesh                  | MEDCoupling    | Notes                                                          |
| ----------------------------------- | ---------------------- | -------------- | -------------------------------------------------------------- |
| Selection by ID                     | ✔️ (`sel.ids` / `nids`)| ✔️             |                                                                |
| Field-based selection               | ✔️ (`Field > x`)       | ✔️             | Selection DSL (`Selection`/`Comparable`) is ergonomic in both. |
| Group-based selection               | ✔️                     | ✔️ (MEDLoader) |                                                                |
| Selection by position (bbox, plane) | ✔️                     | ✔️             | Centroid-based (`bbox`, `rect`, `sphere`, `circle`) and        |
|                                     |                        |                | node-based (`nbbox`, `nrect`, `nsphere`, `ncircle`).           |
| Selection by connectivity patterns  | 🚧                     | ❌             |                                                                |
| Combine selection criteria          | ✔️                     | ❌             | Boolean combinators `&`, `\|`, `^`, `-`, `~`.                  |

---

## 🧩 Stage 3 — Topological Tools

| Functionality                     | UMesh | MEDCoupling | Notes                                                           |
| --------------------------------- | ----- | ----------- | --------------------------------------------------------------- |
| Cell–node adjacency               | ✔️    | ✔️          | Needed for topological algorithms and neighbor queries.         |
| Cell–cell neighbors               | ✔️    | ❌          |                                                                 |
| face–cell neighbors               | ✔️    | ✔️          |                                                                 |
| Node–element inverse connectivity | ⏳    | ✔️          |                                                                 |
| Boundary extraction               | ✔️    | ✔️          | `boundaries` / `boundaries_update`.                             |
| Descending connectivity           | ✔️    | ✔️          | `descend` / `descend_update` (subentities).                     |
| Cell splitting                    | ✔️    | ❌          | `split` (SEG2, TRI3, QUAD4, TET4, HEX8).                        |

➡ **Roadmap additions**:

- [ ] Topological region-growing / connectivity queries
- [x] Connex components computation (`connected_components`)

---

## 🧬 Stage 4 — Python Bindings & FFI

| Feature                             | UMesh | MEDCoupling | Notes                                                      |
| ----------------------------------- | ----- | ----------- | ---------------------------------------------------------- |
| Python Bindings via PyO3/maturin    | ✔️    | ✔️          | Rust-native API with PyO3/maturin                          |
| Selection API in Python             | ✔️    | ✔️          | `mf.sel.*` and `mf.Field(...) >= 0.0`                      |
| Conversion to NumPy Arrays          | ✔️    | ✔️          | For coords, connectivity, fields                           |
| Pythonic Mesh Access (coords, conn) | ✔️    | ✔️          | Rust-style getter wrappers                                 |
| Field transfer in Python            | ✔️    | ✔️          | `mf.transfer.*` + `apply_update`                           |
| C/C++ FFI Interface via `cbindgen`  | ⏳    | ✔️          | Exported symbols with C ABI                                |
| Rust in C/C++ via `extern "C"`      | ⏳    | ✔️          | Allows calling UMesh from legacy code                      |
| Python Submesh Creation             | ✔️    | ✔️          | `mesh.descend()`, `mesh.select(...)`, `mesh.split()`       |
| PyPI Distribution                   | ✔️    | ✔️          | Simple install with `pip install mefikit`                  |

---

## 📐 Stage 5 — Mesh Tools & Geometry Processing

| Feature                                | UMesh      | MEDCoupling | Notes                                                    |
| -------------------------------------- | ---------- | ----------- | -------------------------------------------------------- |
| Cell Measure Computation               | ✔️         | ✔️          | Per-element measure (SEG2, TRI3, QUAD4, TET4, HEX8, ...) |
| Cell Centroid Computation              | ✔️         | ✔️          | Vertex centroid; `tools::centroids` / `ElementGeo`.      |
| Bounding Box Computation               | ✔️         | ✔️          | Useful for acceleration structures                       |
| Mesh Bounding Box                      | ✔️         | ✔️          | Global extent for visualization, filtering, etc.         |
| 2D Mesh-Mesh Overlay                   | ✔️         | ✔️          | `overlay` + `OverlayOperation` (IMPRINT, UNION,          |
|                                        |            |             | INTERSECTION, DIFFERENCE, SYMMETRIC_DIFFERENCE)          |
| 3D Cell Slicing with Plane             | ⏳         | ✔️          | Module stub only                                         |
| Cell-to-Cell Intersection Measure      | 🚧         | ✔️          | Partial: overlay & ConservativeP0 transfer rely on it    |
| Distance to Point / Nearest Cell       | 🚧         | ✔️          | Partial: BVH spatial index / kNN used internally         |
| Cell Normals (2D/3D)                   | ⏳         | ✔️          | Important for post-processing and boundary conditions    |
| Intersections with Line, Plane, Volume | 🚧         | ✔️          | Partial: segment/polygon/polyhedron point-in tests       |
| Parallel Geometry Computation          | ⏳         | ❌          | `rayon` feature + `par_elements()` iterator available    |

---

## 🧪 Stage 6 — Field Tools and Math

| Functionality             | UMesh | MEDCoupling | Notes                                         |
| ------------------------- | ----- | ----------- | --------------------------------------------- |
| Scalar & vector fields    | ✔️    | ✔️          |                                               |
| Field interpolation       | ✔️    | ✔️          | `transfer`: ConstantPiecewise,                |
|                           |       |             | MovingLeastSquares, InverseDistance,          |
|                           |       |             | ConservativeP0 (2D).                          |
| Field reduction / stats   | ⏳    | ❌          |                                               |
| Norms, extrema, threshold | 🚧    | ❌          | Partial: `fieldexpr` math functions +         |
|                           |       |             | comparisons for field-based selections        |

---

## 🔁 Stage 6 — I/O

| Functionality        | UMesh      | MEDCoupling | Notes            |
| -------------------- | ---------- | ----------- | ---------------- |
| Serialization        | ✔️ (serde) | ❌          | json / yaml      |
| I/O from VTK         | ✔️         | ❌/✔️       | read & write     |
| I/O from VTKHDF      | ✔️         | ❌/✔️       | read & write     |
| I/O from MED         | ✔️         | ✔️          |                  |
| I/O from MEDCoupling | ✔️ (python)|             | `UMesh.to_mc()`  |
| I/O from CGNS        | ✔️         | ❌          |                  |
| I/O from meshio      | ✔️ (python)|             | `UMesh.to_meshio()` |

---

## 🚀 Stage 7 — Performance, Parallelism, and WASM

| Functionality      | UMesh              | MEDCoupling | Notes                                            |
| ------------------ | ------------------ | ----------- | ------------------------------------------------ |
| Thread-safe ops    | ✔️ (`Arc`, `Send`) | ❌          | MEDCoupling not thread-safe in Python            |
| WASM support       | ⏳                 | ❌          | MEDCoupling depends on HDF5, not WASM-compatible |
| Parallel iteration | ✔️                 | ❌          | Using the par_elements() rayon iterator API      |

---

## 🛠️ API Coherence & Ergonomics Plan

- [ ] Align `update_field` (rust) / `set_field` (python) — same operation, two
  names; pick one and expose it under the same name on both sides.
- [ ] Resolve the "splitting" overload: `UMesh.split()` (cell refinement,
  element types preserved) vs `connected_components` (mesh splitting). Consider
  renaming/aliasing to `refine()` / `split_cells()`.
- [ ] Reconsider the cryptic `build_cmesh` name; add a self-documenting alias
  such as `build_grid` / `structured_mesh`.
- [ ] Decide a single convention for exposing rust `*Transfer` classes in
  python (suffix dropped today: `ConstantPiecewiseTransfer` →
  `transfer.ConstantPiecewise`).
