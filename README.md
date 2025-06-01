# MeFiKit

**MeFiKit** (*Meshes and Fields Kit*) is a modern, high-performance library for
manipulating unstructured meshes and associated fields. It is designed with a
minimal, clear, and efficient interface, focusing on flexibility, correctness,
and integration in multi-physics simulations and mesh-based data processing
pipelines.

---

## ✨ Key Features

### 🧩 Mesh and Field Core
- Unified, ergonomic mesh format:
  - Supports **mixed element types** in the same mesh
  - Named **fields of doubles** over elements or nodes
  - **Element groups** for flexible subdomain handling

### 🔄 Input/Output Support
- Built-in support for major file formats:
  - `medcoupling`
  - `medfile`
  - `CGNS`
  - Custom formats with `serde`

### 🏗️ Mesh Builders
- Construct meshes from scratch:
  - Structured meshes (grid-like)
  - Extruded meshes (2D to 3D)
  - Fully unstructured meshes (manually)
- Powerfull selection builder:
  - Based on element selection,
  - geometrical criterion,
  - field threshold,
  - etc.

### 🧠 Topological Toolbox
- Utilities for advanced topological operations:
  - **Descending meshes** (edges/faces of volumes, etc.)
  - **Mesh aggregation** (grouping meshes)
  - **Neighbor iterators**
  - **Equivalence classes** of elements
  - **Connected components**
  - **Tetrahedrization**, **polyhedrization**, and reverse operations

### 📐 Geometric Toolbox
- Geometric computation tools:
  - Bounding box trees
  - Element intersections
  - Close node merging
  - Normal and orientation computation
  - Barycenter and volume evaluation

### 🧮 High-Level Algorithms
- High-level, composable mesh operations (API not stable):
  - `aggregate_meshes` – Build a coarse mesh from multiple cell groups.
  - `build_intersection_map` – for **field interpolation** and remapping.
  - `conformize` – Intersect shared faces, snap and merge near-nodes.
  - `crack_from_descending` – Introduce topological cracks along internal faces.
  - `fuse_meshes` – Merge two meshes into one.
  - `intersect_meshes` – Compute boolean mesh intersection.
  - `split_by` – Cut a mesh using another.
  - `substract_with` – Subtract one mesh from another.

### 🔄 Mesh Ownership, Views, and Shared Coordinates
- MeFiKit distinguishes between mesh ownership and views for flexibility and
  performance:
  - `UMesh`: fully owns its data (coordinates, connectivity, fields,
    etc.), suitable for storage, transformation, and I/O.
  - `UMeshView<'a>`: read-only view into external or borrowed mesh
    data; ideal for zero-copy FFI.
- Mefikit supports shared coordinates across meshes for performance:
  - `SharedCoords` wraps coordinates for safe mutability.
  - Shared coordinate arrays can be modified in-place unless exclusive access is
    required (`ensure_unique()` triggers a copy).

### 🛠 In-place vs Out-of-place Operations
- Clean mostly functionnal API:
  - In-place for metadata and non destructive op (`UMeshViewMut`):
    `assign_field`, `merge_close_nodes`, `add_group`, ...
  - Out-of-place for heavy op (`UMeshView`): `build_submesh`, `fuse_meshes`,
    `intersect_meshes`, ...

### 🐍 Python Bindings
- mefikit-python:
  - All functionality is exposed via clean Python bindings in this crate for
    rapid prototyping and integration in data pipelines.

### FFI
- mefikit-ffi:
  - All functionality is exposed via ffi bindings for C/C++ interoperability

---

## 💡 Why MeFiKit?

The internal mesh representation is designed for **simplicity and
performance**, closely matching the file format layout. Unlike MEDCoupling’s
complex (when mixed with MEDFile) and sometimes inefficient structure, MeFiKit
provides:

- 🚀 Better **runtime performance**
- 🧼 Clearer and **simpler interfaces**
- ⚙️ Easier integration and debugging
- 📦 Modern tools and clean build system (Rust/Cargo)
- 🧪 Robust testing & benchmarking suite
- 🧪 Pilot usage of rust in CEA’s **DM2S** simulations

---

## 🧪 Developer Notes

### 📁 Project Structure

```text
umesh/
  ├── umesh_core.rs
  ├── element_block.rs
  ├── element.rs
  ├── connectivity.rs

io/
  ├── medcoupling.rs
  ├── med.rs
  ├── cgns.rs

topology/
  ├── neighbour_iterators.rs
  ├── connex_components.rs
  ├── mesh_aggregator.rs
  ├── descending_mesh.rs
  ├── tetrahedrizer.rs
  ├── polyzer.rs
  ├── unpolyzer.rs

geometry/
  ├── bvh.rs
  ├── intersection.rs
  ├── merge_close_nodes.rs
  ├── normals.rs
  ├── barycenters.rs
  ├── volumes.rs

tools/
  ├── fuser.rs
  ├── domain_intersecter.rs
  ├── connex_decomposer.rs
  ├── cracker.rs
  ├── fields_remapper.rs
  ├── cutter.rs
  ├── renumberer.rs

tests/
  ├── integration/
  ├── performance/
```

### Build Instructions

To build the library, you need to have Rust installed. You can install Rust
using [rustup](https://rustup.rs/). Once you have Rust installed, you can
build the library using the following command:

```bash
cargo build --release
```
This will create a release build of the library in the `target/release`
directory.

### Running Tests

To run the tests, you can use the following command:

```bash
cargo test --release
```
This will run all the tests in the library. You can also run specific tests
by specifying the test name:

```bash
cargo test --release <test_name>
```
This will run only the specified test.

### Running Benchmarks

To run the benchmarks, you can use the following command:

```bash
cargo bench --release
```
This will run all the benchmarks in the library. You can also run specific
benchmarks by specifying the benchmark name:

```bash
cargo bench --release <benchmark_name>
```
This will run only the specified benchmark.

### Generating Documentation

To generate the documentation for the library, you can use the following
command:

```bash
cargo doc --release
```

This will generate the documentation in the `target/doc` directory. You can
open the documentation in your web browser by opening the `index.html` file
in the `target/doc` directory.
You can also view the documentation online at [docs.rs](https://docs.rs/mefikit).
You can publish the documentation to docs.rs using the following command:

```bash
cargo publish --release
```

This will publish the documentation to docs.rs, where it will be available for
public access. You can view the published documentation at
[docs.rs/mefikit](https://docs.rs/mefikit).

### Contributing

If you would like to contribute to the library, please fork the repository
and create a pull request with your changes. We welcome contributions of all
kinds, including bug fixes, new features, and documentation improvements.
Please make sure to follow the coding style and conventions used in the
library. You can run the following command to check the coding style:

```bash
cargo fmt -- --check
```

This will check the coding style and report any issues. You can also run
the following command to automatically format the code:

```bash
cargo fmt
```

This will format the code according to the coding style and conventions used
in the library. Please make sure to run this command before submitting your
pull request.

## License

This library is licensed under the MIT License. See the `LICENSE` file for
more information.

## Acknowledgements

This library is developed as part of the DM2S project at CEA. We would like
to thank the contributors and maintainers of the MEDCoupling library for their
work and support. This library is inspired by the MEDCoupling library and
aims to provide a more performant and user-friendly alternative for mesh
manipulation and analysis.
