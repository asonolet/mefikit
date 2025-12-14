# MeFiKit

![Mefikit logo](./mefibook/src/logo/dessin_rouge_full.png)

**MeFiKit** (_Meshes and Fields Kit_) is a modern, high-performance library for
manipulating unstructured meshes and associated fields. It is designed with a
minimal, clear, and efficient interface, focusing on flexibility, correctness,
and integration in multi-physics simulations and mesh-based data processing
pipelines.

**MeFiKit** is in a very early development phase. You might want to check the [ROADMAP](./ROADMAP.md).

If you are starting with **MeFiKit**, especially on the Python side, check for
the [MeFiBook!](./mefibook/src/SUMMARY.md)

---

## ✨ Key Features

### 🧩 Mesh and Field Core

- Unified, ergonomic `UMesh` structure:
  - Supports **mixed element types** in the same mesh
  - Named **fields of doubles** over elements or nodes
  - **Element groups** for flexible subdomain handling

### 🔄 Input/Output Support

- Built-in (python and rust) support for major file formats:
  - `vtk`
  - `CGNS` (planned)
  - `json` and `yaml` with `serde`
- Python conversions
  - `PyVista`
  - `medcoupling`
  - `meshio`

### 🧮 High-Level mesh operations (Python and rust)

- 🏗️ Mesh Builders
  - `cmesh_builder` - Builds a grid mesh (1d, 2d or 3d).
  - `extrude` - Create an extruded mesh (1d x 1d, 2d x 1d)
  - `duplicate` - Create a mesh by duplication (0d, 1d, 2d, 3d)
  - `aggregate` – Build a mesh from multiple non overlapping cell groups.
- `Selector` - Element selection builder
  - on nodes position
  - on elements position
  - on fields values
  - etc
- 🧠 Topological operations
  - `submesh` – Build the descending connectivity mesh (faces from volumes, etc)
  - `boundaries` – Build the boundaries mesh
  - `crack` – Introduce topological cracks along internal faces.
  - `merge_nodes` - Merges duplicated nodes
- 📐 Geometric operations
  - `snap` - To snap nodes of one mesh on another mesh nodes
  - `fuse` – Merge two meshes into one.
  - `intersect` – Compute boolean mesh intersection.
  - `split` – Cut a mesh using another.
  - `conformize` – Intersect shared faces, snap and merge near-nodes.

### 🧠 Topological Toolbox (rust only)

- Utilities for topological operations on elements:
  - **Descending elements** (edges/faces of volumes, etc.)
  - **Equivalence classes** of elements
  - **Simplexization**

### 📐 Geometric Toolbox (rust only)

- Geometric computation tools:
  - Bounding box trees
  - Element intersections
  - Normal and orientation computation
  - Barycenter and volume evaluation

### 🔄 Mesh Ownership, Views, and Shared Coordinates

- `MeFiKit` distinguishes between mesh ownership and views for flexibility and
  performance:
  - `UMesh`: fully owns its data (coordinates, connectivity, fields,
    etc.), suitable for storage, transformation, and I/O. Useful to share
    arrays using copy-on-write. Maximum performance when staying in rust.
  - `UMeshView<'a>`: read-only view into external or borrowed mesh
    data; ideal for zero-copy FFI.

### 🛠 In-place vs Out-of-place Operations

- Clean mostly functional API:
  - Out-of-place for heavy op (`UMeshView` or `&UMesh`): `compute_submesh`,
    `intersect_meshes`, ...
  - In-place for metadata and non destructive op (`&mut UMesh`):
    `assign_field`, `merge_close_nodes`, `add_group`, `snap`, ...

### 🐍 Python Bindings

- `mefipy`:
  - All high level functionality is exposed via clean Python bindings in this
    crate for rapid prototyping and integration in data pipelines.
- `mefikit`:
  - python package exposing `mefipy`.
  - adding python io conversions through `numpy` to `meshio`, `pyvista`, `medcoupling`.

---

## 💡 Why MeFiKit?

The internal mesh representation is designed for **simplicity and
performance**, closely matching the file format layout. Unlike other tools
`MeFiKit` provides:

- 🧼 **Simpler interface**
- ⚙️ Easier integration and debugging
- 📦 Modern tools and clean build system (Rust/Cargo)
- 🧪 Pilot usage of rust in mesh tools and HPC scientific software

And keeps:

- 🚀 Good **runtime performance**
- 🧪 Robust testing & benchmarking suite

---

## 🧪 Developer Notes

### 📁 Project Structure

```text
src/
├── mesh/          # Mesh & field data model
├── tools/         # The home to all high-level functionnalities
├── io/            # Readers/writers
├── topology/      # Descending/neighbor tools
├── geometry/      # Volumes, bboxes, slicing
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

### Contributing

If you would like to contribute to the library, please fork the repository
and create a pull request with your changes. We welcome contributions of all
kinds, including bug fixes, new features, and documentation improvements.
Please make sure to follow the coding style and conventions used in the
library. You should use `pre-commit` for this purpose.

```bash
uv tool install prek
prek install
git commit -a # pre-commit runs on your committed files
```

This will check the coding style and report any issues. You can also run
the following command to automatically format the code:

### Benchmarks

The `mefkit/benches/` directory contains MeFiKit benchmarks. They use the
[Criterion](https://bheisler.github.io/criterion.rs/book/getting_started.html)
framework.

To launch the benchmarks, run:

```sh
cargo bench
```

To view results as a static and local website:

```sh
firefox ./target/criterion/report/index.html
```

A convenient CLI tool to visualize a summary of the results is `critcmp`:

```sh
cargo install critcmp
critcmp --list
```

If a new benchmark source file `filename.rs` is added inside `benches/`,
**`Cargo.toml` must be adapted accordingly**:

```toml
[[bench]]
name = "filename"
harness = false
```

Note that `filename`, in `Cargo.toml`, is written without the `.rs` extension.
More information in the [Criterion
documentation](https://bheisler.github.io/criterion.rs/book/getting_started.html#step-1---add-dependency-to-cargotoml)

You can create **flamegraphs** using to explore performance issues.

```bash
cargo flamegraph --profile flame --example name_of_the_example
```

## License

This library is licensed under the MIT License. See the `LICENSE` file for
more information.
