# Mefikit

![Mefikit logo](https://github.com/asonolet/mefikit/blob/master/docs/src/logo/mefikit_logo_v2.png)

**Mefikit** (_Meshes and Fields Kit_) is a modern, high-performance library for
manipulating unstructured meshes and associated fields and groups. It is
designed with a minimal, clear, and efficient interface, focusing on
flexibility, correctness, and integration in multi-physics simulations and
mesh-based data processing pipelines.

🚧 **Mefikit** is in early development. Key resources to get started and follow
progress:

- 📚 [Documentation (Mefibook)](https://asonolet.github.io/mefikit)
- 🗺️ [Roadmap](./ROADMAP.md)
- 📝 [Changelog](./CHANGELOG.md)
- 🐍 [PyPI](https://pypi.org/project/mefikit)
- 📦 [Crates.io](https://crates.io/crates/mefikit)
- 💻 [GitHub](https://github.com/asonolet/mefikit)

## 💡 Why _Mefikit_?

_Mefikit_ aims to make mesh-based development **more direct and less
error-prone**, without sacrificing performance. It reduces low-level array
handling so you can focus on **algorithms, physics, and data flow**.

For scientific developers, it offers:

- 🧠 **Higher-level mesh thinking** — express operations on fields and geometry
  instead of indices
- 🧪 **Fast experimentation** — combine topology, geometry, and fields in a
  unified API
- 🔗 **Python ↔ Rust continuity** — prototype and scale with the same concepts
- 🚧 **Early-stage flexibility** — shape core design choices while the project evolves
- ⚡ **Performance-oriented core** — efficient execution without hiding the
  data model

## ✨ Key Features

### 🧩 Mesh and Field Core

- Unified, ergonomic `UMesh` structure:
  - Supports **mixed element types** in the same mesh
  - Named **fields of doubles** over elements or nodes
  - **Element groups** for flexible subdomain handling

### 🧠 Expression DSL for Fields & Mesh Queries (python and rust)

**Mefikit** provides a compact, composable DSL to work with fields and mesh
regions without manual array handling.

```python
T = mf.Field("temperature")
rhoCp = mf.Field("heat_capacity")
V = mf.Field("measure")

energy = rhoCp * T * V

mesh.eval_update("energy", energy)     # compute & store in Rust as a new field
E = mesh.eval(energy)                  # or materialize as NumPy

energy = mf.Field("energy")            # reference the computed field
submesh = mesh.select(energy > 1e6)    # field-based filtering

domain = mf.sel.bbox(p_min, p_max) | mf.sel.sphere(c, r)
submesh = mesh.select(domain & energy > 1e6)  # field and space filtering
```

- **Symbolic expressions**: build computations without touching raw arrays
- **Unified queries**: combine fields, geometry, and groups (`mf.sel.inside(...)`)
- **Efficient execution**: evaluated in Rust, with optional NumPy output

This avoids manual indexing over unstructured meshes and keeps computations
close to the data, while remaining concise and expressive.

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
- 🧠 Topological operations
  - `descend` – Build the descending connectivity mesh (faces from volumes, etc)
  - `boundaries` – Build the boundaries mesh
  - `crack` – Introduce topological cracks along internal faces.
  - `connected_components` – Split the mesh in connected meshes
- 📐 Geometric operations
  - `snap` - To snap nodes of one mesh on another mesh nodes
  - `merge_nodes` - Merges duplicated nodes
  - `fuse` – Merge two meshes into one.
  - `intersect` – Compute boolean mesh intersection.
  - `split` – Cut a mesh using another.
  - `conformize` – Intersect shared faces, snap and merge near-nodes.

### 🧠 Element kit (rust only)

This element kit provide a nice way to implement new features on elements and
use them to build mesh new operations.

- Descending elements (edges/faces of volumes, etc.)
- Equivalence classes of elements
- Simplexization
- Bounding box trees
- Element intersections
- Normal and orientation computation
- Barycentre and volume evaluation
- ...

## 🧪 Developer Notes

### 📁 Project Structure

```text
mefikit/
├── crates/      # The rust core library and pyo3 bindings. You can use it as a rust dependency
├── src/         # The python package
├── docs/        # The Mefikit Book
```

### Rust core library

```text
src/
├── mesh/          # Mesh & field data model, the Element API
├── tools/         # The home to all high-level functionnalities
├── io/            # Readers/writers
├── element_kit/   # Element toolbox used to build higher level functionnalities
```

To build the library, you need to have Rust installed. You can install Rust
using [rustup](https://rustup.rs/). Once you have Rust installed, you can
build the library using the following command:

```bash
cargo build --release
```

This will create a release build of the library in the `target/release`
directory.

### Memory model: Mesh Ownership, Views, and Shared Coordinates

- `UMesh`: fully owns its data (coordinates, connectivity, fields,
  etc.), suitable for storage, transformation, and I/O. Useful to share
  arrays using copy-on-write. Maximum performance when staying in rust.
- `UMeshView<'a>`: read-only view into external or borrowed mesh
  data; ideal for zero-copy FFI.

### API philosophy: Explicit is better than implicit

- Out-of-place functional API for heavy op (`UMeshView` or `&UMesh`): `compute_descending`,
  `intersect_meshes`, ...
- In-place for metadata manipulations and non destructive op (`&mut UMesh`):
  `assign_field`, `merge_close_nodes`, `add_group`, `snap`, ...

### Python package

The crate with the python bindings is called `mefikit-py`. It contains all the PyO3
bindings. This crate is used as the basis of the python `mefikit` package. The
same name was used for the python package for the sake of simplicity.

To build the bindings and the python package please run:

```bash
uv run maturin develop --uv
```

You can then run:

```bash
uv run pytest
```

`uv` won't build the package, it is only in charge of the dependencies.
`maturin` is the only one parametrized for this. Please run `maturin` each time
rust `mefikit` or `mefikit-py` changed.

### Mefibook

```text
docs/
├── src/                # The mdbook root dir
├── python_examples/    # Python notebooks
```

The `mefibook` is a `mdbook` project. Please refer to the `mdbook` documentation.
In two lines, you should:

```bash
cargo binstall mdbook
mdbook serve
```

`Jupyter-notebooks` are executed and converted to markdown using the following:

```bash
uv run make notebooks
```

`uv` is used here because the notebooks need `jupyterlab`, `mefikit` and all its
dependencies to run. As `uv` won't build `mefipy` you need to build it first.

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

The `mefkit/benches/` directory contains `Mefikit` benchmarks. They use the
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

You can create **flamegraphs** to spot performance bottleneck.

```bash
cargo flamegraph --profile flame --example name_of_the_example
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in Mefikit by you shall be dual licensed as above, without any
additional terms or conditions.
