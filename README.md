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
  - **Element families and groups** for flexible subdomain handling (WIP)
- Python bindings for all high-level tools (`build_cmesh`, `sel`, `Field`,
  `transfer`, ...)

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
- **Unified queries**: combine fields and geometry (`mf.sel.bbox`, `mf.sel.sphere`,
  `mf.sel.ids`, ...) with the boolean operators `&`, `|`, `^`, `-` and `~`
- **Efficient execution**: evaluated in Rust, with optional NumPy output

This avoids manual indexing over unstructured meshes and keeps computations
close to the data, while remaining concise and expressive.

### 🔄 Input/Output Support

- Built-in (python and rust) support for major file formats, driven by the
  file extension in `mf.UMesh.read` / `mf.UMesh.write`:
  - `json` and `yaml` with `serde`
  - `vtk` / `vtu`
  - `vtkhdf` / `h5` / `hdf5` (HDF5-based VTK)
  - `CGNS` (planned)
- Python in memory conversions (`UMesh` methods, available when the optional `io`
  dependencies are installed):
  - `to_pyvista()` — `PyVista`
  - `to_mc()` — `medcoupling`
  - `to_meshio()` — `meshio`

### 🧮 High-Level mesh operations (Python and rust)

- 🏗️ Mesh Builders
  - `build_cmesh(*axes)` - Builds a structured grid mesh (1d, 2d or 3d) of
    `SEG2`, `QUAD4` or `HEX8` cells.
  - `extrude`, `extrude_parallel`, `extrude_curv` - Raise the dimension of a
    mesh along a vector, or along a vector per node (parallel/curved extrusion).
- 🧠 Topological operations
  - `descend` / `descend_update` – Build the descending connectivity mesh (faces from volumes, etc)
  - `boundaries` / `boundaries_update` – Build the boundaries mesh
  - `crack` – Introduce topological cracks along internal faces.
  - `connected_components` – Split the mesh in connected meshes
- 📐 Geometric operations
  - `snap` - To snap nodes of one mesh on another mesh nodes
  - `merge_nodes` - Merges duplicated nodes
  - `overlay` – Boolean mesh overlay on 2D meshes: `IMPRINT`, `UNION`,
    `INTERSECTION`, `DIFFERENCE`, `SYMMETRIC_DIFFERENCE` (`OverlayOperation`)
  - `split` – Split the cells into smaller cells of the same element type
- 🔁 Field transfers
  - `mf.transfer.ConstantPiecewise` – Point-location based assignment
  - `mf.transfer.MovingLeastSquares` – MLS regression on the k nearest source cells
  - `mf.transfer.InverseDistance` – Inverse-distance weighted average
  - `mf.transfer.ConservativeP0` – Measure-weighted P0 remapping (2D)
  - `mf.transfer.DistanceWeighting` – Weighting schemes for MLS (`None`,
    `InverseDistance(exponent)`, `Gaussian`)

The transfers separate the (potentially expensive) geometric precompute
performed when the operator is constructed from the `apply_update` call that
transfers a field:

```python
op = mf.transfer.MovingLeastSquares(m_src, m_tgt, k=10)
op.apply_update(m_src, "temperature", m_tgt)
```

### 🧠 Element traits & geometry (rust only)

This element kit provides a nice way to implement new features on elements and
use them to build mesh new operations. It is split between the `element_traits`
module (generic operations on mesh elements - zero copy views) and the
`geometry` module (owned geometric primitives).

- Descending elements (`ElementTopo::subentities`, `to_simplexes` WIP)
- Equivalence classes of elements (`symmetry`, WIP)
- Simplexization (WIP)
- Bounding box trees (`spatial_index`, `SpatiallyIndexable`)
- Element intersections and cutting (`cut`, `segment`, `polygon`, `polyhedron`)
- Measures, centroids and point-in tests (`ElementGeo`)
- Convexity computation (`geometry::convexity`)

## 🧪 Developer Notes

### 📁 Project Structure

```text
mefikit/
├── crates/
│   ├── mefikit/     # The rust core library. You can use it as a rust dependency
│   └── mefikit-py/  # The PyO3 bindings used to build the python package
├── src/             # The python package
├── docs/            # The Mefikit Book
```

### Rust core library

```text
crates/mefikit/src/
├── mesh/            # Mesh & field data model, the Element API
├── element_traits/  # Element toolbox (geo/topo) used to build higher level functionnalities
├── geometry/        # Owned geometric primitives (segment, polygon, polyhedron, region)
├── tools/           # The home to all high-level functionnalities
└── io/              # Readers/writers
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

- Out-of-place functional API for heavy op (`UMeshView` or `&UMesh`):
  `descend`, `boundaries`, `overlay`, `split`, `compute_connected_components`, ...
- In-place for metadata manipulations and non destructive op (`&mut UMesh`):
  `descend_update`, `boundaries_update`, `update_field`, `merge_nodes`, `snap`, ...

Most out-of-place operations also expose an `*_update` variant that adds the
result to the mesh in-place.

### Python package

The PyO3 bindings live in the `crates/mefikit-py` crate (package name
`mefipy`). They are compiled into the `mefikit.mefipy` module which is wrapped
by the python package `mefikit` (in `src/`).

To build the bindings and the python package please run:

```bash
uv tool install maturin
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

This will check the coding style and report any issues.

### Benchmarks

The `crates/mefikit/benches/` directory contains `Mefikit` benchmarks. They use
the [Criterion](https://bheisler.github.io/criterion.rs/book/getting_started.html)
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
