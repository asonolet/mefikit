# MeFiKit

*Meshes and Fields Kit* is a library implementing:

- a convenient unstructured mesh format (mesh with data) with:
    - different kind of elements in the same mesh
    - named fields of double over elements
    - groups of elements
- various input output formats
    - medcoupling
    - medfile
    - cgns
    - serde
- builders for meshes:
    - structured
    - extruded
    - unstructured
- topological toolbox:
    - descending meshes
    - aggregation of meshes
    - iterators over neighbours
    - equivalence classes of elements
    - connected components
    - tetrahedization
    - polyhedrization / unpolyhedrization
- geometric toolbox:
    - bounding box tree computation
    - intersection of elements
    - merge close nodes
    - compute normals, orientations
    - compute barycenters
    - compute volumes
- high level algorithms:
    - split by another mesh (computing intersections on first mesh)
    - conformize with itself (face conforming optional)
    - geometric fusion of meshes, also serves as a "conforming" mesh generator
      (boolean union, face conforming optional)
    - intersection of meshes (boolean intersection)
    - subtraction of meshes (boolean subtraction)
    - cracking of a mesh from a descending mesh
    - intersection_map building for interpolation of fields
- python bindings

The "convenient unstructured mesh format" will be close to the file storage
format for performance and simplicity reasons, **unlike** medcoupling core
format. This alone is a good argument in favour of the reimplementation of the
medcoupling library (let alone the huge part of the medcoupling lib which is
suboptimal, fragile and bogus).

In mefikit those features are supposed to be performant and implemented using
a clear necessary and minimal interface. This allows a better maintainability
of the library.

## Advantages over the current medcoupling

- performance
- ease of use (one main clear data structure and not many, many algorithms)
- clarity (medcoupling ~=? medfile, but mefikit != medfile), modern tools
- rust pilot project in the DM2S

## Mesh Tools Overview

This section describes key mesh operations available in the MEDCoupling library
for combining and manipulating meshes and their spatial domains.

---

### `aggregate_meshes`

The `aggregate_meshes` operation **concatenates multiple meshes** into a single
mesh without modifying element topology or geometry. Nodes and elements from
all input meshes are combined, but no merging or intersection computations are
performed.

- **Result:**
  A composite mesh containing all input elements, which may include overlapping
  or duplicated nodes and elements at interfaces.
- **Use case:**
  Quickly combine disconnected or independent meshes for batch processing or
  grouping.

---

### `fuse_meshes`

The `fuse_meshes` operation performs an advanced **fusion of two meshes by
computing intersections between their elements** and reconstructing a unified
conforming mesh.

- **Key behaviors:**
  - Detects and computes overlapping/intersecting element volumes.
  - Introduces new nodes along intersection boundaries.
  - Splits and merges touching faces, potentially creating polyhedral or cut elements.
  - Produces a mesh free of overlaps or gaps, fully conforming across the combined domain.

- **Result:**
  A single, topologically consistent mesh representing the union of the input
  meshes, with intersection geometry fully accounted for.

- **Use case:**
  Multi-domain simulations requiring conforming interfaces or combined meshes
  with overlapping regions.

---

### `intersect_meshes`

The `intersect_meshes` operation computes the **geometric intersection of the
spatial domains** represented by two input meshes.

- **Key behaviors:**
  - Determines the volume (or area) common to both input meshes.
  - Extracts the mesh elements that lie strictly within this intersection
    region.
  - Resulting mesh represents the shared space only.

- **Result:**
  A mesh corresponding exactly to the overlapping spatial region of the two
  inputs.

- **Use case:**
  Extracting common subdomains, performing Boolean intersection operations, or
  limiting analysis to shared regions.

---

Each of these operations provides powerful tools for mesh manipulation,
enabling complex workflows involving mesh union, intersection, and domain
partitioning in scientific computing and simulation.

## Developer Notes

The library is structured the following way:

- umesh
    - umesh_core
    - element_block
    - element
    - connectivity
- io
    - medcoupling
    - med
    - cgns
- topology
    - neighbour_iterators
    - connex_components
    - mesh_aggregator
    - descending_mesh
    - tetrahedrizer
    - polyzer
    - unpolyzer
- geometry
    - BHV (bounding box hierarchy)
    - intersection
    - merge_close_nodes
    - normals
    - barycenters
    - volumes
- tools
    - fuser
    - domain_intersecter
    - domain_clipper
    - connex_decomposer
    - cracker
    - fields_remapper
    - renumberer
    - cutter
- tests
    - integration tests
    - performance tests
    - regression tests

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

### License

This library is licensed under the MIT License. See the `LICENSE` file for
more information.

### Acknowledgements

This library is developed as part of the DM2S project at CEA. We would like
to thank the contributors and maintainers of the MEDCoupling library for their
work and support. This library is inspired by the MEDCoupling library and
aims to provide a more performant and user-friendly alternative for mesh
manipulation and analysis.
