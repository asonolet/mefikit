# mefikit C++ bindings

C++ access to [mefikit](../../README.md), generated with
[cxx](https://cxx.rs). The whole C++ API is declared in one `#[cxx::bridge]`
module, [`src/ffi.rs`](src/ffi.rs), and reached through a single header:

```cpp
#include <mefikit/mefikit.hpp>

auto mesh = mefikit::UMesh::read("coarse.med");
auto fine = mefikit::UMesh::read("fine.med");

auto op = mefikit::TransferOperator::prepare(*mesh, *fine, mefikit::conservative_p0());
for (int step = 0; step < n_steps; ++step) {
  op->apply_update(*mesh, "temperature", *fine, "temperature", 0.0,
                   mefikit::FieldNature::Intensive);
}
```

## Building

Two steps, because the implementation is Rust and the interface is C++.

```console
$ cargo build -p mefikit-ffi            # or --release
$ cmake -S crates/mefikit-ffi/examples -B build/cpp
$ cmake --build build/cpp
$ ctest --test-dir build/cpp --output-on-failure
```

The first command publishes two things under `target/<profile>/`:

| Path | What it is |
| --- | --- |
| `libmefikit_ffi.so` | the Rust implementation |
| `cxxbridge/include/` | `<mefikit/mefikit.hpp>` plus cxx's generated header |
| `cxxbridge/src/` | the C++ glue (`ffi.rs.cc`, `cxx.cc`) that has to be compiled into your program |

`cargo build -p mefikit-ffi` has to have run before CMake configures: CMake only
picks up what Cargo produced, it does not drive Cargo. Pass
`-DMEFIKIT_REPO_ROOT=/path/to/mefikit` if your CMake project lives outside this
repository.

## Using it from your own CMake project

Copy the target block out of
[`examples/CMakeLists.txt`](examples/CMakeLists.txt):

```cmake
set(MEFIKIT_REPO_ROOT "/path/to/mefikit" CACHE PATH "Repository root")

# The only cargo<->CMake translation needed: which profile directory to read from.
if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "")
  set(MEFIKIT_CARGO_PROFILE_DIR debug)
else()
  set(MEFIKIT_CARGO_PROFILE_DIR release)
endif()
set(MEFIKIT_ARTIFACT_DIR "${MEFIKIT_REPO_ROOT}/target/${MEFIKIT_CARGO_PROFILE_DIR}")

find_library(MEFIKIT_LIBRARY NAMES mefikit_ffi
             PATHS "${MEFIKIT_ARTIFACT_DIR}" NO_DEFAULT_PATH)

add_library(mefikit_cxx_glue STATIC
    "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/src/ffi.rs.cc"
    "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/src/cxx.cc")
set_target_properties(mefikit_cxx_glue PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON POSITION_INDEPENDENT_CODE ON)
target_include_directories(mefikit_cxx_glue PUBLIC "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/include")

add_library(mefikit_ffi SHARED IMPORTED)
set_target_properties(mefikit_ffi PROPERTIES
    IMPORTED_LOCATION "${MEFIKIT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/include")

# The glue archive must be scanned before the shared library.
target_link_libraries(mefikit_cxx_glue PUBLIC mefikit_ffi)

add_library(mefikit::mefikit INTERFACE IMPORTED GLOBAL)
target_link_libraries(mefikit::mefikit INTERFACE mefikit_cxx_glue)
```

then

```cmake
target_link_libraries(my_app PRIVATE mefikit::mefikit)
```

The C++ glue exists because cxx compiles both `ffi.rs.cc` (the generated
declarations) and `cxx.cc` (the out-of-line half of its runtime, giving you
`rust::String`, `rust::Error` and friends) *into* the Rust library, where C++
mangled symbols are not exported. Your program needs its own copy; the build
script stages the exact files that were linked in so the two sides cannot drift.

### Without CMake

Four lines are enough if you do not use CMake:

```console
$ c++ -std=c++17 -I target/debug/cxxbridge/include \
      my_app.cpp \
      target/debug/cxxbridge/src/ffi.rs.cc \
      target/debug/cxxbridge/src/cxx.cc \
      -L target/debug -lmefikit_ffi -Wl,-rpath,$PWD/target/debug \
      -o my_app
```

## The API at a glance

Everything lives in namespace `mefikit`.

### `UMesh`

| | |
| --- | --- |
| `static UMesh from_coords(coords, n_nodes, space_dim)` | empty mesh from a row-major `(n_nodes, space_dim)` array |
| `static UMesh read(path)` | `.med`, `.vtu`, `.vtk`, `.vtkhdf`, `.cgns`, `.json`, `.yaml` |
| `void write(path) const` | same formats, chosen by extension |
| `void add_regular_block(element_type, conn, n_elements)` | fixed node count, row-major `(n_elements, num_nodes)` |
| `void add_poly_block(element_type, conn, offsets)` | `PGON` / `PHED` / `SPLINE`; flat node list plus cumulative end offsets |
| `void validate_structure() const` | the only check for non-finite coordinates and out-of-range connectivity |
| `n_nodes()`, `n_elements()`, `n_elements_of(et)`, `space_dimension()` | counts |
| `is_empty()`, `topological_dimension()`, `element_types()` | shape |
| `set_field(name, blocks, values)` | one block per element type |
| `set_field_uniform(name, et, n_components, values)` | single-block shortcut |
| `field_info(name, et)`, `field_names()` | shapes and names |
| `field_values(name, et)` | zero-copy `rust::Slice<const double>` |

`element_types()` and `field_names()` return `rust::Vec`, which has the subset of
the `std::vector` interface you would expect (`size`, `operator[]`, `begin`,
`end`, `at`). `field_names()` gives `rust::Vec<rust::String>`; `rust::String`
converts to `std::string` with an explicit `static_cast`.

`UMesh::from_coords` rather than `UMesh::new`, because `new` is a C++ keyword and
cannot name a member function.

### Fields

A mefikit field is stored per topological dimension, not per element type: a
`TRI3` and a `QUAD4` on the same surface share one field, and `set_field` requires
the blocks you pass to be exactly the mesh's element types at one dimension.

```cpp
std::vector<mefikit::FieldBlock> blocks{
    {mefikit::ElementType::QUAD4, /*n_components*/ 1, /*offset*/ 0, /*len*/ 4},
    {mefikit::ElementType::TRI3,  /*n_components*/ 1, /*offset*/ 4, /*len*/ 2},
};
std::vector<double> values{/* 6 values */};
mesh->set_field("T", rust::Slice<const mefikit::FieldBlock>(blocks.data(), blocks.size()),
                rust::Slice<const double>(values.data(), values.size()));
```

Anything that does not match — a missing block, the wrong number of values, a
zero-component block — is reported rather than silently accepted. For the common
single-block case `set_field_uniform` does the same with one argument fewer.

### Transfers

`TransferOperator::prepare` walks the geometry and is by far the expensive step.
It holds no reference to either mesh, so build it once and call `apply_update`
once per field and per time step. `mefikit::transfer_field` is the one-shot
shortcut for a single field.

```cpp
auto op = mefikit::TransferOperator::prepare(
    *src, *tgt,
    mefikit::moving_least_squares(6, mefikit::DistanceWeighting::Gaussian));
op->apply_update(*src, "T", *tgt, "T", /*default_value*/ 0.0,
                 mefikit::FieldNature::Intensive);
```

| Factory | Method |
| --- | --- |
| `constant_piecewise(point_location)` | copy the source cell containing the sample point |
| `conservative_p0()` | overlap-measure weighted average, conservative for intensive fields |
| `inverse_distance(k, exponent = 2.0)` | Shepard interpolation over the `k` nearest cells |
| `moving_least_squares(k, weighting, exponent = 2.0)` | local least-squares fit |

`FieldNature::Intensive` for per-unit-measure quantities (temperature,
pressure), `FieldNature::Extensive` for totals (mass, energy). Target cells the
source does not cover get `default_value`.

Only `PointLocation::Centroid` is implemented; `Barycenter` and `StrictInterior`
are rejected with an error, since mefikit does not implement them yet and a panic
crossing the bridge would abort the process.

### What the bindings check for you

mefikit states the preconditions of a transfer as `assert!`s, and a Rust panic
that reaches C++ aborts the process, so `prepare` returns an error instead for
each of them: mismatched space dimensions, an empty source or target, `k` below
one, a non-positive or non-finite exponent, and — for `conservative_p0` and
`constant_piecewise` — cells that do not fill their space dimension. The same
applies to the input side: a second block of an element type the mesh already
has, an empty field name, a field whose blocks disagree on the component count,
and a field size that overflows.

The two checks that walk every element — that coordinates are finite, and that
connectivity stays inside the mesh — are **not** done here. They belong to
mefikit itself, which already has them in `validate_structure`, but does not yet
run them when a block is added, and the bindings do not reimplement it. So
`from_coords` accepts a non-finite coordinate and `add_regular_block` /
`add_poly_block` accept out-of-range indices and a malformed poly offset table.

Call `validate_structure()` on a mesh you built by hand. It is the one call that
reports all three, and it is worth its cost exactly once, after the mesh is
complete:

```cpp
mefikit::UMesh mesh = mefikit::UMesh::from_coords(coords, n_nodes, 2);
mesh.add_regular_block(mefikit::ElementType::QUAD4, conn, n_elements);
// ... other blocks and fields ...
mesh.validate_structure();  // throws mefikit::Error on the first problem
```

A mesh read from a file with `read()` needs no such call: the reader produced it.

### When to prepare again

An operator holds matrices sized by the two meshes it was built from, so
`apply_update` checks that the meshes still have the shape it was prepared for:
the same number of nodes, the same element types, and the same number of cells in
each. Adding or removing a block, or moving cells between element types, means
preparing a new operator. Editing coordinates or connectivity *in place*, which
leaves all of those counts alone, is not detected — nothing in the binding hashes
the geometry. Keep the geometry fixed for an operator's lifetime, which is what
preparing once for many time steps already assumes.

## Error handling

Every fallible method throws a `rust::Error`, which derives from
`std::exception`:

```cpp
try {
  mesh->set_field("T", blocks, values);
} catch (const rust::Error &e) {
  std::cerr << e.what() << "\n";
}
```

`what()` is the Rust `Display` output, so it names the offending element type or
field. Methods that cannot fail are marked `noexcept` in the header.

An enum discriminant this build of mefikit does not know about is an error, not
undefined behaviour: cxx shared enums are open wrappers around an integer, so a
C++ caller can pass any value.

## Safety

`field_values` returns a slice pointing into the mesh's own storage. Keep the
mesh alive for as long as you use it, and do not call a method taking `&mut self`
in between. Nothing enforces that for you: cxx generates `field_values` as a
plain `const` method with no borrow tracking, so upholding the lifetime is the
caller's responsibility. The warning in the header is documentation only.

`rust::Box` and `rust::Vec` own Rust memory and free it on destruction. Do not
copy a `UMesh`; move it (`auto b = std::move(a);`).

## Testing

- `cargo test -p mefikit-ffi` runs the Rust side.
- `ctest --test-dir build/cpp` runs the C++ side, written against the public
  header only, in [`examples/tests/test_mefikit.cpp`](examples/tests/test_mefikit.cpp).

Both suites end with a transfer between the two reference meshes in
`tests/data` (`mesh_27.med` and `mesh_36.med`, 2000 polyhedra each): read both,
put a uniform field on the source, and check that all four methods hand it back
unchanged. That is the only test that touches real `.med` geometry, and it is
where the cost of a wrong answer shows up — a few seconds in Release, a few
times that in Debug.
