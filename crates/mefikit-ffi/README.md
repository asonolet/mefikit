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
| `void validate_structure() const` | cheap consistency check, worth calling on hand-built meshes |
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
in between — that is why the header marks it the way it does.

`rust::Box` and `rust::Vec` own Rust memory and free it on destruction. Do not
copy a `UMesh`; move it (`auto b = std::move(a);`).

## Testing

- `cargo test -p mefikit-ffi` runs the Rust side.
- `ctest --test-dir build/cpp` runs the C++ side, written against the public
  header only, in [`examples/tests/test_mefikit.cpp`](examples/tests/test_mefikit.cpp).
