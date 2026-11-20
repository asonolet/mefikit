# mefikit C++ bindings

Minimal C++ access to mefikit, generated with [`cxx`](https://cxx.rs).
The wrapper exposes only:

- `mefikit::UMesh`: construction from coordinates and regular/poly blocks, plus
  reading and writing one element type's field values;
- `mefikit::ConservativeTransfer`: preparation and application of the P0
  conservative transfer.

No other transfer method or tool is exposed.

```cpp
#include <mefikit/mefikit.hpp>

auto src = mefikit::UMesh::from_coords(coords, n_nodes, 2);
src->add_regular_block(mefikit::ElementType::QUAD4, conn, n_cells);
src->set_field("T", mefikit::ElementType::QUAD4, 1, values);

auto tgt = mefikit::UMesh::from_coords(tgt_coords, tgt_n_nodes, 2);
tgt->add_regular_block(mefikit::ElementType::QUAD4, tgt_conn, tgt_n_cells);

auto transfer = mefikit::ConservativeTransfer::prepare(*src, *tgt, 0.0);
transfer->apply_update(*src, "T", *tgt, "T_target",
                       mefikit::FieldNature::Intensive);
std::vector<double> result =
    mefikit::field_values(*tgt, "T_target", mefikit::ElementType::QUAD4);
```

All fallible calls throw `rust::Error`.

## Build and test

The Rust side must be built first:

```console
$ cargo build -p mefikit-ffi             # or --release
$ cmake -S crates/mefikit-ffi/examples -B build/cpp \
        -DCMAKE_BUILD_TYPE=Release
$ cmake --build build/cpp
$ ctest --test-dir build/cpp --output-on-failure
```

`cargo build` publishes the shared library, cxx's generated header and the C++
glue under `target/<profile>/cxxbridge`.

For a non-CMake build:

```console
$ c++ -std=c++17 -I target/debug/cxxbridge/include \
      my_app.cpp \
      target/debug/cxxbridge/src/ffi.rs.cc \
      target/debug/cxxbridge/src/cxx.cc \
      -L target/debug -lmefikit_ffi -Wl,-rpath,$PWD/target/debug \
      -o my_app
```
