# Standalone CMake project for the mefikit C++ bindings.
#
# It is meant to be read top to bottom as the recipe for consuming mefikit from
# C++: point CMake at the artifacts `cargo build` produced, then link
# `mefikit::mefikit`.
#
#   cmake -S crates/mefikit-ffi/examples -B build/cpp -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/cpp
#   ctest --test-dir build/cpp --output-on-failure
cmake_minimum_required(VERSION 3.20)
project(mefikit_cpp_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(MEFIKIT_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." CACHE PATH
    "Repository root, i.e. the directory holding the workspace Cargo.toml")

# Cargo and CMake disagree about what a build profile is called. This is the only
# translation needed: everything else (debug vs release, opt level, LTO) is cargo's.
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo"
   OR CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
  set(MEFIKIT_CARGO_PROFILE_DIR release)
  set(MEFIKIT_CARGO_FLAGS --release)
else()
  set(MEFIKIT_CARGO_PROFILE_DIR debug)
  set(MEFIKIT_CARGO_FLAGS "")
endif()

# The two artifacts the bindings consist of, both published by
# `cargo build -p mefikit-ffi`:
#   * the shared library holding the Rust implementation,
#   * the include tree holding cxx's generated header plus our umbrella header.
# They sit side by side under target/<profile>, which is what the build script
# arranges so this file does not have to know about cargo's hash directories.
set(MEFIKIT_ARTIFACT_DIR "${MEFIKIT_REPO_ROOT}/target/${MEFIKIT_CARGO_PROFILE_DIR}")
set(MEFIKIT_INCLUDE_DIR "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/include")
set(MEFIKIT_CARGO_HINT
    "cargo build -p mefikit-ffi${MEFIKIT_CARGO_FLAGS}  (from ${MEFIKIT_REPO_ROOT})")

# find_library rather than a hand-built name: it applies the platform's
# lib prefix and extension (.so, .dylib, .dll) for us.
find_library(MEFIKIT_LIBRARY
    NAMES mefikit_ffi
    PATHS "${MEFIKIT_ARTIFACT_DIR}"
    NO_DEFAULT_PATH)

if(NOT MEFIKIT_LIBRARY OR NOT EXISTS "${MEFIKIT_INCLUDE_DIR}/mefikit/mefikit.hpp"
   OR NOT EXISTS "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/src/ffi.rs.cc")
  message(FATAL_ERROR
      "mefikit C++ artifacts not found in ${MEFIKIT_ARTIFACT_DIR}.\n"
      "Build them with:  ${MEFIKIT_CARGO_HINT}\n"
      "Or point -DMEFIKIT_REPO_ROOT=<path> at the repository root.")
endif()

# The C++ half of the bindings. cxx compiled both of these into the Rust library
# already, but C++ mangled symbols are not exported from it, so a C++ program has to
# compile its own copy: ffi.rs.cc is the generated glue for the bridge, cxx.cc the
# out-of-line part of cxx's runtime (rust::String, rust::Error, Slice's internals).
add_library(mefikit_cxx_glue STATIC
    "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/src/ffi.rs.cc"
    "${MEFIKIT_ARTIFACT_DIR}/cxxbridge/src/cxx.cc")
set_target_properties(mefikit_cxx_glue PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON)
target_include_directories(mefikit_cxx_glue PUBLIC "${MEFIKIT_INCLUDE_DIR}")

add_library(mefikit_ffi SHARED IMPORTED)
set_target_properties(mefikit_ffi PROPERTIES
    IMPORTED_LOCATION "${MEFIKIT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${MEFIKIT_INCLUDE_DIR}")

# Order matters on the link line: the glue archive has to be scanned before the
# shared library so that the Rust symbols it calls into are still to come.
target_link_libraries(mefikit_cxx_glue PUBLIC mefikit_ffi)

# The one target consumers link, carrying the include path, the Rust library and the
# C++ glue in the right order.
add_library(mefikit::mefikit INTERFACE IMPORTED GLOBAL)
target_link_libraries(mefikit::mefikit INTERFACE mefikit_cxx_glue)

enable_testing()
add_subdirectory(tests)
