// Minimal C++ API for mefikit.
//
// Only the mesh initialization and the conservative P0 transfer are exposed:
//
//     #include <mefikit/mefikit.hpp>
//
//     auto src = mefikit::UMesh::from_coords(coords, n_nodes, 2);
//     src->add_regular_block(mefikit::ElementType::QUAD4, conn, n_cells);
//     src->set_field("T", mefikit::ElementType::QUAD4, 1, values);
//
//     auto tgt = ...;
//     auto transfer = mefikit::ConservativeTransfer::prepare(*src, *tgt, 0.0);
//     transfer->apply_update(*src, "T", *tgt, "T", mefikit::FieldNature::Intensive);
//     std::vector<double> result = tgt->field_values("T", mefikit::ElementType::QUAD4);
//
// Every fallible call throws `rust::Error`; catch it as `const rust::Error &`.
#pragma once

#include <cstddef>
#include <vector>

#include "rust/cxx.h"
#include "mefikit-ffi/src/ffi.rs.h"

namespace mefikit {

// cxx returns a `rust::Vec` from the bridge. This helper copies it into a
// `std::vector` so callers do not need to depend on cxx's container type.
inline std::vector<double> field_values(const UMesh &mesh, const rust::Str &name,
                                        ElementType element_type) {
  const rust::Vec<double> values = mesh.field_values(name, element_type);
  return std::vector<double>(values.begin(), values.end());
}

} // namespace mefikit
