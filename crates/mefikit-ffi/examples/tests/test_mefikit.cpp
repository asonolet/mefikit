// Tests for the mefikit C++ bindings, written against only the public umbrella
// header. Deliberately dependency-free: CMake just needs a C++17 compiler.
#include <mefikit/mefikit.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void report(bool ok, const char *expr, const char *file, int line, const std::string &note) {
  ++checks;
  if (ok) {
    return;
  }
  ++failures;
  std::fprintf(stderr, "FAIL %s:%d: %s%s%s\n", file, line, expr, note.empty() ? "" : " -- ",
               note.c_str());
}

#define CHECK(expr) report(static_cast<bool>(expr), #expr, __FILE__, __LINE__, "")

// Streamed rather than std::to_string so the macro also works for types that have
// no to_string overload. (Scoped enums are not streamable; those are compared with
// CHECK instead.)
template <typename T>
std::string show(const T &value) {
  std::ostringstream out;
  out << value;
  return out.str();
}

#define CHECK_EQ(lhs, rhs)                                                            \
  do {                                                                                \
    auto &&mevikit_lhs_ = (lhs);                                                       \
    auto &&mevikit_rhs_ = (rhs);                                                       \
    report(mevikit_lhs_ == mevikit_rhs_, #lhs " == " #rhs, __FILE__, __LINE__,         \
           "got " + show(mevikit_lhs_) + " vs " + show(mevikit_rhs_));                 \
  } while (false)

#define CHECK_NEAR(lhs, rhs, tol)                                                     \
  do {                                                                                \
    double mevikit_lhs_ = static_cast<double>(lhs);                                    \
    double mevikit_rhs_ = static_cast<double>(rhs);                                    \
    report(std::fabs(mevikit_lhs_ - mevikit_rhs_) <= (tol), #lhs " ~= " #rhs,           \
           __FILE__, __LINE__, "got " + show(mevikit_lhs_) + " vs " +                   \
                                   show(mevikit_rhs_));                                 \
  } while (false)

// cxx has no implicit std::vector -> Slice conversion, so spell it out once here.
template <typename T>
rust::Slice<const T> slice_of(const std::vector<T> &v) {
  return rust::Slice<const T>(v.data(), v.size());
}

// Builds a structured 2D mesh of n x n QUAD4 cells over [0, 1]^2.
rust::Box<mefikit::UMesh> cmesh(int n) {
  std::vector<double> coords;
  for (int j = 0; j <= n; ++j) {
    for (int i = 0; i <= n; ++i) {
      coords.push_back(static_cast<double>(i) / n);
      coords.push_back(static_cast<double>(j) / n);
    }
  }
  const std::size_t n_nodes = static_cast<std::size_t>(n + 1) * (n + 1);

  std::vector<std::size_t> conn;
  conn.reserve(static_cast<std::size_t>(n) * n * 4);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const std::size_t stride = static_cast<std::size_t>(n + 1);
      const std::size_t base = static_cast<std::size_t>(j) * stride + i;
      conn.push_back(base);
      conn.push_back(base + 1);
      conn.push_back(base + stride + 1);
      conn.push_back(base + stride);
    }
  }

  auto mesh = mefikit::UMesh::from_coords(slice_of(coords), n_nodes, 2);
  mesh->add_regular_block(mefikit::ElementType::QUAD4, slice_of(conn),
                          static_cast<std::size_t>(n) * n);
  return mesh;
}

// A mesh holding both a QUAD4 and a TRI3 block, i.e. two element types at the
// same topological dimension, which is what multi-block fields look like.
rust::Box<mefikit::UMesh> mixed_mesh() {
  const std::vector<double> coords{0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                                   0.0, 1.0, 0.0, 2.0, 1.0, 2.0};
  auto mesh = mefikit::UMesh::from_coords(slice_of(coords), 6, 2);
  mesh->add_regular_block(mefikit::ElementType::QUAD4,
                          slice_of(std::vector<std::size_t>{0, 1, 2, 3}), 1);
  mesh->add_regular_block(mefikit::ElementType::TRI3,
                          slice_of(std::vector<std::size_t>{2, 3, 4, 4, 5, 2}), 2);
  mesh->validate_structure();
  return mesh;
}

void test_topology() {
  auto mesh = cmesh(4);
  CHECK_EQ(mesh->n_nodes(), 25u);
  CHECK_EQ(mesh->n_elements(), 16u);
  CHECK_EQ(mesh->n_elements_of(mefikit::ElementType::QUAD4), 16u);
  CHECK_EQ(mesh->n_elements_of(mefikit::ElementType::TET4), 0u);
  CHECK_EQ(mesh->space_dimension(), 2u);
  CHECK(!mesh->is_empty());
  CHECK(mesh->topological_dimension() == mefikit::Dimension::D2);
  CHECK_EQ(mesh->element_types().size(), 1u);
  CHECK(mesh->element_types()[0] == mefikit::ElementType::QUAD4);
  mesh->validate_structure();
}

void test_empty_mesh() {
  const std::vector<double> coords{0.0, 0.0, 1.0, 0.0};
  auto mesh = mefikit::UMesh::from_coords(slice_of(coords), 2, 2);
  CHECK(mesh->is_empty());
  CHECK_EQ(mesh->n_nodes(), 2u);
  CHECK_EQ(mesh->n_elements(), 0u);
}

void test_poly_block() {
  const std::vector<double> coords{0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
                                   2.0, 0.0, 1.5, 1.5, 0.5, 1.5};
  auto mesh = mefikit::UMesh::from_coords(slice_of(coords), 6, 2);
  // One triangle then one quadrilateral, described by cumulative end offsets.
  mesh->add_poly_block(mefikit::ElementType::PGON,
                       slice_of(std::vector<std::size_t>{0, 1, 2, 2, 3, 4, 5}),
                       slice_of(std::vector<std::size_t>{3, 7}));
  mesh->validate_structure();
  CHECK_EQ(mesh->n_elements_of(mefikit::ElementType::PGON), 2u);
  CHECK(mesh->topological_dimension() == mefikit::Dimension::D2);
}

void test_field_round_trip() {
  auto mesh = cmesh(2);
  const std::vector<double> values{1.0, 2.0, 3.0, 4.0};
  mesh->set_field_uniform("T", mefikit::ElementType::QUAD4, 1, slice_of(values));

  const auto info = mesh->field_info("T", mefikit::ElementType::QUAD4);
  CHECK_EQ(info.n_elements, 4u);
  CHECK_EQ(info.n_components, 1u);

  const auto read_back = mesh->field_values("T", mefikit::ElementType::QUAD4);
  CHECK_EQ(read_back.size(), values.size());
  for (std::size_t i = 0; i < read_back.size(); ++i) {
    CHECK_NEAR(read_back[i], values[i], 0.0);
  }
  CHECK_EQ(mesh->field_names().size(), 1u);
  CHECK_EQ(std::string(mesh->field_names()[0]), std::string("T"));
}

void test_multi_block_field() {
  auto mesh = mixed_mesh();
  // One QUAD4 value then two TRI3 values, laid out exactly as set_field expects.
  const std::vector<mefikit::FieldBlock> blocks{
      {mefikit::ElementType::QUAD4, 1, 0, 1},
      {mefikit::ElementType::TRI3, 1, 1, 2},
  };
  const std::vector<double> values{1.0, 2.0, 3.0};
  mesh->set_field("T", slice_of(blocks), slice_of(values));

  CHECK_NEAR(mesh->field_values("T", mefikit::ElementType::QUAD4)[0], 1.0, 0.0);
  const auto tri = mesh->field_values("T", mefikit::ElementType::TRI3);
  CHECK_EQ(tri.size(), 2u);
  CHECK_NEAR(tri[1], 3.0, 0.0);

  // set_field_uniform is the single-block shortcut and must refuse this mesh.
  bool threw = false;
  try {
    mesh->set_field_uniform("U", mefikit::ElementType::QUAD4, 1, slice_of(values));
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);
}

void test_errors_are_reported() {
  auto mesh = cmesh(2);

  bool threw = false;
  try {
    mesh->field_info("missing", mefikit::ElementType::QUAD4);
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("missing") != std::string::npos);
  }
  CHECK(threw);

  // A discriminant this mefikit build does not know about is an error, not a crash.
  threw = false;
  try {
    mesh->add_regular_block(static_cast<mefikit::ElementType>(250),
                            slice_of(std::vector<std::size_t>{0, 1, 2, 3}), 1);
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);

  // Wrong value count for the declared shape.
  threw = false;
  try {
    mesh->set_field_uniform("T", mefikit::ElementType::QUAD4, 1,
                            slice_of(std::vector<double>{1.0, 2.0}));
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);

  // Unwritable path.
  threw = false;
  try {
    mesh->write("/nonexistent-directory-mefikit/mesh.json");
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);
}

void test_transfer_identity() {
  // Same geometry on both sides with a piecewise-constant method: the transferred
  // field must come back untouched, whatever the field happens to be.
  for (int n : {1, 2, 3}) {
    auto src = cmesh(n);
    auto tgt = cmesh(n);

    std::vector<double> values;
    for (std::size_t i = 0; i < src->n_elements_of(mefikit::ElementType::QUAD4); ++i) {
      values.push_back(static_cast<double>(i) * 1.5 - 4.0);
    }
    src->set_field_uniform("T", mefikit::ElementType::QUAD4, 1, slice_of(values));

    auto op = mefikit::TransferOperator::prepare(
        *src, *tgt, mefikit::constant_piecewise(mefikit::PointLocation::Centroid));
    CHECK(op->target_dimension() == mefikit::Dimension::D2);
    op->apply_update(*src, "T", *tgt, "T", 0.0, mefikit::FieldNature::Intensive);

    const auto transferred = tgt->field_values("T", mefikit::ElementType::QUAD4);
    CHECK_EQ(transferred.size(), values.size());
    for (std::size_t i = 0; i < transferred.size(); ++i) {
      CHECK_NEAR(transferred[i], values[i], 1e-12);
    }
  }
}

void test_transfer_finds_neighbour_cell() {
  // One source cell over [0, 2]^2 holding 7.0, split into two target cells whose
  // centroids both fall inside it, so a piecewise-constant transfer must give 7.0.
  const std::vector<double> src_coords{0.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0};
  auto src = mefikit::UMesh::from_coords(slice_of(src_coords), 4, 2);
  src->add_regular_block(mefikit::ElementType::QUAD4,
                         slice_of(std::vector<std::size_t>{0, 1, 2, 3}), 1);
  src->set_field_uniform("T", mefikit::ElementType::QUAD4, 1,
                         slice_of(std::vector<double>{7.0}));

  // A 2 x 1 grid: (0,0) (1,0) (2,0) (0,2) (1,2) (2,2).
  const std::vector<double> tgt_coords{0.0, 0.0, 1.0, 0.0, 2.0, 0.0,
                                       0.0, 2.0, 1.0, 2.0, 2.0, 2.0};
  auto tgt = mefikit::UMesh::from_coords(slice_of(tgt_coords), 6, 2);
  tgt->add_regular_block(mefikit::ElementType::QUAD4,
                         slice_of(std::vector<std::size_t>{0, 1, 4, 3, 1, 2, 5, 4}), 2);

  mefikit::transfer_field(*src, "T", *tgt, "T",
                          mefikit::constant_piecewise(
                              mefikit::PointLocation::Centroid),
                          0.0, mefikit::FieldNature::Intensive);
  const auto got = tgt->field_values("T", mefikit::ElementType::QUAD4);
  CHECK_EQ(got.size(), 2u);
  CHECK_NEAR(got[0], 7.0, 1e-12);
  CHECK_NEAR(got[1], 7.0, 1e-12);
}

void test_transfer_operator_is_reusable() {
  auto src = cmesh(4);
  std::vector<double> a;
  std::vector<double> b;
  for (std::size_t i = 0; i < 16; ++i) {
    a.push_back(static_cast<double>(i));
    b.push_back(static_cast<double>(i) * 10.0);
  }
  src->set_field_uniform("a", mefikit::ElementType::QUAD4, 1, slice_of(a));
  src->set_field_uniform("b", mefikit::ElementType::QUAD4, 1, slice_of(b));

  auto tgt = cmesh(4);
  auto op = mefikit::TransferOperator::prepare(
      *src, *tgt, mefikit::inverse_distance(4));

  // Two fields, one prepare.
  op->apply_update(*src, "a", *tgt, "a", -1.0, mefikit::FieldNature::Intensive);
  op->apply_update(*src, "b", *tgt, "b", -1.0, mefikit::FieldNature::Intensive);

  const auto ra = tgt->field_values("a", mefikit::ElementType::QUAD4);
  const auto rb = tgt->field_values("b", mefikit::ElementType::QUAD4);
  for (std::size_t i = 0; i < 16; ++i) {
    CHECK_NEAR(ra[i], a[i], 1e-9);
    CHECK_NEAR(rb[i], b[i], 1e-9);
  }

  // Re-applying under a new name replaces nothing and adds a field.
  op->apply_update(*src, "a", *tgt, "a_again", -1.0, mefikit::FieldNature::Intensive);
  const auto again = tgt->field_values("a_again", mefikit::ElementType::QUAD4);
  for (std::size_t i = 0; i < 16; ++i) {
    CHECK_NEAR(again[i], a[i], 1e-9);
  }
  CHECK_NEAR(tgt->field_values("a", mefikit::ElementType::QUAD4)[0], a[0], 1e-9);
}

void test_default_value_fills_uncovered_cells() {
  auto src = cmesh(2);
  src->set_field_uniform("T", mefikit::ElementType::QUAD4, 1,
                         slice_of(std::vector<double>{1.0, 2.0, 3.0, 4.0}));

  // Target is shifted off the source, so nothing is covered: every cell must get
  // the default rather than a NaN.
  const std::vector<double> tgt_coords{5.0, 5.0, 6.0, 5.0, 6.0, 6.0, 5.0, 6.0};
  auto tgt = mefikit::UMesh::from_coords(slice_of(tgt_coords), 4, 2);
  tgt->add_regular_block(mefikit::ElementType::QUAD4,
                         slice_of(std::vector<std::size_t>{0, 1, 2, 3}), 1);

  auto op = mefikit::TransferOperator::prepare(
      *src, *tgt, mefikit::conservative_p0());
  op->apply_update(*src, "T", *tgt, "T", -999.0, mefikit::FieldNature::Intensive);
  CHECK_NEAR(tgt->field_values("T", mefikit::ElementType::QUAD4)[0], -999.0, 0.0);
}

void test_io_round_trip() {
  const std::string path = "mefikit_cpp_test_mesh.json";
  {
    auto mesh = mixed_mesh();
    const std::vector<mefikit::FieldBlock> blocks{
        {mefikit::ElementType::QUAD4, 1, 0, 1},
        {mefikit::ElementType::TRI3, 1, 1, 2},
    };
    mesh->set_field("T", slice_of(blocks),
                    slice_of(std::vector<double>{1.5, 2.5, 3.5}));
    mesh->write(path);
  }

  auto reloaded = mefikit::UMesh::read(path);
  CHECK_EQ(reloaded->n_nodes(), 6u);
  CHECK_EQ(reloaded->n_elements(), 3u);
  CHECK_EQ(reloaded->n_elements_of(mefikit::ElementType::QUAD4), 1u);
  CHECK_EQ(reloaded->n_elements_of(mefikit::ElementType::TRI3), 2u);
  CHECK_NEAR(reloaded->field_values("T", mefikit::ElementType::QUAD4)[0], 1.5, 0.0);
  const auto tri = reloaded->field_values("T", mefikit::ElementType::TRI3);
  CHECK_NEAR(tri[0], 2.5, 0.0);
  CHECK_NEAR(tri[1], 3.5, 0.0);
  std::remove(path.c_str());

  // A missing file is an error, not a crash.
  bool threw = false;
  try {
    auto missing = mefikit::UMesh::read("mefikit_does_not_exist.json");
    (void)missing;
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);
}

// The two reference meshes in the repository's test data: 2000 polyhedra each,
// two different discretizations of the unit cube. Reading a real .med file and
// moving a field between them is the workflow the bindings exist for, so the
// suite does it once against real geometry rather than only small structured
// meshes.
void test_med_transfer_of_a_uniform_field() {
  const std::string dir = MEFIKIT_TEST_DATA_DIR;
  auto src = mefikit::UMesh::read(dir + "/mesh_27.med");
  auto tgt = mefikit::UMesh::read(dir + "/mesh_36.med");

  // Deliberately not calling validate_structure() on these two. It fails, and not
  // because of the bindings: the core's MED reader marks the face boundaries
  // inside a polyhedron with usize::MAX sentinels (crates/mefikit/src/io/med_io.rs,
  // "mefikit convention"), while the core's own validate_structure() rejects any
  // node index >= n_nodes (crates/mefikit/src/mesh/umesh.rs). So a mesh read from
  // a .med file fails the validator the core wrote for it, and the transfer below
  // is what shows the mesh is nonetheless usable.

  CHECK_EQ(src->n_elements(), 2000u);
  CHECK_EQ(tgt->n_elements(), 2000u);
  CHECK_EQ(src->n_elements_of(mefikit::ElementType::PHED), 2000u);
  CHECK_EQ(src->n_elements_of(mefikit::ElementType::TET4), 0u);
  CHECK_EQ(src->space_dimension(), 3u);
  CHECK(src->topological_dimension() == mefikit::Dimension::D3);
  CHECK_EQ(src->element_types().size(), 1u);
  CHECK(src->element_types()[0] == mefikit::ElementType::PHED);

  // A field of 1.0 on every cell of the source. Every method here is an average
  // or a least-squares fit, so a constant field has to come back constant: the
  // value cannot depend on how the two meshes happen to be cut up.
  const std::size_t n_src = src->n_elements_of(mefikit::ElementType::PHED);
  const std::vector<double> ones(n_src, 1.0);
  const std::vector<double> minus_ones(n_src, -1.0);
  src->set_field_uniform("u", mefikit::ElementType::PHED, 1, slice_of(ones));
  src->set_field_uniform("v", mefikit::ElementType::PHED, 1, slice_of(minus_ones));

  // The meshes discretize the same unit cube, so no target cell is left without a
  // source cell to average, and none of them falls back to the default value.
  const double kDefault = 0.0;
  const std::vector<mefikit::TransferMethod> methods{
      mefikit::conservative_p0(),
      mefikit::constant_piecewise(mefikit::PointLocation::Centroid),
      mefikit::inverse_distance(8),
      mefikit::moving_least_squares(8, mefikit::DistanceWeighting::Gaussian),
  };

  for (const auto &method : methods) {
    // One prepare, two fields: the shape of a time-step loop, and the reason
    // prepare() is separate from apply_update().
    auto op = mefikit::TransferOperator::prepare(*src, *tgt, method);
    op->apply_update(*src, "u", *tgt, "u", kDefault, mefikit::FieldNature::Intensive);
    op->apply_update(*src, "v", *tgt, "v", kDefault, mefikit::FieldNature::Intensive);

    const auto got = tgt->field_values("u", mefikit::ElementType::PHED);
    CHECK_EQ(got.size(), 2000u);
    const auto got_minus = tgt->field_values("v", mefikit::ElementType::PHED);
    CHECK_EQ(got_minus.size(), 2000u);

    double worst = 0.0;
    double worst_minus = 0.0;
    std::size_t defaulted = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      worst = std::fmax(worst, std::fabs(got[i] - 1.0));
      worst_minus = std::fmax(worst_minus, std::fabs(got_minus[i] + 1.0));
      defaulted += (got[i] == kDefault);
    }
    report(worst < 1e-5, "a uniform field transfers as a uniform field", __FILE__,
           __LINE__, "worst deviation " + show(worst));
    report(worst_minus < 1e-5, "a second field transfers independently", __FILE__,
           __LINE__, "worst deviation " + show(worst_minus));
    report(defaulted == 0, "every target cell was covered by the source", __FILE__,
           __LINE__, show(defaulted) + " cells took the default value");
  }

  // The other direction, on the same two files: 36 -> 27.
  auto back = mefikit::TransferOperator::prepare(*tgt, *src, mefikit::conservative_p0());
  back->apply_update(*tgt, "u", *src, "u_back", kDefault, mefikit::FieldNature::Intensive);
  const auto reversed = src->field_values("u_back", mefikit::ElementType::PHED);
  CHECK_EQ(reversed.size(), 2000u);
  double worst_back = 0.0;
  for (std::size_t i = 0; i < reversed.size(); ++i) {
    worst_back = std::fmax(worst_back, std::fabs(reversed[i] - 1.0));
  }
  report(worst_back < 1e-5, "the transfer works in the reverse direction too", __FILE__,
         __LINE__, "worst deviation " + show(worst_back));
}

void test_transfer_method_factories() {
  // The factories exist so unused parameters stay zeroed; check the shapes they
  // produce, and that every one of them is accepted by prepare().
  auto src = cmesh(3);
  auto tgt = cmesh(2);

  const std::vector<mefikit::TransferMethod> methods{
      mefikit::constant_piecewise(mefikit::PointLocation::Centroid),
      mefikit::conservative_p0(),
      mefikit::inverse_distance(2, 1.0),
      mefikit::inverse_distance(5),
      mefikit::moving_least_squares(4, mefikit::DistanceWeighting::Gaussian),
      mefikit::moving_least_squares(
          4, mefikit::DistanceWeighting::CompactSupport, 1.5),
  };
  for (const auto &method : methods) {
    bool threw = false;
    try {
      auto op = mefikit::TransferOperator::prepare(*src, *tgt, method);
      (void)op;
    } catch (const rust::Error &) {
      threw = true;
    }
    CHECK(!threw);
  }

  // mefikit has not implemented these yet; the bindings must say so instead of
  // letting the panic cross the bridge, which would abort the process.
  for (auto point : {mefikit::PointLocation::Barycenter,
                     mefikit::PointLocation::StrictInterior}) {
    bool threw = false;
    try {
      auto op = mefikit::TransferOperator::prepare(
          *src, *tgt, mefikit::constant_piecewise(point));
      (void)op;
    } catch (const rust::Error &e) {
      threw = true;
      CHECK(std::string(e.what()).find("not implemented") != std::string::npos);
    }
    CHECK(threw);
  }
}


// Two HEX8 cells over [0, 2] x [0, 1] x [0, 1]: a 3D mesh that is not the same
// size as the one above, to check that an operator notices the difference.
rust::Box<mefikit::UMesh> two_cell_hex_mesh() {
  const std::vector<double> coords{0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 0,
                                   1, 1, 0, 2, 1, 0, 0, 0, 1, 1, 0, 1,
                                   2, 0, 1, 1, 1, 1, 0, 1, 1, 2, 1, 1};
  auto mesh = mefikit::UMesh::from_coords(slice_of(coords), 12, 3);
  mesh->add_regular_block(
      mefikit::ElementType::HEX8,
      slice_of(std::vector<std::size_t>{0, 1, 4, 3, 6, 7, 10, 9, 1, 2, 5, 4,
                                        7, 8, 11, 10}),
      2);
  return mesh;
}

// mefikit states the preconditions of its transfers as assert!s, which abort the
// process rather than unwinding. Every one of those cases has to come back as a
// rust::Error, because a C++ caller has no way to handle a Rust panic.
void test_transfers_that_cannot_work_are_reported() {
  auto a = cmesh(2);
  auto b = cmesh(2);
  const auto p0 = mefikit::conservative_p0();
  const auto id = mefikit::inverse_distance(1, 2.0);

  // Coordinates but no elements at all: nothing to interpolate from.
  auto bare = mefikit::UMesh::from_coords(
      slice_of(std::vector<double>{0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0}), 4, 2);

  const std::vector<std::tuple<const char *, mefikit::UMesh *, mefikit::UMesh *,
                               mefikit::TransferMethod, const char *>>
      cases{
          {"no source cells", &*bare, &*a, p0, "no elements"},
          {"no target cells", &*a, &*bare, p0, "no elements"},
          {"k of zero", &*a, &*b, mefikit::inverse_distance(0, 2.0),
           "k of at least 1"},
          {"exponent of zero", &*a, &*b, mefikit::inverse_distance(3, 0.0),
           "positive exponent"},
          {"negative exponent", &*a, &*b, mefikit::inverse_distance(3, -1.0),
           "positive exponent"},
          {"mls k of zero", &*a, &*b,
           mefikit::moving_least_squares(0, mefikit::DistanceWeighting::Gaussian),
           "k of at least 1"},
      };
  for (const auto &[label, src, tgt, method, expected] : cases) {
    bool threw = false;
    try {
      auto op = mefikit::TransferOperator::prepare(*src, *tgt, method);
      (void)op;
    } catch (const rust::Error &e) {
      threw = true;
      if (std::string(e.what()).find(expected) == std::string::npos) {
        report(false, "error message", __FILE__, __LINE__,
               std::string(label) + ": '" + e.what() + "'");
      }
    }
    report(threw, "transfer refused", __FILE__, __LINE__, label);
  }

  // Meshes in different spaces, and a source of cells that do not fill their
  // space: both are refused for the methods that integrate over cells.
  auto hex = mefikit::UMesh::from_coords(
      slice_of(std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0,
                                   0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                   1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
                                   1.0}),
      8, 3);
  hex->add_regular_block(mefikit::ElementType::HEX8,
                         slice_of(std::vector<std::size_t>{0, 1, 2, 3, 4, 5, 6, 7}),
                         1);
  auto surface = mefikit::UMesh::from_coords(
      slice_of(std::vector<double>{0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0}),
      3, 3);
  surface->add_regular_block(mefikit::ElementType::TRI3,
                             slice_of(std::vector<std::size_t>{0, 1, 2}), 1);

  const std::vector<std::pair<const char *, mefikit::TransferMethod>> cell_methods{
      {"conservative p0", p0},
      {"constant piecewise",
       mefikit::constant_piecewise(mefikit::PointLocation::Centroid)},
  };
  for (const auto &[name, method] : cell_methods) {
    for (const auto &[dir, src, tgt] :
         std::vector<std::tuple<const char *, mefikit::UMesh *, mefikit::UMesh *>>{
             {"surface source", &*surface, &*hex},
             {"surface target", &*hex, &*surface}}) {
      bool threw = false;
      try {
        auto op = mefikit::TransferOperator::prepare(*src, *tgt, method);
        (void)op;
      } catch (const rust::Error &) {
        threw = true;
      }
      report(threw, name, __FILE__, __LINE__, dir);
    }
  }

  // 2D against 3D.
  bool threw = false;
  try {
    auto op = mefikit::TransferOperator::prepare(*a, *hex, p0);
    (void)op;
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("same space dimension") != std::string::npos);
  }
  CHECK(threw);

  // A method that is defined on a lower-dimensional source still works.
  hex->set_field_uniform("T", mefikit::ElementType::HEX8, 1,
                         slice_of(std::vector<double>{1.0}));
  auto op = mefikit::TransferOperator::prepare(*hex, *surface, id);
  op->apply_update(*hex, "T", *surface, "T", 0.0, mefikit::FieldNature::Intensive);
  CHECK_NEAR(surface->field_values("T", mefikit::ElementType::TRI3)[0], 1.0, 1e-12);

  // A source of the right dimensionality but the wrong size: the operator's
  // matrices are sized by the mesh it was built from, so this has to be caught
  // rather than read past the end of.
  auto two_cells = two_cell_hex_mesh();
  two_cells->set_field_uniform("T", mefikit::ElementType::HEX8, 1,
                               slice_of(std::vector<double>{1.0, 1.0}));
  threw = false;
  try {
    op->apply_update(*two_cells, "T", *surface, "T", 0.0,
                     mefikit::FieldNature::Intensive);
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("must not change") != std::string::npos);
  }
  CHECK(threw);

  // The same mesh sizes do not make an operator valid for a different mesh
  // shape: prepare again for a target that has since gained a block.
  auto src = cmesh(2);
  src->set_field_uniform("T", mefikit::ElementType::QUAD4, 1,
                         slice_of(std::vector<double>{1.0, 2.0, 3.0, 4.0}));
  auto tgt = cmesh(2);
  auto op2 = mefikit::TransferOperator::prepare(*src, *tgt, p0);
  op2->apply_update(*src, "T", *tgt, "T", 0.0, mefikit::FieldNature::Intensive);
  tgt->add_regular_block(mefikit::ElementType::TRI3,
                         slice_of(std::vector<std::size_t>{0, 1, 4}), 1);
  threw = false;
  try {
    op2->apply_update(*src, "T", *tgt, "T", 0.0, mefikit::FieldNature::Intensive);
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("prepared for a target mesh") !=
          std::string::npos);
  }
  CHECK(threw);

}

// The core keeps the first block when a second of the same type is added, and
// indexes nodes with the connectivity it is given, so both have to be checked
// where the bindings can see what went wrong.
void test_malformed_meshes_are_rejected() {
  auto mesh = mixed_mesh();

  bool threw = false;
  try {
    mesh->add_regular_block(mefikit::ElementType::QUAD4,
                            slice_of(std::vector<std::size_t>{3, 2, 1, 0}), 1);
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("already has a QUAD4 block") !=
          std::string::npos);
  }
  CHECK(threw);

  // Offsets that do not describe the connectivity handed over.
  const std::vector<std::pair<std::string, std::vector<std::size_t>>> bad_offsets{
      {"backwards", {5, 3}}, {"empty element", {0, 3}}, {"short", {2}}};
  for (const auto &[label, offsets] : bad_offsets) {
    threw = false;
    try {
      mesh->add_poly_block(mefikit::ElementType::PGON,
                           slice_of(std::vector<std::size_t>{0, 1, 2, 3}),
                           slice_of(offsets));
    } catch (const rust::Error &) {
      threw = true;
    }
    report(threw, "poly offsets", __FILE__, __LINE__, label);
  }

  // A node index the mesh does not have.
  auto bare = mefikit::UMesh::from_coords(
      slice_of(std::vector<double>{0.0, 0.0, 1.0, 0.0, 1.0, 1.0}), 3, 2);
  threw = false;
  try {
    bare->add_regular_block(mefikit::ElementType::TRI3,
                            slice_of(std::vector<std::size_t>{0, 1, 99}), 1);
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("node 99") != std::string::npos);
  }
  CHECK(threw);

  // Coordinates that are not finite, and a space mefikit does not work in.
  threw = false;
  try {
    auto nan_mesh = mefikit::UMesh::from_coords(
        slice_of(std::vector<double>{0.0, 0.0, 0.0, std::nan(""), 0.0, 0.0}), 3, 2);
    (void)nan_mesh;
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("not finite") != std::string::npos);
  }
  CHECK(threw);

  threw = false;
  try {
    auto flat = mefikit::UMesh::from_coords(slice_of(std::vector<double>{0.0}), 1, 0);
    (void)flat;
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("space_dim must be") != std::string::npos);
  }
  CHECK(threw);

  // One field, one shape: a transfer reads it as a single array.
  const std::vector<mefikit::FieldBlock> ragged{
      mefikit::FieldBlock{mefikit::ElementType::QUAD4, 1, 0, 1},
      mefikit::FieldBlock{mefikit::ElementType::TRI3, 3, 1, 6},
  };
  threw = false;
  try {
    mesh->set_field("T", slice_of(ragged),
                    slice_of(std::vector<double>{1, 2, 3, 4, 5, 6, 7}));
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("1 components on the first block but 3") !=
          std::string::npos);
  }
  CHECK(threw);

  // A field name is looked up by string everywhere, so an empty one is at best
  // unreachable. (On a single-block mesh, so the check reached is the name and
  // not the number of blocks.)
  auto single = cmesh(2);
  threw = false;
  try {
    single->set_field_uniform("", mefikit::ElementType::QUAD4, 1,
                               slice_of(std::vector<double>{1.0}));
  } catch (const rust::Error &e) {
    threw = true;
    CHECK(std::string(e.what()).find("may not be empty") != std::string::npos);
  }
  CHECK(threw);
}

} // namespace

// Checks is process-wide, so this test puts the level back before returning: a
// Fast left behind would silence the malformed-mesh checks the next test relies
// on.
void test_check_level_is_readable_and_settable() {
  const auto before = mefikit::checks();

  mefikit::set_checks(mefikit::Checks::Fast);
  CHECK(mefikit::checks() == mefikit::Checks::Fast);

  mefikit::set_checks(mefikit::Checks::Full);
  CHECK(mefikit::checks() == mefikit::Checks::Full);

  // Everything outside the whole-mesh scans is still checked in Fast, so a bad
  // field name keeps being an error rather than becoming a wrong answer.
  auto mesh = cmesh(2);
  const auto blocks =
      std::vector<mefikit::FieldBlock>{mefikit::FieldBlock{mefikit::ElementType::QUAD4, 1, 0, 4}};
  bool threw = false;
  try {
    mesh->set_field("", slice_of(blocks), slice_of(std::vector<double>{1, 2, 3, 4}));
  } catch (const rust::Error &) {
    threw = true;
  }
  CHECK(threw);

  mefikit::set_checks(before);
  CHECK(mefikit::checks() == before);
}

int main() {
  test_topology();
  test_empty_mesh();
  test_poly_block();
  test_field_round_trip();
  test_multi_block_field();
  test_errors_are_reported();
  test_transfer_identity();
  test_transfer_finds_neighbour_cell();
  test_transfer_operator_is_reusable();
  test_default_value_fills_uncovered_cells();
  test_io_round_trip();
  test_transfer_method_factories();
  test_transfers_that_cannot_work_are_reported();
  test_malformed_meshes_are_rejected();
  test_med_transfer_of_a_uniform_field();
  test_check_level_is_readable_and_settable();

  std::printf("%d checks, %d failures\n", checks, failures);
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
