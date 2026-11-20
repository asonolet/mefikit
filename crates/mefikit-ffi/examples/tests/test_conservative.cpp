#include <mefikit/mefikit.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

rust::Box<mefikit::UMesh> make_grid(int n, double lo, double hi) {
  std::vector<double> coords;
  coords.reserve(static_cast<std::size_t>((n + 1) * (n + 1) * 2));
  for (int i = 0; i <= n; ++i) {
    const double x = lo + (hi - lo) * static_cast<double>(i) / n;
    for (int j = 0; j <= n; ++j) {
      const double y = lo + (hi - lo) * static_cast<double>(j) / n;
      coords.push_back(x);
      coords.push_back(y);
    }
  }

  std::vector<std::size_t> conn;
  conn.reserve(static_cast<std::size_t>(n * n * 4));
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      const std::size_t n0 = static_cast<std::size_t>(i * (n + 1) + j);
      const std::size_t n1 = n0 + 1;
      const std::size_t n2 = n0 + static_cast<std::size_t>(n + 1) + 1;
      const std::size_t n3 = n0 + static_cast<std::size_t>(n + 1);
      conn.insert(conn.end(), {n0, n1, n2, n3});
    }
  }

  auto mesh = mefikit::UMesh::from_coords(
      coords, static_cast<std::size_t>((n + 1) * (n + 1)), 2);
  mesh->add_regular_block(mefikit::ElementType::QUAD4, conn,
                          static_cast<std::size_t>(n * n));
  return mesh;
}

void check_close(double actual, double expected, const char *what) {
  if (std::abs(actual - expected) > 1e-12) {
    throw std::runtime_error(std::string(what) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

} // namespace

int main() {
  try {
    auto src = make_grid(2, 0.0, 1.0);
    auto tgt = make_grid(3, 0.05, 0.95);

    if (src->num_elements(mefikit::ElementType::QUAD4) != 4) {
      throw std::runtime_error("source should have four QUAD4 cells");
    }
    if (tgt->num_elements(mefikit::ElementType::QUAD4) != 9) {
      throw std::runtime_error("target should have nine QUAD4 cells");
    }

    // Two components per cell, constant over the source mesh.
    std::vector<double> values;
    for (int cell = 0; cell < 4; ++cell) {
      values.push_back(7.0);
      values.push_back(3.0);
    }
    src->set_field("T", mefikit::ElementType::QUAD4, 2, values);

    auto transfer = mefikit::ConservativeTransfer::prepare(*src, *tgt, 0.0);
    transfer->apply_update(*src, "T", *tgt, "T_target",
                           mefikit::FieldNature::Intensive);

    const std::vector<double> result =
        mefikit::field_values(*tgt, "T_target", mefikit::ElementType::QUAD4);
    if (result.size() != 9 * 2) {
      throw std::runtime_error("unexpected target field length");
    }
    for (std::size_t cell = 0; cell < 9; ++cell) {
      check_close(result[2 * cell], 7.0, "component 0");
      check_close(result[2 * cell + 1], 3.0, "component 1");
    }

    // Poly block initialization and field round-trip.
    const std::vector<double> square_coords = {0.0, 0.0, 1.0, 0.0,
                                               1.0, 1.0, 0.0, 1.0};
    const std::vector<std::size_t> polygon = {0, 1, 2, 3};
    const std::vector<std::size_t> offsets = {4};
    auto poly = mefikit::UMesh::from_coords(square_coords, 4, 2);
    poly->add_poly_block(mefikit::ElementType::PGON, polygon, offsets);
    poly->set_field("P", mefikit::ElementType::PGON, 1, {42.0});
    const std::vector<double> poly_values =
        mefikit::field_values(*poly, "P", mefikit::ElementType::PGON);
    if (poly_values.size() != 1) {
      throw std::runtime_error("unexpected PGON field length");
    }
    check_close(poly_values[0], 42.0, "PGON field value");

    // A field spanning several element types is staged until the last block
    // arrives, then committed to the mesh.
    const std::vector<double> mixed_coords = {
        0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0,
    };
    auto mixed = mefikit::UMesh::from_coords(mixed_coords, 6, 2);
    mixed->add_regular_block(mefikit::ElementType::QUAD4, {0, 1, 4, 3}, 1);
    mixed->add_regular_block(mefikit::ElementType::TRI3, {1, 2, 5, 1, 5, 4}, 2);
    mixed->set_field("M", mefikit::ElementType::QUAD4, 1, {10.0});
    mixed->set_field("M", mefikit::ElementType::TRI3, 1, {20.0, 30.0});
    const std::vector<double> mixed_quad =
        mefikit::field_values(*mixed, "M", mefikit::ElementType::QUAD4);
    const std::vector<double> mixed_tri =
        mefikit::field_values(*mixed, "M", mefikit::ElementType::TRI3);
    if (mixed_quad.size() != 1 || mixed_tri.size() != 2) {
      throw std::runtime_error("unexpected mixed field lengths");
    }
    check_close(mixed_quad[0], 10.0, "mixed QUAD4 value");
    check_close(mixed_tri[0], 20.0, "mixed TRI3 value 0");
    check_close(mixed_tri[1], 30.0, "mixed TRI3 value 1");

    std::cout << "conservative P0 C++ wrapper: OK\n";
    return 0;
  } catch (const rust::Error &error) {
    std::cerr << "rust::Error: " << error.what() << '\n';
    return 1;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
