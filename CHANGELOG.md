# CHANGELOG

## Unreleased

## v0.1.5

### Add

- 2D boolean mesh overlay: `UMesh::overlay` / `OverlayOperation` (IMPRINT,
  UNION, INTERSECTION, DIFFERENCE, SYMMETRIC_DIFFERENCE), exposed in Python as
  `UMesh.overlay` / `OverlayOperation`.
- Field transfer module (`tools::transfer`): `ConstantPiecewiseTransfer`,
  `MovingLeastSquaresTransfer`, `InverseDistanceTransfer` and
  `ConservativeP0Transfer` (2D), plus the `DistanceWeighting` schemes. Exposed
  in Python as `mf.transfer.ConstantPiecewise`, `MovingLeastSquares`,
  `InverseDistance`, `ConservativeP0` and `DistanceWeighting`, with the
  geometry precompute separated from the reusable `apply_update` call.
- Cell splitting tool (`tools::split_cells`), exposed in Python as
  `UMesh.split()` (SEG2, TRI3, QUAD4, TET4, HEX8).
- HDF5/VTKHDF reader and writer (`io`), driven by the file extension in
  `mf::read` / `mf::write` and `UMesh.read` / `UMesh.write` (`.vtkhdf`,
  `.h5`, `.hdf5`).
- New `geometry` module (segment, polygon, polyhedron, region, convexity) and
  reworked `element_traits` (cut, measures, centroids, point-in tests).
- New `tools::centroids` (Centroidable) and `tools::spatial_index`
  (SpatiallyIndexable, BVH).
- Python bindings: top-level `Field`, `OverlayOperation` and `transfer`
  exports, `UMesh.set_field`, `UMesh.num_elements`, and PGON support in the
  `to_pyvista()` conversion.

### Change

- `intersect_2d2d` is now private; use `mesh1.overlay(mesh2,
  OverlayOperation::Imprint)` instead.
- New HEX8 node ordering (connectivity convention), breaking for meshes using
  the previous convention.
- Free functions now take `&UMeshView` instead of `UMeshView` by value.

### Fix

- out-of-bounds index in `in_polygon_stable` when the closest vertex is the
  first one
- performance issue due to allocations when computing measure/selection
- I/O by creating the data directory when writing

## Release v0.1.2

### Fix

- wheel does not use cpu-native instructions
