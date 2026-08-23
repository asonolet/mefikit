# CHANGELOG

## Unreleased

### Add

- Surface mesh overlay in 3D space: `Overlayable::overlay_surfaces` /
  `overlay_surfaces(skin1, skin2, tol)` imprints two 2D meshes embedded in 3D
  onto each other wherever they coincide. Faces are clustered into maximal
  coplanar patches, patches of the two surfaces are paired (coplanarity, area
  and bounding box agreement within `tol`), and each pair is processed with the
  2D overlay machinery on a fitted planar frame. Both refined surfaces share
  the same coordinates array so intersection nodes exist once; families and
  fields propagate to the produced faces; untouched faces are copied verbatim.
  Partial overlaps between coplanar patches are rejected through
  `SurfaceOverlayError`.

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
