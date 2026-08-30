# CHANGELOG

## v0.2.0

### Add

- Python bindings: mapping-style field access via `mesh.fields`
  (`mesh.fields[name]` handle, `ref[sel] = value` writes and `ref[sel]` reads
  accepting selections or per-element-type id dicts, whole-domain reductions
  `min`/`max`/`sum`/`mean`/`var`/`std`/`integral`, `values()` bulk export,
  `keys()`/`items()`, `rename(old, new)`, `del mesh.fields[name]`).
- Python bindings: mapping-style element groups via `mesh.groups`
  (`mesh.groups[name] = selection | {etype: ids}`, `add`/`remove`, `ids()`,
  `rename(old, new)`, `del mesh.groups[name]`).
- Lazy selection API: `UMesh.select(expr)` returns a lightweight
  `SelectionResult` (`ids()`, `len()`, regional reductions accepting field
  expressions, `to_mesh(with_fields=True)` to materialize a sub-mesh).
- Wildcard selector: `mf.sel.all()` / core `ElementSelection::All`, also
  reachable through `None`, `...` or `[:]` wherever a selector is expected.
- Bare field names as strings are accepted wherever a field value or
  expression is expected (`mesh.fields["copy"] = "T"`, `ref[sel] = "T"`,
  `mesh.select(sel).mean("T")`); unknown names raise a Python error.
- Core: `mf::sel::all()` factory and field view/arc types exported through the
  prelude.
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

- Removed the `UMesh`-level group methods (`select_to_group`,
  `add_to_group`, `remove_from_group`, `delete_group`, `rename_group`,
  `set_groups`, `group_names`, `has_group`); use the `mesh.groups` mapping.
- Removed the tuple-key form `mesh.fields["name", sel] = value`; use
  `mesh.fields["name"][sel] = value`.
- `FieldRef.to_dict()` renamed to `values()`.
- `UMesh.select()` now returns a lazy `SelectionResult` instead of a
  materialized `UMesh`; use `.to_mesh()` where the sub-mesh is needed.
- `intersect_2d2d` is now private; use `mesh1.overlay(mesh2,
  OverlayOperation::Imprint)` instead.
- New HEX8 node ordering (connectivity convention), breaking for meshes using
  the previous convention.
- Free functions now take `&UMeshView` instead of `UMeshView` by value.

### Fix

- Referencing an unknown field name in a selection reduction or field
  assignment now raises a Python error instead of aborting the process.
- Row ordering when writing field values through element selections: rows are
  written directly in selection order instead of being re-indexed by local
  element ids.
- out-of-bounds index in `in_polygon_stable` when the closest vertex is the
  first one
- performance issue due to allocations when computing measure/selection
- I/O by creating the data directory when writing

## Release v0.1.2

### Fix

- wheel does not use cpu-native instructions
