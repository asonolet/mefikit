# CHANGELOG

## [0.3.2](https://github.com/asonolet/mefikit/compare/v0.3.1...v0.3.2) - 2026-09-11

### Add

- reorient
- update README/CHANGELOG and bump version
- add group["toto"].to_mesh and .select("group") = .select(mf.sel.group("group"))
- add dim filtering for fields / selection
- Normal, Nx, Ny, Nz, matmul op, vec/tensor ops
- fieldexpr centroids
- unpolyze PGON, PHED-TET4 and PHED-HEX8
- polyze UMesh
- *(py)* fields, groups and selection new API
- add python groups api
- conservativeP0 2D
- adding inverse distance fast interpolation
- python bindings to transfer tool
- split meshes ([#13](https://github.com/asonolet/mefikit/pull/13))
- take &UMeshView instead of UMeshView by value

### Documentation

- demo of different Transfers
- add intersect2d2d example

### Fix

- med_io
- rm faulty publish = false on mefipy (not published anyway)
- python type-stubs
- broadcasting and document fields ops
- io is optional, no default feature
- remove useless dim passed into "update_field"
- HEX8 new cell order

### Other

- release v0.3.1
- fix release process to have one tag
- upgrade mefikit version to 0.3.0
- HEX/TET node ordering, rebase issues
- lstsq transfer
- upgrade crate deps
- change intersect_2d2d to overlay API
- bump version to 0.1.4
- bump to 0.1.2 to publish new wheels
- bump mefikit-py version
- prepare for release

### Refactor

- remove unused fields from element API
- rename DistanceWeighting None variant to Constant
- replace squared radius r2 with linear radius r in selection API
- remove dead code
- use a main-python tree structure

## [0.3.1](https://github.com/asonolet/mefikit/compare/v0.3.0...v0.3.1) - 2026-09-06

### Add

- reorient
- update README/CHANGELOG and bump version
- add group["toto"].to_mesh and .select("group") = .select(mf.sel.group("group"))
- add dim filtering for fields / selection
- Normal, Nx, Ny, Nz, matmul op, vec/tensor ops
- fieldexpr centroids
- unpolyze PGON, PHED-TET4 and PHED-HEX8
- polyze UMesh
- *(py)* fields, groups and selection new API
- add python groups api
- conservativeP0 2D
- adding inverse distance fast interpolation
- python bindings to transfer tool
- split meshes ([#13](https://github.com/asonolet/mefikit/pull/13))
- take &UMeshView instead of UMeshView by value

### Documentation

- demo of different Transfers
- add intersect2d2d example

### Fix

- rm faulty publish = false on mefipy (not published anyway)
- python type-stubs
- broadcasting and document fields ops
- io is optional, no default feature
- remove useless dim passed into "update_field"
- HEX8 new cell order

### Other

- set up release process
- fix release process to have one tag
- upgrade mefikit version to 0.3.0
- HEX/TET node ordering, rebase issues
- lstsq transfer
- upgrade crate deps
- change intersect_2d2d to overlay API
- bump version to 0.1.4
- bump to 0.1.2 to publish new wheels
- bump mefikit-py version
- prepare for release

### Refactor

- remove unused fields from element API
- rename DistanceWeighting None variant to Constant
- replace squared radius r2 with linear radius r in selection API
- remove dead code
- use a main-python tree structure

## v0.3.0

### Add

- 3D cell intersection: `Polyhedron::convex_intersection_volume` computes the
  intersection volume of two convex polyhedra, robust to translation, scale and
  warped/nearly-colinear faces.
- ConservativeP0 field transfer now supports 3D source/target meshes via the
  intersection volume (2D intersection area remains supported).
- Mesh reorientation tool (`tools::reorient`, `Reorientable` trait, `mf::reorient`
  free function) exposed in Python as `UMesh.reorient()`: rewinds 2D cells
  counter-clockwise, TET4/HEX8 to positive volume and PHED faces outward, while
  preserving blocks, fields and groups (`Arc`-cheap on an owned `UMesh`).
- MED writer handles polygonal/polyhedral cells (PGON / "POG", PHED / "POE"),
  writes the MED geometry ("GEO") attribute on cell groups and round-trips
  fields.

### Fix

- Polyhedron volume computation is translation invariant and scale-aware;
  warped and nearly-colinear faces are managed, and faces are no longer clipped
  by their own plane.
- HEX8 → poly connectivity winding and the last cell of `to_mc`.

### Performance

- Element API refactor removes unused fields (3–20% gains).

## v0.2.1

### Fix

- Export med with "PFL" attribute on polyhedron element types.
- Used nodes do not return `usize::MAX` (especially not in polyhedron case),
  fixes polyhedron merge_nodes / snap

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
