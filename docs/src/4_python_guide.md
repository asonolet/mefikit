# Python guide

This page is a compact reference for the Python API. The notebooks grouped at
the end of the book show the same features in context; here they are gathered
as tables.

Meshes are `UMesh` objects. Fields and element groups live in two dict-like
mappings on the mesh, and selections are lazy views that only evaluate when
queried.

Some insights on how to use the current Python library:

- Use autocompletion with the tool you like, type hints are provided! A missing
  type hint is a bug, please report it.
- Only high level whole mesh operations are supported through python. For finer
  grain ops either reach me or implement it in rust consuming the `mefikit` rust
  crate.
- Use lazy expressions wherever possible, they are fast, reusable, expressive
  and less error prone than manual indexing. They are inspired from `polars`, a
  DataFrame python-rust lib.
- Visualize with `pyvista`. It lacks type hints and the syntax may not feel
  familiar but the overall experience is really better than anything else.

## The fields mapping

`mesh.fields` behaves like a dict keyed by field name. Each entry returns a
`FieldRef`, a handle to read, reduce, or write the stored values.

| Operation | Call | Notes |
|---|---|---|
| list names | `mesh.fields.keys()` / `items()` / `len()` / `name in mesh.fields` | sorted, deterministic |
| get a handle | `ref = mesh.fields["T"]` | `KeyError` if missing |
| create / replace | `mesh.fields["T"] = value` | see accepted values below |
| delete | `del mesh.fields["T"]` | removes every instance |
| rename | `mesh.fields.rename("T", "T2")` | `KeyError` / `ValueError` on bad names |
| bulk export | `mesh.fields.to_dict()` | `{name: {etype: array}}`; `mesh.fields.values()` returns the `FieldRef` list |
| per-etype values | `ref.values()` | `{etype: array}` |
| single array | `ref.numpy()` | one array when the mesh has one element type |
| metadata | `ref.shape`, `ref.dimension()`, `len(ref)` | component shape, mesh dimension, element count |

Accepted values for creation and writes:

- a `float` — broadcast to every row
- an `np.ndarray` — full column or per-block rows
- a dict `{etype: array}` — per element type
- an expression: `mf.Field("T") * 2` or another field object
- a string naming an existing field (e.g. `"T"`) — copies it

Reductions over all elements carrying the field:
`min()`, `max()`, `sum()`, `mean()`, `var(ddof=0)`, `std(ddof=0)`,
`integral()` (measure-weighted).

### Partial reads and writes

`ref[selector]` gathers the selected rows as `{etype: array}`; assigning
through `ref[selector] = value` writes them. Selectors are:

- wildcards: `...`, `:` (full slice), or `None`
- an ids dict: `{"QUAD4": [0, 3]}`
- any selection expression: `mf.sel.rect(...)`, `mf.Field("T") > 1.0`, ...

## The groups mapping

`mesh.groups` behaves like a dict keyed by group name. Each entry is a
`GroupRef`.

| Operation | Call |
|---|---|
| create / replace | `mesh.groups["wall"] = sel_expr` or `= {"QUAD4": [0, 1]}` |
| grow / shrink | `ref.add(source)` / `ref.remove(source)` |
| element ids | `ref.ids()` → `{etype: uint64 array}`, `len(ref)` |
| rename | `mesh.groups.rename("old", "new")` |
| delete | `del mesh.groups["wall"]` |

Groups feed back into selections through
`mf.sel.group("wall")` and `mf.sel.exclude_group("wall")`.

## Selections

Selection factories live in the `mf.sel` module:

| Factory | Elements matched by |
|---|---|
| `bbox(min, max)` / `sphere(center, r)` | centroid position (3D) |
| `rect(min, max)` / `circle(center, r)` | centroid position (2D); bounds are min-inclusive / max-exclusive |
| `nbbox` / `nsphere` / `nrect` / `ncircle(..., all)` | node positions, with all/any semantics |
| `ids({"ETYPE": [...]})` | explicit element ids |
| `types(["QUAD4", ...])` | element types |
| `group(name)` / `exclude_group(name)` | membership in a named group |
| `all()` — also `None`, `...`, `[:]` where a selector is expected | everything |

Selections compose with `&`, `|`, `^`, `-`, `~`. Field thresholds produce
selections too: `mf.Field("T") > 1.0`.

Two families of spatial selectors are available (showcased in the
[selection](./python_examples/selection.md) notebook):

- the `n*` variants (`nbbox`, `nrect`, `nsphere`, `ncircle`, `nids`) match
  **node** positions and take an `all=` flag (all vs. any node of the element
  must match);
- `bbox`, `rect`, `sphere`, `circle`, `ids` match **element centroids**.

### Lazy results

`mesh.select(expr)` does not build a mesh; it returns a lightweight
`SelectionResult` that re-evaluates on every call:

- `result.ids()` → `{etype: array}`
- `len(result)`
- reductions with any field expression: `min/max/sum/mean(expr)`,
  `var/std(expr, ddof=0)`, `integral(expr)`
- `result.to_mesh(with_fields=True)` materializes a sub-mesh when needed

```python
hot = mesh.select(mf.Field("energy") > 1e6)
print(hot.mean("energy"))
submesh = hot.to_mesh()
```

## Mesh modification

Most topological tools return a new `UMesh`; the `*_update` variants operate
in-place and return a new mesh only when the result displaced elements
(otherwise `None`).

| Operation | Call |
|---|---|
| build structured grid (SEG2/QUAD4/HEX8) | `mf.build_cmesh(*axes)` |
| descending/finer connectivity | `mesh.descend(src_dim, target_dim)` / `descend_update(...)` |
| boundaries of a dimension | `mesh.boundaries(src_dim, target_dim)` / `boundaries_update(...)` |
| connected parts | `mesh.connected_components(src_dim, link_dim, with_fields)` |
| crack / snap / merge nodes | `mesh.crack(cut)`, `mesh.snap(ref, eps)`, `mesh.merge_nodes(eps)` |
| extrude | `mesh.extrude(along)`, `extrude_parallel(...)`, `extrude_curv(...)` |
| split / polygonize | `mesh.split()`, `mesh.polyze()` / `unpolyze()` |
| boolean overlay | `mesh.overlay(mesh2, operation=None)` |
| surface imprint in 3D | `mesh.overlay_surfaces(mesh2, tol=1e-9)` → `mf.SurfaceOverlay` |
| join meshes as-is | `mf.aggregate(meshes)`, `mf.concat(a, b)` |
| stitch volumes at shared boundaries | `mf.stitch(meshes, tol=1e-9)` → `UMesh` |

`mf.stitch` makes two or more volume meshes (`TET4`, `HEX8` or `PHED`) conformal
wherever their boundaries coincide, so that they can be used as a single mesh:

```python
import mefikit as mf

plate = mf.build_cmesh([0, 2], [0, 2], [0, 1])  # one HEX8
block = mf.build_cmesh([0, 1, 2], [0, 1, 2], [1, 2])  # four HEX8
stitched = mf.stitch([plate, block])  # one PHED mesh, 5 cells

# The interface on z = 1 is now shared: every face there belongs to two cells.
coords = stitched.coords()
```

The result is a single polyhedral mesh: interface nodes exist once, families are
relabeled per input mesh, and fields and groups are dropped. Coincident interfaces
must be piecewise planar within `tol`. This is an *imprint only* operation:
overlapping volumes are not detected, so the output may contain overlapping cells.

## Geometric transforms

Pure coordinate transformations, each returning a **new** `UMesh` (the source
is left untouched):

| Operation | Call |
|---|---|
| translate by a vector | `mesh.translate([x, y, z])` |
| scale per axis / uniformly | `mesh.scale([sx, sy, sz])` / `mesh.scale_uniform(s)` |
| rotate around an axis through the origin | `mesh.rotate(axis, angle)` |
| rotate around an axis through a point | `mesh.rotate_about(center, axis, angle)` |
| mirror across a plane through the origin | `mesh.mirror(normal)` |
| mirror across a plane through a point | `mesh.mirror_about(point, normal)` |
| apply an arbitrary affine transform | `mesh.transform(mf.Transform(matrix))` |
| repeat a shape with a step transform | `mesh.duplicate(step, n)` |

Angles are in radian. `mf.Transform` also offers the `identity`, `translation`,
`scaling`, `rotation`, `reflection` and `from_matrix` constructors; `transform`
also accepts a plain 4x4 numpy array.

Field expressions (notably `mf.M` for the on-the-fly measure) can be evaluated
without a stored field:

- `mesh.eval(expr, dim=None)` → `{etype: array}`, e.g. `mf.M` or `mf.Field("T") * 2`
- `mesh.eval_update(name, expr, dim=None)` stores the result in-place
- `mesh.measure()` → per-type measures; `mesh.measure_update()` materializes a
  `"Measure"` field (usually unnecessary, prefer `mf.M`)

## Input / output

`UMesh.read` and `UMesh.write` build a mesh from — or dump it to — a file, the
format being selected by the file extension (`.json`, `.yaml`, `.vtk`, `.vtu`,
`.vtkhdf`, `.cgns`, `.med`):

| Operation | Call |
|---|---|
| read a mesh from disk | `mf.UMesh.read(path)` |
| write a mesh to disk | `mesh.write(path)` |
| in-memory PyVista object | `mesh.to_pyvista(dim=None, with_fields=True)` |
| in-memory meshio object | `mesh.to_meshio()` |
| in-memory medcoupling twin | `mesh.to_mc(lev=None)` / `mf.UMesh.from_mc(mc_mesh)` |

`to_pyvista` copies the mesh (and, by default, its fields) into a
`pyvista.UnstructuredGrid`; `to_meshio` produces a `meshio.Mesh`; the
medcoupling pair converts to / from a `MEDCouplingUMesh`. See the
[Input/Output](./python_examples/input_output.md) notebook and the `.med`
round-trip in [mefikit vs. medcoupling](./python_examples/compare_medcoupling.md).

## Field transfers

Remapping a field from a **source** mesh onto a **target** mesh is done with
the `mf.transfer` operators. All of them share the same prepare / apply split:
construction builds the interpolation coefficients once, and applying the
operator to any field is then a single fast sparse product:

| Operator | Kind | Conserves | Signature |
|---|---|---|---|
| `ConstantPiecewise` | cell-based, point location | no | `(src, tgt, def_val=0.0)` |
| `InverseDistance` | meshless, `k`-nearest | no | `(src, tgt, k=4, exponent=2.0, def_val=0.0)` |
| `MovingLeastSquares` | meshless, local fit | no | `(src, tgt, k=10, weighting=DistanceWeighting.Constant(), def_val=0.0)` |
| `ConservativeP0` | volumetric overlap | **yes** | `(src, tgt, def_val=0.0)` |

Weighting kernels for `MovingLeastSquares` come from
`mf.transfer.DistanceWeighting`: `Constant()`, `InverseDistance(exponent)`
and `Gaussian()`.

Once built, every operator is used the same way:

- `tr(expr, extensive=False)` → the transferred field, as a `Field` on `tgt`
- `tr.eval(expr, extensive=False)` → `{etype: array}`
- `tr.apply_update(src, name, tgt, tgt_field_name=None, def_val=0.0, extensive=False)`
  writes the result into a target field in place

`extensive=True` treats the field as extensive (mass, energy): `ConservativeP0`
then keeps the raw measure-weighted sum, whereas intensive fields are
normalized by the target cell measure. Cells uncovered by any source cell keep
the transfer `def_val`. The [Field transfers](./python_examples/transfers.md)
notebook walks through each operator; the timing and correctness comparison
with medcoupling is at the end of [mefikit vs.
medcoupling](./python_examples/compare_medcoupling.md).
