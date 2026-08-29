# Python guide

This page is a compact reference for the Python API. The notebooks under
[Python Examples](./python_examples/SUMMARY.md) show the same features in
context; here they are gathered as tables.

Meshes are `UMesh` objects. Fields and element groups live in two dict-like
mappings on the mesh, and selections are lazy views that only evaluate when
queried.

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

Field expressions (notably `mf.M` for the on-the-fly measure) can be evaluated
without a stored field:

- `mesh.eval(expr, dim=None)` → `{etype: array}`, e.g. `mf.M` or `mf.Field("T") * 2`
- `mesh.eval_update(name, expr, dim=None)` stores the result in-place
- `mesh.measure()` → per-type measures; `mesh.measure_update()` materializes a
  `"Measure"` field (usually unnecessary, prefer `mf.M`)
