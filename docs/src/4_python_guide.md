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
| bulk export | `mesh.fields.to_dict()` | `{name: {etype: array}}` |
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
