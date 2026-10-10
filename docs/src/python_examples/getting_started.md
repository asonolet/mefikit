# Getting started

`mefikit` builds, reads, writes and transforms **unstructured meshes** and the
**fields** living on them, from Python. This notebook is the shortest path from
zero to a mesh with a field, a selection and a figure — everything below runs
as-is.

The whole book is organized so that you can copy any code cell into your own
script: the [Python guide](../4_python_guide.md) gathers the API in reference
tables, and the examples build on top of it.

## Install

Mefikit is released on [PyPI](https://pypi.org/project/mefikit). The `[all]`
extra pulls the optional I/O and plotting dependencies (`medcoupling`,
`meshio`, `pyvista`):

```shell
pip install "mefikit[all]"      # or: uv add "mefikit[all]"
```

A plain `pip install mefikit` also works if you only need the core — numpy is
the sole mandatory dependency — and you can add the extras later with
`pip install mefikit[io]`. Building from source (for the Rust API) is covered
in the [README](https://github.com/asonolet/mefikit).

## A first mesh

A `UMesh` is an unstructured mesh: **one global node-coordinates array** plus a
set of **element blocks**, one per element type. `mf.build_cmesh(*axes)`
builds a structured cartesian grid in a single call — give it the node
coordinates along each axis and it fills the cells for you.


```python
import mefikit as mf

mesh = mf.build_cmesh(range(2), range(5), range(4))
print("coords :", mesh.coords().shape)
print("blocks :", {et: conn.shape for et, conn in mesh.blocks().items()})
```

    coords : (40, 3)
    blocks : {'HEX8': (12, 8)}


## Fields in one line

Data is attached to a mesh through the dict-like `mesh.fields` mapping. You
assign a **field expression** and mefikit evaluates it lazily, on demand —
the expression is a tree, not a stored array. Geometry primitives exist ready
to use: `mf.X`/`mf.Y`/`mf.Z` (node or cell centroids), `mf.M` (measure:
length / area / volume), `mf.N` (normals).


```python
import numpy as np

mesh = mf.build_cmesh(
    np.linspace(0.0, 1.0, 30), np.linspace(0.0, 1.0, 20), np.linspace(0.0, 1.0, 10)
)
mesh.fields["T"] = 1.0 + 2.0 * mf.X  # a linear temperature profile in x

ref = mesh.fields["T"]
print("min / mean / max :", ref.min(), ref.mean(), ref.max())
print("values           :", {et: a.shape for et, a in ref.values().items()})
```

    min / mean / max : 1.0344827586206897 2.0 2.9655172413793105
    values           : {'HEX8': (4959,)}


## Selecting elements

`mesh.select(expr)` returns a **lazy view**: nothing is computed until you
reduce it or materialize it. Spatial filters from `mf.sel` combine with field
thresholds; `sum`/`mean`/... reduce over the selection, and `.to_mesh()`
builds a real sub-mesh when you need one.


```python
hot = mesh.select(mf.Field("T") > 1.25)
print("hot cells   :", len(hot))
print("hot volume  :", hot.sum(mf.M))
print("mean pos    :", hot.mean(mf.C))

bubble = mesh.select(mf.sel.sphere([0.5, 0.5, 0.5], 0.3))
print("bubble cells:", len(bubble))
bubble_mesh = bubble.to_mesh()  # materialize a real UMesh when needed
```

    hot cells   : 4275
    hot volume  : 0.8620689655172395
    mean pos    : [0.56896552 0.5        0.5       ]
    bubble cells: 559


## Plotting

`UMesh` converts to [PyVista](https://docs.pyvista.org) in one call; fields
ride along, so volume rendering and slicing work with no glue code.


```python
mesh.to_pyvista().plot(scalars="T", show_edges=True)
```


    Widget(value='<iframe src="http://localhost:42367/index.html?ui=P_0x7236419e7a10_0&reconnect=auto" class="pyvi…


## The mental model in one glance

Three rules cover most of the API:

- **A mesh is coordinates + blocks.** Node coordinates are one array; each
  block stores the connectivity of one element type (`HEX8`, `QUAD4`, ...).
  The [UMesh basics](./umesh_basics.md) notebook explores this in depth.
- **Fields live per element type, and expressions are lazy.** `mesh.fields["T"]`
  must cover *all* elements of its type. Assignments accept any expression
  (`mf.X`, `mf.Field("T") * 2`, ...) which is materialized only when you ask:
  a reduction, a `select(...)`, or `mesh.eval(...)`. `mf.M` computes the
  measure on the fly instead of storing it. See the
  [Fields](./fields.md) notebook.
- **Selections are lazy views.** `select(...)` never builds a mesh; reductions
  re-evaluate on each call and `.to_mesh()` materializes when you truly need a
  sub-mesh. See the [Selection](./selection.md) notebook.

Mechanical detail that bites: most `*_update` methods operate **in place** and
return a *new* mesh **only** when the operation displaced elements — otherwise
they return `None`.

## Where to go next

| Goal | Page |
|---|---|
| understand what a `UMesh` contains | [UMesh basics](./umesh_basics.md) |
| field expressions, `eval`, reductions | [Fields](./fields.md) |
| spatial filters and groups | [Selection](./selection.md) |
| move fields between meshes | [Field transfers](./transfers.md) |
| topology tools (descend, crack, ...) | [Topological tools](./topological_tools.md) |
| geometry tools (overlay, merge, snap, ...) | [Geometric tools](./geometric_tools.md) |
| build meshes by sweeping / transforms | [Extrusions](./extrusions.md), [Geometric transforms](./geometric_transforms.md) |
| read / write `.med` and interchange | [Input/Output](./input_output.md) |
| a full use case end to end | [Bubbles](./example_bubbles.md) |
| how mefikit compares to a mature library | [mefikit vs. medcoupling](./compare_medcoupling.md) |

The [Python guide](../4_python_guide.md) is the condensed API reference behind
all of these examples.
