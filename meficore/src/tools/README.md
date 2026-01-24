# GridBuilder

A utility to construct regular structured meshes in 1D, 2D, or 3D.

`RegularUMeshBuilder` is a convenience API to generate a [`UMesh`] from axis-aligned
coordinates, forming a structured grid of cells such as line segments (1D), quadrilaterals (2D),
or hexahedrons (3D).

## Overview

The mesh is built by specifying a list of 1 to 3 coordinate axes (x, y, and z), and then taking
their cartesian product to create the node coordinates. The order in which axes are added
determines their role:

- First axis: `x`
- Second axis: `y`
- Third axis (if present): `z`

The grid is then populated by iterating **first along x, then y, then z**, following Fortran-like
(column-major) ordering. This guarantees consistent indexing of nodes and elements when moving
between 1D, 2D, and 3D cases.

## Node and Cell Indexing

The node indices increase along the x-axis first, then y, then z. This means:

- In 1D: `n[i]` is at position `x[i]`
- In 2D: `n[j * nx + i]` is at `(x[i], y[j])`
- In 3D: `n[k * ny * nx + j * nx + i]` is at `(x[i], y[j], z[k])`

Elements are created as:

- 1D: `SEG2` (line segments of 2 nodes)
- 2D: `QUAD4` (quads of 4 nodes)
- 3D: `HEX8` (hexahedra of 8 nodes)

## Example: 2D Mesh

```text
let builder = RegularUMeshBuilder::new()
    .add_axis(vec![0.0, 1.0, 2.0]) // x-axis
    .add_axis(vec![0.0, 1.0]);     // y-axis
let mesh = builder.build();
```

This produces:

Nodes (in order):

```text
(0.0, 0.0)
(1.0, 0.0)
(2.0, 0.0)
(0.0, 1.0)
(1.0, 1.0)
(2.0, 1.0)
```

Elements (QUAD4):

```text
(0, 3, 4, 1)
(1, 4, 5, 2)
```

Layout (indices of nodes):

```text
 3           4           5
 +-----------+-----------+
 |     0     |     1     |
 +-----------+-----------+
 0           1           2
```

## Panics

- Panics if more than 3 axes are added.
- Panics if coordinate vectors are malformed or too short to build elements.
- Panics in `build()` if mesh dimensionality is unsupported (>3D).

## Guarantees

- Coordinate memory layout is flat and matches `ArcArray2<f64>`
- Connectivity arrays are valid and compact
- Node and element ordering is deterministic and predictable

## Use Cases

- Quick generation of test meshes
- Structured meshing for simple domains
- Unit testing and geometry prototyping

## Limitations

- Only supports axis-aligned Cartesian grids (no transformations or distortions)
- Element types are fixed (`SEG2`, `QUAD4`, `HEX8`) — no adaptive mesh refinement
- No boundary tagging or geometry metadata

## See Also

- [`UMesh`] — Main mesh container structure
- [`ElementBlock`] — Block-level data for mesh elements

# Selection

## API Design

```python
eids, mesh = umesh.select(
    mf.sel.bbox([0, 0, 0], [1, 1, 1])
    & mf.sel.types(["QUAD4"])
)
```

---

## 🔎 Geometrical Criterion Implementation Notes

| Method             | Implementation hint                  | Notes                                |
| ------------------ | ------------------------------------ | ------------------------------------ |
| `bbox`/`rectangle` | Use centroid or bounding box overlap | Simple AABB test                     |
| `sphere`/`circle`  | Distance from centroid or nodes      | Use squared distance for performance |

---

## ⚠️ Considerations

- For polyhedral cells or degenerate shapes, centroids might not be enough — but good for most use cases.
