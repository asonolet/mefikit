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


# SelectionBuilder

## 🧱 Rust API Design

Extend your `SelectionBuilder` with new methods like:

```rust
impl<'a> SelectionBuilder<'a> {
    pub fn in_bounding_box(mut self, min: [f64; 3], max: [f64; 3]) -> Self { ... }

    pub fn near_point(mut self, point: [f64; 3], tol: f64) -> Self { ... }

    pub fn along_plane(mut self, point: [f64; 3], normal: [f64; 3], tol: f64) -> Self { ... }

    pub fn along_segment(mut self, a: [f64; 3], b: [f64; 3], tol: f64) -> Self { ... }
}
```

Each method would push a closure to a list of selection criteria, evaluated per cell (e.g., using the **cell centroid** as a proxy). Example:

```rust
self.criteria.push(Box::new(move |mesh, cell_id| {
    let centroid = mesh.compute_centroid(cell_id);
    centroid[0] >= min[0] && centroid[0] <= max[0] &&
    centroid[1] >= min[1] && centroid[1] <= max[1] &&
    centroid[2] >= min[2] && centroid[2] <= max[2]
}))
```

You can later optimize with bounding boxes of full elements if needed.

---

## 🐍 Python API Design

Expose these methods through your `PySelectionBuilder`:

```python
builder = umesh.select()
    .in_bounding_box([0, 0, 0], [1, 1, 1])
    .along_plane([0, 0, 0], [0, 1, 0], tol=1e-6)
    .into_view()
```

### Optional: Named Criteria for Expressiveness

```python
builder = umesh.select()
    .near_point([1.0, 0.0, 0.0], tol=0.01)
    .filter_by_group("fluid")
```

---

## 🔎 Geometrical Criterion Implementation Notes

| Method            | Implementation hint                  | Notes                                    |
| ----------------- | ------------------------------------ | ---------------------------------------- |
| `in_bounding_box` | Use centroid or bounding box overlap | Simple AABB test                         |
| `near_point`      | Distance from centroid or nodes      | Use squared distance for performance     |
| `along_plane`     | Point-to-plane distance              | Dot product + abs comparison             |
| `along_segment`   | Project point, clamp, distance       | More complex but useful in CAD/CFD tools |

---

## ⚠️ Considerations

* For polyhedral cells or degenerate shapes, centroids might not be enough — but good for most use cases.
* You may want a toggle to apply selection on **cells** vs. **nodes** in the future.
* All geometric methods should take an optional `tol: f64` (with a default in Python).

---

## ✅ Summary

Adding `in_bounding_box`, `near_point`, `along_plane`, and `along_segment` to your `SelectionBuilder`:

* Is idiomatic and composable in Rust
* Maps cleanly to Python with `pyo3`
* Provides powerful filtering UX, especially for preprocessing and subdomain analysis
* Helps you stand out vs. more rigid tools like MEDCoupling

