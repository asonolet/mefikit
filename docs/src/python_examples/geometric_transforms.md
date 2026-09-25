# Geometric transforms


```python
import numpy as np

import mefikit as mf

coords = np.array(
    [
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)
mesh = mf.UMesh(coords)
mesh.add_regular_block("QUAD4", np.array([[0, 1, 3, 2]], dtype=np.uint))
```

**All angles are given in radians.** A `Transform` is an affine transformation
represented by a 4x4 homogeneous matrix (last row `0 0 0 1`). `UMesh` objects
are transformed out-of-place: each transform returns a new mesh and leaves the
source unchanged.

## Out-of-place transforms


```python
translated = mesh.translate([10.0, 0.0])
assert np.isclose(translated.coords()[1, 0], 11.0)

rotated = mesh.rotate([0.0, 0.0, 1.0], np.pi / 2)
assert np.allclose(rotated.coords()[1], [0.0, 1.0], atol=1e-9)

scaled = mesh.scale([2.0, 3.0])
assert np.allclose(scaled.coords()[3], [2.0, 3.0])

uniform = mesh.scale_uniform(4.0)
assert np.isclose(uniform.coords()[3, 0], 4.0)

mirrored = mesh.mirror([1.0, 0.0, 0.0])
assert np.isclose(mirrored.coords()[1, 0], -1.0)
```

The input mesh is never modified.

## Unidirectional (left-to-right) composition

`Transform` supports composition in both conventions:

- `a @ b` (matrix product) applies `b` first;
- `a.then(b)` applies `a`, then `b`.


```python
tr = mf.Transform.translation([1.0, 0.0]).then(
    mf.Transform.rotation([0.0, 0.0, 1.0], np.pi / 2)
)
out = mesh.transform(tr)
assert np.allclose(out.coords()[1], [0.0, 2.0], atol=1e-9)

matrix = mf.Transform.translation([1.0, 2.0, 3.0]) @ mf.Transform.scaling([2.0])
assert np.allclose(
    matrix.matrix(),
    mf.Transform.translation([1.0, 2.0, 3.0]).matrix()
    @ mf.Transform.scaling([2.0]).matrix(),
)
```

Rather than a `Transform`, `mesh.transform(...)` also accepts a raw 4x4
`numpy` array, and `Transform.from_matrix(...)` wraps one.


```python
mat = np.eye(4)
mat[0, 3] = 7.0
assert np.isclose(mesh.transform(mat).coords()[0, 0], 7.0)
```

Dimensional consistency is enforced: trying to move a 2D (or 1D) mesh out of
its plane (or line) raises a `ValueError`.


```python
try:
    mesh.translate([0.0, 0.0, 0.5])
except ValueError as err:
    print(err)
```

    This transform moves the mesh out of its plane/line: rows beyond the space dimension must leave the extra coordinates unchanged.


## Duplicating a mesh

`duplicate(step, n)` returns `n` copies, each transformed by the powers
`step`, `step @ step`, ... of `step`.


```python
column = mesh.duplicate(mf.Transform.translation([0.0, 3.0, 0.0]), 3)
assert column.coords().shape[0] == 12
assert np.allclose(
    np.sort(np.unique(column.coords()[:, 1])), [0.0, 1.0, 3.0, 4.0, 6.0, 7.0]
)
```

Arbitrary arrangements can be built with the module-level `aggregate` /
`concat` functions, which concatenate meshes while preserving blocks, fields,
families and groups (element ids are relabelled so that the resulting mesh
stays valid).


```python
line = mf.concat(mesh, mesh.translate([5.0, 0.0]))
three = mf.aggregate([mesh, mesh.translate([5.0, 0.0]), mesh.translate([10.0, 0.0])])
assert line.coords().shape[0] == 8
assert three.coords().shape[0] == 12
```

Transforms preserve the mesh metadata: blocks, fields, families and groups
survive untouched.

## Reconstructing coordinates

For non-affine coordinate changes, use `UMesh.from_mesh`. It accepts a
same-shaped coordinate array and preserves the selected source connectivity,
fields, families and groups. Omitting `coords` reuses the source coordinates.


```python
warped = mf.UMesh.from_mesh(
    mesh,
    np.column_stack((mesh.coords()[:, 0] ** 2, mesh.coords()[:, 1])),
)
assert np.allclose(warped.coords()[:, 0], mesh.coords()[:, 0] ** 2)
assert np.allclose(mesh.coords(), coords)
```

`from_mesh` can select source element blocks by topological dimension or by
element type. The two selectors are mutually exclusive; the source mesh is
never changed.


```python
surfaces = mf.UMesh.from_mesh(mesh, dim=2)
quads = mf.UMesh.from_mesh(mesh, element_types=["QUAD4"])
```

Coordinate arrays are structurally validated when a mesh is reconstructed;
geometric quality checks are not performed.
