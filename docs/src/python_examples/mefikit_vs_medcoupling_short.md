# mefikit vs. medcoupling

## Goal

Compare two Python APIs for unstructured mesh and field operations:

- **medcoupling**: mature C++ library with broad MED ecosystem.
- **mefikit**: Rust core with Python bindings, focused on concise APIs and performance.

All comparisons use equivalent meshes and cross-check numerical results.

## 1. Reimplemented features

| Feature | mefikit | medcoupling |
|---|---|---|
| Structured mesh | `build_cmesh(*axes)` | `MEDCouplingCMesh` + `buildUnstructured()` |
| MED I/O | `UMesh.read()` / `write()` | `ReadMeshFromFile()` / `write()` |
| Cell measures | `mf.M` | `getMeasureField()` |
| Field expressions | `mf.Field` DSL | `DataArrayDouble` + explicit field setup |
| Selection | `select()` | Explicit field/array operations |
| Descending connectivity | `descend()` | `buildDescendingConnectivity()` |
| Boundary mesh | `boundaries()` | `buildBoundaryMesh()` |
| Merge duplicated nodes | `merge_nodes()` | `mergeNodes()` |
| 2D overlay/imprint | `overlay()` | `Intersect2DMeshes()` |
| Conservative P0/P0 remap | `ConservativeP0` | `MEDCouplingRemapper.prepare("P0P0")` |
| Polyhedral meshes | Supported | Supported |
| Meshless transfers | `ConstantPiecewise`, `MovingLeastSquares` | More manual/lower-level |
| Mixed element types | Supported | Supported |

mefikit also provides a direct bridge to medcoupling:

```python
mc_mesh = mf_mesh.to_mc()
```

## 2. Same mesh, different API

### Structured mesh

```python
import numpy as np
import mefikit as mf
import medcoupling as mc

x = np.linspace(0.0, 1.0, 6)

# mefikit
mf_mesh = mf.build_cmesh(x, x)

# medcoupling
cmesh = mc.MEDCouplingCMesh()
cmesh.setCoords(
    mc.DataArrayDouble(x),
    mc.DataArrayDouble(x),
)
mc_mesh = cmesh.buildUnstructured()
mc_mesh.setMeshDimension(2)
```

Both produce 25 QUAD4 cells and 36 nodes.

### Fields and selection

```python
# mefikit
mf_mesh.fields["Measure"] = mf.M
mf_mesh.fields["T"] = 1.0 + mf.X**2 + 0.5 * mf.Y

hot = mf_mesh.select(mf.Field("T") > 1.5)
mean_hot = hot.mean("T")
```

```python
# medcoupling
measure = mc_mesh.getMeasureField(True)

centers = np.asarray(mc_mesh.computeCellCenterOfMass().toNumPyArray())
values = 1.0 + centers[:, 0] ** 2 + 0.5 * centers[:, 1]

field = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
field.setArray(mc.DataArrayDouble(values))
field.setMesh(mc_mesh)
field.setNature(mc.IntensiveConservation)
```

**API difference:** mefikit exposes symbolic expressions and selection directly on meshes. medcoupling uses explicit arrays and field objects.

### Descending connectivity

```python
# mefikit
faces = mesh.descend()
```

```python
# medcoupling
faces = mesh.buildDescendingConnectivity()[0]
```

### Merge duplicated nodes

```python
# mefikit
merged = mesh.merge_nodes()
```

```python
# medcoupling
mesh.mergeNodes(1e-12)
```

Both produce the same used-node count. mefikit rewires connectivity without compacting the coordinate array; medcoupling compacts the nodes.

### Conservative P0/P0 remap

```python
# mefikit
op = mf.transfer.ConservativeP0(source, target)
op.apply_update(source, "T", target, "T", def_val=0.0)
```

```python
# medcoupling
remap = mc.MEDCouplingRemapper()
remap.prepare(source_mc, target_mc, "P0P0")
result = remap.transferField(source_field, 0.0)
```

Both use a prepare/apply workflow. The transferred values match to numerical precision.

## 3. Performance

Measurements are medians of repeated runs on the same geometries. Values are from the benchmark notebook; they are machine-dependent.

| Operation | Mesh | mefikit (ms) | medcoupling (ms) | Ratio |
|---|---|---:|---:|---:|
| Remap prepare | 2D QUAD4 | 27.045 | 24.817 | 0.9× |
| Remap transfer | 2D QUAD4 | 0.181 | 3.905 | 21.5× |
| Remap prepare | 3D HEX8 | 162.564 | 692.317 | 4.3× |
| Remap transfer | 3D HEX8 | 0.094 | 1.235 | 13.2× |
| Remap prepare | 3D polyhedral | 356.550 | 8746.085 | 24.5× |
| Remap transfer | 3D polyhedral | 0.099 | 2.171 | 22.0× |
| Merge nodes | 3D HEX8 | 0.538 | 4.019 | 7.5× |
| Descend | 3D HEX8 | 56.470 | 103.696 | 1.8× |
| Overlay | 2D QUAD4 | 2.071 | 68.392 | 33.0× |
| Crack | 3D HEX8 | 235.632 | 1624.416 | 6.9× |

**Main result:** mefikit is substantially faster for most tested operations, especially polyhedral remap preparation and 2D overlay.

The 2D remap preparation is the exception: medcoupling is slightly faster in this case.

### Polyhedral remap scaling

| Cells | mefikit prepare | medcoupling prepare | medcoupling / mefikit |
|---:|---:|---:|---:|
| 100 | 2.59 ms | 153.5 ms | 59× |
| 200 | 7.14 ms | 568.9 ms | 80× |
| 400 | 23.26 ms | 2347.4 ms | 101× |

The benchmark shows a growing advantage for mefikit as the polyhedral mesh size increases.

## 4. Correctness

The tested operations produced equivalent results:

- Structured meshes: same cell and node counts.
- Measures: same total area/volume.
- Field expressions: maximum difference near machine precision.
- P0/P0 remap: transferred fields match within tolerance.
- Merge nodes: same resulting node count.
- Descending connectivity: same face count.
- Overlay: unit area preserved.
- Crack: same node count.

## Feature comparison

| Feature | mefikit | medcoupling |
| --- | --- | --- |
| MED I/O | Supported | Supported |
| CGNS I/O | Supported | Not supported |
| VTKHDF I/O | Supported | Not supported |
| Structured meshes | Supported | Supported |
| Polyhedral meshes | Supported | Supported |
| Field expressions | Symbolic `Field` API | Explicit arrays and field objects |
| Field selection | High-level `select()` API | More explicit field/array operations |
| Conservative P0/P0 remapping | Supported | Supported |
| Meshless interpolation | MLS and distance-based operators | No equivalent high-level API |
| Distance kernels | Gaussian, inverse distance, inverse squared distance, etc. | Not available through an equivalent API |
| Descending connectivity | Supported | Supported |
| Boundary extraction | Supported | Supported |
| Node merging | Supported | Supported |
| 2D overlay | Supported | Supported |
| 3D conformization | Not supported | Supported |
| 2D manifold overlay | Not supported | Supported |
| Quadratic polygon overlay | Not supported | Supported |
| Slicing | Not supported | Supported |
| Mixed element types | Supported | Supported |

## Data-model distinction

Mefikit unifies mesh and MED-file concepts in `UMesh`. A `UMesh` can contain:

- mesh levels;
- groups;
- fields;
- MED metadata;
- topology and geometry.

The core medcoupling object, `MEDCouplingUMesh`, represents a single mesh level. MED-file organization is handled separately.
This makes the mefikit model more convenient when working with groups, fields, and multiple levels as one object.


```python
mesh = mf.UMesh.read("mesh.med")

mesh.groups
mesh.fields
```

The bridge to medcoupling remains available:

```python
mesh_mc = mesh.to_mc()
```

## Meshless interpolation

mefikit provides high-level meshless interpolation operators, including:

```python
op = mf.transfer.MovingLeastSquares(
    source,
    target,
    kernel="gaussian",
)
```

Supported weighting approaches include:

* MLS;
* distance weighting;
* inverse-distance weighting;
* inverse-squared-distance weighting;
* Gaussian and other distance kernels.

This is a clear mefikit capability that is not represented by an equivalent high-level medcoupling API.

## Conclusions

- API: mefikit provides a shorter, higher-level API for common mesh and field operations.
- Performance: mefikit was faster in most tested cases, especially polyhedral remapping.
- I/O: mefikit supports MED, CGNS, and VTKHDF workflows.
- Interpolation: mefikit provides meshless interpolation with MLS and configurable distance kernels.
- Data model: `UMesh` unifies mesh levels, groups, fields, and MED metadata.
- Geometry: medcoupling remains stronger for 3D conformization, 2D manifold overlay, quadratic polygon overlay, and slicing.
- Maturity: medcoupling remains the broader and more established ecosystem.
- Positioning: mefikit is particularly attractive for new projects, polyhedral
  meshes, and performance-sensitive remapping workflows. Not a drop-in
  replacement for now.
