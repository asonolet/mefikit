# Field transfers

*Figures use PyVista; verbose plotting boilerplate is omitted.*

This notebook shows how to remap fields between meshes with the
`mf.transfer` operators: interpolation, extrapolation and conservative
remapping, all sharing the same prepare / apply split.

The setup below is self-contained: a log-spaced box mesh carrying a `"Measure"`
field (cell volumes) and an analytic expression `"4 * M2"`, plus a couple of
spatial selections used to crop the source. If anything looks unfamiliar, the
[Fields](./fields.md) and
[Selection](./selection.md) notebooks cover their meaning.


```python
import numpy as np

import mefikit as mf

x = np.logspace(-5, 0.0, 1000)
mesh2 = mf.build_cmesh(x, x)

mesh2.fields["Measure"] = mf.M

m = mf.Field("Measure")
m2 = mf.Field("4 * M2")
mesh2.fields["4 * M2"] = m * m * 4.0

r = mf.sel.rect([0.25, 0.25], [0.7, 0.7])
c = mf.sel.circle([0.875, 0.875], 0.05)
```

## Transferring fields

A transfer maps a field from a **source** mesh onto a **target** mesh. Every
`mf.transfer` operator reduces to the same operation — each target cell is a
weighted sum of source cells:

    t_j = sum_i w_ji s_i

The weights are computed **once**, at construction time (the *prepare* step);
applying the operator to any field is then a single, fast matrix-vector
product. Reuse one operator for many fields, as long as the two meshes do not
change.

What differs between the four operators is only *how the weights are
obtained*:

| Operator | Kind | Sampling / support | Conserves the integral | Typical use |
|---|---|---|---|---|
| `ConstantPiecewise` | cell-based | target cell center located in a source cell (centroid) | no | faithful piecewise-constant copy, coarsening |
| `InverseDistance` | meshless | `k` nearest source cell centers, weights `1 / r**exponent` | no | cheap local fallback, unrelated point clouds |
| `MovingLeastSquares` | meshless | `k` nearest source cell centers, local fit with a distance kernel | no | smooth fields, trend reconstruction |
| `ConservativeP0` | overlap | measure-weighted intersection (same-dim and downcast) | **yes** | keep the target integral exact, ALE / coupling, volume → surface / line |

Some shared rules:

- **Interpolation methods** (`ConstantPiecewise`, `InverseDistance`,
  `MovingLeastSquares`) combine a few source cells per target cell and never
  conserve the integral.
- **`ConservativeP0`** weights each contribution by the intersection measure.
  For **intensive** fields (temperature, density) it normalizes by the target
  cell measure afterwards — enough to behave like an interpolation — while
  **extensive** fields (mass, energy) keep the raw weighted sum. Pass
  `extensive=True` to treat a field as extensive.
- Target cells covered by **no** source cell keep the **default value**
  `def_val` (defaults to `0.0`; `np.nan` makes the uncovered cells explicit).
- All operators share one calling convention: build them with
  `tr = mf.transfer.Op(src, tgt, ...)`, then assign the lazy result to a
  target field with `tgt.fields[name] = tr(expr)`. `tr.eval(expr)` returns the
  per-type arrays directly.

### Cross-dimensional `ConservativeP0`

`ConservativeP0` also transfers from higher- to lower-dimensional cells in the
same space:

| Source | Target | Intersection measure |
|---|---|---|
| 3D volume | 2D surface | area |
| 3D volume | 1D line | length |
| 3D volume | 0D point | 1 if the point is in the cell |
| 2D surface (2D or 3D space) | 1D line | length |
| 2D surface (2D or 3D space) | 0D point | 1 if the point is in the cell |
| 1D line | 0D point | 1 if the point is on the segment |

The class, signature and prepare / apply split are unchanged:

```python
# src is a volume mesh, tgt a surface mesh (for example a cutting plane)
tr = mf.transfer.ConservativeP0(src, tgt)
tr.apply_update(src, "T", tgt, "T")
```

For an **intensive** field the result is the area- (or length-) weighted average
over the target cell, normalized by the **whole** target measure: a target cell
that is only half covered by the source mesh is attenuated by one half. For an
**extensive** field the raw measure-weighted sum is kept, i.e. the integral of
the source field over the target cell. For point targets the target measure is
the counting measure (`1`), and each point is assigned to a single source cell
(on a conforming interface the lowest source index wins).

The usual assumptions apply: convex, first-order cells (`TET4`, `HEX8`,
`WEDGE6`, `PHED`, `TRI3`, `QUAD4`, `PGON`, `SEG2`, `VERTEX`), and 2D cells
embedded in 3D are planar up to a small tolerance. Same-dimensional transfers
still require both meshes to be full-dimensional.

### ConstantPiecewise Transfer

The transfer is very simple. It is based on the cells of src_mesh and the cells center of the target mesh. It assigns to a cell from the target the value of the cell in which the center is located in. This is a point location based value assignment. By default the centroid (mean of cell nodes) is used because it is fast to compute and is accurate with regular cells.


```python
m_src = mesh2.select((m2 > 4e-9) - r - c).to_mesh()
m_tgt = mf.build_cmesh(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))

# The transfer is computed between source and target geometry.
# This step is computationnaly heavy, but done once.
cpt = mf.transfer.ConstantPiecewise(m_src, m_tgt, def_val=np.nan)
```


```python
# The transfer is applied. This step is much faster.
m_tgt.fields["Projection"] = cpt("Measure")
```


```python
m_tgt.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_8_0.png)



As you can see the `"Measure"` field from m_src was used to compute the
`"Projection"` field on m_tgt. Both meshes are not completely overlapping but
that is not an issue: cells from m_tgt whose center is not in a cell from m_src
take a default value `def_val`. Default is `0.0`, but any floating point value,
such as `np.nan`, is accepted.

This interpolation is a good fit when coarsening a mesh and you do not need
conservation. It is fast, though not as fast as the meshless methods, because
of the `is_in_cell` exact geometrical query.

### InverseDistance Transfer

The meshless counterpart of the piecewise-constant copy: each target cell
center samples its `k` nearest source cell centers and receives their weighted
mean, with weights `w_i ~ 1 / r**exponent` (defaults: `k = 4`,
`exponent = 2.0`). The weights are non-negative and normalized, so the value is
a convex combination of the neighbours — it stays inside the source range,
never overshoots, and is cheap to evaluate.

Use it when `ConstantPiecewise` leaves holes (uncovered cells) or when the
source and target meshes are unrelated point clouds. A higher `exponent`
concentrates the weight on the closest neighbour; `k = 1` degenerates to exact
nearest-neighbour sampling.

### MovingLeastSquares Transfer

The most flexible of the meshless methods: for each target cell center it
fits a local least-squares polynomial through the `k` nearest source centers
(default `k = 10`) and evaluates it at the target point. The neighbour weights
are shaped by a distance kernel, one of the `mf.DistanceWeighting`
choices:

- `Constant()` — a plain (linear) least-squares fit over the `k` neighbours;
- `InverseDistance(exponent)` — neighbours weighted by `(h / r)**exponent`
  where `h` is the distance to the farthest selected neighbour;
- `Gaussian()` — a gaussian kernel.

It smooths the field and reconstructs trends, but it is the only method that
can **overshoot** the source values and **extrapolate** outside the source
domain. Watch the boundaries — using `def_val=np.nan` as target for the
uncovered cells makes the extrapolated region visible.


```python
m_src = mesh2.select((m2 > 4e-9) - r - c).to_mesh()
m_tgt = mf.build_cmesh(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))

# The transfer is computed between source and target geometry.
# This step is computationnaly heavy, but done once.
mlsqt = mf.transfer.MovingLeastSquares(m_src, m_tgt, k=10, def_val=np.nan)
```


```python
# The transfer is applied. This step is much faster.
m_tgt.fields["Projection"] = mlsqt("Measure")
```


```python
m_tgt.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_14_0.png)



As you can see the projection gets a value everywhere, even outside the initial domain. This can lead to bad extrapolations, so it is to your responsability. Here as an example the extrapolated Measure can have negative values.

Inside the domain the interpolation works like a charm.

### Transfer methods comparison

Which method to use? `ConstantPiecewise` (CPW) is local and cheap, `MovingLeastSquares` (MLS) smooths the field but can extrapolate outside the source, `InverseDistance` (ID) is a tunable local fallback, and `ConservativeP0` conserves the integral.

Every method is timed on the same grid pair, split between the one-shot
**prepare** (slow) and the repeated **apply** (fast). The timing loop is hidden
for readability — it only fills the chart below.



![png](transfers_files/transfers_18_0.png)





![png](transfers_files/transfers_18_1.png)





![png](transfers_files/transfers_18_2.png)





![png](transfers_files/transfers_18_3.png)





![png](transfers_files/transfers_18_4.png)





![png](transfers_files/transfers_18_5.png)





![png](transfers_files/transfers_18_6.png)





![png](transfers_files/transfers_18_7.png)





![png](transfers_files/transfers_18_8.png)





![png](transfers_files/transfers_18_9.png)





![png](transfers_files/transfers_18_10.png)





![png](transfers_files/transfers_19_0.png)



## Transfer of 3D fields

The same benchmark on a hexahedral grid pair — here `ConstantPiecewise`,
`InverseDistance` and `ConservativeP0` are timed. The prepare / apply split is
unchanged:



![png](transfers_files/transfers_22_0.png)





![png](transfers_files/transfers_22_1.png)





![png](transfers_files/transfers_22_2.png)





![png](transfers_files/transfers_22_3.png)





![png](transfers_files/transfers_22_4.png)





![png](transfers_files/transfers_23_0.png)



## Transfer between surfaces

Transfers do not apply to volumetric meshes only: they also work between surface
meshes, i.e. 2D domains embedded in 3D space. Here `sin_mesh` builds such a surface
by extruding a sine line along an increasing height. The cell `Measure` is remapped
from a coarse surface to a finer one with the `InverseDistance` interpolation.


```python
def sin_mesh(n=20):
    x = np.linspace(0.0, 1.0, n)
    coords = np.c_[x, 0.1 * np.sin(x * 2.0 * np.pi)]
    conn = np.c_[np.arange(0, n - 1, dtype=np.uint), np.arange(1, n, dtype=np.uint)]
    res = mf.UMesh(coords)
    res.add_regular_block("SEG2", conn)
    return res.extrude(np.linspace(0.0, 1.0, n) ** 2)
```


```python
coarse = sin_mesh(9)
coarse.measure_update()
coarse.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_26_0.png)




```python
fine = sin_mesh(20)
fine.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_27_0.png)




```python
tr = mf.transfer.InverseDistance(coarse, fine)
fine.fields["Measure"] = tr("Measure")
```


```python
fine.to_pyvista().plot()
```



![png](transfers_files/transfers_29_0.png)



## Comparison to medcoupling


```python
import medcoupling as mc
from medcoupling import MEDCouplingRemapper
```


```python
m_src = mf.build_cmesh(
    np.logspace(-2.0, 0.0, 20), np.logspace(-2.0, 0.0, 20), np.linspace(0.0, 0.1, 5)
)
m_tgt = mf.build_cmesh(
    np.linspace(-0.05, 1.1, 40), np.linspace(-0.05, 1.1, 40), np.linspace(0.0, 0.1, 3)
)
m_src.fields["Measure"] = mf.M

t0 = time.time()
tr = mf.transfer.ConservativeP0(m_src, m_tgt)
t1 = time.time()
m_tgt.fields["Measure"] = tr("Measure")
t2 = time.time()

mf_prepare = t1 - t0
mf_apply = t2 - t1
```



![png](transfers_files/transfers_33_0.png)




```python
mc_src = m_src.to_mc()
mc_tgt = m_tgt.to_mc()

f_src = mc_src.getMeasureField(True)
f_src.setNature(mc.IntensiveConservation)

remap = MEDCouplingRemapper()
t0 = time.time()
remap.prepare(mc_src, mc_tgt, "P0P0")
t1 = time.time()
f_tgt = remap.transferField(f_src, 0.0)
t2 = time.time()

mc_prepare = t1 - t0
mc_apply = t2 - t1
```


```python
# Identical fields, whichever library computed them
mf_f = m_tgt.fields["Measure"].numpy()
mc_f = f_tgt.getArray().toNumPyArray()
assert np.allclose(mc_f, mf_f)
```



![png](transfers_files/transfers_36_0.png)



## Using transfer as a FieldExpressions


```python
trsf = mf.transfer.ConservativeP0(m_src, m_tgt)
rho = 3.0 * mf.X + mf.Y.square()
cp = 10.0 * mf.M
m_src.fields["rhoCp"] = rho * cp
m_tgt.fields["temp"] = trsf(rho * cp)
```

> **About the default value and the zeros**
>
> Target cells that are **not covered** by any source cell keep the transfer `def_val`, which defaults to `0.0`. With the log-pressure source `m_src` confined to ~[0.001, 0.1] above, most of the large `m_tgt` (up to 1.5) is outside the source, so most target cells are legitimately zero. The earlier cells set `def_val=np.nan` on purpose so that uncovered cells show up as `NaN` instead of `0.0`; use `def_val=np.nan` (or any other sentinel) the same way with `eval`/`__call__`:
>
> ```python
> trsf = mf.transfer.ConservativeP0(m_src, m_tgt, def_val=np.nan)
> m_tgt.fields["temp"] = trsf(rho)   # uncovered cells -> NaN, covered cells -> rho
> ```
>
> Cells covered by the source are transferred correctly; only the uncovered cells
> differ because of the chosen default value.



![png](transfers_files/transfers_40_0.png)

## Saving and reusing a prepared operator

Building a transfer operator is the expensive part; applying it is a sparse
matrix-vector product. An operator can be written to disk and loaded back in a
later session:

```python
trsf = mf.transfer.ConservativeP0(m_src, m_tgt, def_val=np.nan)
trsf.save("transfer.npz")

# src_mesh is required; tgt_mesh is optional but enables stronger validation.
loaded = mf.transfer.load("transfer.npz", src_mesh=m_src, tgt_mesh=m_tgt)
m_tgt.fields["temp"] = loaded("temp")
```

The file is a standard NumPy `.npz` archive, so it can also be opened without
mefikit:

```python
import scipy.sparse as sp

matrix = sp.load_npz("transfer.npz")  # CSR matrix
```

The CSR matrix has one row per target cell and one column per source cell.
Blocks are stored by element type; the file metadata records each block's
element type, size and explicit row/column offset. `mf.transfer.load` uses those
offsets to remap columns to the current in-memory element-type order, so files
remain valid if new element types are inserted into the enum later. The extra
`mefikit_meta` array contains the JSON metadata; SciPy ignores it.

On load, mefikit checks the CSR structure (row pointers, index bounds, array
lengths), the method parameters, and, when meshes are supplied, the per-element-
type block counts of both meshes. A mismatch raises `ValueError` with the
expected and actual blocks. Validation is structural: it does not compare cell
coordinates, so a saved operator is only meaningful for the geometry it was
built from. Pass `compressed=False` to `save` if you prefer uncompressed
(faster) I/O.

## Parallel transfer with MPI

When the source and target meshes are split across MPI ranks,
`mf.transfer.DistributedConservativeP0` performs the same conservative P0
transfer on each rank's local pieces and exchanges only the source values that
are actually needed. MPI is used from Python through `mpi4py`; the Rust
extension itself stays MPI-free. Install the optional dependency with:

```bash
pip install "mefikit[mpi]"
```

The example below builds one local piece per rank. Running it normally executes
the single-rank path; running it with `mpirun -n N` splits the cells over the
ranks (the piece construction is only here to make the example self-contained).



```python
import numpy as np
from mpi4py import MPI

import mefikit as mf

comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()
print(f"rank {rank}/{size}")

x = np.linspace(0.0, 1.0, 17)
src_full = mf.build_cmesh(x, x)
tgt_full = mf.build_cmesh(np.linspace(0.0, 1.0, 13), np.linspace(0.0, 1.0, 13))
values = (np.arange(src_full.num_elements(), dtype=float) + 1.0).reshape(-1, 1)
src_full.set_field("T", {"QUAD4": values})


def local_piece(mesh, et, start, stop):
    if stop <= start:
        return None
    ids = np.arange(start, stop, dtype=np.uintp)
    return mesh.select({et: ids}).to_mesh(False).remove_unused_coords()


s_lo = rank * src_full.num_elements() // size
s_hi = (rank + 1) * src_full.num_elements() // size
t_lo = rank * tgt_full.num_elements() // size
t_hi = (rank + 1) * tgt_full.num_elements() // size

src = local_piece(src_full, "QUAD4", s_lo, s_hi)
tgt = local_piece(tgt_full, "QUAD4", t_lo, t_hi)
if src is not None:
    src.set_field("T", {"QUAD4": np.ascontiguousarray(values[s_lo:s_hi])})

print(f"local source cells: {0 if src is None else src.num_elements()}")
print(f"local target cells: {0 if tgt is None else tgt.num_elements()}")
```

    rank 0/1
    local source cells: 256
    local target cells: 144



```python
transfer = mf.transfer.DistributedConservativeP0(src, tgt, comm=comm)
local = transfer.eval("T")
local_values = (
    np.asarray(local["QUAD4"]).reshape(-1, 1) if "QUAD4" in local else np.empty((0, 1))
)
print(f"local transferred shape: {local_values.shape}")

# extensive=True keeps the integral: the target integral equals the source one.
extensive = transfer.eval("T", extensive=True)
local_integral = (
    float(np.asarray(extensive["QUAD4"]).sum()) if "QUAD4" in extensive else 0.0
)
total = comm.allreduce(local_integral, op=MPI.SUM)
if rank == 0:
    print(f"distributed extensive integral: {total:.12f}")

# Store the result on the local target mesh, as with the sequential operators.
transfer.apply_update(src, "T", tgt, "T_mpi")
if tgt is not None:
    print(f"target field T_mpi: {tgt.fields['T_mpi'].numpy().shape}")

# Gather the local pieces on rank 0 and compare with the sequential operator.
gathered = comm.gather(local_values, root=0)
if rank == 0:
    distributed = np.concatenate(gathered, axis=0)
    sequential = np.asarray(
        mf.transfer.ConservativeP0(src_full, tgt_full).eval("T")["QUAD4"]
    ).reshape(-1, 1)
    print(
        "max |distributed - sequential| = "
        f"{np.max(np.abs(distributed - sequential)):.3e}"
    )
```

    local transferred shape: (144, 1)
    distributed extensive integral: 128.500000000000
    target field T_mpi: (144, 1)
    max |distributed - sequential| = 0.000e+00


The operator is prepared once and can be reused for any number of fields, for
example in a time loop. Intensive fields are normalized by the target cell
measure, while `extensive=True` keeps the raw measure-weighted sum. For a
one-shot transfer, use `mf.transfer.distributed(src, expr, tgt, comm=...)`; for
repeated fields, keep the `DistributedConservativeP0` object.
