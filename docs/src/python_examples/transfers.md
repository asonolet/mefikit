# Field transfers

This notebook shows how to remap fields between meshes with the
`mf.transfer` operators: interpolation, extrapolation and conservative
remapping, all sharing the same prepare / apply split.


```python
import numpy as np
import pyvista as pv

import mefikit as mf

pv.set_plot_theme("dark")
pv.set_jupyter_backend("static")

# Same mesh and selection context as the Fields notebook.
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

The transfer function can be
- interpolation
- extrapolation
- conservative
- non-conservative
- using cells
- using cell centers and point clouds methods
- etc

They are many.

### ConstantPiecewise Transfer

The transfer is very simple. It is based on the cells of src_mesh and the cells center of the target mesh. It assigns to a cell from the target the value of the cell in which the center is located in. This is a point location based value assignment. By default the centroid (mean of cell nodes) is used because it is fast to comupute and is accurate with regular cells.


```python
m_src = mesh2.select((m2 > 4e-9) - r - c).to_mesh()
m_tgt = mf.build_cmesh(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))

# The transfer is computed between source and target geometry.
# This step is computationnaly heavy, but done once.
cpt = mf.transfer.ConstantPiecewise(m_src, m_tgt)
```


```python
# The transfer is applied. This step is much faster.
cpt.apply_update(m_src, "Measure", m_tgt, tgt_field_name="Projection", def_val=np.nan)
```


```python
m_tgt.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_7_0.png)



As you can see the `"Measure"` field from m_src was used to compute the `"Projection"` field on m_tgt. Both mesh are not completly overlapping but that is not an issue. Cells from m_tgt whose center is not in a cell from m_src take a default value `def_val`. Default is 0.0 but any floating point value, such as `np.nan` is accepted.

This interpolation is good when coarseing a mesh and you do not need conservation. It might be useful in other circumstances I do not know of. It is quite fast but not that much because of the `is_in_cell` exact geometrical query.

### MovingMean Transfer

This Transfer is based on m_src cell center positions and m_tgt cell centers positions. It is a "meshless" operation as it does not care about connectivity. There are several options :

- normal mean
- weighted mean

Pros :

- it is extremly fast to compute
- it does not overshoot / undershoot

Cons :

- It lacks precision

### MovingLeastSquare Transfer

This Transfer is based on m_src cell center positions and m_tgt cell centers positions. It is a "meshless" operation as it does not care about connectivity. There are several options :

- linear least square : the projection is the least square linear approx of the solution (can be an extrapolation)
- weighted least square : the projection is the weighted least square linear approx of the solution, there are several possibilities for the weighting function but it depends on the relative distance to the target interpolation point.


```python
m_src = mesh2.select((m2 > 4e-9) - r - c).to_mesh()
m_tgt = mf.build_cmesh(np.linspace(0.0, 1.5, 20), np.linspace(0.0, 1.5, 20))

# The transfer is computed between source and target geometry.
# This step is computationnaly heavy, but done once.
mlsqt = mf.transfer.MovingLeastSquares(m_src, m_tgt, k=10)
```


```python
# The transfer is applied. This step is much faster.
mlsqt.apply_update(m_src, "Measure", m_tgt, tgt_field_name="Projection", def_val=np.nan)
```


```python
m_tgt.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_13_0.png)



As you can see the projection gets a value everywhere, even outside the initial domain. This can lead to bad extrapolations, so it is to your responsability. Here as an example the extrapolated Measure can have negative values.

Inside the domain the interpolation works like a charm.

### Transfer methods comparison


```python
def compare_src_tgt(m_src, m_tgt):
    scale = (1.0 - 1e-2) / (1.1 + 0.05)

    pt = pv.Plotter(shape=(1, 2))
    pt.subplot(0, 0)
    pt.add_text("Source")
    pt.add_mesh(m_src.to_pyvista(), show_edges=True, clim=[0.0, 0.06])
    pt.camera_position = "xy"
    pt.camera.zoom(scale)
    pt.subplot(0, 1)
    pt.add_text("Target")
    pt.add_mesh(
        m_tgt.to_pyvista(), clim=[0.0, 0.06], below_color="pink", above_color="red"
    )
    pt.add_mesh(m_src.descend().to_pyvista(), show_edges=True, line_width=1)
    pt.camera_position = "xy"
    pt.show()
```


```python
import time

transfers = (
    mf.transfer.ConstantPiecewise,
    lambda src, tgt: mf.transfer.MovingLeastSquares(src, tgt, k=5),
    mf.transfer.MovingLeastSquares,
    lambda src, tgt: mf.transfer.MovingLeastSquares(src, tgt, k=20),
    mf.transfer.MovingLeastSquares,
    lambda src, tgt: mf.transfer.MovingLeastSquares(
        src, tgt, weighting=mf.transfer.DistanceWeighting.Gaussian()
    ),
    lambda src, tgt: mf.transfer.MovingLeastSquares(
        src, tgt, weighting=mf.transfer.DistanceWeighting.InverseDistance(1.0)
    ),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=3),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=5),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=10),
    mf.transfer.ConservativeP0,
)
trasfers_labels = (
    "CPW",
    "MLS k5",
    "MLS k10",
    "MLS k20",
    "MLS",
    "MLS gaussian",
    "MLS inv_dist",
    "ID k3",
    "ID k5",
    "ID k10",
    "ConservativeP0",
)
prepare_times = []
apply_times = []

for T, label in zip(transfers, trasfers_labels):
    m_src = mf.build_cmesh(np.logspace(-2.0, 0.0, 20), np.logspace(-2.0, 0.0, 20))
    m_tgt = mf.build_cmesh(np.linspace(-0.05, 1.1, 40), np.linspace(-0.05, 1.1, 40))
    m_src.fields["Measure"] = mf.M
    t0 = time.time()
    tr = T(m_src, m_tgt)
    t1 = time.time()
    tr.apply_update(m_src, "Measure", m_tgt, label + " Transfered Measure")
    t2 = time.time()

    prepare_times.append((t1 - t0) * 1000.0)
    apply_times.append((t2 - t1) * 1000.0)

    compare_src_tgt(m_src, m_tgt)
```



![png](transfers_files/transfers_17_0.png)





![png](transfers_files/transfers_17_1.png)





![png](transfers_files/transfers_17_2.png)





![png](transfers_files/transfers_17_3.png)





![png](transfers_files/transfers_17_4.png)





![png](transfers_files/transfers_17_5.png)





![png](transfers_files/transfers_17_6.png)





![png](transfers_files/transfers_17_7.png)





![png](transfers_files/transfers_17_8.png)





![png](transfers_files/transfers_17_9.png)





![png](transfers_files/transfers_17_10.png)




```python
import matplotlib.pyplot as plt

chart_data = {
    "Prepare": prepare_times,
    "Apply": apply_times,
}

fig, ax = plt.subplots(figsize=(10, 5))

res = ax.grouped_bar(chart_data, tick_labels=trasfers_labels, group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)

# Add some text for labels, title, etc.
ax.set_ylabel("Time (ms)")
ax.set_title("Time per step")
ax.legend(loc="upper left", ncols=3)
fig.tight_layout()
plt.show()
```



![png](transfers_files/transfers_18_0.png)



## Transfer of 3D fields


```python
def compare_src_tgt_3d(m_src, m_tgt):
    vmin = m_src.select(mf.sel.all()).min(mf.M) * 0.99
    vmax = m_src.select(mf.sel.all()).max(mf.M) * 1.01
    scale = (1.0 - 1e-2) / (1.1 + 0.05)

    pt = pv.Plotter(shape=(1, 2))
    pt.subplot(0, 0)
    pt.add_text("Source")
    pt.add_mesh(m_src.to_pyvista(), show_edges=True, clim=[vmin, vmax])
    # pt.view_xy()
    pt.camera.zoom(scale)

    pt.subplot(0, 1)
    pt.add_text("Target")
    pt.add_mesh(
        m_tgt.to_pyvista(), clim=[vmin, vmax], below_color="pink", above_color="red"
    )
    pt.add_mesh(m_src.descend(target_dim=1).to_pyvista(), show_edges=True, line_width=1)
    # pt.view_xy()
    pt.show()
```


```python
import time

transfers = (
    mf.transfer.ConstantPiecewise,
    # lambda src, tgt: mf.transfer.MovingLeastSquares(src, tgt, k=5),
    # mf.transfer.MovingLeastSquares,
    # lambda src, tgt: mf.transfer.MovingLeastSquares(src, tgt, k=20),
    # mf.transfer.MovingLeastSquares,
    # lambda src, tgt: mf.transfer.MovingLeastSquares(
    #     src, tgt, weighting=mf.transfer.DistanceWeighting.Gaussian()
    # ),
    # lambda src, tgt: mf.transfer.MovingLeastSquares(
    #     src, tgt, weighting=mf.transfer.DistanceWeighting.InverseDistance(1.0)
    # ),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=3),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=5),
    lambda src, tgt: mf.transfer.InverseDistance(src, tgt, k=10),
    mf.transfer.ConservativeP0,
)
trasfers_labels = (
    "CPW",
    "ID k3",
    "ID k5",
    "ID k10",
    "ConservativeP0",
)
prepare_times = []
apply_times = []

for T, label in zip(transfers, trasfers_labels):
    m_src = mf.build_cmesh(
        np.logspace(-2.0, 0.0, 20), np.logspace(-2.0, 0.0, 20), np.linspace(0.0, 0.1, 3)
    )
    m_tgt = mf.build_cmesh(
        np.linspace(-0.05, 1.1, 40),
        np.linspace(-0.05, 1.1, 40),
        np.linspace(0.0, 0.1, 3),
    )
    m_src.fields["Measure"] = mf.M
    t0 = time.time()
    tr = T(m_src, m_tgt)
    t1 = time.time()
    tr.apply_update(m_src, "Measure", m_tgt, label + " Transfered Measure")
    t2 = time.time()

    prepare_times.append((t1 - t0) * 1000.0)
    apply_times.append((t2 - t1) * 1000.0)

    compare_src_tgt_3d(m_src, m_tgt)
```



![png](transfers_files/transfers_21_0.png)





![png](transfers_files/transfers_21_1.png)





![png](transfers_files/transfers_21_2.png)





![png](transfers_files/transfers_21_3.png)





![png](transfers_files/transfers_21_4.png)




```python
import matplotlib.pyplot as plt

chart_data = {
    "Prepare": prepare_times,
    "Apply": apply_times,
}

fig, ax = plt.subplots(figsize=(10, 5))

res = ax.grouped_bar(chart_data, tick_labels=trasfers_labels, group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)

# Add some text for labels, title, etc.
ax.set_ylabel("Time (ms)")
ax.set_title("Time per step")
ax.legend(loc="upper left", ncols=3)
fig.tight_layout()
plt.show()
```



![png](transfers_files/transfers_22_0.png)



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



![png](transfers_files/transfers_25_0.png)




```python
fine = sin_mesh(20)
fine.to_pyvista().plot(show_edges=True)
```



![png](transfers_files/transfers_26_0.png)




```python
tr = mf.transfer.InverseDistance(coarse, fine)
tr.apply_update(coarse, "Measure", fine)
```


```python
fine.to_pyvista().plot()
```



![png](transfers_files/transfers_28_0.png)



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
tr.apply_update(m_src, "Measure", m_tgt)
t2 = time.time()

mf_prepare = t1 - t0
mf_apply = t2 - t1
```


```python
pt = pv.Plotter(shape=(1, 2))
pt.subplot(0, 0)
pt.add_text("Source")
pt.add_mesh(m_src.to_pyvista(), show_edges=True)
pt.camera.zoom(0.99 / 1.15)

pt.subplot(0, 1)
pt.add_text("Target")
pt.add_mesh(
    m_src.descend(target_dim=1).to_pyvista(),
    show_edges=True,
    line_width=2,
    color="black",
)
pt.add_mesh(m_tgt.to_pyvista(), show_edges=True)
pt.show()
```



![png](transfers_files/transfers_32_0.png)




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
# Les champs produits sont identiques
mf_f = m_tgt.fields["Measure"].numpy()
mc_f = f_tgt.getArray().toNumPyArray()
assert np.allclose(mc_f, mf_f)
```


```python
chart_data = {
    "Mefikit": [mf_prepare, mf_apply],
    "Medcoupling": [mc_prepare, mc_apply],
}

fig, ax = plt.subplots(figsize=(10, 5))

res = ax.grouped_bar(chart_data, tick_labels=["Prepare", "Apply"], group_spacing=1)
for container in res.bar_containers:
    ax.bar_label(container, padding=3)

# Add some text for labels, title, etc.
ax.set_ylabel("Time (s)")
ax.set_title("Time per step")
ax.legend(loc="upper right", ncols=3)
fig.tight_layout()
plt.show()
```



![png](transfers_files/transfers_35_0.png)



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
> Cells covered by the source are transferred correctly and match `apply_update` exactly.


```python
pt = pv.Plotter()

box = mf.sel.bbox([-np.inf] * 3, [0.9, np.inf, np.inf])
pvm = m_tgt.select(box).to_mesh().to_pyvista()
pvm.active_scalars_name = "temp"
pt.add_mesh(pvm.shrink(0.8), show_edges=True)

pvs = m_src.to_pyvista()
pvs.active_scalars_name = "rhoCp"
pt.add_mesh(pvs, style="wireframe", line_width=2)

pt.show()
```



![png](transfers_files/transfers_39_0.png)
