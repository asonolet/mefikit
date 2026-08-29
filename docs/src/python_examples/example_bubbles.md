# Bubbles example


```python

```


```python
import numpy as np
import pyvista as pv

import mefikit as mf

rng = np.random.default_rng(seed=123)
pv.set_plot_theme("dark")
pv.set_jupyter_backend("static")
```

## Setup


```python
xmax = 5.0
ymax = 1.0
r = 0.17
nb = 15
nr = 12.5
```


```python
nx = int(xmax / r * nr)
ny = int(ymax / r * nr)
print(f"Number of elements : {nx * ny * ny:,}")
```

    Number of elements : 1,955,743



```python
xc = rng.uniform(r, xmax - r, nb)
yc = rng.uniform(r, ymax - r, nb)
zc = rng.uniform(r, ymax - r, nb)
spheres = [mf.sel.sphere([x, y, z], r) for x, y, z in zip(xc, yc, zc)]
sphere_union = spheres[0]
for s in spheres[1:]:
    sphere_union = sphere_union | s
```


```python
x = np.linspace(0.0, xmax, nx)
y = np.linspace(0.0, ymax, ny)
volumes = mf.build_cmesh(x, y, y)
```


```python
volumes.boundaries().to_pyvista().plot(opacity=0.4)
```



![png](example_bubbles_files/example_bubbles_8_0.png)



## Selecting bubbles


```python
# `select` returns a lazy view: materialize it into a sub-mesh with `.to_mesh()`.
inner_bubbles = volumes.select(sphere_union).to_mesh()
interface = inner_bubbles.boundaries()
cracked = volumes.crack(interface)
```


```python
# Named groups live in a dict-like mapping on the mesh:
# assign any selection expression (or {etype: ids} dict) to tag elements.
volumes.groups["bubbles"] = sphere_union
print(len(volumes.groups["bubbles"]), "elements tagged in group 'bubbles'")
```

    105990 elements tagged in group 'bubbles'


## Cracking and connected components


```python
cracked.boundaries().to_pyvista().plot(opacity=0.4)
```



![png](example_bubbles_files/example_bubbles_13_0.png)




```python
bubble_groups = inner_bubbles.connected_components()
```


```python
pv.global_theme.color_cycler = "default"
pl = pv.Plotter()
for c in bubble_groups:
    compo = c.to_pyvista()
    pl.add_mesh(compo)
pl.add_mesh(volumes.boundaries(target_dim=1).to_pyvista())
pl.show()
pv.global_theme.color_cycler = None
```



![png](example_bubbles_files/example_bubbles_15_0.png)




```python
clip1 = mf.sel.bbox([-np.inf] * 3, [np.inf, ymax / 3.0, np.inf])
pl = pv.Plotter()
pl.add_mesh(volumes.select(clip1 & ~sphere_union).to_mesh().to_pyvista())
pl.add_mesh(interface.to_pyvista(), opacity=0.4)
pl.show()
```



![png](example_bubbles_files/example_bubbles_16_0.png)



## Computing statistics


```python
bubble_volumes = volumes.select("bubbles").sum(mf.M)
print(nb * 4.0 / 3.0 * np.pi * r**3.0)
bubble_volumes
```

    0.3086928941417331





    0.2793115007084435




```python
bubbles_mean_pos = volumes.select("bubbles").mean(mf.C)
print(bubbles_mean_pos)
```

    [2.49923348 0.4928932  0.5314482 ]
