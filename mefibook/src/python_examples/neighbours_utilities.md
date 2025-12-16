# Neighbours related topological operations

Here are some examples about the:
- `submesh` method
- `boundaries` method


```python
import mefikit as mf
import numpy as np
import pyvista as pv

pv.set_plot_theme("dark")
pv.set_jupyter_backend("static")
```


```python
x = range(3)
y = np.linspace(0.0, 3.0, 5, endpoint=True)
z = np.logspace(-0.5, 1, 4, endpoint=True)
volumes = mf.build_cmesh(x, y, z)
```

## Submesh functionaly

This functionality is able to compute the descending connectivity of the porvided mesh.
It can act on elements of dimension 1, 2 or 3.


```python
faces = volumes.submesh()
edges = faces.submesh()
vertex = edges.submesh()

plotter = pv.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_mesh(faces.to_pyvista().shrink(0.8), show_edges=True)
plotter.subplot(0, 1)
plotter.add_mesh(edges.to_pyvista().shrink(0.8))
plotter.subplot(0, 2)
plotter.add_mesh(vertex.to_pyvista())
plotter.show()
```



![png](neighbours_utilities_files/neighbours_utilities_4_0.png)



## Submesh in one go

You might want to directly access either the node mesh or the edges mesh. You can ! And going into one step is ever faster than chaining mutliple `.submesh()` calls.


```python
edges = volumes.submesh(target_dim=1)
nodes = volumes.submesh(target_dim=0)

plotter = pv.Plotter(shape=(1, 2))
plotter.subplot(0, 0)
plotter.add_mesh(edges.to_pyvista().shrink(0.8))
plotter.subplot(0, 1)
plotter.add_mesh(vertex.to_pyvista())
plotter.show()
```



![png](neighbours_utilities_files/neighbours_utilities_6_0.png)



## Boundaries computation

As it is very common to compute boundaries on a mesh (for boundary counditions for ex), there is a custom `boundaries` computation method.


```python
face_bounds = volumes.boundaries()
edge_bounds = volumes.boundaries(target_dim=1)
vertex_bounds = volumes.boundaries(target_dim=0)

plotter = pv.Plotter(shape=(1, 3))
plotter.subplot(0, 0)
plotter.add_mesh(face_bounds.to_pyvista().shrink(0.8), show_edges=True)
plotter.subplot(0, 1)
plotter.add_mesh(edge_bounds.to_pyvista().shrink(0.8))
plotter.subplot(0, 2)
plotter.add_mesh(vertex_bounds.to_pyvista())
plotter.show()
```



![png](neighbours_utilities_files/neighbours_utilities_8_0.png)
