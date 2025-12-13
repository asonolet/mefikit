# Example using `PyVista` to visualize a mesh created with `mefikit`


```python
import mefikit as mf
import numpy as np
import pyvista as pv
pv.set_jupyter_backend('static')
```


```python
volumes = mf.build_cmesh(range(2), np.linspace(0., 1.0, 5), np.logspace(0., 1.0, 5) / 10.0)
volumes.to_pyvista().plot(show_edges=True)
```



![png](mefikit_demo_files/mefikit_demo_2_0.png)




```python
faces = volumes.submesh()
faces.to_pyvista().shrink(0.8).plot(show_edges=True)
```



![png](mefikit_demo_files/mefikit_demo_3_0.png)




```python
edges = faces.submesh()
edges.to_pyvista().plot()
```



![png](mefikit_demo_files/mefikit_demo_4_0.png)




```python
nodes = edges.submesh()
nodes.to_pyvista().plot()
```



![png](mefikit_demo_files/mefikit_demo_5_0.png)
