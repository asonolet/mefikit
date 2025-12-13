```python
import mefikit as mf
import pyvista as pv
pv.set_jupyter_backend('static')
```


```python
a = mf.data.cmesh3()
```


```python
a.to_pyvista().shrink(0.9).plot(show_edges=True)
```



![png](mefikit_demo_files/mefikit_demo_2_0.png)
