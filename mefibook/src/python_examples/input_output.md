# `mefikit.UMesh`


```python
import mefikit as mf
import numpy as np
```


```python
volumes = mf.build_cmesh(
    range(2), np.linspace(0.0, 1.0, 5), np.logspace(0.0, 1.0, 5) / 10.0
)
```

## Memory exports

- Through numpy arrays manipulations:
    - medcoupling
    - meshio
    - pyvista
- Through `string` translation to `Python`:
    - json


```python

```


```python
volumes.to_mc()
```




    MEDCouplingUMesh C++ instance at 0xb855fc0. Name : "mf_UMesh". Not set !




```python
volumes.to_pyvista()
```





<table style='width: 100%;'>
<tr><th>UnstructuredGrid</th><th>Information</th></tr>
<tr><td>N Cells</td><td>16</td></tr>
<tr><td>N Points</td><td>50</td></tr>
<tr><td>X Bounds</td><td>0.000e+00, 1.000e+00</td></tr>
<tr><td>Y Bounds</td><td>0.000e+00, 1.000e+00</td></tr>
<tr><td>Z Bounds</td><td>1.000e-01, 1.000e+00</td></tr>
<tr><td>N Arrays</td><td>0</td></tr>
</table>






```python
volumes.to_meshio()
```




    <meshio mesh object>
      Number of points: 50
      Number of cells:
        hexahedron: 16




```python
volumes.to_json()
```




    '{"coords":{"v":1,"dim":[50,3],"data":[0.0,0.0,0.1,1.0,0.0,0.1,0.0,0.25,0.1,1.0,0.25,0.1,0.0,0.5,0.1,1.0,0.5,0.1,0.0,0.75,0.1,1.0,0.75,0.1,0.0,1.0,0.1,1.0,1.0,0.1,0.0,0.0,0.17782794100389226,1.0,0.0,0.17782794100389226,0.0,0.25,0.17782794100389226,1.0,0.25,0.17782794100389226,0.0,0.5,0.17782794100389226,1.0,0.5,0.17782794100389226,0.0,0.75,0.17782794100389226,1.0,0.75,0.17782794100389226,0.0,1.0,0.17782794100389226,1.0,1.0,0.17782794100389226,0.0,0.0,0.31622776601683794,1.0,0.0,0.31622776601683794,0.0,0.25,0.31622776601683794,1.0,0.25,0.31622776601683794,0.0,0.5,0.31622776601683794,1.0,0.5,0.31622776601683794,0.0,0.75,0.31622776601683794,1.0,0.75,0.31622776601683794,0.0,1.0,0.31622776601683794,1.0,1.0,0.31622776601683794,0.0,0.0,0.5623413251903491,1.0,0.0,0.5623413251903491,0.0,0.25,0.5623413251903491,1.0,0.25,0.5623413251903491,0.0,0.5,0.5623413251903491,1.0,0.5,0.5623413251903491,0.0,0.75,0.5623413251903491,1.0,0.75,0.5623413251903491,0.0,1.0,0.5623413251903491,1.0,1.0,0.5623413251903491,0.0,0.0,1.0,1.0,0.0,1.0,0.0,0.25,1.0,1.0,0.25,1.0,0.0,0.5,1.0,1.0,0.5,1.0,0.0,0.75,1.0,1.0,0.75,1.0,0.0,1.0,1.0,1.0,1.0,1.0]},"element_blocks":{"HEX8":{"cell_type":"HEX8","connectivity":{"Regular":{"v":1,"dim":[16,8],"data":[0,1,3,2,10,11,13,12,2,3,5,4,12,13,15,14,4,5,7,6,14,15,17,16,6,7,9,8,16,17,19,18,10,11,13,12,20,21,23,22,12,13,15,14,22,23,25,24,14,15,17,16,24,25,27,26,16,17,19,18,26,27,29,28,20,21,23,22,30,31,33,32,22,23,25,24,32,33,35,34,24,25,27,26,34,35,37,36,26,27,29,28,36,37,39,38,30,31,33,32,40,41,43,42,32,33,35,34,42,43,45,44,34,35,37,36,44,45,47,46,36,37,39,38,46,47,49,48]}},"fields":{},"families":{"v":1,"dim":[128],"data":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},"groups":{}}}}'



# File read/write on rust side

- On rust side, file I/O with the `read`/`write` methods:
    - vtk
    - yaml
    - json

Note that for now the vtk reader/writer is only based on the binary vtk 2.0 file format (which is quite old). No rust crate is doing better. I am planning on implementing a HDF5/CGNS compliant rust reader/writer (based on the hdf5 lib) in order to support a more HPC file format.


```python
for ext in ("vtk", "yaml", "json"):
    volumes.write(f"data/volumes.{ext}")
    volumes_from_disk = mf.UMesh.read("data/volumes.vtk")
    assert volumes_from_disk
    assert (
        volumes != volumes_from_disk
    )  # this is a new instance, with a different memory adress
```
