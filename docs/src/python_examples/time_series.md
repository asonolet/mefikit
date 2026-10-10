# Time series

*Figures are omitted; the API is the focus here.*


```python
import numpy as np

import mefikit as mf
```

## Creating a time series

A time series is a set of fields sharing a common **base name**, each step stored as a field
named `"<base>_step_<n>_time_<t>"`. Steps are numbered automatically (`0, 1, 2, ...`), and the
time stamps are yours to choose: they need not be sorted nor unique.


```python
mesh = mf.build_cmesh(np.linspace(0.0, 1.0, 3), np.linspace(0.0, 1.0, 3))
```


```python
mesh.time_series["T"].add(20.0, time=0.0)
```




    FieldRef("T_step_0_time_0")




```python
mesh.time_series["T"].add(40.0, time=0.5)
mesh.time_series["T"].add(60.0, time=1.0)
```




    FieldRef("T_step_2_time_1")




```python
mesh.time_series
```




    TimeSeriesMapping(["T"])




```python
mesh.fields.keys()  # the same data, seen as raw steps
```




    ['T_step_0_time_0', 'T_step_1_time_0.5', 'T_step_2_time_1']



## Step access

`mesh.time_series` maps base names to `TimeSeries` handles. A handle knows its base, the
dimension it lives on, and its ordered steps.


```python
s = mesh.time_series["T"]
```


```python
s.base, s.dim()
```




    ('T', 2)




```python
s.steps()
```




    [0, 1, 2]




```python
s.times()
```




    [0.0, 0.5, 1.0]




```python
s.at(0).numpy()  # exact step
```




    array([20., 20., 20., 20.])




```python
s.at_time(0.5).numpy()  # exact time
```




    array([40., 40., 40., 40.])




```python
s.nearest(0.7).numpy()  # closest time
```




    array([40., 40., 40., 40.])




```python
s.latest().numpy()  # last step
```




    array([60., 60., 60., 60.])




```python
s[1]  # int -> step
s.nearest(0.9)  # float lookups go through nearest()/at_time()
```




    FieldRef("T_step_2_time_1")



`TimeSeries` also supports the usual collection protocol.


```python
list(s)  # (time, FieldRef) pairs, in step order
```




    [(0.0, FieldRef("T_step_0_time_0")),
     (0.5, FieldRef("T_step_1_time_0.5")),
     (1.0, FieldRef("T_step_2_time_1"))]




```python
s.items()  # (step, time, FieldRef) triples
```




    [(0, 0.0, FieldRef("T_step_0_time_0")),
     (1, 0.5, FieldRef("T_step_1_time_0.5")),
     (2, 1.0, FieldRef("T_step_2_time_1"))]




```python
(len(s), 1 in s, 7 in s)
```




    (3, True, False)




```python
s  # compact repr
```




    TimeSeries('T', dim=2) {0:0, 1:0.5, 2:1}



## Growing a series

`add` appends the next step (`max step + 1`) with the given time. Values accept the same inputs
as `fields`: scalars, arrays, per-block dicts, field expressions,


```python
mesh.time_series["T"].add(np.array([1.0, 2.0, 3.0, 4.0]), time=1.5)
```




    FieldRef("T_step_3_time_1.5")




```python
s.times()
```




    [0.0, 0.5, 1.0, 1.5]




```python
s.at(3).numpy()
```




    array([1., 2., 3., 4.])



## Mutating steps

Timestamps can be rewritten in place, and steps removed again.


```python
s.set_time(0, -0.1)  # re-stamp step 0
```


```python
s.times()
```




    [-0.1, 0.5, 1.0, 1.5]




```python
s.remove(1)
print(s.steps(), s.times())
```

    [0, 2, 3] [-0.1, 1.0, 1.5]


`TimeSeries` is also subscript-settable / deletable through dunder methods.


```python
del s[3]
print(s.steps(), s.times())
```

    [0, 2] [-0.1, 1.0]


## Renaming a series

Renaming moves every step to a new base name, and the handle follows.


```python
s.rename("Temperature")
print(s.base)
```

    Temperature



```python
mesh.time_series
```




    TimeSeriesMapping(["Temperature"])




```python
sorted(mesh.fields.keys())
```




    ['Temperature_step_0_time_-0.1', 'Temperature_step_2_time_1']



## Coexisting with plain fields

Static, non-stepped fields live alongside the series and are untouched by any of the
time-series helpers.


```python
mesh.fields["steady"] = 3.0
print(sorted(mesh.fields.keys()))
```

    ['Temperature_step_0_time_-0.1', 'Temperature_step_2_time_1', 'steady']


## Limitations

Time series are a light naming scheme over ordinary fields: they do **not** model MED's
multi-step fields. MED import/export currently only handle the first step of a series; a
conversion with time-stepping is not provided (yet).
