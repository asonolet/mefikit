# mefikit vs. medcoupling

**Same geometry. Same operations. Same numbers. Measured.**

---

## Two libraries, one goal

- Manipulate unstructured meshes and their fields.
- Both work on the very same `.med` files.
- This talk compares them fairly: same meshes, same operations, same numbers.

---

## medcoupling

- Mature C++ library, **the reference implementation** of the MED world.
- Decades of API surface, embedded in many industrial codes (Salome, ...).
- Advanced field machinery, spline remapping, a huge ecosystem.

---

## mefikit

- Young library, **Rust core** with Python bindings.
- Same goal as medcoupling, with a more concise API and a fast core.
- Aimed at the common operations first: polyhedral meshes, fields, transfers.

---

## The roadmap, honestly

- The long-term goal is to go further than mefikit's current scope — one day.
- Today, many features are not implemented: mefikit **complements**, it does not replace.
- medcoupling is used in many other codes; moving to mefikit has not started — it is not even near its end.
- Here, we only compare **interfaces** and **timings** on a handful of operations.

---

## Method: fair by construction

- Both libraries always work on the **same geometry**.
- Every result is **cross-checked** with the other library — agreement ≈ 1e-15.
- Timings are **medians of repeated runs** on one machine.

---

## Building a grid

`build_cmesh(*axes)` ⊗ `MEDCouplingCMesh` + `buildUnstructured()`

::::: columns
::: column
```python
# mefikit
x = np.linspace(0.0, 1.0, 6)
mf_mesh = mf.build_cmesh(x, x)
```
:::
::: column
```python
# medcoupling
c = mc.MEDCouplingCMesh()
c.setCoords(mc.DataArrayDouble(x), mc.DataArrayDouble(x))
mc_mesh = c.buildUnstructured()
mc_mesh.setMeshDimension(2)
```
:::
:::::

Identical: **25 QUAD4 cells, 36 nodes.**

---

## The same grid, side by side

![Same cartesian grid, two engines](compare_medcoupling_files/compare_medcoupling_8_0.png){width=92%}

---

## Fields and measures

`mf.M`, `mf.Field` DSL ⊗ `DataArrayDouble` + field objects

::::: columns
::: column
```python
# mefikit
mesh.fields["Measure"] = mf.M
mesh.fields["T"] = 1.0 + mf.X**2
                 + 0.5 * mf.Y
hot = mesh.select(mf.Field("T") > 1.5)
print(hot.mean("T"))
```
:::
::: column
```python
# medcoupling
m = mesh.getMeasureField(True)
c = mesh.computeCellCenterOfMass()
T = 1.0 + c[:, 0] ** 2 + 0.5 * c[:, 1]
f = mc.MEDCouplingFieldDouble(mc.ON_CELLS, mc.ONE_TIME)
f.setArray(mc.DataArrayDouble(T))
f.setMesh(mesh)
```
:::
:::::

Same values: measure sum = 1.0, max |ΔT| ≈ 1.3e-15.

---

## Fields — same values

![Same field, two engines](compare_medcoupling_files/compare_medcoupling_13_0.png){width=92%}

---

## Faces of a volume mesh

`descend()` ⊗ `buildDescendingConnectivity()`

::::: columns
::: column
```python
# mefikit
faces = mesh.descend()
```
:::
::: column
```python
# medcoupling
desc = mesh.buildDescendingConnectivity()
faces = desc[0]
```
:::
:::::

Identical: **240 faces** on a 4³ hexa grid.

---

## Faces — visual

![Faces of a volume mesh](compare_medcoupling_files/compare_medcoupling_16_0.png){width=92%}

---

## Merge duplicated nodes

`merge_nodes()` ⊗ `mergeNodes(1e-12)`

::::: columns
::: column
```python
# mefikit
merged = cracked.merge_nodes()
```
:::
::: column
```python
# medcoupling
m = cracked_mc.deepCopy()
m.mergeNodes(1e-12)
```
:::
:::::

- Cracked hex stack: **128 → 50 nodes**, **16 → 1 component**, both sides.
- mefikit rewires connectivity (no node compaction); medcoupling compacts.

---

## Merge — visual

![Cracked vs merged](compare_medcoupling_files/compare_medcoupling_19_0.png){width=88%}

---

## 2D overlay / imprint

`overlay(operation=...)` ⊗ `Intersect2DMeshes()`

::::: columns
::: column
```python
# mefikit
imprint = g1.overlay(g2, mf.OverlayOperation.IMPRINT)
```
:::
::: column
```python
# medcoupling
imprint = mc.MEDCouplingUMesh.Intersect2DMeshes(g1m, g2m, 1e-12)[0]
```
:::
:::::

Unit area preserved — **1.0 on both sides.**

---

## Overlay — visual

![Overlay imprint](compare_medcoupling_files/compare_medcoupling_22_0.png){width=92%}

---

## Polyhedral boundaries

`boundaries()` ⊗ `buildBoundaryMesh()`

::::: columns
::: column
```python
# mefikit
bnd = mf_src.boundaries()
```
:::
::: column
```python
# medcoupling
bnd = mc_src.buildBoundaryMesh(True)
```
:::
:::::

Real 2000-cell PHED meshes (`mesh_36.med`, `mesh_27.med`): **888 faces each**.

---

## Polyhedra — visual

![Polyhedral boundary](compare_medcoupling_files/compare_medcoupling_29_0.png){width=92%}

---

## Conservative P0/P0 transfer

Prepare once, apply many. `ConservativeP0` ⊗ `MEDCouplingRemapper`

::::: columns
::: column
```python
# mefikit
op = mf.transfer.ConservativeP0(src, tgt)
op.apply_update(src, "T", tgt, "T", def_val=0.0)
```
:::
::: column
```python
# medcoupling
remap = mc.MEDCouplingRemapper()
remap.prepare(src, tgt, "P0P0")
f_tgt = remap.transferField(f_src, 0.0)
```
:::
:::::

Transferred fields match to ≈ 3.6e-15.

---

## Transfer — visual

![P0/P0 transfer, source to target](compare_medcoupling_files/compare_medcoupling_25_0.png){width=92%}

---

## Performance: the honest framing

- Only a **handful of operations** are timed — no general claim.
- Same geometry on every run, results **machine-dependent**.
- ms, medians of several runs.

---

## Prepare / build — one-off cost

| Operation | mefikit | medcoupling | ratio |
|---|---:|---:|---:|
| Remap 2D · QUAD4 (9216 cells) | 27.0 ms | 24.8 ms | 0.9× |
| Remap 3D · HEX8 (4096 cells) | 162.6 ms | 692.3 ms | 4.3× |
| Remap 3D · polyhedral | 356.6 ms | 8746.1 ms | 24.5× |
| Merge nodes · 3D HEX8 | 0.5 ms | 4.0 ms | 7.5× |
| Descend · 24³ HEX8 | 56.5 ms | 103.7 ms | 1.8× |
| Overlay · 32² QUAD4 | 2.1 ms | 68.4 ms | 33.0× |
| Crack · 20³ HEX8 | 235.6 ms | 1624.4 ms | 6.9× |

> ratio = medcoupling / mefikit — **> 1 ⇒ mefikit faster**.

---

## Prepare / build — all operations

![Prepare and build times](compare_medcoupling_files/compare_medcoupling_38_0.png){width=92%}

On this set, only 2D remap prepare goes medcoupling's way (0.9×).

---

## Transfer / apply — per-step cost

| P0/P0 remap | mefikit | medcoupling | ratio |
|---|---:|---:|---:|
| 2D QUAD4 | 0.18 ms | 3.90 ms | 21.5× |
| 3D HEX8 | 0.09 ms | 1.24 ms | 13.2× |
| 3D polyhedral | 0.10 ms | 2.17 ms | 22.0× |

Sub-millisecond on both sides — the cost is in **prepare**, not **apply**.

---

## Polyhedral remap — a first real gap

![Polyhedral remap scaling](compare_medcoupling_files/compare_medcoupling_32_0.png){width=88%}

> 101× faster at 400 cells on this test, and the gap grows with size — fields still match to ~1e-14.

---

## Every ratio at a glance

![Speedup ratios per operation](compare_medcoupling_files/compare_medcoupling_39_0.png){width=88%}

> > 1 ⇒ mefikit faster. **9 of 10 workloads** go mefikit's way.

---

## Feature comparison

| Operation | medcoupling | mefikit |
|---|---|---|
| Structured grid | `CMesh` + `buildUnstructured()` | `build_cmesh(*axes)` |
| Per-cell measure | `getMeasureField()` + plumbing | `mf.M` — symbolic |
| Faces / boundary | `buildDescendingConnectivity`, `buildBoundaryMesh` | `descend()`, `boundaries()` |
| Merge nodes | `mergeNodes()` | `merge_nodes()` |
| 2D overlay / imprint | `Intersect2DMeshes()` | `overlay()` |
| P0/P0 conservative remap | `MEDCouplingRemapper.prepare("P0P0")` | `ConservativeP0` |
| Meshless transfers | lower-level / manual | `ConstantPiecewise`, `MovingLeastSquares` |
| File formats | MED, VTK, ENSIGHT, ... | med, vtk/vtu, vtkhdf, cgns, json, yaml |
| Core | C++ (Python bindings) | Rust (first-class Python) |

---

## What mefikit offers today

- A **concise API** for common mesh and field operations.
- Strong speed on **some** operations — polyhedral remap, overlay.
- Polyhedral meshes handled from the start.

---

## Where medcoupling stays ahead

- **Breadth**: spline remapping, advanced field machinery, big ecosystem.
- **Proven** in many production codes over decades.
- mefikit still has a lot to implement before reaching that level.

---

## Conclusion: step by step

- mefikit is faster on several operations — real, but **partial**.
- Many features remain to be written; the API is still evolving.
- medcoupling lives in many other codes; adopting mefikit has **not started**.
- The long-term aim is to go further — built on medcoupling's strengths, one step at a time.

---

## Try it

- Full side-by-side notebook: `docs/python_examples/compare_medcoupling.ipynb`.
- Same workloads live in `tests/bench_vs_medcoupling.py`.
- [Roadmap](https://github.com/asonolet/mefikit/blob/master/ROADMAP.md) and the other notebooks of this book.
