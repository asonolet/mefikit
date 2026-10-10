# Element Conventions

This chapter defines the node ordering and face orientation conventions used
throughout `mefikit`. These conventions follow the **VTK** convention for
element node ordering and face definitions.

**Note:** TET10 uses MEDFile mid-side node numbering (deferred migration).

## Element types

| Type   | Dim | Nodes | Regularity | Description                           |
| ------ | --- | ----- | ---------- | ------------------------------------- |
| VERTEX | 0D  | 1     | Regular    | Point                                 |
| SEG2   | 1D  | 2     | Regular    | Linear segment                        |
| SEG3   | 1D  | 3     | Regular    | Quadratic segment                     |
| SEG4   | 1D  | 4     | Regular    | Cubic segment                         |
| TRI3   | 2D  | 3     | Regular    | Linear triangle                       |
| TRI6   | 2D  | 6     | Regular    | Quadratic triangle                    |
| TRI7   | 2D  | 7     | Regular    | Quadratic triangle + centroid         |
| QUAD4  | 2D  | 4     | Regular    | Linear quadrilateral                  |
| QUAD8  | 2D  | 8     | Regular    | Quadratic quadrilateral (serendipity) |
| QUAD9  | 2D  | 9     | Regular    | Biquadratic quadrilateral             |
| TET4   | 3D  | 4     | Regular    | Linear tetrahedron                    |
| TET10  | 3D  | 10    | Regular    | Quadratic tetrahedron                 |
| HEX8   | 3D  | 8     | Regular    | Linear hexahedron                     |
| HEX21  | 3D  | 21    | Regular    | Tricubic hexahedron                   |
| SPLINE | 1D  | var.  | Poly       | Polyline                              |
| PGON   | 2D  | var.  | Poly       | Polygon                               |
| PHED   | 3D  | var.  | Poly       | Polyhedron                            |

## Poly element representation

Poly elements (SPLINE, PGON, PHED) store their connectivity as a flat array of
node indices. **Faces are delimited by `usize::MAX` sentinel values.**

- **PGON**: nodes form a single face, no sentinel needed.
  `[n0, n1, n2, n3]` — a 4-node polygon.

- **PHED**: each face is a closed polygon (PGON), separated by sentinels.

  ```
  [f0_n0, f0_n1, f0_n2, MAX, f1_n0, f1_n1, f1_n2, f1_n3, MAX, ...]
  ```

  The last face has no trailing sentinel.

## Face orientation convention

All subentities (edges of 2D elements, faces of 3D elements) are defined with
**consistent counter-clockwise (CCW) winding when viewed from outside the
element**. This ensures that:

- Each outward-facing normal follows the right-hand rule.
- Shared lower-dimensional entities are traversed in **opposite directions**
  by adjacent higher-dimensional elements.

The subentity definitions below are the canonical reference for node orderings.

## 2D elements

### TRI3 / TRI6 / TRI7 — Triangle

Nodes 0, 1, 2 are the three vertices. Edges follow CCW winding:

```
        2
       / \
      /   \
     /     \
    0-------1
```

| Edge | Nodes  | Description |
| ---- | ------ | ----------- |
| 0    | [0, 1] | Edge 0→1    |
| 1    | [1, 2] | Edge 1→2    |
| 2    | [2, 0] | Edge 2→0    |

For **TRI6/TRI7**, mid-side nodes are placed as: node 3 on edge 01, node 4 on
edge 12, node 5 on edge 20. TRI7 additionally has a centroid node (node 6).

### QUAD4 / QUAD8 / QUAD9 — Quadrilateral

Nodes 0–3 are the four vertices, numbered counter-clockwise:

```
    3-------2
    |       |
    |       |
    0-------1
```

| Edge | Nodes  | Description |
| ---- | ------ | ----------- |
| 0    | [0, 1] | Bottom      |
| 1    | [1, 2] | Right       |
| 2    | [2, 3] | Top         |
| 3    | [3, 0] | Left        |

## 3D elements

### TET4 — Tetrahedron

Nodes 0–3 are the four vertices. The tetrahedron is defined by its four
triangular faces (VTK convention):

```
          3
         /|\
        / | \
       /  |  \
      /   |   \
     /    |    \
    2-----+-----1
     \    |    /
      \   |   /
       \  |  /
        \ | /
         \|/
          0
```

| Face | Nodes     | Description            |
| ---- | --------- | ---------------------- |
| 0    | [0, 1, 3] | Opposite node 2        |
| 1    | [1, 2, 3] | Opposite node 0        |
| 2    | [2, 0, 3] | Opposite node 1        |
| 3    | [0, 2, 1] | Base (opposite node 3) |

Each face lists the three nodes that do **not** include the opposite vertex,
in CCW order when viewed from outside the element.

For **TET10**, the vertex convention is: nodes 0–3 vertices, mid-side nodes
4 (edge 01), 5 (edge 12), 6 (edge 02), 7 (edge 03), 8 (edge 13), 9 (edge 23).

### HEX8 — Hexahedron

Nodes 0–7 are the eight vertices of a unit cube, numbered following the
**VTK** convention:

```
        7---------6
       /|        /|
      / |       / |
     /  |      /  |
    4---------5   |
    |   3-----|---2
    |  /      |  /
    | /       | /
    |/        |/
    0---------1
```

Bottom face: nodes 0, 1, 2, 3 (CCW viewed from below).
Top face: nodes 4, 5, 6, 7 (CCW viewed from above).
Node 4 is directly above node 0, node 5 above node 1, etc.

| Face | Nodes        | Description    |
| ---- | ------------ | -------------- |
| 0    | [0, 3, 2, 1] | Bottom (z = 0) |
| 1    | [4, 5, 6, 7] | Top (z = 1)    |
| 2    | [0, 1, 5, 4] | Front (y = 0)  |
| 3    | [2, 3, 7, 6] | Back (y = 1)   |
| 4    | [1, 2, 6, 5] | Right (x = 1)  |
| 5    | [3, 0, 4, 7] | Left (x = 0)   |

All faces are wound CCW when viewed from outside the element.

**Differences with MEDFile:** MEDFile HEX8 has a different node numbering:
MED node 0 is top-left-front, while VTK node 0 is bottom-left-front. The
MED→VTK node permutation is `[4,5,6,7,0,1,2,3]` (self-inverse). Do not mix VTK
and MEDFile conventions

For **HEX21**, the vertex convention is: nodes 0–7 vertices, mid-side nodes
8 (edge 01), 9 (edge 12), 10 (edge 23), 11 (edge 30), 12 (edge 45),
13 (edge 56), 14 (edge 67), 15 (edge 74), 16 (edge 04), 17 (edge 15),
18 (edge 26), 19 (edge 37).

## PHED — Polyhedron

A polyhedron is stored as a sequence of polygonal faces, each wound CCW when
viewed from outside the element. Faces are delimited by `usize::MAX` sentinels
in the flat connectivity array.

The constraints for a valid PHED are:

1. **Each face is a closed polygon** — its nodes form a simple, non-self-intersecting loop.
2. **Consistent orientation** — every edge shared by exactly two faces is
   traversed in opposite directions by those two faces. Equivalently, the
   outward normal of every face follows the right-hand rule.
3. **Closed surface** — the collection of faces forms a topologically closed
   manifold (no dangling edges).
