# `mefikit` Mesh Format

The **`mefikit` format** defines a compact, extensible, and expressive way to
represent unstructured meshes composed of heterogeneous elements — including
vertices, edges, faces, and volumetric cells. Supported element types include,
but are not limited to:

- **0D:** vertices
- **1D:** segments, polylines
- **2D:** triangles, quadrilaterals, polygons
- **3D:** tetrahedra, hexahedra, prisms, pyramids, polyhedra

In addition, any element can carry fields (scalar, vector, or tensor) that
reside directly on its topological dimension. This enables workflows akin to
VTK field data. Fields must be defined upon all elements of a given topological dimension.
