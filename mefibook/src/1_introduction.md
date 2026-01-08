# Introduction

![Mefikit logo](./logo/mefikit_logo_v2.png)

## About `mefikit`

**`mefikit`** — short for **Mesh and Field Kit** — is a comprehensive library for
generating, manipulating, and analyzing unstructured meshes together with
associated scalar, vector, and tensor fields. Its goal is to provide a unified
in-memory representation of meshes and fields, along with a set of robust,
efficient tools that support numerical simulation workflows in research,
engineering, and scientific computing.

`mefikit` focuses on three core principles:

1. **A flexible mesh model** capable of representing mixed-element unstructured
   meshes.
2. **A consistent field and group architecture** that attaches data to mesh
   entities across dimensions.
3. **Zero-copy, iterator-based access patterns** to efficiently navigate and
   process large meshes.

This design positions `mefikit` as a modern core library for algorithm
development.

## `mefikit` provides

- `umesh` the unstructured mesh container supporting mixed element types,
  fields, and groups.
- `io` modules for reading and writing meshes in various formats
  (e.g., VTK, serde_json, serde_yaml).
- `topology` tools for analyzing mesh connectivity, computing submeshes,
  neighbours, domain frontier, etc.
- `geometry` tools for computing element measures, centroids, etc.
- `Selector` utilities for querying and filtering mesh elements
  based on geometric or topological criteria.

## Definitions

### **Element**

An **element** is an abstract mesh entity specified by:

- **its type** (e.g., triangle, hexahedron, polygon, polyhedron)
- **its connectivity** (indices referencing the global coordinate array)

In `mefikit`, elements are **ephemeral, zero-copy views** constructed on the fly
when iterating through a mesh block. They do not own memory; instead they refer
to underlying index and coordinate buffers. This behavior mirrors VTK’s “cell
iterator” pattern while improving cache locality and reducing overhead.

It is only accessible through the rust API. The python bindings do not expose
elements directly because python is only intended to be used for high-level
scripting. Do not fear of going to the rust side for performance critical code,
`mefikit` was designed to be simple to use for rust newcommers.

### **Topological Dimension**

The **topological dimension** of an element is the dimension of the
mathematical object it represents:

- 0D → vertices
- 1D → edges / segments
- 2D → faces
- 3D → volumes

This categorization is independent of the embedding space and is central for
field association and mesh validity checks.

### **Spatial Dimension**

The **spatial dimension** is the dimension of the coordinate system in which
the mesh is embedded, typically **2D or 3D**. All element coordinates must be
consistent with this spatial dimension. For example, one may have:

- 2D elements in 2D (pure surface mesh)
- 2D elements in 3D (surface embedded in 3D)
- 3D volumetric elements in 3D

`mefikit` does not assume a fixed coordinate system beyond this requirement.

### **Mesh**

A **mesh** in `mefikit` is a container that holds:

- the global node coordinate array,
- a set of **blocks**, each containing elements of a given type,
- **fields** (scalar/vector/tensor) attached to elements of a given topological dimension,
- **groups**, i.e., labeled subsets of elements for boundary conditions,
  materials, or post-processing.

Properties:

- You may attach **any number of fields or groups**.
- **Groups** may overlap arbitrarily.
- A **field must cover all elements** of its associated topological dimension
  (similar to Gmsh’s “node data” or “element data” consistency).

If these constraints seem restrictive, `mefikit` encourages using **multiple
meshes** (e.g., one per partition, or one per physical domain) to better match
specialized workflows.

### Topologically Valid Mesh

A **topologically valid mesh** in `mefikit` satisfies the following conditions:

1. **No overlapping higher-dimensional elements**
   Two 3D elements may share faces or edges, but they must not occupy the same
   region of space. (Duplicate faces in a hex-dominant mesh are allowed and
   interpreted in context.)

2. **Lower-dimensional entities derive consistently from higher-dimensional ones**
   - All edges must be edges of some face or volume.
   - All faces must be boundaries of some volume.
   - No “floating” or orphaned lower-dimension elements are allowed.
     This is consistent with the expectations of many finite-element and
     finite-volume codes.

3. **Non-degenerate geometry**
   Elements must not be geometrically degenerate:
   - triangles must have non-zero area,
   - tetrahedra must have non-zero volume,
   - polygons must be well-defined and non-self-intersecting, etc.
     These checks are similar to the geometric validation steps performed by
     MeshGems and various meshing libraries.
