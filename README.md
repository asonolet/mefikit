# MeFiKit

*Meshes and Fields Kit* is a library implementing:

- a convenient mesh format (mesh with data) with:
    - different kind of elements in the same mesh
    - named fields of double over any element
    - params (named double) attached to element kinds
    - groups and families (named group of element) attached to any element
- various input output formats
    - medfile
    - cgns
- base structured / extruded / unstructured mesh construction abilities
- descending meshes computation
- aggregation of meshes
- mesh cracks
- Intersection of meshes
- Gluing of meshes with non conform common faces
- terahedization
- measuring (volume)
- AMR abilities ?

The "convenient mesh format" will be close to the file storage format for
performance and simplicity reasons, **unlike** medcoupling core format. This
alone is a good argument in favour of the reimplementation of the medcoupling
library (let alone the huge part of the medcoupling lib which is suboptimal,
fragile and bogus).

In mefikit those features are supposed to be performant and implemented using
a clear necessary and minimal interface. This allows a better maintainability
of the library.

The library is structured the following way:

- mefikit_core
    - umesh
    - element_block
    - element
    - connectivity
- io
    - med
    - cgns
- tools
    - intersecter
    - cracker
    - tetrahedrizer
    - ...

## Advantages over the current medcoupling

- performance
- clarity (medcoupling ~= medfile, but mefikit != medfile)
- rust pilot project in the DM2S
- ease of use (one main clear data structure and not many, many algorithms)
