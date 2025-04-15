# MeFiKit

*Meshes and Fields Kit* is a library implementing:

- a convenient meshfields format (mesh with data)
- medfile input/output
- cgns input/output
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
    - mesh
    - fields
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
