# Topological basic operations

## Subgraph and neighbours computation

The subgraph computation is done supposing that the mesh is topologically
valid. The subgraph of a volume mesh is defined by a set of either faces, edges
or vertices elements depending on the subgraph relative dimension.

The subgraph computation and the neighbours computation are tightly linked. The
neighbours are defined as the elements that share a common lower dimensional
element. For example, two volumes are neighbours if they share a common face,
two faces are neighbours if they share a common edge, and so on.

In `mefikit`, the neighbours computation builds an adjacency graph, having as
nodes the elements of the mesh and as edges the shared lower dimensional
elements. Then, the neighbours of an element can be found by looking at its
adjacent nodes in the graph.

## Connectivity equivalence

There is different ways to represent the connectivity of an element. Here are
the different class of equivalence:

- exact representation equality: two elements are equivalent if their
  connectivity is exactly the same, including the order of the nodes.
- rotational equivalence: two elements are equivalent if their connectivity can
  be made the same by rotating the order of the nodes. The topological shape is
  strictly equivalent under all operations, preserving the measure orientation.
- chiral equivalence: two elements are equivalent if their connectivity can be made
  the same by reversing the order of the nodes and rotating the nodes. This
  equivalence does not preserve the measure orientation (if it is positive or
  negative).
- node set equivalence: two elements are equivalent if they have the same set
  of nodes, regardless of the order. This equivalence is theoretical only, as
  it does not preserve the shape of the element. But it is very useful to detect
  duplicate elements in a mesh.
