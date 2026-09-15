# Python Examples

Following are some python examples of what `mefikit` can do from python.

The Python API is first class access to `mefikit`. As `mefikit` is in early
development phase, ideas concerning the python api are welcome !

Some insights on how to use the current python lib:

- Use autocompletion with the tool you like, type hints are provided! A missing
  type hint is a bug, please report it.
- Only high level whole mesh operations are supported through python. For finer
  grain ops either reach me or implement it in rust consuming the `mefikit` rust
  crate.
- Use lazy expressions wherever possible, they are fast, reusable, expressive
  and less error prone than manual indexing. They are inspired from `polars`, a
  DataFrame python-rust lib.
- Visualize with `pyvista`. It lacks type hints and the syntax may not feel
  familiar but the overall experience is really better than anything else.
