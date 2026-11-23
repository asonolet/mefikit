# TODO

Remaining items from the `dev` review. None of these blocks the current
best-effort quadratic/concave policy; they are follow-ups.

## Deferred by decision

- **Epsilon behaviour** in `is_conform` and the cross-dimensional transfer is
  heavily tested but not translation/scale invariant. It is intentionally left
  untouched for now; do not change it without adding the corresponding
  translation/scale regression tests.
- **Concave input** is assumed convex and is not checked by most tools. This is
  documented globally in `docs/src/reference.md`; do not add per-tool convexity
  checks or reject concave cells.

## Correctness and edge cases

- `is_conform`: concave `PHED` volume intersections still use convex clipping;
  either decompose concave cells or document the limitation more explicitly.
- `is_conform`: 1D/0D overlaps are not checked; `NoVolumeCells` is also returned
  for empty 2D meshes. Consider renaming and a conformance-specific error type
  instead of `StitchError`.
- `slice`: a 2D surface embedded in 3D currently returns an empty slice; either
  support the plane/line intersection or reject it explicitly.
- `slice`: an empty slice can still panic in meshless transfers (empty target).
  Guard the callers or document the requirement.
- `slice`: concave 3D/`PHED` faces are not checked; non-closed loops are
  silently dropped; annulus/hole sections are emitted as separate filled
  polygons.
- `slice`: the internal tolerance is not configurable; families/groups are not
  propagated consistently; curved quadratic cells are only linearized.
- `groups`: empty group assignment currently deletes/never creates the group.
  Decide whether empty groups are stored or whether assignment means deletion,
  then document and test it.
- `groups`: the bulk additive `add_to_groups` API was removed; provide a
  single-rebuild replacement if bulk additions are common.
- `groups`: CGNS `ZoneSubRegion_t` groups and `EdgeCenter` fields are not
  visible in ParaView; consider `Family_t`/`ZoneBC_t` or document the
  interoperability limit.
- `io`: CGNS/`.vtkhdf` writers drop mixed-dimension fields by design; document
  the read and write behaviour for third-party files.
- `io`: make saves atomic (write a temp file, then rename) and explicitly flush
  the zip writer; map `Encode`/write failures to `OSError` rather than
  `ValueError` where appropriate.
- `io`: CGNS reserved node names can still collide with generated group/field
  nodes. Surface a clear error before creating anything.
- `transfer`: the saved SciPy matrix is name-sorted by element type, while field
  arrays flatten in enum order. Document the column order and/or export a helper
  that applies the matrix to field dictionaries.
- `transfer`: `load` trusts `shape.npy` before allocating the source remap;
  cap sizes against the actual arrays and use `try_reserve`.
- `transfer`: store the space dimension in the v1 metadata; validate
  method-specific construction invariants such as ConstantPiecewise
  full-dimensionality.
- `transfer`: `ElementType::name()` is documented as the serialized name but
  serde still uses variant names. Unify the representation or fix the doc.
- `mpi`: document that `eval`, `__call__`, `apply_update`, and `distributed`
  are collective operations.
- `mpi`: 0D point ties are resolved in local owner-rank order, not global source
  order; preserve global order or document the distributed tie rule.
- `mpi`: fixed tags 4101-4103 can collide with other traffic on the same
  communicator; use per-operator tags or a duplicated communicator.
- `mpi`: add a distributed `PHED` regression test now that `aggregate` preserves
  the face separators and `validate_structure` accepts them.
- `aggregate`: only the PHED sentinel is special-cased; check other poly types
  if new separators are introduced.

## Tests, tooling, and docs

- Add a CI job that runs `pytest` and `mpiexec -n 4 pytest
  tests/test_transfer_mpi.py`; plain `pytest` only exercises `COMM_WORLD` size 1.
- Document/run `cargo test --all-features` (or `--workspace`): the documented
  `cargo test -p mefikit` does not compile the IO tests.
- Remove or use `tests/data/overlapping_wedges_tet.med`; move
  `tests/conformize/main.py` to an example or delete it.
- Regenerate `uv.lock` with the repository uv version; the MPI commit downgraded
  its revision (5 -> 3) and produced marker churn without version changes.
- Stubs: the remaining `_FieldInput` aliases on reductions should be checked
  against the runtime; add a docstring/final marker to `ConformanceChecks`.
- Docs: qualify “Conserves: yes” for downcast transfers (only same-dimension
  transfers conserve the source integral), and restore the `Gradient.apply_update`
  row removed from the reference table.
- `slice`: add a MedCoupling cross-check for length/area if its slicing API is
  usable from Python.
