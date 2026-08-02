# CHANGELOG

## Release v0.1.2

### Fix

- wheel does not use cpu-native instructions

## Unreleased

### Add

- `UMesh::overlay` / `OverlayOperation` (IMPRINT, UNION, INTERSECTION, DIFFERENCE, SYMMETRIC_DIFFERENCE) for 2D mesh overlay, exposed in Python as `UMesh.overlay`

### Change

- `intersect_2d2d` is now private; use `mesh1.overlay(mesh2, OverlayOperation::Imprint)` instead

### Fix

- out-of-bounds index in `in_polygon_stable` when the closest vertex is the first one
