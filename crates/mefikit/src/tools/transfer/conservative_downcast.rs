//! Cross-dimensional conservative P0 transfer (source dimension greater than target
//! dimension).
//!
//! [`super::conservative_p0::prepare`] dispatches here when the source cells have a higher
//! topological dimension than the target cells. Each target cell is intersected with the source
//! cells and weighted by the measure of the intersection in the **target** dimension:
//!
//! - `3D → 2D`: area of a planar target polygon clipped by a convex source polyhedron;
//! - `3D → 1D` / `2D → 1D`: length of a target segment clipped by a convex source cell;
//! - `→ 0D`: the counting measure, i.e. one target point belongs to exactly one source cell
//!   (on a shared interface the cell with the smallest global index wins).
//!
//! The apply-time sparse product is shared with the same-dimensional path
//! ([`super::operator::TransferOperator`]); this module only builds the `(indices, weights)`
//! rows.
//!
//! # Assumptions
//!
//! - Source and target cells are convex and first-order (`TET4`, `HEX8`, `WEDGE6`, `PHED`,
//!   `TRI3`, `QUAD4`, `PGON`, `SEG2`, `VERTEX`).
//! - 2D cells embedded in 3D space are planar up to a small tolerance; a slightly warped
//!   polygon is handled through its best-fit plane / Newell area.
//! - The target mesh is conforming for `0D` targets: a point is assigned to a single source
//!   cell, even when it lies exactly on an interface.

use std::collections::BTreeMap;

use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;

use super::conservative_p0::source_offsets;
use super::operator::{RowSparse, TransferMethod, TransferOperator};
use crate::element_traits::ElementGeo;
use crate::geometry::{
    HalfSpace, PlaneFrame, Polygon, Segment, clip_polygon, clip_segment, into_ccw2, point_inside,
    signed_area2,
};
use crate::mesh::{Dimension, Element, ElementId, ElementType, UMeshView};
use crate::tools::spatial_index::SpatiallyIndexable;

/// Relative tolerance used to decide on which side of a clipping plane a point lies.
const CLIP_EPS: f64 = 1e-12;
/// Relative tolerance on the distance to a source surface plane for `2D → 1D/0D` transfers.
const PLANE_TOL: f64 = 1e-9;
/// Relative thickness below which a clipped measure is considered zero.
const MEASURE_EPS: f64 = 1e-14;
/// Relative inflation of the BVH query boxes.
const BVH_EPS: f64 = 1e-9;

/// Builds the cross-dimensional conservative P0 operator.
pub(crate) fn prepare(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    tgt_dim: Dimension,
    space: usize,
) -> TransferOperator {
    validate_pair(space, src_dim, tgt_dim);
    validate_source_types(mesh_src, src_dim);
    validate_target_types(mesh_tgt, tgt_dim);

    let (src_offsets, n_src) = source_offsets(mesh_src, src_dim);
    let data = match space {
        2 => build_2d_space(mesh_src, mesh_tgt, src_dim, tgt_dim, &src_offsets),
        3 => build_3d_space(mesh_src, mesh_tgt, src_dim, tgt_dim, &src_offsets),
        _ => unreachable!("validated_dims only accepts 2D and 3D spaces"),
    };

    TransferOperator::build(
        TransferMethod::ConservativeP0,
        src_dim,
        tgt_dim,
        n_src,
        data,
    )
}

/// Checks that the dimension pair is one of the supported downcast combinations.
fn validate_pair(space: usize, src_dim: Dimension, tgt_dim: Dimension) {
    let supported = matches!(
        (space, u8::from(src_dim), u8::from(tgt_dim)),
        (2, 2, 1)
            | (2, 2, 0)
            | (2, 1, 0)
            | (3, 3, 2)
            | (3, 3, 1)
            | (3, 3, 0)
            | (3, 2, 1)
            | (3, 2, 0)
            | (3, 1, 0)
    );
    assert!(
        supported,
        "Conservative P0 does not support a {src_dim:?} source to a {tgt_dim:?} target in a {space}D space"
    );
}

/// Panics on a source element type that the cross-dimensional path cannot handle.
fn validate_source_types(mesh_src: &UMeshView, src_dim: Dimension) {
    for (et, _) in mesh_src.blocks() {
        if et.dimension() == src_dim {
            assert_supported(*et, src_dim, "source");
        }
    }
}

/// Panics on a target element type that the cross-dimensional path cannot handle.
fn validate_target_types(mesh_tgt: &UMeshView, tgt_dim: Dimension) {
    for (et, _) in mesh_tgt.blocks() {
        if et.dimension() == tgt_dim {
            assert_supported(*et, tgt_dim, "target");
        }
    }
}

fn assert_supported(et: ElementType, dim: Dimension, role: &str) {
    let supported = match dim {
        Dimension::D0 => et == ElementType::VERTEX,
        Dimension::D1 => et == ElementType::SEG2,
        Dimension::D2 => matches!(
            et,
            ElementType::TRI3 | ElementType::QUAD4 | ElementType::PGON
        ),
        Dimension::D3 => matches!(
            et,
            ElementType::TET4 | ElementType::HEX8 | ElementType::WEDGE6 | ElementType::PHED
        ),
    };
    assert!(
        supported,
        "{role} element type {et:?} is not supported for a {dim:?} cross-dimensional Conservative P0 transfer"
    );
}

// ---------------------------------------------------------------------------
// Row assembly
// ---------------------------------------------------------------------------

struct OverlapRow {
    measure: f64,
    src_idx: Vec<usize>,
    weights: Vec<f64>,
}

impl OverlapRow {
    fn new(measure: f64) -> Self {
        Self {
            measure,
            src_idx: Vec::new(),
            weights: Vec::new(),
        }
    }

    fn push(&mut self, index: usize, weight: f64) {
        self.src_idx.push(index);
        self.weights.push(weight);
    }
}

fn collect_rows(rows: Vec<OverlapRow>) -> RowSparse {
    let n = rows.len();
    let mut target_measure = nd::Array1::zeros(n);
    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut src_idx = Vec::new();
    let mut weights = Vec::new();
    row_ptr.push(0);
    for (j, row) in rows.into_iter().enumerate() {
        target_measure[j] = row.measure;
        src_idx.extend(row.src_idx);
        weights.extend(row.weights);
        row_ptr.push(src_idx.len());
    }
    RowSparse {
        target_measure,
        row_ptr,
        src_idx,
        weights,
    }
}

/// Returns `true` when a clipped measure is not a numerical artefact.
fn significant(overlap: f64, target_measure: f64) -> bool {
    overlap > MEASURE_EPS * target_measure.max(f64::MIN_POSITIVE)
}

// ---------------------------------------------------------------------------
// 3D-space builders
// ---------------------------------------------------------------------------

fn build_3d_space(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    tgt_dim: Dimension,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    match tgt_dim {
        Dimension::D2 => build_3d_to_2d(mesh_src, mesh_tgt, src_offsets),
        Dimension::D1 => build_3d_to_1d(mesh_src, mesh_tgt, src_dim, src_offsets),
        Dimension::D0 => build_3d_to_0d(mesh_src, mesh_tgt, src_dim, src_offsets),
        _ => unreachable!("the same-dimensional path handles 3D targets"),
    }
}

fn build_3d_to_2d(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    let index = mesh_src.bvh3();
    let mut data = Vec::new();
    for (&tgt_et, block) in mesh_tgt.blocks() {
        if tgt_et.dimension() != Dimension::D2 {
            continue;
        }
        let rows: Vec<OverlapRow> = block
            .par_iter(mesh_tgt.coords())
            .map(|elem| {
                let polygon: Vec<[f64; 3]> = elem.coords3().copied().collect();
                let measure = surface_area3(&polygon);
                let mut row = OverlapRow::new(measure);
                if measure <= 0.0 {
                    return row;
                }

                let [min, max] = elem.bounds3();
                let eps = bvh_epsilon(&min, &max);
                let candidates = index.in_bounds(
                    [min[0] - eps, min[1] - eps, min[2] - eps],
                    [max[0] + eps, max[1] + eps, max[2] + eps],
                );
                for (src_et, ids) in candidates.iter_blocks() {
                    if src_et.dimension() != Dimension::D3 {
                        continue;
                    }
                    let offset = src_offsets[src_et];
                    for &i in ids {
                        let src = mesh_src.element(ElementId::new(*src_et, i));
                        let overlap = volume_surface_overlap(&src, &polygon);
                        if significant(overlap, measure) {
                            row.push(offset + i, overlap);
                        }
                    }
                }
                row
            })
            .collect();
        data.push((tgt_et, collect_rows(rows)));
    }
    data
}

fn build_3d_to_1d(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    let index = mesh_src.bvh3();
    let mut data = Vec::new();
    for (&tgt_et, block) in mesh_tgt.blocks() {
        if tgt_et.dimension() != Dimension::D1 {
            continue;
        }
        let rows: Vec<OverlapRow> = block
            .par_iter(mesh_tgt.coords())
            .map(|elem| {
                let a = *elem.coord3_ref(0);
                let b = *elem.coord3_ref(1);
                let measure = Segment::new(a, b).length();
                let mut row = OverlapRow::new(measure);
                if measure <= 0.0 {
                    return row;
                }

                let [min, max] = elem.bounds3();
                let eps = bvh_epsilon(&min, &max);
                let candidates = index.in_bounds(
                    [min[0] - eps, min[1] - eps, min[2] - eps],
                    [max[0] + eps, max[1] + eps, max[2] + eps],
                );
                for (src_et, ids) in candidates.iter_blocks() {
                    if src_et.dimension() != src_dim {
                        continue;
                    }
                    let offset = src_offsets[src_et];
                    for &i in ids {
                        let src = mesh_src.element(ElementId::new(*src_et, i));
                        let overlap = match src_dim {
                            Dimension::D3 => volume_segment_overlap(&src, a, b),
                            Dimension::D2 => surface_segment_overlap(&src, a, b),
                            _ => unreachable!("downcast sources are 2D or 3D here"),
                        };
                        if significant(overlap, measure) {
                            row.push(offset + i, overlap);
                        }
                    }
                }
                row
            })
            .collect();
        data.push((tgt_et, collect_rows(rows)));
    }
    data
}

fn build_3d_to_0d(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    let index = mesh_src.bvh3();
    let mut data = Vec::new();
    for (&tgt_et, block) in mesh_tgt.blocks() {
        if tgt_et.dimension() != Dimension::D0 {
            continue;
        }
        let rows: Vec<OverlapRow> = block
            .par_iter(mesh_tgt.coords())
            .map(|elem| {
                let point = *elem.coord3_ref(0);
                let mut row = OverlapRow::new(1.0);
                let [min, max] = elem.bounds3();
                let eps = bvh_epsilon(&min, &max);
                let candidates = index.in_bounds(
                    [min[0] - eps, min[1] - eps, min[2] - eps],
                    [max[0] + eps, max[1] + eps, max[2] + eps],
                );

                let mut best: Option<(usize, bool)> = None;
                for (src_et, ids) in candidates.iter_blocks() {
                    if src_et.dimension() != src_dim {
                        continue;
                    }
                    let offset = src_offsets[src_et];
                    for &i in ids {
                        let src = mesh_src.element(ElementId::new(*src_et, i));
                        let containment = match src_dim {
                            Dimension::D3 => volume_point_containment(&src, &point),
                            Dimension::D2 => surface_point_containment(&src, &point),
                            Dimension::D1 => segment_point_containment3(&src, &point),
                            _ => None,
                        };
                        if let Some(strict) = containment {
                            update_best_source(&mut best, offset + i, strict);
                        }
                    }
                }
                if let Some((index, _)) = best {
                    row.push(index, 1.0);
                }
                row
            })
            .collect();
        data.push((tgt_et, collect_rows(rows)));
    }
    data
}

// ---------------------------------------------------------------------------
// 2D-space builders
// ---------------------------------------------------------------------------

fn build_2d_space(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    tgt_dim: Dimension,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    match tgt_dim {
        Dimension::D1 => build_2d_to_1d(mesh_src, mesh_tgt, src_offsets),
        Dimension::D0 => build_2d_to_0d(mesh_src, mesh_tgt, src_dim, src_offsets),
        _ => unreachable!("the same-dimensional path handles 2D targets"),
    }
}

fn build_2d_to_1d(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    let index = mesh_src.bvh2();
    let mut data = Vec::new();
    for (&tgt_et, block) in mesh_tgt.blocks() {
        if tgt_et.dimension() != Dimension::D1 {
            continue;
        }
        let rows: Vec<OverlapRow> = block
            .iter(mesh_tgt.coords())
            .map(|elem| {
                let a = *elem.coord2_ref(0);
                let b = *elem.coord2_ref(1);
                let measure = Segment::new(a, b).length();
                let mut row = OverlapRow::new(measure);
                if measure <= 0.0 {
                    return row;
                }

                let [min, max] = elem.bounds2();
                let eps = bvh_epsilon(&min, &max);
                let candidates =
                    index.in_bounds([min[0] - eps, min[1] - eps], [max[0] + eps, max[1] + eps]);
                for (src_et, ids) in candidates.iter_blocks() {
                    if src_et.dimension() != Dimension::D2 {
                        continue;
                    }
                    let offset = src_offsets[src_et];
                    for &i in ids {
                        let src = mesh_src.element(ElementId::new(*src_et, i));
                        let points: Vec<[f64; 2]> = src.coords2().copied().collect();
                        let Some(planes) = polygon2_half_spaces(&points) else {
                            continue;
                        };
                        let scale = scale2(&points).max(scale2(&[a, b]));
                        let eps = CLIP_EPS * scale;
                        let overlap = clip_segment(a, b, &planes, eps)
                            .map(|(p, q)| distance2(p, q))
                            .unwrap_or(0.0);
                        if significant(overlap, measure) {
                            row.push(offset + i, overlap);
                        }
                    }
                }
                row
            })
            .collect();
        data.push((tgt_et, collect_rows(rows)));
    }
    data
}

fn build_2d_to_0d(
    mesh_src: &UMeshView,
    mesh_tgt: &UMeshView,
    src_dim: Dimension,
    src_offsets: &BTreeMap<ElementType, usize>,
) -> Vec<(ElementType, RowSparse)> {
    let index = mesh_src.bvh2();
    let mut data = Vec::new();
    for (&tgt_et, block) in mesh_tgt.blocks() {
        if tgt_et.dimension() != Dimension::D0 {
            continue;
        }
        let rows: Vec<OverlapRow> = block
            .iter(mesh_tgt.coords())
            .map(|elem| {
                let point = *elem.coord2_ref(0);
                let mut row = OverlapRow::new(1.0);
                let [min, max] = elem.bounds2();
                let eps = bvh_epsilon(&min, &max);
                let candidates =
                    index.in_bounds([min[0] - eps, min[1] - eps], [max[0] + eps, max[1] + eps]);

                let mut best: Option<(usize, bool)> = None;
                for (src_et, ids) in candidates.iter_blocks() {
                    if src_et.dimension() != src_dim {
                        continue;
                    }
                    let offset = src_offsets[src_et];
                    for &i in ids {
                        let src = mesh_src.element(ElementId::new(*src_et, i));
                        let containment = match src_dim {
                            Dimension::D2 => polygon2_point_containment(&src, &point),
                            Dimension::D1 => segment_point_containment2(&src, &point),
                            _ => None,
                        };
                        if let Some(strict) = containment {
                            update_best_source(&mut best, offset + i, strict);
                        }
                    }
                }
                if let Some((index, _)) = best {
                    row.push(index, 1.0);
                }
                row
            })
            .collect();
        data.push((tgt_et, collect_rows(rows)));
    }
    data
}

/// Keeps a single source cell for a target point: a strictly containing cell wins over a
/// boundary cell, and ties are broken by the smallest global source index. This gives a
/// deterministic owner for points exactly on a conforming interface.
fn update_best_source(best: &mut Option<(usize, bool)>, index: usize, strict: bool) {
    let replace = match *best {
        None => true,
        Some((best_index, best_strict)) => {
            (strict && !best_strict) || (strict == best_strict && index < best_index)
        }
    };
    if replace {
        *best = Some((index, strict));
    }
}

// ---------------------------------------------------------------------------
// Source geometry helpers
// ---------------------------------------------------------------------------

/// Half-spaces `n · x <= d` of a convex polyhedron, oriented with its vertex centroid on the
/// interior side. This does not depend on the winding of the input faces.
///
/// Returns the half-spaces and the maximum distance of a face vertex to its best-fit face plane.
/// The latter is used as an extra clipping tolerance so that a target surface lying on a slightly
/// warped source face is not clipped by the planar approximation of that face.
fn polyhedron_half_spaces(polyhedron: &crate::geometry::Polyhedron) -> (Vec<HalfSpace<3>>, f64) {
    let reference = polyhedron.centroid();
    let mut planes = Vec::with_capacity(polyhedron.num_faces());
    let mut warp = 0.0_f64;
    for face in polyhedron.faces() {
        let normal = face.normal();
        if normal == [0.0; 3] {
            continue;
        }
        let Some(first) = face.iter().next() else {
            continue;
        };
        let d = dot3(normal, *first);
        for point in face.iter() {
            warp = warp.max((dot3(normal, *point) - d).abs());
        }
        let side = dot3(normal, reference) - d;
        let plane = HalfSpace { n: normal, d };
        planes.push(if side > 0.0 { plane.flipped() } else { plane });
    }
    (planes, warp)
}

/// Half-planes of a convex 2D polygon, oriented inward. Returns `None` for a degenerate polygon.
fn polygon2_half_spaces(points: &[[f64; 2]]) -> Option<Vec<HalfSpace<2>>> {
    if points.len() < 3 {
        return None;
    }
    let mut points = points.to_vec();
    into_ccw2(&mut points);
    if signed_area2(&points) <= 0.0 {
        return None;
    }
    let n = points.len();
    Some(
        (0..n)
            .map(|i| {
                let a = points[i];
                let b = points[(i + 1) % n];
                // Interior is on the left of the CCW edge; the right normal gives n · x <= d.
                HalfSpace::new([b[1] - a[1], -(b[0] - a[0])], a)
            })
            .collect(),
    )
}

/// A source surface polygon embedded in 3D, represented in its best-fit plane.
struct Surface3 {
    plane: PlaneFrame,
    half_planes: Vec<HalfSpace<2>>,
    scale: f64,
    warp: f64,
}

impl Surface3 {
    fn new(points: &[[f64; 3]]) -> Option<Self> {
        if points.len() < 3 || surface_area3(points) <= 0.0 {
            return None;
        }
        let plane = PlaneFrame::from_points(points);
        let warp = plane.max_deviation(points);
        let mut projected: Vec<[f64; 2]> = points.iter().map(|p| plane.project(p)).collect();
        if signed_area2(&projected) < 0.0 {
            projected.reverse();
        }
        let half_planes = polygon2_half_spaces(&projected)?;
        Some(Self {
            plane,
            half_planes,
            scale: scale3(points),
            warp,
        })
    }

    fn clip_epsilon(&self) -> f64 {
        CLIP_EPS * self.scale
    }

    fn plane_tolerance(&self) -> f64 {
        PLANE_TOL * self.scale + self.warp
    }

    fn segment_overlap(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        if self.plane.max_deviation(&[a, b]) > self.plane_tolerance() {
            return 0.0;
        }
        let pa = self.plane.project(&a);
        let pb = self.plane.project(&b);
        clip_segment(pa, pb, &self.half_planes, self.clip_epsilon())
            .map(|(p, q)| distance2(p, q))
            .unwrap_or(0.0)
    }

    /// Returns `Some(true)` when the point is strictly inside, `Some(false)` when it is on the
    /// boundary, and `None` when it is outside (or off the plane).
    fn point_containment(&self, point: &[f64; 3]) -> Option<bool> {
        if self.plane.max_deviation(std::slice::from_ref(point)) > self.plane_tolerance() {
            return None;
        }
        let projected = self.plane.project(point);
        let eps = self.clip_epsilon();
        if !point_inside(&projected, &self.half_planes, eps) {
            None
        } else {
            Some(
                self.half_planes
                    .iter()
                    .all(|half_space| half_space.distance(&projected) < -eps),
            )
        }
    }
}

fn volume_surface_overlap(src: &Element, polygon: &[[f64; 3]]) -> f64 {
    let polyhedron = src.to_polyhedron();
    if polyhedron.volume() <= 0.0 {
        return 0.0;
    }
    let (half_spaces, warp) = polyhedron_half_spaces(&polyhedron);
    let scale = scale3_iter(polyhedron.iter()).max(scale3(polygon));
    let clipped = clip_polygon(polygon, &half_spaces, CLIP_EPS * scale + warp);
    surface_area3(&clipped)
}

fn volume_segment_overlap(src: &Element, a: [f64; 3], b: [f64; 3]) -> f64 {
    let polyhedron = src.to_polyhedron();
    if polyhedron.volume() <= 0.0 {
        return 0.0;
    }
    let (half_spaces, warp) = polyhedron_half_spaces(&polyhedron);
    let scale = scale3_iter(polyhedron.iter()).max(scale3(&[a, b]));
    clip_segment(a, b, &half_spaces, CLIP_EPS * scale + warp)
        .map(|(p, q)| distance3(p, q))
        .unwrap_or(0.0)
}

fn volume_point_containment(src: &Element, point: &[f64; 3]) -> Option<bool> {
    let polyhedron = src.to_polyhedron();
    if polyhedron.volume() <= 0.0 {
        return None;
    }
    let (half_spaces, warp) = polyhedron_half_spaces(&polyhedron);
    let scale = scale3_iter(polyhedron.iter()).max(scale3(std::slice::from_ref(point)));
    let eps = CLIP_EPS * scale + warp;
    if !point_inside(point, &half_spaces, eps) {
        None
    } else {
        Some(
            half_spaces
                .iter()
                .all(|half_space| half_space.distance(point) < -eps),
        )
    }
}

fn surface_segment_overlap(src: &Element, a: [f64; 3], b: [f64; 3]) -> f64 {
    let points: Vec<[f64; 3]> = src.coords3().copied().collect();
    Surface3::new(&points).map_or(0.0, |surface| surface.segment_overlap(a, b))
}

fn surface_point_containment(src: &Element, point: &[f64; 3]) -> Option<bool> {
    let points: Vec<[f64; 3]> = src.coords3().copied().collect();
    Surface3::new(&points).and_then(|surface| surface.point_containment(point))
}

fn polygon2_point_containment(src: &Element, point: &[f64; 2]) -> Option<bool> {
    let points: Vec<[f64; 2]> = src.coords2().copied().collect();
    let planes = polygon2_half_spaces(&points)?;
    let scale = scale2(&points).max(scale2(std::slice::from_ref(point)));
    let eps = CLIP_EPS * scale;
    if !point_inside(point, &planes, eps) {
        None
    } else {
        Some(planes.iter().all(|plane| plane.distance(point) < -eps))
    }
}

fn segment_point_containment2(src: &Element, point: &[f64; 2]) -> Option<bool> {
    let a = *src.coord2_ref(0);
    let b = *src.coord2_ref(1);
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [point[0] - a[0], point[1] - a[1]];
    let norm2 = dot2(ab, ab);
    if norm2 == 0.0 {
        return None;
    }
    let t = dot2(ap, ab) / norm2;
    let closest = [a[0] + t * ab[0], a[1] + t * ab[1]];
    let scale = scale2(&[a, b, *point]);
    let eps = CLIP_EPS * scale;
    if t < -CLIP_EPS || t > 1.0 + CLIP_EPS || distance2(closest, *point) > eps {
        None
    } else {
        Some(t > CLIP_EPS && t < 1.0 - CLIP_EPS)
    }
}

fn segment_point_containment3(src: &Element, point: &[f64; 3]) -> Option<bool> {
    let a = *src.coord3_ref(0);
    let b = *src.coord3_ref(1);
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ap = [point[0] - a[0], point[1] - a[1], point[2] - a[2]];
    let norm2 = dot3(ab, ab);
    if norm2 == 0.0 {
        return None;
    }
    let t = dot3(ap, ab) / norm2;
    let closest = [a[0] + t * ab[0], a[1] + t * ab[1], a[2] + t * ab[2]];
    let scale = scale3(&[a, b, *point]);
    let eps = CLIP_EPS * scale;
    if t < -CLIP_EPS || t > 1.0 + CLIP_EPS || distance3(closest, *point) > eps {
        None
    } else {
        Some(t > CLIP_EPS && t < 1.0 - CLIP_EPS)
    }
}

// ---------------------------------------------------------------------------
// Small numeric helpers
// ---------------------------------------------------------------------------

fn surface_area3(points: &[[f64; 3]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    Polygon::<3>::unknown(points.iter().copied()).area()
}

fn scale3(points: &[[f64; 3]]) -> f64 {
    scale3_iter(points.iter())
}

fn scale3_iter<'a>(points: impl Iterator<Item = &'a [f64; 3]>) -> f64 {
    points
        .flat_map(|point| point.iter())
        .fold(1.0_f64, |scale, value| scale.max(value.abs()))
}

fn scale2(points: &[[f64; 2]]) -> f64 {
    points
        .iter()
        .flat_map(|point| point.iter())
        .fold(1.0_f64, |scale, value| scale.max(value.abs()))
}

fn bvh_epsilon<const D: usize>(min: &[f64; D], max: &[f64; D]) -> f64 {
    let scale = min
        .iter()
        .chain(max.iter())
        .fold(1.0_f64, |scale, value| scale.max(value.abs()));
    BVH_EPS * scale
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn distance2(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1]).hypot(a[2] - b[2])
}

#[cfg(test)]
mod tests;
