//! Stitch several 3D volume meshes into a single conforming polyhedral mesh.
//!
//! Given two or more volume meshes lying in a common 3D space, [`stitch`] refines the boundary
//! faces that two or more meshes have in common (up to `tol`) so that they become mutually
//! conformal, shares the resulting interface nodes, and reassembles everything as one
//! polyhedral (`PHED`) mesh.
//!
//! # Algorithm
//!
//! 1. Every volume cell is converted to its polyhedral equivalent and its faces are counted.
//!    Faces owned by a single cell are the mesh boundary; they are collected into one global
//!    skin.
//! 2. Boundary faces are clustered into coplanar patches and the coincident patches of distinct
//!    meshes are grouped into planar regions (patches with parallel *or antiparallel* normals
//!    match, since the two sides of a volume interface wind oppositely).
//! 3. Each region is refined independently: the faces of every source mesh are projected on a
//!    common plane and the 2D overlay machinery is applied iteratively so that every side is cut
//!    by every other side's edges. Child pieces keep a reference to the original face they come
//!    from.
//! 4. All input coordinates and produced interface vertices are welded globally (within `tol`),
//!    which makes coincident interface nodes shared by every mesh.
//! 5. Each volume cell is rebuilt as a `PHED` whose boundary faces are replaced by their refined
//!    children; interior faces are kept as-is. Families are relabeled per input mesh so domains
//!    stay distinguishable. Fields and groups are **not** propagated.
//!
//! Only first-order volume cells (`TET4`, `HEX8`) and already-poly `PHED` cells are supported;
//! quadratic cells (`TET10`, `HEX21`) are rejected. Coincident interfaces must be piecewise
//! planar within `tol`.
//!
//! # Known limitations
//!
//! - Interfaces are refined with the 2D overlay machinery shared with [`overlay_surfaces`], whose
//!   cell walker (`walk_dart_map` in `crates/mefikit/src/element_traits/cut.rs`) is not yet
//!   complete: it panics when a cutting segment enters a cell without cutting it. Triangular
//!   interface faces (`TET4` elements, or `PHED` cells with triangular faces) can trigger it.
//! - Fields and groups are not propagated to the output.
//! - Overlapping volumes are not detected: the result may contain overlapping cells.

mod diagnose;
#[cfg(test)]
mod tests;

pub use diagnose::{ConformanceIssue, ConformanceReport, is_conform};

use std::collections::BTreeMap;
use std::collections::hash_map::Entry;
use std::fmt;

use ndarray as nd;
use rustc_hash::{FxBuildHasher, FxHashMap};
use smallvec::SmallVec;

use super::surface::{
    FaceData, PARALLEL_NORMAL_COS_EPS, Patch, bboxes_overlap, cluster_coplanar_patches,
    collect_surface_faces, plane_distance,
};
use super::{compute_overlay, cut_cells_all, merge_on_reference_coords};
use crate::element_traits::ElementTopo;
use crate::geometry::{PlaneFrame, Polygon, bounds_iter, newell_normal3, signed_area2};
use crate::mesh::{Dimension, ElementId, ElementIds, ElementLike, ElementType, UMesh, UMeshView};
use crate::tools::Descendable;
use crate::tools::spatial_index::{SpIdx3, SpatiallyIndexable};

/// Tolerances and errors of [`stitch`].
#[derive(Clone, Debug, PartialEq)]
pub enum StitchError {
    /// At least two meshes are required.
    NotEnoughMeshes {
        /// Number of meshes actually given.
        found: usize,
    },
    /// The tolerance must be finite and non-negative.
    InvalidTolerance {
        /// The offending tolerance.
        tol: f64,
    },
    /// Meshes must be embedded in 3D space.
    InvalidSpaceDimension {
        /// Mesh ordinal.
        mesh: usize,
        /// Its spatial dimension.
        found: usize,
    },
    /// Only volume meshes (made exclusively of `D3` cells) are supported.
    NonVolumeElement {
        /// Mesh ordinal.
        mesh: usize,
        /// The offending element type.
        element_type: ElementType,
    },
    /// Quadratic or otherwise unsupported volume element.
    UnsupportedElementType {
        /// Mesh ordinal.
        mesh: usize,
        /// The offending element type.
        element_type: ElementType,
    },
    /// A region deviates from planarity by more than the tolerance.
    NonPlanarRegion {
        /// Region ordinal.
        region: usize,
        /// Measured maximum deviation.
        deviation: f64,
        /// The tolerance used.
        tol: f64,
    },
    /// A produced piece could not be matched back to one of its parent faces.
    UnmatchedPiece {
        /// Region ordinal.
        region: usize,
    },
    /// A produced piece is degenerate and has no usable interior point.
    DegeneratePiece {
        /// Region ordinal.
        region: usize,
    },
    /// A face of the input skin could not be collected.
    Surface(super::surface::SurfaceOverlayError),
    /// An internal operation failed.
    Internal(String),
}

impl fmt::Display for StitchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotEnoughMeshes { found } => {
                write!(f, "stitch requires at least two meshes, found {found}")
            }
            Self::InvalidTolerance { tol } => {
                write!(
                    f,
                    "stitch tolerance must be finite and non-negative, found {tol}"
                )
            }
            Self::InvalidSpaceDimension { mesh, found } => write!(
                f,
                "mesh {mesh} must be embedded in 3d space, found spatial dimension {found}"
            ),
            Self::NonVolumeElement { mesh, element_type } => write!(
                f,
                "mesh {mesh} contains element type {element_type:?} which is not a volume element"
            ),
            Self::UnsupportedElementType { mesh, element_type } => write!(
                f,
                "mesh {mesh} contains unsupported element type {element_type:?}; only TET4, \
                 HEX8 and PHED are supported"
            ),
            Self::NonPlanarRegion {
                region,
                deviation,
                tol,
            } => write!(
                f,
                "region {region} deviates from planarity by {deviation} which exceeds the \
                 tolerance {tol}; coincident interfaces must be piecewise planar"
            ),
            Self::UnmatchedPiece { region } => write!(
                f,
                "could not match a piece of region {region} back to its parent face"
            ),
            Self::DegeneratePiece { region } => {
                write!(f, "region {region} produced a degenerate piece")
            }
            Self::Surface(e) => write!(f, "{e}"),
            Self::Internal(e) => write!(f, "internal stitching error: {e}"),
        }
    }
}

impl std::error::Error for StitchError {}

impl From<super::surface::SurfaceOverlayError> for StitchError {
    fn from(e: super::surface::SurfaceOverlayError) -> Self {
        Self::Surface(e)
    }
}

/// Maps each parent cell id to the ids of the pieces it was split into.
type ParentMap = Vec<(ElementId, Vec<ElementId>)>;

/// Number of nodes a face ring holds inline, chosen to cover every `TET4` and `HEX8` face.
const RING_INLINE: usize = 8;

/// Number of 2D points a projected face ring holds inline, matching [`RING_INLINE`].
const PTS_INLINE: usize = 8;

/// Child face rings in 3D, indexed by the input face (a [`FaceData`] index) they refine.
type ChildrenByFace = Vec<(usize, Vec<Vec<[f64; 3]>>)>;

/// A projected 2D face of a region: input face index, CCW polygon and bounding box.
type ProjectedFace = (usize, Polygon<2>, [[f64; 2]; 2]);

/// One boundary face of one input mesh.
struct SkinFace {
    /// Ordinal of the owning mesh.
    mesh: usize,
    /// Oriented node ring (global coordinates), as it appears in its cell.
    ring: SmallVec<[usize; RING_INLINE]>,
}

/// The cells using a given face, plus the mesh and oriented ring of the first of them.
struct FaceUse {
    /// Every cell using the face, in insertion order; two cells for an interior interface.
    cells: SmallVec<[ElementId; 2]>,
    /// Ordinal of the mesh the first cell belongs to.
    mesh: usize,
    /// Oriented node ring of the first cell, in global node ids.
    ring: SmallVec<[usize; RING_INLINE]>,
}

/// Splits a polyhedral connectivity (faces separated by [`usize::MAX`]) into face rings.
///
/// Rings of at most [`RING_INLINE`] nodes, which covers every face of a `TET4` or `HEX8` cell,
/// are held inline so that walking the faces of a volume mesh allocates nothing.
fn split_phed(conn: &[usize]) -> Vec<SmallVec<[usize; RING_INLINE]>> {
    let mut faces = Vec::new();
    let mut cur: SmallVec<[usize; RING_INLINE]> = SmallVec::new();
    for &n in conn {
        if n == usize::MAX {
            if !cur.is_empty() {
                faces.push(std::mem::take(&mut cur));
            }
        } else {
            cur.push(n);
        }
    }
    if !cur.is_empty() {
        faces.push(cur);
    }
    faces
}

/// Element type of a boundary face with `n` nodes.
fn face_etype(n: usize) -> ElementType {
    match n {
        3 => ElementType::TRI3,
        4 => ElementType::QUAD4,
        _ => ElementType::PGON,
    }
}

/// Joins a set of face rings into a `PHED` connectivity using [`usize::MAX`] separators.
fn join_phed(rings: &[Vec<usize>]) -> Vec<usize> {
    let mut conn = Vec::new();
    for (i, ring) in rings.iter().enumerate() {
        if i > 0 {
            conn.push(usize::MAX);
        }
        conn.extend_from_slice(ring);
    }
    conn
}

/// Returns `true` when the two patches are coplanar and overlap, accepting antiparallel normals.
fn patches_coincident_abs(p: &Patch, q: &Patch, tol: f64) -> bool {
    let n1 = p.frame.normal();
    let n2 = q.frame.normal();
    let dot = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2];
    dot.abs() >= 1.0 - PARALLEL_NORMAL_COS_EPS
        && plane_distance(p, q) <= tol
        && bboxes_overlap(p.bounds, q.bounds, tol)
}

/// Groups coincident patches of distinct meshes into planar regions.
///
/// Returns, for every region holding at least two meshes, the list of its patch indices.
fn group_regions_k(
    patches: &[Patch],
    patch_mesh: &[usize],
    face_patch: &FxHashMap<ElementId, usize>,
    bvh: &SpIdx3,
    tol: f64,
) -> Vec<Vec<usize>> {
    let np = patches.len();
    let mut parent: Vec<usize> = (0..np).collect();
    fn root_compress(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        let mut y = x;
        while parent[y] != y {
            let next = parent[y];
            parent[y] = r;
            y = next;
        }
        r
    }

    let pad = tol.max(f64::EPSILON);
    for (pi, p) in patches.iter().enumerate() {
        let min = [
            p.bounds[0][0] - pad,
            p.bounds[0][1] - pad,
            p.bounds[0][2] - pad,
        ];
        let max = [
            p.bounds[1][0] + pad,
            p.bounds[1][1] + pad,
            p.bounds[1][2] + pad,
        ];
        for eid in bvh.in_bounds(min, max).iter() {
            let Some(&pj) = face_patch.get(&eid) else {
                continue;
            };
            if pj == pi || patch_mesh[pj] == patch_mesh[pi] {
                continue;
            }
            if patches_coincident_abs(p, &patches[pj], tol) {
                let ri = root_compress(&mut parent, pi);
                let rj = root_compress(&mut parent, pj);
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }

    let mut members: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for pi in 0..np {
        let r = root_compress(&mut parent, pi);
        members.entry(r).or_default().push(pi);
    }
    let mut regions = Vec::new();
    for (_r, ps) in members {
        let mut meshes: Vec<usize> = ps.iter().map(|&p| patch_mesh[p]).collect();
        meshes.sort_unstable();
        meshes.dedup();
        if meshes.len() >= 2 {
            regions.push(ps);
        }
    }
    regions
}

fn point_in_bbox(p: [f64; 2], bb: [[f64; 2]; 2], pad: f64) -> bool {
    (0..2).all(|k| p[k] >= bb[0][k] - pad && p[k] <= bb[1][k] + pad)
}

/// Builds the counter-clockwise projected 2D mesh of a set of faces of a region.
fn build_group_mesh(faces: &[FaceData], idxs: &[usize], frame: &PlaneFrame) -> UMesh {
    let mut gid_to_local: FxHashMap<usize, usize> = FxHashMap::default();
    let mut xy: Vec<[f64; 2]> = Vec::new();
    let mut rings: Vec<(ElementType, Vec<usize>)> = Vec::new();
    for &fi in idxs {
        let f = &faces[fi];
        let mut local: Vec<usize> = f
            .ring
            .iter()
            .zip(&f.pts)
            .map(|(&g, p)| {
                *gid_to_local.entry(g).or_insert_with(|| {
                    xy.push(frame.project(p));
                    xy.len() - 1
                })
            })
            .collect();
        let projected: Vec<[f64; 2]> = local.iter().map(|&l| xy[l]).collect();
        if signed_area2(&projected) < 0.0 {
            local.reverse();
        }
        rings.push((f.et, local));
    }

    let mut coords = nd::Array2::<f64>::zeros((xy.len(), 2));
    for (i, q) in xy.iter().enumerate() {
        coords[(i, 0)] = q[0];
        coords[(i, 1)] = q[1];
    }
    let mut mesh = UMesh::new(coords.into_shared());
    for (et, ring) in rings {
        mesh.add_element(et, &ring, None);
    }
    mesh
}

/// Cuts two 2D meshes by each other's edges and keeps every piece of both, together with the
/// parent maps of each side.
fn imprint_both(m1: &UMesh, m2: UMesh, tol: f64) -> (UMesh, ParentMap, ParentMap) {
    let (m2, _weld) = merge_on_reference_coords(m2, m1.view(), tol);
    let e1 = m1.descend(Some(Dimension::D2), Some(Dimension::D1));
    let e2 = m2.descend(Some(Dimension::D2), Some(Dimension::D1));
    let bvh1 = e1.view().bvh2();
    let bvh2 = e2.view().bvh2();

    let (mut shell, seg) = compute_overlay(&e1, &e2, &bvh2);

    let mut parents1 = Vec::new();
    cut_cells_all(&mut shell, m1, &e2.view(), &bvh2, &seg, Some(&mut parents1));
    let mut parents2 = Vec::new();
    cut_cells_all(
        &mut shell,
        &m2,
        &e1.view(),
        &bvh1,
        &seg,
        Some(&mut parents2),
    );

    (shell, parents1, parents2)
}

/// Refines the faces of one planar region.
///
/// Returns, for each input face (as an index into `faces`), the list of its child rings in 3D.
fn process_region(
    region: usize,
    faces: &[FaceData],
    face_mesh: &[usize],
    idxs: &[usize],
    tol: f64,
) -> Result<ChildrenByFace, StitchError> {
    // Fit the region plane on its largest face to avoid cancellation between opposite normals.
    let reference = idxs
        .iter()
        .copied()
        .max_by(|&a, &b| {
            let na = newell_normal3(&faces[a].pts);
            let nb = newell_normal3(&faces[b].pts);
            let la = na[0] * na[0] + na[1] * na[1] + na[2] * na[2];
            let lb = nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2];
            la.total_cmp(&lb)
        })
        .expect("region is non-empty");
    let frame = PlaneFrame::from_points(&faces[reference].pts);

    let all_pts: Vec<[f64; 3]> = idxs
        .iter()
        .flat_map(|&fi| faces[fi].pts.iter().copied())
        .collect();
    let deviation = frame.max_deviation(&all_pts);
    if deviation > tol {
        return Err(StitchError::NonPlanarRegion {
            region,
            deviation,
            tol,
        });
    }

    // Group the region faces by source mesh.
    let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &fi in idxs {
        groups.entry(face_mesh[fi]).or_default().push(fi);
    }

    let group_ids: Vec<usize> = groups.keys().copied().collect();
    let mut group_meshes: Vec<UMesh> = groups
        .values()
        .map(|idxs| build_group_mesh(faces, idxs, &frame))
        .collect();

    if group_meshes.len() < 2 {
        return Ok(Vec::new());
    }

    let mut acc = group_meshes.remove(0);
    let mut tag: FxHashMap<ElementId, usize> = acc
        .elements_of_dim(Dimension::D2)
        .map(|e| (e.id(), group_ids[0]))
        .collect();

    for (gi, gmesh) in group_meshes.into_iter().enumerate() {
        let gid = group_ids[gi + 1];
        let (new_acc, p1, p2) = imprint_both(&acc, gmesh, tol);
        let mut new_tag: FxHashMap<ElementId, usize> = FxHashMap::default();
        for (old, pieces) in &p1 {
            let t = tag[old];
            for pid in pieces {
                new_tag.insert(*pid, t);
            }
        }
        for (_, pieces) in &p2 {
            for pid in pieces {
                new_tag.insert(*pid, gid);
            }
        }
        acc = new_acc;
        tag = new_tag;
    }

    // Original faces of every group, in projected coordinates, for containment tests.
    // Rings are forced counter-clockwise because `contains_stable` requires that convention, and
    // the `Polygon` is built once here instead of on every containment test.
    let mut per_group: BTreeMap<usize, Vec<ProjectedFace>> = BTreeMap::new();
    for &fi in idxs {
        let mut ring: SmallVec<[[f64; 2]; PTS_INLINE]> =
            faces[fi].pts.iter().map(|p| frame.project(p)).collect();
        if signed_area2(&ring) < 0.0 {
            ring.reverse();
        }
        let bb = bounds_iter(ring.iter().copied());
        per_group.entry(face_mesh[fi]).or_default().push((
            fi,
            Polygon::unknown(ring.iter().copied()),
            bb,
        ));
    }

    let pad = tol.max(f64::EPSILON);
    // Characteristic size of the region: an area below `tol * extent` is indistinguishable from
    // mere contact at the requested tolerance.
    let mut extent: f64 = 0.0;
    {
        for k in 0..3 {
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for q in &all_pts {
                lo = lo.min(q[k]);
                hi = hi.max(q[k]);
            }
            extent = extent.max(hi - lo);
        }
    }
    let area_eps = tol * extent;

    // Assign each produced piece to the input face it comes from, and measure how much of the
    // region is a genuine interface (covered by every participating mesh).
    let mut shared_area = 0.0;
    let mut assigned: Vec<(Option<usize>, Vec<[f64; 3]>)> = Vec::new();
    for cell in acc.elements_of_dim(Dimension::D2) {
        let co = cell.connectivity();
        let pts2: Vec<[f64; 2]> = (0..co.len())
            .map(|i| {
                let g = co[i];
                [acc.coords()[(g, 0)], acc.coords()[(g, 1)]]
            })
            .collect();
        let area = signed_area2(&pts2).abs();
        if area <= area_eps {
            // Zero-area slivers produced when a cut lies exactly on a boundary edge.
            continue;
        }
        let interior = Polygon::unknown(pts2.iter().copied())
            .strict_interior_point()
            .ok_or(StitchError::DegeneratePiece { region })?;

        let gid = *tag
            .get(&cell.id())
            .ok_or(StitchError::UnmatchedPiece { region })?;
        let contains = |candidates: &[ProjectedFace]| -> Option<usize> {
            candidates.iter().find_map(|(fi, poly, bb)| {
                (point_in_bbox(interior, *bb, pad) && poly.contains_stable(&interior))
                    .then_some(*fi)
            })
        };
        let parent = per_group
            .get(&gid)
            .map(Vec::as_slice)
            .and_then(&contains)
            .ok_or(StitchError::UnmatchedPiece { region })?;
        // The piece is interface area as soon as two distinct meshes cover it. It need not be
        // covered by *every* mesh of the region: a block touching two others on different faces
        // shares each part of its own face with only one of them. The parent group is known to
        // cover the piece, so only the other groups are tested.
        let covering = 1 + per_group
            .iter()
            .filter(|(g, c)| **g != gid && contains(c).is_some())
            .count();
        if covering >= 2 {
            shared_area += area;
        }

        let ring3d: Vec<[f64; 3]> = pts2.iter().map(|q| frame.deproject(q)).collect();
        assigned.push((Some(parent), ring3d));
    }

    // Faces that only touch along an edge or at a point are not interfaces: leave them untouched.
    if shared_area <= area_eps {
        return Ok(Vec::new());
    }

    let mut children: FxHashMap<usize, Vec<Vec<[f64; 3]>>> = FxHashMap::default();
    for (fi, ring3d) in assigned {
        let fi = fi.ok_or(StitchError::UnmatchedPiece { region })?;
        children.entry(fi).or_default().push(ring3d);
    }

    Ok(children.into_iter().collect())
}

/// Welds coincident points (within `tol`) into unique coordinates.
///
/// Returns the unique coordinates together with, for every input point, the index of its
/// representative in the unique array.
fn weld_points(points: &[[f64; 3]], tol: f64) -> (Vec<[f64; 3]>, Vec<usize>) {
    let n = points.len();

    let close = |a: &[f64; 3], b: &[f64; 3]| {
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol && (a[2] - b[2]).abs() <= tol
    };

    // Points are bucketed into a `tol`-sized grid and compared against the 27 neighbouring cells.
    // Bucketing is what makes this correct and cheap: two points within `tol` may be arbitrarily
    // far apart in any global ordering, so a candidate that is skipped as "too far" along the
    // other axes would otherwise never be paired.
    let cell_of = |p: &[f64; 3]| {
        [
            (p[0] / tol).floor() as i64,
            (p[1] / tol).floor() as i64,
            (p[2] / tol).floor() as i64,
        ]
    };

    let mut clusters: Vec<usize> = vec![usize::MAX; n];
    let mut canonical: Vec<usize> = Vec::new();
    // Buckets hold a single cluster almost always, so they are kept inline to spare one
    // allocation per occupied cell.
    let mut grid: FxHashMap<[i64; 3], SmallVec<[usize; 1]>> =
        FxHashMap::with_capacity_and_hasher(n, FxBuildHasher);
    for i in 0..n {
        let cell = cell_of(&points[i]);
        let mut hit: Option<usize> = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let neighbour = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    let Some(list) = grid.get(&neighbour) else {
                        continue;
                    };
                    for &cid in list {
                        if close(&points[i], &points[canonical[cid]]) {
                            hit = Some(cid);
                            break 'search;
                        }
                    }
                }
            }
        }
        match hit {
            None => {
                let cid = canonical.len();
                canonical.push(i);
                grid.entry(cell).or_default().push(cid);
                clusters[i] = cid;
            }
            Some(cid) => {
                clusters[i] = cid;
                // The lowest input index represents the cluster, so that the nodes of the input
                // meshes win over the nodes that the imprint created for them.
                if i < canonical[cid] {
                    canonical[cid] = i;
                    grid.entry(cell).or_default().push(cid);
                }
            }
        }
    }

    (canonical.iter().map(|&i| points[i]).collect(), clusters)
}

fn point3<S: nd::Data<Elem = f64>>(coords: &nd::ArrayBase<S, nd::Ix2>, i: usize) -> [f64; 3] {
    [coords[(i, 0)], coords[(i, 1)], coords[(i, 2)]]
}

fn dot3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Whether `et` is one of the cell types [`stitch`] and [`conformize`] accept.
fn supported_cell_type(et: ElementType) -> bool {
    matches!(
        et,
        ElementType::TET4 | ElementType::HEX8 | ElementType::PHED
    )
}

/// Checks that `mesh` (mesh number `k` of the operation) can be stitched: embedded in 3D space
/// and made only of supported volume cells.
///
/// Shared by [`stitch`] and [`conformize`]; [`is_conform`] reports the same conditions as
/// issues of the report instead of raising them.
fn check_volume_mesh(mesh: &UMeshView, k: usize) -> Result<(), StitchError> {
    if mesh.space_dimension() != 3 {
        return Err(StitchError::InvalidSpaceDimension {
            mesh: k,
            found: mesh.space_dimension(),
        });
    }
    for e in mesh.elements() {
        let et = e.element_type();
        if et.dimension() != Dimension::D3 {
            return Err(StitchError::NonVolumeElement {
                mesh: k,
                element_type: et,
            });
        }
        if !supported_cell_type(et) {
            return Err(StitchError::UnsupportedElementType {
                mesh: k,
                element_type: et,
            });
        }
    }
    Ok(())
}

/// Stitches `meshes` into a single conforming polyhedral mesh.
///
/// See the module documentation for the algorithm, guarantees and limitations.
pub fn stitch(meshes: &[UMeshView], tol: f64) -> Result<UMesh, StitchError> {
    if meshes.len() < 2 {
        return Err(StitchError::NotEnoughMeshes {
            found: meshes.len(),
        });
    }
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(StitchError::InvalidTolerance { tol });
    }

    // Phase 0: validate and gather the global coordinate array.
    let mut coords_views: Vec<nd::ArrayView2<f64>> = Vec::with_capacity(meshes.len());
    let mut node_offset: Vec<usize> = Vec::with_capacity(meshes.len());
    let mut acc = 0usize;
    for (k, m) in meshes.iter().enumerate() {
        check_volume_mesh(m, k)?;
        node_offset.push(acc);
        acc += m.coords().nrows();
        coords_views.push(m.coords());
    }
    let n_input = acc;
    let gcoords = nd::concatenate(nd::Axis(0), &coords_views)
        .map_err(|e| StitchError::Internal(e.to_string()))?;

    // Phase 1: count volume faces and collect the boundary skin.
    let face_uses = count_face_uses(meshes, &node_offset);

    // The family labels of the output are those of each mesh, shifted past the previous ones.
    let mut family_shifts: Vec<usize> = Vec::with_capacity(meshes.len());
    let mut fam_acc = 0usize;
    for m in meshes {
        let mut max_fam = 0usize;
        for cell in m.elements_of_dim(Dimension::D3) {
            max_fam = max_fam.max(*cell.family);
        }
        family_shifts.push(fam_acc);
        fam_acc += max_fam + 1;
    }

    let mut skin = UMesh::new(gcoords.to_shared());
    let mut skin_faces: Vec<SkinFace> = Vec::new();
    let mut skin_index_by_key: FxHashMap<SmallVec<[usize; RING_INLINE]>, usize> =
        FxHashMap::with_capacity_and_hasher(face_uses.len(), FxBuildHasher);
    let mut eid_to_skin: FxHashMap<ElementId, usize> = FxHashMap::default();

    // Boundary faces are the ones used by a single cell. Sorting them by node ids keeps the
    // output independent of the hash map iteration order.
    let mut boundary: Vec<(SmallVec<[usize; RING_INLINE]>, FaceUse)> = face_uses
        .into_iter()
        .filter(|(_, u)| u.cells.len() == 1)
        .collect();
    boundary.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (key, use_) in boundary {
        let si = skin_faces.len();
        let eid = skin.add_element(face_etype(use_.ring.len()), &use_.ring, None);
        eid_to_skin.insert(eid, si);
        skin_index_by_key.insert(key, si);
        skin_faces.push(SkinFace {
            mesh: use_.mesh,
            ring: use_.ring,
        });
    }

    if skin_faces.is_empty() {
        // No boundary at all (degenerate input); still return a poly version.
        return polyze_all(meshes, &node_offset, &family_shifts, &gcoords, tol);
    }

    // Phase 2: cluster patches and group coincident patches of distinct meshes into regions.
    let faces = collect_surface_faces(&skin.view())?;
    let face_mesh: Vec<usize> = faces
        .iter()
        .map(|f| skin_faces[eid_to_skin[&f.id]].mesh)
        .collect();
    let face_skin_idx: Vec<usize> = faces.iter().map(|f| eid_to_skin[&f.id]).collect();
    let (patches, face_patch) = cluster_coplanar_patches(&faces, tol);
    let patch_mesh: Vec<usize> = patches.iter().map(|p| face_mesh[p.faces[0]]).collect();

    let bvh = skin.view().bvh3();
    let regions = group_regions_k(&patches, &patch_mesh, &face_patch, &bvh, tol);

    // Phase 3: refine every region.
    let mut region_outputs: Vec<(usize, Vec<Vec<[f64; 3]>>)> = Vec::new();
    for (region, patch_idxs) in regions.iter().enumerate() {
        let idxs: Vec<usize> = patch_idxs
            .iter()
            .flat_map(|&p| patches[p].faces.iter().copied())
            .collect();
        let children = process_region(region, &faces, &face_mesh, &idxs, tol)?;
        region_outputs.extend(children);
    }

    // Phase 4: flatten child rings into a global point list and index children per skin face.
    let mut child_points: Vec<[f64; 3]> = Vec::new();
    let mut face_children: Vec<Option<Vec<Vec<usize>>>> = vec![None; skin_faces.len()];
    for (fi, rings) in region_outputs {
        let si = face_skin_idx[fi];
        let mut idx_rings = Vec::with_capacity(rings.len());
        for ring in rings {
            let mut ir = Vec::with_capacity(ring.len());
            for p in ring {
                ir.push(child_points.len());
                child_points.push(p);
            }
            idx_rings.push(ir);
        }
        face_children[si] = Some(idx_rings);
    }

    // Phase 5: weld all nodes globally.
    let mut all_points: Vec<[f64; 3]> = Vec::with_capacity(n_input + child_points.len());
    for i in 0..n_input {
        all_points.push(point3(&gcoords, i));
    }
    all_points.extend_from_slice(&child_points);
    let (unique, id_of) = weld_points(&all_points, tol);

    let mut coords = nd::Array2::<f64>::zeros((unique.len(), 3));
    for (i, p) in unique.iter().enumerate() {
        coords[(i, 0)] = p[0];
        coords[(i, 1)] = p[1];
        coords[(i, 2)] = p[2];
    }
    let mut out = UMesh::new(coords.into_shared());

    // Phase 6: rebuild every volume cell as a PHED.
    for (k, m) in meshes.iter().enumerate() {
        let shift = node_offset[k];
        let fshift = family_shifts[k];
        for cell in m.elements_of_dim(Dimension::D3) {
            let (_, conn) = cell.to_poly();
            let faces_local = split_phed(&conn);
            let mut rings: Vec<Vec<usize>> = Vec::with_capacity(faces_local.len());
            for face in &faces_local {
                let mut key: SmallVec<[usize; RING_INLINE]> =
                    face.iter().map(|&g| g + shift).collect();
                key.sort_unstable();
                let si = skin_index_by_key.get(&key).copied();
                let children = si.and_then(|si| face_children[si].as_ref());
                match (si, children) {
                    (Some(si), Some(child_rings)) => {
                        let parent_ring = &skin_faces[si].ring;
                        let ppts: Vec<[f64; 3]> =
                            parent_ring.iter().map(|&g| point3(&gcoords, g)).collect();
                        let pn = newell_normal3(&ppts);
                        for cr in child_rings {
                            let cpts: Vec<[f64; 3]> = cr.iter().map(|&j| child_points[j]).collect();
                            let cn = newell_normal3(&cpts);
                            let mut ring: Vec<usize> =
                                cr.iter().map(|&j| id_of[n_input + j]).collect();
                            if dot3(&cn, &pn) < 0.0 {
                                ring.reverse();
                            }
                            rings.push(ring);
                        }
                    }
                    _ => {
                        rings.push(face.iter().map(|&g| id_of[g + shift]).collect());
                    }
                }
            }
            let phed = join_phed(&rings);
            out.add_element(ElementType::PHED, &phed, Some(*cell.family + fshift));
        }
    }

    Ok(out)
}

/// Fallback used when no boundary face exists: converts every volume cell to its poly form.
fn polyze_all(
    meshes: &[UMeshView],
    node_offset: &[usize],
    family_shifts: &[usize],
    gcoords: &nd::Array2<f64>,
    tol: f64,
) -> Result<UMesh, StitchError> {
    let (unique, id_of) = {
        let points: Vec<[f64; 3]> = (0..gcoords.nrows()).map(|i| point3(gcoords, i)).collect();
        weld_points(&points, tol)
    };
    let mut coords = nd::Array2::<f64>::zeros((unique.len(), 3));
    for (i, p) in unique.iter().enumerate() {
        coords[(i, 0)] = p[0];
        coords[(i, 1)] = p[1];
        coords[(i, 2)] = p[2];
    }
    let mut out = UMesh::new(coords.into_shared());
    for (k, m) in meshes.iter().enumerate() {
        let shift = node_offset[k];
        let fshift = family_shifts[k];
        for cell in m.elements_of_dim(Dimension::D3) {
            let (_, conn) = cell.to_poly();
            let rings: Vec<Vec<usize>> = split_phed(&conn)
                .iter()
                .map(|face| face.iter().map(|&g| id_of[g + shift]).collect())
                .collect();
            out.add_element(
                ElementType::PHED,
                &join_phed(&rings),
                Some(*cell.family + fshift),
            );
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// Shared face accounting, used by `stitch`, by `conformize` below, and by the `diagnose`
// submodule, followed by `conformize`: `stitch` applied to a single mesh split into its parts.
// ------------------------------------------------------------------------------------------

/// Uses of every face, keyed by the sorted (global) node ids of the face.
type FaceMap = FxHashMap<SmallVec<[usize; RING_INLINE]>, FaceUse>;

/// The part of every cell, as computed by [`count_volume_faces`].
type PartMap = FxHashMap<ElementId, usize>;

/// Counts the uses of every face of every volume cell of `meshes`, one map entry per face.
///
/// Node ids are shifted by `node_offset[k]` so that the faces of distinct meshes stay distinct,
/// which makes the map the shared face index of the whole operation: [`stitch`] takes the
/// boundary from it (a face used once), [`count_volume_faces`] the interfaces (a face used
/// twice) and [`is_conform`] the defects (a face used more than twice). A `TET4` or `HEX8` cell
/// has six faces, which sizes the map well.
fn count_face_uses(meshes: &[UMeshView], node_offset: &[usize]) -> FaceMap {
    let n_cells: usize = meshes.iter().map(|m| m.num_elements()).sum();
    let mut uses: FaceMap = FxHashMap::with_capacity_and_hasher(6 * n_cells, FxBuildHasher);
    for (k, m) in meshes.iter().enumerate() {
        let shift = node_offset[k];
        for cell in m.elements_of_dim(Dimension::D3) {
            let id = cell.id();
            let (_, conn) = cell.to_poly();
            for face in split_phed(&conn) {
                let mut key: SmallVec<[usize; RING_INLINE]> =
                    face.iter().map(|&g| g + shift).collect();
                key.sort_unstable();
                match uses.entry(key) {
                    Entry::Occupied(mut slot) => slot.get_mut().cells.push(id),
                    Entry::Vacant(slot) => {
                        let mut ring = face;
                        for g in &mut ring {
                            *g += shift;
                        }
                        slot.insert(FaceUse {
                            cells: smallvec::smallvec![id],
                            mesh: k,
                            ring,
                        });
                    }
                }
            }
        }
    }
    uses
}

/// Counts the faces of every volume cell of `mesh`, returning the uses of each face together with
/// the part of every cell.
///
/// A part is a group of cells linked by a chain of conformal faces, i.e. what is left of the mesh
/// once every interface that still has to be conformized is cut. [`conformize`] stitches the
/// parts, and [`is_conform`] compares nodes across them. Cells sharing no face with any other
/// cell form a part of their own.
fn count_volume_faces(mesh: &UMeshView) -> (FaceMap, PartMap) {
    let n_cells = mesh.elements_of_dim(Dimension::D3).count();
    let uses = count_face_uses(std::slice::from_ref(mesh), &[0]);

    // Cells sharing a face are in the same part. Union-find over those pairs, keyed by the
    // first cell of the pair so that each edge is inserted once.
    let mut parent: FxHashMap<ElementId, ElementId> =
        FxHashMap::with_capacity_and_hasher(n_cells, FxBuildHasher);
    fn find(parent: &mut FxHashMap<ElementId, ElementId>, x: ElementId) -> ElementId {
        let mut root = x;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        let mut y = x;
        while let Some(&p) = parent.get(&y) {
            if p == root {
                break;
            }
            parent.insert(y, root);
            y = p;
        }
        root
    }
    let mut edges: Vec<(ElementId, ElementId)> = Vec::new();
    for u in uses.values() {
        if let [a, b] = u.cells.as_slice() {
            edges.push((*a, *b));
        }
    }
    edges.sort_unstable();
    for (a, b) in edges {
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
    }

    let mut part_of: FxHashMap<ElementId, usize> =
        FxHashMap::with_capacity_and_hasher(n_cells, FxBuildHasher);
    let mut part_of_root: FxHashMap<ElementId, usize> =
        FxHashMap::with_capacity_and_hasher(n_cells, FxBuildHasher);
    let mut next_part = 0usize;
    // Walking the cells in order keeps the part numbering independent of the hash iteration
    // order, which would otherwise leak into the family labels of the result. A cell that shares
    // no face with any other cell is its own part, so every volume cell gets an entry.
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let id = cell.id();
        let root = if parent.contains_key(&id) {
            find(&mut parent, id)
        } else {
            id
        };
        let part = match part_of_root.get(&root) {
            Some(&p) => p,
            None => {
                part_of_root.insert(root, next_part);
                next_part += 1;
                next_part - 1
            }
        };
        part_of.insert(id, part);
    }
    (uses, part_of)
}

/// Conformizes a single 3D volume mesh, that is [`stitch`] applied to a mesh with itself.
///
/// The mesh is split into its *parts*: groups of cells linked by a chain of conforming faces.
/// Two blocks that merely touch, each carrying its own copy of the interface nodes, are two
/// parts, and are made conformal to each other exactly as [`stitch`] would do for two separate
/// meshes. A mesh whose interfaces are already conformal has a single part and comes back
/// unchanged, apart from the conversion of its cells to `PHED`.
///
/// Like [`stitch`], the result is a single `PHED` mesh whose families are relabeled (per part),
/// with fields and groups dropped; interfaces are imprinted only, so overlapping volumes are not
/// detected and faces shared by more than two cells are not repaired. See [`is_conform`] for the
/// diagnostic and the module documentation for the algorithm, guarantees and limitations.
pub fn conformize(mesh: &UMeshView, tol: f64) -> Result<UMesh, StitchError> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(StitchError::InvalidTolerance { tol });
    }
    check_volume_mesh(mesh, 0)?;

    let parts = parts_of(mesh);
    if parts.len() < 2 {
        // Nothing to conformize. The cell count drives the decision rather than the presence of
        // a boundary, so that a mesh with no boundary face at all still takes the fast path.
        return Ok(crate::tools::polyze::polyze(mesh));
    }
    let views: Vec<UMeshView> = parts.iter().map(UMesh::view).collect();
    stitch(&views, tol)
}

/// Splits `mesh` into the parts [`conformize`] stitches, in a stable order.
fn parts_of(mesh: &UMeshView) -> Vec<UMesh> {
    let (_, part_of) = count_volume_faces(mesh);
    if part_of.is_empty() {
        // No volume cell at all: one empty part, which sends `conformize` down the fast path.
        return vec![mesh.to_shared()];
    }
    let n_parts = part_of.values().copied().max().map_or(0, |m| m + 1);
    let mut ids: Vec<ElementIds> = vec![ElementIds::new(); n_parts];
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let p = part_of[&cell.id()];
        ids[p].add(cell.element_type(), cell.index());
    }
    // `extract` exists on owned meshes only, so the parts are cut out of one shared copy.
    let owned = mesh.to_shared();
    ids.iter().map(|part| owned.extract(part, false)).collect()
}
