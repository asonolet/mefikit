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
use crate::geometry::{PlaneFrame, Polygon, newell_normal3};
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

/// Signed area of a polygon in the plane.
fn signed_area2(pts: &[[f64; 2]]) -> f64 {
    let mut a = 0.0;
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * a
}

fn bbox2(pts: &[[f64; 2]]) -> [[f64; 2]; 2] {
    let mut bb = [[f64::INFINITY; 2], [f64::NEG_INFINITY; 2]];
    for p in pts {
        for k in 0..2 {
            bb[0][k] = bb[0][k].min(p[k]);
            bb[1][k] = bb[1][k].max(p[k]);
        }
    }
    bb
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
        let bb = bbox2(&ring);
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
            let remapped: Vec<usize> = conn.iter().map(|&g| id_of[g + shift]).collect();
            out.add_element(ElementType::PHED, &remapped, Some(*cell.family + fshift));
        }
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------
// `conformize` and `is_conform`: `stitch` applied to a single mesh, and the diagnostic that
// tells whether the mesh needs it.
// ------------------------------------------------------------------------------------------

/// Why a mesh is not conformal, and where.
///
/// Every variant carries the cells involved, and the coordinates of the place where the problem
/// is centred, so that a report can be turned into a selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConformanceIssue {
    /// Two nodes are within `tol` of each other, but the cells that use them meet only through
    /// those two nodes: they should have been welded into one.
    ///
    /// This is the block-on-block case, which [`conformize`] repairs by making the interface
    /// conformal.
    MergedNodes {
        /// Centroid of the two nodes.
        center: [f64; 3],
        /// The two nodes, `node_a` first.
        nodes: [usize; 2],
        /// Cells using `node_a`.
        cells_a: Vec<ElementId>,
        /// Cells using `node_b`.
        cells_b: Vec<ElementId>,
    },
    /// Two or more cells share a face, which leaves the domain ill-defined.
    OverlappingFaces {
        /// Centroid of the shared face.
        center: [f64; 3],
        /// The cells sharing the face, more than two.
        cells: Vec<ElementId>,
    },
    /// The mesh is not embedded in 3D space, so its volume cannot be checked.
    InvalidSpaceDimension {
        /// The spatial dimension that was found.
        found: usize,
    },
    /// The mesh holds a cell that [`conformize`] cannot handle.
    UnsupportedCells {
        /// The offending cells. More than one entry is reported.
        cells: Vec<ElementId>,
    },
    /// The mesh has no volume cell at all.
    NoVolumeCells,
}

/// Formats element ids as `HEX8#3, TRI3#0`, for issue messages.
fn fmt_cells(cells: &[ElementId]) -> String {
    cells
        .iter()
        .map(|id| format!("{:?}#{}", id.element_type(), id.index()))
        .collect::<Vec<_>>()
        .join(", ")
}

impl fmt::Display for ConformanceIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MergedNodes {
                center,
                nodes,
                cells_a,
                cells_b,
            } => write!(
                f,
                "cells {} and {} meet through the separate nodes {} and {} although those nodes \
                 are at the same location {center:?}; the interface between them is not conformal",
                fmt_cells(cells_a),
                fmt_cells(cells_b),
                nodes[0],
                nodes[1],
            ),
            Self::OverlappingFaces { center, cells } => write!(
                f,
                "cells {} all share the same face, whose centroid is {center:?}; a face can be \
                 shared by at most two cells",
                fmt_cells(cells),
            ),
            Self::InvalidSpaceDimension { found } => write!(
                f,
                "the mesh must be embedded in 3d space, found spatial dimension {found}",
            ),
            Self::UnsupportedCells { cells } => write!(
                f,
                "the mesh holds cells that cannot be conformized: {}",
                fmt_cells(cells),
            ),
            Self::NoVolumeCells => write!(f, "the mesh holds no volume cell"),
        }
    }
}

/// The outcome of [`is_conform`].
#[derive(Clone, Debug, PartialEq)]
pub struct ConformanceReport {
    /// The problems found, ordered from the most structural to the most numerical.
    pub issues: Vec<ConformanceIssue>,
    /// The total number of problems found, which may be larger than `issues.len()`.
    pub n_issues: usize,
    /// Whether `issues` was capped, in which case it does not list everything.
    pub truncated: bool,
}

impl ConformanceReport {
    /// Whether the mesh is conformal.
    pub fn is_conform(&self) -> bool {
        self.n_issues == 0
    }
}

/// Ranks an issue so that the report is deterministic and the most structural problems come
/// first: an input that cannot be conformized at all is more useful to report than a couple of
/// duplicated nodes.
fn issue_rank(issue: &ConformanceIssue) -> u8 {
    match issue {
        ConformanceIssue::NoVolumeCells
        | ConformanceIssue::InvalidSpaceDimension { .. }
        | ConformanceIssue::UnsupportedCells { .. } => 0,
        ConformanceIssue::OverlappingFaces { .. } => 1,
        ConformanceIssue::MergedNodes { .. } => 2,
    }
}

/// A duplicated node pair found by [`is_conform`]: the midpoint of the pair, the nodes, and the
/// cells using each of them.
struct NodeDuplicate {
    center: [f64; 3],
    nodes: [usize; 2],
    cells_a: Vec<ElementId>,
    cells_b: Vec<ElementId>,
}

/// Checks whether `mesh` is internally conformal and, when it is not, reports why and where.
///
/// Three kinds of defect are reported:
///
/// - a face shared by more than two cells, which leaves the domain ill-defined;
/// - two nodes within `tol` of each other that are used by cells which are not already linked
///   by a chain of conformal faces. Those cells meet through those two nodes where they should
///   meet through a shared, conformal interface; [`conformize`] welds the pair away;
/// - input that [`conformize`] would reject (not embedded in 3D, no volume cell, unsupported
///   cell type), reported as a single self-explanatory issue rather than an error, since the
///   missing conformance is then a consequence of the input being unsupported. Such an issue is
///   always included, even when `max_issues` is `Some(0)`.
///
/// Interfaces that overlap without sharing any coincident node are *not* detected: recognizing
/// them costs the geometric imprinting that [`conformize`] performs, so [`is_conform`] only
/// reports what is cheap to know beforehand. Similarly, duplicate nodes within one part are not
/// reported, because [`conformize`] leaves them alone.
///
/// `max_issues` caps the length of the report so that a large mesh cannot produce an unbounded
/// one; `None` keeps everything. [`ConformanceReport::n_issues`] always holds the true total.
///
/// The mesh is only read. See [`conformize`] for the corresponding operation.
pub fn is_conform(
    mesh: &UMeshView,
    tol: f64,
    max_issues: Option<usize>,
) -> Result<ConformanceReport, StitchError> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(StitchError::InvalidTolerance { tol });
    }
    let cap = max_issues.unwrap_or(usize::MAX);

    // An input that `conformize` would reject is reported as-is: the missing conformance is a
    // consequence of the input being unsupported, not a separate defect.
    if mesh.space_dimension() != 3 {
        return Ok(single_issue(ConformanceIssue::InvalidSpaceDimension {
            found: mesh.space_dimension(),
        }));
    }
    if mesh.elements_of_dim(Dimension::D3).count() == 0 {
        return Ok(single_issue(ConformanceIssue::NoVolumeCells));
    }
    let unsupported: Vec<ElementId> = mesh
        .elements()
        .filter(|e| {
            let et = e.element_type();
            et.dimension() != Dimension::D3 || !supported_cell_type(et)
        })
        .map(|e| e.id())
        .take(cap.max(1))
        .collect();
    if !unsupported.is_empty() {
        return Ok(single_issue(ConformanceIssue::UnsupportedCells {
            cells: unsupported,
        }));
    }

    let (uses, part_of) = count_volume_faces(mesh);

    let mut issues: Vec<ConformanceIssue> = Vec::new();
    let mut n_issues = 0usize;

    // A face used by more than two cells cannot be reconciled by refining: report it first, as
    // it hides whatever happens on that face.
    let mut overlapping: Vec<&FaceUse> = uses.values().filter(|u| u.cells.len() > 2).collect();
    overlapping.sort_by_key(|u| u.cells[0]);
    for u in overlapping {
        n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::OverlappingFaces {
                center: ring_centroid(&mesh.coords(), &u.ring),
                cells: u.cells.to_vec(),
            });
        }
    }

    // Nodes that are within `tol` of each other are candidates for welding. A candidate pair is
    // only a defect when its two cells are not already linked by a chain of conformal faces: the
    // nodes legitimately shared by neighbouring cells are not duplicates.
    let clusters = weld_clusters(mesh, tol);
    let mut cells_using: FxHashMap<usize, Vec<ElementId>> =
        FxHashMap::with_capacity_and_hasher(mesh.coords().nrows(), FxBuildHasher);
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let id = cell.id();
        for &n in cell.connectivity().iter() {
            cells_using.entry(n).or_default().push(id);
        }
    }
    let coords = mesh.coords();
    let mut duplicates: Vec<NodeDuplicate> = Vec::new();
    for members in clusters {
        let Some((a, rest)) = members.split_first() else {
            continue;
        };
        let cells_a = cells_using.get(a).cloned().unwrap_or_default();
        for &b in rest {
            let cells_b = cells_using.get(&b).cloned().unwrap_or_default();
            if cells_a.is_empty() || cells_b.is_empty() {
                continue;
            }
            if same_part(&part_of, &cells_a, &cells_b) {
                continue;
            }
            duplicates.push(NodeDuplicate {
                center: [
                    (coords[(*a, 0)] + coords[(b, 0)]) / 2.0,
                    (coords[(*a, 1)] + coords[(b, 1)]) / 2.0,
                    (coords[(*a, 2)] + coords[(b, 2)]) / 2.0,
                ],
                nodes: [*a, b],
                cells_a: cells_a.clone(),
                cells_b,
            });
        }
    }
    duplicates.sort_by(|x, y| {
        x.center[0]
            .total_cmp(&y.center[0])
            .then_with(|| x.center[1].total_cmp(&y.center[1]))
            .then_with(|| x.center[2].total_cmp(&y.center[2]))
            .then_with(|| x.nodes.cmp(&y.nodes))
    });
    for dup in duplicates {
        n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::MergedNodes {
                center: dup.center,
                nodes: dup.nodes,
                cells_a: dup.cells_a,
                cells_b: dup.cells_b,
            });
        }
    }

    issues.sort_by_key(issue_rank);
    Ok(ConformanceReport {
        truncated: n_issues > issues.len(),
        issues,
        n_issues,
    })
}

/// A report holding a single, self-explanatory issue.
fn single_issue(issue: ConformanceIssue) -> ConformanceReport {
    ConformanceReport {
        issues: vec![issue],
        n_issues: 1,
        truncated: false,
    }
}

/// Tells whether any cell of `cells_a` and any cell of `cells_b` belong to the same part, i.e.
/// are linked by a chain of conformal faces.
fn same_part(part_of: &FxHashMap<ElementId, usize>, a: &[ElementId], b: &[ElementId]) -> bool {
    a.iter()
        .filter_map(|c| part_of.get(c))
        .any(|pa| b.iter().filter_map(|c| part_of.get(c)).any(|pb| pa == pb))
}

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

/// Groups the nodes of `mesh` that are within `tol` of each other.
///
/// Returns one list of node ids per cluster of more than one node, in increasing order. The
/// clusters are the very ones [`weld_points`] builds (this only regroups its per-point result),
/// so [`is_conform`] reports exactly the node pairs that [`stitch`] welds together, and every
/// member of a cluster lies within `tol` of its first node.
fn weld_clusters(mesh: &UMeshView, tol: f64) -> Vec<Vec<usize>> {
    let coords = mesh.coords();
    let points: Vec<[f64; 3]> = (0..coords.nrows()).map(|i| point3(&coords, i)).collect();
    let (_, cluster_of) = weld_points(&points, tol);
    let mut groups: FxHashMap<usize, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(points.len(), FxBuildHasher);
    for (i, &c) in cluster_of.iter().enumerate() {
        groups.entry(c).or_default().push(i);
    }
    let mut out: Vec<Vec<usize>> = groups.into_values().filter(|g| g.len() > 1).collect();
    out.sort();
    out
}

/// Centroid of a node ring.
fn ring_centroid<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    ring: &[usize],
) -> [f64; 3] {
    let mut acc = [0.0; 3];
    for &n in ring {
        let p = point3(coords, n);
        for k in 0..3 {
            acc[k] += p[k];
        }
    }
    let inv = 1.0 / ring.len().max(1) as f64;
    [acc[0] * inv, acc[1] * inv, acc[2] * inv]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element_traits::ElementGeo;
    use crate::geometry::{cross2, signed_area2};
    use crate::mesh::ElementType;
    use crate::tools::RegularUMeshBuilder;
    use std::collections::BTreeSet;

    fn box_mesh(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
        RegularUMeshBuilder::new()
            .add_axis(xs.to_vec())
            .add_axis(ys.to_vec())
            .add_axis(zs.to_vec())
            .build()
    }

    /// Structured hex mesh over the node grid `xs × ys × zs` keeping only the cells listed in
    /// `cells` as `(i, j, k)` grid indices. Nodes are shared between neighbouring cells, and
    /// the `HEX8` winding matches [`RegularUMeshBuilder`].
    fn hex_block(xs: &[f64], ys: &[f64], zs: &[f64], cells: &[(usize, usize, usize)]) -> UMesh {
        let (nx, ny) = (xs.len(), ys.len());
        let node = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let flat: Vec<f64> = (0..zs.len())
            .flat_map(|k| {
                ys.iter()
                    .flat_map(move |y| xs.iter().map(move |x| [*x, *y, zs[k]]))
            })
            .flatten()
            .collect();
        let mut mesh = UMesh::new(
            nd::Array2::from_shape_vec((nx * ny * zs.len(), 3), flat)
                .unwrap()
                .into_shared(),
        );
        for &(i, j, k) in cells {
            mesh.add_element(
                ElementType::HEX8,
                &[
                    node(i, j, k),
                    node(i + 1, j, k),
                    node(i + 1, j + 1, k),
                    node(i, j + 1, k),
                    node(i, j, k + 1),
                    node(i + 1, j, k + 1),
                    node(i + 1, j + 1, k + 1),
                    node(i, j + 1, k + 1),
                ],
                None,
            );
        }
        mesh
    }

    /// Rotates the mesh coordinates around `axis = value` by `angle` radians.
    fn rotate_about_axis(mesh: &UMesh, axis: usize, value: f64, angle: f64) -> UMesh {
        let (c, s) = (angle.cos(), angle.sin());
        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
        let mut coords = mesh.coords().to_owned();
        for mut row in coords.outer_iter_mut() {
            let (du, dv) = (row[u] - value, row[v] - value);
            row[u] = value + c * du - s * dv;
            row[v] = value + s * du + c * dv;
        }
        let mut rotated = UMesh::new(coords.into_shared());
        for cell in mesh.elements() {
            rotated.add_element(cell.element_type(), cell.connectivity(), Some(*cell.family));
        }
        rotated
    }

    fn total_volume(mesh: &UMesh) -> f64 {
        mesh.elements_of_dim(Dimension::D3)
            .map(|c| c.measure3())
            .sum()
    }

    /// Every face of the mesh in its stored winding, together with the cells using it, keyed by
    /// the sorted node ids so that a face shared by two cells appears only once.
    fn face_table(mesh: &UMesh) -> BTreeMap<Vec<usize>, (Vec<usize>, Vec<usize>)> {
        let mut table: BTreeMap<Vec<usize>, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
        for cell in mesh.elements_of_dim(Dimension::D3) {
            let (_, conn) = cell.to_poly();
            for ring in split_phed(&conn) {
                let mut key = ring.to_vec();
                key.sort_unstable();
                let entry = table.entry(key).or_default();
                entry.0 = ring.to_vec();
                entry.1.push(cell.id().index());
            }
        }
        table
    }

    /// Points of a face ring, in its stored winding.
    fn face_points(mesh: &UMesh, ring: &[usize]) -> Vec<[f64; 3]> {
        let coords = mesh.coords();
        ring.iter()
            .map(|&g| [coords[(g, 0)], coords[(g, 1)], coords[(g, 2)]])
            .collect()
    }

    /// Summary of a validated `stitch` result.
    #[derive(Debug)]
    struct Report {
        cells: usize,
        interface_faces: usize,
        interface_area: f64,
        /// Interface area covered by each pair of input meshes, keyed by the pair of families.
        by_pair: BTreeMap<(usize, usize), f64>,
        volume: f64,
    }

    /// Validates a `stitch` result and returns a summary of its interface.
    ///
    /// Asserts that, within `tol`:
    /// - every cell is a `PHED`;
    /// - the total volume is conserved, since imprinting only subdivides matter;
    /// - every face is used by at most two cells, so no two cells overlap;
    /// - every face is planar and has at least three nodes;
    /// - faces shared by two cells are consistently wound, whatever the convention;
    /// - the interface covers exactly the area that the input meshes really share, pair by pair.
    ///   A face left unshared inside a contact region (a missing imprint) or a face shared that
    ///   should not be (a spurious one) moves that area, so this is what pins down conformity;
    /// - no two output nodes are within `tol` of each other, so welding did not leave
    ///   duplicates behind.
    ///
    /// Families are assumed to be relabeled in input order, which is what every fixture here
    /// feeds in (all input cells have family 0).
    fn check_result(out: &UMesh, inputs: &[&UMesh], tol: f64) -> Report {
        const PLANAR_TOL: f64 = 1e-12;
        for cell in out.elements() {
            assert_eq!(
                cell.element_type(),
                ElementType::PHED,
                "every output cell must be a PHED"
            );
        }
        let volume = total_volume(out);
        let expected: f64 = inputs.iter().map(|m| total_volume(m)).sum();
        assert!(
            (volume - expected).abs() <= 1e-9 * expected.abs().max(1.0),
            "volume is not conserved: got {volume}, expected {expected}"
        );

        let coords = out.coords();
        for i in 0..coords.nrows() {
            for j in i + 1..coords.nrows() {
                let d: f64 = (0..3)
                    .map(|k| (coords[(i, k)] - coords[(j, k)]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                assert!(d > tol, "nodes {i} and {j} are {d} apart, within {tol}");
            }
        }

        let mut interface_faces = 0;
        let mut interface_area = 0.0;
        let mut by_pair: BTreeMap<(usize, usize), f64> = BTreeMap::new();
        let mut winding = 0usize;
        for (key, (ring, cells)) in face_table(out) {
            assert!(
                cells.len() <= 2,
                "face {key:?} is used by {} cells, so two cells overlap",
                cells.len()
            );
            assert!(ring.len() >= 3, "face {key:?} is degenerate");
            let pts = face_points(out, &ring);
            let frame = PlaneFrame::from_points(&pts);
            assert!(
                frame.max_deviation(&pts) <= PLANAR_TOL,
                "face {key:?} is not planar (deviation {})",
                frame.max_deviation(&pts)
            );
            if cells.len() != 2 {
                continue;
            }
            let [c0, c1] =
                [cells[0], cells[1]].map(|i| out.element(ElementId::new(ElementType::PHED, i)));
            let normal = Polygon::unknown(pts.iter().copied()).normal();
            let d: f64 = (0..3)
                .map(|k| normal[k] * (c1.centroid3()[k] - c0.centroid3()[k]))
                .sum();
            let side = usize::from(d > 0.0);
            assert!(
                winding == 0 || winding == side,
                "face {key:?} is wound inconsistently with the other shared faces"
            );
            winding = side;
            if *c0.family == *c1.family {
                continue;
            }
            let area = Polygon::unknown(pts.iter().copied()).area();
            interface_faces += 1;
            interface_area += area;
            *by_pair
                .entry(((*c0.family).min(*c1.family), (*c0.family).max(*c1.family)))
                .or_default() += area;
        }
        for i in 0..inputs.len() {
            for j in i + 1..inputs.len() {
                let got = by_pair.get(&(i, j)).copied().unwrap_or(0.0);
                let want = contact_area(inputs[i], inputs[j], tol);
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "the interface between meshes {i} and {j} covers {got}, \
                     but the two meshes really share {want}"
                );
            }
        }
        Report {
            cells: out.num_elements(),
            interface_faces,
            interface_area,
            by_pair,
            volume,
        }
    }

    /// Area of the region covered by the boundary faces that `a` and `b` share, computed without
    /// `stitch`: every pair of coplanar boundary faces is clipped against each other. Only valid
    /// for meshes whose faces are all convex, which is the case for every fixture below.
    fn contact_area(a: &UMesh, b: &UMesh, tol: f64) -> f64 {
        fn boundary_faces(mesh: &UMesh) -> Vec<Vec<[f64; 3]>> {
            face_table(mesh)
                .into_values()
                .filter(|(_, cells)| cells.len() == 1)
                .map(|(ring, _)| face_points(mesh, &ring))
                .collect()
        }
        /// `true` if the two faces lie on the same plane, whatever side of it they are on.
        fn coplanar(pa: &[[f64; 3]], pb: &[[f64; 3]], tol: f64) -> bool {
            let na = Polygon::unknown(pa.iter().copied()).normal();
            let nb = Polygon::unknown(pb.iter().copied()).normal();
            if (0..3).map(|k| na[k] * nb[k]).sum::<f64>().abs() < 1.0 - 1e-9 {
                return false;
            }
            pb.iter().all(|q| {
                let d: f64 = (0..3).map(|k| na[k] * (q[k] - pa[0][k])).sum();
                d.abs() <= tol
            })
        }
        /// Area of the intersection of two convex 2D polygons given in any winding.
        fn overlap_2d(p: &[[f64; 2]], q: &[[f64; 2]]) -> f64 {
            let mut clip = q.to_vec();
            if signed_area2(&clip) < 0.0 {
                clip.reverse();
            }
            let mut out = p.to_vec();
            for i in 0..clip.len() {
                if out.len() < 3 {
                    return 0.0;
                }
                let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
                let (input, mut next) = (out.clone(), Vec::new());
                for k in 0..input.len() {
                    let (p, q) = (input[k], input[(k + 1) % input.len()]);
                    let (p_in, q_in) = (cross2(a, b, p) >= 0.0, cross2(a, b, q) >= 0.0);
                    let cut = || {
                        let (dax, day) = (b[0] - a[0], b[1] - a[1]);
                        let (dpx, dpy) = (q[0] - p[0], q[1] - p[1]);
                        let t =
                            ((p[0] - a[0]) * dpy - (p[1] - a[1]) * dpx) / (dax * dpy - day * dpx);
                        [a[0] + t * dax, a[1] + t * day]
                    };
                    match (p_in, q_in) {
                        (true, true) => next.push(q),
                        (true, false) => next.push(cut()),
                        (false, true) => {
                            next.push(cut());
                            next.push(q);
                        }
                        (false, false) => {}
                    }
                }
                out = next;
            }
            signed_area2(&out).abs()
        }
        let mut total = 0.0;
        for pa in boundary_faces(a) {
            for pb in boundary_faces(b) {
                if !coplanar(&pa, &pb, tol) {
                    continue;
                }
                let frame = PlaneFrame::from_points(&pa);
                let p: Vec<[f64; 2]> = pa.iter().map(|q| frame.project(q)).collect();
                let q: Vec<[f64; 2]> = pb.iter().map(|r| frame.project(r)).collect();
                total += overlap_2d(&p, &q);
            }
        }
        total
    }

    /// Faces lying on the plane `coords[axis] = value`, mapped to how many cells use them.
    fn faces_on_plane(
        mesh: &UMesh,
        axis: usize,
        value: f64,
        tol: f64,
    ) -> BTreeMap<Vec<usize>, usize> {
        let mut counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
        for cell in mesh.elements_of_dim(Dimension::D3) {
            let (_, conn) = cell.to_poly();
            for face in split_phed(&conn) {
                if face
                    .iter()
                    .all(|&g| (mesh.coords()[(g, axis)] - value).abs() <= tol)
                {
                    let mut key = face.to_vec();
                    key.sort_unstable();
                    *counts.entry(key).or_default() += 1;
                }
            }
        }
        counts
    }

    /// Number of interface faces on the plane `axis = value`, asserting that every face there is
    /// shared by exactly two cells (i.e. the interface is fully internal and conforming).
    fn count_interface_faces(mesh: &UMesh, axis: usize, value: f64, tol: f64) -> usize {
        let counts = faces_on_plane(mesh, axis, value, tol);
        assert!(
            counts.values().all(|&c| c == 2),
            "every interface face must be shared by exactly two cells, got {counts:?}"
        );
        counts.len()
    }

    #[test]
    fn test_stitch_requires_two_meshes() {
        let m = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let views = vec![m.view()];
        let e = stitch(&views, 1e-9).unwrap_err();
        assert_eq!(e, StitchError::NotEnoughMeshes { found: 1 });
    }

    #[test]
    fn test_stitch_rejects_bad_tolerance() {
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        assert_eq!(
            stitch(&views, -1.0).unwrap_err(),
            StitchError::InvalidTolerance { tol: -1.0 }
        );
        assert!(matches!(
            stitch(&views, f64::NAN).unwrap_err(),
            StitchError::InvalidTolerance { .. }
        ));
    }

    #[test]
    fn test_stitch_rejects_2d_mesh() {
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0])
            .add_axis(vec![0.0, 1.0])
            .build();
        let views = vec![a.view(), b.view()];
        assert_eq!(
            stitch(&views, 1e-9).unwrap_err(),
            StitchError::InvalidSpaceDimension { mesh: 1, found: 2 }
        );
    }

    #[test]
    fn test_stitch_rejects_surface_mesh_in_3d() {
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let mut b = UMesh::new(
            nd::arr2(&[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ])
            .to_shared(),
        );
        b.add_element(ElementType::QUAD4, &[0, 1, 2, 3], None);
        let views = vec![a.view(), b.view()];
        assert_eq!(
            stitch(&views, 1e-9).unwrap_err(),
            StitchError::NonVolumeElement {
                mesh: 1,
                element_type: ElementType::QUAD4
            }
        );
    }

    #[test]
    fn test_stitch_matching_interfaces() {
        // Two boxes with identical interface discretization.
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 2);
        assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 1);
        assert_eq!(out.coords().nrows(), 12);
        assert!(
            out.elements()
                .all(|e| e.element_type() == ElementType::PHED)
        );
    }

    #[test]
    fn test_stitch_mismatched_interfaces() {
        // A: 1x1x1 hex, B: 2x2x1 hexes sitting on top, interface is 1 quad vs 4 quads.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 5);
        // 8 + 18 nodes, 4 of which are shared interface corners.
        assert_eq!(out.coords().nrows(), 22);
        // The single quad of A became 4 quads shared with the 4 quads of B.
        assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 4);
    }

    #[test]
    fn test_stitch_partial_overlap() {
        // B's bottom face sticks out of A's top face on the +x side.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 3.0], &[0.0, 1.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 2);
        // A's top quad is split by B's edge x = 2 and B's bottom quad is split by A's edge y = 1,
        // yielding 3 faces on z = 1: one shared interface and one exclusive face per mesh.
        let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
        assert_eq!(counts.len(), 3);
        assert_eq!(counts.values().filter(|&&c| c == 2).count(), 1);
        assert_eq!(counts.values().filter(|&&c| c == 1).count(), 2);
    }

    #[test]
    fn test_stitch_three_meshes() {
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
        let c = box_mesh(&[0.0, 2.0], &[0.0, 1.0, 2.0], &[2.0, 3.0]);
        let views = vec![a.view(), b.view(), c.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 5);
        // A|B: A's 2x2 top is split in 2 by B's x = 1 edge; both pieces are shared with B.
        assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 2);
        // B|C: B's top is 2x1 (x 0..2) and C's bottom is 2x2, so B's top is split in 2 by C's
        // y = 1 edge and C's bottom in 2 by B's x = 1 edge: all 4 pieces are shared.
        assert_eq!(count_interface_faces(&out, 2, 2.0, 1e-12), 4);
    }

    #[test]
    fn test_stitch_accepts_poly_input() {
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
        let a_poly = crate::tools::polyze::polyze(&a.view());
        let b_poly = crate::tools::polyze::polyze(&b.view());
        let views = vec![a_poly.view(), b_poly.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 5);
        assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 4);
    }

    #[test]
    fn test_stitch_families_are_relabeled() {
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();
        let fams: Vec<usize> = out.elements().map(|e| *e.family).collect();
        assert!(fams.contains(&0));
        assert!(fams.contains(&1));
    }

    #[test]
    fn test_stitch_disjoint_meshes() {
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = box_mesh(&[5.0, 6.0], &[0.0, 1.0], &[0.0, 1.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();
        assert_eq!(out.num_elements(), 2);
        // The two boxes only touch the plane z = 1 on their own tops, which stay boundary faces.
        let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
        assert_eq!(counts.len(), 2);
        assert!(counts.values().all(|&c| c == 1));
    }

    /// Signed volume of the tetrahedron `t`, six times its actual volume.
    fn tet6v(coords: &nd::Array2<f64>, t: [usize; 4]) -> f64 {
        let p = |i: usize| [coords[(i, 0)], coords[(i, 1)], coords[(i, 2)]];
        let (a, b, c, d) = (p(t[0]), p(t[1]), p(t[2]), p(t[3]));
        let u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let v = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let w = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
        u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
            + u[2] * (v[0] * w[1] - v[1] * w[0])
    }

    /// A hexahedral grid of `xs` x `ys` x `zs` split into 6 tetrahedra per hex (Kuhn's
    /// decomposition around the `0-6` diagonal), oriented consistently.
    fn tet_box(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
        let kuhn = [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ];
        let mut coords: Vec<[f64; 3]> = Vec::new();
        let mut tets: Vec<[usize; 4]> = Vec::new();
        let (nx, ny, nz) = (xs.len(), ys.len(), zs.len());
        for k in 0..nz - 1 {
            for j in 0..ny - 1 {
                for i in 0..nx - 1 {
                    let base = coords.len();
                    let local: Vec<[f64; 3]> = [
                        (0, 0, 0),
                        (1, 0, 0),
                        (1, 1, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                        (1, 0, 1),
                        (1, 1, 1),
                        (0, 1, 1),
                    ]
                    .iter()
                    .map(|(di, dj, dk)| [xs[i + di], ys[j + dj], zs[k + dk]])
                    .collect();
                    coords.extend_from_slice(&local);
                    let local_arr = nd::Array2::from_shape_vec(
                        (8, 3),
                        local.iter().flatten().copied().collect(),
                    )
                    .unwrap();
                    for t in kuhn {
                        let t = [t[0] + base, t[1] + base, t[2] + base, t[3] + base];
                        let t = if tet6v(
                            &local_arr,
                            [t[0] - base, t[1] - base, t[2] - base, t[3] - base],
                        ) >= 0.0
                        {
                            t
                        } else {
                            [t[0], t[2], t[1], t[3]]
                        };
                        tets.push(t);
                    }
                }
            }
        }
        let arr = nd::Array2::from_shape_vec(
            (coords.len(), 3),
            coords.iter().flatten().copied().collect(),
        )
        .unwrap();
        let mut mesh = UMesh::new(arr.into_shared());
        for t in tets {
            mesh.add_element(ElementType::TET4, &t, None);
        }
        mesh
    }

    /// Total area of the faces lying on the plane `axis = value`.
    fn total_area_on_plane(mesh: &UMesh, axis: usize, value: f64, tol: f64) -> f64 {
        faces_on_plane(mesh, axis, value, tol)
            .keys()
            .map(|k| {
                let pts: Vec<[f64; 3]> = k
                    .iter()
                    .map(|&g| {
                        [
                            mesh.coords()[(g, 0)],
                            mesh.coords()[(g, 1)],
                            mesh.coords()[(g, 2)],
                        ]
                    })
                    .collect();
                Polygon::unknown(pts).area()
            })
            .sum()
    }

    /// Known limitation: see `walk_dart_map` in `crates/mefikit/src/element_traits/cut.rs`, which
    /// still `todo!()`s when a cutting segment enters a cell without cutting it. Triangular
    /// interfaces hit that path, so this test is disabled until the cutter is fixed. The same bug
    /// is reachable from the public `overlay_surfaces` API with plain triangles.
    #[test]
    #[ignore = "blocked on the unimplemented rewind in walk_dart_map"]
    fn test_stitch_tetrahedra() {
        // One 2x2x1 hex split into 6 tets (2 triangles on the interface) against a 2x2x1 grid of
        // hexes split into tets (8 triangles on the interface).
        let a = tet_box(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = tet_box(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
        let views = vec![a.view(), b.view()];
        let out = stitch(&views, 1e-9).unwrap();

        assert_eq!(out.num_elements(), 6 + 24);
        // Every triangle of the common refinement is shared by exactly two cells.
        assert!(count_interface_faces(&out, 2, 1.0, 1e-12) > 8);
        // The interface is covered exactly twice: once by A, once by B.
        assert!((total_area_on_plane(&out, 2, 1.0, 1e-12) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_stitch_reports_non_planar_region() {
        // Two quads from distinct meshes whose nodes are not coplanar.
        let mk = |z_last: f64| FaceData {
            id: ElementId::new(ElementType::QUAD4, 0),
            et: ElementType::QUAD4,
            ring: vec![0, 1, 2, 3],
            pts: vec![
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, z_last],
                [0.0, 1.0, z_last],
            ],
            normal: [0.0, 0.0, 1.0],
            bounds: [[0.0, 0.0, 0.5], [1.0, 1.0, 1.5]],
        };
        let faces = vec![mk(1.0), mk(1.5)];
        let e = process_region(0, &faces, &[0, 1], &[0, 1], 1e-9).unwrap_err();
        match e {
            StitchError::NonPlanarRegion {
                region, deviation, ..
            } => {
                assert_eq!(region, 0);
                assert!(deviation > 1e-9);
            }
            other => panic!("expected NonPlanarRegion, got {other:?}"),
        }
    }

    #[test]
    fn test_weld_points() {
        let pts = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1e-13],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let (uniq, ids) = weld_points(&pts, 1e-9);
        assert_eq!(uniq.len(), 3);
        assert_eq!(ids[1], ids[3]);
        assert_eq!(ids[0], ids[2]);
        assert_ne!(ids[0], ids[4]);
    }

    #[test]
    fn test_weld_points_paired_against_a_far_neighbour() {
        // Points 0 and 1 differ by a rounding error in x, but point 2 sorts between them and is a
        // whole cell away in z, so neither an anchored nor a chained scan over the sorted order
        // pairs 0 with 1. Point 3 is the same story in y. This is what a rotated interface leaves
        // behind: an imprint node computed by projecting a 2D intersection back to 3D next to the
        // input mesh's own node for the same corner.
        let eps = f64::EPSILON / 2.0;
        let pts = vec![
            [0.0, 0.0, 0.0],
            [0.0, 0.0, eps],
            [0.0, 0.0, 1.0],
            [eps, 0.0, 0.0],
        ];
        let (uniq, ids) = weld_points(&pts, 1e-9);
        assert_eq!(uniq.len(), 2, "0 and 1, and 0 and 3, are all coincident");
        assert_eq!(ids[0], ids[1]);
        assert_eq!(ids[0], ids[3]);
        assert_ne!(ids[0], ids[2]);
        // The lowest input index represents the cluster.
        assert_eq!(uniq[ids[0]], [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_weld_points_snap_imprints_onto_input_nodes() {
        // An imprint node that lands within `tol` of an input node must be welded onto it, and the
        // input node's own coordinates must win so the output keeps them exactly. `stitch` feeds
        // the input mesh nodes first, so the lower index must represent the cluster even when the
        // imprint sorts before it.
        let eps = f64::EPSILON / 2.0;
        let pts = vec![[0.0, 0.0, 0.0], [0.0, -eps, 0.0]];
        let (uniq, ids) = weld_points(&pts, 1e-9);
        assert_eq!(uniq.len(), 1);
        assert_eq!(uniq[0], [0.0, 0.0, 0.0]);
        assert_eq!(ids[0], ids[1]);
    }

    // ------------------------------------------------------------------------------------------
    // Hex-dominant fixtures. These are the shapes `stitch` is meant for: structured hex blocks and
    // already-conformized hex blocks (i.e. `PHED` cells whose faces are no longer all quads).
    // ------------------------------------------------------------------------------------------

    #[test]
    fn test_stitch_hex_grids_offset_in_plane() {
        // A is a single 2x2 hex; B is a 3x2 hex grid whose in-plane lines share no node with A.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(
            &[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0],
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        // A's top quad becomes the 3x2 grid of B, so A stays a single PHED with 12 nodes.
        assert_eq!(rep.cells, 7);
        assert_eq!(rep.interface_faces, 6);
        assert!((rep.interface_area - 4.0).abs() < 1e-12);
        assert!((rep.volume - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_stitch_hex_grids_rotated_in_plane() {
        // A is a 2x2 hex, B is a 2x2 hex block rotated 45 degrees about the interface centre, so
        // the contact region is an octagon and both sides are cut into non-quadrilateral faces.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = rotate_about_axis(
            &box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]),
            2,
            1.0,
            std::f64::consts::FRAC_PI_4,
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        // Square minus the four corner triangles cut by the rotated square's edges x + y = 2 +- r.
        let r = 2.0 - std::f64::consts::SQRT_2;
        let expected = 4.0 - 2.0 * r * r;
        assert!((rep.interface_area - expected).abs() < 1e-12);
        assert!((rep.volume - 8.0).abs() < 1e-12);
        // The two in-plane lines of the rotated grid cross at the centre, so they cut the
        // octagon into four pentagons: three of its nodes plus the centre and one crossing.
        assert_eq!(rep.interface_faces, 4);
        let face_sizes: BTreeSet<usize> =
            face_table(&out).into_keys().map(|key| key.len()).collect();
        assert!(
            face_sizes.contains(&5),
            "expected pentagonal faces, got {face_sizes:?}"
        );
    }

    #[test]
    fn test_stitch_hex_blocks_on_a_slanted_plane() {
        // The interface is the plane x + y + z = 3 rather than a coordinate plane.
        let a = rotate_about_axis(
            &box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]),
            0,
            0.0,
            std::f64::consts::FRAC_PI_4,
        );
        let b = rotate_about_axis(
            &box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]),
            0,
            0.0,
            std::f64::consts::FRAC_PI_4,
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert_eq!(rep.cells, 2);
        assert_eq!(rep.interface_faces, 1);
        assert!((rep.interface_area - 4.0).abs() < 1e-12);
    }

    #[test]
    fn test_stitch_hex_blocks_interlocking_on_two_planes() {
        // A is a stepped block: a 2x1x1 hex next to a 1x1x2 hex. B fills the notch, so the two
        // meshes share a z = 1 face *and* an x = 2 face. A2's x = 2 face is a quad that B's
        // x = 2 quad only covers halfway, so it gains a node and becomes a PHED face pair.
        let a = hex_block(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0],
            &[0.0, 1.0, 2.0],
            &[(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 1)],
        );
        let b = hex_block(
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0],
            &[1.0, 2.0],
            &[(0, 0, 0), (1, 0, 0)],
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        // z = 1 over x in [0, 2] (area 2) plus x = 2 over z in [1, 2] (area 1).
        assert!((rep.interface_area - 3.0).abs() < 1e-12);
        assert!((rep.volume - 6.0).abs() < 1e-12);
        // Two interface faces on z = 1 (x = [0, 1] and [1, 2]) and one on x = 2 (z = [1, 2]).
        assert_eq!(rep.interface_faces, 3);
        assert_eq!(count_interface_faces(&out, 2, 1.0, 1e-12), 3);
        assert_eq!(count_interface_faces(&out, 0, 2.0, 1e-12), 2);
    }

    #[test]
    fn test_stitch_hex_block_on_l_shaped_footprint() {
        // A is a 2x2 grid of hexes, B an L-shaped block covering three of its four top faces.
        let a = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[0.0, 1.0]);
        let b = hex_block(
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0],
            &[1.0, 2.0],
            &[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert_eq!(rep.cells, 7);
        assert_eq!(rep.interface_faces, 3);
        assert!((rep.interface_area - 3.0).abs() < 1e-12);
        assert!((rep.interface_area - contact_area(&a, &b, 1e-9)).abs() < 1e-12);
        // The uncovered top face of A stays a boundary face.
        let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
        assert_eq!(counts.len(), 4);
        assert_eq!(counts.values().filter(|&&c| c == 1).count(), 1);
        assert_eq!(counts.values().filter(|&&c| c == 2).count(), 3);
    }

    #[test]
    fn test_stitch_two_interface_regions_between_the_same_pair() {
        // One pair of meshes, two disjoint contact patches: region grouping must not merge them.
        let a = box_mesh(&[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = hex_block(
            &[0.0, 1.0, 2.0, 3.0],
            &[0.0, 1.0],
            &[1.0, 2.0],
            &[(0, 0, 0), (2, 0, 0)],
        );
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert_eq!(rep.cells, 5);
        assert_eq!(rep.interface_faces, 2);
        assert!((rep.interface_area - 2.0).abs() < 1e-12);
        // A's middle top face is left alone: it belongs to no contact region.
        let counts = faces_on_plane(&out, 2, 1.0, 1e-12);
        assert_eq!(counts.values().filter(|&&c| c == 1).count(), 1);
        assert_eq!(counts.values().filter(|&&c| c == 2).count(), 2);
    }

    #[test]
    fn test_stitch_l_shaped_hex_block_against_a_grid() {
        // The "hex-like" case: a concave PHED-shaped block (three hexes in an L) against a grid
        // whose in-plane lines cut its exposed faces into non-quadrilateral pieces.
        let a = hex_block(
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0, 2.0],
            &[0.0, 1.0],
            &[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
        );
        let b = box_mesh(&[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert!((rep.interface_area - 3.0).abs() < 1e-12);
        assert!((rep.volume - 7.0).abs() < 1e-12);
        // A's three unit-square top faces are each cut once by B's x lines at 2/3 and 4/3, so
        // the interface is six rectangles of area 1/3 rather than three unit squares.
        assert_eq!(rep.interface_faces, 6);
        assert_eq!(rep.by_pair.get(&(0, 1)), Some(&3.0));
    }

    #[test]
    fn test_stitch_four_hex_blocks_in_a_chain() {
        // Four blocks stacked in a chain, every pair mismatched in-plane.
        let ys_list: [Vec<f64>; 4] = [
            vec![0.0, 1.0, 2.0],
            vec![0.0, 0.5, 1.0, 1.5, 2.0],
            vec![0.0, 1.0, 2.0],
            vec![0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
        ];
        let blocks: Vec<UMesh> = ys_list
            .iter()
            .enumerate()
            .map(|(k, ys)| {
                let k = k as f64;
                box_mesh(&[0.0, 1.0, 2.0], ys, &[k, k + 1.0])
            })
            .collect();
        let views: Vec<_> = blocks.iter().map(UMesh::view).collect();
        let out = stitch(&views, 1e-9).unwrap();
        let rep = check_result(&out, &blocks.iter().collect::<Vec<_>>(), 1e-9);

        // Three interfaces, each covering the full 2x2 footprint, so twelve in total.
        assert!((rep.interface_area - 12.0).abs() < 1e-9);
        assert!((rep.volume - 16.0).abs() < 1e-12);
        assert!(rep.interface_faces >= 6);
    }

    #[test]
    fn test_stitch_hex_blocks_touching_only_along_an_edge() {
        // Zero-area contact: the blocks share a line segment, so nothing must be imprinted.
        let a = box_mesh(&[0.0, 1.0], &[0.0, 1.0], &[0.0, 1.0]);
        let b = box_mesh(&[1.0, 2.0], &[0.0, 1.0], &[1.0, 2.0]);
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert_eq!(rep.cells, 2);
        assert_eq!(rep.interface_faces, 0);
        assert_eq!(rep.interface_area, 0.0);
        assert!((rep.volume - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_stitch_result_can_be_stitched_again() {
        // Stitching an already-conformized result (PHED cells with refined faces) against a new
        // hex block must behave like any other PHED input.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
        let first = stitch(&[a.view(), b.view()], 1e-9).unwrap();

        // Stack a third block on top of the conformed result, with its own in-plane mismatch.
        let top = box_mesh(
            &[0.0, 2.0 / 3.0, 4.0 / 3.0, 2.0],
            &[0.0, 1.0, 2.0],
            &[2.0, 3.0],
        );
        for cell in top.elements() {
            assert_eq!(cell.element_type(), ElementType::HEX8);
        }
        let second = stitch(&[first.view(), top.view()], 1e-9).unwrap();
        let rep = check_result(&second, &[&first, &top], 1e-9);

        assert!((rep.interface_area - 8.0).abs() < 1e-9);
        assert!((rep.volume - 12.0).abs() < 1e-9);
    }

    #[test]
    fn test_stitch_conformized_hex_against_finer_hex() {
        // `polyze` turns the hexes into PHED cells; stitching a polyze'd block against a finer
        // hex grid is the closest stand-in for "already conformized hexa blocks".
        let a =
            crate::tools::polyze::polyze(&box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]).view());
        let b = box_mesh(&[0.0, 0.5, 1.0, 1.5, 2.0], &[0.0, 1.0, 2.0], &[1.0, 2.0]);
        let out = stitch(&[a.view(), b.view()], 1e-9).unwrap();
        let rep = check_result(&out, &[&a, &b], 1e-9);

        assert!((rep.interface_area - 4.0).abs() < 1e-12);
        assert!((rep.volume - 8.0).abs() < 1e-12);
        // A's single top quad is split 4x2 by B, so it becomes a PHED with 4 + 8 + 4 nodes.
        assert_eq!(rep.cells, 9);
        assert_eq!(rep.interface_faces, 8);
    }

    #[test]
    fn test_stitch_third_block_touching_both_others_with_partial_overlap() {
        // A plate under a wider, refined block, with a third block butting against both of them
        // along x. The contact patch on x = 0 therefore carries three meshes, and each part of it
        // is covered by the third block and one of the two others only.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 1.0, 2.2], &[0.0, 1.0, 2.5], &[1.0, 2.0]);
        let c = box_mesh(&[-1.0, -0.5, 0.0], &[0.0, 2.0], &[0.0, 2.0]);
        let views = vec![a.view(), b.view(), c.view()];
        let out = stitch(&views, 1e-9).unwrap();

        let rep = check_result(&out, &[&a, &b, &c], 1e-9);

        // 1 + 4 + 2 input cells, none of them split since C's faces already match B's lines.
        assert_eq!(rep.cells, 7);
        assert!((rep.interface_area - 8.0).abs() < 1e-12);
        assert_eq!(rep.by_pair.get(&(0, 1)), Some(&4.0));
        assert_eq!(rep.by_pair.get(&(0, 2)), Some(&2.0));
        assert_eq!(rep.by_pair.get(&(1, 2)), Some(&2.0));
    }

    #[test]
    fn test_stitch_three_blocks_where_the_third_touches_both_others() {
        // A and B stack along z while C touches them both along x, so the interface region of
        // C carries three meshes and no part of it is covered by all of them at once.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
        let c = box_mesh(&[2.0, 3.0], &[0.0, 2.0], &[0.0, 2.0]);
        let views = vec![a.view(), b.view(), c.view()];
        let out = stitch(&views, 1e-9).unwrap();

        let rep = check_result(&out, &[&a, &b, &c], 1e-9);

        assert_eq!(rep.cells, 3);
        assert_eq!(rep.interface_faces, 3);
        // A|B on z = 1, A|C on x = 2 over z 0..1, B|C on x = 2 over z 1..2.
        assert!((rep.interface_area - 8.0).abs() < 1e-12);
        assert!((rep.volume - 12.0).abs() < 1e-12);
    }

    #[test]
    fn test_stitch_three_blocks_where_the_third_is_imprinted_on_both() {
        // Same assembly, but C is refined along y and z, so both of its faces have to imprint
        // their lines into the faces of A and B.
        let a = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[0.0, 1.0]);
        let b = box_mesh(&[0.0, 2.0], &[0.0, 2.0], &[1.0, 2.0]);
        let c = box_mesh(&[2.0, 3.0], &[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
        let views = vec![a.view(), b.view(), c.view()];
        let out = stitch(&views, 1e-9).unwrap();

        let rep = check_result(&out, &[&a, &b, &c], 1e-9);

        // C is refined 2x2, so the total cell count and volume are those of the three inputs.
        assert_eq!(rep.cells, 6);
        // One face for A|B, then A's and B's x = 2 faces are each split in 2 by C's y = 1 line.
        assert_eq!(rep.interface_faces, 5);
        assert!((rep.interface_area - 8.0).abs() < 1e-12);
        assert_eq!(rep.by_pair.get(&(0, 1)), Some(&4.0));
        assert_eq!(rep.by_pair.get(&(0, 2)), Some(&2.0));
        assert_eq!(rep.by_pair.get(&(1, 2)), Some(&2.0));
        assert!((rep.volume - 12.0).abs() < 1e-12);
    }
}
