//! Overlay of two surface meshes embedded in 3D space.
//!
//! [`Overlayable::overlay_surfaces`] imprints two 2D meshes living in 3D space onto each
//! other wherever they coincide. The typical use-case is two volume meshes sharing an
//! interface with mismatched boundary tessellations: the operation refines each side with
//! the other side's interface edges so that both become mutually conformal.
//!
//! # Algorithm
//!
//! 1. Faces of each surface are clustered into maximal coplanar **patches** (faces touching
//!    through a shared node and lying in the same plane).
//! 2. Patches of the two surfaces are **grouped** into regions: two patches belong to the same
//!    region when they are coplanar within `tol` and their bounding boxes overlap. Grouping is
//!    transitive, so a small patch lying strictly inside a larger face forms a single region
//!    with it; footprints do not have to match.
//! 3. Each region is processed independently: both sides are projected on a common planar
//!    [`PlaneFrame`](crate::geometry::PlaneFrame), and the classic 2D overlay machinery is
//!    applied — the intersections are computed once so that both sides share the resulting
//!    node ids. This is an *imprint*: each side keeps its whole footprint, only refined.
//! 4. Pieces are reassembled into two refined meshes sharing the same coordinates array,
//!    together with parent maps relating input faces to produced elements.
//!
//! # Guarantees
//!
//! - Both refined meshes tile their input footprints exactly (area preserving up to `tol`)
//! - Intersection nodes are shared by both sides (single node id in the returned meshes)
//! - Families of input faces propagate to their pieces
//! - Untouched faces are copied verbatim (type and connectivity preserved; fields and groups
//!   are **not** carried over, see below)
//!
//! # Output layout
//!
//! `refined1` and `refined2` share the very same coordinates array, laid out as
//! `[skin1 nodes; skin2 nodes; added intersection nodes]`. Consequently:
//!
//! - `refined1` refers to node ids in `[0, n1 + n2)` where `n1 = skin1.coords().nrows()`
//! - `refined2` mostly refers to node ids in `[n1, n1 + n2)` (its own nodes, offset by `n1`),
//!   except on coincident areas where its nodes are welded onto first-surface nodes and the
//!   ids are in `[0, n1)` instead
//! - a node of an added intersection is placed on the best-fit plane of the region, i.e. on
//!   the first surface's side of the region; the second surface therefore deviates from its
//!   own geometry by at most the region deviation (`tol`)
//!
//! Fields and groups are not propagated: only element families are. Use the parent maps to
//! rebuild them if needed.
//!
//! # Assumptions
//!
//! - Input surfaces are valid (non-self-intersecting) and first-order (TRI3, QUAD4, PGON);
//!   lower dimensional elements are ignored
//! - Coincident areas are piecewise planar within `tol`; a region's two sides must lie in a
//!   common plane (up to the region's characteristic length scaled tolerance)
//!
//! Non-planar (curved) areas are therefore *not* imprinted: faces must be coplanar with their
//! counterpart within `tol`. Genuinely curved surfaces that share no common plane are copied
//! verbatim on both sides.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use rustc_hash::FxHashMap;

use ndarray as nd;

use super::{compute_overlay, cut_cells_all, merge_on_reference_coords};
use crate::geometry::{PlaneFrame, newell_normal3};
use crate::mesh::{Dimension, ElementId, ElementLike, ElementType, Regularity, UMesh, UMeshView};
use crate::prelude::ElementGeo;
use crate::tools::Descendable;
use crate::tools::spatial_index::{SpIdx3, SpatiallyIndexable};

/// Cosine threshold above which two normals are considered parallel.
const PARALLEL_NORMAL_COS_EPS: f64 = 1e-8;
/// Relative area threshold under which a face is flagged degenerate (w.r.t. its max edge
/// length squared).
const DEGENERATE_AREA_EPS: f64 = 1e-16;

/// Result of [`Overlayable::overlay_surfaces`].
///
/// `refined1` and `refined2` hold the imprinted faces of the first and second input surface
/// respectively. They **share the same coordinates array**, laid out as
/// `[skin1 nodes; skin2 nodes; added intersection nodes]`; intersection nodes created on the
/// coincident areas exist once and are referenced by both sides.
///
/// As a consequence, `refined2`'s node ids are offset by `skin1.coords().nrows()` and the
/// prefix of the shared coordinates array is unused by `refined2`. See the module
/// documentation for the full layout.
#[derive(Clone, Debug)]
pub struct SurfaceOverlay {
    /// Refined faces of the first input surface.
    pub refined1: UMesh,
    /// Refined faces of the second input surface.
    pub refined2: UMesh,
    /// Input face id (in the first surface) -> emitted element ids (in `refined1`).
    pub parents1: FxHashMap<ElementId, Vec<ElementId>>,
    /// Input face id (in the second surface) -> emitted element ids (in `refined2`).
    pub parents2: FxHashMap<ElementId, Vec<ElementId>>,
}

/// Error conditions of [`Overlayable::overlay_surfaces`].
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceOverlayError {
    /// Both inputs must be 2D surfaces embedded in 3D space.
    InvalidSpaceDimension {
        /// The offending spatial dimension.
        found: usize,
    },
    /// Quadratic or spline faces are not supported.
    UnsupportedElementType {
        /// The offending face.
        face: ElementId,
        /// Its element type.
        element_type: ElementType,
    },
    /// A face with null area was found.
    DegenerateFace {
        /// The offending face.
        face: ElementId,
    },
    /// A region deviates from planarity by more than the allowed tolerance.
    NonPlanarRegion {
        /// Ordinal of the region (in region order).
        region: usize,
        /// Measured maximum deviation to the fitted plane.
        deviation: f64,
        /// The tolerance used.
        tol: f64,
    },
    /// A face of a region could not be matched back to its parent, which indicates duplicated
    /// or otherwise inconsistent connectivity.
    UnmatchedParentRing {
        /// Ordinal of the region (in region order).
        region: usize,
    },
    /// Two distinct input faces map to the same produced element, which indicates that the
    /// input surface contains duplicated faces.
    DuplicateParent {
        /// The offending input face.
        face: ElementId,
    },
}

impl fmt::Display for SurfaceOverlayError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSpaceDimension { found } => write!(
                f,
                "surface overlay requires 2d surfaces embedded in 3d space, found spatial \
                 dimension {found}"
            ),
            Self::UnsupportedElementType { face, element_type } => write!(
                f,
                "unsupported element type {element_type:?} for face {face:?}"
            ),
            Self::DegenerateFace { face } => write!(f, "degenerate (null area) face {face:?}"),
            Self::NonPlanarRegion {
                region,
                deviation,
                tol,
            } => write!(
                f,
                "region {region} deviates from planarity by {deviation} which exceeds the \
                 tolerance {tol}; coincident surfaces must be piecewise planar"
            ),
            Self::UnmatchedParentRing { region } => write!(
                f,
                "could not match an input face of region {region} back to its parent; the \
                 input surface may contain a degenerate or duplicated connectivity"
            ),
            Self::DuplicateParent { face } => write!(
                f,
                "input face {face:?} was produced more than once; the input surface contains \
                 duplicated faces"
            ),
        }
    }
}

impl std::error::Error for SurfaceOverlayError {}

/// Imprints two surface meshes embedded in 3D space onto each other.
///
/// See the module documentation for the guarantees and assumptions. This is the free
/// function behind [`Overlayable::overlay_surfaces`].
pub fn overlay_surfaces(
    skin1: &UMeshView,
    skin2: &UMeshView,
    tol: f64,
) -> Result<SurfaceOverlay, SurfaceOverlayError> {
    assert!(
        tol.is_finite() && tol >= 0.0,
        "tol must be finite and non-negative"
    );
    for view in [skin1.view(), skin2.view()] {
        if view.space_dimension() != 3 {
            return Err(SurfaceOverlayError::InvalidSpaceDimension {
                found: view.space_dimension(),
            });
        }
    }

    // Phase 1: face collection and coplanar clustering per surface.
    let faces1 = collect_surface_faces(skin1)?;
    let faces2 = collect_surface_faces(skin2)?;
    let (patches1, _face_patch1) = cluster_coplanar_patches(&faces1, tol);
    let (patches2, face_patch2) = cluster_coplanar_patches(&faces2, tol);

    // Acceleration structure over the elements of skin2 for patch pairing queries.
    let bvh_skin2 = skin2.view().bvh3();

    // Phase 2: group coincident patches into disjoint regions. A region is a maximal set of
    // patches of the two surfaces lying in a common plane and overlapping; footprints are
    // allowed to differ (a small patch may sit strictly inside a larger face).
    let regions = group_regions(&patches1, &patches2, &face_patch2, &bvh_skin2, tol);
    let covered1: BTreeSet<usize> = regions
        .iter()
        .flat_map(|r| r.patches1.iter().copied())
        .collect();
    let covered2: BTreeSet<usize> = regions
        .iter()
        .flat_map(|r| r.patches2.iter().copied())
        .collect();

    // Global id layout: `[skin1 nodes; skin2 nodes; added intersection nodes]`.
    let n1 = skin1.coords().nrows();
    let n2 = skin2.coords().nrows();
    let added_base = n1 + n2;

    // Phase 3: flatten the regions into face-index jobs and process them independently.
    let jobs: Vec<(Vec<usize>, Vec<usize>)> = regions
        .iter()
        .map(|r| {
            let idxs1 = r
                .patches1
                .iter()
                .flat_map(|&p| patches1[p].faces.iter().copied())
                .collect();
            let idxs2 = r
                .patches2
                .iter()
                .flat_map(|&p| patches2[p].faces.iter().copied())
                .collect();
            (idxs1, idxs2)
        })
        .collect();

    #[cfg(feature = "rayon")]
    let outputs: Vec<Result<RegionOutput, SurfaceOverlayError>> = {
        use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
        jobs.par_iter()
            .enumerate()
            .map(|(region, (idxs1, idxs2))| {
                process_region(
                    region,
                    &faces1[..],
                    idxs1,
                    &faces2[..],
                    idxs2,
                    n1,
                    added_base,
                    tol,
                )
            })
            .collect()
    };
    #[cfg(not(feature = "rayon"))]
    let outputs: Vec<Result<RegionOutput, SurfaceOverlayError>> = jobs
        .iter()
        .enumerate()
        .map(|(region, (idxs1, idxs2))| {
            process_region(
                region,
                &faces1[..],
                idxs1,
                &faces2[..],
                idxs2,
                n1,
                added_base,
                tol,
            )
        })
        .collect();
    let mut outputs = outputs.into_iter().collect::<Result<Vec<_>, _>>()?;

    // Resolve added-node ids: within a region they were encoded as
    // `added_base + local_index`; give each region a disjoint slice of the added range.
    let mut shift = 0usize;
    let mut added_all: Vec<[f64; 3]> = Vec::new();
    for out in &mut outputs {
        for ring in out.rings_mut() {
            for g in ring.iter_mut() {
                if *g >= added_base {
                    *g += shift;
                }
            }
        }
        added_all.append(&mut out.added);
        shift = added_all.len();
    }

    // Deduplication of the added nodes against the existing coordinates and between
    // themselves (neighbouring regions may create identical border intersections). Single
    // lex-sort based pass, non-quadratic.
    let (coords, final_gids): (nd::ArcArray2<f64>, Vec<usize>) =
        dedup_added_coords(skin1.coords(), skin2.coords(), &added_all, tol);
    for ring in outputs.iter_mut().flat_map(|o| o.rings_mut()) {
        for g in ring.iter_mut() {
            if *g >= added_base {
                *g = final_gids[*g - added_base];
            }
        }
    }

    // Phase 4: assemble the refined meshes.
    let mut refined1 = UMesh::new(coords.clone());
    let mut refined2 = UMesh::new(coords.clone());
    let mut parents1: FxHashMap<ElementId, Vec<ElementId>> = FxHashMap::default();
    let mut parents2: FxHashMap<ElementId, Vec<ElementId>> = FxHashMap::default();

    for out in &outputs {
        emit_pieces(&mut refined1, skin1, &out.pieces1, &mut parents1)?;
        emit_pieces(&mut refined2, skin2, &out.pieces2, &mut parents2)?;
    }

    // Faces outside any region are copied verbatim so that the refined meshes cover exactly
    // their input footprints.
    for (i1, p1) in patches1.iter().enumerate() {
        if !covered1.contains(&i1) {
            for &fi in &p1.faces {
                copy_verbatim(skin1, faces1[fi].id, 0, &mut refined1, &mut parents1)?;
            }
        }
    }
    for (i2, p2) in patches2.iter().enumerate() {
        if !covered2.contains(&i2) {
            for &fi in &p2.faces {
                copy_verbatim(skin2, faces2[fi].id, n1, &mut refined2, &mut parents2)?;
            }
        }
    }

    Ok(SurfaceOverlay {
        refined1,
        refined2,
        parents1,
        parents2,
    })
}

/// A maximal set of patches of the two surfaces lying in a common plane and overlapping.
///
/// Patches are named by their index in their surface's patch list. Footprints are allowed to
/// differ: a small patch may be strictly contained in a larger face, which is the typical
/// "small part glued on a bigger face" configuration.
struct Region {
    patches1: Vec<usize>,
    patches2: Vec<usize>,
}

/// Groups coincident patches of the two surfaces into disjoint regions.
///
/// A patch of the first surface is a candidate for a patch of the second surface when their
/// normals are parallel, their planes coincide within `tol` and their bounding boxes overlap
/// (padded by `tol`). Regions are the connected components of the resulting bipartite graph:
/// a patch belongs to at most one region, and every patch of a region is imprinted with every
/// patch of the other surface in that region.
fn group_regions(
    patches1: &[Patch],
    patches2: &[Patch],
    face_patch2: &FxHashMap<ElementId, usize>,
    bvh_skin2: &SpIdx3,
    tol: f64,
) -> Vec<Region> {
    let n1 = patches1.len();
    // Union-find over `n1 + patches2.len()` nodes: `i1` for the first surface and `n1 + i2`
    // for the second one.
    let mut parent: Vec<usize> = (0..n1 + patches2.len()).collect();
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
    for (i1, p1) in patches1.iter().enumerate() {
        let min = [
            p1.bounds[0][0] - pad,
            p1.bounds[0][1] - pad,
            p1.bounds[0][2] - pad,
        ];
        let max = [
            p1.bounds[1][0] + pad,
            p1.bounds[1][1] + pad,
            p1.bounds[1][2] + pad,
        ];
        let mut seen: BTreeSet<usize> = BTreeSet::new();
        for eid in bvh_skin2.in_bounds(min, max).iter() {
            let Some(&i2) = face_patch2.get(&eid) else {
                continue;
            };
            if !seen.insert(i2) {
                continue;
            }
            if patches_are_coincident(p1, &patches2[i2], tol) {
                let r1 = root_compress(&mut parent, i1);
                let r2 = root_compress(&mut parent, n1 + i2);
                if r1 != r2 {
                    parent[r2] = r1;
                }
            }
        }
    }

    let mut members1: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i1 in 0..n1 {
        let r = root_compress(&mut parent, i1);
        members1.entry(r).or_default().push(i1);
    }
    let mut members2: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i2 in 0..patches2.len() {
        let r = root_compress(&mut parent, n1 + i2);
        members2.entry(r).or_default().push(i2);
    }

    let mut roots: BTreeSet<usize> = BTreeSet::new();
    roots.extend(members1.keys().copied());
    roots.extend(members2.keys().copied());
    let mut regions = Vec::new();
    for r in roots {
        let p1 = members1.get(&r).cloned().unwrap_or_default();
        let p2 = members2.get(&r).cloned().unwrap_or_default();
        // Components holding patches of a single surface are not regions: those patches are
        // copied verbatim.
        if p1.is_empty() || p2.is_empty() {
            continue;
        }
        regions.push(Region {
            patches1: p1,
            patches2: p2,
        });
    }
    regions
}

/// Returns `true` when two patches lie in a common plane within `tol` and their bounding
/// boxes overlap (padded by `tol`). Footprints may differ.
fn patches_are_coincident(p1: &Patch, p2: &Patch, tol: f64) -> bool {
    let n1 = p1.frame.normal();
    let n2 = p2.frame.normal();
    let parallel = n1[0] * n2[0] + n1[1] * n2[1] + n1[2] * n2[2] >= 1.0 - PARALLEL_NORMAL_COS_EPS;
    parallel && plane_distance(p1, p2) <= tol && bboxes_overlap(p1.bounds, p2.bounds, tol)
}

/// A produced face piece, expressed in the global node id space.
///
/// Rings of newly created intersection nodes use temporary ids `>= added_base`; they are
/// resolved during assembly (see [`overlay_surfaces`]).
#[derive(Clone, Debug)]
struct Piece {
    et: ElementType,
    ring: Vec<usize>,
}

/// Output of one matched region.
#[derive(Default)]
struct RegionOutput {
    pieces1: Vec<(ElementId, Vec<Piece>)>,
    pieces2: Vec<(ElementId, Vec<Piece>)>,
    /// Deprojected intersection coordinates, indexed by `id - added_base`.
    added: Vec<[f64; 3]>,
}

impl RegionOutput {
    fn rings_mut(&mut self) -> impl Iterator<Item = &mut Vec<usize>> {
        self.pieces1
            .iter_mut()
            .flat_map(|(_, ps)| ps.iter_mut())
            .chain(self.pieces2.iter_mut().flat_map(|(_, ps)| ps.iter_mut()))
            .map(|p| &mut p.ring)
    }
}

/// Per-face geometric data collected once per surface.
struct FaceData {
    id: ElementId,
    et: ElementType,
    /// Node ring in the surface coordinate space.
    ring: Vec<usize>,
    pts: Vec<[f64; 3]>,
    /// Unit Newell normal.
    normal: [f64; 3],
    bounds: [[f64; 3]; 2],
}

/// A maximal coplanar group of faces of one surface.
struct Patch {
    faces: Vec<usize>,
    frame: PlaneFrame,
    bounds: [[f64; 3]; 2],
}

/// Collects the D2 faces of `view`, rejecting unsupported or degenerate ones.
fn collect_surface_faces(view: &UMeshView) -> Result<Vec<FaceData>, SurfaceOverlayError> {
    let mut faces = Vec::new();
    for cell in view.elements_of_dim(Dimension::D2) {
        let et = cell.element_type();
        if !matches!(
            et,
            ElementType::TRI3 | ElementType::QUAD4 | ElementType::PGON
        ) {
            return Err(SurfaceOverlayError::UnsupportedElementType {
                face: cell.id(),
                element_type: et,
            });
        }
        let ring = cell.connectivity().to_vec();
        let pts: Vec<[f64; 3]> = (0..cell.num_nodes()).map(|i| *cell.coord3_ref(i)).collect();

        let nv = newell_normal3(&pts);
        let norm = (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt();
        let area = 0.5 * norm;
        // Degeneracy: area negligible w.r.t. the squared longest edge.
        let max_edge2 = pts
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let b = pts[(i + 1) % pts.len()];
                let dx = b[0] - a[0];
                let dy = b[1] - a[1];
                let dz = b[2] - a[2];
                dx * dx + dy * dy + dz * dz
            })
            .fold(0.0, f64::max);
        if norm == 0.0 || area <= DEGENERATE_AREA_EPS * max_edge2 {
            return Err(SurfaceOverlayError::DegenerateFace { face: cell.id() });
        }
        let normal = [nv[0] / norm, nv[1] / norm, nv[2] / norm];

        let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        for p in &pts {
            for k in 0..3 {
                bounds[0][k] = bounds[0][k].min(p[k]);
                bounds[1][k] = bounds[1][k].max(p[k]);
            }
        }

        faces.push(FaceData {
            id: cell.id(),
            et,
            ring,
            pts,
            normal,
            bounds,
        });
    }
    Ok(faces)
}

/// Clusters faces sharing an edge and lying in the same plane into maximal patches.
///
/// Returns the patches together with a map assigning every face to its patch index.
fn cluster_coplanar_patches(
    faces: &[FaceData],
    tol: f64,
) -> (Vec<Patch>, FxHashMap<ElementId, usize>) {
    // Union-find with path compression over face indices.
    let mut parent: Vec<usize> = (0..faces.len()).collect();
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

    // Faces sharing at least one node and lying in the same plane are joined. Node
    // connectivity (rather than full edge sharing) keeps T-junction tessellations, such as
    // an L-shaped footprint tiled by two rectangles, within a single patch.
    let mut node_faces: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for (fi, f) in faces.iter().enumerate() {
        for &g in &f.ring {
            node_faces.entry(g).or_default().push(fi);
        }
    }
    for bucket in node_faces.into_values() {
        if bucket.len() < 2 {
            continue;
        }
        let head = bucket[0];
        for &other in &bucket[1..] {
            if faces_are_coplanar(&faces[head], &faces[other], tol) {
                let rh = root_compress(&mut parent, head);
                let ro = root_compress(&mut parent, other);
                if rh != ro {
                    parent[ro] = rh;
                }
            }
        }
    }

    let mut members_of: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    for fi in 0..faces.len() {
        members_of
            .entry(root_compress(&mut parent, fi))
            .or_default()
            .push(fi);
    }
    let mut sorted_roots: Vec<usize> = members_of.keys().copied().collect();
    sorted_roots.sort_unstable();

    let mut patches = Vec::with_capacity(sorted_roots.len());
    let mut face_patch: FxHashMap<ElementId, usize> = FxHashMap::default();
    for (patch_idx, r) in sorted_roots.into_iter().enumerate() {
        let members = &members_of[&r];
        for &fi in members {
            face_patch.insert(faces[fi].id, patch_idx);
        }
        let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        let mut pts = Vec::new();
        for &fi in members {
            let [fmin, fmax] = faces[fi].bounds;
            for (bk, fk) in bounds[0].iter_mut().zip(fmin) {
                *bk = bk.min(fk);
            }
            for (bk, fk) in bounds[1].iter_mut().zip(fmax) {
                *bk = bk.max(fk);
            }
            pts.extend_from_slice(&faces[fi].pts);
        }
        let frame = PlaneFrame::from_points(&pts);
        patches.push(Patch {
            faces: members.clone(),
            frame,
            bounds,
        });
    }
    (patches, face_patch)
}

/// Returns `true` when both faces lie in a common plane within `tol`.
///
/// Every vertex of `b` is checked against the plane of `a`, not just the first one, so a face
/// that only touches the plane at a single vertex is not merged into the patch.
fn faces_are_coplanar(a: &FaceData, b: &FaceData, tol: f64) -> bool {
    let dot = a.normal[0] * b.normal[0] + a.normal[1] * b.normal[1] + a.normal[2] * b.normal[2];
    if dot < 1.0 - PARALLEL_NORMAL_COS_EPS {
        return false;
    }
    b.pts
        .iter()
        .all(|p| plane_offset(a.normal, a.pts[0], p) <= tol)
}

/// Absolute distance between the planes of two patches, evaluated at the origin of the
/// second patch frame. Meaningful only when both normals are parallel.
fn plane_distance(p1: &Patch, p2: &Patch) -> f64 {
    plane_offset(p1.frame.normal(), p1.frame.origin(), &p2.frame.origin())
}

/// Absolute signed-plane distance of point `x` from the plane `(n, o)`.
fn plane_offset(n: [f64; 3], o: [f64; 3], x: &[f64; 3]) -> f64 {
    let d = [x[0] - o[0], x[1] - o[1], x[2] - o[2]];
    (n[0] * d[0] + n[1] * d[1] + n[2] * d[2]).abs()
}

/// Returns `true` when two bounding boxes overlap within `pad` on every axis.
fn bboxes_overlap(a: [[f64; 3]; 2], b: [[f64; 3]; 2], pad: f64) -> bool {
    (0..3).all(|k| a[1][k] + pad >= b[0][k] && b[1][k] + pad >= a[0][k])
}

/// Builds the temporary projected mesh of the given faces.
///
/// Returns the mesh together with the global node id of each local node. Coordinates are
/// stored as plain 2D `[u, v]` rows as required by the 2D overlay machinery; element types
/// are preserved so untouched cells can be recognized and copied back verbatim.
fn build_projected_mesh(
    faces: &[FaceData],
    idxs: &[usize],
    frame: &PlaneFrame,
) -> (UMesh, Vec<usize>) {
    let mut gid_to_local: FxHashMap<usize, usize> = FxHashMap::default();
    let mut locals: Vec<usize> = Vec::new();
    let mut xy: Vec<[f64; 2]> = Vec::new();
    for &fi in idxs {
        let f = &faces[fi];
        for (&g, p) in f.ring.iter().zip(&f.pts) {
            if let std::collections::hash_map::Entry::Vacant(e) = gid_to_local.entry(g) {
                e.insert(locals.len());
                locals.push(g);
                xy.push(frame.project(p));
            }
        }
    }

    let mut coords = nd::Array2::<f64>::zeros((xy.len(), 2));
    for (i, q) in xy.iter().enumerate() {
        coords[(i, 0)] = q[0];
        coords[(i, 1)] = q[1];
    }
    let mut mesh = UMesh::new(coords.into_shared());

    // Regular blocks are grouped per element type; polygons go to a poly block.
    let mut regular: BTreeMap<ElementType, Vec<usize>> = BTreeMap::new();
    let mut poly_conn: Vec<usize> = Vec::new();
    let mut poly_offsets: Vec<usize> = Vec::new();
    for &fi in idxs {
        let f = &faces[fi];
        let ring: Vec<usize> = f.ring.iter().map(|g| gid_to_local[g]).collect();
        match f.et.regularity() {
            Regularity::Regular => regular.entry(f.et).or_default().extend(ring),
            Regularity::Poly => {
                poly_conn.extend(ring);
                poly_offsets.push(poly_conn.len());
            }
        }
    }
    for (et, flat) in regular {
        let width = et.num_nodes().expect("regular elements have a node count");
        let nrows = flat.len() / width;
        let conn = nd::Array2::from_shape_vec((nrows, width), flat)
            .expect("regular block connectivity is rectangular");
        mesh.add_regular_block(et, conn.into_shared(), None);
    }
    if !poly_offsets.is_empty() {
        mesh.add_poly_block(
            ElementType::PGON,
            nd::ArcArray1::from_vec(poly_conn),
            nd::ArcArray1::from_vec(poly_offsets),
            None,
        );
    }

    (mesh, locals)
}

/// Registers a shell added-intersection node and returns its region-local global encoding
/// (`added_base + contiguous index`), deprojecting it into 3D once.
#[allow(clippy::too_many_arguments)]
fn register_added(
    g: usize,
    shell: &UMesh,
    frame: &PlaneFrame,
    added_index: &mut FxHashMap<usize, usize>,
    added_xyz: &mut Vec<[f64; 3]>,
    added_base: usize,
) -> usize {
    let local = *added_index.entry(g).or_insert_with(|| {
        let q = [shell.coords()[(g, 0)], shell.coords()[(g, 1)]];
        added_xyz.push(frame.deproject(&q));
        added_xyz.len() - 1
    });
    added_base + local
}

/// Translates a piece ring from the projected shell coordinate space
/// `[projA; projB; added]` to the final global id space
/// `[skin1 nodes; skin2 nodes (+ n1); temporary added ids]`.
///
/// The second side may be welded onto first-side nodes, hence any ring may reference both
/// spaces; the `n1` shift applies to second-side nodes only.
#[allow(clippy::too_many_arguments)]
fn translate_ring(
    ring: &[usize],
    locals_a: &[usize],
    locals_b: &[usize],
    n1: usize,
    shell: &UMesh,
    shell_added_base: usize,
    frame: &PlaneFrame,
    added_index: &mut FxHashMap<usize, usize>,
    added_xyz: &mut Vec<[f64; 3]>,
    added_base: usize,
) -> Vec<usize> {
    let na = locals_a.len();
    ring.iter()
        .map(|&g| {
            if g < na {
                locals_a[g]
            } else if g < shell_added_base {
                n1 + locals_b[g - na]
            } else {
                register_added(g, shell, frame, added_index, added_xyz, added_base)
            }
        })
        .collect()
}

/// Processes one matched pair of patches.
///
/// Both sides are checked for planarity against a frame fitted on the first side only (the
/// two sides wind in opposite directions, so a combined fit could cancel), then projected to
/// XY where the classic overlay machinery runs. The intersections are computed once so that
/// both sides share the resulting node ids.
#[allow(clippy::too_many_arguments)]
fn process_region(
    region: usize,
    faces1: &[FaceData],
    idxs1: &[usize],
    faces2: &[FaceData],
    idxs2: &[usize],
    n1: usize,
    added_base: usize,
    tol: f64,
) -> Result<RegionOutput, SurfaceOverlayError> {
    let pts1: Vec<[f64; 3]> = idxs1
        .iter()
        .flat_map(|&fi| faces1[fi].pts.iter().copied())
        .collect();
    let pts2: Vec<[f64; 3]> = idxs2
        .iter()
        .flat_map(|&fi| faces2[fi].pts.iter().copied())
        .collect();
    let frame = PlaneFrame::from_points(&pts1);
    let deviation = frame.max_deviation(&pts1).max(frame.max_deviation(&pts2));
    if deviation > tol {
        return Err(SurfaceOverlayError::NonPlanarRegion {
            region,
            deviation,
            tol,
        });
    }

    let (mesh_a, locals_a) = build_projected_mesh(faces1, idxs1, &frame);
    let (mesh_b_raw, locals_b) = build_projected_mesh(faces2, idxs2, &frame);
    let na = mesh_a.coords().nrows();
    let nb = mesh_b_raw.coords().nrows();

    // Weld the second side onto the first so both share the node id space. The weld map
    // (merged-space B node -> A node) lets us predict the merged ring of every second-side
    // face, used later to identify produced pieces.
    let (mesh_b, weld) = merge_on_reference_coords(mesh_b_raw, mesh_a.view(), tol);

    let edges_a = mesh_a.descend(Some(Dimension::D2), Some(Dimension::D1));
    let edges_b = mesh_b.descend(Some(Dimension::D2), Some(Dimension::D1));
    let bvh_edges_a = edges_a.view().bvh2();
    let bvh_edges_b = edges_b.view().bvh2();

    // Shell coordinate layout: `[projA; projB; added intersections]`.
    let (mut shell, seg_intersections) = compute_overlay(&edges_a, &edges_b, &bvh_edges_b);
    let shell_added_base = na + nb;

    let mut parents_a: Vec<(ElementId, Vec<ElementId>)> = Vec::new();
    let mut parents_b: Vec<(ElementId, Vec<ElementId>)> = Vec::new();
    cut_cells_all(
        &mut shell,
        &mesh_a,
        &edges_b.view(),
        &bvh_edges_b,
        &seg_intersections,
        Some(&mut parents_a),
    );
    cut_cells_all(
        &mut shell,
        &mesh_b,
        &edges_a.view(),
        &bvh_edges_a,
        &seg_intersections,
        Some(&mut parents_b),
    );

    // Sorted parent ring -> position in `idxs`, used to recover the global face id of each
    // produced group of pieces without relying on element id ordering across blocks.
    //
    // Side 1 cells keep pure local-A rings, translated back through `locals_a`. Side 2
    // cells may be welded onto A nodes, so their faces are keyed by their predicted ring
    // in the merged space.
    let pos_to_face1: FxHashMap<Vec<usize>, usize> = idxs1
        .iter()
        .enumerate()
        .map(|(pos, &fi)| {
            let mut key = faces1[fi].ring.clone();
            key.sort_unstable();
            (key, pos)
        })
        .collect();
    let lb_index: FxHashMap<usize, usize> =
        locals_b.iter().enumerate().map(|(l, &g)| (g, l)).collect();
    let pos_to_face2: FxHashMap<Vec<usize>, usize> = idxs2
        .iter()
        .enumerate()
        .map(|(pos, &fi)| {
            let mut key: Vec<usize> = faces2[fi]
                .ring
                .iter()
                .map(|g| {
                    let l = lb_index[g];
                    match weld.get(&(l + na)) {
                        Some(&a) => a,
                        None => l + na,
                    }
                })
                .collect();
            key.sort_unstable();
            (key, pos)
        })
        .collect();

    let mut added_index: FxHashMap<usize, usize> = FxHashMap::default();
    let mut added_xyz: Vec<[f64; 3]> = Vec::new();
    let mut out = RegionOutput::default();

    let mut collect = |parents: &[(ElementId, Vec<ElementId>)],
                       subject: &UMesh,
                       is_side1: bool,
                       pieces_out: &mut Vec<(ElementId, Vec<Piece>)>|
     -> Result<(), SurfaceOverlayError> {
        for &(cell_id, ref piece_ids) in parents {
            let parent_cell = subject.element(cell_id);
            let parent_ring: Vec<usize> = parent_cell.connectivity().to_vec();
            let lookup_key: Vec<usize> = if is_side1 {
                let mut k: Vec<usize> = parent_ring.iter().map(|&g| locals_a[g]).collect();
                k.sort_unstable();
                k
            } else {
                let mut k = parent_ring.clone();
                k.sort_unstable();
                k
            };
            let mut pieces = Vec::with_capacity(piece_ids.len());
            for &pid in piece_ids.iter() {
                let pe = shell.element(pid);
                let ring_shell: Vec<usize> = pe.connectivity().to_vec();
                // Coincident boundaries are reported as boundary intersections: a cell
                // crossed by nothing comes back as a single polygon with its original
                // ring up to rotation, which must keep its original type.
                let verbatim = piece_ids.len() == 1 && {
                    let mut sorted_shell = ring_shell.clone();
                    sorted_shell.sort_unstable();
                    let mut sorted_parent = parent_ring.clone();
                    sorted_parent.sort_unstable();
                    sorted_shell == sorted_parent
                };
                let et = if verbatim {
                    parent_cell.element_type()
                } else {
                    pe.element_type()
                };
                let ring = translate_ring(
                    &ring_shell,
                    &locals_a,
                    &locals_b,
                    n1,
                    &shell,
                    shell_added_base,
                    &frame,
                    &mut added_index,
                    &mut added_xyz,
                    added_base,
                );
                pieces.push(Piece { et, ring });
            }
            let pos = if is_side1 {
                pos_to_face1.get(&lookup_key)
            } else {
                pos_to_face2.get(&lookup_key)
            }
            .ok_or(SurfaceOverlayError::UnmatchedParentRing { region })?;
            let face_id = if is_side1 {
                faces1[*pos].id
            } else {
                faces2[*pos].id
            };
            pieces_out.push((face_id, pieces));
        }
        Ok(())
    };

    collect(&parents_a, &mesh_a, true, &mut out.pieces1)?;
    collect(&parents_b, &mesh_b, false, &mut out.pieces2)?;
    out.added = added_xyz;
    Ok(out)
}

/// Appends the pieces of one side to `refined`, propagating the family from the parent face in
/// `skin`, and records the parent map entries.
///
/// Returns an error if a face is emitted twice, which can only happen when the input surface
/// contains duplicated faces.
fn emit_pieces(
    refined: &mut UMesh,
    skin: &UMeshView,
    pieces: &[(ElementId, Vec<Piece>)],
    parents: &mut FxHashMap<ElementId, Vec<ElementId>>,
) -> Result<(), SurfaceOverlayError> {
    for &(face_id, ref plist) in pieces {
        let cell = skin.element(face_id);
        let family = *cell.family;
        let mut ids = Vec::with_capacity(plist.len());
        for piece in plist {
            // Neither fields nor groups are propagated by the overlay; use the parent maps to
            // rebuild them.
            let id = refined.add_element(piece.et, &piece.ring, Some(family));
            ids.push(id);
        }
        if parents.insert(face_id, ids).is_some() {
            return Err(SurfaceOverlayError::DuplicateParent { face: face_id });
        }
    }
    Ok(())
}

/// Copies an untouched face of `skin` verbatim into `refined` (its nodes shifted by `offset`)
/// and records the parent map entry.
///
/// Returns an error if a face is emitted twice, which can only happen when the input surface
/// contains duplicated faces.
fn copy_verbatim(
    skin: &UMeshView,
    face_id: ElementId,
    offset: usize,
    refined: &mut UMesh,
    parents: &mut FxHashMap<ElementId, Vec<ElementId>>,
) -> Result<(), SurfaceOverlayError> {
    let cell = skin.element(face_id);
    let ring: Vec<usize> = cell.connectivity().iter().map(|g| g + offset).collect();
    let id = refined.add_element(cell.element_type(), &ring, Some(*cell.family));
    if parents.insert(face_id, vec![id]).is_some() {
        return Err(SurfaceOverlayError::DuplicateParent { face: face_id });
    }
    Ok(())
}

/// Deduplicates the deprojected added intersection nodes against the input coordinates and
/// between themselves (neighbouring regions may create identical border intersections).
///
/// Returns the final coordinates array `[skin1; skin2; kept added nodes]` together with the
/// final global id of each added node: an existing input node id when the point coincides
/// with an input node within `tol`, otherwise a fresh id past the input ranges. A single
/// lexicographic sort drives the pass, keeping it non-quadratic.
fn dedup_added_coords(
    coords1: nd::ArrayView2<f64>,
    coords2: nd::ArrayView2<f64>,
    added: &[[f64; 3]],
    tol: f64,
) -> (nd::ArcArray2<f64>, Vec<usize>) {
    let n1 = coords1.nrows();
    let n_base = n1 + coords2.nrows();
    let total = n_base + added.len();
    if added.is_empty() {
        let coords = nd::concatenate![nd::Axis(0), coords1, coords2];
        return (coords.into_shared(), Vec::new());
    }

    let point_of = |i: usize| -> [f64; 3] {
        if i < n1 {
            [coords1[(i, 0)], coords1[(i, 1)], coords1[(i, 2)]]
        } else if i < n_base {
            let j = i - n1;
            [coords2[(j, 0)], coords2[(j, 1)], coords2[(j, 2)]]
        } else {
            added[i - n_base]
        }
    };
    let close = |a: &[f64; 3], b: &[f64; 3]| {
        (a[0] - b[0]).abs() <= tol && (a[1] - b[1]).abs() <= tol && (a[2] - b[2]).abs() <= tol
    };

    let mut order: Vec<usize> = (0..total).collect();
    order.sort_unstable_by(|&a, &b| {
        let pa = point_of(a);
        let pb = point_of(b);
        pa[0]
            .total_cmp(&pb[0])
            .then_with(|| pa[1].total_cmp(&pb[1]))
            .then_with(|| pa[2].total_cmp(&pb[2]))
    });

    // Collapse each run onto the member of smallest original id: input nodes win over added
    // ones thanks to the id layout. A run is grown against its first (anchor) point rather
    // than chained point-to-point, so a node farther than `tol` from the anchor is never
    // absorbed by a sequence of closer neighbours.
    let mut final_gids = vec![usize::MAX; added.len()];
    let mut kept: Vec<[f64; 3]> = Vec::new();
    let mut next_new = n_base;
    let mut start = 0usize;
    while start < total {
        let anchor = point_of(order[start]);
        let mut end = start + 1;
        while end < total && close(&anchor, &point_of(order[end])) {
            end += 1;
        }
        let canonical = order[start..end].iter().copied().min().expect("non-empty");
        let gid = if canonical < n_base {
            canonical
        } else {
            let g = next_new;
            next_new += 1;
            kept.push(point_of(canonical));
            g
        };
        for &m in &order[start..end] {
            if m >= n_base {
                final_gids[m - n_base] = gid;
            }
        }
        start = end;
    }

    let mut extra = nd::Array2::<f64>::zeros((kept.len(), 3));
    for (i, p) in kept.iter().enumerate() {
        extra[(i, 0)] = p[0];
        extra[(i, 1)] = p[1];
        extra[(i, 2)] = p[2];
    }
    let coords = nd::concatenate![nd::Axis(0), coords1, coords2, extra];
    (coords.into_shared(), final_gids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray as nd;

    const TOL: f64 = 1e-9;

    /// Maps the unit square onto a 3D plane through the given transform.
    fn grid_coords(n: usize, map: impl Fn(f64, f64) -> [f64; 3]) -> nd::ArcArray2<f64> {
        let mut c = nd::Array2::<f64>::zeros(((n + 1) * (n + 1), 3));
        let mut k = 0;
        for j in 0..=n {
            for i in 0..=n {
                let p = map(i as f64 / n as f64, j as f64 / n as f64);
                c[(k, 0)] = p[0];
                c[(k, 1)] = p[1];
                c[(k, 2)] = p[2];
                k += 1;
            }
        }
        c.into_shared()
    }

    fn nid(i: usize, j: usize, n: usize) -> usize {
        j * (n + 1) + i
    }

    /// Structured `n x n` quad surface over the unit square.
    fn quad_surface(map: impl Fn(f64, f64) -> [f64; 3], n: usize) -> UMesh {
        let mut m = UMesh::new(grid_coords(n, &map));
        let mut flat = Vec::new();
        for j in 0..n {
            for i in 0..n {
                flat.extend([
                    nid(i, j, n),
                    nid(i + 1, j, n),
                    nid(i + 1, j + 1, n),
                    nid(i, j + 1, n),
                ]);
            }
        }
        let conn = nd::Array2::from_shape_vec((n * n, 4), flat).unwrap();
        m.add_regular_block(ElementType::QUAD4, conn.into_shared(), None);
        m
    }

    /// Structured triangulated surface over the unit square (two TRI3 per cell).
    fn tri_surface(map: impl Fn(f64, f64) -> [f64; 3], n: usize) -> UMesh {
        let mut m = UMesh::new(grid_coords(n, &map));
        let mut flat = Vec::new();
        for j in 0..n {
            for i in 0..n {
                flat.extend([nid(i, j, n), nid(i + 1, j, n), nid(i + 1, j + 1, n)]);
                flat.extend([nid(i, j, n), nid(i + 1, j + 1, n), nid(i, j + 1, n)]);
            }
        }
        let conn = nd::Array2::from_shape_vec((2 * n * n, 3), flat).unwrap();
        m.add_regular_block(ElementType::TRI3, conn.into_shared(), None);
        m
    }

    fn z_plane() -> impl Fn(f64, f64) -> [f64; 3] {
        |x, y| [x, y, 0.0]
    }

    /// Orthonormal basis of the plane with normal (1, 1, 1) through the origin.
    fn tilted_plane() -> impl Fn(f64, f64) -> [f64; 3] {
        let u = [1.0 / 2f64.sqrt(), -1.0 / 2f64.sqrt(), 0.0];
        let v = [1.0 / 6f64.sqrt(), 1.0 / 6f64.sqrt(), -2.0 / 6f64.sqrt()];
        move |s, t| {
            [
                s * u[0] + t * v[0],
                s * u[1] + t * v[1],
                s * u[2] + t * v[2],
            ]
        }
    }

    /// Doubly-curved map: no two distinct grid faces share a plane.
    fn curved_plane() -> impl Fn(f64, f64) -> [f64; 3] {
        |x, y| [x, y, 0.3 * (x * x + y * y)]
    }

    fn total_area(view: &UMeshView) -> f64 {
        view.elements_of_dim(Dimension::D2)
            .map(|c| {
                let pts: Vec<[f64; 3]> = (0..c.connectivity().len())
                    .map(|i| *c.coord3_ref(i))
                    .collect();
                let nv = newell_normal3(&pts);
                0.5 * (nv[0] * nv[0] + nv[1] * nv[1] + nv[2] * nv[2]).sqrt()
            })
            .sum()
    }

    fn count_type(view: &UMeshView, et: ElementType) -> usize {
        view.elements_of_dim(Dimension::D2)
            .filter(|c| c.element_type() == et)
            .count()
    }

    #[test]
    fn identical_grids_match_2d_imprint() {
        // Reference: plain 2D imprint of two identical grids.
        use crate::tools::{OverlayOperation, Overlayable};
        let xy = crate::mesh_examples::make_imesh_2d(4);
        let imprint = xy.overlay(xy.clone(), OverlayOperation::Imprint);
        assert_eq!(imprint.num_elements_of_dim(Dimension::D2), 16);

        // Surface path on the same grids lifted to z = 0.
        let skin1 = quad_surface(z_plane(), 4);
        let skin2 = quad_surface(z_plane(), 4);
        let out =
            overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("identical grids overlay");
        assert_eq!(
            out.refined1.num_elements_of_dim(Dimension::D2),
            imprint.num_elements_of_dim(Dimension::D2)
        );
        assert_eq!(count_type(&out.refined1.view(), ElementType::QUAD4), 16);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 1.0, epsilon = 1e-10);
        // No intersection nodes: final coordinates are exactly `[skin1; skin2]`.
        assert_eq!(out.refined1.coords().nrows(), 50);
    }

    #[test]
    fn tri_vs_quad_grid() {
        let skin1 = quad_surface(z_plane(), 4);
        let skin2 = tri_surface(z_plane(), 4);
        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("tri vs quad");

        // Each quad is cut by one triangulation diagonal into two polygons.
        assert_eq!(count_type(&out.refined1.view(), ElementType::PGON), 32);
        // Triangles are untouched and keep their type.
        assert_eq!(count_type(&out.refined2.view(), ElementType::TRI3), 32);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn rotated_plane_normal_111() {
        let skin1 = quad_surface(tilted_plane(), 3);
        let skin2 = tri_surface(tilted_plane(), 3);
        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("tilted plane");
        assert_eq!(count_type(&out.refined1.view(), ElementType::PGON), 18);
        assert_eq!(count_type(&out.refined2.view(), ElementType::TRI3), 18);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 1.0, epsilon = 1e-10);
        // Both sides share the same coordinates array.
        assert_eq!(out.refined1.coords(), out.refined2.coords());
    }

    #[test]
    fn l_shaped_pgon_patch() {
        // L-shaped polygon: unit square minus its upper right quadrant.
        let coords = nd::ArcArray2::from_shape_vec(
            (6, 3),
            vec![
                0.0, 0.0, 0.0, // 0
                2.0, 0.0, 0.0, // 1
                2.0, 1.0, 0.0, // 2
                1.0, 1.0, 0.0, // 3
                1.0, 2.0, 0.0, // 4
                0.0, 2.0, 0.0, // 5
            ],
        )
        .unwrap();
        let mut skin1 = UMesh::new(coords);
        skin1.add_element(ElementType::PGON, &[0, 1, 2, 3, 4, 5], None);

        // Same footprint tiled by two rectangles meeting at a T-junction (they share only
        // part of an edge, hence a single node): node based clustering keeps them in one
        // patch.
        let coords2 = nd::ArcArray2::from_shape_vec(
            (7, 3),
            vec![
                0.0, 0.0, 0.0, // 0
                1.0, 0.0, 0.0, // 1
                2.0, 0.0, 0.0, // 2
                2.0, 1.0, 0.0, // 3
                1.0, 1.0, 0.0, // 4
                1.0, 2.0, 0.0, // 5
                0.0, 2.0, 0.0, // 6
            ],
        )
        .unwrap();
        let mut skin2 = UMesh::new(coords2);
        skin2.add_element(ElementType::QUAD4, &[0, 1, 5, 6], None);
        skin2.add_element(ElementType::QUAD4, &[1, 2, 3, 4], None);

        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("L shape");
        // The interface between the two rectangles (x = 1, y in [0, 1]) crosses the L
        // interior: both sides end up split along it.
        assert_eq!(out.refined1.num_elements_of_dim(Dimension::D2), 2);
        assert_eq!(out.refined2.num_elements_of_dim(Dimension::D2), 2);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn small_patch_inside_big_face_imprints_both_sides() {
        // Unit square tiled by 4 faces on side 1, single [0, 2]^2 face on side 2.
        let skin1 = quad_surface(z_plane(), 2);
        let coords2 = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0],
        )
        .unwrap();
        let mut skin2 = UMesh::new(coords2);
        skin2.add_element(ElementType::QUAD4, &[0, 1, 2, 3], None);

        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("small patch");
        // Side 1 is fully covered by side 2's face, so its mesh is left as is.
        assert_eq!(out.refined1.num_elements_of_dim(Dimension::D2), 4);
        // Side 2's big face is cut by side 1's outer boundary (and internal edges): the
        // annular region plus the inner square sum back to the full footprint.
        assert_eq!(out.refined2.num_elements_of_dim(Dimension::D2), 5);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 4.0, epsilon = 1e-10);
        assert_eq!(out.refined1.coords(), out.refined2.coords());
    }

    #[test]
    fn cross_grid_imprint_shares_intersection_nodes() {
        // Two non-nested tessellations of the same unit square: their edges cross at
        // non-node points, so new shared intersection nodes must be created.
        let skin1 = quad_surface(z_plane(), 2);
        let skin2 = quad_surface(z_plane(), 3);
        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("cross grid");

        assert_eq!(out.refined1.num_elements_of_dim(Dimension::D2), 16);
        assert_eq!(out.refined2.num_elements_of_dim(Dimension::D2), 16);
        assert_abs_diff_eq!(total_area(&out.refined1.view()), 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(total_area(&out.refined2.view()), 1.0, epsilon = 1e-10);
        // `[skin1 (9 nodes); skin2 (16 nodes); 4 edge crossings]`.
        assert_eq!(out.refined1.coords().nrows(), 29);
        assert_eq!(out.refined1.coords(), out.refined2.coords());
    }

    #[test]
    fn curved_surfaces_are_left_untouched() {
        let skin1 = quad_surface(curved_plane(), 2);
        let skin2 = quad_surface(curved_plane(), 3);
        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("curved");

        // The two tessellations share no common plane, so no region is imprinted and both
        // surfaces are copied verbatim (area preserved, no shared intersection node).
        assert_eq!(out.refined1.num_elements_of_dim(Dimension::D2), 4);
        assert_eq!(out.refined2.num_elements_of_dim(Dimension::D2), 9);
        assert_abs_diff_eq!(
            total_area(&out.refined1.view()),
            total_area(&skin1.view()),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            total_area(&out.refined2.view()),
            total_area(&skin2.view()),
            epsilon = 1e-10
        );
        assert_eq!(out.refined1.coords().nrows(), 9 + 16);
    }

    #[test]
    fn disjoint_faces_copied_verbatim() {
        let skin1 = quad_surface(z_plane(), 2);

        // Second surface: one face matching skin1's footprint plus a far away face.
        let coords2 = nd::ArcArray2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                1.0, 1.0, 0.0, //
                0.0, 1.0, 0.0, //
                2.0, 0.0, 0.0, //
                3.0, 0.0, 0.0, //
                3.0, 1.0, 0.0, //
                2.0, 1.0, 0.0, //
            ],
        )
        .unwrap();
        let mut skin2 = UMesh::new(coords2);
        let matched_id = skin2.add_element(ElementType::QUAD4, &[0, 1, 2, 3], None);
        let far_id = skin2.add_element(ElementType::QUAD4, &[4, 5, 6, 7], None);

        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("disjoint faces");
        let ids = &out.parents2[&far_id];
        assert_eq!(ids.len(), 1);
        let piece = out.refined2.element(ids[0]);
        assert_eq!(piece.element_type(), ElementType::QUAD4);
        // Nodes shifted past the first surface coordinates.
        assert_eq!(
            piece.connectivity(),
            &[
                4 + skin1.coords().nrows() as usize,
                5 + skin1.coords().nrows() as usize,
                6 + skin1.coords().nrows() as usize,
                7 + skin1.coords().nrows() as usize
            ][..]
        );
        let _ = matched_id;
    }

    #[test]
    fn families_propagate_to_pieces() {
        let coords1 = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut skin1 = UMesh::new(coords1);
        skin1.add_element(ElementType::QUAD4, &[0, 1, 2, 3], Some(5));

        // Same footprint, different tessellation: the quad is cut into pieces which must
        // all carry its family.
        let mut skin2 = UMesh::new(grid_coords(1, z_plane()));
        skin2.add_element(ElementType::TRI3, &[0, 1, 3], Some(7));
        skin2.add_element(ElementType::TRI3, &[0, 3, 2], Some(7));

        let out = overlay_surfaces(&skin1.view(), &skin2.view(), TOL).expect("families");
        assert!(!out.parents1[&ElementId::new(ElementType::QUAD4, 0)].is_empty());
        for ids in out.parents1.values() {
            for id in ids {
                assert_eq!(*out.refined1.element(*id).family, 5);
            }
        }
        for ids in out.parents2.values() {
            for id in ids {
                assert_eq!(*out.refined2.element(*id).family, 7);
            }
        }
    }

    #[test]
    fn degenerate_face_rejected() {
        let coords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        .unwrap();
        let mut skin = UMesh::new(coords);
        skin.add_element(ElementType::PGON, &[0, 1, 2, 3], None);
        let empty = UMesh::new(nd::ArcArray2::zeros((0, 3)));

        let err = overlay_surfaces(&skin.view(), &empty.view(), TOL).unwrap_err();
        assert!(matches!(err, SurfaceOverlayError::DegenerateFace { .. }));
    }
}
