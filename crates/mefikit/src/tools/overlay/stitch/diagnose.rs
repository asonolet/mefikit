//! The conformance diagnostic of [`stitch`](super): [`is_conform`], the [`ConformanceReport`]
//! it returns, and the [`ConformanceIssue`]s that carry the findings.
//!
//! Everything in this file is about *telling* whether a mesh needs conformizing, never about
//! conformizing it; the operation itself is [`super::conformize`].

use std::fmt;

use ndarray as nd;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::super::surface::{FaceData, PARALLEL_NORMAL_COS_EPS, collect_surface_faces};
use super::{StitchError, point3, split_phed, supported_cell_type, weld_points};
use crate::element_traits::{ElementGeo, ElementTopo};
use crate::geometry::{Intersection, Intersections, PlaneFrame, Polygon, Polyhedron, Segment};
use crate::mesh::{Dimension, ElementId, ElementLike, ElementType, UMesh, UMeshView};
use crate::tools::spatial_index::SpatiallyIndexable;

/// Why a mesh is not conformal, and where.
///
/// Every variant carries the cells involved, and the coordinates of the place where the problem
/// is centred, so that a report can be turned into a selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConformanceIssue {
    /// A cell uses two distinct nodes that lie within `tol` of each other: the cell folds onto
    /// itself around that location. Welding the two nodes into one is what removes the fold.
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
    /// Two cells have coincident, overlapping boundary faces that are not the same face: the
    /// interface between them is only partially shared (a T-junction, or two different
    /// tessellations of the same interface).
    NonConformingFaces {
        /// A point inside the overlap of the two faces.
        center: [f64; 3],
        /// The two cells whose faces overlap without matching.
        cells: [ElementId; 2],
    },
    /// Two cells overlap in volume: their matter intersects although they are distinct cells.
    IntersectingCells {
        /// Centroid of the two cells.
        center: [f64; 3],
        /// The two overlapping cells, `cell_a` first.
        cells: [ElementId; 2],
        /// Volume of the intersection.
        volume: f64,
    },
    /// The mesh is not embedded in 3D space, so its volume cannot be checked.
    InvalidSpaceDimension {
        /// The spatial dimension that was found.
        found: usize,
    },
    /// The mesh holds a cell that [conformize](super::conformize) cannot handle.
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
            Self::NonConformingFaces { center, cells } => write!(
                f,
                "cells {} and {} have overlapping boundary faces around {center:?} that are not \
                 the same face; their interface is not conformal",
                fmt_cells(&cells[..1]),
                fmt_cells(&cells[1..]),
            ),
            Self::IntersectingCells {
                center,
                cells,
                volume,
            } => write!(
                f,
                "cells {} and {} overlap in volume {volume} around {center:?}; cells must not \
                 intersect one another",
                fmt_cells(&cells[..1]),
                fmt_cells(&cells[1..]),
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
        ConformanceIssue::NonConformingFaces { .. } => 2,
        ConformanceIssue::IntersectingCells { .. } => 3,
        ConformanceIssue::MergedNodes { .. } => 4,
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

/// Checks whether `mesh` is conformal and, when it is not, reports why and where.
///
/// A mesh is conformal when its cells do not intersect one another and two neighbour cells that
/// share any node share a *whole* common face. Such a mesh descends cleanly to its faces: each
/// interior face belongs to exactly two cells and appears once, so the descending connectivity
/// holds a single face rather than two coincident copies.
///
/// Four kinds of defect are reported:
///
/// - [`ConformanceIssue::IntersectingCells`]: two cells overlap in volume;
/// - [`ConformanceIssue::OverlappingFaces`]: a face is shared by more than two cells, which
///   leaves the domain ill-defined;
/// - [`ConformanceIssue::MergedNodes`]: two nodes lie within `tol` of each other but are not the
///   same node, and the cells using them do not meet through a full common face (a cell that
///   folds onto itself, or two cells that share only part of a face);
/// - input that [conformize](super::conformize) would reject (not embedded in 3D, no volume cell,
///   unsupported cell type), reported as a single self-explanatory issue rather than an error,
///   since the missing conformance is then a consequence of the input being unsupported. Such an
///   issue is always included, even when `max_issues` is `Some(0)`.
///
/// Coincident nodes are identified (welded within `tol`) before the shared-face test, so two
/// neighbour cells that carry their own copy of a *complete* interface are conformal; only a
/// *partially* shared interface, where some nodes coincide and others differ, is reported.
///
/// `max_issues` caps the length of the report so that a large mesh cannot produce an unbounded
/// one; `None` keeps everything. [`ConformanceReport::n_issues`] always holds the true total.
///
/// The mesh is only read. See [conformize](super::conformize) for the corresponding operation.
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

    let coords = mesh.coords();
    let mut issues: Vec<ConformanceIssue> = Vec::new();
    let mut n_issues = 0usize;

    intersecting_cells(mesh, &mut issues, &mut n_issues, cap);

    // Node identification: every node is replaced by the representative of its coincident
    // cluster, so that a cell using two ids for the same location is detected as a fold.
    let canonical = canonical_nodes(mesh, tol);

    // Cells that fold onto themselves: two of their nodes are coincident but distinct.
    let mut merged: Vec<NodeDuplicate> = Vec::new();
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let conn = cell.connectivity();
        let mut seen: FxHashMap<usize, usize> =
            FxHashMap::with_capacity_and_hasher(conn.len(), FxBuildHasher);
        let mut repeats: Vec<[usize; 2]> = Vec::new();
        for (pos, &node) in conn.iter().enumerate() {
            if node == usize::MAX {
                continue;
            }
            if let Some(prev) = seen.insert(canonical[node], pos) {
                let (a, b) = (conn[prev].min(node), conn[prev].max(node));
                if a != b {
                    repeats.push([a, b]);
                }
            }
        }
        repeats.sort_unstable();
        repeats.dedup();
        for [a, b] in repeats {
            let cells = vec![cell.id()];
            merged.push(NodeDuplicate {
                center: midpoint(&coords, a, b),
                nodes: [a, b],
                cells_a: cells.clone(),
                cells_b: cells,
            });
        }
    }

    // A face spelled with the very same node ids by more than two cells cannot be reconciled by
    // refining: report it first, as it hides whatever happens on that face.
    let mut face_cells: FxHashMap<Vec<usize>, Vec<ElementId>> = FxHashMap::with_capacity_and_hasher(
        mesh.elements_of_dim(Dimension::D3).count() * 6,
        FxBuildHasher,
    );
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (_, conn) = cell.to_poly();
        for ring in split_phed(&conn) {
            let mut key = ring.to_vec();
            key.sort_unstable();
            face_cells.entry(key).or_default().push(cell.id());
        }
    }
    let mut overlapping: Vec<(&Vec<usize>, &Vec<ElementId>)> =
        face_cells.iter().filter(|(_, cs)| cs.len() > 2).collect();
    overlapping.sort_by_key(|(_, cs)| cs[0]);
    for (key, cs) in overlapping {
        n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::OverlappingFaces {
                center: key_centroid(&coords, key),
                cells: cs.clone(),
            });
        }
    }

    // Two cells whose coincident boundary faces overlap geometrically without being the same
    // face have a non-conforming interface (a T-junction, or two different tessellations of the
    // same interface). This is the case the topological checks above cannot see.
    let mut non_conf_pairs: FxHashMap<[ElementId; 2], ()> = FxHashMap::default();
    non_conforming_faces(
        mesh,
        tol,
        &mut issues,
        &mut n_issues,
        cap,
        &mut non_conf_pairs,
    );

    // Two cells that share three nodes or more but have no face in common meet through a face
    // that is only partially shared: one side spells the interface with a node the other side
    // does not have (a "descending face" T-junction that shares node ids). Two cells touching
    // along an edge or a vertex share fewer than three nodes and are left alone.
    partially_shared_faces(mesh, &mut issues, &mut n_issues, cap, &mut non_conf_pairs);

    for dup in merged {
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

/// Appends one [`ConformanceIssue::IntersectingCells`] per pair of volume cells that overlap.
///
/// A bounding-volume hierarchy rejects the pairs that cannot touch; the remaining pairs are
/// tested by clipping one cell's polyhedron against the other's.
fn intersecting_cells(
    mesh: &UMeshView,
    issues: &mut Vec<ConformanceIssue>,
    n_issues: &mut usize,
    cap: usize,
) {
    let cells: Vec<ElementId> = mesh
        .elements_of_dim(Dimension::D3)
        .map(|c| c.id())
        .collect();
    let index = mesh.bvh3();
    let mut polys: FxHashMap<ElementId, Polyhedron> =
        FxHashMap::with_capacity_and_hasher(cells.len(), FxBuildHasher);
    let mut pairs: Vec<(ElementId, ElementId, f64, [f64; 3])> = Vec::new();
    for &cell in &cells {
        let poly = mesh.element(cell).to_polyhedron();
        let [min, max] = poly.bounds();
        let scale = min
            .iter()
            .chain(max.iter())
            .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
        let eps = 1e-12 * scale;
        let candidates = index.in_bounds(
            [min[0] - eps, min[1] - eps, min[2] - eps],
            [max[0] + eps, max[1] + eps, max[2] + eps],
        );
        for (other_et, indices) in candidates.0 {
            if other_et.dimension() != Dimension::D3 {
                continue;
            }
            for &i in &indices {
                let other = ElementId::new(other_et, i);
                if other <= cell {
                    continue;
                }
                let poly_b = polys
                    .entry(other)
                    .or_insert_with(|| mesh.element(other).to_polyhedron());
                let vol = poly.convex_intersection_volume_impl(poly_b);
                if vol > volume_eps(scale) {
                    let c = poly_centroid(&poly);
                    pairs.push((cell, other, vol, c));
                }
            }
        }
        polys.insert(cell, poly);
    }
    pairs.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    for (a, b, volume, center) in pairs {
        *n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::IntersectingCells {
                center,
                cells: [a, b],
                volume,
            });
        }
    }
}

/// Floor below which an intersection volume counts as no overlap, relative to `scale`.
fn volume_eps(scale: f64) -> f64 {
    1e-12 * scale.powi(3)
}

/// Appends one [`ConformanceIssue::NonConformingFaces`] per pair of distinct cells whose
/// coincident boundary faces overlap without being the same face.
///
/// Every face of every volume cell is collected in a 2D mesh and clustered by a bounding-volume
/// hierarchy on its bounding box. A candidate pair is a defect when the two faces come from
/// distinct cells, lie in a common plane within `tol`, are *not* spelled with the same node ids
/// and overlap with a positive area. This is the T-junction / mismatched-tessellation interface
/// that leaves two distinct faces where there should be one.
fn non_conforming_faces(
    mesh: &UMeshView,
    tol: f64,
    issues: &mut Vec<ConformanceIssue>,
    n_issues: &mut usize,
    cap: usize,
    seen: &mut FxHashMap<[ElementId; 2], ()>,
) {
    let coords = mesh.coords();
    let mut faces = UMesh::new(coords.to_shared());
    let mut owner: Vec<ElementId> = Vec::new();
    let mut keys: Vec<Vec<usize>> = Vec::new();
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (_, conn) = cell.to_poly();
        for ring in split_phed(&conn) {
            faces.add_element(ElementType::PGON, &ring, None);
            owner.push(cell.id());
            let mut key = ring.to_vec();
            key.sort_unstable();
            keys.push(key);
        }
    }
    let Ok(fdata) = collect_surface_faces(&faces.view()) else {
        return;
    };
    let index = faces.view().bvh3();

    let mut pairs: Vec<(ElementId, ElementId, [f64; 3])> = Vec::new();
    for (i, fi) in fdata.iter().enumerate() {
        let [min, max] = fi.bounds;
        let pad = tol.max(f64::EPSILON);
        let candidates = index.in_bounds(
            [min[0] - pad, min[1] - pad, min[2] - pad],
            [max[0] + pad, max[1] + pad, max[2] + pad],
        );
        for (et, indices) in candidates.0 {
            if et != ElementType::PGON {
                continue;
            }
            for &j in &indices {
                if j <= i || owner[i] == owner[j] || keys[i] == keys[j] {
                    continue;
                }
                let fj = &fdata[j];
                if !planes_coincident(fi, fj, tol) {
                    continue;
                }
                let Some(center) = faces_overlap(fi, fj) else {
                    continue;
                };
                let pair = if owner[i] < owner[j] {
                    [owner[i], owner[j]]
                } else {
                    [owner[j], owner[i]]
                };
                if seen.insert(pair, ()).is_some() {
                    continue;
                }
                pairs.push((pair[0], pair[1], center));
            }
        }
    }
    pairs.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    for (a, b, center) in pairs {
        *n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::NonConformingFaces {
                center,
                cells: [a, b],
            });
        }
    }
}

/// Appends one [`ConformanceIssue::NonConformingFaces`] per pair of distinct cells that share
/// three nodes or more without sharing a whole face.
///
/// Two cells meeting along a full face spell it with the same node set; two cells meeting only
/// along an edge or a vertex share two nodes or one. Any pair in between shares a *part* of a
/// face — the interface is only partially shared, and one side carries a node the other lacks —
/// so the mesh is not conformal. Node ids are compared as given: coincident-but-distinct ids are
/// handled by the geometric check, this one catches the case the two sides share ids.
fn partially_shared_faces(
    mesh: &UMeshView,
    issues: &mut Vec<ConformanceIssue>,
    n_issues: &mut usize,
    cap: usize,
    seen: &mut FxHashMap<[ElementId; 2], ()>,
) {
    let coords = mesh.coords();
    // The faces of every cell, and the cells reaching every node.
    let mut faces: Vec<(ElementId, Vec<usize>)> = Vec::new();
    let mut node_faces: FxHashMap<usize, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(coords.nrows(), FxBuildHasher);
    for cell in mesh.elements_of_dim(Dimension::D3) {
        let (_, conn) = cell.to_poly();
        for ring in split_phed(&conn) {
            let idx = faces.len();
            for &n in &ring {
                node_faces.entry(n).or_default().push(idx);
            }
            faces.push((cell.id(), ring.to_vec()));
        }
    }

    // Count the nodes every pair of faces shares. A pair reaching three nodes from distinct cells
    // is a partial interface; its two faces necessarily have different node sets.
    let mut shared: FxHashMap<(usize, usize), usize> =
        FxHashMap::with_capacity_and_hasher(faces.len(), FxBuildHasher);
    for face_ids in node_faces.values() {
        for (a_pos, &a) in face_ids.iter().enumerate() {
            for &b in &face_ids[(a_pos + 1)..] {
                let (lo, hi) = (a.min(b), a.max(b));
                *shared.entry((lo, hi)).or_insert(0) += 1;
            }
        }
    }

    let mut pairs: Vec<(ElementId, ElementId, [f64; 3])> = Vec::new();
    for ((lo, hi), count) in shared {
        if count < 3 {
            continue;
        }
        let (owner_a, ring_a) = &faces[lo];
        let (owner_b, ring_b) = &faces[hi];
        if owner_a == owner_b {
            continue;
        }
        // A whole shared face is exactly a common node set; a partial one is not.
        let mut ka = ring_a.clone();
        ka.sort_unstable();
        let mut kb = ring_b.clone();
        kb.sort_unstable();
        if ka == kb {
            continue;
        }
        let pair = if owner_a < owner_b {
            [*owner_a, *owner_b]
        } else {
            [*owner_b, *owner_a]
        };
        if seen.insert(pair, ()).is_some() {
            continue;
        }
        let common: Vec<usize> = ring_a
            .iter()
            .copied()
            .filter(|n| ring_b.contains(n))
            .collect();
        pairs.push((pair[0], pair[1], key_centroid(&coords, &common)));
    }
    pairs.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    for (a, b, center) in pairs {
        *n_issues += 1;
        if issues.len() < cap {
            issues.push(ConformanceIssue::NonConformingFaces {
                center,
                cells: [a, b],
            });
        }
    }
}

/// Tells whether two coplanar-in-the-large faces lie in a common plane within `tol` (their
/// normals may be antiparallel, as the two sides of an interface are wound oppositely).
fn planes_coincident(a: &FaceData, b: &FaceData, tol: f64) -> bool {
    let (na, nb) = (a.normal, b.normal);
    let dot = na[0] * nb[0] + na[1] * nb[1] + na[2] * nb[2];
    if dot.abs() < 1.0 - PARALLEL_NORMAL_COS_EPS {
        return false;
    }
    let d = [
        b.pts[0][0] - a.pts[0][0],
        b.pts[0][1] - a.pts[0][1],
        b.pts[0][2] - a.pts[0][2],
    ];
    (na[0] * d[0] + na[1] * d[1] + na[2] * d[2]).abs() <= tol
}

/// Returns a point inside the geometric overlap of two coplanar faces, or `None` when they do
/// not overlap with a positive area. Both faces are projected on the first face's plane.
fn faces_overlap(a: &FaceData, b: &FaceData) -> Option<[f64; 3]> {
    let frame = PlaneFrame::from_points(&a.pts);
    let pa: Vec<[f64; 2]> = a.pts.iter().map(|p| frame.project(p)).collect();
    let pb: Vec<[f64; 2]> = b.pts.iter().map(|p| frame.project(p)).collect();

    // A shared vertex or edge alone is not a non-conforming interface: require a positive-area
    // overlap, witnessed by a strictly-interior vertex or a proper edge crossing. Both polygons
    // are made counter-clockwise, as `contains_stable` assumes that winding.
    let poly_a = Polygon::unknown(pa.iter().copied()).into_ccw();
    let poly_b = Polygon::unknown(pb.iter().copied()).into_ccw();
    for p in &pb {
        if poly_a.contains_stable(p) {
            return Some(frame.deproject(p));
        }
    }
    for p in &pa {
        if poly_b.contains_stable(p) {
            return Some(frame.deproject(p));
        }
    }
    for i in 0..pa.len() {
        let s1 = Segment::new(pa[i], pa[(i + 1) % pa.len()]);
        for j in 0..pb.len() {
            let s2 = Segment::new(pb[j], pb[(j + 1) % pb.len()]);
            if let Intersections::One(Intersection::New(p)) = s1.intersect(s2) {
                return Some(frame.deproject(&p));
            }
        }
    }
    None
}

/// Centroid of a polyhedron, averaging its vertices.
fn poly_centroid(poly: &Polyhedron) -> [f64; 3] {
    let n = poly.len().max(1) as f64;
    let mut acc = [0.0; 3];
    for p in poly.iter() {
        for k in 0..3 {
            acc[k] += p[k];
        }
    }
    [acc[0] / n, acc[1] / n, acc[2] / n]
}

/// Maps every node of `mesh` to the representative of its coincident cluster.
///
/// Nodes at most `tol` apart (Chebyshev distance) share a representative. With `tol == 0`, every
/// node is its own representative.
fn canonical_nodes(mesh: &UMeshView, tol: f64) -> Vec<usize> {
    let n = mesh.coords().nrows();
    if tol == 0.0 {
        return (0..n).collect();
    }
    let points: Vec<[f64; 3]> = (0..n).map(|i| point3(&mesh.coords(), i)).collect();
    let (_, cluster_of) = weld_points(&points, tol);
    cluster_of
}

/// Midpoint of two nodes.
fn midpoint<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    a: usize,
    b: usize,
) -> [f64; 3] {
    [
        (coords[(a, 0)] + coords[(b, 0)]) / 2.0,
        (coords[(a, 1)] + coords[(b, 1)]) / 2.0,
        (coords[(a, 2)] + coords[(b, 2)]) / 2.0,
    ]
}

/// Centroid of a set of node ids.
fn key_centroid<S: nd::Data<Elem = f64>>(
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
