//! The conformance diagnostic of [`stitch`](super): [`is_conform`], the [`ConformanceReport`]
//! it returns, and the [`ConformanceIssue`]s that carry the findings.
//!
//! Everything in this file is about *telling* whether a mesh needs conformizing, never about
//! conformizing it; the operation itself is [`super::conformize`].

use std::fmt;

use ndarray as nd;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{StitchError, point3, split_phed, supported_cell_type, weld_points};
use crate::element_traits::{ElementGeo, ElementTopo};
use crate::geometry::Polyhedron;
use crate::mesh::{Dimension, ElementId, ElementLike, UMeshView};
use crate::tools::spatial_index::SpatiallyIndexable;

/// Why a mesh is not conformal, and where.
///
/// Every variant carries the cells involved, and the coordinates of the place where the problem
/// is centred, so that a report can be turned into a selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConformanceIssue {
    /// Two nodes lie within `tol` of each other but are not the same node, and the cells using
    /// them do not meet through a full, common face.
    ///
    /// Either one cell uses both nodes (it folds onto itself), or two neighbour cells share only
    /// part of a face: they meet through a face which has some nodes in common and others that
    /// differ. Welding the two nodes into one is what makes the interface conformal.
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
        ConformanceIssue::IntersectingCells { .. } => 2,
        ConformanceIssue::MergedNodes { .. } => 3,
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
    // cluster, so that the shared-face test below compares geometry rather than node numbering.
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

    // Faces keyed by their coincident-node set, so that two cells spelling the same face with
    // different node ids still produce one key.
    let cells: Vec<ElementId> = mesh
        .elements_of_dim(Dimension::D3)
        .map(|c| c.id())
        .collect();
    let mut face_cells: FxHashMap<Vec<usize>, Vec<ElementId>> =
        FxHashMap::with_capacity_and_hasher(cells.len() * 6, FxBuildHasher);
    // For every cell, the keys of its faces, used for the shared-face test.
    let mut cell_faces: FxHashMap<ElementId, Vec<Vec<usize>>> =
        FxHashMap::with_capacity_and_hasher(cells.len(), FxBuildHasher);
    // Nodes of every cell, in coincident coordinates, and the cells reaching each of them.
    let mut node_cells: FxHashMap<usize, Vec<ElementId>> =
        FxHashMap::with_capacity_and_hasher(coords.nrows(), FxBuildHasher);
    for &cell in &cells {
        let (_, conn) = mesh.element(cell).to_poly();
        let mut keys: Vec<Vec<usize>> = Vec::new();
        let mut own_nodes: FxHashMap<usize, ()> = FxHashMap::default();
        for ring in split_phed(&conn) {
            let mut key: Vec<usize> = ring.iter().map(|&n| canonical[n]).collect();
            for &n in &key {
                own_nodes.entry(n).or_insert(());
            }
            key.sort_unstable();
            key.dedup();
            face_cells.entry(key.clone()).or_default().push(cell);
            keys.push(key);
        }
        for &n in own_nodes.keys() {
            node_cells.entry(n).or_default().push(cell);
        }
        cell_faces.insert(cell, keys);
    }

    // A face used by more than two cells cannot be reconciled by refining: report it first, as
    // it hides whatever happens on that face.
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

    // Cells that share a node through *distinct* coincident ids must meet through a full common
    // face; otherwise the interface is only partially shared. Cells that share the very same
    // node id (touching along an edge or a vertex) are never a defect on that account.
    //
    // Walking the cells sharing each coincident node and pairing them reaches every candidate
    // pair once. A pair is kept when it shares no canonical face and the shared node is spelled
    // with two different original ids.
    let share_face = |a: ElementId, b: ElementId| -> bool {
        cell_faces[&a]
            .iter()
            .any(|ka| cell_faces[&b].iter().any(|kb| ka == kb))
    };
    let mut seen_pairs: FxHashMap<(ElementId, ElementId), ()> =
        FxHashMap::with_capacity_and_hasher(cells.len(), FxBuildHasher);
    for (&node, sharing) in &node_cells {
        for i in 0..sharing.len() {
            for j in (i + 1)..sharing.len() {
                let (a, b) = (sharing[i], sharing[j]);
                if a == b || share_face(a, b) {
                    continue;
                }
                let Some((na, nb)) = distinct_ids(mesh, &canonical, node, a, b) else {
                    continue;
                };
                let pair = if a < b { (a, b) } else { (b, a) };
                if seen_pairs.insert(pair, ()).is_some() {
                    continue;
                }
                merged.push(NodeDuplicate {
                    center: midpoint(&coords, na, nb),
                    nodes: [na, nb],
                    cells_a: vec![pair.0],
                    cells_b: vec![pair.1],
                });
            }
        }
    }

    merged.sort_by(|x, y| {
        x.center[0]
            .total_cmp(&y.center[0])
            .then_with(|| x.center[1].total_cmp(&y.center[1]))
            .then_with(|| x.center[2].total_cmp(&y.center[2]))
            .then_with(|| x.nodes.cmp(&y.nodes))
    });
    merged.dedup_by(|x, y| x.nodes == y.nodes && x.cells_a == y.cells_a && x.cells_b == y.cells_b);
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

/// Returns one original node id from `a` and one from `b` that both map to `node`, when the two
/// cells spell that coincident location with *different* ids; `None` when they share the very
/// same id.
fn distinct_ids(
    mesh: &UMeshView,
    canonical: &[usize],
    node: usize,
    a: ElementId,
    b: ElementId,
) -> Option<(usize, usize)> {
    let ids_of = |cell: ElementId| -> Vec<usize> {
        let mut ids: Vec<usize> = mesh
            .element(cell)
            .connectivity()
            .iter()
            .copied()
            .filter(|&n| n != usize::MAX && canonical[n] == node)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let ids_a = ids_of(a);
    let ids_b = ids_of(b);
    let (na, nb) = (
        ids_a.iter().copied().find(|x| !ids_b.contains(x)),
        ids_b.iter().copied().find(|x| !ids_a.contains(x)),
    );
    match (na, nb) {
        (Some(x), Some(y)) => Some((x.min(y), x.max(y))),
        _ => None,
    }
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
