//! The conformance diagnostic of [`stitch`](super): [`is_conform`], the [`ConformanceReport`]
//! it returns, and the [`ConformanceIssue`]s that carry the findings.
//!
//! Everything in this file is about *telling* whether a mesh needs conformizing, never about
//! conformizing it; the operation itself is [`super::conformize`].

use std::fmt;

use ndarray as nd;
use rustc_hash::{FxBuildHasher, FxHashMap};

use super::{FaceUse, StitchError, count_volume_faces, point3, supported_cell_type, weld_points};
use crate::mesh::{Dimension, ElementId, ElementLike, UMeshView};

/// Why a mesh is not conformal, and where.
///
/// Every variant carries the cells involved, and the coordinates of the place where the problem
/// is centred, so that a report can be turned into a selection.
#[derive(Clone, Debug, PartialEq)]
pub enum ConformanceIssue {
    /// Two nodes are within `tol` of each other, but the cells that use them meet only through
    /// those two nodes: they should have been welded into one.
    ///
    /// This is the block-on-block case, which [conformize](super::conformize) repairs by making the interface
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
///   meet through a shared, conformal interface; [conformize](super::conformize) welds the pair away;
/// - input that [conformize](super::conformize) would reject (not embedded in 3D, no volume cell, unsupported
///   cell type), reported as a single self-explanatory issue rather than an error, since the
///   missing conformance is then a consequence of the input being unsupported. Such an issue is
///   always included, even when `max_issues` is `Some(0)`.
///
/// Interfaces that overlap without sharing any coincident node are *not* detected: recognizing
/// them costs the geometric imprinting that [conformize](super::conformize) performs, so [`is_conform`] only
/// reports what is cheap to know beforehand. Similarly, duplicate nodes within one part are not
/// reported, because [conformize](super::conformize) leaves them alone.
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

/// Groups the nodes of `mesh` that are within `tol` of each other.
///
/// Returns one list of node ids per cluster of more than one node, in increasing order. The
/// clusters are the very ones [`weld_points`] builds (this only regroups its per-point result),
/// so [`is_conform`] reports exactly the node pairs that [stitch](super::stitch) welds together, and every
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
