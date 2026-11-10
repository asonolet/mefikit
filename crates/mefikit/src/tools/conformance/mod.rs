//! Mesh conformance diagnostics: [`is_conform`], the [`ConformanceReport`] it returns, and the
//! [`ConformanceIssue`]s that carry the findings.
//!
//! Everything in this file is about *telling* whether a mesh needs conformizing, never about
//! conformizing it; the operation itself is [`crate::tools::overlay::conformize`].

use std::fmt;

use ndarray as nd;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::element_traits::{ElementGeo, ElementTopo};
use crate::geometry::{
    Intersection, Intersections, PlaneFrame, Polygon, Polyhedron, Segment, into_ccw2, signed_area2,
};
use crate::mesh::{Dimension, ElementId, ElementLike, ElementType, UMesh, UMeshView};
use crate::tools::overlay::StitchError;
use crate::tools::overlay::stitch::{point3, split_phed, weld_points};
use crate::tools::overlay::surface::{FaceData, PARALLEL_NORMAL_COS_EPS, collect_surface_faces};
use crate::tools::spatial_index::SpatiallyIndexable;

/// Why a mesh is not conformal, and where.
///
/// Every variant carries the cells involved, and the coordinates of the place where the problem
/// is centred, so that a report can be turned into a selection.
#[non_exhaustive]
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
    /// A cell is self-intersecting (a "butterfly"): its boundary crosses itself, for instance
    /// because one of its edges passes through one of its faces. Its measure is not well-defined
    /// until the cell is fixed or split.
    ButterflyCell {
        /// Centroid of the cell.
        center: [f64; 3],
        /// The self-intersecting cell.
        cell: ElementId,
    },
    /// Two cells overlap: their matter intersects although they are distinct cells.
    IntersectingCells {
        /// Centroid of the two cells.
        center: [f64; 3],
        /// The two overlapping cells, `cell_a` first.
        cells: [ElementId; 2],
        /// Measure of the intersection: a volume in 3D, an area in 2D.
        volume: f64,
    },
    /// The mesh is not embedded in a space of high enough dimension for its cells.
    InvalidSpaceDimension {
        /// The spatial dimension that was found.
        found: usize,
    },
    /// The mesh holds a cell that [conformize](crate::tools::overlay::conformize) cannot handle.
    UnsupportedCells {
        /// The offending cells. More than one entry is reported.
        cells: Vec<ElementId>,
    },
    /// The mesh has no cell at all.
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
            Self::ButterflyCell { center, cell } => write!(
                f,
                "cell {} crosses itself around {center:?}; its boundary self-intersects and its \
                 measure is not well-defined",
                fmt_cells(std::slice::from_ref(cell)),
            ),
            Self::IntersectingCells {
                center,
                cells,
                volume,
            } => write!(
                f,
                "cells {} and {} overlap with measure {volume} around {center:?}; cells must not \
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

/// Which conformance checks [`is_conform_with`] runs.
///
/// The checks are independent groups; [`Self::FAST`] runs the cheap structural and per-cell
/// checks, while [`Self::FULL`] adds the two geometric searches over cell pairs. Combine them
/// with `|`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConformanceChecks(u8);

impl ConformanceChecks {
    /// Run no check beyond input validation.
    pub const NONE: Self = Self(0);
    /// Overlapping faces, partially shared faces, and merged nodes.
    pub const TOPOLOGY: Self = Self(1 << 0);
    /// Self-intersecting (butterfly) cells.
    pub const BUTTERFLY: Self = Self(1 << 1);
    /// Cells whose volumes (3D) or areas (2D) overlap.
    pub const INTERSECTIONS: Self = Self(1 << 2);
    /// Coplanar faces (3D) or edges (2D) that overlap without being the same face.
    pub const FACE_OVERLAP: Self = Self(1 << 3);
    /// The fast checks: structural topology and butterfly cells.
    pub const FAST: Self = Self(Self::TOPOLOGY.0 | Self::BUTTERFLY.0);
    /// Every check.
    pub const FULL: Self = Self(Self::FAST.0 | Self::INTERSECTIONS.0 | Self::FACE_OVERLAP.0);

    /// The raw bits, for bindings.
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Builds a set from raw bits, dropping unknown bits.
    pub const fn from_bits_truncate(bits: u8) -> Self {
        Self(bits & Self::FULL.0)
    }

    /// Whether every check in `other` is enabled.
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Whether any check in `other` is enabled.
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// The union of two check sets.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether no check is enabled.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl Default for ConformanceChecks {
    fn default() -> Self {
        Self::FULL
    }
}

impl std::ops::BitOr for ConformanceChecks {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for ConformanceChecks {
    fn bitor_assign(&mut self, rhs: Self) {
        *self = self.union(rhs);
    }
}

/// Options for [`is_conform_with`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConformanceOptions {
    /// The checks to run.
    pub checks: ConformanceChecks,
    /// Cap on `issues.len()`, or `None` to keep every issue.
    pub max_issues: Option<usize>,
}

impl ConformanceOptions {
    /// The given checks, with no issue cap.
    pub const fn new(checks: ConformanceChecks) -> Self {
        Self {
            checks,
            max_issues: None,
        }
    }

    /// The fast checks, with no issue cap.
    pub const fn fast() -> Self {
        Self::new(ConformanceChecks::FAST)
    }

    /// Sets the issue cap.
    pub const fn with_max_issues(mut self, max_issues: Option<usize>) -> Self {
        self.max_issues = max_issues;
        self
    }
}

impl Default for ConformanceOptions {
    fn default() -> Self {
        Self::new(ConformanceChecks::FULL)
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
        ConformanceIssue::OverlappingFaces { .. } | ConformanceIssue::ButterflyCell { .. } => 1,
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

/// The boundary faces of one volume cell, collected once and shared by every check that needs
/// them.
struct VolumeCellFaces {
    id: ElementId,
    rings: Vec<Vec<usize>>,
}

fn collect_volume_faces(mesh: &UMeshView) -> Vec<VolumeCellFaces> {
    mesh.elements_of_dim(Dimension::D3)
        .map(|cell| {
            let (_, conn) = cell.to_poly();
            let rings = split_phed(&conn).into_iter().map(|r| r.to_vec()).collect();
            VolumeCellFaces {
                id: cell.id(),
                rings,
            }
        })
        .collect()
}

/// Shared accumulator for the issue list and its uncapped count.
struct IssueSink<'a> {
    issues: &'a mut Vec<ConformanceIssue>,
    n_issues: &'a mut usize,
    cap: usize,
}

impl IssueSink<'_> {
    fn push(&mut self, issue: ConformanceIssue) {
        *self.n_issues += 1;
        if self.issues.len() < self.cap {
            self.issues.push(issue);
        }
    }
}

/// Checks whether `mesh` is conformal and, when it is not, reports why and where.
///
/// The check follows the mesh's topological dimension. In 3D a mesh is conformal when its cells
/// do not intersect one another and two neighbour cells that share any node share a *whole*
/// common face; in 2D the same rule applies to common edges. Lower-dimensional elements are
/// ignored when higher-dimensional ones are present, so a volume mesh may carry its boundary
/// surface without being rejected.
///
/// The reported defects are:
///
/// - [`ConformanceIssue::ButterflyCell`]: a cell's boundary crosses itself (an edge passes
///   through a face, or two non-adjacent edges of a face cross), so its measure is ill-defined;
/// - [`ConformanceIssue::IntersectingCells`]: two cells overlap in volume (3D) or area (2D);
/// - [`ConformanceIssue::OverlappingFaces`]: a face (3D) or edge (2D) is shared by more than two
///   cells, which leaves the domain ill-defined;
/// - [`ConformanceIssue::NonConformingFaces`]: two cells meet through a face/edge that is only
///   partially shared;
/// - [`ConformanceIssue::MergedNodes`]: two nodes lie within `tol` of each other but are not the
///   same node, and the cells using them do not meet through a full common face (a cell that
///   folds onto itself, or two cells that share only part of a face);
/// - an input that cannot be diagnosed (too few coordinate dimensions, no cell, or a cell type
///   whose polyhedral form is not available), reported as a single self-explanatory issue rather
///   than an error. Such an issue is always included, even when `max_issues` is `Some(0)`.
///
/// Coincident nodes are identified (welded within `tol`) before the shared-face test, so two
/// neighbour cells that carry their own copy of a *complete* interface are conformal; only a
/// *partially* shared interface, where some nodes coincide and others differ, is reported.
///
/// `max_issues` caps the length of the report so that a large mesh cannot produce an unbounded
/// one; `None` keeps everything. [`ConformanceReport::n_issues`] always holds the true total.
///
/// The mesh is only read. See [conformize](crate::tools::overlay::conformize) for the corresponding operation.
pub fn is_conform(
    mesh: &UMeshView,
    tol: f64,
    max_issues: Option<usize>,
) -> Result<ConformanceReport, StitchError> {
    is_conform_with(
        mesh,
        tol,
        ConformanceOptions {
            checks: ConformanceChecks::FULL,
            max_issues,
        },
    )
}

/// Like [`is_conform`], but chooses which checks to run.
///
/// [`ConformanceOptions::default`] runs every check, so this is the general entry point when
/// only a subset is needed. Use [`ConformanceOptions::fast`] for the structural and per-cell
/// checks without the two geometric searches over cell pairs.
pub fn is_conform_with(
    mesh: &UMeshView,
    tol: f64,
    options: ConformanceOptions,
) -> Result<ConformanceReport, StitchError> {
    if !(tol.is_finite() && tol >= 0.0) {
        return Err(StitchError::InvalidTolerance { tol });
    }
    let checks = options.checks;
    let cap = options.max_issues.unwrap_or(usize::MAX);

    let Some(top) = mesh.topological_dimension() else {
        return Ok(single_issue(ConformanceIssue::NoVolumeCells));
    };
    let space = mesh.space_dimension();
    let min_space = match top {
        Dimension::D0 => 0,
        Dimension::D1 => 1,
        Dimension::D2 => 2,
        Dimension::D3 => 3,
    };
    if space < min_space {
        return Ok(single_issue(ConformanceIssue::InvalidSpaceDimension {
            found: space,
        }));
    }

    // Only the top-dimensional cells take part in conformance; lower-dimensional elements are
    // allowed alongside them (boundary surfaces, embedded lines, ...).
    let unsupported: Vec<ElementId> = mesh
        .elements_of_dim(top)
        .filter(|e| !diagnostic_cell_type(e.element_type(), top))
        .map(|e| e.id())
        .take(cap.max(1))
        .collect();
    if !unsupported.is_empty() {
        return Ok(single_issue(ConformanceIssue::UnsupportedCells {
            cells: unsupported,
        }));
    }

    if checks.is_empty() {
        return Ok(ConformanceReport {
            issues: Vec::new(),
            n_issues: 0,
            truncated: false,
        });
    }

    let coords = mesh.coords();
    let mut issues: Vec<ConformanceIssue> = Vec::new();
    let mut n_issues = 0usize;

    let need_faces = checks.intersects(
        ConformanceChecks::TOPOLOGY
            | ConformanceChecks::BUTTERFLY
            | ConformanceChecks::FACE_OVERLAP,
    );
    let cell_faces = if need_faces && top == Dimension::D3 {
        collect_volume_faces(mesh)
    } else {
        Vec::new()
    };

    // Node identification: every node is replaced by the representative of its coincident
    // cluster, so that a cell using two ids for the same location is detected as a fold.
    // Surface cells always need it; volume cells only for the checks that inspect faces.
    let need_canonical = top == Dimension::D2
        || checks.intersects(
            ConformanceChecks::TOPOLOGY
                | ConformanceChecks::BUTTERFLY
                | ConformanceChecks::FACE_OVERLAP,
        );
    let canonical = if need_canonical {
        canonical_nodes(mesh, tol)
    } else {
        Vec::new()
    };

    if top == Dimension::D2 {
        return Ok(is_conform_surface(mesh, tol, cap, &canonical, checks));
    }
    if top != Dimension::D3 {
        // 1D and 0D meshes have no self-intersection or shared-facet notion to check here.
        return Ok(ConformanceReport {
            issues: Vec::new(),
            n_issues: 0,
            truncated: false,
        });
    }

    // Butterfly detection works on the identified nodes, so that a fold spelled with coincident
    // duplicate ids is left to the `MergedNodes` diagnostic instead of being reported as a
    // self-intersection. The set is also needed by the face checks, which skip self-intersecting
    // cells, even when the butterfly issues themselves are not requested.
    let need_butterfly = checks.intersects(
        ConformanceChecks::TOPOLOGY
            | ConformanceChecks::BUTTERFLY
            | ConformanceChecks::FACE_OVERLAP,
    );
    let found_butterflies = if need_butterfly {
        find_butterfly_cells(mesh, tol, &canonical, &cell_faces)
    } else {
        Vec::new()
    };
    let butterfly_cells: FxHashSet<ElementId> =
        found_butterflies.iter().map(|(cell, _)| *cell).collect();
    if checks.contains(ConformanceChecks::BUTTERFLY) {
        for (cell, center) in found_butterflies {
            n_issues += 1;
            if issues.len() < cap {
                issues.push(ConformanceIssue::ButterflyCell { center, cell });
            }
        }
    }

    if checks.contains(ConformanceChecks::INTERSECTIONS) {
        intersecting_cells(mesh, &mut issues, &mut n_issues, cap);
    }

    // Cells that fold onto themselves: two of their nodes are coincident but distinct.
    let mut merged: Vec<NodeDuplicate> = Vec::new();
    if checks.contains(ConformanceChecks::TOPOLOGY) {
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
    }

    // A face spelled with the very same node ids by more than two cells cannot be reconciled by
    // refining: report it first, as it hides whatever happens on that face.
    if checks.contains(ConformanceChecks::TOPOLOGY) {
        let mut face_cells: FxHashMap<Vec<usize>, Vec<ElementId>> =
            FxHashMap::with_capacity_and_hasher(
                mesh.elements_of_dim(Dimension::D3).count() * 6,
                FxBuildHasher,
            );
        for cell in &cell_faces {
            for ring in &cell.rings {
                let mut key = ring.clone();
                key.sort_unstable();
                face_cells.entry(key).or_default().push(cell.id);
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
    }

    // Two cells whose coincident boundary faces overlap geometrically without being the same
    // face have a non-conforming interface (a T-junction, or two different tessellations of the
    // same interface). This is the case the topological checks above cannot see.
    let mut non_conf_pairs: FxHashMap<[ElementId; 2], ()> = FxHashMap::default();
    if checks.intersects(ConformanceChecks::TOPOLOGY | ConformanceChecks::FACE_OVERLAP) {
        let mut sink = IssueSink {
            issues: &mut issues,
            n_issues: &mut n_issues,
            cap,
        };
        if checks.contains(ConformanceChecks::FACE_OVERLAP) {
            non_conforming_faces(
                mesh,
                tol,
                &canonical,
                &butterfly_cells,
                &cell_faces,
                &mut non_conf_pairs,
                &mut sink,
            );
        }

        // Two cells that share three nodes or more but have no face in common meet through a
        // face that is only partially shared: one side spells the interface with a node the
        // other side does not have (a "descending face" T-junction that shares node ids). Two
        // cells touching along an edge or a vertex share fewer than three nodes and are left
        // alone.
        if checks.contains(ConformanceChecks::TOPOLOGY) {
            partially_shared_faces(
                mesh,
                tol,
                &butterfly_cells,
                &cell_faces,
                &mut non_conf_pairs,
                &mut sink,
            );
        }
    }

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

/// Whether a top-dimensional cell type can be analyzed by [`is_conform`].
fn diagnostic_cell_type(et: ElementType, dim: Dimension) -> bool {
    match dim {
        Dimension::D3 => matches!(
            et,
            ElementType::TET4 | ElementType::WEDGE6 | ElementType::HEX8 | ElementType::PHED
        ),
        Dimension::D2 => matches!(
            et,
            ElementType::TRI3
                | ElementType::TRI6
                | ElementType::TRI7
                | ElementType::QUAD4
                | ElementType::QUAD8
                | ElementType::QUAD9
                | ElementType::PGON
        ),
        Dimension::D1 => matches!(
            et,
            ElementType::SEG2 | ElementType::SEG3 | ElementType::SEG4 | ElementType::PLINE
        ),
        Dimension::D0 => et == ElementType::VERTEX,
    }
}

/// Boundary ring of a 2D cell, in winding order.
///
/// Quadratic cells store their mid-side nodes after the corners; the ring interleaves them so
/// that consecutive nodes are actual polygon vertices. A centre node, when present, is excluded.
fn surface_ring(et: ElementType, conn: &[usize]) -> Vec<usize> {
    match et {
        ElementType::TRI3 | ElementType::QUAD4 | ElementType::PGON => conn.to_vec(),
        ElementType::TRI6 | ElementType::TRI7 => {
            vec![conn[0], conn[3], conn[1], conn[4], conn[2], conn[5]]
        }
        ElementType::QUAD8 | ElementType::QUAD9 => vec![
            conn[0], conn[4], conn[1], conn[5], conn[2], conn[6], conn[3], conn[7],
        ],
        _ => Vec::new(),
    }
}

/// One 2D cell with its boundary and precomputed face data.
struct SurfaceCell {
    id: ElementId,
    /// Raw connectivity, before node identification.
    raw: Vec<usize>,
    /// Identified boundary ring.
    ring: Vec<usize>,
    face: FaceData,
}

impl SurfaceCell {
    fn new<S: nd::Data<Elem = f64>>(
        coords: &nd::ArrayBase<S, nd::Ix2>,
        cell: crate::mesh::Element<'_>,
        canonical: &[usize],
        et: ElementType,
    ) -> Self {
        let raw = cell.connectivity().to_vec();
        let ring: Vec<usize> = surface_ring(et, &raw)
            .into_iter()
            .map(|n| canonical[n])
            .collect();
        let pts: Vec<[f64; 3]> = ring.iter().map(|&n| point3(coords, n)).collect();
        let raw_normal = crate::geometry::newell_normal3(&pts);
        let norm = (raw_normal[0].powi(2) + raw_normal[1].powi(2) + raw_normal[2].powi(2)).sqrt();
        let normal = if norm > 0.0 {
            [
                raw_normal[0] / norm,
                raw_normal[1] / norm,
                raw_normal[2] / norm,
            ]
        } else {
            [0.0; 3]
        };
        let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        for p in &pts {
            for k in 0..3 {
                bounds[0][k] = bounds[0][k].min(p[k]);
                bounds[1][k] = bounds[1][k].max(p[k]);
            }
        }
        let face = FaceData {
            id: cell.id(),
            et,
            ring: ring.clone(),
            pts: pts.clone(),
            normal,
            bounds,
        };
        Self {
            id: cell.id(),
            raw,
            ring,
            face,
        }
    }
}

/// One boundary edge of a surface cell.
struct SurfaceEdge {
    owner: ElementId,
    key: [usize; 2],
    a: [f64; 3],
    b: [f64; 3],
}

/// BVH over boundary edges, in the mesh's coordinate dimension.
enum EdgeIndex<'a> {
    D2(&'a crate::tools::spatial_index::SpIdx2),
    D3(&'a crate::tools::spatial_index::SpIdx3),
}

/// Returns a point where two edges overlap or properly cross, or `None` when they only touch at
/// a shared vertex.
///
/// Collinear edges must overlap with a positive length (a T-junction); non-collinear edges must
/// cross away from both endpoints.
fn segments_overlap_center(
    a: &[f64; 3],
    b: &[f64; 3],
    c: &[f64; 3],
    d: &[f64; 3],
    eps: f64,
) -> Option<[f64; 3]> {
    let d1 = sub3(b, a);
    let d2 = sub3(d, c);
    let r = sub3(a, c);
    let a11 = dot3v(&d1, &d1);
    let a22 = dot3v(&d2, &d2);
    if a11 <= 0.0 || a22 <= 0.0 {
        return None;
    }
    let a12 = dot3v(&d1, &d2);
    let denom = a11 * a22 - a12 * a12;

    if denom <= 64.0 * f64::EPSILON * a11 * a22 {
        let ar = cross3(&d1, &r);
        if dot3v(&ar, &ar) / a11 > eps * eps {
            return None;
        }
        let tc = dot3v(&sub3(c, a), &d1) / a11;
        let td = dot3v(&sub3(d, a), &d1) / a11;
        let (lo, hi) = if tc <= td { (tc, td) } else { (td, tc) };
        let lo = lo.max(0.0);
        let hi = hi.min(1.0);
        let slack = eps / a11.sqrt().max(f64::MIN_POSITIVE);
        if hi <= lo + slack {
            return None;
        }
        return Some(add3(a, &scale3(&d1, 0.5 * (lo + hi))));
    }

    let b1 = dot3v(&d1, &r);
    let b2 = dot3v(&d2, &r);
    let s = (a12 * b2 - a22 * b1) / denom;
    let t = (a11 * b2 - a12 * b1) / denom;
    let s_slack = eps / a11.sqrt().max(f64::MIN_POSITIVE);
    let t_slack = eps / a22.sqrt().max(f64::MIN_POSITIVE);
    if !(-s_slack..=1.0 + s_slack).contains(&s) || !(-t_slack..=1.0 + t_slack).contains(&t) {
        return None;
    }
    let p = add3(a, &scale3(&d1, s));
    let q = add3(c, &scale3(&d2, t));
    if dot3v(&sub3(&p, &q), &sub3(&p, &q)) > eps * eps {
        return None;
    }
    let at_end_a = s <= s_slack || s >= 1.0 - s_slack;
    let at_end_b = t <= t_slack || t >= 1.0 - t_slack;
    if at_end_a && at_end_b {
        return None;
    }
    Some(add3(&p, &scale3(&sub3(&q, &p), 0.5)))
}

/// Area of the overlap of two coplanar faces. Exact for convex faces, an approximation
/// otherwise; only the sign matters for the conformance report.
fn coplanar_overlap_area(a: &FaceData, b: &FaceData) -> f64 {
    let frame = PlaneFrame::from_points(&a.pts);
    let mut pa: Vec<[f64; 2]> = a.pts.iter().map(|p| frame.project(p)).collect();
    let mut pb: Vec<[f64; 2]> = b.pts.iter().map(|p| frame.project(p)).collect();
    into_ccw2(&mut pa);
    into_ccw2(&mut pb);
    signed_area2(&clip_convex(&pa, &pb)).abs()
}

/// Sutherland-Hodgman clip of convex `subject` by convex `clip`, both counter-clockwise.
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut output = subject.to_vec();
    for i in 0..clip.len() {
        if output.len() < 3 {
            return Vec::new();
        }
        let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
        let input = output;
        output = Vec::new();
        for k in 0..input.len() {
            let p = input[k];
            let q = input[(k + 1) % input.len()];
            let p_in = cross2(&sub2(b, a), &sub2(p, a)) >= 0.0;
            let q_in = cross2(&sub2(b, a), &sub2(q, a)) >= 0.0;
            match (p_in, q_in) {
                (true, true) => output.push(q),
                (true, false) => output.push(clip_intersection(a, b, p, q)),
                (false, true) => {
                    output.push(clip_intersection(a, b, p, q));
                    output.push(q);
                }
                (false, false) => {}
            }
        }
    }
    output
}

/// Intersection of two segments known to cross, used by [`clip_convex`].
fn clip_intersection(a: [f64; 2], b: [f64; 2], p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    let r = sub2(b, a);
    let s = sub2(q, p);
    let denom = cross2(&r, &s);
    if denom == 0.0 {
        return q;
    }
    let t = cross2(&sub2(p, a), &s) / denom;
    add2(a, &scale2(&r, t))
}

/// Checks a surface mesh: self-crossing cells, overlapping cells, and shared-edge conformance.
fn is_conform_surface(
    mesh: &UMeshView,
    tol: f64,
    cap: usize,
    canonical: &[usize],
    checks: ConformanceChecks,
) -> ConformanceReport {
    let coords = mesh.coords();
    let space = mesh.space_dimension();
    let mut issues: Vec<ConformanceIssue> = Vec::new();
    let mut n_issues = 0usize;
    let cells: Vec<SurfaceCell> = mesh
        .elements_of_dim(Dimension::D2)
        .map(|cell| {
            let et = cell.element_type();
            SurfaceCell::new(&coords, cell, canonical, et)
        })
        .collect();

    // A surface cell is a polygon; it is a butterfly when its boundary crosses itself. The set
    // is needed to suppress cascading face reports even when butterfly issues are not requested.
    let mut butterfly_ids: FxHashSet<ElementId> = FxHashSet::default();
    for cell in &cells {
        if cell_is_butterfly(&coords, std::slice::from_ref(&cell.ring), tol) {
            butterfly_ids.insert(cell.id);
            if checks.contains(ConformanceChecks::BUTTERFLY) {
                n_issues += 1;
                if issues.len() < cap {
                    issues.push(ConformanceIssue::ButterflyCell {
                        center: cell_centroid(&coords, &cell.raw),
                        cell: cell.id,
                    });
                }
            }
        }
    }

    // A cell that uses two ids for one location folds onto itself.
    let mut merged: Vec<NodeDuplicate> = Vec::new();
    if checks.contains(ConformanceChecks::TOPOLOGY) {
        for cell in &cells {
            let mut seen: FxHashMap<usize, usize> =
                FxHashMap::with_capacity_and_hasher(cell.raw.len(), FxBuildHasher);
            let mut repeats: Vec<[usize; 2]> = Vec::new();
            for (pos, &node) in cell.raw.iter().enumerate() {
                if let Some(prev) = seen.insert(cell.ring[pos], pos) {
                    let other = cell.raw[prev];
                    let (a, b) = (other.min(node), other.max(node));
                    if a != b {
                        repeats.push([a, b]);
                    }
                }
            }
            repeats.sort_unstable();
            repeats.dedup();
            for [a, b] in repeats {
                let ids = vec![cell.id];
                merged.push(NodeDuplicate {
                    center: midpoint(&coords, a, b),
                    nodes: [a, b],
                    cells_a: ids.clone(),
                    cells_b: ids,
                });
            }
        }
    }

    let mut seen: FxHashMap<[ElementId; 2], ()> = FxHashMap::default();
    let mut partial: Vec<(ElementId, ElementId, [f64; 3])> = Vec::new();

    // Topological census of the boundary edges.
    if checks.contains(ConformanceChecks::TOPOLOGY) {
        let mut facet_cells: FxHashMap<[usize; 2], Vec<ElementId>> =
            FxHashMap::with_capacity_and_hasher(cells.len() * 4, FxBuildHasher);
        let mut cell_edges: Vec<Vec<[usize; 2]>> = Vec::with_capacity(cells.len());
        for cell in &cells {
            let mut edges = Vec::with_capacity(cell.ring.len());
            for i in 0..cell.ring.len() {
                let a = cell.ring[i];
                let b = cell.ring[(i + 1) % cell.ring.len()];
                let key = if a <= b { [a, b] } else { [b, a] };
                edges.push(key);
                facet_cells.entry(key).or_default().push(cell.id);
            }
            cell_edges.push(edges);
        }
        let mut overlapping: Vec<(&[usize; 2], &Vec<ElementId>)> =
            facet_cells.iter().filter(|(_, cs)| cs.len() > 2).collect();
        overlapping.sort_by_key(|(key, _)| **key);
        for (key, cs) in overlapping {
            n_issues += 1;
            if issues.len() < cap {
                issues.push(ConformanceIssue::OverlappingFaces {
                    center: key_centroid(&coords, key),
                    cells: cs.clone(),
                });
            }
        }

        // Two cells sharing at least two nodes must share a whole edge; otherwise an edge is only
        // partially shared (a T-junction).
        let mut node_cells: FxHashMap<usize, Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(coords.nrows(), FxBuildHasher);
        for (ci, cell) in cells.iter().enumerate() {
            let mut unique = cell.ring.clone();
            unique.sort_unstable();
            unique.dedup();
            for &n in &unique {
                node_cells.entry(n).or_default().push(ci);
            }
        }
        let mut shared: FxHashMap<(usize, usize), usize> = FxHashMap::default();
        for bucket in node_cells.values() {
            for (pos, &a) in bucket.iter().enumerate() {
                for &b in &bucket[(pos + 1)..] {
                    *shared.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
        for ((a, b), count) in shared {
            if count < 2 {
                continue;
            }
            if cell_edges[a].iter().any(|e| cell_edges[b].contains(e)) {
                continue;
            }
            let common: Vec<usize> = cells[a]
                .ring
                .iter()
                .copied()
                .filter(|n| cells[b].ring.contains(n))
                .collect();
            let pair = if cells[a].id < cells[b].id {
                [cells[a].id, cells[b].id]
            } else {
                [cells[b].id, cells[a].id]
            };
            seen.insert(pair, ());
            partial.push((pair[0], pair[1], key_centroid(&coords, &common)));
        }
    }

    // An edge that only partially overlaps another cell's edge is a T-junction, even when the
    // two cells share only one of the large edge's endpoints. Edges are compared geometrically
    // through a small BVH over the boundary segments.
    if checks.contains(ConformanceChecks::FACE_OVERLAP) {
        let mut edge_mesh = UMesh::new(coords.to_shared());
        let mut edge_geo: Vec<SurfaceEdge> = Vec::new();
        for cell in &cells {
            for i in 0..cell.ring.len() {
                let a = cell.ring[i];
                let b = cell.ring[(i + 1) % cell.ring.len()];
                let key = if a <= b { [a, b] } else { [b, a] };
                edge_mesh.add_element(ElementType::PLINE, &[a, b], None);
                edge_geo.push(SurfaceEdge {
                    owner: cell.id,
                    key,
                    a: point3(&coords, a),
                    b: point3(&coords, b),
                });
            }
        }
        let edge_index2;
        let edge_index3;
        let edge_index = if space == 2 {
            edge_index2 = edge_mesh.view().bvh2();
            Some(EdgeIndex::D2(&edge_index2))
        } else {
            edge_index3 = edge_mesh.view().bvh3();
            Some(EdgeIndex::D3(&edge_index3))
        };
        let mut geometric: Vec<(ElementId, ElementId, [f64; 3])> = Vec::new();
        for (i, edge) in edge_geo.iter().enumerate() {
            let pad = tol.max(f64::EPSILON);
            let [min, max] = segment_bounds(&edge.a, &edge.b);
            let found = match edge_index {
                Some(EdgeIndex::D2(index)) => {
                    index.in_bounds([min[0] - pad, min[1] - pad], [max[0] + pad, max[1] + pad])
                }
                Some(EdgeIndex::D3(index)) => index.in_bounds(
                    [min[0] - pad, min[1] - pad, min[2] - pad],
                    [max[0] + pad, max[1] + pad, max[2] + pad],
                ),
                None => continue,
            };
            for other in found.iter() {
                if other.element_type() != ElementType::PLINE {
                    continue;
                }
                let j = other.index();
                if j <= i {
                    continue;
                }
                let other_edge = &edge_geo[j];
                if edge.owner == other_edge.owner || edge.key == other_edge.key {
                    continue;
                }
                let Some(center) =
                    segments_overlap_center(&edge.a, &edge.b, &other_edge.a, &other_edge.b, tol)
                else {
                    continue;
                };
                let pair = if edge.owner < other_edge.owner {
                    [edge.owner, other_edge.owner]
                } else {
                    [other_edge.owner, edge.owner]
                };
                if seen.insert(pair, ()).is_some() {
                    continue;
                }
                geometric.push((pair[0], pair[1], center));
            }
        }
        partial.extend(geometric);
    }

    if checks.intersects(ConformanceChecks::TOPOLOGY | ConformanceChecks::FACE_OVERLAP) {
        partial.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        for (a, b, center) in partial {
            if butterfly_ids.contains(&a) || butterfly_ids.contains(&b) {
                continue;
            }
            n_issues += 1;
            if issues.len() < cap {
                issues.push(ConformanceIssue::NonConformingFaces {
                    center,
                    cells: [a, b],
                });
            }
        }
    }

    // Coplanar cells overlapping with a positive area are intersecting. Non-coplanar surface
    // cells are left to the butterfly/edge checks; a surface folded through itself is reported
    // by the cells it crosses.
    if checks.contains(ConformanceChecks::INTERSECTIONS) {
        let index_by_id: FxHashMap<ElementId, usize> =
            cells.iter().enumerate().map(|(i, c)| (c.id, i)).collect();
        let bvh2;
        let bvh3;
        let index2 = if space == 2 {
            bvh2 = mesh.bvh2();
            Some(&bvh2)
        } else {
            None
        };
        let index3 = if space == 3 {
            bvh3 = mesh.bvh3();
            Some(&bvh3)
        } else {
            None
        };
        let mut pairs: Vec<(ElementId, ElementId, f64, [f64; 3])> = Vec::new();
        for (i, cell) in cells.iter().enumerate() {
            let pad = tol.max(f64::EPSILON);
            let found = if let Some(index) = index2 {
                index.in_bounds(
                    [cell.face.bounds[0][0] - pad, cell.face.bounds[0][1] - pad],
                    [cell.face.bounds[1][0] + pad, cell.face.bounds[1][1] + pad],
                )
            } else if let Some(index) = index3 {
                index.in_bounds(
                    [
                        cell.face.bounds[0][0] - pad,
                        cell.face.bounds[0][1] - pad,
                        cell.face.bounds[0][2] - pad,
                    ],
                    [
                        cell.face.bounds[1][0] + pad,
                        cell.face.bounds[1][1] + pad,
                        cell.face.bounds[1][2] + pad,
                    ],
                )
            } else {
                continue;
            };
            for other in found.iter() {
                if other.element_type().dimension() != Dimension::D2 {
                    continue;
                }
                let Some(&j) = index_by_id.get(&other) else {
                    continue;
                };
                if j <= i {
                    continue;
                }
                if butterfly_ids.contains(&cell.id) || butterfly_ids.contains(&cells[j].id) {
                    continue;
                }
                if !planes_coincident(&cell.face, &cells[j].face, tol) {
                    continue;
                }
                if let Some(center) = faces_overlap(&cell.face, &cells[j].face, tol) {
                    let area = coplanar_overlap_area(&cell.face, &cells[j].face);
                    pairs.push((cell.id, cells[j].id, area, center));
                }
            }
        }
        pairs.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
        for (a, b, measure, center) in pairs {
            n_issues += 1;
            if issues.len() < cap {
                issues.push(ConformanceIssue::IntersectingCells {
                    center,
                    cells: [a, b],
                    volume: measure,
                });
            }
        }
    }

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
    ConformanceReport {
        truncated: n_issues > issues.len(),
        issues,
        n_issues,
    }
}

/// Centroid of the nodes used by one cell.
fn cell_centroid<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    conn: &[usize],
) -> [f64; 3] {
    let mut acc = [0.0; 3];
    for &n in conn {
        let p = point3(coords, n);
        for k in 0..3 {
            acc[k] += p[k];
        }
    }
    let inv = 1.0 / conn.len().max(1) as f64;
    [acc[0] * inv, acc[1] * inv, acc[2] * inv]
}

/// Appends one [`ConformanceIssue::ButterflyCell`] per volume cell whose boundary crosses itself.
///
/// A cell is a butterfly when two of its edges intersect away from a shared vertex, or when an
/// edge passes through a face it does not belong to. Both are found from the cell's face rings:
/// each face is checked on its own (a bow-tie quad) and every edge is tested against the faces it
/// does not touch. Adjacent faces and edges are skipped topologically, so a shared vertex or a
/// shared edge is never reported.
///
/// Returns the self-intersecting cells. Their faces are not simple polygons, so the face-based
/// checks skip them instead of reporting cascading false overlaps next to the real defect.
fn find_butterfly_cells(
    mesh: &UMeshView,
    tol: f64,
    canonical: &[usize],
    cell_faces: &[VolumeCellFaces],
) -> Vec<(ElementId, [f64; 3])> {
    let coords = mesh.coords();
    let mut found: Vec<(ElementId, [f64; 3])> = Vec::new();
    for cell in cell_faces {
        let rings: Vec<Vec<usize>> = cell
            .rings
            .iter()
            .map(|r| r.iter().map(|&n| canonical[n]).collect())
            .collect();
        if cell_is_butterfly(&coords, &rings, tol) {
            found.push((cell.id, mesh.element(cell.id).centroid3()));
        }
    }
    found.sort_by_key(|(id, _)| *id);
    found
}

/// Precomputed geometry of one face of a cell, shared by every edge-versus-face test.
struct FaceGeom {
    /// Node ids of the face ring (already identified).
    ring: Vec<usize>,
    /// Node coordinates, in ring order.
    points: Vec<[f64; 3]>,
    /// Best-fit plane of the face.
    frame: PlaneFrame,
    /// Face ring projected on [`Self::frame`].
    projected: Vec<[f64; 2]>,
    /// Projected ring made counter-clockwise, for the point-in-polygon test.
    poly: Polygon<2>,
    /// Bounding box of the face.
    bounds: [[f64; 3]; 2],
    /// Coordinate magnitude of the face, for relative tolerances.
    scale: f64,
    /// False for a degenerate (zero-area) face, which cannot be pierced meaningfully.
    valid: bool,
}

impl FaceGeom {
    fn new<S: nd::Data<Elem = f64>>(coords: &nd::ArrayBase<S, nd::Ix2>, ring: &[usize]) -> Self {
        let points: Vec<[f64; 3]> = ring.iter().map(|&n| point3(coords, n)).collect();
        let scale = points
            .iter()
            .flat_map(|p| p.iter())
            .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
        let raw_normal = crate::geometry::newell_normal3(&points);
        let normal_norm =
            (raw_normal[0].powi(2) + raw_normal[1].powi(2) + raw_normal[2].powi(2)).sqrt();
        let frame = PlaneFrame::from_points(&points);
        let projected: Vec<[f64; 2]> = points.iter().map(|p| frame.project(p)).collect();
        let poly = Polygon::unknown(projected.iter().copied()).into_ccw();
        let mut bounds = [[f64::INFINITY; 3], [f64::NEG_INFINITY; 3]];
        for p in &points {
            for k in 0..3 {
                bounds[0][k] = bounds[0][k].min(p[k]);
                bounds[1][k] = bounds[1][k].max(p[k]);
            }
        }
        Self {
            ring: ring.to_vec(),
            points,
            frame,
            projected,
            poly,
            bounds,
            scale,
            valid: normal_norm > 64.0 * f64::EPSILON * scale * scale,
        }
    }
}

/// Tells whether the closed surface made of `rings` crosses itself.
fn cell_is_butterfly<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    rings: &[Vec<usize>],
    tol: f64,
) -> bool {
    let faces: Vec<FaceGeom> = rings
        .iter()
        .map(|ring| FaceGeom::new(coords, ring))
        .collect();

    // A face may already be self-intersecting, as in a bow-tie quad.
    if faces
        .iter()
        .any(|face| face_self_intersects(&face.points, tol))
    {
        return true;
    }

    // Any other self-intersection has an edge crossing a face. Faces sharing a node with that
    // edge are its incident faces (or the faces around one of its endpoints), so they are skipped.
    for (fi, face) in faces.iter().enumerate() {
        if !face.valid {
            continue;
        }
        let n = face.ring.len();
        for i in 0..n {
            let a = face.ring[i];
            let b = face.ring[(i + 1) % n];
            let pa = face.points[i];
            let pb = face.points[(i + 1) % n];
            let [emin, emax] = segment_bounds(&pa, &pb);
            for (gi, other) in faces.iter().enumerate() {
                if gi == fi
                    || !other.valid
                    || other.ring.iter().any(|&n| n == a || n == b)
                    || !boxes_overlap3(&emin, &emax, &other.bounds, tol)
                {
                    continue;
                }
                if edge_pierces_face(&pa, &pb, other, tol) {
                    return true;
                }
            }
        }
    }
    false
}

/// Bounding box of a segment.
fn segment_bounds(a: &[f64; 3], b: &[f64; 3]) -> [[f64; 3]; 2] {
    let mut lo = [0.0; 3];
    let mut hi = [0.0; 3];
    for k in 0..3 {
        lo[k] = a[k].min(b[k]);
        hi[k] = a[k].max(b[k]);
    }
    [lo, hi]
}

/// Whether two 3D bounding boxes overlap once padded by `pad`.
fn boxes_overlap3(a0: &[f64; 3], a1: &[f64; 3], b: &[[f64; 3]; 2], pad: f64) -> bool {
    (0..3).all(|k| a0[k] - pad <= b[1][k] && a1[k] + pad >= b[0][k])
}

/// Tells whether two non-adjacent edges of a face cross each other.
fn face_self_intersects(points: &[[f64; 3]], tol: f64) -> bool {
    let n = points.len();
    if n < 4 {
        return false;
    }
    let scale = points
        .iter()
        .flat_map(|p| p.iter())
        .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
    let eps = tol.max(64.0 * f64::EPSILON * scale);
    for i in 0..n {
        for j in (i + 2)..n {
            // The last and first edges share a vertex and are adjacent.
            if i == 0 && j == n - 1 {
                continue;
            }
            if segments_intersect3(
                &points[i],
                &points[(i + 1) % n],
                &points[j],
                &points[(j + 1) % n],
                eps,
            ) {
                return true;
            }
        }
    }
    false
}

/// Tells whether two 3D segments share a point, within `eps`.
///
/// Non-parallel segments are intersected through their closest points; parallel ones are reported
/// only when they are collinear and overlap.
fn segments_intersect3(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3], eps: f64) -> bool {
    let d1 = sub3(b, a);
    let d2 = sub3(d, c);
    let r = sub3(a, c);
    let a11 = dot3v(&d1, &d1);
    let a22 = dot3v(&d2, &d2);
    if a11 <= 0.0 || a22 <= 0.0 {
        return false;
    }
    let a12 = dot3v(&d1, &d2);
    let b1 = dot3v(&d1, &r);
    let b2 = dot3v(&d2, &r);
    let denom = a11 * a22 - a12 * a12;

    if denom <= 64.0 * f64::EPSILON * a11 * a22 {
        // Parallel segments: only a collinear overlap counts. `c` must lie on the line through
        // `a` along `d1`.
        let ar = cross3(&d1, &r);
        let line_dist2 = dot3v(&ar, &ar) / a11;
        if line_dist2 > eps * eps {
            return false;
        }
        let tc = dot3v(&sub3(c, a), &d1) / a11;
        let td = dot3v(&sub3(d, a), &d1) / a11;
        let (lo, hi) = if tc <= td { (tc, td) } else { (td, tc) };
        let slack = eps / a11.sqrt().max(f64::MIN_POSITIVE);
        return hi >= -slack && lo <= 1.0 + slack;
    }

    let s = (a12 * b2 - a22 * b1) / denom;
    let t = (a11 * b2 - a12 * b1) / denom;
    let slack = 1e-12;
    if !(-slack..=1.0 + slack).contains(&s) || !(-slack..=1.0 + slack).contains(&t) {
        return false;
    }
    let p = add3(a, &scale3(&d1, s));
    let q = add3(c, &scale3(&d2, t));
    dot3v(&sub3(&p, &q), &sub3(&p, &q)) <= eps * eps
}

/// Tells whether the segment `a`-`b` crosses `face` away from that face's boundary nodes.
///
/// The segment is first cut at the face's plane. A segment lying in the plane is tested against
/// the projected polygon directly, which catches coplanar folds as well.
fn edge_pierces_face(a: &[f64; 3], b: &[f64; 3], face: &FaceGeom, tol: f64) -> bool {
    let scale = face
        .scale
        .max(a.iter().fold(1.0_f64, |acc, &c| acc.max(c.abs())))
        .max(b.iter().fold(1.0_f64, |acc, &c| acc.max(c.abs())));
    let eps = tol.max(64.0 * f64::EPSILON * scale);
    let n = face.frame.normal();
    let origin = face.frame.origin();
    let da = dot3v(&sub3(a, &origin), &n);
    let db = dot3v(&sub3(b, &origin), &n);

    if da.abs() <= eps && db.abs() <= eps {
        let qa = face.frame.project(a);
        let qb = face.frame.project(b);
        return coplanar_segment_hits_face(qa, qb, face, eps);
    }
    if (da > eps && db > eps) || (da < -eps && db < -eps) {
        return false;
    }
    let t = da / (da - db);
    let p = add3(a, &scale3(&sub3(b, a), t));
    face.poly.contains_stable(&face.frame.project(&p))
}

/// Tells whether the segment `qa`-`qb`, lying in a face's plane, enters its projected polygon.
///
/// Only a strict interior part of the segment counts. Touching or running along the polygon
/// boundary is a T-junction between neighbouring coplanar faces, not a self-intersection.
fn coplanar_segment_hits_face(qa: [f64; 2], qb: [f64; 2], face: &FaceGeom, eps: f64) -> bool {
    if point_strictly_inside(qa, face, eps) || point_strictly_inside(qb, face, eps) {
        return true;
    }

    // Cut the segment at every boundary it meets, then probe the middle of each piece.
    let mut ts = vec![0.0, 1.0];
    let r = sub2(qb, qa);
    let n = face.projected.len();
    for i in 0..n {
        let a = face.projected[i];
        let b = face.projected[(i + 1) % n];
        if let Some(t) = segment_intersection_param(qa, qb, a, b, eps)
            && t > 0.0
            && t < 1.0
        {
            ts.push(t);
        }
    }
    ts.sort_by(f64::total_cmp);
    for w in ts.windows(2) {
        let t = 0.5 * (w[0] + w[1]);
        let p = add2(qa, &scale2(&r, t));
        if point_strictly_inside(p, face, eps) {
            return true;
        }
    }
    false
}

/// Whether `p` is inside the face polygon and away from its boundary by more than `eps`.
fn point_strictly_inside(p: [f64; 2], face: &FaceGeom, eps: f64) -> bool {
    if !face.poly.contains_stable(&p) {
        return false;
    }
    let n = face.projected.len();
    for i in 0..n {
        let a = face.projected[i];
        let b = face.projected[(i + 1) % n];
        if point_segment_distance2(p, a, b) <= eps * eps {
            return false;
        }
    }
    true
}

/// Squared distance from `p` to the segment `a`-`b`.
fn point_segment_distance2(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub2(b, a);
    let ap = sub2(p, a);
    let denom = dot2(&ab, &ab);
    let t = if denom > 0.0 {
        (dot2(&ap, &ab) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let d = sub2(p, add2(a, &scale2(&ab, t)));
    dot2(&d, &d)
}

/// Parameter along `p`-`q` where it meets the line `a`-`b`, if the two segments intersect.
fn segment_intersection_param(
    p: [f64; 2],
    q: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    eps: f64,
) -> Option<f64> {
    let r = sub2(q, p);
    let s = sub2(b, a);
    let denom = cross2(&r, &s);
    let scale = (dot2(&r, &r) * dot2(&s, &s)).sqrt();
    if denom.abs() <= 64.0 * f64::EPSILON * scale {
        return None;
    }
    let ap = sub2(a, p);
    let t = cross2(&ap, &s) / denom;
    let u = cross2(&ap, &r) / denom;
    let t_slack = eps / dot2(&r, &r).sqrt().max(f64::MIN_POSITIVE);
    let u_slack = eps / dot2(&s, &s).sqrt().max(f64::MIN_POSITIVE);
    if (-t_slack..=1.0 + t_slack).contains(&t) && (-u_slack..=1.0 + u_slack).contains(&u) {
        Some(t)
    } else {
        None
    }
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add2(a: [f64; 2], b: &[f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale2(a: &[f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}

fn dot2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: &[f64; 2], b: &[f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale3(a: &[f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

fn dot3v(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Builds a cell polyhedron with outward face winding when the cell is convex and inverted.
///
/// The intersection-volume algorithm assumes outward-oriented faces. Orientation is not part of
/// conformance, so an otherwise valid but inverted cell must not produce a false intersection.
fn oriented_polyhedron(mesh: &UMeshView, cell: ElementId) -> Polyhedron {
    let elem = mesh.element(cell);
    let mut poly = elem.to_polyhedron();
    let inverted = match elem.element_type() {
        ElementType::TET4 | ElementType::WEDGE6 | ElementType::HEX8 => elem.measure3() < 0.0,
        _ => poly.is_convex() && poly.signed_volume() < 0.0,
    };
    if inverted {
        poly.into_ccw();
    }
    poly
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
        let poly = oriented_polyhedron(mesh, cell);
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
                    .or_insert_with(|| oriented_polyhedron(mesh, other));
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
    canonical: &[usize],
    butterfly: &FxHashSet<ElementId>,
    cell_faces: &[VolumeCellFaces],
    seen: &mut FxHashMap<[ElementId; 2], ()>,
    sink: &mut IssueSink<'_>,
) {
    let coords = mesh.coords();
    let mut faces = UMesh::new(coords.to_shared());
    let mut owner: Vec<ElementId> = Vec::new();
    let mut keys: Vec<Vec<usize>> = Vec::new();
    for cell in cell_faces {
        for ring in &cell.rings {
            faces.add_element(ElementType::PGON, ring, None);
            owner.push(cell.id);
            // Compare identified nodes: a complete interface spelled with separate coincident
            // ids is still one face, while a partial interface keeps a different node set.
            let mut key: Vec<usize> = ring.iter().map(|&n| canonical[n]).collect();
            key.sort_unstable();
            key.dedup();
            keys.push(key);
        }
    }
    let Ok(mut fdata) = collect_surface_faces(&faces.view()) else {
        return;
    };
    // Use identified node ids for the shared-endpoint test, matching the rest of the diagnostic:
    // coincident nodes within `tol` are the same node, while nodes farther apart remain distinct.
    for face in &mut fdata {
        for node in &mut face.ring {
            *node = canonical[*node];
        }
    }
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
                if butterfly.contains(&owner[i]) || butterfly.contains(&owner[j]) {
                    continue;
                }
                let fj = &fdata[j];
                if !planes_coincident(fi, fj, tol) {
                    continue;
                }
                let Some(center) = faces_overlap(fi, fj, tol) else {
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
        sink.push(ConformanceIssue::NonConformingFaces {
            center,
            cells: [a, b],
        });
    }
}

/// Appends one [`ConformanceIssue::NonConformingFaces`] per pair of distinct cells that share
/// three non-collinear nodes or more without sharing a whole face.
///
/// Two cells meeting along a full face spell it with the same node set; two cells meeting only
/// along an edge or a vertex share two nodes or one. Any pair in between shares a *part* of a
/// face — the interface is only partially shared, and one side carries a node the other lacks —
/// so the mesh is not conformal. Node ids are compared as given: coincident-but-distinct ids are
/// handled by the geometric check, this one catches the case the two sides share ids.
///
/// Three shared nodes on one line are only a shared edge, with an extra midpoint on it; that is
/// still conformal. A partial face needs the common nodes to span a positive area.
fn partially_shared_faces(
    mesh: &UMeshView,
    tol: f64,
    butterfly: &FxHashSet<ElementId>,
    cell_faces: &[VolumeCellFaces],
    seen: &mut FxHashMap<[ElementId; 2], ()>,
    sink: &mut IssueSink<'_>,
) {
    let coords = mesh.coords();
    // The faces of every cell, indexed both by node and by node triple. Which index gives the
    // smaller candidate set depends on the mesh: isolated polyhedra have almost no shared nodes,
    // while a fine hex/wedge grid has many faces around each node. Build the cheap node index
    // first, then choose the strategy with the lower estimated pair count.
    let n_faces: usize = cell_faces.iter().map(|cell| cell.rings.len()).sum();
    let mut faces: Vec<(ElementId, Vec<usize>)> = Vec::with_capacity(n_faces);
    let mut face_nodes: Vec<Vec<usize>> = Vec::with_capacity(n_faces);
    let mut node_faces: FxHashMap<usize, Vec<usize>> =
        FxHashMap::with_capacity_and_hasher(coords.nrows(), FxBuildHasher);
    let mut triple_work = 0usize;
    for cell in cell_faces {
        for ring in &cell.rings {
            let idx = faces.len();
            let mut nodes = ring.clone();
            nodes.sort_unstable();
            nodes.dedup();
            for &n in &nodes {
                node_faces.entry(n).or_default().push(idx);
            }
            let k = nodes.len();
            if k >= 3 {
                triple_work =
                    triple_work.saturating_add(k.saturating_mul(k - 1).saturating_mul(k - 2) / 6);
            }
            faces.push((cell.id, ring.clone()));
            face_nodes.push(nodes);
        }
    }
    let node_work: usize = node_faces
        .values()
        .map(|bucket| bucket.len().saturating_mul(bucket.len().saturating_sub(1)) / 2)
        .sum();

    // A triple insertion allocates a bucket and is more expensive than a node-pair count, so
    // prefer the node index unless it is much larger. The factor was measured on a PHED mesh
    // with isolated cells and on a fine wedge grid with many faces per node.
    let mut candidates: Vec<(usize, usize)> = if node_work <= triple_work.saturating_mul(4) {
        let mut shared: FxHashMap<(usize, usize), usize> =
            FxHashMap::with_capacity_and_hasher(faces.len(), FxBuildHasher);
        for bucket in node_faces.values() {
            for (pos, &a) in bucket.iter().enumerate() {
                for &b in &bucket[(pos + 1)..] {
                    *shared.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
        }
        shared
            .into_iter()
            .filter(|(_, count)| *count >= 3)
            .map(|(pair, _)| pair)
            .collect()
    } else {
        let mut triple_faces: FxHashMap<[usize; 3], Vec<usize>> =
            FxHashMap::with_capacity_and_hasher(triple_work, FxBuildHasher);
        for (idx, nodes) in face_nodes.iter().enumerate() {
            for i in 0..nodes.len().saturating_sub(2) {
                for j in (i + 1)..nodes.len().saturating_sub(1) {
                    for k in (j + 1)..nodes.len() {
                        triple_faces
                            .entry([nodes[i], nodes[j], nodes[k]])
                            .or_default()
                            .push(idx);
                    }
                }
            }
        }
        let mut candidate_pairs: FxHashSet<(usize, usize)> = FxHashSet::default();
        for bucket in triple_faces.values() {
            for (pos, &a) in bucket.iter().enumerate() {
                for &b in &bucket[(pos + 1)..] {
                    candidate_pairs.insert((a.min(b), a.max(b)));
                }
            }
        }
        candidate_pairs.into_iter().collect()
    };
    candidates.sort_unstable();

    let mut pairs: Vec<(ElementId, ElementId, [f64; 3])> = Vec::new();
    for (lo, hi) in candidates {
        let (owner_a, _) = &faces[lo];
        let (owner_b, _) = &faces[hi];
        if owner_a == owner_b {
            continue;
        }
        if butterfly.contains(owner_a) || butterfly.contains(owner_b) {
            continue;
        }
        // A whole shared face is exactly a common node set; a partial one is not.
        let nodes_a = &face_nodes[lo];
        let nodes_b = &face_nodes[hi];
        if nodes_a == nodes_b {
            continue;
        }
        let mut common: Vec<usize> = Vec::with_capacity(nodes_a.len().min(nodes_b.len()));
        let (mut ia, mut ib) = (0, 0);
        while ia < nodes_a.len() && ib < nodes_b.len() {
            match nodes_a[ia].cmp(&nodes_b[ib]) {
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
                std::cmp::Ordering::Equal => {
                    common.push(nodes_a[ia]);
                    ia += 1;
                    ib += 1;
                }
            }
        }
        if common.len() < 3 || !common_nodes_span_area(&coords, &common, tol) {
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
        pairs.push((pair[0], pair[1], key_centroid(&coords, &common)));
    }
    pairs.sort_by(|x, y| x.0.cmp(&y.0).then_with(|| x.1.cmp(&y.1)));
    for (a, b, center) in pairs {
        sink.push(ConformanceIssue::NonConformingFaces {
            center,
            cells: [a, b],
        });
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
///
/// A shared vertex or edge alone is not an overlap: the witness must be a vertex strictly inside
/// the other face, away from its boundary, or a proper crossing of two edges. Boundary-sensitive
/// containment alone is not enough here, because two neighbouring faces naturally share an edge
/// and a face of a self-intersecting cell can make an outside vertex look contained.
///
/// A face can also lie inside another one with all its vertices on the other's boundary (for
/// example a triangle tiling half of a quad). There is no interior witness and no proper edge
/// crossing then, so convex pairs fall back to their clipped area.
fn faces_overlap(a: &FaceData, b: &FaceData, tol: f64) -> Option<[f64; 3]> {
    let frame = PlaneFrame::from_points(&a.pts);
    let mut pa: Vec<[f64; 2]> = a.pts.iter().map(|p| frame.project(p)).collect();
    let mut pb: Vec<[f64; 2]> = b.pts.iter().map(|p| frame.project(p)).collect();
    into_ccw2(&mut pa);
    into_ccw2(&mut pb);
    let scale = a
        .pts
        .iter()
        .chain(&b.pts)
        .flat_map(|p| p.iter())
        .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
    let eps = tol.max(64.0 * f64::EPSILON * scale);

    let poly_a = Polygon::unknown(pa.iter().copied());
    let poly_b = Polygon::unknown(pb.iter().copied());
    if let Some(p) = pb.iter().find(|p| strictly_inside(&poly_a, &pa, p, eps)) {
        return Some(frame.deproject(p));
    }
    if let Some(p) = pa.iter().find(|p| strictly_inside(&poly_b, &pb, p, eps)) {
        return Some(frame.deproject(p));
    }
    for i in 0..pa.len() {
        let a0 = a.ring[i];
        let a1 = a.ring[(i + 1) % pa.len()];
        let s1 = Segment::new(pa[i], pa[(i + 1) % pa.len()]);
        for j in 0..pb.len() {
            let b0 = b.ring[j];
            let b1 = b.ring[(j + 1) % pb.len()];
            // Edges incident to the same node meet at that node, not at a proper crossing. The
            // intersection computation is ill-conditioned for nearly collinear edges, so relying
            // on its endpoint classification alone produces false crossings. Compare node ids,
            // not coordinates: distinct close nodes may still describe a real small overlap.
            if a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 {
                continue;
            }
            let s2 = Segment::new(pb[j], pb[(j + 1) % pb.len()]);
            if let Intersections::One(Intersection::New(p)) = s1.intersect(s2) {
                return Some(frame.deproject(&p));
            }
        }
    }
    if poly_a.is_convex() && poly_b.is_convex() {
        let clipped = clip_convex(&pa, &pb);
        if clipped.len() >= 3 && signed_area2(&clipped).abs() > 1e-12 * scale * scale {
            let center = Polygon::unknown(clipped.iter().copied()).geometric_centroid();
            return Some(frame.deproject(&center));
        }
    }
    None
}

/// Whether `p` is inside `poly` and farther than `eps` from every edge of `ring`.
fn strictly_inside(poly: &Polygon<2>, ring: &[[f64; 2]], p: &[f64; 2], eps: f64) -> bool {
    if !poly.contains(p) {
        return false;
    }
    let n = ring.len();
    (0..n).all(|i| point_segment_distance2(*p, ring[i], ring[(i + 1) % n]) > eps * eps)
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

/// Maps every node of `mesh` to the node id representing its coincident cluster.
///
/// Nodes at most `tol` apart (Chebyshev distance) share a representative. With `tol == 0`, every
/// node is its own representative.
fn canonical_nodes(mesh: &UMeshView, tol: f64) -> Vec<usize> {
    let n = mesh.coords().nrows();
    if tol == 0.0 {
        return (0..n).collect();
    }
    let points: Vec<[f64; 3]> = (0..n).map(|i| point3(&mesh.coords(), i)).collect();
    let (unique, cluster_of) = weld_points(&points, tol);

    // `weld_points` numbers clusters densely, so its cluster ids are not node ids. Map each
    // cluster back to its representative node (the lowest input index), which is what the
    // face-based checks need.
    let mut representative = vec![usize::MAX; unique.len()];
    for (node, &cluster) in cluster_of.iter().enumerate() {
        representative[cluster] = representative[cluster].min(node);
    }
    cluster_of
        .iter()
        .map(|&cluster| representative[cluster])
        .collect()
}

/// Midpoint of two nodes.
fn midpoint<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    a: usize,
    b: usize,
) -> [f64; 3] {
    let z = if coords.ncols() > 2 {
        (coords[(a, 2)] + coords[(b, 2)]) / 2.0
    } else {
        0.0
    };
    [
        (coords[(a, 0)] + coords[(b, 0)]) / 2.0,
        (coords[(a, 1)] + coords[(b, 1)]) / 2.0,
        z,
    ]
}

/// Whether the given nodes span a positive area, i.e. are not all collinear.
///
/// Used to tell a partially shared face from a shared edge that merely carries an extra
/// collinear node. The tolerance is a length: a deviation from the line below `tol` is ignored.
fn common_nodes_span_area<S: nd::Data<Elem = f64>>(
    coords: &nd::ArrayBase<S, nd::Ix2>,
    nodes: &[usize],
    tol: f64,
) -> bool {
    if nodes.len() < 3 {
        return false;
    }
    let origin = point3(coords, nodes[0]);
    let diffs: Vec<[f64; 3]> = nodes[1..]
        .iter()
        .map(|&n| sub3(&point3(coords, n), &origin))
        .collect();
    let scale = nodes
        .iter()
        .map(|&n| point3(coords, n))
        .flat_map(|p| p.into_iter())
        .fold(1.0_f64, |acc, c| acc.max(c.abs()));
    let eps = tol.max(64.0 * f64::EPSILON * scale);
    let extent = diffs
        .iter()
        .map(|d| dot3v(d, d).sqrt())
        .fold(0.0_f64, f64::max);
    if extent <= eps {
        return false;
    }
    let Some(dir) = diffs.iter().find(|d| dot3v(d, d).sqrt() > eps) else {
        return false;
    };
    let area_eps = eps * extent;
    diffs.iter().any(|d| {
        let n = cross3(dir, d);
        dot3v(&n, &n).sqrt() > area_eps
    })
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

#[cfg(test)]
mod tests;
