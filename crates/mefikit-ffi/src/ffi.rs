//! The `cxx` bridge: the single source of truth for the C++ API surface.
//!
//! Everything C++ can see is declared in the [`bridge`] module below. The
//! generated header lands at `<target>/cxxbridge/mefikit-ffi/include/mefikit-ffi/src/ffi.rs.h`
//! and is re-exported to users as `<mefikit/mefikit.hpp>`.
//!
//! Method bodies are **not** here: cxx only emits the declarations and resolves
//! each signature to an associated function or free function living in this
//! module's parent. `UMesh` and `TransferOperator` are newtypes declared at the
//! bottom of this file (cxx requires opaque Rust types to be defined in the
//! same crate as the bridge); their methods are implemented in
//! [`crate::mesh`] and [`crate::transfer`].

use std::fmt::{self, Display, Formatter};

use ndarray as nd;

use mefikit::prelude as mf;

// Free functions declared in the bridge are resolved in this module, so the
// crate-root re-export has to be visible here rather than only in `lib`.
pub use crate::transfer::transfer_field;

/// Anything that can go wrong in the C++ API, surfaced as a `rust::Error` C++
/// exception (catch it as `const rust::Error &` and read `what()`).
#[derive(Debug)]
pub enum Error {
    /// A `mefikit::read` / `mefikit::write` failure.
    Io(mf::MefikitIOError),
    /// A rejected argument: missing field, wrong element type, bad shape, ...
    InvalidArgument(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<mf::MefikitIOError> for Error {
    fn from(e: mf::MefikitIOError) -> Self {
        Error::Io(e)
    }
}

impl From<nd::ShapeError> for Error {
    fn from(e: nd::ShapeError) -> Self {
        Error::InvalidArgument(e.to_string())
    }
}

/// An owned mesh, exposed to C++ as an opaque `mefikit::UMesh` held by
/// `rust::Box`.
///
/// The indirection is required by cxx (opaque types must be defined in the
/// bridge's own crate) and is also a useful place to keep the FFI surface stable
/// while `mefikit::UMesh` evolves. The wrapped value is private on purpose: the
/// methods in [`crate::mesh`] check their arguments before handing them to
/// `mefikit`, and a public field would let a Rust caller skip those checks.
pub struct UMesh(pub(crate) mf::UMesh);

/// A transfer operator precomputed between a source and a target mesh.
///
/// This is the expensive half of a transfer: build it once, then reuse it for
/// every field and every time step, as long as the two meshes do not change.
///
/// The bookkeeping fields are not decoration. `mefikit`'s transfer panics — with
/// an `assert!` — when the meshes handed to
/// [`apply_update`](crate::transfer::TransferOperator::apply_update) are not
/// the ones it was built from, and a panic unwinding out of the library would
/// abort the C++ process instead of raising a catchable `rust::Error`. Recording
/// what was prepared lets [`crate::transfer`] check the preconditions itself and
/// report them.
pub struct TransferOperator {
    pub(crate) operator: mf::TransferOperator,
    /// Dimension of the source cells, i.e. the source mesh's topological
    /// dimension at prepare time.
    pub(crate) src_dim: mf::Dimension,
    /// Number of source cells at `src_dim`, matching `TransferOperator::n_src`
    /// in the core.
    pub(crate) n_src: usize,
    /// Shape of the source mesh at prepare time.
    pub(crate) src_shape: MeshShape,
    /// Shape of the target mesh at prepare time.
    pub(crate) tgt_shape: MeshShape,
}

/// Enough of a mesh's shape to notice that a transfer operator went stale.
///
/// Coordinates are not hashed: a mesh edited in place without changing its node
/// or element counts is not detected, which is why the documentation insists the
/// geometry stay fixed for the operator's lifetime.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct MeshShape {
    pub(crate) n_nodes: usize,
    /// Element count of every block, keyed by element type, so that moving a
    /// cell from one type to another is noticed even when the total is the same.
    pub(crate) blocks: Vec<(mf::ElementType, usize)>,
}

impl MeshShape {
    pub(crate) fn of(mesh: &UMesh) -> Self {
        Self {
            n_nodes: mesh.n_nodes(),
            blocks: Self::blocks_of(mesh).collect(),
        }
    }

    fn blocks_of(mesh: &UMesh) -> impl Iterator<Item = (mf::ElementType, usize)> {
        mesh.0.element_types().map(|element_type| {
            let n = mesh.0.block(*element_type).map_or(0, |block| block.len());
            (*element_type, n)
        })
    }

    /// Whether `mesh` still has this shape, without building a [`MeshShape`]
    /// for it first.
    ///
    /// `apply_update` needs this twice per field, so on the path that matters it
    /// allocates nothing; a [`MeshShape`] is only built to describe a mismatch
    /// after one has been found.
    pub(crate) fn matches(&self, mesh: &UMesh) -> bool {
        mesh.n_nodes() == self.n_nodes && Self::blocks_of(mesh).eq(self.blocks.iter().copied())
    }
}

impl std::fmt::Display for MeshShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} nodes and blocks ", self.n_nodes)?;
        for (i, (element_type, n)) in self.blocks.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{element_type:?} x {n}")?;
        }
        Ok(())
    }
}

#[cxx::bridge(namespace = "mefikit")]
pub mod bridge {
    /// The element kinds supported by mefikit. Mirrors
    /// `mefikit::mesh::ElementType`; the conversions live in [`crate::types`].
    enum ElementType {
        VERTEX,
        SEG2,
        SEG3,
        SEG4,
        SPLINE,
        TRI3,
        TRI6,
        TRI7,
        QUAD4,
        QUAD8,
        QUAD9,
        PGON,
        TET4,
        TET10,
        HEX8,
        HEX21,
        PHED,
    }

    /// Topological dimension of an element, or of a field.
    enum Dimension {
        D0,
        D1,
        D2,
        D3,
    }

    /// How a field behaves when the supporting cells change.
    ///
    /// `Intensive` is a per-unit-measure quantity (temperature, pressure) and is
    /// normalized by the target cell measure under `ConservativeP0`.
    /// `Extensive` is a total quantity (mass, energy) and is transferred as a raw sum.
    enum FieldNature {
        Intensive,
        Extensive,
    }

    /// Where inside a target cell the interpolation sample point is taken.
    enum PointLocation {
        Centroid,
        Barycenter,
        StrictInterior,
    }

    /// Kernel turning a neighbour distance into a moving least-squares weight.
    enum DistanceWeighting {
        Constant,
        InverseDistance,
        CompactSupport,
        Gaussian,
    }

    /// Which of the bindings' own input checks to run.
    ///
    /// The checks that cost time proportional to the size of the mesh are the
    /// only ones `Fast` gives up: the scan for non-finite coordinates in
    /// `from_coords`, and the scan for out-of-range node indices in
    /// `add_regular_block` and `add_poly_block`. Everything whose cost does not
    /// grow with the mesh -- the transfer preconditions, the field block layout,
    /// the element type of every block -- runs either way.
    ///
    /// Turning them off means a malformed mesh is not rejected where it was
    /// built. mefikit indexes coordinates and connectivity without checking, so
    /// what used to be an error becomes undefined behaviour; the checks are
    /// there because that is a bad trade, and `Fast` is a deliberate one.
    ///
    /// Set it once at startup: it is a process-wide setting, and the
    /// `MEFIKIT_FFI_CHECKS` environment variable does the same thing before the
    /// first mesh is built.
    enum Checks {
        /// Only the checks that do not grow with the size of the input.
        Fast,
        /// Every check. The default.
        Full,
    }

    /// Discriminant of [`TransferMethod`]; the other fields of that struct are
    /// only read by the variant selected here.
    enum TransferMethodKind {
        ConstantPiecewise,
        ConservativeP0,
        InverseDistance,
        MovingLeastSquares,
    }

    /// How to build a [`TransferOperator`].
    ///
    /// Prefer the ready-made values returned by `mefikit::constant_piecewise()`,
    /// `mefikit::conservative_p0()`, `mefikit::inverse_distance()` and
    /// `mefikit::moving_least_squares()` (inline helpers declared in
    /// `mefikit/mefikit.hpp`) over filling this in by hand, so that the
    /// parameters a variant does not use stay zeroed.
    #[derive(Copy, Clone)]
    struct TransferMethod {
        kind: TransferMethodKind,
        point_location: PointLocation,
        k: usize,
        exponent: f64,
        weighting: DistanceWeighting,
    }

    /// One element type's contribution to [`set_field`](UMesh::set_field).
    ///
    /// `values[offset..offset + len]` must hold `n_elements * n_components`
    /// values in row-major order.
    #[derive(Copy, Clone, Debug)]
    struct FieldBlock {
        element_type: ElementType,
        n_components: usize,
        offset: usize,
        len: usize,
    }

    /// Shape of a field carried by one element type.
    #[derive(Copy, Clone, Debug)]
    struct FieldInfo {
        n_elements: usize,
        n_components: usize,
    }

    extern "Rust" {
        /// An unstructured mesh. Opaque in C++; see `crates/mefikit-ffi/README.md`.
        type UMesh;

        /// A precomputed source -> target interpolation. Opaque in C++.
        type TransferOperator;

        /// Creates an empty mesh from a row-major `(n_nodes, space_dim)` array of
        /// coordinates. Element blocks are added afterwards.
        ///
        /// This is the constructor, spelled `from_coords` rather than `new`
        /// because `new` is a C++ keyword and cannot name a member function.
        #[Self = "UMesh"]
        fn from_coords(coords: &[f64], n_nodes: usize, space_dim: usize) -> Result<Box<UMesh>>;

        /// Adds a fixed-node-count block. `conn` is row-major
        /// `(n_elements, num_nodes(et))`; each element type may appear once.
        fn add_regular_block(
            self: &mut UMesh,
            element_type: ElementType,
            conn: &[usize],
            n_elements: usize,
        ) -> Result<()>;

        /// Adds a variable-node-count block (`PGON`, `PHED`, `SPLINE`).
        /// `conn` is the flat node-index list and `offsets` holds one cumulative
        /// end index per element, so `offsets.back() == conn.size()` and
        /// `offsets.size() == n_elements`.
        fn add_poly_block(
            self: &mut UMesh,
            element_type: ElementType,
            conn: &[usize],
            offsets: &[usize],
        ) -> Result<()>;

        /// Checks the mesh is internally consistent (connectivity in range, field
        /// shapes, poly offsets). Cheap; worth calling on meshes built by hand.
        fn validate_structure(self: &UMesh) -> Result<()>;

        /// Number of nodes.
        fn n_nodes(self: &UMesh) -> usize;
        /// Total number of elements, over all element types.
        fn n_elements(self: &UMesh) -> usize;
        /// Number of elements carried by one element type.
        fn n_elements_of(self: &UMesh, element_type: ElementType) -> usize;
        /// Coordinates are 1, 2 or 3 dimensional.
        fn space_dimension(self: &UMesh) -> usize;
        /// True for a mesh with no element block yet.
        fn is_empty(self: &UMesh) -> bool;
        /// Highest topological dimension present. Reports `Dimension::D0` for an
        /// empty mesh, so check `is_empty()` first if that matters.
        fn topological_dimension(self: &UMesh) -> Dimension;
        /// Element types present, in ascending order.
        fn element_types(self: &UMesh) -> Vec<ElementType>;

        /// Sets the field `name` from one block per element type. Every block must
        /// belong to the same dimension, and that set of element types must match
        /// the mesh's blocks at that dimension exactly.
        fn set_field(
            self: &mut UMesh,
            name: &str,
            blocks: &[FieldBlock],
            values: &[f64],
        ) -> Result<()>;

        /// Convenience for the common case of a single-block mesh: fills the
        /// block of `element_type` with `values` interpreted as
        /// `(values.size() / n_components, n_components)`.
        fn set_field_uniform(
            self: &mut UMesh,
            name: &str,
            element_type: ElementType,
            n_components: usize,
            values: &[f64],
        ) -> Result<()>;

        /// Shape of field `name` on element type `element_type`.
        fn field_info(self: &UMesh, name: &str, element_type: ElementType) -> Result<FieldInfo>;

        /// Zero-copy view of field `name` on element type `element_type`, as a
        /// row-major `(n_elements, n_components)` array.
        ///
        /// The returned slice points into the mesh: the mesh must outlive it, and
        /// no method taking `&mut self` may run while it is alive.
        unsafe fn field_values<'a>(
            self: &'a UMesh,
            name: &str,
            element_type: ElementType,
        ) -> Result<&'a [f64]>;

        /// Names of every field carried by the mesh.
        fn field_names(self: &UMesh) -> Vec<String>;

        /// Precomputes the transfer from `src` to `tgt`. This is the expensive
        /// step: call it once, then `apply_update` as often as needed.
        #[Self = "TransferOperator"]
        fn prepare(
            src: &UMesh,
            tgt: &UMesh,
            method: &TransferMethod,
        ) -> Result<Box<TransferOperator>>;

        /// Transfers field `field_name` of `src` onto `tgt` under
        /// `tgt_field_name`, replacing it if it already exists. `default_value`
        /// fills target cells the source does not cover.
        fn apply_update(
            self: &TransferOperator,
            src: &UMesh,
            field_name: &str,
            tgt: &mut UMesh,
            tgt_field_name: &str,
            default_value: f64,
            nature: FieldNature,
        ) -> Result<()>;

        /// Topological dimension of the cells the operator writes to.
        fn target_dimension(self: &TransferOperator) -> Dimension;

        /// Reads a mesh. The format follows the extension: `.med`, `.vtu`,
        /// `.vtk`, `.vtkhdf`, `.cgns`, `.json`, `.yaml`.
        #[Self = "UMesh"]
        fn read(path: &str) -> Result<Box<UMesh>>;

        /// Writes a mesh, same formats as `read`.
        fn write(self: &UMesh, path: &str) -> Result<()>;

        /// One-shot transfer: prepares an operator and immediately applies it.
        /// Convenient when a single field is transferred once; use
        /// `TransferOperator::prepare` + `apply_update` otherwise.
        fn transfer_field(
            src: &UMesh,
            field_name: &str,
            tgt: &mut UMesh,
            tgt_field_name: &str,
            method: &TransferMethod,
            default_value: f64,
            nature: FieldNature,
        ) -> Result<()>;

        /// Sets how much of the input the bindings check, for the rest of the
        /// process. See [`Checks`]; the default is `Checks::Full`.
        fn set_checks(checks: Checks) -> Result<()>;

        /// The level currently in force.
        fn checks() -> Checks;
    }
}

/// Sets how much of the input the bindings check. See [`Checks`].
///
/// # Errors
///
/// [`Error::InvalidArgument`] if `checks` carries a discriminant this build
/// does not know about; a cxx shared enum is an open wrapper around an integer,
/// so a C++ caller can pass any value.
pub fn set_checks(checks: bridge::Checks) -> Result<(), Error> {
    crate::checks::set(checks)
}

/// The level currently in force. See [`Checks`].
#[must_use]
pub fn checks() -> bridge::Checks {
    crate::checks::get()
}
