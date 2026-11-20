//! The cxx bridge: the complete C++ API surface.
//!
//! The bridge only declares the two concepts requested for the wrapper: a mesh
//! (`UMesh`) and a conservative P0 transfer (`ConservativeTransfer`). Method
//! bodies live in [`crate::mesh`] and [`crate::transfer`].

use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

use ndarray as nd;

use mefikit::prelude as mf;

/// An error crossing the C++ boundary as a `rust::Error`.
#[derive(Debug)]
pub enum Error {
    /// Invalid input or a precondition of the conservative transfer.
    InvalidArgument(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidArgument(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {}

impl From<nd::ShapeError> for Error {
    fn from(error: nd::ShapeError) -> Self {
        Self::InvalidArgument(error.to_string())
    }
}

/// An owned mefikit mesh, opaque to C++.
pub struct UMesh {
    pub(crate) mesh: mf::UMesh,
    /// Field blocks staged until every element type at their dimension is set.
    pub(crate) pending_fields:
        BTreeMap<String, BTreeMap<mf::ElementType, nd::ArcArray<f64, nd::IxDyn>>>,
}

/// The conservative P0 operator, prepared once and applied many times.
pub struct ConservativeTransfer {
    pub(crate) operator: mf::TransferOperator,
    pub(crate) default_value: f64,
    pub(crate) src_dim: mf::Dimension,
    pub(crate) n_src: usize,
    pub(crate) src_shape: MeshShape,
    pub(crate) tgt_shape: MeshShape,
}

/// Node and block counts used to reject a mesh that changed after preparation.
pub(crate) struct MeshShape {
    n_nodes: usize,
    blocks: Vec<(mf::ElementType, usize)>,
}

impl MeshShape {
    pub(crate) fn of(mesh: &UMesh) -> Self {
        Self {
            n_nodes: mesh.mesh.coords().nrows(),
            blocks: mesh
                .mesh
                .element_types()
                .map(|element_type| {
                    let n = mesh
                        .mesh
                        .block(*element_type)
                        .map_or(0, |block| block.len());
                    (*element_type, n)
                })
                .collect(),
        }
    }

    pub(crate) fn matches(&self, mesh: &UMesh) -> bool {
        self.n_nodes == mesh.mesh.coords().nrows()
            && self
                .blocks
                .iter()
                .copied()
                .eq(mesh.mesh.element_types().map(|element_type| {
                    (
                        *element_type,
                        mesh.mesh
                            .block(*element_type)
                            .map_or(0, |block| block.len()),
                    )
                }))
    }
}

#[cxx::bridge(namespace = "mefikit")]
pub mod bridge {
    /// Element kinds understood by mefikit.
    enum ElementType {
        VERTEX,
        SEG2,
        SEG3,
        SEG4,
        PLINE,
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
        WEDGE6,
        WEDGE15,
        PHED,
    }

    /// How a field behaves under the conservative transfer.
    enum FieldNature {
        Intensive,
        Extensive,
    }

    extern "Rust" {
        /// An unstructured mesh. Construct it with `from_coords`, then add
        /// regular or poly element blocks.
        type UMesh;

        /// A conservative P0 transfer prepared for a source and a target mesh.
        type ConservativeTransfer;

        /// Creates an empty mesh from a row-major `(n_nodes, space_dim)` array.
        #[Self = "UMesh"]
        fn from_coords(
            coords: &CxxVector<f64>,
            n_nodes: usize,
            space_dim: usize,
        ) -> Result<Box<UMesh>>;

        /// Adds a fixed-node-count element block. `conn` is row-major
        /// `(n_elements, num_nodes(element_type))`.
        fn add_regular_block(
            self: &mut UMesh,
            element_type: ElementType,
            conn: &CxxVector<usize>,
            n_elements: usize,
        ) -> Result<()>;

        /// Adds a variable-node-count (`PGON`, `PHED`, `PLINE`) block.
        ///
        /// `conn` is the flat node list and `offsets` holds one cumulative end
        /// index per element, so `offsets.size()` is the element count.
        fn add_poly_block(
            self: &mut UMesh,
            element_type: ElementType,
            conn: &CxxVector<usize>,
            offsets: &CxxVector<usize>,
        ) -> Result<()>;

        /// Sets field `name` on one element type.
        ///
        /// `values` is row-major `(num_elements(element_type), n_components)`.
        /// Calling this once per element type builds a field spanning several
        /// blocks at the same topological dimension.
        fn set_field(
            self: &mut UMesh,
            name: &str,
            element_type: ElementType,
            n_components: usize,
            values: &CxxVector<f64>,
        ) -> Result<()>;

        /// Returns field `name` on one element type as row-major
        /// `(num_elements(element_type), n_components)` values.
        fn field_values(self: &UMesh, name: &str, element_type: ElementType) -> Result<Vec<f64>>;

        /// Number of cells carried by one element type, or 0 if the block is absent.
        fn num_elements(self: &UMesh, element_type: ElementType) -> usize;

        /// Prepares a conservative P0 transfer from `src` to `tgt`.
        ///
        /// This is the expensive step. The returned operator can be applied to
        /// any number of source fields as long as the meshes do not change.
        #[Self = "ConservativeTransfer"]
        fn prepare(
            src: &UMesh,
            tgt: &UMesh,
            default_value: f64,
        ) -> Result<Box<ConservativeTransfer>>;

        /// Transfers `field_name` from `src` onto `tgt` under `tgt_field_name`.
        ///
        /// Target cells not covered by the source keep the `default_value` passed
        /// to `prepare`.
        fn apply_update(
            self: &ConservativeTransfer,
            src: &UMesh,
            field_name: &str,
            tgt: &mut UMesh,
            tgt_field_name: &str,
            nature: FieldNature,
        ) -> Result<()>;
    }
}
