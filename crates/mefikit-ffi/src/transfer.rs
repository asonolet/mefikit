//! `TransferOperator`: the prepare-once / apply-many half of a transfer.

use mefikit::prelude as mf;
use mefikit::tools::Transfer as _;

use crate::ffi::bridge::{Dimension, FieldNature, TransferMethod};
use crate::ffi::{Error, MeshShape, TransferOperator, UMesh};
use crate::mesh::FieldComponents;

impl TransferOperator {
    /// Precomputes the transfer from `src` to `tgt`.
    ///
    /// This walks the geometry (k-nearest-neighbours, or cell overlap, depending
    /// on `method`) and is by far the expensive part of a transfer. The result
    /// holds no reference to either mesh and stays valid as long as their
    /// geometry does not change, so build it once and call
    /// [`apply_update`](Self::apply_update) for every field and time step.
    ///
    /// # Errors
    ///
    /// `mefikit` states the mesh and method preconditions of a transfer as
    /// `assert!`s, which abort the process when reached from C++. They are
    /// checked here instead, so every one of them comes back as a message naming
    /// what to change.
    pub fn prepare(src: &UMesh, tgt: &UMesh, method: &TransferMethod) -> Result<Box<Self>, Error> {
        let method = method.to_core()?;
        let src_dim = check_transfer_preconditions(src, tgt, &method)?;

        let operator = mf::TransferOperator::new(&src.0.view(), &tgt.0.view(), method);

        // What the core asserts on at apply time, recorded now so that it can be
        // checked as an error later. `src_dim` is the source mesh's topological
        // dimension, which is what both `Transfer::apply` and this crate's
        // `field(name, None)` mean by "the source dimension".
        let n_src = src.core().num_elements_of_dim(src_dim);

        Ok(Box::new(Self {
            operator,
            src_dim,
            n_src,
            src_shape: MeshShape::of(src),
            tgt_shape: MeshShape::of(tgt),
        }))
    }

    /// Transfers field `field_name` of `src` onto `tgt` under `tgt_field_name`.
    ///
    /// Target cells not covered by the source mesh get `default_value`. The
    /// field is replaced if `tgt_field_name` already exists on `tgt`.
    pub fn apply_update(
        &self,
        src: &UMesh,
        field_name: &str,
        tgt: &mut UMesh,
        tgt_field_name: &str,
        default_value: f64,
        nature: FieldNature,
    ) -> Result<(), Error> {
        // The core reads this field with `field(name, None)`, which resolves to
        // the source mesh's topological dimension and unwraps it; on an empty
        // source mesh that is a panic. Everything below mirrors the two `assert!`s
        // in `Transfer::apply`, which would also panic.
        let src_dim = src
            .core_dimension()
            .ok_or_else(|| Error::InvalidArgument("the source mesh has no elements".to_owned()))?;
        if src_dim != self.src_dim {
            return Err(Error::InvalidArgument(format!(
                "this operator was prepared from a source at dimension {:?} but the source mesh \
                 given here is at dimension {src_dim:?}",
                self.src_dim
            )));
        }
        if src.core().num_elements_of_dim(src_dim) != self.n_src {
            return Err(Error::InvalidArgument(format!(
                "this operator was prepared for a source with {} cells at dimension {} but the \
                 source mesh given here has {}; the geometry must not change between prepare and \
                 apply",
                self.n_src,
                u8::from(self.src_dim),
                src.core().num_elements_of_dim(src_dim)
            )));
        }
        self.check_shape(src, &self.src_shape, "source")?;
        self.check_shape(tgt, &self.tgt_shape, "target")?;

        let field = src.0.field(field_name, None).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "source mesh has no field named '{field_name}' at its topological dimension"
            ))
        })?;
        // The core concatenates the field's per-element-type parts, which only
        // lines up if they share a shape. The view just resolved is walked
        // directly, rather than looking every block up again.
        check_field_components(&field, field_name, src_dim)?;
        self.operator.apply_update(
            &mut tgt.0,
            tgt_field_name,
            &field,
            nature.to_core()?,
            default_value,
        );
        Ok(())
    }

    /// Topological dimension of the cells this operator writes to.
    #[must_use]
    pub fn target_dimension(&self) -> Dimension {
        Dimension::from(self.operator.tgt_dim())
    }

    /// Rejects a mesh whose shape changed since the operator was prepared.
    fn check_shape(&self, mesh: &UMesh, expected: &MeshShape, role: &str) -> Result<(), Error> {
        if expected.matches(mesh) {
            return Ok(());
        }
        let now = MeshShape::of(mesh);
        Err(Error::InvalidArgument(format!(
            "this operator was prepared for a {role} mesh with {expected}, but the {role} mesh \
             given here has {now}; prepare a new operator for it"
        )))
    }
}

/// Rejects a field whose per-element-type parts disagree on the component
/// count.
///
/// The core reads a field as a single array, so ragged parts have to be caught
/// before it panics. `set_field` keeps that from happening for fields the
/// bindings build, but not for one read from a file.
fn check_field_components(
    field: &mf::FieldView<ndarray::IxDyn>,
    name: &str,
    dimension: mf::Dimension,
) -> Result<(), Error> {
    let mut components: Option<(usize, mf::ElementType)> = None;
    for (element_type, array) in &field.0 {
        if element_type.dimension() != dimension {
            continue;
        }
        let (_, n_components) = FieldComponents::of(array);
        match components {
            Some((expected, first)) if expected != n_components => {
                return Err(Error::InvalidArgument(format!(
                    "source mesh field '{name}' has {expected} components on {first:?} and \
                     {n_components} on {element_type:?}; it needs one shape across every \
                     element type at dimension {} to be transferred",
                    u8::from(dimension)
                )));
            }
            Some(_) => {}
            None => components = Some((n_components, *element_type)),
        }
    }
    Ok(())
}

/// Prepares a transfer and applies it in one call.
///
/// Convenient when a single field is transferred once; when you transfer several
/// fields (or several time steps) onto the same meshes, use
/// [`TransferOperator::prepare`] instead so the precompute is paid for once.
pub fn transfer_field(
    src: &UMesh,
    field_name: &str,
    tgt: &mut UMesh,
    tgt_field_name: &str,
    method: &TransferMethod,
    default_value: f64,
    nature: FieldNature,
) -> Result<(), Error> {
    let operator = TransferOperator::prepare(src, tgt, method)?;
    operator.apply_update(src, field_name, tgt, tgt_field_name, default_value, nature)
}

/// Checks everything `mefikit` asserts about the two meshes and the method when
/// building an operator.
///
/// The core's `validated_dims` plus the per-method asserts. `mefikit` documents
/// these as panics; the bindings turn each one into an error, so the checks
/// mirror the core's rather than being an independent opinion about what a
/// transfer needs. They have to be kept in step with it deliberately.
/// Returns the source mesh's topological dimension, which the caller needs
/// afterwards and which is only known once the mesh has been found to be
/// non-empty.
fn check_transfer_preconditions(
    src: &UMesh,
    tgt: &UMesh,
    method: &mf::TransferMethod,
) -> Result<mf::Dimension, Error> {
    let name = method_name(method);

    let src_space = src.space_dimension();
    let tgt_space = tgt.space_dimension();
    if src_space != tgt_space {
        return Err(Error::InvalidArgument(format!(
            "{name} needs both meshes in the same space dimension, got {src_space}D and \
             {tgt_space}D"
        )));
    }
    if !(2..=3).contains(&src_space) {
        return Err(Error::InvalidArgument(format!(
            "{name} is only supported in 2D and 3D space, got {src_space}D"
        )));
    }
    let src_dim = src
        .core_dimension()
        .ok_or_else(|| Error::InvalidArgument("the source mesh has no elements".to_owned()))?;
    let tgt_dim = tgt
        .core_dimension()
        .ok_or_else(|| Error::InvalidArgument("the target mesh has no elements".to_owned()))?;

    // ConservativeP0 and ConstantPiecewise work off cell measures, cell centroids
    // and a bounding volume hierarchy, all of which need full-dimensional cells.
    // InverseDistance and MovingLeastSquares interpolate at target sample points
    // and do support a lower-dimensional target, so this is per method.
    let needs_full_dimensional = matches!(
        method,
        mf::TransferMethod::ConservativeP0 | mf::TransferMethod::ConstantPiecewise { .. }
    );
    if needs_full_dimensional {
        for (role, dim) in [("source", src_dim), ("target", tgt_dim)] {
            if u8::from(dim) as usize != src_space {
                return Err(Error::InvalidArgument(format!(
                    "{name} needs full-dimensional {role} cells, but the {role} mesh has \
                     topological dimension {dim:?} in {src_space}D space"
                )));
            }
        }
    }

    let (k, positive_exponent) = match *method {
        mf::TransferMethod::InverseDistance { k, exponent } => (Some(k), Some(exponent)),
        mf::TransferMethod::MovingLeastSquares { k, .. } => (Some(k), None),
        _ => (None, None),
    };
    if k.is_some_and(|k| k == 0) {
        return Err(Error::InvalidArgument(format!(
            "{name} needs k of at least 1, got 0"
        )));
    }
    if positive_exponent.is_some_and(|e| !e.is_finite() || e <= 0.0) {
        return Err(Error::InvalidArgument(format!(
            "{name} needs a positive exponent, got {positive_exponent:?}"
        )));
    }

    // A source whose cells are all empty would make the core divide the field
    // length by the cell count when applying, which is a division by zero.
    if src.core().num_elements_of_dim(src_dim) == 0 {
        return Err(Error::InvalidArgument(format!(
            "the source mesh has no cells at dimension {} to interpolate from",
            u8::from(src_dim)
        )));
    }

    Ok(src_dim)
}

/// The name of a method, for error messages.
fn method_name(method: &mf::TransferMethod) -> &'static str {
    match method {
        mf::TransferMethod::ConstantPiecewise { .. } => "Constant piecewise",
        mf::TransferMethod::ConservativeP0 => "Conservative P0",
        mf::TransferMethod::InverseDistance { .. } => "Inverse-distance",
        mf::TransferMethod::MovingLeastSquares { .. } => "Moving least-squares",
    }
}
