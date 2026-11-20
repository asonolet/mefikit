//! Conservative P0 transfer: prepare once, apply many times.

use mefikit::prelude as mf;
use mefikit::tools::Transfer as _;

use crate::ffi::bridge::FieldNature;
use crate::ffi::{ConservativeTransfer, Error, MeshShape, UMesh};

impl ConservativeTransfer {
    /// Precomputes the conservative P0 transfer from `src` to `tgt`.
    pub fn prepare(src: &UMesh, tgt: &UMesh, default_value: f64) -> Result<Box<Self>, Error> {
        let src_dim = check_preconditions(src, tgt)?;
        let operator = mf::TransferOperator::new(
            &src.mesh.view(),
            &tgt.mesh.view(),
            mf::TransferMethod::ConservativeP0,
        );
        Ok(Box::new(Self {
            operator,
            default_value,
            src_dim,
            n_src: src.mesh.num_elements_of_dim(src_dim),
            src_shape: MeshShape::of(src),
            tgt_shape: MeshShape::of(tgt),
        }))
    }

    /// Applies the prepared operator to one source field.
    pub fn apply_update(
        &self,
        src: &UMesh,
        field_name: &str,
        tgt: &mut UMesh,
        tgt_field_name: &str,
        nature: FieldNature,
    ) -> Result<(), Error> {
        let src_dim = src
            .mesh
            .topological_dimension()
            .ok_or_else(|| Error::InvalidArgument("the source mesh has no elements".to_owned()))?;
        if src_dim != self.src_dim {
            return Err(Error::InvalidArgument(format!(
                "this operator was prepared from a source at dimension {:?}, but the source \
                 mesh given here is {src_dim:?}",
                self.src_dim
            )));
        }
        if src.mesh.num_elements_of_dim(src_dim) != self.n_src {
            return Err(Error::InvalidArgument(format!(
                "this operator was prepared for {} source cells at dimension {:?}, but the \
                 source mesh given here has {}",
                self.n_src,
                self.src_dim,
                src.mesh.num_elements_of_dim(src_dim)
            )));
        }
        if !self.src_shape.matches(src) {
            return Err(Error::InvalidArgument(
                "the source mesh changed since the operator was prepared".to_owned(),
            ));
        }
        if !self.tgt_shape.matches(tgt) {
            return Err(Error::InvalidArgument(
                "the target mesh changed since the operator was prepared".to_owned(),
            ));
        }

        let field = src
            .mesh
            .field(field_name, Some(self.src_dim))
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "source mesh has no field named '{field_name}' at its topological dimension"
                ))
            })?;
        self.operator.apply_update(
            &mut tgt.mesh,
            tgt_field_name,
            &field,
            nature.to_core()?,
            self.default_value,
        );
        Ok(())
    }
}

fn check_preconditions(src: &UMesh, tgt: &UMesh) -> Result<mf::Dimension, Error> {
    let src_space = src.mesh.space_dimension();
    let tgt_space = tgt.mesh.space_dimension();
    if src_space != tgt_space {
        return Err(Error::InvalidArgument(format!(
            "conservative P0 needs both meshes in the same space dimension, got {src_space}D \
             and {tgt_space}D"
        )));
    }
    if !(2..=3).contains(&src_space) {
        return Err(Error::InvalidArgument(format!(
            "conservative P0 is only supported in 2D and 3D space, got {src_space}D"
        )));
    }
    let src_dim = src
        .mesh
        .topological_dimension()
        .ok_or_else(|| Error::InvalidArgument("the source mesh has no elements".to_owned()))?;
    let tgt_dim = tgt
        .mesh
        .topological_dimension()
        .ok_or_else(|| Error::InvalidArgument("the target mesh has no elements".to_owned()))?;
    if src_dim < tgt_dim {
        return Err(Error::InvalidArgument(
            "conservative P0 cannot transfer from lower-dimensional source cells to \
             higher-dimensional target cells"
                .to_owned(),
        ));
    }
    if src_dim == tgt_dim && (u8::from(src_dim) as usize != src_space) {
        return Err(Error::InvalidArgument(format!(
            "same-dimensional conservative P0 needs full-dimensional source and target cells, \
             got topological {src_dim:?} in {src_space}D space"
        )));
    }
    if src.mesh.num_elements_of_dim(src_dim) == 0 {
        return Err(Error::InvalidArgument(
            "the source mesh has no cells at its topological dimension".to_owned(),
        ));
    }
    if tgt.mesh.num_elements_of_dim(tgt_dim) == 0 {
        return Err(Error::InvalidArgument(
            "the target mesh has no cells at its topological dimension".to_owned(),
        ));
    }
    Ok(src_dim)
}
