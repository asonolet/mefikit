//! Transfer preparation and application for C++.

use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};

use mefikit::prelude as mf;
use mefikit::tools::Transfer as _;

use crate::ffi::bridge::{DistanceWeighting, FieldNature};
use crate::ffi::{Error, MeshShape, Transfer, UMesh};

impl Transfer {
    /// Precomputes the conservative P0 transfer from `src` to `tgt`.
    pub fn prepare_conservative_p0(
        src: &UMesh,
        tgt: &UMesh,
        default_value: f64,
    ) -> Result<Box<Self>, Error> {
        check_conservative_preconditions(src, tgt)?;
        Self::build(src, tgt, default_value, mf::TransferMethod::ConservativeP0)
    }

    /// Precomputes the piecewise-constant transfer from `src` to `tgt`.
    pub fn prepare_constant_piecewise(
        src: &UMesh,
        tgt: &UMesh,
        default_value: f64,
    ) -> Result<Box<Self>, Error> {
        let (src_dim, _tgt_dim, space) = check_common(src, tgt)?;
        if u8::from(src_dim) as usize != space {
            return Err(Error::InvalidArgument(format!(
                "constant-piecewise needs a full-dimensional source mesh, got topological \
                 {src_dim:?} in {space}D space"
            )));
        }
        Self::build(
            src,
            tgt,
            default_value,
            mf::TransferMethod::ConstantPiecewise {
                point_location: mf::PointLocation::Centroid,
            },
        )
    }

    /// Precomputes the inverse-distance (`k`-nearest-neighbours Shepard) transfer.
    pub fn prepare_inverse_distance(
        src: &UMesh,
        tgt: &UMesh,
        k: usize,
        exponent: f64,
        default_value: f64,
    ) -> Result<Box<Self>, Error> {
        check_point_method(src, tgt, k, "inverse-distance")?;
        if !exponent.is_finite() || exponent <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "inverse-distance needs a finite, positive exponent, got {exponent}"
            )));
        }
        Self::build(
            src,
            tgt,
            default_value,
            mf::TransferMethod::InverseDistance { k, exponent },
        )
    }

    /// Precomputes the moving least-squares transfer.
    pub fn prepare_moving_least_squares(
        src: &UMesh,
        tgt: &UMesh,
        k: usize,
        weighting: DistanceWeighting,
        exponent: f64,
        default_value: f64,
    ) -> Result<Box<Self>, Error> {
        check_point_method(src, tgt, k, "moving least-squares")?;
        let weighting = weighting.to_core(exponent)?;
        Self::build(
            src,
            tgt,
            default_value,
            mf::TransferMethod::MovingLeastSquares { k, weighting },
        )
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

    fn build(
        src: &UMesh,
        tgt: &UMesh,
        default_value: f64,
        method: mf::TransferMethod,
    ) -> Result<Box<Self>, Error> {
        let operator = catch_unwind(AssertUnwindSafe(|| {
            mf::TransferOperator::new(&src.mesh.view(), &tgt.mesh.view(), method)
        }))
        .map_err(|payload| Error::InvalidArgument(panic_message(payload)))?;

        let src_dim = src
            .mesh
            .topological_dimension()
            .ok_or_else(|| Error::InvalidArgument("the source mesh has no elements".to_owned()))?;
        Ok(Box::new(Self {
            operator,
            default_value,
            src_dim,
            n_src: src.mesh.num_elements_of_dim(src_dim),
            src_shape: MeshShape::of(src),
            tgt_shape: MeshShape::of(tgt),
        }))
    }
}

fn check_common(src: &UMesh, tgt: &UMesh) -> Result<(mf::Dimension, mf::Dimension, usize), Error> {
    let src_space = src.mesh.space_dimension();
    let tgt_space = tgt.mesh.space_dimension();
    if src_space != tgt_space {
        return Err(Error::InvalidArgument(format!(
            "transfers need both meshes in the same space dimension, got {src_space}D and \
             {tgt_space}D"
        )));
    }
    if !(2..=3).contains(&src_space) {
        return Err(Error::InvalidArgument(format!(
            "transfers are only supported in 2D and 3D space, got {src_space}D"
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
    Ok((src_dim, tgt_dim, src_space))
}

fn check_conservative_preconditions(src: &UMesh, tgt: &UMesh) -> Result<(), Error> {
    let (src_dim, tgt_dim, src_space) = check_common(src, tgt)?;
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
    Ok(())
}

fn check_point_method(src: &UMesh, tgt: &UMesh, k: usize, method: &str) -> Result<(), Error> {
    let (src_dim, _tgt_dim, _space) = check_common(src, tgt)?;
    if src_dim != mf::Dimension::D2 && src_dim != mf::Dimension::D3 {
        return Err(Error::InvalidArgument(format!(
            "{method} needs source cells in 2D or 3D, got topological {src_dim:?}"
        )));
    }
    if k == 0 {
        return Err(Error::InvalidArgument(format!(
            "{method} needs at least one neighbour, got k = 0"
        )));
    }
    Ok(())
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "the transfer could not be prepared".to_owned()
    }
}
