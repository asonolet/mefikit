//! Conversions between the enums generated in the bridge and their `mefikit`
//! counterparts.
//!
//! The bridge declares its own `ElementType` / `Dimension` / ... because a cxx
//! shared enum is a *new* type, not a re-export of the Rust one.
//!
//! Note the asymmetry in direction. cxx generates a shared enum as an open
//! `#[repr(transparent)] struct` around its integer, so a C++ caller can hand us
//! any discriminant — every bridge -> `mefikit` conversion is therefore fallible
//! and reports an unknown discriminant as an [`Error::InvalidArgument`] rather
//! than panicking. `mefikit` -> bridge is the other way round: the source is a
//! closed enum, so those `match`es are exhaustive and adding a variant to a
//! `mefikit` enum does break the build here instead of silently mistranslating.

use std::fmt::{self, Debug, Formatter};

use mefikit::mesh::{Dimension as CoreDimension, ElementType as CoreElementType};
use mefikit::tools::{
    DistanceWeighting as CoreDistanceWeighting, FieldNature as CoreFieldNature,
    PointLocation as CorePointLocation, TransferMethod as CoreTransferMethod,
};

use crate::ffi::Error;
use crate::ffi::bridge::{
    Dimension, DistanceWeighting, ElementType, FieldNature, PointLocation, TransferMethod,
    TransferMethodKind,
};

/// Declares a fallible bridge -> `mefikit` conversion plus the `Debug` impl cxx
/// does not generate, which delegates to the `mefikit` enum so that messages name
/// elements the way the rest of mefikit does.
macro_rules! mirrored_enum {
    (
        $bridge:ty => $core:ty,
        variants { $($bridge_var:ident => $core_var:ident),* $(,)? }
    ) => {
        impl $bridge {
            /// Interprets a value that arrived from C++ as a `mefikit` value.
            ///
            /// # Errors
            ///
            /// Returns [`Error::InvalidArgument`] if the discriminant is not one
            /// this build of `mefikit` knows about.
            pub fn to_core(self) -> Result<$core, Error> {
                match self.repr {
                    $(repr if repr == Self::$bridge_var.repr => Ok(<$core>::$core_var),)*
                    repr => Err(Error::InvalidArgument(format!(
                        concat!(stringify!($bridge), " discriminant {} is not known to this \
                                mefikit build"),
                        repr,
                    ))),
                }
            }
        }

        impl From<$core> for $bridge {
            fn from(value: $core) -> Self {
                match value {
                    $(<$core>::$core_var => Self::$bridge_var,)*
                }
            }
        }

        impl Debug for $bridge {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self.to_core() {
                    Ok(core) => Debug::fmt(&core, f),
                    Err(_) => write!(f, concat!(stringify!($bridge), "({})"), self.repr),
                }
            }
        }
    };
}

mirrored_enum! {
    ElementType => CoreElementType,
    variants {
        VERTEX => VERTEX, SEG2 => SEG2, SEG3 => SEG3, SEG4 => SEG4, SPLINE => SPLINE,
        TRI3 => TRI3, TRI6 => TRI6, TRI7 => TRI7, QUAD4 => QUAD4, QUAD8 => QUAD8,
        QUAD9 => QUAD9, PGON => PGON, TET4 => TET4, TET10 => TET10, HEX8 => HEX8,
        HEX21 => HEX21, PHED => PHED,
    }
}

mirrored_enum! {
    Dimension => CoreDimension,
    variants { D0 => D0, D1 => D1, D2 => D2, D3 => D3 }
}

mirrored_enum! {
    FieldNature => CoreFieldNature,
    variants { Intensive => Intensive, Extensive => Extensive }
}

mirrored_enum! {
    PointLocation => CorePointLocation,
    variants { Centroid => Centroid, Barycenter => Barycenter, StrictInterior => StrictInterior }
}

impl Debug for TransferMethodKind {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_str(match self.repr {
            x if x == Self::ConstantPiecewise.repr => "ConstantPiecewise",
            x if x == Self::ConservativeP0.repr => "ConservativeP0",
            x if x == Self::InverseDistance.repr => "InverseDistance",
            x if x == Self::MovingLeastSquares.repr => "MovingLeastSquares",
            repr => return write!(f, "TransferMethodKind({repr})"),
        })
    }
}

impl DistanceWeighting {
    /// Builds a weighting from its bridge discriminant plus the exponent that
    /// `InverseDistance` and `CompactSupport` need.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if the discriminant is not one this
    /// build of `mefikit` knows about.
    pub fn with_exponent(self, exponent: f64) -> Result<CoreDistanceWeighting, Error> {
        match self.repr {
            x if x == Self::Constant.repr => Ok(CoreDistanceWeighting::Constant),
            x if x == Self::InverseDistance.repr => {
                Ok(CoreDistanceWeighting::InverseDistance { exponent })
            }
            x if x == Self::CompactSupport.repr => {
                Ok(CoreDistanceWeighting::CompactSupport { exponent })
            }
            x if x == Self::Gaussian.repr => Ok(CoreDistanceWeighting::Gaussian),
            repr => Err(Error::InvalidArgument(format!(
                "DistanceWeighting discriminant {repr} is not known to this mefikit build"
            ))),
        }
    }
}

impl TransferMethod {
    /// Copy the value of the source cell containing the sample point.
    #[must_use]
    pub fn constant_piecewise(point_location: PointLocation) -> Self {
        Self {
            kind: TransferMethodKind::ConstantPiecewise,
            point_location,
            k: 0,
            exponent: 0.0,
            weighting: DistanceWeighting::Constant,
        }
    }

    /// Overlap-measure weighted average, conservative for intensive fields.
    #[must_use]
    pub fn conservative_p0() -> Self {
        Self {
            kind: TransferMethodKind::ConservativeP0,
            point_location: PointLocation::Centroid,
            k: 0,
            exponent: 0.0,
            weighting: DistanceWeighting::Constant,
        }
    }

    /// Shepard interpolation over the `k` nearest source cells.
    #[must_use]
    pub fn inverse_distance(k: usize, exponent: f64) -> Self {
        Self {
            kind: TransferMethodKind::InverseDistance,
            point_location: PointLocation::Centroid,
            k,
            exponent,
            weighting: DistanceWeighting::Constant,
        }
    }

    /// Local least-squares fit over the `k` nearest source cells.
    #[must_use]
    pub fn moving_least_squares(k: usize, weighting: DistanceWeighting) -> Self {
        Self {
            kind: TransferMethodKind::MovingLeastSquares,
            point_location: PointLocation::Centroid,
            k,
            exponent: 2.0,
            weighting,
        }
    }

    /// Flattens the struct into the `mefikit` transfer method it describes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidArgument`] if a discriminant is not one this build
    /// of `mefikit` knows about, or if the combination is not supported.
    pub fn to_core(self) -> Result<CoreTransferMethod, Error> {
        let Self {
            kind,
            point_location,
            k,
            exponent,
            weighting,
        } = self;

        let kind = kind.repr;
        if kind == TransferMethodKind::ConstantPiecewise.repr {
            // mefikit still has todo!()s for these two, and a panic across the bridge
            // aborts the process instead of unwinding into a catchable C++ exception.
            // Reject them here so the caller gets a message they can act on.
            let point_location = match point_location.repr {
                x if x == PointLocation::Centroid.repr => CorePointLocation::Centroid,
                x if x == PointLocation::Barycenter.repr => {
                    return Err(unsupported_point_location("Barycenter"));
                }
                x if x == PointLocation::StrictInterior.repr => {
                    return Err(unsupported_point_location("StrictInterior"));
                }
                repr => {
                    return Err(Error::InvalidArgument(format!(
                        "PointLocation discriminant {repr} is not known to this mefikit build"
                    )));
                }
            };
            return Ok(CoreTransferMethod::ConstantPiecewise { point_location });
        }
        if kind == TransferMethodKind::ConservativeP0.repr {
            return Ok(CoreTransferMethod::ConservativeP0);
        }
        if kind == TransferMethodKind::InverseDistance.repr {
            return Ok(CoreTransferMethod::InverseDistance { k, exponent });
        }
        if kind == TransferMethodKind::MovingLeastSquares.repr {
            return Ok(CoreTransferMethod::MovingLeastSquares {
                k,
                weighting: weighting.with_exponent(exponent)?,
            });
        }
        Err(Error::InvalidArgument(format!(
            "TransferMethodKind discriminant {kind} is not known to this mefikit build"
        )))
    }
}

fn unsupported_point_location(name: &str) -> Error {
    Error::InvalidArgument(format!(
        "PointLocation::{name} is not implemented by mefikit yet; \
         use PointLocation::Centroid"
    ))
}
