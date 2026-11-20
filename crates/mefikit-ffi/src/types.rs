//! Fallible conversions from cxx's open enums to mefikit's closed enums.

use std::fmt::{self, Debug, Formatter};

use mefikit::mesh::ElementType as CoreElementType;
use mefikit::tools::FieldNature as CoreFieldNature;

use crate::ffi::Error;
use crate::ffi::bridge::{ElementType, FieldNature};

macro_rules! mirrored_enum {
    (
        $bridge:ty => $core:ty,
        variants { $($bridge_var:ident => $core_var:ident),* $(,)? }
    ) => {
        impl $bridge {
            /// Converts a value received from C++.
            pub fn to_core(self) -> Result<$core, Error> {
                match self.repr {
                    $(repr if repr == Self::$bridge_var.repr => Ok(<$core>::$core_var),)*
                    repr => Err(Error::InvalidArgument(format!(
                        concat!(stringify!($bridge), " discriminant {} is unknown"),
                        repr,
                    ))),
                }
            }
        }

        impl Debug for $bridge {
            fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
                match self.to_core() {
                    Ok(value) => Debug::fmt(&value, f),
                    Err(_) => write!(f, concat!(stringify!($bridge), "({})"), self.repr),
                }
            }
        }
    };
}

mirrored_enum! {
    ElementType => CoreElementType,
    variants {
        VERTEX => VERTEX,
        SEG2 => SEG2,
        SEG3 => SEG3,
        SEG4 => SEG4,
        PLINE => PLINE,
        TRI3 => TRI3,
        TRI6 => TRI6,
        TRI7 => TRI7,
        QUAD4 => QUAD4,
        QUAD8 => QUAD8,
        QUAD9 => QUAD9,
        PGON => PGON,
        TET4 => TET4,
        TET10 => TET10,
        HEX8 => HEX8,
        HEX21 => HEX21,
        WEDGE6 => WEDGE6,
        WEDGE15 => WEDGE15,
        PHED => PHED,
    }
}

mirrored_enum! {
    FieldNature => CoreFieldNature,
    variants {
        Intensive => Intensive,
        Extensive => Extensive,
    }
}
