//! Distance kernels shared by every meshless reconstruction operator.
//!
//! A kernel turns the distance from an evaluation point to one of its `k` nearest source points
//! into a weight of the local weighted least-squares fit. The same kernels are used by the
//! moving least-squares transfer and by the gradient tool (and can be reused by any future
//! meshless operator, e.g. divergence or laplacian).

/// How the distance from a target interpolation point to its `k` nearest source points is turned
/// into a weight of the local (moving) least-squares fit.
///
/// All kernels are written as a function of `s^2 = (r / h)^2`, where `r` is the distance from the
/// target point to a source point and `h` is a characteristic length chosen per target point (the
/// distance to its farthest selected neighbour).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DistanceWeighting {
    /// All selected neighbours get the same weight: a plain linear least-squares fit.
    Constant,
    /// Inverse-distance weight `w = (h / r)^exponent` (`2.0` gives the usual inverse squared
    /// distance).
    InverseDistance { exponent: f64 },
    /// Compact-support moving least-squares kernel `w = (1 - s^2)^exponent` for `s < 1`, zero
    /// otherwise (`exponent = 2` or `3` are the usual choices).
    CompactSupport { exponent: f64 },
    /// Gaussian kernel `w = exp(-s^2)`.
    Gaussian,
}

impl DistanceWeighting {
    /// Evaluates the kernel on the squared scaled distance `s2 = (r / h)^2`.
    pub(crate) fn kernel(self, s2: f64) -> f64 {
        match self {
            Self::Constant => 1.0,
            Self::InverseDistance { exponent } => s2.powf(-0.5 * exponent),
            Self::CompactSupport { exponent } => {
                if s2 >= 1.0 {
                    0.0
                } else {
                    (1.0 - s2).powf(exponent)
                }
            }
            Self::Gaussian => (-s2).exp(),
        }
    }
}
