//! Dimension types for mesh elements and fields.
//!
//! Provides the [`Dimension`] enum representing 0D, 1D, 2D, and 3D spaces.

/// Represents the topological dimension of a mesh element or field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Dimension {
    /// Zero-dimensional (points/vertices).
    D0,
    /// One-dimensional (lines/segments).
    D1,
    /// Two-dimensional (surfaces/faces).
    D2,
    /// Three-dimensional (volumes).
    D3,
}

impl From<Dimension> for u8 {
    fn from(dim: Dimension) -> u8 {
        match dim {
            Dimension::D0 => 0,
            Dimension::D1 => 1,
            Dimension::D2 => 2,
            Dimension::D3 => 3,
        }
    }
}

impl TryFrom<u8> for Dimension {
    type Error = String;
    fn try_from(i: u8) -> Result<Dimension, String> {
        match i {
            3 => Ok(Dimension::D3),
            2 => Ok(Dimension::D2),
            1 => Ok(Dimension::D1),
            0 => Ok(Dimension::D0),
            _ => Err("i is higher than 3, could not convert into a dimension".to_owned()),
        }
    }
}

impl TryFrom<usize> for Dimension {
    type Error = String;
    fn try_from(i: usize) -> Result<Dimension, String> {
        match i {
            3 => Ok(Dimension::D3),
            2 => Ok(Dimension::D2),
            1 => Ok(Dimension::D1),
            0 => Ok(Dimension::D0),
            _ => Err("i is higher than 3, could not convert into a dimension".to_owned()),
        }
    }
}

impl std::ops::Add<u8> for Dimension {
    type Output = Dimension;
    fn add(self, i: u8) -> Self {
        let dim: u8 = self.into();
        let sum = i + dim;
        sum.try_into().unwrap()
    }
}

impl std::ops::Sub<u8> for Dimension {
    type Output = Dimension;
    fn sub(self, i: u8) -> Self {
        let dim: u8 = self.into();
        let sub = dim - i;
        sub.try_into().unwrap()
    }
}

impl std::ops::Add for Dimension {
    type Output = Dimension;
    fn add(self, rhs: Self) -> Self {
        let dim: u8 = self.into();
        let rhs: u8 = rhs.into();
        let sum = rhs + dim;
        sum.try_into().unwrap()
    }
}

impl std::ops::Sub for Dimension {
    type Output = Dimension;
    fn sub(self, rhs: Self) -> Self {
        let rhs: u8 = rhs.into();
        let dim: u8 = self.into();
        let sub = dim - rhs;
        sub.try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_from_u8() {
        assert_eq!(Dimension::try_from(0u8), Ok(Dimension::D0));
        assert_eq!(Dimension::try_from(1u8), Ok(Dimension::D1));
        assert_eq!(Dimension::try_from(2u8), Ok(Dimension::D2));
        assert_eq!(Dimension::try_from(3u8), Ok(Dimension::D3));
        assert!(Dimension::try_from(4u8).is_err());
    }

    #[test]
    fn test_dimension_from_usize() {
        assert_eq!(Dimension::try_from(0usize), Ok(Dimension::D0));
        assert_eq!(Dimension::try_from(1usize), Ok(Dimension::D1));
        assert_eq!(Dimension::try_from(2usize), Ok(Dimension::D2));
        assert_eq!(Dimension::try_from(3usize), Ok(Dimension::D3));
        assert!(Dimension::try_from(4usize).is_err());
    }

    #[test]
    fn test_dimension_into_u8() {
        assert_eq!(u8::from(Dimension::D0), 0);
        assert_eq!(u8::from(Dimension::D1), 1);
        assert_eq!(u8::from(Dimension::D2), 2);
        assert_eq!(u8::from(Dimension::D3), 3);
    }

    #[test]
    fn test_dimension_add_u8() {
        assert_eq!(Dimension::D1 + 1, Dimension::D2);
        assert_eq!(Dimension::D2 + 1, Dimension::D3);
        assert_eq!(Dimension::D0 + 2, Dimension::D2);
    }

    #[test]
    fn test_dimension_sub_u8() {
        assert_eq!(Dimension::D2 - 1, Dimension::D1);
        assert_eq!(Dimension::D3 - 1, Dimension::D2);
        assert_eq!(Dimension::D2 - 2, Dimension::D0);
    }

    #[test]
    fn test_dimension_add_dimension() {
        assert_eq!(Dimension::D1 + Dimension::D1, Dimension::D2);
        assert_eq!(Dimension::D0 + Dimension::D3, Dimension::D3);
    }

    #[test]
    fn test_dimension_sub_dimension() {
        assert_eq!(Dimension::D2 - Dimension::D1, Dimension::D1);
        assert_eq!(Dimension::D3 - Dimension::D2, Dimension::D1);
    }

    #[test]
    fn test_dimension_ordering() {
        assert!(Dimension::D0 < Dimension::D1);
        assert!(Dimension::D1 < Dimension::D2);
        assert!(Dimension::D2 < Dimension::D3);
    }
}
