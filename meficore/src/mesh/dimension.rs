#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
pub enum Dimension {
    D0,
    D1,
    D2,
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
