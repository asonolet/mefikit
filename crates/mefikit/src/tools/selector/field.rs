use crate::tools::fieldexpr::FieldExpr;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub enum FieldSelection {
    Gt(Arc<FieldExpr>, Arc<FieldExpr>),
    Geq(Arc<FieldExpr>, Arc<FieldExpr>),
    Lt(Arc<FieldExpr>, Arc<FieldExpr>),
    Leq(Arc<FieldExpr>, Arc<FieldExpr>),
    Eq(Arc<FieldExpr>, Arc<FieldExpr>),
    Neq(Arc<FieldExpr>, Arc<FieldExpr>),
}

pub trait Comparable {
    fn gt(self, other: Self) -> FieldSelection;
    fn geq(self, other: Self) -> FieldSelection;
    fn lt(self, other: Self) -> FieldSelection;
    fn leq(self, other: Self) -> FieldSelection;
    fn eq(self, other: Self) -> FieldSelection;
    fn neq(self, other: Self) -> FieldSelection;
}

impl Comparable for FieldExpr {
    fn gt(self, other: Self) -> FieldSelection {
        FieldSelection::Gt(Arc::new(self), Arc::new(other))
    }
    fn geq(self, other: Self) -> FieldSelection {
        FieldSelection::Geq(Arc::new(self), Arc::new(other))
    }
    fn lt(self, other: Self) -> FieldSelection {
        FieldSelection::Lt(Arc::new(self), Arc::new(other))
    }
    fn leq(self, other: Self) -> FieldSelection {
        FieldSelection::Leq(Arc::new(self), Arc::new(other))
    }
    fn eq(self, other: Self) -> FieldSelection {
        FieldSelection::Eq(Arc::new(self), Arc::new(other))
    }
    fn neq(self, other: Self) -> FieldSelection {
        FieldSelection::Neq(Arc::new(self), Arc::new(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::fieldexpr::{arr, field};
    use ndarray;

    #[test]
    fn test_field_gt() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.gt(b);
        match selection {
            FieldSelection::Gt(..) => (),
            _ => panic!("Expected Gt"),
        }
    }

    #[test]
    fn test_field_geq() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.geq(b);
        match selection {
            FieldSelection::Geq(..) => (),
            _ => panic!("Expected Geq"),
        }
    }

    #[test]
    fn test_field_lt() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.lt(b);
        match selection {
            FieldSelection::Lt(..) => (),
            _ => panic!("Expected Lt"),
        }
    }

    #[test]
    fn test_field_leq() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.leq(b);
        match selection {
            FieldSelection::Leq(..) => (),
            _ => panic!("Expected Leq"),
        }
    }

    #[test]
    fn test_field_eq() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.eq(b);
        match selection {
            FieldSelection::Eq(..) => (),
            _ => panic!("Expected Eq"),
        }
    }

    #[test]
    fn test_field_neq() {
        let a = field("A");
        let b = arr(ndarray::arr0(1.0));
        let selection = a.neq(b);
        match selection {
            FieldSelection::Neq(..) => (),
            _ => panic!("Expected Neq"),
        }
    }
}
