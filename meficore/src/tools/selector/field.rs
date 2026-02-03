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
