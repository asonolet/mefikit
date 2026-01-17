use std::sync::Arc;

pub enum FieldSelection {
    Gt(Arc<FieldExpr>, f64),
    Geq(Arc<FieldExpr>, f64),
    Eq(Arc<FieldExpr>, f64),
    Lt(Arc<FieldExpr>, f64),
    Leq(Arc<FieldExpr>, f64),
}

pub enum FieldExpr {
    Scalar(f64),
    Field(String),
    BinarayExpr {
        operator: FieldOp,
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
}

#[derive(Copy, Clone)]
pub enum FieldOp {
    Add,
    Mul,
    Sub,
    Div,
    Pow,
}
