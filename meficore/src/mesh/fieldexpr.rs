use ndarray::{self as nd};
use smallvec::SmallVec;
use std::{
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::mesh::fields::*;
use crate::mesh::{Dimension, UMeshView};

#[derive(Clone, Debug)]
pub enum FieldExpr {
    Array(nd::ArrayD<f64>), // This array is meant to be broadcasted
    Field(String),
    BinarayExpr {
        operator: BinaryOp,
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
    UnaryExpr {
        operator: UnaryOp,
        expr: Arc<FieldExpr>,
    },
    Centroids,
    X,
    Y,
    Z,
    Index(Arc<FieldExpr>, SmallVec<[usize; 2]>),
}

#[derive(Copy, Clone, Debug)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
    Pow,
}

#[derive(Copy, Clone, Debug)]
pub enum UnaryOp {
    Sin,
    Sqrt,
    Square,
    Cos,
    Exp,
    Ln,
    Log10,
    Abs,
    Tan,
}

impl FieldExpr {
    pub fn sin(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sin,
            expr: Arc::new(self),
        }
    }
    pub fn cos(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Cos,
            expr: Arc::new(self),
        }
    }
    pub fn sqrt(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sqrt,
            expr: Arc::new(self),
        }
    }
    pub fn square(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Square,
            expr: Arc::new(self),
        }
    }
    pub fn exp(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Exp,
            expr: Arc::new(self),
        }
    }
    pub fn ln(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Ln,
            expr: Arc::new(self),
        }
    }
    pub fn log10(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Log10,
            expr: Arc::new(self),
        }
    }
    pub fn tan(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Tan,
            expr: Arc::new(self),
        }
    }
    pub fn abs(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Abs,
            expr: Arc::new(self),
        }
    }
    pub fn pow(self, other: Self) -> Self {
        Self::BinarayExpr {
            operator: BinaryOp::Pow,
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }
}

pub fn field(name: &str) -> FieldExpr {
    FieldExpr::Field(name.to_owned())
}

pub fn arr(arr: nd::ArrayD<f64>) -> FieldExpr {
    FieldExpr::Array(arr)
}

impl Add for FieldExpr {
    type Output = FieldExpr;

    fn add(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinarayExpr {
            operator: BinaryOp::Add,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Sub for FieldExpr {
    type Output = FieldExpr;

    fn sub(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinarayExpr {
            operator: BinaryOp::Sub,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Mul for FieldExpr {
    type Output = FieldExpr;

    fn mul(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinarayExpr {
            operator: BinaryOp::Mul,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl Div for FieldExpr {
    type Output = FieldExpr;

    fn div(self, rhs: FieldExpr) -> FieldExpr {
        FieldExpr::BinarayExpr {
            operator: BinaryOp::Div,
            left: Arc::new(self),
            right: Arc::new(rhs),
        }
    }
}

impl FieldExpr {
    pub fn index(self, index: &[usize]) -> Self {
        Self::Index(Arc::new(self), index.into())
    }
}

pub trait Evaluable {
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCow<'a>;
}

impl Evaluable for FieldExpr {
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCow<'a> {
        let dim = match dim {
            Some(d) => d,
            None => mesh.topological_dimension().unwrap(),
        };
        let elems: Vec<_> = mesh
            .element_types()
            .filter(|et| et.dimension() == dim)
            .cloned()
            .collect();
        match self {
            FieldExpr::Array(arr) => FieldCow::from_array(arr.view().into(), elems.as_slice()),
            FieldExpr::Field(name) => mesh.field(name, None).unwrap().into(),
            FieldExpr::BinarayExpr {
                operator,
                left,
                right,
            } => {
                let left_eval = left.evaluate(mesh, Some(dim));
                let right_eval = right.evaluate(mesh, Some(dim));
                match operator {
                    BinaryOp::Add => (&left_eval + &right_eval).into(),
                    BinaryOp::Sub => (&left_eval - &right_eval).into(),
                    BinaryOp::Mul => (&left_eval * &right_eval).into(),
                    BinaryOp::Div => (&left_eval / &right_eval).into(),
                    BinaryOp::Pow => left_eval.map_zip(&right_eval, |a, b| a.powf(b)).into(),
                }
            }
            FieldExpr::UnaryExpr { operator, expr } => {
                let expr_eval = expr.evaluate(mesh, Some(dim));
                match operator {
                    UnaryOp::Sin => expr_eval.mapv(|x| x.sin()).into(),
                    UnaryOp::Cos => expr_eval.mapv(|x| x.cos()).into(),
                    UnaryOp::Tan => expr_eval.mapv(|x| x.tan()).into(),
                    UnaryOp::Sqrt => expr_eval.mapv(|x| x.sqrt()).into(),
                    UnaryOp::Square => expr_eval.mapv(|x| x.powi(2)).into(),
                    UnaryOp::Exp => expr_eval.mapv(|x| x.exp()).into(),
                    UnaryOp::Ln => expr_eval.mapv(|x| x.ln()).into(),
                    UnaryOp::Log10 => expr_eval.mapv(|x| x.log10()).into(),
                    UnaryOp::Abs => expr_eval.mapv(|x| x.abs()).into(),
                }
            }
            // FieldExpr::Centroids => mesh.centroids().to_owned(),
            // FieldExpr::X => mesh.coords().slice(nd::s![.., 0]).to_owned(),
            // FieldExpr::Y => mesh.coords().slice(nd::s![.., 1]).to_owned(),
            // FieldExpr::Z => mesh.coords().slice(nd::s![.., 2]).to_owned(),
            // FieldExpr::Rcyl => mesh.coords().slice(nd::s![.., 0]).to_owned(),
            // FieldExpr::Rsph => mesh.coords().slice(nd::s![.., 0]).to_owned(),
            // FieldExpr::Theta => mesh.coords().slice(nd::s![.., 1]).to_owned(),
            // FieldExpr::Phi => mesh.coords().slice(nd::s![.., 2]).to_owned(),
            // FieldExpr::Index(expr, index) => {
            //     let eval = expr.evaluate(mesh);
            //     eval[.., [index.try_into().unwrap()]].to_owned()
            // }
            _ => todo!(),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn compose_expr() {
        let a = field("toto");
        let b = field("exponent");
        let c = arr(nd::array![1.0].into_dyn());
        let _res = a.pow(b) + c;
    }
}
