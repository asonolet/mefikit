//! Field expression system for computing derived fields.
//!
//! Provides a domain-specific language for building and evaluating
//! field expressions using mathematical operations.

use ndarray::{self as nd};
use smallvec::SmallVec;
use std::{
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::mesh::{Dimension, FieldArcD, FieldCowD, FieldOwnedD, UMesh, UMeshBase, UMeshView};

/// An expression tree for field computations.
#[derive(Clone, Debug)]
pub enum FieldExpr {
    /// A broadcastable constant array.
    Array(nd::Array<f64, nd::IxDyn>),
    /// A reference to a named field in the mesh.
    Field(String),
    /// A binary operation between two expressions.
    BinarayExpr {
        operator: BinaryOp,
        left: Arc<FieldExpr>,
        right: Arc<FieldExpr>,
    },
    /// A unary operation on an expression.
    UnaryExpr {
        operator: UnaryOp,
        expr: Arc<FieldExpr>,
    },
    /// Element centroids (not yet implemented).
    Centroids,
    /// X coordinate (not yet implemented).
    X,
    /// Y coordinate (not yet implemented).
    Y,
    /// Z coordinate (not yet implemented).
    Z,
    /// Index into a multi-component field.
    Index(Arc<FieldExpr>, SmallVec<[usize; 2]>),
}

/// Binary operations available in field expressions.
#[derive(Copy, Clone, Debug)]
pub enum BinaryOp {
    /// Addition.
    Add,
    /// Multiplication.
    Mul,
    /// Subtraction.
    Sub,
    /// Division.
    Div,
    /// Power (a^b).
    Pow,
}

/// Unary operations available in field expressions.
#[derive(Copy, Clone, Debug)]
pub enum UnaryOp {
    /// Sine function.
    Sin,
    /// Square root.
    Sqrt,
    /// Squaring (x^2).
    Square,
    /// Cosine function.
    Cos,
    /// Exponential function.
    Exp,
    /// Natural logarithm.
    Ln,
    /// Base-10 logarithm.
    Log10,
    /// Absolute value.
    Abs,
    /// Tangent function.
    Tan,
}

impl FieldExpr {
    /// Applies the sine function to this expression.
    pub fn sin(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sin,
            expr: Arc::new(self),
        }
    }

    /// Applies the cosine function to this expression.
    pub fn cos(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Cos,
            expr: Arc::new(self),
        }
    }

    /// Applies the square root to this expression.
    pub fn sqrt(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Sqrt,
            expr: Arc::new(self),
        }
    }

    /// Squares this expression.
    pub fn square(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Square,
            expr: Arc::new(self),
        }
    }

    /// Applies the exponential function to this expression.
    pub fn exp(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Exp,
            expr: Arc::new(self),
        }
    }

    /// Applies the natural logarithm to this expression.
    pub fn ln(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Ln,
            expr: Arc::new(self),
        }
    }

    /// Applies the base-10 logarithm to this expression.
    pub fn log10(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Log10,
            expr: Arc::new(self),
        }
    }

    /// Applies the tangent function to this expression.
    pub fn tan(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Tan,
            expr: Arc::new(self),
        }
    }

    /// Applies the absolute value to this expression.
    pub fn abs(self) -> Self {
        Self::UnaryExpr {
            operator: UnaryOp::Abs,
            expr: Arc::new(self),
        }
    }

    /// Raises this expression to the power of `other`.
    pub fn pow(self, other: Self) -> Self {
        Self::BinarayExpr {
            operator: BinaryOp::Pow,
            left: Arc::new(self),
            right: Arc::new(other),
        }
    }
}

/// Creates a field expression referencing a named field.
pub fn field(name: &str) -> FieldExpr {
    FieldExpr::Field(name.to_owned())
}

/// Creates a field expression from a constant array.
pub fn arr<D: nd::Dimension>(arr: nd::Array<f64, D>) -> FieldExpr {
    FieldExpr::Array(arr.into_dyn())
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
    /// Selects a component from a multi-component field.
    pub fn index(self, index: &[usize]) -> Self {
        Self::Index(Arc::new(self), index.into())
    }
}

/// Trait for evaluating field expressions on a mesh.
pub trait Evaluable {
    /// Evaluates the expression on the given mesh and returns the result as a field.
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCowD<'a>;
}

impl Evaluable for FieldExpr {
    fn evaluate<'a>(&'a self, mesh: &'a UMeshView<'a>, dim: Option<Dimension>) -> FieldCowD<'a> {
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
            FieldExpr::Array(arr) => FieldCowD::from_array(arr.view().into(), elems.as_slice()),
            FieldExpr::Field(name) => mesh.field(name, Some(dim)).unwrap().into(),
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
            // FieldExpr::Measure => mesh.measure().to_owned(),
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

/// Trait for evaluating field expressions on a mesh.
pub trait MeshEvaluable {
    /// Evaluates an expression and returns the result as a new field.
    fn eval_field(&self, dim: Option<Dimension>, expr: FieldExpr) -> FieldOwnedD;
}

/// Trait for evaluating and storing field expressions.
pub trait MeshEvalUpdatable: MeshEvaluable {
    /// Evaluates an expression and stores the result as a named field in the mesh.
    fn eval_update_field(
        &mut self,
        name: &str,
        dim: Option<Dimension>,
        expr: FieldExpr,
    ) -> Option<FieldArcD>;
}

impl<N, C, F, G> MeshEvaluable for UMeshBase<N, C, F, G>
where
    N: nd::Data<Elem = f64>,
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    fn eval_field(&self, dim: Option<Dimension>, expr: FieldExpr) -> FieldOwnedD {
        expr.evaluate(&self.view(), dim).to_owned()
    }
}

impl MeshEvalUpdatable for UMesh {
    fn eval_update_field(
        &mut self,
        name: &str,
        dim: Option<Dimension>,
        expr: FieldExpr,
    ) -> Option<FieldArcD> {
        let field = self.eval_field(dim, expr);
        self.update_field(name, field.into_shared(), dim)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mesh_examples as me;
    use crate::tools::Measurable;

    #[test]
    fn compose_expr() {
        let a = field("toto");
        let b = field("exponent");
        let c = arr(nd::arr0(1.0));
        let _res = a.pow(b) + c;
    }

    #[test]
    fn measure_squared() {
        let mut m = me::make_imesh_2d(10);
        m.measure_update("M", None);
        let mes_squared5 = field("M").square() * arr(nd::arr0(5.));
        m.eval_field(None, mes_squared5);
    }
}
