use ndarray::{self as nd, Dimension};
use smallvec::SmallVec;
use std::{
    collections::BTreeMap,
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::mesh::{ElementType, UMeshView};

#[derive(Clone, Debug)]
pub enum FieldExpr {
    Array(nd::ArrayD<f64>),
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

pub struct Field<'a>(BTreeMap<ElementType, nd::CowArray<'a, f64, nd::IxDyn>>);

impl<'a> From<BTreeMap<ElementType, nd::CowArray<'a, f64, nd::IxDyn>>> for Field<'a> {
    fn from(map: BTreeMap<ElementType, nd::CowArray<'a, f64, nd::IxDyn>>) -> Self {
        Field(map)
    }
}

impl Field<'_> {
    pub fn view(&self) -> &BTreeMap<ElementType, nd::CowArray<'_, f64, nd::IxDyn>> {
        &self.0
    }
    pub fn is_coherent(&self) -> bool {
        let first_array = self.0.values().next().unwrap();
        let size_dim = &first_array.shape()[1..];
        for (_elem_type, array) in &self.0 {
            if &array.shape()[1..] != size_dim {
                return false;
            }
        }
        true
    }
    pub fn is_compatible_with(&self, other: &Field<'_>) -> bool {
        for (elem_type, left_array) in &self.0 {
            match other.0.get(elem_type) {
                Some(right_array) => {
                    if right_array.shape() != left_array.shape() {
                        return false;
                    }
                }
                None => return false,
            }
        }
        true
    }
}

impl Add<&Field<'_>> for &Field<'_> {
    type Output = Field<'static>;

    fn add(self, rhs: &Field<'_>) -> Field<'static> {
        let mut result = BTreeMap::new();
        if !self.is_compatible_with(rhs) {
            let dim0 = self
                .0
                .iter()
                .map(|(k, a)| (*k, a.dim()))
                .collect::<Vec<_>>();
            let dim1 = rhs.0.iter().map(|(k, a)| (*k, a.dim())).collect::<Vec<_>>();
            panic!("Fields with shapes {dim0:?}, {dim1:?} are not compatible for addition");
        }
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let sum_array = left_array + right_array;
                result.insert(*elem_type, sum_array.into_owned().into());
            }
        }
        Field(result)
    }
}

impl Sub<&Field<'_>> for &Field<'_> {
    type Output = Field<'static>;

    fn sub(self, rhs: &Field<'_>) -> Field<'static> {
        if !self.is_compatible_with(rhs) {
            let dim0: Vec<_> = self.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            let dim1: Vec<_> = rhs.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            panic!("Fields with shapes {dim0:?}, {dim1:?} are not compatible for addition");
        }
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let diff_array = left_array - right_array;
                result.insert(*elem_type, diff_array.into_owned().into());
            }
        }
        Field(result)
    }
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
    fn evaluate<'a>(&'a self, mesh: UMeshView<'a>) -> nd::CowArray<'a, f64, nd::IxDyn>;
}

impl Evaluable for FieldExpr {
    fn evaluate<'a>(
        &'a self,
        mesh: UMeshView<'a>,
    ) -> BTreeMap<ElementType, nd::CowArray<'a, f64, nd::IxDyn>> {
        match self {
            FieldExpr::Array(arr) => arr.view().into(),
            // FieldExpr::Field(name) => mesh.field(name).unwrap().to_owned(),
            FieldExpr::BinarayExpr {
                operator,
                left,
                right,
            } => {
                let left_eval = left.evaluate(mesh.clone());
                let right_eval = right.evaluate(mesh.clone());
                match operator {
                    BinaryOp::Add => (&left_eval + &right_eval).into(),
                    BinaryOp::Sub => (&left_eval - &right_eval).into(),
                    BinaryOp::Mul => (&left_eval * &right_eval).into(),
                    BinaryOp::Div => (&left_eval / &right_eval).into(),
                    BinaryOp::Pow => {
                        // find the greatest dimension and broadcast accordingly
                        let greatest_dim = if left_eval.ndim() > right_eval.ndim() {
                            left_eval.dim()
                        } else {
                            right_eval.dim()
                        };
                        let mut res = nd::ArrayD::<f64>::zeros(greatest_dim);
                        nd::Zip::from(&mut res)
                            .and_broadcast(&left_eval)
                            .and_broadcast(&right_eval)
                            .for_each(|a, &b, &c| *a = b.powf(c));
                        res.into()
                    }
                }
            }
            FieldExpr::UnaryExpr { operator, expr } => {
                let expr_eval = expr.evaluate(mesh.clone());
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
            // FieldExpr::Index(expr, index) => {
            //     let eval = expr.evaluate(mesh);
            //     eval[.., [index.try_into().unwrap()]].to_owned()
            // }
            _ => todo!(),
        }
    }
}
