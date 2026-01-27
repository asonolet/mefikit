use derive_where::derive_where;
use ndarray::{self as nd, ArrayBase};
use smallvec::SmallVec;
use std::{
    collections::{BTreeMap, HashSet},
    ops::{Add, Div, Mul, Sub},
    sync::Arc,
};

use crate::mesh::{Dimension, ElementType, UMeshView};

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

#[derive_where(Clone, Debug; S: nd::RawDataClone)]
pub struct FieldBase<S: nd::Data<Elem = f64>>(
    pub BTreeMap<ElementType, nd::ArrayBase<S, nd::IxDyn>>,
);
type FieldView<'a> = FieldBase<nd::ViewRepr<&'a f64>>;
type FieldOwned = FieldBase<nd::OwnedRepr<f64>>;
type FieldArc = FieldBase<nd::OwnedArcRepr<f64>>;
type FieldCow<'a> = FieldBase<nd::CowRepr<'a, f64>>;

impl<S> FieldBase<S>
where
    S: nd::Data<Elem = f64>,
{
    pub fn new(map: BTreeMap<ElementType, nd::ArrayBase<S, nd::IxDyn>>) -> Self {
        let res = Self(map);
        res.is_coherent();
        res
    }
    pub fn view(&self) -> FieldView<'_> {
        FieldView::new(
            self.0
                .iter()
                .map(|(k, v)| (*k, v.view()))
                .collect::<BTreeMap<_, _>>(),
        )
    }
    pub fn is_coherent(&self) -> bool {
        let first_array = self
            .0
            .values()
            .next()
            .expect("A field should not be empty.");
        let size_dim = &first_array.shape()[1..];
        for array in self.0.values() {
            if &array.shape()[1..] != size_dim {
                return false;
            }
        }
        true
    }
    pub fn is_strictly_compatible_with(&self, other: &Self) -> bool {
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
    pub fn may_be_compatible_with(&self, other: &Self) -> bool {
        let elems1 = self.0.keys().collect::<HashSet<_>>();
        let elems2 = other.0.keys().collect::<HashSet<_>>();
        elems1 == elems2
    }
    pub fn panic_if_not_strictly_compatible_with(&self, other: &Self) {
        if !self.is_strictly_compatible_with(other) {
            let dim0: Vec<_> = self.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            let dim1: Vec<_> = other.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            panic!("Fields with shapes {dim0:?}, {dim1:?} are not compatible for operation");
        }
    }
    pub fn panic_if_incompatible_with(&self, other: &Self) {
        if !self.may_be_compatible_with(other) {
            let elems1: Vec<_> = self.0.keys().collect();
            let elems2: Vec<_> = other.0.keys().collect();
            panic!(
                "Fields with element types {elems1:?}, {elems2:?} are not compatible for operation"
            );
        }
    }
    pub fn mapv<F>(&self, mut f: F) -> FieldOwned
    where
        F: FnMut(f64) -> f64,
    {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            let mapped_array = array.mapv(|x| f(x));
            result.insert(*elem_type, mapped_array.into_owned());
        }
        FieldOwned::new(result)
    }
    pub fn map_zip<F>(&self, other: &Self, mut f: F) -> FieldOwned
    where
        F: FnMut(f64, f64) -> f64,
    {
        self.panic_if_incompatible_with(other);
        let mut result = BTreeMap::new();
        let greatest_dim = if self.ndim() > other.ndim() {
            self.dim()
        } else {
            other.dim()
        };
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = other.0.get(elem_type) {
                let mut res = nd::ArrayD::<f64>::zeros(greatest_dim.clone());
                nd::Zip::from(&mut res)
                    .and_broadcast(left_array)
                    .and_broadcast(right_array)
                    .for_each(|a, &b, &c| *a = f(b, c));
                result.insert(*elem_type, res.into_owned());
            }
        }
        FieldOwned::new(result)
    }
    pub fn ndim(&self) -> usize {
        let first_array = self.0.values().next().unwrap();
        first_array.ndim()
    }

    pub fn dim(&self) -> nd::IxDyn {
        let first_array = self.0.values().next().unwrap();
        nd::IxDyn(&first_array.shape()[1..])
    }
    pub fn to_owned(&self) -> FieldOwned {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            result.insert(*elem_type, array.to_owned());
        }
        FieldOwned::new(result)
    }
    pub fn to_shared(&self) -> FieldArc {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            result.insert(*elem_type, array.to_shared());
        }
        FieldArc::new(result)
    }
    pub fn from_array<T>(array: ArrayBase<T, nd::IxDyn>, elems: &[ElementType]) -> FieldBase<T>
    where
        T: nd::Data<Elem = f64> + nd::RawDataClone,
    {
        let mut result = BTreeMap::new();
        for elem_type in elems {
            result.insert(*elem_type, array.clone());
        }
        FieldBase::new(result)
    }
}

impl<'a> From<FieldView<'a>> for FieldCow<'a> {
    fn from(value: FieldView<'a>) -> Self {
        let mut result: BTreeMap<ElementType, nd::CowArray<_, _>> = BTreeMap::new();
        for (elem_type, array) in value.0 {
            result.insert(elem_type, array.into());
        }
        FieldCow::new(result)
    }
}

impl<'a> From<FieldOwned> for FieldCow<'a> {
    fn from(value: FieldOwned) -> Self {
        let mut result: BTreeMap<ElementType, nd::CowArray<_, _>> = BTreeMap::new();
        for (elem_type, array) in value.0 {
            result.insert(elem_type, array.into());
        }
        FieldCow::new(result)
    }
}

impl<S> Add<&FieldBase<S>> for &FieldBase<S>
where
    S: nd::Data<Elem = f64>,
{
    type Output = FieldOwned;

    fn add(self, rhs: &FieldBase<S>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let sum_array = left_array + right_array;
                result.insert(*elem_type, sum_array.into_owned());
            }
        }
        FieldOwned::new(result)
    }
}

impl<S> Sub<&FieldBase<S>> for &FieldBase<S>
where
    S: nd::Data<Elem = f64>,
{
    type Output = FieldOwned;

    fn sub(self, rhs: &FieldBase<S>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let diff_array = left_array - right_array;
                result.insert(*elem_type, diff_array.into_owned());
            }
        }
        FieldOwned::new(result)
    }
}

impl<S> Mul<&FieldBase<S>> for &FieldBase<S>
where
    S: nd::Data<Elem = f64>,
{
    type Output = FieldOwned;

    fn mul(self, rhs: &FieldBase<S>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let prod_array = left_array * right_array;
                result.insert(*elem_type, prod_array.into_owned());
            }
        }
        FieldOwned::new(result)
    }
}

impl<S> Div<&FieldBase<S>> for &FieldBase<S>
where
    S: nd::Data<Elem = f64>,
{
    type Output = FieldOwned;

    fn div(self, rhs: &FieldBase<S>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = rhs.0.get(elem_type) {
                let div_array = left_array / right_array;
                result.insert(*elem_type, div_array.into_owned());
            }
        }
        FieldOwned::new(result)
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
