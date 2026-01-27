use derive_where::derive_where;
use ndarray::{self as nd, ArrayBase};
use std::{
    collections::{BTreeMap, HashSet},
    ops::{Add, Div, Mul, Sub},
};

use crate::mesh::ElementType;

#[derive_where(Clone, Debug; S: nd::RawDataClone)]
pub struct FieldBase<S: nd::Data<Elem = f64>>(
    pub BTreeMap<ElementType, nd::ArrayBase<S, nd::IxDyn>>,
);
pub type FieldView<'a> = FieldBase<nd::ViewRepr<&'a f64>>;
pub type FieldOwned = FieldBase<nd::OwnedRepr<f64>>;
pub type FieldArc = FieldBase<nd::OwnedArcRepr<f64>>;
pub type FieldCow<'a> = FieldBase<nd::CowRepr<'a, f64>>;

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
