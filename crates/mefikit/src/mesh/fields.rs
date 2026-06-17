//! Field data structures for storing per-element values.
//!
//! Fields associate data arrays with element types, enabling storage of
//! scalar, vector, or tensor values on mesh elements.

use derive_where::derive_where;
use ndarray::{self as nd, ArrayBase, Axis};
use std::{
    collections::{BTreeMap, HashSet},
    ops::{Add, Div, Mul, Sub},
};

use crate::mesh::{Dimension, ElementIds, ElementType};

/// A generic field container mapping element types to data arrays.
///
/// Fields store per-element data (e.g., temperature, displacement) organized
/// by element type. The data arrays have shape `(num_elements, ...)` where
/// trailing dimensions represent the field's tensor structure.
#[derive_where(Clone, Debug; S: nd::RawDataClone)]
pub struct FieldBase<S: nd::Data<Elem = f64>, D: nd::Dimension>(
    pub BTreeMap<ElementType, nd::ArrayBase<S, D>>,
);
/// A view into a field with borrowed data.
pub type FieldView<'a, D> = FieldBase<nd::ViewRepr<&'a f64>, D>;
/// An owned field with uniquely owned data.
pub type FieldOwned<D> = FieldBase<nd::OwnedRepr<f64>, D>;
/// A shared (reference-counted) field.
pub type FieldArc<D> = FieldBase<nd::OwnedArcRepr<f64>, D>;
/// A copy-on-write field that can borrow or own data.
pub type FieldCow<'a, D> = FieldBase<nd::CowRepr<'a, f64>, D>;
/// A dynamic-dimension field view.
pub type FieldViewD<'a> = FieldBase<nd::ViewRepr<&'a f64>, nd::IxDyn>;
/// A dynamic-dimension owned field.
pub type FieldOwnedD = FieldBase<nd::OwnedRepr<f64>, nd::IxDyn>;
/// A dynamic-dimension shared field.
pub type FieldArcD = FieldBase<nd::OwnedArcRepr<f64>, nd::IxDyn>;
/// A dynamic-dimension copy-on-write field.
pub type FieldCowD<'a> = FieldBase<nd::CowRepr<'a, f64>, nd::IxDyn>;

impl<S, D> FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    /// Creates a new field from a map, validating coherence.
    ///
    /// # Panics
    /// Panics if the field map is empty or if arrays have incompatible shapes.
    pub fn new(map: BTreeMap<ElementType, nd::ArrayBase<S, D>>) -> Self {
        let res = Self(map);
        res.is_coherent();
        res
    }

    /// Returns a view of this field.
    pub fn view(&self) -> FieldView<'_, D> {
        FieldView::new(
            self.0
                .iter()
                .map(|(k, v)| (*k, v.view()))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    /// Returns the topological dimension of the field's elements, or `None` if empty.
    pub fn dimension(&self) -> Option<Dimension> {
        self.0.keys().next().map(|e| e.dimension())
    }

    /// Checks if all arrays in the field have compatible shapes.
    ///
    /// Returns `true` if all element types share the same dimension and
    /// all arrays have the same trailing dimensions.
    pub fn is_coherent(&self) -> bool {
        let first_array = self
            .0
            .values()
            .next()
            .expect("A field should not be empty.");
        if !self
            .0
            .keys()
            .all(|e| e.dimension() == self.dimension().unwrap())
        {
            return false;
        }
        if first_array.ndim() == 0 {
            for array in self.0.values() {
                if array.ndim() != 0 {
                    return false;
                }
            }
            return true;
        }
        let size_dim = &first_array.shape()[1..];
        for array in self.0.values() {
            if &array.shape()[1..] != size_dim {
                return false;
            }
        }
        true
    }

    /// Returns `true` if this field has the same element types and array shapes as `other`.
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

    /// Returns `true` if this field has the same element types as `other`.
    pub fn may_be_compatible_with(&self, other: &Self) -> bool {
        let elems1 = self.0.keys().collect::<HashSet<_>>();
        let elems2 = other.0.keys().collect::<HashSet<_>>();
        elems1 == elems2
    }

    /// Panics if fields are not strictly compatible.
    pub fn panic_if_not_strictly_compatible_with(&self, other: &Self) {
        if !self.is_strictly_compatible_with(other) {
            let dim0: Vec<_> = self.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            let dim1: Vec<_> = other.0.iter().map(|(k, a)| (*k, a.dim())).collect();
            panic!("Fields with shapes {dim0:?}, {dim1:?} are not compatible for operation");
        }
    }

    /// Panics if fields have different element types.
    pub fn panic_if_incompatible_with(&self, other: &Self) {
        if !self.may_be_compatible_with(other) {
            let elems1: Vec<_> = self.0.keys().collect();
            let elems2: Vec<_> = other.0.keys().collect();
            panic!(
                "Fields with element types {elems1:?}, {elems2:?} are not compatible for operation"
            );
        }
    }

    /// Applies a function element-wise to all values, returning a new owned field.
    pub fn mapv<F>(&self, mut f: F) -> FieldOwned<D>
    where
        F: FnMut(f64) -> f64,
    {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            let mapped_array = array.mapv(&mut f);
            result.insert(*elem_type, mapped_array.into_owned());
        }
        FieldOwned::new(result)
    }

    /// Applies a binary function element-wise to this field and another.
    pub fn map_zip<F>(&self, other: &Self, mut f: F) -> FieldOwned<nd::IxDyn>
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

    /// Returns element IDs where a binary predicate holds.
    pub fn map_zip_where<F>(&self, other: &Self, mut f: F) -> ElementIds
    where
        F: FnMut(f64, f64) -> bool,
    {
        self.panic_if_incompatible_with(other);
        let mut result = BTreeMap::new();
        let greatest_dim = if self.ndim() > other.ndim() {
            self.full_dim()
        } else {
            other.full_dim()
        };
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = other.0.get(elem_type) {
                let mut res = nd::ArrayD::<bool>::from_elem(greatest_dim, false);
                nd::Zip::from(&mut res)
                    .and_broadcast(left_array)
                    .and_broadcast(right_array)
                    .for_each(|a, &b, &c| *a = f(b, c));
                if res.ndim() == 1 {
                    res.insert_axis_inplace(Axis(1));
                }
                result.insert(
                    *elem_type,
                    res.rows()
                        .into_iter()
                        .enumerate()
                        .filter_map(|(i, b)| {
                            if b.into_iter().all(|&x| x) {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect(),
                );
            }
        }
        ElementIds(result)
    }

    /// Returns element IDs where this field is greater than `other`.
    pub fn gt(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a > b)
    }

    /// Returns element IDs where this field is greater than or equal to `other`.
    pub fn ge(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a >= b)
    }

    /// Returns element IDs where this field is less than `other`.
    pub fn lt(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a < b)
    }

    /// Returns element IDs where this field is less than or equal to `other`.
    pub fn le(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a <= b)
    }

    /// Returns element IDs where this field equals `other`.
    pub fn eq(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a == b)
    }

    /// Returns element IDs where this field does not equal `other`.
    pub fn neq(&self, other: &Self) -> ElementIds {
        self.map_zip_where(other, |a, b| a != b)
    }

    /// Returns the number of dimensions of the field arrays.
    pub fn ndim(&self) -> usize {
        let first_array = self.0.values().next().unwrap();
        first_array.ndim()
    }

    /// Returns the trailing dimensions (excluding the element count).
    pub fn dim(&self) -> nd::IxDyn {
        let first_array = self.0.values().next().unwrap();
        nd::IxDyn(&first_array.shape()[1..])
    }

    /// Returns the full shape of the first array.
    pub fn full_dim(&self) -> &[usize] {
        self.0.values().next().unwrap().shape()
    }

    /// Converts this field to an owned field.
    pub fn to_owned(&self) -> FieldOwned<D> {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            result.insert(*elem_type, array.to_owned());
        }
        FieldOwned::new(result)
    }

    /// Converts this field to a shared (reference-counted) field.
    pub fn to_shared(&self) -> FieldArc<D> {
        let mut result = BTreeMap::new();
        for (elem_type, array) in &self.0 {
            result.insert(*elem_type, array.to_shared());
        }
        FieldArc::new(result)
    }

    /// Consumes this field and returns a shared version.
    pub fn into_shared(self) -> FieldArc<D>
    where
        S: nd::DataOwned,
    {
        let mut result = BTreeMap::new();
        for (elem_type, array) in self.0 {
            result.insert(elem_type, array.into_shared());
        }
        FieldArc::new(result)
    }

    /// Creates a field by broadcasting a single array to multiple element types.
    pub fn from_array<T>(array: ArrayBase<T, D>, elems: &[ElementType]) -> FieldBase<T, D>
    where
        T: nd::Data<Elem = f64> + nd::RawDataClone,
    {
        let mut result = BTreeMap::new();
        for elem_type in elems {
            result.insert(*elem_type, array.clone());
        }
        FieldBase::new(result)
    }

    /// Converts this field to use dynamic dimensions.
    pub fn into_dyn(self) -> FieldBase<S, nd::IxDyn> {
        let mut result = BTreeMap::new();
        for (elem_type, array) in self.0 {
            result.insert(elem_type, array.into_dyn());
        }
        FieldBase::new(result)
    }
}

impl<'a, D: nd::Dimension> From<FieldView<'a, D>> for FieldCow<'a, D> {
    fn from(value: FieldView<'a, D>) -> Self {
        let mut result: BTreeMap<ElementType, nd::CowArray<_, _>> = BTreeMap::new();
        for (elem_type, array) in value.0 {
            result.insert(elem_type, array.into());
        }
        FieldCow::new(result)
    }
}

impl<'a, D: nd::Dimension> From<FieldOwned<D>> for FieldCow<'a, D> {
    fn from(value: FieldOwned<D>) -> Self {
        let mut result: BTreeMap<ElementType, nd::CowArray<_, _>> = BTreeMap::new();
        for (elem_type, array) in value.0 {
            result.insert(elem_type, array.into());
        }
        FieldCow::new(result)
    }
}

impl<S, D> Add<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<D>;

    /// Element-wise addition of two fields.
    fn add(self, rhs: &FieldBase<S, D>) -> Self::Output {
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

impl<S, D> Sub<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<D>;

    /// Element-wise subtraction of two fields.
    fn sub(self, rhs: &FieldBase<S, D>) -> Self::Output {
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

impl<S, D> Mul<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<D>;

    /// Element-wise multiplication of two fields.
    fn mul(self, rhs: &FieldBase<S, D>) -> Self::Output {
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

impl<S, D> Div<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<D>;

    /// Element-wise division of two fields.
    fn div(self, rhs: &FieldBase<S, D>) -> Self::Output {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use ndarray as nd;

    #[test]
    fn test_fieldbase_new() {
        let mut map = BTreeMap::new();
        map.insert(ElementType::QUAD4, nd::arr0(1.0).into_dyn());
        let field = FieldBase::new(map);
        assert_eq!(field.dimension(), Some(crate::mesh::Dimension::D2));
    }

    #[test]
    fn test_fieldbase_view() {
        let mut map = BTreeMap::new();
        map.insert(ElementType::QUAD4, nd::arr0(1.0).into_dyn());
        let field = FieldBase::new(map);
        let view = field.view();
        assert_eq!(view.dimension(), Some(crate::mesh::Dimension::D2));
    }

    #[test]
    fn test_fieldbase_dimension() {
        let mut map = BTreeMap::new();
        map.insert(ElementType::SEG2, nd::arr0(1.0).into_dyn());
        let field = FieldBase::new(map);
        assert_eq!(field.dimension(), Some(crate::mesh::Dimension::D1));
    }

    #[test]
    fn test_fieldbase_is_coherent() {
        let mut map = BTreeMap::new();
        map.insert(ElementType::QUAD4, nd::arr0(1.0).into_dyn());
        map.insert(ElementType::TRI3, nd::arr0(2.0).into_dyn());
        let field = FieldBase::new(map);
        assert!(field.is_coherent());
    }

    #[test]
    fn test_fieldbase_is_strictly_compatible_with() {
        let mut map1 = BTreeMap::new();
        map1.insert(ElementType::QUAD4, nd::arr0(1.0).into_dyn());
        let field1 = FieldBase::new(map1);

        let mut map2 = BTreeMap::new();
        map2.insert(ElementType::QUAD4, nd::arr0(2.0).into_dyn());
        let field2 = FieldBase::new(map2);

        assert!(field1.is_strictly_compatible_with(&field2));
    }

    #[test]
    fn test_fieldbase_may_be_compatible_with() {
        let mut map1 = BTreeMap::new();
        map1.insert(ElementType::QUAD4, nd::arr0(1.0).into_dyn());
        let field1 = FieldBase::new(map1);

        let mut map2 = BTreeMap::new();
        map2.insert(ElementType::QUAD4, nd::arr0(2.0).into_dyn());
        let field2 = FieldBase::new(map2);

        assert!(field1.may_be_compatible_with(&field2));
    }

    #[test]
    fn test_fieldbase_mapv() {
        let mut map = BTreeMap::new();
        map.insert(ElementType::QUAD4, nd::arr1(&[1.0, 2.0, 3.0]).into_dyn());
        let field = FieldBase::new(map);
        let mapped = field.mapv(|x| x * 2.0);
        let result = mapped.0.get(&ElementType::QUAD4).unwrap();
        assert_eq!(result[0], 2.0);
        assert_eq!(result[1], 4.0);
        assert_eq!(result[2], 6.0);
    }
}
