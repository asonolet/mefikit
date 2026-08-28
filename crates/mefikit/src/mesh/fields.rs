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

    /// Applies a binary function element-wise with broadcast to this field and another.
    pub fn map_zip_broadcast<F>(&self, other: &Self, f: F) -> FieldOwned<nd::IxDyn>
    where
        F: Fn(f64, f64) -> f64 + Copy,
    {
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = other.0.get(elem_type) {
                let res = broadcast_binary_op(
                    &left_array.view().into_dyn(),
                    &right_array.view().into_dyn(),
                    f,
                );
                result.insert(*elem_type, res.into_owned());
            }
        }
        FieldOwned::new(result)
    }

    /// Applies a per-element matrix product (shorthand `@`) between this field
    /// and another, propagating numpy `matmul` semantics along the first (element)
    /// axis and broadcasting a second operand that lacks the element axis.
    ///
    /// Supported per-element contracts (element axis `n` kept on the first axis):
    /// - `[k] @ [k] -> [1]` (vector dot, e.g. `[n, 3] @ [3] -> [n, 1]`)
    /// - `[m, k] @ [k] -> [m]` (e.g. `[n, 3, 3] @ [n, 3] -> [n, 3]`)
    /// - `[k] @ [k, p] -> [p]`
    /// - `[m, k] @ [k, p] -> [m, p]` (e.g. `[n, 3, 3] @ [3, 3] -> [n, 3, 3]`)
    pub fn map_matmul(&self, other: &Self) -> FieldOwned<nd::IxDyn> {
        self.panic_if_incompatible_with(other);
        let mut result = BTreeMap::new();
        for (elem_type, left_array) in &self.0 {
            if let Some(right_array) = other.0.get(elem_type) {
                let res = matmul_broadcast_op(
                    left_array.view().into_dyn(),
                    right_array.view().into_dyn(),
                );
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
    type Output = FieldOwned<nd::IxDyn>;

    /// Element-wise addition of two fields, with broadcasting.
    fn add(self, rhs: &FieldBase<S, D>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        self.map_zip_broadcast(rhs, |a, b| a + b)
    }
}

impl<S, D> Sub<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<nd::IxDyn>;

    /// Element-wise subtraction of two fields, with broadcasting.
    fn sub(self, rhs: &FieldBase<S, D>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        self.map_zip_broadcast(rhs, |a, b| a - b)
    }
}

fn broadcast_binary_op<S1, S2, T, F>(
    lhs: &nd::ArrayBase<S1, nd::IxDyn>,
    rhs: &nd::ArrayBase<S2, nd::IxDyn>,
    op: F,
) -> nd::ArrayD<T>
where
    S1: nd::Data<Elem = T>,
    S2: nd::Data<Elem = T>,
    T: Clone,
    F: Fn(T, T) -> T + Copy,
{
    if lhs.shape() == rhs.shape() {
        return nd::Zip::from(lhs)
            .and(rhs)
            .map_collect(|a, b| op(a.clone(), b.clone()));
    }

    if let Some(rhs) = rhs.broadcast(lhs.raw_dim()) {
        return nd::Zip::from(lhs)
            .and(rhs)
            .map_collect(|a, b| op(a.clone(), b.clone()));
    }

    if let Some(lhs) = lhs.broadcast(rhs.raw_dim()) {
        return nd::Zip::from(lhs)
            .and(rhs)
            .map_collect(|a, b| op(a.clone(), b.clone()));
    }

    panic!(
        "incompatible shapes: {:?} and {:?}",
        lhs.shape(),
        rhs.shape()
    );
}

/// Per-element matrix product between a field block (with leading element axis) and
/// another block (same element axis or a plain constant array to broadcast across it).
///
/// Mirrors numpy `matmul` semantics propagated along the first axis, except that a
/// vector·vector product yields a `[n, 1]` column-vector rather than a bare `[n]`.
fn matmul_broadcast_op(
    lhs: nd::ArrayViewD<'_, f64>,
    rhs: nd::ArrayViewD<'_, f64>,
) -> nd::ArrayD<f64> {
    let n = lhs.shape()[0];
    let lhs_shape = lhs.shape().to_vec();
    let rhs_shape = rhs.shape().to_vec();

    // The right operand carries the element axis only if its leading axis equals `n`
    // and it has at least two dimensions (so the leading axis is not a lone vector
    // component of a broadcast constant).
    let r_has_n = rhs.ndim() >= 2 && rhs.shape()[0] == n;

    // Per-element tensor ranks: 1 = vector, 2 = matrix.
    let l_rank = lhs.ndim() - 1;
    let r_rank = if r_has_n { rhs.ndim() - 1 } else { rhs.ndim() };
    let l_is_vec = l_rank == 1;
    let r_is_vec = r_rank == 1;

    // Left operand becomes a `[n, lm, lk]` matrix stack, contracted on its last axis.
    // Returns `(stack, contraction, other)` so here that is `(stack, lk, lm)`.
    let (l_stack, lk, lm) = operand_to_matrix(
        lhs, n, l_rank, /*contraction_is_first=*/ false, /*offset=*/ 1,
    );
    // Right operand becomes a `[elems, rk, rp]` matrix stack, contracted on its first axis
    // (the rows); for a shared broadcast constant `elems` is 1.
    let (r_stack, rk, rp) = operand_to_matrix(
        rhs,
        if r_has_n { n } else { 1 },
        r_rank,
        /*contraction_is_first=*/ true,
        /*offset=*/ if r_has_n { 1 } else { 0 },
    );

    if lk != rk {
        panic!(
            "matmul inner dimensions do not match: lhs {:?}, rhs {:?}",
            lhs_shape, rhs_shape
        );
    }

    let mut out = nd::Array3::<f64>::zeros((n, lm, rp));
    for i in 0..n {
        let lrow = l_stack.index_axis(nd::Axis(0), i);
        let rrow = r_stack.index_axis(nd::Axis(0), if r_has_n { i } else { 0 });
        let product = lrow.dot(&rrow);
        out.slice_mut(nd::s![i, .., ..]).assign(&product);
    }

    // Assemble the trailing result dims, keeping the element axis first:
    // - vec·vec `[k]@[k]`      -> `[1]`
    // - mat·vec `[m,k]@[k]`    -> `[m]`
    // - vec·mat `[k]@[k,p]`    -> `[p]`
    // - mat·mat `[m,k]@[k,p]`  -> `[m,p]`
    let mut final_shape: Vec<usize> = vec![n];
    if l_is_vec && r_is_vec {
        final_shape.push(1);
    } else {
        if !l_is_vec {
            final_shape.push(lm);
        }
        if !r_is_vec {
            final_shape.push(rp);
        }
    }

    out.into_shape_with_order(nd::IxDyn(&final_shape))
        .expect("matmul result shape is always valid")
}

/// Converts an operand's per-element tensor into a `[elems, rows, cols]` matrix stack.
///
/// A vector tensor `[k]` becomes `[1, k]` (a row vector) when used on the left and
/// `[k, 1]` (a column vector) when used on the right; a matrix tensor `[m, k]` is kept as
/// `[m, k]` in both cases. `contraction_is_first` selects which operand side we represent,
/// and `offset` is the index of the first per-element tensor axis (1 for field operands,
/// which carry the leading element axis, 0 for shared broadcast constants).
///
/// Returns `(stack, contraction_dim, other_dim)`.
fn operand_to_matrix(
    block: nd::ArrayViewD<'_, f64>,
    elems: usize,
    rank: usize,
    contraction_is_first: bool,
    offset: usize,
) -> (nd::Array3<f64>, usize, usize) {
    let shape = block.shape();
    let (rows, cols, contraction) = match (rank, contraction_is_first) {
        // left vector -> [1, k]
        (1, false) => (1, shape[offset], shape[offset]),
        // right vector -> [k, 1]
        (1, true) => (shape[offset], 1, shape[offset]),
        // matrix -> [m, k]; left contracts on k, right on m
        (2, _) => (
            shape[offset],
            shape[offset + 1],
            shape[offset + if contraction_is_first { 0 } else { 1 }],
        ),
        _ => panic!(
            "matmul operands must be vector- or matrix-valued per element, got shape {:?}",
            block.shape()
        ),
    };
    let stack = block
        .to_owned()
        .into_shape_with_order((elems, rows, cols))
        .unwrap();
    (
        stack,
        contraction,
        if contraction_is_first { cols } else { rows },
    )
}

impl<S> Mul<&FieldBase<S, nd::IxDyn>> for &FieldBase<S, nd::IxDyn>
where
    S: nd::Data<Elem = f64>,
{
    type Output = FieldOwned<nd::IxDyn>;

    /// Element-wise multiplication of two fields, with broadcasting.
    fn mul(self, rhs: &FieldBase<S, nd::IxDyn>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        self.map_zip_broadcast(rhs, |a, b| a * b)
    }
}

impl<S, D> Div<&FieldBase<S, D>> for &FieldBase<S, D>
where
    S: nd::Data<Elem = f64>,
    D: nd::Dimension,
{
    type Output = FieldOwned<nd::IxDyn>;

    /// Element-wise division of two fields, with broadcasting.
    fn div(self, rhs: &FieldBase<S, D>) -> Self::Output {
        self.panic_if_incompatible_with(rhs);
        self.map_zip_broadcast(rhs, |a, b| a / b)
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

    fn mul_field(lhs: nd::ArrayD<f64>, rhs: nd::ArrayD<f64>) -> nd::ArrayD<f64> {
        let mut lm = BTreeMap::new();
        lm.insert(ElementType::QUAD4, lhs);
        let lf = FieldBase::new(lm);

        let mut rm = BTreeMap::new();
        rm.insert(ElementType::QUAD4, rhs);
        let rf = FieldBase::new(rm);

        (&lf * &rf)
            .0
            .remove(&ElementType::QUAD4)
            .unwrap()
            .into_dyn()
    }

    #[test]
    fn test_mul_broadcast_scalar() {
        // [n, 3] * [] -> [n, 3]
        let lhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let rhs = nd::arr0(2.0).into_dyn();
        let res = mul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]].into_dyn()
        );
    }

    #[test]
    fn test_mul_broadcast_vector() {
        // [n, 3] * [3] -> [n, 3]
        let lhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let rhs = nd::array![10.0, 100.0, 1000.0].into_dyn();
        let res = mul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[10.0, 200.0, 3000.0], [40.0, 500.0, 6000.0]].into_dyn()
        );
    }

    #[test]
    fn test_mul_broadcast_matrix() {
        // [n, 3, 3] * [3, 3] -> [n, 3, 3]
        let lhs = nd::array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]].into_dyn();
        let rhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into_dyn();
        let res = mul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[[1.0, 4.0, 9.0], [16.0, 25.0, 36.0], [49.0, 64.0, 81.0]]].into_dyn()
        );
    }

    #[test]
    fn test_mul_same_shape() {
        // [n, 3] * [n, 3] -> [n, 3] (no broadcast)
        let lhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let rhs = nd::array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]].into_dyn();
        let res = mul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[2.0, 6.0, 12.0], [20.0, 30.0, 42.0]].into_dyn()
        );
    }

    #[test]
    fn test_add_broadcast() {
        // [n, 3] + [3] -> [n, 3]
        let mut lm = BTreeMap::new();
        lm.insert(ElementType::QUAD4, nd::array![[1.0, 2.0, 3.0]].into_dyn());
        let lf = FieldBase::new(lm);
        let mut rm = BTreeMap::new();
        rm.insert(ElementType::QUAD4, nd::arr1(&[10.0, 20.0, 30.0]).into_dyn());
        let rf = FieldBase::new(rm);
        let res = (&lf + &rf).0.remove(&ElementType::QUAD4).unwrap();
        assert_eq!(res, nd::array![[11.0, 22.0, 33.0]].into_dyn());
    }

    #[test]
    fn test_div_broadcast_scalar() {
        // [n, 3] / [] -> [n, 3]
        let mut lm = BTreeMap::new();
        lm.insert(ElementType::QUAD4, nd::array![[2.0, 4.0, 6.0]].into_dyn());
        let lf = FieldBase::new(lm);
        let mut rm = BTreeMap::new();
        rm.insert(ElementType::QUAD4, nd::arr0(2.0).into_dyn());
        let rf = FieldBase::new(rm);
        let res = (&lf / &rf).0.remove(&ElementType::QUAD4).unwrap();
        assert_eq!(res, nd::array![[1.0, 2.0, 3.0]].into_dyn());
    }

    #[test]
    #[should_panic(expected = "incompatible shapes")]
    fn test_mul_incompatible_shapes_panics() {
        // [2, 3] * [2, 5] -> panic
        let lhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let rhs = nd::array![[1.0, 2.0, 3.0, 4.0, 5.0], [6.0, 7.0, 8.0, 9.0, 10.0]].into_dyn();
        let _ = mul_field(lhs, rhs);
    }

    #[test]
    fn test_map_zip_broadcast_matches_ops() {
        // map_zip_broadcast with mul should equal * operator
        let mut lm = BTreeMap::new();
        lm.insert(ElementType::QUAD4, nd::array![[1.0, 2.0, 3.0]].into_dyn());
        let lf = FieldBase::new(lm);
        let mut rm = BTreeMap::new();
        rm.insert(ElementType::QUAD4, nd::arr1(&[2.0, 3.0, 4.0]).into_dyn());
        let rf = FieldBase::new(rm);

        let via_method = lf
            .map_zip_broadcast(&rf, |a, b| a * b)
            .0
            .remove(&ElementType::QUAD4)
            .unwrap();
        let via_op = (&lf * &rf).0.remove(&ElementType::QUAD4).unwrap();
        assert_eq!(via_method, via_op);
    }

    fn matmul_field(lhs: nd::ArrayD<f64>, rhs: nd::ArrayD<f64>) -> nd::ArrayD<f64> {
        let mut lm = BTreeMap::new();
        lm.insert(ElementType::QUAD4, lhs);
        let lf = FieldBase::new(lm);

        let mut rm = BTreeMap::new();
        rm.insert(ElementType::QUAD4, rhs);
        let rf = FieldBase::new(rm);

        lf.map_matmul(&rf)
            .0
            .remove(&ElementType::QUAD4)
            .unwrap()
            .into_dyn()
    }

    #[test]
    fn test_matmul_vector_times_const_vector() {
        // [n, 3] @ [3] -> [n, 1]
        let lhs = nd::array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let rhs = nd::array![1.0, 2.0, 3.0].into_dyn();
        let res = matmul_field(lhs, rhs);
        assert_eq!(res, nd::array![[14.0], [32.0]].into_dyn());
    }

    #[test]
    fn test_matmul_matrix_times_const_matrix() {
        // [n, 3, 3] @ [3, 3] -> [n, 3, 3]
        let lhs = nd::array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]].into_dyn();
        let rhs = nd::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]].into_dyn();
        let res = matmul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]].into_dyn()
        );
    }

    #[test]
    fn test_matmul_matrix_field_times_vector_field() {
        // [n, 3, 3] @ [n, 3] -> [n, 3]
        let lhs = nd::array![
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        ]
        .into_dyn();
        let rhs = nd::array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn();
        let res = matmul_field(lhs, rhs);
        assert_eq!(
            res,
            nd::array![[1.0, 2.0, 3.0], [6.0, 15.0, 24.0]].into_dyn()
        );
    }

    #[test]
    fn test_matmul_vector_times_const_matrix() {
        // [n, 3] @ [3, 2] -> [n, 2]
        let lhs = nd::array![[1.0, 2.0, 3.0]].into_dyn();
        let rhs = nd::array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]].into_dyn();
        let res = matmul_field(lhs, rhs);
        assert_eq!(res, nd::array![[4.0, 5.0]].into_dyn());
    }

    #[test]
    #[should_panic(expected = "inner dimensions do not match")]
    fn test_matmul_incompatible_inner_dim_panics() {
        // [n, 3] @ [4] -> panic
        let lhs = nd::array![[1.0, 2.0, 3.0]].into_dyn();
        let rhs = nd::array![1.0, 2.0, 3.0, 4.0].into_dyn();
        let _ = matmul_field(lhs, rhs);
    }
}
