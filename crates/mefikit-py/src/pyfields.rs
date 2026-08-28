use std::collections::{BTreeMap, HashSet};

use numpy as np;
use numpy::ndarray as nd;
use pyo3::exceptions::{PyKeyError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyDictMethods, PyIterator, PyList, PyTuple};

use mefikit::mesh::{Dimension, ElementType};
use mefikit::prelude as mf;
use mefikit::tools::{
    MeshSelect,
    fieldexpr::{FieldExpr, MeshEvalUpdatable, MeshEvaluable},
};

use super::element::{etype_to_str, try_str_to_etype};
use super::pyfield::PyField;
use super::pyumesh::PyUMesh;
use super::select::{Selector, extract_selector};

pub(crate) enum ReduceOp {
    Min,
    Max,
    Sum,
    Mean,
    Var,
    Std,
}

pub(crate) enum ReduceVal {
    Scalar(f64),
    Array(nd::ArrayD<f64>),
}

impl ReduceVal {
    fn from_array(arr: nd::ArrayD<f64>) -> Self {
        if arr.ndim() == 0 {
            ReduceVal::Scalar(arr.first().copied().unwrap_or(f64::NAN))
        } else {
            ReduceVal::Array(arr)
        }
    }
}

pub(crate) fn reduce_to_py(py: Python<'_>, val: ReduceVal) -> Py<PyAny> {
    match val {
        ReduceVal::Scalar(f) => f
            .into_pyobject(py)
            .expect("float conversion")
            .unbind()
            .into_any(),
        ReduceVal::Array(a) => np::PyArray::from_owned_array(py, a).unbind().into_any(),
    }
}

fn elem_count(view: &nd::ArrayViewD<'_, f64>) -> usize {
    if view.ndim() == 0 { 1 } else { view.shape()[0] }
}

fn block_stat(view: &nd::ArrayViewD<'_, f64>, op: &ReduceOp) -> nd::ArrayD<f64> {
    if view.ndim() == 0 {
        return view.to_owned();
    }
    match op {
        ReduceOp::Min => view.fold_axis(nd::Axis(0), f64::INFINITY, |a, &b| a.min(b)),
        ReduceOp::Max => view.fold_axis(nd::Axis(0), f64::NEG_INFINITY, |a, &b| a.max(b)),
        _ => view.mean_axis(nd::Axis(0)).expect("non-empty axis"),
    }
}

fn accumulate(total: &mut Option<nd::ArrayD<f64>>, term: nd::ArrayD<f64>) {
    match total {
        None => *total = Some(term),
        Some(t) => *t += &term,
    }
}

pub(crate) fn reduce_blocks(
    views: &[nd::ArrayViewD<'_, f64>],
    op: ReduceOp,
    ddof: usize,
) -> PyResult<ReduceVal> {
    let non_empty: Vec<nd::ArrayViewD<'_, f64>> = views
        .iter()
        .filter(|v| v.ndim() == 0 || v.shape().first().is_some_and(|&n| n > 0))
        .cloned()
        .collect();
    let op_name = match op {
        ReduceOp::Min => "min",
        ReduceOp::Max => "max",
        ReduceOp::Sum => "sum",
        ReduceOp::Mean => "mean",
        ReduceOp::Var => "var",
        ReduceOp::Std => "std",
    };
    let empty_err =
        || PyValueError::new_err(format!("cannot compute {op_name} over zero elements"));

    match op {
        ReduceOp::Min | ReduceOp::Max => {
            let mut acc: Option<nd::ArrayD<f64>> = None;
            for v in &non_empty {
                let stat = block_stat(v, &op);
                match &mut acc {
                    None => acc = Some(stat),
                    Some(a) => a.zip_mut_with(&stat, |x, y| {
                        *x = if matches!(op, ReduceOp::Min) {
                            x.min(*y)
                        } else {
                            x.max(*y)
                        };
                    }),
                }
            }
            acc.map(ReduceVal::from_array).ok_or_else(empty_err)
        }
        ReduceOp::Sum => {
            let mut total: Option<nd::ArrayD<f64>> = None;
            for v in &non_empty {
                let s = if v.ndim() == 0 {
                    v.to_owned()
                } else {
                    v.sum_axis(nd::Axis(0))
                };
                accumulate(&mut total, s);
            }
            total.map(ReduceVal::from_array).ok_or_else(empty_err)
        }
        ReduceOp::Mean | ReduceOp::Var | ReduceOp::Std => {
            let total_n: usize = non_empty.iter().map(|v| elem_count(v)).sum();
            if total_n == 0 {
                return Err(empty_err());
            }
            let n_f = total_n as f64;
            let mut mean_acc: Option<nd::ArrayD<f64>> = None;
            for v in &non_empty {
                let term = block_stat(v, &ReduceOp::Mean) * (elem_count(v) as f64 / n_f);
                accumulate(&mut mean_acc, term);
            }
            let mean = mean_acc.ok_or_else(empty_err)?;
            if matches!(op, ReduceOp::Mean) {
                return Ok(ReduceVal::from_array(mean));
            }
            let mut m2_acc: Option<nd::ArrayD<f64>> = None;
            for v in &non_empty {
                let sq = if v.ndim() == 0 {
                    v.to_owned()
                } else {
                    v.mapv(|x| x * x)
                        .mean_axis(nd::Axis(0))
                        .expect("non-empty axis")
                };
                let term = sq * (elem_count(v) as f64 / n_f);
                accumulate(&mut m2_acc, term);
            }
            let m2 = m2_acc.ok_or_else(empty_err)?;
            let mut var = m2 - &mean * &mean;
            if ddof > 0 {
                if total_n <= ddof {
                    return Err(PyValueError::new_err(format!(
                        "ddof ({ddof}) must be smaller than the number of elements ({total_n})"
                    )));
                }
                let corr = n_f / (total_n - ddof) as f64;
                var.mapv_inplace(|x| x * corr);
            }
            if matches!(op, ReduceOp::Var) {
                Ok(ReduceVal::from_array(var))
            } else {
                Ok(ReduceVal::from_array(var.mapv(f64::sqrt)))
            }
        }
    }
}

pub(crate) fn measure_weighted(
    meas: &nd::Array1<f64>,
    val: nd::ArrayViewD<'_, f64>,
) -> PyResult<nd::ArrayD<f64>> {
    if val.ndim() < 1 {
        return Err(PyValueError::new_err(
            "cannot integrate a field without an element axis",
        ));
    }
    let n = val.shape()[0];
    if meas.len() != n {
        return Err(PyValueError::new_err(format!(
            "measure length {} does not match the {} elements of the field",
            meas.len(),
            n
        )));
    }
    if val.ndim() == 1 {
        return Ok((val.to_owned() * meas).into_dyn());
    }
    let mut mshape = vec![n];
    mshape.extend(std::iter::repeat_n(1usize, val.ndim() - 1));
    let expanded =
        nd::ArrayD::from_shape_vec(mshape, meas.to_vec()).expect("measure reshape is always valid");
    Ok(val.to_owned() * expanded)
}

enum FieldValue {
    Expr(Box<FieldExpr>),
    Broadcast(nd::ArcArrayD<f64>),
    PerBlock(BTreeMap<ElementType, nd::ArcArrayD<f64>>),
}

fn extract_field_value(value: &Bound<'_, PyAny>) -> PyResult<FieldValue> {
    if let Ok(expr) = value.extract::<PyField>() {
        return Ok(FieldValue::Expr(Box::new(expr.into())));
    }
    if let Ok(name) = value.extract::<String>() {
        return Ok(FieldValue::Expr(Box::new(mf::fieldexpr::field(&name))));
    }
    if let Ok(scalar) = value.extract::<f64>() {
        return Ok(FieldValue::Broadcast(
            nd::arr0(scalar).into_dyn().to_shared(),
        ));
    }
    if let Ok(arr) = value.extract::<np::PyReadonlyArray<'_, f64, nd::IxDyn>>() {
        return Ok(FieldValue::Broadcast(arr.as_array().to_shared()));
    }
    if let Ok(dict) = value.cast::<PyDict>() {
        let mut map = BTreeMap::new();
        for (key, val) in dict.iter() {
            let et_str: String = key.extract()?;
            let et = try_str_to_etype(&et_str).map_err(PyValueError::new_err)?;
            let arr = val.extract::<np::PyReadonlyArray<'_, f64, nd::IxDyn>>()?;
            map.insert(et, arr.as_array().to_shared());
        }
        if map.is_empty() {
            return Err(PyValueError::new_err("empty per-block field dict"));
        }
        return Ok(FieldValue::PerBlock(map));
    }
    Err(PyTypeError::new_err(
        "expected a Field expression, a float, a numpy array or a dict {\"ETYPE\": array}",
    ))
}

fn find_instances<'a>(inner: &'a mf::UMesh, name: &str) -> Vec<(Dimension, mf::FieldViewD<'a>)> {
    inner
        .fields()
        .filter(|(n, _)| n == name)
        .map(|(_, f)| (f.dimension().expect("a stored field is never empty"), f))
        .collect()
}

fn highest_instance<'a>(
    instances: &'a [(Dimension, mf::FieldViewD<'a>)],
) -> Option<&'a (Dimension, mf::FieldViewD<'a>)> {
    instances.iter().max_by_key(|(d, _)| u8::from(*d))
}

fn resolve_field<'a>(
    inner: &'a mf::UMesh,
    name: &str,
) -> PyResult<(Dimension, mf::FieldViewD<'a>)> {
    let instances = find_instances(inner, name);
    let best = instances
        .iter()
        .max_by_key(|(d, _)| u8::from(*d))
        .map(|(d, f)| (*d, f.clone()));
    best.ok_or_else(|| {
        PyKeyError::new_err(format!("no field named '{name}' available on this mesh"))
    })
}

fn target_dim(inner: &mf::UMesh, name: &str) -> PyResult<Dimension> {
    if let Some((d, _)) = highest_instance(&find_instances(inner, name)) {
        return Ok(*d);
    }
    inner.topological_dimension().ok_or_else(|| {
        PyValueError::new_err("cannot infer a target dimension: the mesh has no elements")
    })
}

fn etypes_at_dim(inner: &mf::UMesh, dim: Dimension) -> Vec<ElementType> {
    inner
        .element_types()
        .filter(|et| et.dimension() == dim)
        .copied()
        .collect()
}

fn broadcast_fill(comp: &nd::ArcArrayD<f64>, n_elem: usize) -> PyResult<nd::ArcArrayD<f64>> {
    let mut shape = vec![n_elem];
    shape.extend(comp.shape().iter().copied());
    let broadcast = comp
        .broadcast(shape)
        .ok_or_else(|| PyValueError::new_err("cannot broadcast the provided array"))?;
    Ok(broadcast.to_owned().into_shared())
}

fn expr_field_references<'a>(expr: &'a FieldExpr, out: &mut Vec<&'a str>) {
    match expr {
        FieldExpr::Field(name) => out.push(name),
        FieldExpr::BinaryExpr { left, right, .. } => {
            expr_field_references(left, out);
            expr_field_references(right, out);
        }
        FieldExpr::UnaryExpr { expr, .. } => expr_field_references(expr, out),
        FieldExpr::Index(inner, _) => expr_field_references(inner, out),
        _ => {}
    }
}

pub(crate) fn validate_expr_fields(
    m: &mf::UMesh,
    expr: &FieldExpr,
    dim: Dimension,
) -> PyResult<()> {
    let mut refs = Vec::new();
    expr_field_references(expr, &mut refs);
    for name in refs {
        if m.field(name, Some(dim)).is_none() {
            return Err(PyValueError::new_err(format!(
                "expression references field '{name}' which is not available on dimension {} elements",
                u8::from(dim)
            )));
        }
    }
    Ok(())
}

fn selection_field_references<'a>(sel: &'a mf::Selection, out: &mut Vec<&'a str>) {
    match sel {
        mf::Selection::FieldSelection(fs) => {
            let (a, b) = match fs {
                mf::FieldSelection::Gt(a, b)
                | mf::FieldSelection::Geq(a, b)
                | mf::FieldSelection::Lt(a, b)
                | mf::FieldSelection::Leq(a, b)
                | mf::FieldSelection::Eq(a, b)
                | mf::FieldSelection::Neq(a, b) => (a, b),
            };
            expr_field_references(a, out);
            expr_field_references(b, out);
        }
        mf::Selection::BinarayExpr(e) => {
            selection_field_references(&e.left, out);
            selection_field_references(&e.right, out);
        }
        mf::Selection::NotExpr(e) => selection_field_references(&e.0, out),
        _ => {}
    }
}

fn validate_selection_fields(m: &mf::UMesh, sel: &mf::Selection, dim: Dimension) -> PyResult<()> {
    let mut refs = Vec::new();
    selection_field_references(sel, &mut refs);
    for name in refs {
        if m.field(name, Some(dim)).is_none() {
            return Err(PyValueError::new_err(format!(
                "selection references field '{name}' which is not available on dimension {} elements",
                u8::from(dim)
            )));
        }
    }
    Ok(())
}

fn split_element_array(
    m: &mf::UMesh,
    dim: Dimension,
    arr: &nd::ArcArrayD<f64>,
) -> PyResult<BTreeMap<ElementType, nd::ArcArrayD<f64>>> {
    let mut map = BTreeMap::new();
    let mut offset = 0usize;
    for et in etypes_at_dim(m, dim) {
        let n = m.block(et).expect("block exists").len();
        map.insert(
            et,
            arr.slice_axis(nd::Axis(0), (offset..offset + n).into())
                .to_owned()
                .into_shared(),
        );
        offset += n;
    }
    Ok(map)
}

fn assign_broadcast(m: &mut mf::UMesh, name: &str, arr: &nd::ArcArrayD<f64>) -> PyResult<()> {
    let instances = find_instances(m, name);
    if let Some((dim, field)) = highest_instance(&instances) {
        let dim = *dim;
        let first = field
            .0
            .values()
            .next()
            .expect("a stored field is never empty");
        if first.ndim() < 1 {
            return Err(PyValueError::new_err(
                "cannot reassign a degenerate 0-dimensional field with an array",
            ));
        }
        let comp_shape = first.shape()[1..].to_vec();
        let is_component = arr.ndim() == 0
            || (arr.ndim() == comp_shape.len() && arr.shape().to_vec() == comp_shape);
        let is_element_array = !is_component
            && arr.ndim() == comp_shape.len() + 1
            && arr.shape()[1..] == comp_shape[..]
            && arr.shape()[0] == m.num_elements_of_dim(dim);
        if !is_component && !is_element_array {
            return Err(PyValueError::new_err(format!(
                "array of shape {:?} matches neither a component of shape {:?} nor an element array of shape (n, {:?})",
                arr.shape(),
                comp_shape,
                comp_shape,
            )));
        }
        let map = if is_component {
            let mut map = BTreeMap::new();
            for et in etypes_at_dim(m, dim) {
                let n = m.block(et).expect("block exists").len();
                map.insert(et, broadcast_fill(arr, n)?);
            }
            map
        } else {
            split_element_array(m, dim, arr)?
        };
        m.update_field(name, mf::FieldArcD::new(map));
        return Ok(());
    }
    let dim = target_dim(m, name)?;
    let total_n = m.num_elements_of_dim(dim);
    let as_element_array = arr.ndim() >= 1 && arr.shape()[0] == total_n;
    let map = if as_element_array {
        split_element_array(m, dim, arr)?
    } else {
        let mut map = BTreeMap::new();
        for et in etypes_at_dim(m, dim) {
            let n = m.block(et).expect("block exists").len();
            map.insert(et, broadcast_fill(arr, n)?);
        }
        map
    };
    m.update_field(name, mf::FieldArcD::new(map));
    Ok(())
}

fn assign_per_block(
    m: &mut mf::UMesh,
    name: &str,
    map: BTreeMap<ElementType, nd::ArcArrayD<f64>>,
) -> PyResult<()> {
    let mut dims: HashSet<u8> = HashSet::new();
    for et in map.keys() {
        dims.insert(u8::from(et.dimension()));
    }
    if dims.len() != 1 {
        return Err(PyValueError::new_err(
            "all element types in a per-block field must share the same dimension",
        ));
    }
    let dim = map.keys().next().expect("non-empty").dimension();
    let expected_types = etypes_at_dim(m, dim);
    for et in &expected_types {
        if !map.contains_key(et) {
            return Err(PyValueError::new_err(format!(
                "per-block assignment must cover all '{}' blocks at this dimension",
                etype_to_str(*et)
            )));
        }
    }
    let mut comp_tail: Option<Vec<usize>> = None;
    for (et, arr) in map.iter() {
        let n = m.block(*et).expect("block exists").len();
        if arr.ndim() < 1 || arr.shape()[0] != n {
            return Err(PyValueError::new_err(format!(
                "block '{}' expects an array with {} elements along its first axis, got shape {:?}",
                etype_to_str(*et),
                n,
                arr.shape()
            )));
        }
        let tail = arr.shape()[1..].to_vec();
        match &comp_tail {
            None => comp_tail = Some(tail),
            Some(prev) if prev != &tail => {
                return Err(PyValueError::new_err(
                    "all arrays in a per-block field must share the same trailing shape",
                ));
            }
            _ => {}
        }
    }
    m.update_field(name, mf::FieldArcD::new(map));
    Ok(())
}

fn assign_field(m: &mut mf::UMesh, name: &str, value: FieldValue) -> PyResult<()> {
    match value {
        FieldValue::Expr(expr) => {
            let mut refs = Vec::new();
            expr_field_references(&expr, &mut refs);
            for name in refs {
                let exists = m.blocks().any(|(_, b)| b.fields.contains_key(name));
                if !exists {
                    return Err(PyValueError::new_err(format!(
                        "expression references unknown field '{name}'"
                    )));
                }
            }
            m.eval_update_field(name, None, *expr);
            Ok(())
        }
        FieldValue::Broadcast(arr) => assign_broadcast(m, name, &arr),
        FieldValue::PerBlock(map) => assign_per_block(m, name, map),
    }
}

enum RowSource {
    Comp(nd::ArcArrayD<f64>),
    Rows(BTreeMap<ElementType, nd::ArrayD<f64>>),
}

type BlockSnapshot = Vec<(ElementType, nd::ArcArrayD<f64>)>;

fn fill_rows(dst: &mut nd::ArrayD<f64>, idxs: &[usize], comp: &nd::ArcArrayD<f64>) {
    for &i in idxs {
        let mut row = dst.index_axis_mut(nd::Axis(0), i);
        for (d, s) in row.iter_mut().zip(comp.iter()) {
            *d = *s;
        }
    }
}

fn write_rows(dst: &mut nd::ArrayD<f64>, idxs: &[usize], src: &nd::ArrayD<f64>) {
    for (j, &i) in idxs.iter().enumerate() {
        let mut row = dst.index_axis_mut(nd::Axis(0), i);
        let src_row = src.index_axis(nd::Axis(0), j);
        for (d, s) in row.iter_mut().zip(src_row.iter()) {
            *d = *s;
        }
    }
}

#[pyclass]
#[pyo3(name = "FieldsMapping")]
pub struct PyFieldsMapping {
    pub(crate) mesh: Py<PyUMesh>,
}

impl PyFieldsMapping {
    fn with_inner<R>(&self, f: impl FnOnce(&mf::UMesh) -> R) -> R {
        Python::attach(|py| f(&self.mesh.bind(py).borrow().inner))
    }

    fn sorted_names(&self) -> Vec<String> {
        self.with_inner(|m| {
            let mut names: Vec<String> = m.fields().map(|(n, _)| n).collect();
            names.sort_unstable();
            names.dedup();
            names
        })
    }

    fn make_ref(&self, name: String) -> PyFieldRef {
        PyFieldRef {
            mesh: Python::attach(|py| self.mesh.clone_ref(py)),
            name,
        }
    }
}

#[pymethods]
impl PyFieldsMapping {
    pub fn keys(&self) -> Vec<String> {
        self.sorted_names()
    }

    pub fn values(&self) -> Vec<PyFieldRef> {
        self.sorted_names()
            .into_iter()
            .map(|name| self.make_ref(name))
            .collect()
    }

    #[pyo3(name = "items")]
    pub fn items_pairs(&self) -> Vec<(String, PyFieldRef)> {
        self.sorted_names()
            .into_iter()
            .map(|name| {
                let r = self.make_ref(name.clone());
                (name, r)
            })
            .collect()
    }

    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for name in self.sorted_names() {
            let r = self.make_ref(name.clone());
            dict.set_item(name, r.values(py)?)?;
        }
        Ok(dict)
    }

    pub fn __len__(&self) -> usize {
        self.sorted_names().len()
    }

    pub fn __contains__(&self, name: &str) -> bool {
        self.with_inner(|m| !find_instances(m, name).is_empty())
    }

    pub fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Bound<'_, PyIterator>> {
        let list = PyList::new(slf.py(), slf.keys())?;
        list.as_any().try_iter()
    }

    pub fn __getitem__(&self, name: &str) -> PyResult<PyFieldRef> {
        let exists = self.with_inner(|m| !find_instances(m, name).is_empty());
        if !exists {
            return Err(PyKeyError::new_err(name.to_string()));
        }
        Ok(self.make_ref(name.to_string()))
    }

    pub fn __setitem__(&self, name: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = extract_field_value(value)?;
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            assign_field(&mut guard.inner, name, value)
        })
    }

    pub fn rename(&self, old_name: &str, new_name: &str) -> PyResult<()> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;
            let dims: Vec<Dimension> = find_instances(inner, old_name)
                .iter()
                .map(|(d, _)| *d)
                .collect();
            if dims.is_empty() {
                return Err(PyKeyError::new_err(old_name.to_string()));
            }
            if !find_instances(inner, new_name).is_empty() {
                return Err(PyValueError::new_err(format!(
                    "field '{new_name}' already exists"
                )));
            }
            for dim in dims {
                let field = inner
                    .remove_field(old_name, Some(dim))
                    .expect("field instances were just resolved");
                inner.update_field(new_name, field);
            }
            Ok(())
        })
    }

    pub fn __delitem__(&self, name: &str) -> PyResult<()> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let dims: Vec<Dimension> = find_instances(&guard.inner, name)
                .iter()
                .map(|(d, _)| *d)
                .collect();
            if dims.is_empty() {
                return Err(PyKeyError::new_err(name.to_string()));
            }
            for dim in dims {
                guard.inner.remove_field(name, Some(dim));
            }
            Ok(())
        })
    }

    pub fn __repr__(&self) -> String {
        format!("FieldsMapping({:?})", self.sorted_names())
    }
}

#[pyclass]
#[pyo3(name = "FieldRef")]
pub struct PyFieldRef {
    pub(crate) mesh: Py<PyUMesh>,
    pub(crate) name: String,
}

impl PyFieldRef {
    fn with_resolved<'py, R>(
        &self,
        py: Python<'py>,
        f: impl FnOnce(Python<'py>, &mf::UMesh, Dimension, &mf::FieldViewD<'_>) -> PyResult<R>,
    ) -> PyResult<R> {
        let mesh = self.mesh.bind(py);
        let inner = &mesh.borrow().inner;
        let (dim, field) = resolve_field(inner, &self.name)?;
        f(py, inner, dim, &field)
    }
}

#[pymethods]
impl PyFieldRef {
    #[getter]
    pub fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        self.with_resolved(py, |py, _m, _dim, field| {
            let first = field
                .0
                .values()
                .next()
                .expect("a stored field is never empty");
            let comps: Vec<usize> = if first.ndim() <= 1 {
                vec![1]
            } else {
                first.shape()[1..].to_vec()
            };
            PyTuple::new(py, comps)
        })
    }

    pub fn dimension(&self) -> PyResult<u8> {
        Python::attach(|py| self.with_resolved(py, |_py, _m, dim, _field| Ok(u8::from(dim))))
    }

    pub fn len(&self) -> PyResult<usize> {
        self.__len__()
    }

    pub fn __len__(&self) -> PyResult<usize> {
        Python::attach(|py| {
            self.with_resolved(py, |_py, m, dim, _field| Ok(m.num_elements_of_dim(dim)))
        })
    }

    pub fn numpy<'py>(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        self.with_resolved(py, |py, _m, _dim, field| {
            let views: Vec<nd::ArrayViewD<'_, f64>> = field.0.values().map(|a| a.view()).collect();
            if views.len() == 1 {
                return Ok(np::PyArray::from_array(py, &views[0]).unbind().into_any());
            }
            let stacked = nd::concatenate(nd::Axis(0), &views)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            Ok(np::PyArray::from_owned_array(py, stacked)
                .unbind()
                .into_any())
        })
    }

    pub fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.with_resolved(py, |py, _m, _dim, field| {
            let dict = PyDict::new(py);
            for (et, arr) in field.0.iter() {
                dict.set_item(etype_to_str(*et), np::PyArray::from_array(py, arr))?;
            }
            Ok(dict)
        })
    }

    pub fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        sel: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let selector = extract_selector(sel)?;
        self.with_resolved(py, |py, m, dim, field| {
            let eids = match selector {
                Selector::Expr(expr) => m.select_ids(expr, Some(dim)),
                Selector::Ids(ids) => ids,
            };
            let dict = PyDict::new(py);
            for (et, arr) in field.0.iter() {
                let Some(idxs) = eids.get(et) else { continue };
                if idxs.is_empty() {
                    continue;
                }
                let gathered = arr.select(nd::Axis(0), idxs);
                dict.set_item(
                    etype_to_str(*et),
                    np::PyArray::from_owned_array(py, gathered),
                )?;
            }
            Ok(dict)
        })
    }

    pub fn __setitem__(&self, sel: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let selector = extract_selector(sel)?;
        let value = extract_field_value(value)?;
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let mut guard = mesh.borrow_mut();
            let inner = &mut guard.inner;

            let (dim, comp_shape, snapshot): (Dimension, Vec<usize>, BlockSnapshot) = {
                let (resolved_dim, field) = resolve_field(inner, &self.name)?;
                let first = field
                    .0
                    .values()
                    .next()
                    .expect("a stored field is never empty");
                if first.ndim() < 1 {
                    return Err(PyValueError::new_err(
                        "cannot partially rewrite a degenerate 0-dimensional field",
                    ));
                }
                let snap: BlockSnapshot =
                    field.0.iter().map(|(et, a)| (*et, a.to_shared())).collect();
                (resolved_dim, first.shape()[1..].to_vec(), snap)
            };

            let eids = match selector {
                Selector::Expr(expr) => {
                    validate_selection_fields(inner, &expr, dim)?;
                    inner.select_ids(expr, Some(dim))
                }
                Selector::Ids(ids) => ids,
            };

            let mut targets: BTreeMap<ElementType, Vec<usize>> = BTreeMap::new();
            let mut total_selected = 0usize;
            for (et, _) in snapshot.iter() {
                if let Some(idxs) = eids.get(et).filter(|idxs| !idxs.is_empty()) {
                    total_selected += idxs.len();
                    targets.insert(*et, idxs.clone());
                }
            }
            if targets.is_empty() {
                return Err(PyValueError::new_err(
                    "the selection does not intersect any element carrying this field",
                ));
            }

            let source = match value {
                FieldValue::Expr(expr) => {
                    validate_expr_fields(inner, &expr, dim)?;
                    let evaluated = inner.eval_field(Some(dim), *expr);
                    let mut rows = BTreeMap::new();
                    for (et, idxs) in targets.iter() {
                        let ev = evaluated.0.get(et).ok_or_else(|| {
                            PyValueError::new_err(format!(
                                "expression did not produce values for block '{}'",
                                etype_to_str(*et)
                            ))
                        })?;
                        rows.insert(*et, ev.select(nd::Axis(0), idxs));
                    }
                    RowSource::Rows(rows)
                }
                FieldValue::PerBlock(map) => {
                    for (et, arr) in map.iter() {
                        let Some(idxs) = targets.get(et) else {
                            return Err(PyValueError::new_err(format!(
                                "block '{}' does not intersect the selection",
                                etype_to_str(*et)
                            )));
                        };
                        if arr.shape()[0] != idxs.len() {
                            return Err(PyValueError::new_err(format!(
                                "block '{}' expects {} selected values, got {}",
                                etype_to_str(*et),
                                idxs.len(),
                                arr.shape()[0]
                            )));
                        }
                    }
                    let rows = map
                        .into_iter()
                        .filter(|(et, _)| targets.contains_key(et))
                        .map(|(et, arr)| (et, arr.into_owned()))
                        .collect();
                    RowSource::Rows(rows)
                }
                FieldValue::Broadcast(arr) => {
                    let is_component = arr.ndim() == 0
                        || (arr.ndim() == comp_shape.len() && arr.shape().to_vec() == comp_shape);
                    if is_component {
                        RowSource::Comp(arr)
                    } else if arr.ndim() == comp_shape.len() + 1
                        && arr.shape()[1..] == comp_shape[..]
                        && targets.len() == 1
                        && arr.shape()[0] == total_selected
                    {
                        let mut rows = BTreeMap::new();
                        for et in targets.keys() {
                            rows.insert(*et, arr.to_owned());
                        }
                        RowSource::Rows(rows)
                    } else {
                        return Err(PyValueError::new_err(format!(
                            "array of shape {:?} cannot be written onto components of shape {:?}; provide a dict {{\"ETYPE\": array}} of shape (n_selected, {:?}) for multi-block selections",
                            arr.shape(),
                            comp_shape,
                            comp_shape
                        )));
                    }
                }
            };

            let mut new_map: BTreeMap<ElementType, nd::ArcArrayD<f64>> = BTreeMap::new();
            for (et, base) in snapshot.iter() {
                if let Some(idxs) = targets.get(et) {
                    let mut owned = base.to_owned();
                    match &source {
                        RowSource::Comp(comp) => fill_rows(&mut owned, idxs, comp),
                        RowSource::Rows(rows) => {
                            write_rows(
                                &mut owned,
                                idxs,
                                rows.get(et).expect("rows validated per target"),
                            );
                        }
                    }
                    new_map.insert(*et, owned.into_shared());
                } else {
                    new_map.insert(*et, base.clone());
                }
            }
            inner.update_field(&self.name, mf::FieldArcD::new(new_map));
            Ok(())
        })
    }

    pub fn min(&self) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Min, 0)
    }

    pub fn max(&self) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Max, 0)
    }

    pub fn sum(&self) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Sum, 0)
    }

    pub fn mean(&self) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Mean, 0)
    }

    #[pyo3(signature = (ddof=0))]
    pub fn var(&self, ddof: usize) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Var, ddof)
    }

    #[pyo3(signature = (ddof=0))]
    pub fn std(&self, ddof: usize) -> PyResult<Py<PyAny>> {
        self.reduce_pub(ReduceOp::Std, ddof)
    }

    pub fn integral(&self) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let (_, field) = resolve_field(inner, &self.name)?;
            let dim = field.dimension().expect("a stored field is never empty");
            let measures = mf::measure(&inner.view(), Some(dim));
            let mut products: Vec<nd::ArrayD<f64>> = Vec::new();
            for (et, arr) in field.0.iter() {
                let meas = measures.get(et).ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "missing measure for element type {}",
                        etype_to_str(*et)
                    ))
                })?;
                products.push(measure_weighted(meas, arr.view())?);
            }
            let views: Vec<nd::ArrayViewD<'_, f64>> = products.iter().map(|p| p.view()).collect();
            let reduced = reduce_blocks(&views, ReduceOp::Sum, 0)?;
            Ok(reduce_to_py(py, reduced))
        })
    }

    pub fn __repr__(&self) -> String {
        format!("FieldRef({:?})", self.name)
    }
}

impl PyFieldRef {
    fn reduce_pub(&self, op: ReduceOp, ddof: usize) -> PyResult<Py<PyAny>> {
        Python::attach(|py| {
            let mesh = self.mesh.bind(py);
            let inner = &mesh.borrow().inner;
            let (_, field) = resolve_field(inner, &self.name)?;
            let views: Vec<nd::ArrayViewD<'_, f64>> = field.0.values().map(|a| a.view()).collect();
            let reduced = reduce_blocks(&views, op, ddof)?;
            Ok(reduce_to_py(py, reduced))
        })
    }
}
