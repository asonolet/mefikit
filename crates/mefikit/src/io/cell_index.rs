//! Canonical cell numbering used by the unstructured IO writers.
//!
//! Formats that store one flat array of cell data (VTKHDF `CellData`/`GroupData`,
//! CGNS `PointList`) need a deterministic way to map between a flat "global cell
//! index" and the `(ElementType, local index)` pair used by
//! [`UMesh`](crate::mesh::UMesh). [`BlockLayout`] provides exactly that: it walks
//! the element blocks in `mesh.blocks()` order (sorted by `ElementType`) and, in
//! each block, in local index order. This matches the order in which mefikit
//! writes VTKHDF cells and CGNS `ElementRange`s.
//!
//! Readers must **not** use this layout: files produced by other tools may number
//! their cells (or CGNS elements) differently. They build an explicit
//! file-index-to-`(ElementType, local)` mapping while parsing the connectivity.

use super::error::MefikitIOError;
use crate::mesh::{Dimension, ElementIds, ElementType, FieldBase, UMesh, UMeshBase};
use ndarray as nd;
use std::collections::BTreeMap;

struct BlockEntry {
    element_type: ElementType,
    start: usize,
    len: usize,
}

/// A deterministic, format-independent numbering of every cell of a mesh.
pub struct BlockLayout {
    entries: Vec<BlockEntry>,
}

impl BlockLayout {
    /// Builds the canonical layout of `mesh`.
    pub fn new<N, C, F, G>(mesh: &UMeshBase<N, C, F, G>) -> Self
    where
        N: nd::Data<Elem = f64>,
        C: nd::Data<Elem = usize>,
        F: nd::Data<Elem = f64>,
        G: nd::Data<Elem = usize>,
    {
        let mut entries = Vec::new();
        let mut start = 0;
        for (element_type, block) in mesh.blocks() {
            let len = block.len();
            entries.push(BlockEntry {
                element_type: *element_type,
                start,
                len,
            });
            start += len;
        }
        Self { entries }
    }

    /// Iterates over `(element_type, start, len)` in canonical order.
    pub fn entries(&self) -> impl Iterator<Item = (ElementType, usize, usize)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.element_type, entry.start, entry.len))
    }

    fn entry(&self, element_type: ElementType) -> Option<&BlockEntry> {
        self.entries
            .iter()
            .find(|entry| entry.element_type == element_type)
    }

    /// Converts a per-block local index into a global cell index.
    pub fn type_local_to_global(&self, element_type: ElementType, local: usize) -> Option<usize> {
        let entry = self.entry(element_type)?;
        (local < entry.len).then_some(entry.start + local)
    }

    /// Global (0-based) cell indices of every element belonging to `group`, in
    /// canonical order.
    pub fn group_global_ids<N, C, F, G>(
        &self,
        mesh: &UMeshBase<N, C, F, G>,
        group: &str,
    ) -> Vec<i64>
    where
        N: nd::Data<Elem = f64>,
        C: nd::Data<Elem = usize>,
        F: nd::Data<Elem = f64>,
        G: nd::Data<Elem = usize>,
    {
        let mut ids = Vec::new();
        for (element_type, block) in mesh.blocks() {
            for local in block.group_elements_local(group) {
                if let Some(global) = self.type_local_to_global(*element_type, local) {
                    ids.push(global as i64);
                }
            }
        }
        ids
    }
}

/// Rebuilds an [`ElementIds`] map from an explicit list of
/// `(element type, local index)` pairs.
pub fn element_ids_from_type_local_rows(
    rows: impl IntoIterator<Item = (ElementType, usize)>,
) -> ElementIds {
    let mut per_type: BTreeMap<ElementType, Vec<usize>> = BTreeMap::new();
    for (element_type, local) in rows {
        per_type.entry(element_type).or_default().push(local);
    }
    let mut ids = ElementIds::new();
    for (element_type, indices) in per_type {
        ids.add_block(element_type, indices);
    }
    ids
}

/// Flattens the cell-centered field `name` at dimension `dim` into a single
/// array of shape `(n_cells_at_dim, *components)`, in canonical cell order.
///
/// Returns `None` if the field is not defined at `dim`.
pub fn flatten_field_for_dim<N, C, F, G>(
    mesh: &UMeshBase<N, C, F, G>,
    name: &str,
    dim: Dimension,
) -> Option<nd::ArrayD<f64>>
where
    N: nd::Data<Elem = f64>,
    C: nd::Data<Elem = usize>,
    F: nd::Data<Elem = f64>,
    G: nd::Data<Elem = usize>,
{
    let field = mesh.field(name, Some(dim))?;
    let first = field.0.values().next()?;
    if first.ndim() == 0 {
        return None;
    }
    let trailing: Vec<usize> = first.shape()[1..].to_vec();
    let total: usize = field.0.values().map(|array| array.shape()[0]).sum();

    let mut shape = vec![total];
    shape.extend_from_slice(&trailing);
    let mut flat = nd::ArrayD::<f64>::zeros(nd::IxDyn(&shape));

    let mut offset = 0;
    for (element_type, _) in mesh.blocks().filter(|(et, _)| et.dimension() == dim) {
        let array = &field.0[element_type];
        let len = array.shape()[0];
        if trailing.is_empty() {
            flat.slice_mut(nd::s![offset..offset + len]).assign(array);
        } else {
            // `flat` has extra component axes: copy one element at a time so the
            // trailing dimensions are preserved.
            for i in 0..len {
                flat.index_axis_mut(nd::Axis(0), offset + i)
                    .assign(&array.index_axis(nd::Axis(0), i));
            }
        }
        offset += len;
    }
    Some(flat)
}

/// Rebuilds a cell-centered field at dimension `dim` from a flat array whose
/// rows are aligned with `rows` (`(element type, local index)` pairs).
///
/// Every element of `dim` must be covered by `rows`; otherwise the field is
/// only partially defined and an error is returned.
pub fn field_from_type_local_rows(
    mesh: &UMesh,
    dim: Dimension,
    data: &nd::ArrayViewD<f64>,
    rows: &[(ElementType, usize)],
) -> Result<FieldBase<nd::OwnedArcRepr<f64>, nd::IxDyn>, MefikitIOError> {
    if data.ndim() == 0 {
        return Err(MefikitIOError::MalformedFile(
            "field data has no element dimension".to_string(),
        ));
    }
    if data.shape()[0] != rows.len() {
        return Err(MefikitIOError::MalformedFile(format!(
            "field has {} rows but {} cell indices",
            data.shape()[0],
            rows.len()
        )));
    }

    let trailing: Vec<usize> = data.shape()[1..].to_vec();
    let mut values: BTreeMap<ElementType, nd::ArrayD<f64>> = BTreeMap::new();
    let mut covered: BTreeMap<ElementType, usize> = BTreeMap::new();

    for element_type in mesh.element_types().filter(|et| et.dimension() == dim) {
        let len = mesh.block(*element_type).unwrap().len();
        let mut shape = vec![len];
        shape.extend_from_slice(&trailing);
        values.insert(
            *element_type,
            nd::ArrayD::from_elem(nd::IxDyn(&shape), f64::NAN),
        );
        covered.insert(*element_type, 0);
    }

    if values.is_empty() {
        return Err(MefikitIOError::MalformedFile(format!(
            "no elements at dimension {dim:?} to attach the field to"
        )));
    }

    for (row, &(element_type, local)) in rows.iter().enumerate() {
        let Some(array) = values.get_mut(&element_type) else {
            return Err(MefikitIOError::MalformedFile(format!(
                "field row {row} refers to {element_type:?}, which is not at {dim:?}"
            )));
        };
        if local >= array.shape()[0] {
            return Err(MefikitIOError::MalformedFile(format!(
                "field row {row} refers to {element_type:?} element {local}, which is out of range"
            )));
        }
        array
            .index_axis_mut(nd::Axis(0), local)
            .assign(&data.index_axis(nd::Axis(0), row));
        *covered.get_mut(&element_type).unwrap() += 1;
    }

    for (element_type, array) in &values {
        if covered[element_type] != array.shape()[0] {
            return Err(MefikitIOError::MalformedFile(format!(
                "field covers only {} of {} {element_type:?} elements",
                covered[element_type],
                array.shape()[0]
            )));
        }
    }

    let shared: BTreeMap<ElementType, nd::ArrayBase<nd::OwnedArcRepr<f64>, nd::IxDyn>> = values
        .into_iter()
        .map(|(element_type, array)| (element_type, array.into_shared()))
        .collect();
    Ok(FieldBase::new(shared))
}

/// Returns the global (0-based) cell indices of every element at `dim`, in
/// canonical order.
pub fn global_ids_of_dim(layout: &BlockLayout, dim: Dimension) -> Vec<usize> {
    let mut ids = Vec::new();
    for (element_type, start, len) in layout.entries() {
        if element_type.dimension() == dim {
            ids.extend(start..start + len);
        }
    }
    ids
}
