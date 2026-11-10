//! Save and load a [`TransferOperator`] as a SciPy-compatible NPZ archive.
//!
//! The file is a regular NumPy `.npz` archive: a zip containing `.npy` arrays. The CSR matrix
//! entries use the same names and dtypes as `scipy.sparse.save_npz`, so the raw matrix can be
//! opened with `scipy.sparse.load_npz`. Extra arrays carry the mefikit metadata and are ignored
//! by SciPy.
//!
//! # Block ordering
//!
//! A transfer operator is a block matrix: rows are target cells grouped by element type and
//! columns are source cells grouped by element type. The file stores the block descriptors with
//! **explicit** row/column offsets; it never infers them from the [`ElementType`] declaration
//! order. On load the source columns are remapped to the current in-memory element-type order, so
//! inserting a new element type in the middle of the enum does not invalidate existing files.
//!
//! Blocks are written in name-sorted order so the file layout itself is independent of the enum
//! order; the offsets remain authoritative for arbitrary files.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use ndarray as nd;
use npyz::WriterBuilder;
use serde::{Deserialize, Serialize};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use super::operator::{RowSparse, TransferMethod, TransferOperator};
use super::transfer_trait::PointLocation;
use crate::io::MefikitIOError;
use crate::mesh::{Dimension, ElementType, UMeshView};
use crate::tools::meshless::DistanceWeighting;

const MAGIC: &str = "mefikit-transfer";
const VERSION: u32 = 1;

const FORMAT: &str = "format.npy";
const SHAPE: &str = "shape.npy";
const DATA: &str = "data.npy";
const INDICES: &str = "indices.npy";
const INDPTR: &str = "indptr.npy";
const TARGET_MEASURE: &str = "target_measure.npy";
const META: &str = "mefikit_meta.npy";
const DEFAULT_VALUE: &str = "mefikit_default.npy";

/// Options for [`TransferOperator::save_with`].
#[derive(Clone, Debug)]
pub struct TransferSaveOptions {
    /// Deflate-compress the archive entries.
    pub compressed: bool,
    /// Optional default value restored by [`TransferOperator::load`].
    ///
    /// This is a Python-wrapper convenience: it is stored in the file but is not used by the
    /// Rust operator itself.
    pub default_value: Option<f64>,
}

impl Default for TransferSaveOptions {
    fn default() -> Self {
        Self {
            compressed: true,
            default_value: None,
        }
    }
}

/// Metadata stored in the `mefikit_meta` array.
#[derive(Debug, Serialize, Deserialize)]
struct TransferMetadata {
    magic: String,
    version: u32,
    method: MethodMetadata,
    src_dim: u8,
    tgt_dim: u8,
    source_blocks: Vec<SourceBlock>,
    target_blocks: Vec<TargetBlock>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceBlock {
    element_type: ElementType,
    count: u64,
    column_offset: u64,
}

#[derive(Debug, Serialize, Deserialize)]
struct TargetBlock {
    element_type: ElementType,
    count: u64,
    row_offset: u64,
}

/// Stable, file-level representation of a transfer method.
///
/// This deliberately does not serialize the Rust enums directly: the on-disk names are part of
/// the file format and must stay stable even if the internal enums are reorganised.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MethodMetadata {
    ConstantPiecewise {
        point_location: PointLocationMetadata,
    },
    ConservativeP0,
    InverseDistance {
        k: usize,
        exponent: f64,
    },
    MovingLeastSquares {
        k: usize,
        weighting: WeightingMetadata,
    },
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum PointLocationMetadata {
    Centroid,
    Barycenter,
    StrictInterior,
}

#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum WeightingMetadata {
    Constant,
    InverseDistance { exponent: f64 },
    CompactSupport { exponent: f64 },
    Gaussian,
}

impl From<PointLocation> for PointLocationMetadata {
    fn from(value: PointLocation) -> Self {
        match value {
            PointLocation::Centroid => Self::Centroid,
            PointLocation::Barycenter => Self::Barycenter,
            PointLocation::StrictInterior => Self::StrictInterior,
        }
    }
}

impl From<PointLocationMetadata> for PointLocation {
    fn from(value: PointLocationMetadata) -> Self {
        match value {
            PointLocationMetadata::Centroid => Self::Centroid,
            PointLocationMetadata::Barycenter => Self::Barycenter,
            PointLocationMetadata::StrictInterior => Self::StrictInterior,
        }
    }
}

impl From<DistanceWeighting> for WeightingMetadata {
    fn from(value: DistanceWeighting) -> Self {
        match value {
            DistanceWeighting::Constant => Self::Constant,
            DistanceWeighting::InverseDistance { exponent } => Self::InverseDistance { exponent },
            DistanceWeighting::CompactSupport { exponent } => Self::CompactSupport { exponent },
            DistanceWeighting::Gaussian => Self::Gaussian,
        }
    }
}

impl TryFrom<WeightingMetadata> for DistanceWeighting {
    type Error = String;

    fn try_from(value: WeightingMetadata) -> Result<Self, Self::Error> {
        let valid_exponent = |exponent: f64| {
            if exponent.is_finite() && exponent > 0.0 {
                Ok(exponent)
            } else {
                Err(format!(
                    "kernel exponent should be positive, got {exponent}"
                ))
            }
        };
        match value {
            WeightingMetadata::Constant => Ok(Self::Constant),
            WeightingMetadata::InverseDistance { exponent } => Ok(Self::InverseDistance {
                exponent: valid_exponent(exponent)?,
            }),
            WeightingMetadata::CompactSupport { exponent } => Ok(Self::CompactSupport {
                exponent: valid_exponent(exponent)?,
            }),
            WeightingMetadata::Gaussian => Ok(Self::Gaussian),
        }
    }
}

impl From<TransferMethod> for MethodMetadata {
    fn from(value: TransferMethod) -> Self {
        match value {
            TransferMethod::ConstantPiecewise { point_location } => Self::ConstantPiecewise {
                point_location: point_location.into(),
            },
            TransferMethod::ConservativeP0 => Self::ConservativeP0,
            TransferMethod::InverseDistance { k, exponent } => {
                Self::InverseDistance { k, exponent }
            }
            TransferMethod::MovingLeastSquares { k, weighting } => Self::MovingLeastSquares {
                k,
                weighting: weighting.into(),
            },
        }
    }
}

impl TryFrom<MethodMetadata> for TransferMethod {
    type Error = String;

    fn try_from(value: MethodMetadata) -> Result<Self, Self::Error> {
        match value {
            MethodMetadata::ConstantPiecewise { point_location } => Ok(Self::ConstantPiecewise {
                point_location: point_location.into(),
            }),
            MethodMetadata::ConservativeP0 => Ok(Self::ConservativeP0),
            MethodMetadata::InverseDistance { k, exponent } => {
                if k == 0 {
                    return Err("inverse-distance k should be at least 1".to_string());
                }
                if !(exponent.is_finite() && exponent > 0.0) {
                    return Err(format!(
                        "inverse-distance exponent should be positive, got {exponent}"
                    ));
                }
                Ok(Self::InverseDistance { k, exponent })
            }
            MethodMetadata::MovingLeastSquares { k, weighting } => {
                if k == 0 {
                    return Err("moving least-squares k should be at least 1".to_string());
                }
                Ok(Self::MovingLeastSquares {
                    k,
                    weighting: weighting.try_into()?,
                })
            }
        }
    }
}

impl TransferOperator {
    /// Saves the operator to a SciPy-compatible `.npz` file.
    ///
    /// Uses deflate compression and does not store a default value. See
    /// [`TransferOperator::save_with`] for the Python-facing options.
    pub fn save(&self, path: &Path) -> Result<(), MefikitIOError> {
        self.save_with(path, &TransferSaveOptions::default())
    }

    /// Saves the operator with explicit options.
    pub fn save_with(
        &self,
        path: &Path,
        options: &TransferSaveOptions,
    ) -> Result<(), MefikitIOError> {
        self.validate_structure()?;

        let n_src = self.n_src();
        let n_tgt = self.n_tgt();
        let nnz: usize = self.data().iter().map(|(_, rows)| rows.weights.len()).sum();

        let mut source_blocks = self.source_blocks().to_vec();
        source_blocks.sort_by_key(|(et, _)| et.name());

        let mut target_blocks: Vec<(&ElementType, &RowSparse)> =
            self.data().iter().map(|(et, rows)| (et, rows)).collect();
        target_blocks.sort_by_key(|(et, _)| et.name());

        // File column order is name-sorted; build the map from the current in-memory source
        // order to the file order.
        let mut file_source_offsets = BTreeMap::new();
        let mut column_offset = 0usize;
        for (et, count) in &source_blocks {
            file_source_offsets.insert(*et, column_offset);
            column_offset += *count;
        }
        if column_offset != n_src {
            return Err(invalid("source block counts do not sum to n_src"));
        }

        let mut current_source_offsets = BTreeMap::new();
        let mut running = 0usize;
        for (et, count) in self.source_blocks() {
            current_source_offsets.insert(*et, running);
            running += *count;
        }
        let mut source_index_map = vec![0usize; n_src];
        for (et, count) in self.source_blocks() {
            let from = current_source_offsets[et];
            let to = file_source_offsets[et];
            for local in 0..*count {
                source_index_map[from + local] = to + local;
            }
        }

        // File row order is name-sorted; build the global CSR indptr in that order.
        let mut indptr = Vec::with_capacity(n_tgt + 1);
        indptr.push(0usize);
        let mut nnz_offset = 0usize;
        for (_, rows) in &target_blocks {
            let base = nnz_offset;
            for &p in &rows.row_ptr[1..] {
                indptr.push(base + p);
            }
            nnz_offset += rows.src_idx.len();
        }
        if nnz_offset != nnz {
            return Err(invalid("target row pointers do not sum to nnz"));
        }

        let i64_from_usize = |value: usize, what: &str| {
            i64::try_from(value)
                .map_err(|_| invalid(format!("{what} ({value}) does not fit in int64")))
        };
        let n_src_i64 = i64_from_usize(n_src, "n_src")?;
        let n_tgt_i64 = i64_from_usize(n_tgt, "n_tgt")?;

        let use_i32 = n_src <= i32::MAX as usize && nnz <= i32::MAX as usize;

        let metadata = TransferMetadata {
            magic: MAGIC.to_string(),
            version: VERSION,
            method: self.method().into(),
            src_dim: u8::from(self.src_dim()),
            tgt_dim: u8::from(self.tgt_dim()),
            source_blocks: source_blocks
                .iter()
                .map(|(et, count)| SourceBlock {
                    element_type: *et,
                    count: u64::try_from(*count).expect("usize fits in u64"),
                    column_offset: u64::try_from(file_source_offsets[et])
                        .expect("usize fits in u64"),
                })
                .collect(),
            target_blocks: target_blocks
                .iter()
                .scan(0u64, |offset, (et, rows)| {
                    let count =
                        u64::try_from(rows.target_measure.len()).expect("usize fits in u64");
                    let block = TargetBlock {
                        element_type: **et,
                        count,
                        row_offset: *offset,
                    };
                    *offset += count;
                    Some(block)
                })
                .collect(),
        };
        let metadata =
            serde_json::to_vec(&metadata).map_err(|e| MefikitIOError::Encode(e.to_string()))?;

        let file = File::create(path)?;
        let mut zip = ZipWriter::new(BufWriter::new(file));
        let zip_options = file_options(options.compressed);

        write_format(&mut zip, zip_options)?;
        write_npy(&mut zip, SHAPE, zip_options, &[2], [n_tgt_i64, n_src_i64])?;
        write_npy(
            &mut zip,
            DATA,
            zip_options,
            &[nnz as u64],
            target_blocks
                .iter()
                .flat_map(|(_, rows)| rows.weights.iter().copied()),
        )?;
        if use_i32 {
            write_npy(
                &mut zip,
                INDICES,
                zip_options,
                &[nnz as u64],
                target_blocks.iter().flat_map(|(_, rows)| {
                    rows.src_idx.iter().map(|&idx| {
                        i32::try_from(source_index_map[idx]).expect("checked to fit in i32")
                    })
                }),
            )?;
            write_npy(
                &mut zip,
                INDPTR,
                zip_options,
                &[(n_tgt + 1) as u64],
                indptr
                    .iter()
                    .map(|&p| i32::try_from(p).expect("checked to fit in i32")),
            )?;
        } else {
            write_npy(
                &mut zip,
                INDICES,
                zip_options,
                &[nnz as u64],
                target_blocks.iter().flat_map(|(_, rows)| {
                    rows.src_idx.iter().map(|&idx| {
                        i64::try_from(source_index_map[idx]).expect("checked to fit in i64")
                    })
                }),
            )?;
            write_npy(
                &mut zip,
                INDPTR,
                zip_options,
                &[(n_tgt + 1) as u64],
                indptr
                    .iter()
                    .map(|&p| i64::try_from(p).expect("checked to fit in i64")),
            )?;
        }
        write_npy(
            &mut zip,
            TARGET_MEASURE,
            zip_options,
            &[n_tgt as u64],
            target_blocks
                .iter()
                .flat_map(|(_, rows)| rows.target_measure.iter().copied()),
        )?;
        write_npy(
            &mut zip,
            META,
            zip_options,
            &[metadata.len() as u64],
            metadata.iter().copied(),
        )?;
        if let Some(default) = options.default_value {
            write_npy(&mut zip, DEFAULT_VALUE, zip_options, &[1], [default])?;
        }

        zip.finish()
            .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
        Ok(())
    }

    /// Loads an operator from a `.npz` file.
    ///
    /// Returns the operator and the optional default value stored in the file. The file is
    /// validated structurally before being returned; use [`TransferOperator::validate`] to also
    /// check it against the meshes it will be applied to.
    pub fn load(path: &Path) -> Result<(Self, Option<f64>), MefikitIOError> {
        let file = File::open(path)?;
        let mut zip =
            ZipArchive::new(BufReader::new(file)).map_err(|e| malformed(format!("{e}")))?;

        let meta_bytes = read_array::<u8, _>(&mut zip, META, None)?;
        let metadata: TransferMetadata = serde_json::from_slice(&meta_bytes)
            .map_err(|e| MefikitIOError::Parse(format!("invalid transfer metadata: {e}")))?;
        if metadata.magic != MAGIC {
            return Err(invalid(format!(
                "not a mefikit transfer file (magic is {:?})",
                metadata.magic
            )));
        }
        if metadata.version != VERSION {
            return Err(invalid(format!(
                "unsupported transfer file version {} (expected {VERSION})",
                metadata.version
            )));
        }

        let format: Vec<Vec<u8>> = read_array(&mut zip, FORMAT, Some(&[]))?;
        if format.len() != 1 || format[0].as_slice() != b"csr" {
            return Err(invalid("transfer file is not a CSR matrix"));
        }

        let shape: Vec<i64> = read_array(&mut zip, SHAPE, Some(&[2]))?;
        let n_tgt = usize_from_i64(shape[0], "n_tgt")?;
        let n_src = usize_from_i64(shape[1], "n_src")?;

        let src_dim = Dimension::try_from(metadata.src_dim)
            .map_err(|e| invalid(format!("invalid source dimension: {e}")))?;
        let tgt_dim = Dimension::try_from(metadata.tgt_dim)
            .map_err(|e| invalid(format!("invalid target dimension: {e}")))?;
        let method = TransferMethod::try_from(metadata.method)
            .map_err(|e| invalid(format!("invalid transfer method: {e}")))?;
        if matches!(method, TransferMethod::ConservativeP0) && src_dim < tgt_dim {
            return Err(invalid(format!(
                "conservative P0 cannot transfer from {src_dim:?} to {tgt_dim:?}"
            )));
        }

        let file_source = checked_source_blocks(&metadata.source_blocks, n_src, src_dim)?;
        let file_target = checked_target_blocks(&metadata.target_blocks, n_tgt, tgt_dim, method)?;

        let indptr = read_indices(&mut zip, INDPTR, n_tgt + 1)?;
        validate_indptr(&indptr, n_tgt)?;
        let nnz = indptr[n_tgt];
        let mut indices = read_indices(&mut zip, INDICES, nnz)?;
        let data: Vec<f64> = read_array(&mut zip, DATA, Some(&[nnz as u64]))?;
        let target_measure: Vec<f64> = read_array(&mut zip, TARGET_MEASURE, Some(&[n_tgt as u64]))?;

        let default_value = if zip.file_names().any(|name| name == DEFAULT_VALUE) {
            let values: Vec<f64> = read_array(&mut zip, DEFAULT_VALUE, Some(&[1]))?;
            Some(values[0])
        } else {
            None
        };

        // Map the file's source columns to the current in-memory element-type order.
        let mut current_source = file_source.clone();
        current_source.sort_by_key(|(et, _, _)| *et);
        let mut new_offsets = BTreeMap::new();
        let mut offset = 0usize;
        for (et, count, _) in &current_source {
            new_offsets.insert(*et, offset);
            offset += *count;
        }
        let source_blocks: Vec<(ElementType, usize)> = current_source
            .iter()
            .map(|(et, count, _)| (*et, *count))
            .collect();

        let mut remap = vec![0usize; n_src];
        for (et, count, old_offset) in &file_source {
            let old_offset = *old_offset;
            let new_offset = new_offsets[et];
            for local in 0..*count {
                remap[old_offset + local] = new_offset + local;
            }
        }
        for idx in &mut indices {
            let old = *idx;
            if old >= n_src {
                return Err(invalid(format!(
                    "source index {old} is out of range for {n_src} source cells"
                )));
            }
            *idx = remap[old];
        }

        let mut target: Vec<(ElementType, RowSparse)> = Vec::with_capacity(file_target.len());
        for (et, count, row_offset) in &file_target {
            let row_offset = *row_offset;
            let block_start = indptr[row_offset];
            let block_end = indptr[row_offset + count];
            if block_start > block_end {
                return Err(invalid(format!(
                    "target block {et:?} has decreasing row pointers"
                )));
            }
            let row_ptr = indptr[row_offset..=row_offset + count]
                .iter()
                .map(|&p| p - block_start)
                .collect();
            target.push((
                *et,
                RowSparse {
                    target_measure: nd::Array1::from_vec(
                        target_measure[row_offset..row_offset + count].to_vec(),
                    ),
                    row_ptr,
                    src_idx: indices[block_start..block_end].to_vec(),
                    weights: data[block_start..block_end].to_vec(),
                },
            ));
        }
        target.sort_by_key(|(et, _)| *et);

        let operator = TransferOperator::build(method, src_dim, tgt_dim, source_blocks, target);
        operator.validate_structure()?;
        Ok((operator, default_value))
    }

    /// Checks that the operator is structurally consistent and matches a source mesh, plus an
    /// optional target mesh.
    pub fn validate(
        &self,
        mesh_src: &UMeshView,
        mesh_tgt: Option<&UMeshView>,
    ) -> Result<(), MefikitIOError> {
        self.validate_structure()?;

        if mesh_src.topological_dimension() != Some(self.src_dim()) {
            return Err(invalid(format!(
                "source mesh topological dimension is {:?}, operator expects {:?}",
                mesh_src.topological_dimension(),
                self.src_dim()
            )));
        }
        let space = mesh_src.space_dimension();
        if !(2..=3).contains(&space) {
            return Err(invalid(format!(
                "source mesh space dimension {space} is not supported (expected 2 or 3)"
            )));
        }

        let expected_source: Vec<(ElementType, usize)> = mesh_src
            .blocks()
            .filter(|(et, _)| et.dimension() == self.src_dim())
            .map(|(et, block)| (*et, block.len()))
            .collect();
        if expected_source != self.source_blocks() {
            return Err(invalid(format!(
                "source mesh blocks {:?} do not match operator blocks {:?}",
                expected_source,
                self.source_blocks()
            )));
        }

        if let Some(mesh_tgt) = mesh_tgt {
            if mesh_tgt.space_dimension() != space {
                return Err(invalid(format!(
                    "source and target meshes have different space dimensions: {space} and {}",
                    mesh_tgt.space_dimension()
                )));
            }
            if mesh_tgt.topological_dimension() != Some(self.tgt_dim()) {
                return Err(invalid(format!(
                    "target mesh topological dimension is {:?}, operator expects {:?}",
                    mesh_tgt.topological_dimension(),
                    self.tgt_dim()
                )));
            }
            let expected_target: Vec<(ElementType, usize)> = match self.method() {
                TransferMethod::InverseDistance { .. }
                | TransferMethod::MovingLeastSquares { .. } => mesh_tgt
                    .blocks()
                    .map(|(et, block)| (*et, block.len()))
                    .collect(),
                TransferMethod::ConstantPiecewise { .. } | TransferMethod::ConservativeP0 => {
                    mesh_tgt
                        .blocks()
                        .filter(|(et, _)| et.dimension() == self.tgt_dim())
                        .map(|(et, block)| (*et, block.len()))
                        .collect()
                }
            };
            let actual_target: Vec<(ElementType, usize)> = self.target_blocks().collect();
            if expected_target != actual_target {
                return Err(invalid(format!(
                    "target mesh blocks {expected_target:?} do not match operator blocks {actual_target:?}"
                )));
            }
        }

        Ok(())
    }

    fn validate_structure(&self) -> Result<(), MefikitIOError> {
        let _ = TransferMethod::try_from(MethodMetadata::from(self.method()))
            .map_err(|e| invalid(format!("invalid transfer method: {e}")))?;
        if matches!(self.method(), TransferMethod::ConservativeP0)
            && self.src_dim() < self.tgt_dim()
        {
            return Err(invalid(format!(
                "conservative P0 cannot transfer from {:?} to {:?}",
                self.src_dim(),
                self.tgt_dim()
            )));
        }

        let n_src = self.n_src();
        let mut source_seen = BTreeSet::new();
        for (et, count) in self.source_blocks() {
            if !source_seen.insert(*et) {
                return Err(invalid(format!("duplicate source block {et:?}")));
            }
            if et.dimension() != self.src_dim() {
                return Err(invalid(format!(
                    "source block {et:?} has dimension {:?}, expected {:?}",
                    et.dimension(),
                    self.src_dim()
                )));
            }
            if *count > n_src {
                return Err(invalid(format!(
                    "source block {et:?} count is out of range"
                )));
            }
        }

        let mut target_seen = BTreeSet::new();
        let require_target_dim = matches!(
            self.method(),
            TransferMethod::ConstantPiecewise { .. } | TransferMethod::ConservativeP0
        );
        for (et, rows) in self.data() {
            if !target_seen.insert(*et) {
                return Err(invalid(format!("duplicate target block {et:?}")));
            }
            if et.dimension() > self.tgt_dim()
                || (require_target_dim && et.dimension() != self.tgt_dim())
            {
                return Err(invalid(format!(
                    "target block {et:?} has dimension {:?}, expected {:?}",
                    et.dimension(),
                    self.tgt_dim()
                )));
            }
            let n_rows = rows.target_measure.len();
            if rows.row_ptr.len() != n_rows + 1 {
                return Err(invalid(format!(
                    "target block {et:?} has {} row pointers for {n_rows} rows",
                    rows.row_ptr.len()
                )));
            }
            if rows.row_ptr.first() != Some(&0) {
                return Err(invalid(format!(
                    "target block {et:?} row pointers do not start at 0"
                )));
            }
            if rows.row_ptr.last().copied() != Some(rows.src_idx.len()) {
                return Err(invalid(format!(
                    "target block {et:?} row pointers do not end at nnz"
                )));
            }
            if rows.src_idx.len() != rows.weights.len() {
                return Err(invalid(format!(
                    "target block {et:?} has {} source indices but {} weights",
                    rows.src_idx.len(),
                    rows.weights.len()
                )));
            }
            if rows.row_ptr.windows(2).any(|w| w[0] > w[1]) {
                return Err(invalid(format!(
                    "target block {et:?} row pointers are not monotonic"
                )));
            }
            if rows.src_idx.iter().any(|&idx| idx >= n_src) {
                return Err(invalid(format!(
                    "target block {et:?} contains a source index out of range"
                )));
            }
            if matches!(self.method(), TransferMethod::ConservativeP0)
                && rows
                    .target_measure
                    .iter()
                    .any(|&m| !m.is_finite() || m <= 0.0)
            {
                return Err(invalid(format!(
                    "target block {et:?} has a non-positive target measure"
                )));
            }
        }

        Ok(())
    }
}

fn file_options(compressed: bool) -> SimpleFileOptions {
    SimpleFileOptions::default()
        .compression_method(if compressed {
            CompressionMethod::Deflated
        } else {
            CompressionMethod::Stored
        })
        .large_file(true)
}

fn write_format<W: Write + Seek>(
    zip: &mut ZipWriter<W>,
    options: SimpleFileOptions,
) -> Result<(), MefikitIOError> {
    zip.start_file(FORMAT, options)
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    let dtype = npyz::DType::Plain(
        "|S3"
            .parse()
            .map_err(|e: npyz::ParseTypeStrError| MefikitIOError::Encode(e.to_string()))?,
    );
    let mut writer = npyz::WriteOptions::new()
        .dtype(dtype)
        .shape(&[])
        .writer(zip)
        .begin_nd()
        .map_err(encode)?;
    let value: &[u8] = b"csr";
    writer.push(value).map_err(encode)?;
    writer.finish().map_err(encode)?;
    Ok(())
}

fn write_npy<W, T>(
    zip: &mut ZipWriter<W>,
    name: &str,
    options: SimpleFileOptions,
    shape: &[u64],
    values: impl IntoIterator<Item = T>,
) -> Result<(), MefikitIOError>
where
    W: Write + Seek,
    T: npyz::AutoSerialize,
{
    zip.start_file(name, options)
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    let mut writer = npyz::WriteOptions::new()
        .default_dtype()
        .shape(shape)
        .writer(zip)
        .begin_nd()
        .map_err(encode)?;
    writer.extend(values).map_err(encode)?;
    writer.finish().map_err(encode)?;
    Ok(())
}

fn open_npy<'a, R: Read + Seek>(
    zip: &'a mut ZipArchive<R>,
    name: &str,
) -> Result<npyz::NpyFile<zip::read::ZipFile<'a, R>>, MefikitIOError> {
    let file = zip.by_name(name).map_err(|e| match e {
        zip::result::ZipError::FileNotFound => invalid(format!("missing array '{name}'")),
        other => invalid(format!("cannot open array '{name}': {other}")),
    })?;
    npyz::NpyFile::new(file).map_err(|e| invalid(format!("invalid NPY array '{name}': {e}")))
}

fn read_array<T, R>(
    zip: &mut ZipArchive<R>,
    name: &str,
    expected_shape: Option<&[u64]>,
) -> Result<Vec<T>, MefikitIOError>
where
    T: npyz::Deserialize,
    R: Read + Seek,
{
    let npy = open_npy(zip, name)?;
    if let Some(expected) = expected_shape
        && npy.shape() != expected
    {
        return Err(invalid(format!(
            "array '{name}' has shape {:?}, expected {expected:?}",
            npy.shape()
        )));
    }
    npy.into_vec::<T>()
        .map_err(|e| invalid(format!("cannot read array '{name}': {e}")))
}

fn read_indices<R: Read + Seek>(
    zip: &mut ZipArchive<R>,
    name: &str,
    expected_len: usize,
) -> Result<Vec<usize>, MefikitIOError> {
    let npy = open_npy(zip, name)?;
    let expected = [expected_len as u64];
    if npy.shape() != expected {
        return Err(invalid(format!(
            "array '{name}' has shape {:?}, expected {expected:?}",
            npy.shape()
        )));
    }
    match npy.try_data::<i32>() {
        Ok(data) => data
            .map(|item| {
                item.map_err(|e| invalid(format!("cannot read array '{name}': {e}")))
                    .and_then(|value| usize::try_from(value).map_err(|_| negative_index(name)))
            })
            .collect(),
        Err(npy) => match npy.try_data::<i64>() {
            Ok(data) => data
                .map(|item| {
                    item.map_err(|e| invalid(format!("cannot read array '{name}': {e}")))
                        .and_then(|value| usize::try_from(value).map_err(|_| negative_index(name)))
                })
                .collect(),
            Err(_) => Err(invalid(format!(
                "array '{name}' must contain 32- or 64-bit signed integers"
            ))),
        },
    }
}

fn checked_source_blocks(
    blocks: &[SourceBlock],
    n_src: usize,
    src_dim: Dimension,
) -> Result<Vec<(ElementType, usize, usize)>, MefikitIOError> {
    let mut result = Vec::with_capacity(blocks.len());
    let mut seen = BTreeSet::new();
    let mut offset = 0usize;
    for block in blocks {
        if !seen.insert(block.element_type) {
            return Err(invalid(format!(
                "duplicate source block {:?}",
                block.element_type
            )));
        }
        if block.element_type.dimension() != src_dim {
            return Err(invalid(format!(
                "source block {:?} has dimension {:?}, expected {src_dim:?}",
                block.element_type,
                block.element_type.dimension()
            )));
        }
        let count = usize_from_u64(block.count, "source block count")?;
        let block_offset = usize_from_u64(block.column_offset, "source block offset")?;
        if block_offset != offset {
            return Err(invalid(format!(
                "source block {:?} starts at {block_offset}, expected {offset}",
                block.element_type
            )));
        }
        offset = offset
            .checked_add(count)
            .ok_or_else(|| invalid("source block counts overflow"))?;
        result.push((block.element_type, count, block_offset));
    }
    if offset != n_src {
        return Err(invalid(format!(
            "source blocks cover {offset} columns, expected {n_src}"
        )));
    }
    Ok(result)
}

fn checked_target_blocks(
    blocks: &[TargetBlock],
    n_tgt: usize,
    tgt_dim: Dimension,
    method: TransferMethod,
) -> Result<Vec<(ElementType, usize, usize)>, MefikitIOError> {
    let mut result = Vec::with_capacity(blocks.len());
    let mut seen = BTreeSet::new();
    let mut offset = 0usize;
    let require_target_dim = matches!(
        method,
        TransferMethod::ConstantPiecewise { .. } | TransferMethod::ConservativeP0
    );
    for block in blocks {
        if !seen.insert(block.element_type) {
            return Err(invalid(format!(
                "duplicate target block {:?}",
                block.element_type
            )));
        }
        if block.element_type.dimension() > tgt_dim
            || (require_target_dim && block.element_type.dimension() != tgt_dim)
        {
            return Err(invalid(format!(
                "target block {:?} has dimension {:?}, expected {tgt_dim:?}",
                block.element_type,
                block.element_type.dimension()
            )));
        }
        let count = usize_from_u64(block.count, "target block count")?;
        let block_offset = usize_from_u64(block.row_offset, "target block offset")?;
        if block_offset != offset {
            return Err(invalid(format!(
                "target block {:?} starts at {block_offset}, expected {offset}",
                block.element_type
            )));
        }
        offset = offset
            .checked_add(count)
            .ok_or_else(|| invalid("target block counts overflow"))?;
        result.push((block.element_type, count, block_offset));
    }
    if offset != n_tgt {
        return Err(invalid(format!(
            "target blocks cover {offset} rows, expected {n_tgt}"
        )));
    }
    Ok(result)
}

fn validate_indptr(indptr: &[usize], n_tgt: usize) -> Result<(), MefikitIOError> {
    if indptr.len() != n_tgt + 1 {
        return Err(invalid(format!(
            "indptr has {} entries, expected {}",
            indptr.len(),
            n_tgt + 1
        )));
    }
    if indptr.first() != Some(&0) {
        return Err(invalid("indptr does not start at 0"));
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(invalid("indptr is not monotonic"));
    }
    Ok(())
}

fn usize_from_u64(value: u64, what: &str) -> Result<usize, MefikitIOError> {
    usize::try_from(value).map_err(|_| invalid(format!("{what} ({value}) does not fit in usize")))
}

fn usize_from_i64(value: i64, what: &str) -> Result<usize, MefikitIOError> {
    usize::try_from(value).map_err(|_| invalid(format!("{what} ({value}) is not a valid size")))
}

fn negative_index(name: &str) -> MefikitIOError {
    invalid(format!("array '{name}' contains a negative index"))
}

fn encode(error: impl std::fmt::Display) -> MefikitIOError {
    MefikitIOError::Encode(error.to_string())
}

fn malformed(message: String) -> MefikitIOError {
    MefikitIOError::MalformedFile(message)
}

fn invalid(message: impl Into<String>) -> MefikitIOError {
    MefikitIOError::MalformedFile(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples::{make_imesh_2d, make_mesh_2d_multi_simple};
    use crate::tools::meshless::DistanceWeighting;

    fn assert_same_operator(left: &TransferOperator, right: &TransferOperator) {
        assert_eq!(left.method(), right.method());
        assert_eq!(left.src_dim(), right.src_dim());
        assert_eq!(left.tgt_dim(), right.tgt_dim());
        assert_eq!(left.source_blocks(), right.source_blocks());
        assert_eq!(left.data().len(), right.data().len());
        for ((left_et, left_rows), (right_et, right_rows)) in
            left.data().iter().zip(right.data().iter())
        {
            assert_eq!(left_et, right_et);
            assert_eq!(left_rows.target_measure, right_rows.target_measure);
            assert_eq!(left_rows.row_ptr, right_rows.row_ptr);
            assert_eq!(left_rows.src_idx, right_rows.src_idx);
            assert_eq!(left_rows.weights, right_rows.weights);
        }
    }

    #[test]
    fn roundtrip_all_methods() {
        let dir = tempfile::tempdir().unwrap();
        let src = make_imesh_2d(2);
        let tgt = make_imesh_2d(2);
        let methods = [
            TransferMethod::ConstantPiecewise {
                point_location: PointLocation::Centroid,
            },
            TransferMethod::ConservativeP0,
            TransferMethod::InverseDistance {
                k: 4,
                exponent: 2.0,
            },
            TransferMethod::MovingLeastSquares {
                k: 4,
                weighting: DistanceWeighting::Constant,
            },
        ];

        for method in methods {
            let op = TransferOperator::new(&src.view(), &tgt.view(), method);
            let path = dir.path().join("operator.npz");
            op.save_with(
                &path,
                &TransferSaveOptions {
                    compressed: true,
                    default_value: Some(f64::NAN),
                },
            )
            .unwrap();
            let (loaded, default) = TransferOperator::load(&path).unwrap();
            assert!(default.unwrap().is_nan());
            assert_same_operator(&op, &loaded);
            loaded.validate(&src.view(), Some(&tgt.view())).unwrap();
            std::fs::remove_file(path).unwrap();
        }
    }

    /// The source has QUAD4 and PGON blocks. They are name-sorted in the file (PGON, QUAD4)
    /// while the in-memory order is the enum order (QUAD4, PGON), so this test exercises the
    /// column remapping on load.
    #[test]
    fn roundtrip_mixed_source_remaps_columns() {
        let dir = tempfile::tempdir().unwrap();
        let src = make_mesh_2d_multi_simple();
        let tgt = make_imesh_2d(1);
        let method = TransferMethod::ConstantPiecewise {
            point_location: PointLocation::Centroid,
        };
        let op = TransferOperator::new(&src.view(), &tgt.view(), method);
        let path = dir.path().join("mixed.npz");
        op.save(&path).unwrap();
        let (loaded, default) = TransferOperator::load(&path).unwrap();
        assert!(default.is_none());
        assert_same_operator(&op, &loaded);
        loaded.validate(&src.view(), Some(&tgt.view())).unwrap();
    }

    #[test]
    fn validate_rejects_mismatched_meshes() {
        let src = make_imesh_2d(2);
        let tgt = make_imesh_2d(2);
        let op = TransferOperator::new(&src.view(), &tgt.view(), TransferMethod::ConservativeP0);
        let other = make_imesh_2d(3);
        assert!(op.validate(&other.view(), Some(&tgt.view())).is_err());
        assert!(op.validate(&src.view(), Some(&other.view())).is_err());
    }

    #[test]
    fn load_rejects_a_non_transfer_npz() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-transfer.npz");
        let file = File::create(&path).unwrap();
        let mut zip = ZipWriter::new(BufWriter::new(file));
        let options = file_options(false);
        write_npy(&mut zip, SHAPE, options, &[2], [0i64, 0]).unwrap();
        zip.finish().unwrap();
        assert!(TransferOperator::load(&path).is_err());
    }
}
