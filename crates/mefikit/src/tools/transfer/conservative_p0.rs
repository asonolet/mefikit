//! Conservative P0 transfer between full-dimensional meshes of the same space dimension.
//!
//! Each target cell value is the measure-weighted average of the source cell values it overlaps,
//! so the transferred field is exactly conservative: for an intensive field the integral of the
//! target field over the covered target cells equals the integral of the source field over the
//! source cells, and for an extensive field the per-cell values sum to the source total.
//!
//! The overlap measures are precomputed once at construction time and stored as a sparse matrix,
//! so [`Transfer::apply`] is a sparse matrix-vector product that can be reused for any field as
//! long as the meshes do not change.

use std::collections::BTreeMap;

use ndarray as nd;

use super::transfer_trait::{FieldNature, Transfer};
use crate::element_traits::ElementGeo;
use crate::geometry::{Polyhedron, cross2, into_ccw2, signed_area2};
use crate::mesh::{Dimension, ElementId, ElementType, FieldOwnedD, FieldViewD, UMeshView};
use crate::tools::spatial_index::SpatiallyIndexable;

/// Conservative P0 transfer operator.
///
/// The operator precomputes, for every target cell, the measure of its intersection with every
/// overlapping source cell (the intersection area in 2D, the intersection volume in 3D). At apply
/// time each target cell accumulates the source values weighted by the overlap measures: the raw
/// sum for [`FieldNature::Extensive`] fields and the sum normalized by the target cell measure for
/// [`FieldNature::Intensive`] fields. A target cell not covered by the source mesh keeps the
/// `default` value.
#[derive(Debug, Clone)]
pub struct ConservativeP0Transfer {
    /// Topological dimension of the full-dimensional source cells.
    src_dim: Dimension,
    tgt_dim: Dimension,
    /// Number of source cells the operator was built from, flattened over element types in
    /// BTreeMap order.
    n_src: usize,
    /// Sparse overlap data, grouped by target element type.
    data: Vec<(ElementType, OverlapMatrix)>,
}

/// Sparse (CSR) overlap matrix between source and target cells for a single target element type.
#[derive(Debug, Clone)]
struct OverlapMatrix {
    /// Measure (area or volume) of each target cell.
    target_measure: nd::Array1<f64>,
    /// Row pointer: the overlaps of target cell `j` live in `src_idx[row_ptr[j]..row_ptr[j + 1]]`.
    row_ptr: Vec<usize>,
    /// Global index (in the flattened source array) of each overlapping source cell.
    src_idx: Vec<usize>,
    /// Overlap measure `|s_i ∩ t_j|` of each pair.
    overlap: Vec<f64>,
}

impl ConservativeP0Transfer {
    /// Builds a conservative P0 transfer operator from source cells to target cells.
    ///
    /// Both meshes must be full-dimensional (their topological dimension must match the space
    /// dimension `D`, which must be 2 or 3): the transfer computes the intersection area (in 2D)
    /// or intersection volume (in 3D) of the source and target cells, so every cell must be a full
    /// `D`-dimensional region (the intersection of a full-dimensional cell with a lower-dimensional
    /// cell has zero measure). The cells must be convex: this is always the case for `TRI3`,
    /// `QUAD4`, `TET4` and `HEX8` elements, while `PGON` and `PHED` cells are assumed convex.
    ///
    /// # Panics
    ///
    /// - If `mesh_src` and `mesh_tgt` do not share the same space dimension, or if it is not 2 or 3.
    /// - If either mesh is empty or not full-dimensional.
    pub fn new(mesh_src: &UMeshView, mesh_tgt: &UMeshView) -> Self {
        let src_space = mesh_src.space_dimension();
        let tgt_space = mesh_tgt.space_dimension();
        assert_eq!(
            src_space, tgt_space,
            "Source and target meshes should share the same space dimension, got source = {src_space}D and target = {tgt_space}D"
        );
        assert!(
            (2..=3).contains(&src_space),
            "Conservative P0 transfer is only supported in 2D and 3D space, got {src_space}D"
        );
        let src_dim = mesh_src
            .topological_dimension()
            .expect("Source mesh should not be empty");
        let tgt_dim = mesh_tgt
            .topological_dimension()
            .expect("Target mesh should not be empty");
        let src_dim_usize = u8::from(src_dim) as usize;
        assert_eq!(
            src_dim_usize, src_space,
            "Source mesh should be full-dimensional (topological dimension = space dimension), got topological {src_dim:?} in a {src_space}D space"
        );
        let tgt_dim_usize = u8::from(tgt_dim) as usize;
        assert_eq!(
            tgt_dim_usize, src_space,
            "Target mesh should be full-dimensional (topological dimension = space dimension), got topological {tgt_dim:?} in a {src_space}D space"
        );

        // Flatten the source cells in BTreeMap element-type order, so the global index stored in
        // the overlap matrix matches the concatenation of the field arrays done at apply time.
        let mut src_offsets: BTreeMap<ElementType, usize> = BTreeMap::new();
        let mut n_src = 0;
        for (et, block) in mesh_src.blocks() {
            if u8::from(et.dimension()) as usize == src_space {
                src_offsets.insert(*et, n_src);
                n_src += block.len();
            }
        }

        let data = match src_space {
            2 => Self::build_overlap_2d(mesh_src, mesh_tgt, &src_offsets),
            3 => Self::build_overlap_3d(mesh_src, mesh_tgt, &src_offsets),
            _ => unreachable!(),
        };
        Self {
            src_dim,
            tgt_dim,
            n_src,
            data,
        }
    }

    /// Builds the source-pair overlap areas for all target cells of a 2D mesh.
    fn build_overlap_2d(
        mesh_src: &UMeshView,
        mesh_tgt: &UMeshView,
        src_offsets: &BTreeMap<ElementType, usize>,
    ) -> Vec<(ElementType, OverlapMatrix)> {
        let index = mesh_src.bvh2();
        let mut data = Vec::new();
        for (_, block) in mesh_tgt.blocks() {
            let tgt_et = block.element_type();
            if tgt_et.dimension() != Dimension::D2 {
                continue;
            }
            let n = block.len();
            let mut target_measure = nd::Array1::zeros(n);
            let mut row_ptr = Vec::with_capacity(n + 1);
            row_ptr.push(0);
            let mut src_idx = Vec::new();
            let mut overlap = Vec::new();
            for (j, elem) in block.iter(mesh_tgt.coords()).enumerate() {
                let mut pgon: Vec<[f64; 2]> = elem.coords2().copied().collect();
                into_ccw2(&mut pgon);
                target_measure[j] = signed_area2(&pgon).abs();

                let [min, max] = elem.bounds2();
                // The BVH stores f32 boxes: inflate the query by a small epsilon so a source cell
                // sharing a boundary with the target cell is never missed by the broad phase. The
                // narrow phase below is exact in f64 and discards any spurious candidate.
                let scale = min
                    .iter()
                    .chain(max.iter())
                    .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
                let eps = 1e-6 * scale;
                let candidates =
                    index.in_bounds([min[0] - eps, min[1] - eps], [max[0] + eps, max[1] + eps]);

                for (src_et, indices) in candidates.0 {
                    if src_et.dimension() != Dimension::D2 {
                        continue;
                    }
                    let offset = src_offsets[&src_et];
                    for &i in &indices {
                        let src_elem = mesh_src.element(ElementId::new(src_et, i));
                        let mut src_pgon: Vec<[f64; 2]> = src_elem.coords2().copied().collect();
                        into_ccw2(&mut src_pgon);
                        let area = convex_intersection_area(&src_pgon, &pgon);
                        if area > 1e-15 {
                            src_idx.push(offset + i);
                            overlap.push(area);
                        }
                    }
                }
                row_ptr.push(src_idx.len());
            }
            data.push((
                tgt_et,
                OverlapMatrix {
                    target_measure,
                    row_ptr,
                    src_idx,
                    overlap,
                },
            ));
        }
        data
    }

    /// Builds the source-pair overlap volumes for all target cells of a 3D mesh.
    fn build_overlap_3d(
        mesh_src: &UMeshView,
        mesh_tgt: &UMeshView,
        src_offsets: &BTreeMap<ElementType, usize>,
    ) -> Vec<(ElementType, OverlapMatrix)> {
        // Number of source cells per element type, used to size the lazy polyhedron cache.
        let src_block_len: BTreeMap<ElementType, usize> = mesh_src
            .blocks()
            .map(|(et, block)| (*et, block.len()))
            .collect();

        let index = mesh_src.bvh3();
        // Each source cell may overlap several target cells; cache its polyhedron so it is only
        // built once instead of once per overlapping target cell.
        let mut poly_cache: BTreeMap<ElementType, Vec<Option<Polyhedron>>> = BTreeMap::new();
        let mut data = Vec::new();
        for (_, block) in mesh_tgt.blocks() {
            let tgt_et = block.element_type();
            if tgt_et.dimension() != Dimension::D3 {
                continue;
            }
            let n = block.len();
            let mut target_measure = nd::Array1::zeros(n);
            let mut row_ptr = Vec::with_capacity(n + 1);
            row_ptr.push(0);
            let mut src_idx = Vec::new();
            let mut overlap = Vec::new();
            for (j, elem) in block.iter(mesh_tgt.coords()).enumerate() {
                let tgt_poly = elem.to_polyhedron();
                target_measure[j] = tgt_poly.volume();

                let [min, max] = tgt_poly.bounds();
                let scale = min
                    .iter()
                    .chain(max.iter())
                    .fold(1.0_f64, |acc, &c| acc.max(c.abs()));
                let eps = 1e-6 * scale;
                let candidates = index.in_bounds(
                    [min[0] - eps, min[1] - eps, min[2] - eps],
                    [max[0] + eps, max[1] + eps, max[2] + eps],
                );

                for (src_et, indices) in candidates.0 {
                    if src_et.dimension() != Dimension::D3 {
                        continue;
                    }
                    let offset = src_offsets[&src_et];
                    let n_src_et = src_block_len[&src_et];
                    let slots = poly_cache
                        .entry(src_et)
                        .or_insert_with(|| vec![None; n_src_et]);
                    for &i in &indices {
                        if slots[i].is_none() {
                            slots[i] =
                                Some(mesh_src.element(ElementId::new(src_et, i)).to_polyhedron());
                        }
                        let src_poly = slots[i].as_ref().unwrap();
                        // The candidates already come from a BVH AABB query, so skip the redundant
                        // AABB reject inside the intersection and clip directly.
                        let vol = tgt_poly.convex_intersection_volume_impl(src_poly);
                        if vol > 1e-15 {
                            src_idx.push(offset + i);
                            overlap.push(vol);
                        }
                    }
                }
                row_ptr.push(src_idx.len());
            }
            data.push((
                tgt_et,
                OverlapMatrix {
                    target_measure,
                    row_ptr,
                    src_idx,
                    overlap,
                },
            ));
        }
        data
    }

    /// Evaluates the operator on a flat source array `src` of shape `(n_src, ...)` for the target
    /// element type `et`, producing a `(n_tgt, ...)` array.
    fn apply_on_array(
        &self,
        et: ElementType,
        src: &nd::ArrayViewD<f64>,
        field_nature: FieldNature,
        default: f64,
    ) -> nd::ArrayD<f64> {
        let n_src = src.shape()[0];
        assert_eq!(
            n_src, self.n_src,
            "The field should have one entry per source cell, got {n_src} for {}",
            self.n_src
        );
        let n_compo = src.len() / n_src;
        let (_, matrix) = self.data.iter().find(|(e, _)| *e == et).unwrap();
        let n_tgt = matrix.target_measure.len();

        let mut out_shape = src.raw_dim();
        out_shape[0] = n_tgt;
        let src = src.view().into_shape_with_order((n_src, n_compo)).unwrap();

        let mut tgt = nd::Array::zeros((n_tgt, n_compo));
        for j in 0..n_tgt {
            let (lo, hi) = (matrix.row_ptr[j], matrix.row_ptr[j + 1]);
            if lo == hi {
                tgt.row_mut(j).fill(default);
                continue;
            }
            let mut row = tgt.row_mut(j);
            for p in lo..hi {
                let w = matrix.overlap[p];
                nd::Zip::from(&mut row)
                    .and(src.row(matrix.src_idx[p]))
                    .for_each(|d, &s| {
                        *d += w * s;
                    });
            }
            if field_nature == FieldNature::Intensive {
                let inv = 1.0 / matrix.target_measure[j];
                row *= inv;
            }
        }

        tgt.into_shape_with_order(out_shape).unwrap()
    }
}

impl Transfer for ConservativeP0Transfer {
    fn apply(&self, field: &FieldViewD, field_nature: FieldNature, default: f64) -> FieldOwnedD {
        assert_eq!(
            field.dimension(),
            Some(self.src_dim),
            "The field should be defined on {:?} source cells, got {:?}",
            self.src_dim,
            field.dimension()
        );
        // Concatenate the per-element-type source arrays in the same order used to build the
        // operator (BTreeMap order), so the source indices stored in the overlap matrix are valid
        // whatever the source element types are.
        let src_views: Vec<nd::ArrayViewD<f64>> = field.0.values().map(|a| a.view()).collect();
        let src = nd::concatenate(nd::Axis(0), src_views.as_slice()).unwrap();

        let mut res = BTreeMap::new();
        for (et, _) in &self.data {
            res.insert(
                *et,
                self.apply_on_array(*et, &src.view(), field_nature, default),
            );
        }
        FieldOwnedD::new(res)
    }

    fn tgt_dim(&self) -> Dimension {
        self.tgt_dim
    }
}

/// Area of the intersection of two convex polygons.
fn convex_intersection_area(p: &[[f64; 2]], q: &[[f64; 2]]) -> f64 {
    signed_area2(&clip_convex(p, q)).abs()
}

/// Clips the convex polygon `subject` (CCW) by the convex polygon `clip` (CCW), keeping the part
/// on or to the left of each directed edge of `clip` (Sutherland–Hodgman).
fn clip_convex(subject: &[[f64; 2]], clip: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut output = subject.to_vec();
    for i in 0..clip.len() {
        if output.len() < 3 {
            return Vec::new();
        }
        let (a, b) = (clip[i], clip[(i + 1) % clip.len()]);
        let input = output;
        output = Vec::new();
        for k in 0..input.len() {
            let p = input[k];
            let q = input[(k + 1) % input.len()];
            let p_in = cross2(a, b, p) >= 0.0;
            let q_in = cross2(a, b, q) >= 0.0;
            match (p_in, q_in) {
                (true, true) => output.push(q),
                (true, false) => output.push(segment_intersection(a, b, p, q)),
                (false, true) => {
                    output.push(segment_intersection(a, b, p, q));
                    output.push(q);
                }
                (false, false) => {}
            }
        }
    }
    output
}

/// Intersection point of the segments `[a, b]` and `[p, q]`.
///
/// The segments are assumed to cross; the degenerate collinear case (which never reaches here
/// because collinear points are kept by the inside test) returns `q`.
fn segment_intersection(a: [f64; 2], b: [f64; 2], p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
    let (dax, day) = (b[0] - a[0], b[1] - a[1]);
    let (dpx, dpy) = (q[0] - p[0], q[1] - p[1]);
    let denom = dax * dpy - day * dpx;
    if denom == 0.0 {
        return q;
    }
    let t = ((p[0] - a[0]) * dpy - (p[1] - a[1]) * dpx) / denom;
    [a[0] + t * dax, a[1] + t * day]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{FieldOwnedD, UMesh};
    use crate::mesh_examples as me;
    use crate::tools::grid::RegularUMeshBuilder;

    fn source_with_field(values: nd::Array<f64, nd::IxDyn>) -> UMesh {
        let mut source = me::make_imesh_2d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(ElementType::QUAD4, values)]));
        source.update_field("f", field.into_shared());
        source
    }

    fn field_view(source: &UMesh) -> FieldViewD<'_> {
        source.field("f", Some(Dimension::D2)).unwrap()
    }

    /// Area of each 2D cell of a mesh, used to check the conservation integrals in tests.
    fn cell_areas(mesh: &UMesh) -> Vec<f64> {
        mesh.elements_of_dim(Dimension::D2)
            .map(|e| {
                let mut pgon: Vec<[f64; 2]> = e.coords2().copied().collect();
                into_ccw2(&mut pgon);
                signed_area2(&pgon).abs()
            })
            .collect()
    }

    /// A constant intensive field is reproduced exactly.
    #[test]
    fn transfer_constant_intensive() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_2d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr.shape(), &[16]);
        assert!(arr.iter().all(|&v| v == 7.0));
    }

    /// A vector field (several components per cell) is transferred component-wise.
    #[test]
    fn transfer_vector_field() {
        let mut source = me::make_imesh_2d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![[7.0, 3.0]].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let target = me::make_imesh_2d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(arr.shape(), &[16, 2]);
        assert!(arr.iter().all(|&v| v == 7.0 || v == 3.0));
    }

    /// A nested refined grid: every target cell lies inside a single source cell, so the overlap
    /// ratio is exactly one and each target cell receives the source value.
    #[test]
    fn transfer_nested_ratios() {
        let mut source = me::make_imesh_2d(2);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![1.0, 2.0, 3.0, 4.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let target = me::make_imesh_2d(4);

        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &out.0[&ElementType::QUAD4];
        // Target cell (i, j) of a 4×4 grid lies in source cell (i / 2, j / 2).
        let expected: Vec<f64> = (0..16)
            .map(|k| {
                let sx = (k % 4) / 2;
                let sy = (k / 4) / 2;
                [1.0, 2.0, 3.0, 4.0][sy * 2 + sx]
            })
            .collect();
        assert_eq!(arr.iter().copied().collect::<Vec<f64>>(), expected);
    }

    /// Partial overlaps are weighted by the overlap ratio: an intensive field averages over the
    /// whole target cell, an extensive field sums the overlapped source quantities.
    #[test]
    fn transfer_partial_overlap() {
        let source = source_with_field(nd::array![7.0].into_dyn());

        // Target cell exactly half of the source cell.
        let mut half = UMesh::new(
            nd::ArcArray2::from_shape_vec((4, 2), vec![0.5, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5, 1.0])
                .unwrap(),
        );
        half.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let op = ConservativeP0Transfer::new(&source.view(), &half.view());
        let int = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let ext = op.apply(&field_view(&source), FieldNature::Extensive, 0.0);
        assert_eq!(int.0[&ElementType::QUAD4][0], 7.0);
        assert_eq!(ext.0[&ElementType::QUAD4][0], 3.5);

        // Target cell overlapping only a quarter of the source cell.
        let mut quarter = UMesh::new(
            nd::ArcArray2::from_shape_vec((4, 2), vec![0.5, 0.0, 1.5, 0.0, 1.5, 0.5, 0.5, 0.5])
                .unwrap(),
        );
        quarter.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let op = ConservativeP0Transfer::new(&source.view(), &quarter.view());
        let int = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let ext = op.apply(&field_view(&source), FieldNature::Extensive, 0.0);
        assert_eq!(int.0[&ElementType::QUAD4][0], 3.5);
        assert_eq!(ext.0[&ElementType::QUAD4][0], 1.75);
    }

    /// The transfer is exactly conservative: the integral of the target field equals the integral
    /// of the source field, for a target mesh offset from the source cells.
    #[test]
    fn transfer_mass_conservation() {
        let mut source = me::make_imesh_2d(2);
        let values: Vec<f64> = source
            .elements_of_dim(Dimension::D2)
            .map(|e| {
                let c = e.centroid2();
                1.0 + c[0] + 2.0 * c[1]
            })
            .collect();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::Array::from_iter(values).into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        // A target grid offset by half a source cell so the overlaps are general.
        let axis: Vec<f64> = (0..15).map(|k| -0.25 + 0.1 * k as f64).collect();
        let target = RegularUMeshBuilder::new()
            .add_axis(axis.clone())
            .add_axis(axis)
            .build();

        let src_total: f64 = field_view(&source)
            .0
            .values()
            .zip(cell_areas(&source))
            .map(|(arr, area)| area * arr.iter().sum::<f64>())
            .sum();

        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let ext = op.apply(&field_view(&source), FieldNature::Extensive, 0.0);
        let ext_total: f64 = ext.0.values().map(|a| a.sum()).sum();
        assert!(
            (ext_total - src_total).abs() < 1e-9,
            "extensive total {ext_total} != source total {src_total}"
        );

        let int = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let int_total: f64 = int
            .0
            .values()
            .zip(cell_areas(&target))
            .map(|(arr, area)| area * arr.iter().sum::<f64>())
            .sum();
        assert!(
            (int_total - src_total).abs() < 1e-9,
            "intensive integral {int_total} != source total {src_total}"
        );
    }

    /// Target cells not covered by the source mesh keep the default value.
    #[test]
    fn transfer_default_uncovered() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0, 2.0])
            .add_axis(vec![0.0, 1.0, 2.0])
            .build();
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 99.0);
        let arr = &field.0[&ElementType::QUAD4];
        assert_eq!(
            arr.iter().copied().collect::<Vec<f64>>(),
            vec![7.0, 99.0, 99.0, 99.0]
        );
    }

    /// Disjoint meshes produce only default values.
    #[test]
    fn transfer_no_overlap() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let target = RegularUMeshBuilder::new()
            .add_axis(vec![2.0, 3.0])
            .add_axis(vec![2.0, 3.0])
            .build();
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view(&source), FieldNature::Intensive, 99.0);
        assert!(field.0[&ElementType::QUAD4].iter().all(|&v| v == 99.0));
    }

    /// A triangular source cell clips the target cells along the hypotenuse.
    #[test]
    fn transfer_triangular_source() {
        let mut source = UMesh::new(
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap(),
        );
        source.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 2]]).to_shared(), None);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::TRI3,
            nd::array![7.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        let target = me::make_imesh_2d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let out = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        let arr = &out.0[&ElementType::QUAD4];
        for (k, &v) in arr.iter().enumerate() {
            let i = k % 4;
            let j = k / 4;
            if i + j <= 2 {
                assert_eq!(v, 7.0, "cell ({i}, {j}) should be fully covered");
            } else if i + j >= 4 {
                assert_eq!(v, 0.0, "cell ({i}, {j}) should be outside");
            } else {
                assert!((0.0..7.0).contains(&v), "cell ({i}, {j}) partially covered");
            }
        }
    }

    /// Source cells of several element types are correctly addressed by the overlap matrix.
    #[test]
    fn transfer_mixed_source_types() {
        let mut source = UMesh::new(
            nd::ArcArray2::from_shape_vec(
                (6, 2),
                vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            )
            .unwrap(),
        );
        source.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 3]]).to_shared(), None);
        source.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[1, 2, 5, 4]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([
            (ElementType::TRI3, nd::array![20.0].into_dyn()),
            (ElementType::QUAD4, nd::array![10.0].into_dyn()),
        ]));
        source.update_field("f", field.into_shared());

        let mut target = UMesh::new(
            nd::ArcArray2::from_shape_vec(
                (6, 2),
                vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
            )
            .unwrap(),
        );
        target.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 4, 3], [1, 2, 5, 4]]).to_shared(),
            None,
        );

        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        // Left cell overlaps the triangle (area 1/2, extensive 10), right cell the quad (10).
        let ext = op.apply(&field_view(&source), FieldNature::Extensive, 0.0);
        assert_eq!(
            ext.0[&ElementType::QUAD4]
                .iter()
                .copied()
                .collect::<Vec<f64>>(),
            vec![10.0, 10.0]
        );
        let int = op.apply(&field_view(&source), FieldNature::Intensive, 0.0);
        assert_eq!(
            int.0[&ElementType::QUAD4]
                .iter()
                .copied()
                .collect::<Vec<f64>>(),
            vec![10.0, 10.0]
        );
    }

    /// The same operator can evaluate several fields without rebuilding the precompute.
    #[test]
    fn transfer_reuse() {
        let mut source = me::make_imesh_2d(1);
        let f1 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![7.0].into_dyn(),
        )]));
        let f2 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::QUAD4,
            nd::array![3.0].into_dyn(),
        )]));
        source.update_field("f1", f1.into_shared());
        source.update_field("f2", f2.into_shared());
        let target = me::make_imesh_2d(2);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let r1 = op.apply(
            &source.field("f1", Some(Dimension::D2)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        let r2 = op.apply(
            &source.field("f2", Some(Dimension::D2)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert!(r1.0[&ElementType::QUAD4].iter().all(|&v| v == 7.0));
        assert!(r2.0[&ElementType::QUAD4].iter().all(|&v| v == 3.0));
    }

    /// `apply_update` stores the transferred field on the target mesh.
    #[test]
    fn transfer_apply_update() {
        let source = source_with_field(nd::array![7.0].into_dyn());
        let mut target = me::make_imesh_2d(2);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let old = op.apply_update(
            &mut target,
            "transferred",
            &field_view(&source),
            FieldNature::Intensive,
            0.0,
        );
        assert!(old.is_none());
        let field = target.field("transferred", Some(Dimension::D2)).unwrap();
        assert!(field.0[&ElementType::QUAD4].iter().all(|&v| v == 7.0));
    }

    fn source3d_with_field(values: nd::Array<f64, nd::IxDyn>) -> UMesh {
        let mut source = me::make_imesh_3d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(ElementType::HEX8, values)]));
        source.update_field("f", field.into_shared());
        source
    }

    fn field_view_3d(source: &UMesh) -> FieldViewD<'_> {
        source.field("f", Some(Dimension::D3)).unwrap()
    }

    /// Volume of each 3D cell of a mesh, used to check the conservation integrals in tests.
    fn cell_volumes(mesh: &UMesh) -> Vec<f64> {
        mesh.elements_of_dim(Dimension::D3)
            .map(|e| e.measure3())
            .collect()
    }

    /// Builds a single HEX8 cell whose 8 nodes are enumerated with x fastest, then y, then z.
    fn hex_cell(coords: [[f64; 3]; 8]) -> UMesh {
        let flat: Vec<f64> = coords.iter().copied().flatten().collect();
        let mut mesh = UMesh::new(nd::ArcArray2::from_shape_vec((8, 3), flat).unwrap());
        mesh.add_regular_block(
            ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        mesh
    }

    /// A constant intensive field is reproduced exactly by the 3D transfer.
    #[test]
    fn transfer3d_constant_intensive() {
        let source = source3d_with_field(nd::array![7.0].into_dyn());
        let target = me::make_imesh_3d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::HEX8];
        assert_eq!(arr.shape(), &[64]);
        assert!(arr.iter().all(|&v| (v - 7.0).abs() < 1e-10));
    }

    /// A vector field (several components per cell) is transferred component-wise in 3D.
    #[test]
    fn transfer3d_vector_field() {
        let mut source = me::make_imesh_3d(1);
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::array![[7.0, 3.0]].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let target = me::make_imesh_3d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let arr = &field.0[&ElementType::HEX8];
        assert_eq!(arr.shape(), &[64, 2]);
        assert!(
            arr.iter()
                .all(|&v| (v - 7.0).abs() < 1e-10 || (v - 3.0).abs() < 1e-10)
        );
    }

    /// A nested refined grid: every target cell lies inside a single source cell, so the
    /// overlap ratio is exactly one and each target cell receives the source value.
    #[test]
    fn transfer3d_nested_ratios() {
        let mut source = me::make_imesh_3d(2);
        let values: Vec<f64> = (0..8).map(|k| (k + 1) as f64).collect();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::Array::from_iter(values).into_dyn(),
        )]));
        source.update_field("f", field.into_shared());
        let target = me::make_imesh_3d(4);

        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let out = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let arr = &out.0[&ElementType::HEX8];
        // Target cell index k = ((z * 4) + y) * 4 + x maps to source (x/2, y/2, z/2) numbered
        // as sz * 4 + sy * 2 + sx.
        let expected: Vec<f64> = (0..64)
            .map(|k| {
                let (z, y, x) = (k / 16, (k % 16) / 4, k % 4);
                (4 * (z / 2) + 2 * (y / 2) + (x / 2) + 1) as f64
            })
            .collect();
        assert_eq!(arr.len(), expected.len());
        for (got, want) in arr.iter().zip(&expected) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }

    /// Partial overlaps are weighted by the overlap ratio: an intensive field averages over the
    /// whole target cell, an extensive field sums the overlapped source quantities.
    #[test]
    fn transfer3d_partial_overlap() {
        let source = source3d_with_field(nd::array![7.0].into_dyn());

        // Target cell exactly half of the source cell in x: [0, 1/2] × [0, 1] × [0, 1].
        let half = hex_cell([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.5, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]);
        let op = ConservativeP0Transfer::new(&source.view(), &half.view());
        let int = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let ext = op.apply(&field_view_3d(&source), FieldNature::Extensive, 0.0);
        assert!((int.0[&ElementType::HEX8][0] - 7.0).abs() < 1e-10);
        assert!((ext.0[&ElementType::HEX8][0] - 3.5).abs() < 1e-10);

        // Target cell overlapping only a quarter of the source cell in x and y.
        let quarter = hex_cell([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.5, 0.5, 1.0],
            [0.0, 0.5, 1.0],
        ]);
        let op = ConservativeP0Transfer::new(&source.view(), &quarter.view());
        let int = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let ext = op.apply(&field_view_3d(&source), FieldNature::Extensive, 0.0);
        assert!((int.0[&ElementType::HEX8][0] - 7.0).abs() < 1e-10);
        assert!((ext.0[&ElementType::HEX8][0] - 1.75).abs() < 1e-10);
    }

    /// The 3D transfer is exactly conservative: for a target mesh offset from the source cells,
    /// the integral of the target field equals the integral of the source field.
    #[test]
    fn transfer3d_mass_conservation() {
        let mut source = me::make_imesh_3d(2);
        let values: Vec<f64> = source
            .elements_of_dim(Dimension::D3)
            .map(|e| {
                let mut sum = [0.0; 3];
                let mut n = 0usize;
                for p in e.coords3() {
                    n += 1;
                    for k in 0..3 {
                        sum[k] += p[k];
                    }
                }
                let c = [sum[0] / n as f64, sum[1] / n as f64, sum[2] / n as f64];
                1.0 + c[0] + 2.0 * c[1] + 3.0 * c[2]
            })
            .collect();
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::Array::from_iter(values).into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        // A target grid offset by a fraction of a source cell so the overlaps are general.
        let axis: Vec<f64> = (0..15).map(|k| -0.25 + 0.1 * k as f64).collect();
        let target = RegularUMeshBuilder::new()
            .add_axis(axis.clone())
            .add_axis(axis.clone())
            .add_axis(axis)
            .build();

        let src_total: f64 = field_view_3d(&source)
            .0
            .values()
            .zip(cell_volumes(&source))
            .map(|(arr, vol)| vol * arr.iter().sum::<f64>())
            .sum();

        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let ext = op.apply(&field_view_3d(&source), FieldNature::Extensive, 0.0);
        let ext_total: f64 = ext.0.values().map(|a| a.sum()).sum();
        assert!(
            (ext_total - src_total).abs() < 1e-8,
            "extensive total {ext_total} != source total {src_total}"
        );

        let int = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let int_total: f64 = int
            .0
            .values()
            .zip(cell_volumes(&target))
            .map(|(arr, vol)| vol * arr.iter().sum::<f64>())
            .sum();
        assert!(
            (int_total - src_total).abs() < 1e-8,
            "intensive integral {int_total} != source total {src_total}"
        );
    }

    /// Target cells not covered by the source mesh keep the default value.
    #[test]
    fn transfer3d_default_uncovered() {
        let source = source3d_with_field(nd::array![7.0].into_dyn());
        let target = RegularUMeshBuilder::new()
            .add_axis(vec![0.0, 1.0, 2.0])
            .add_axis(vec![0.0, 1.0, 2.0])
            .add_axis(vec![0.0, 1.0, 2.0])
            .build();
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view_3d(&source), FieldNature::Intensive, 99.0);
        let arr = &field.0[&ElementType::HEX8];
        assert_eq!(arr.shape(), &[8]);
        for (k, &v) in arr.iter().enumerate() {
            let expected = if k == 0 { 7.0 } else { 99.0 };
            assert!((v - expected).abs() < 1e-10, "cell {k}: got {v}");
        }
    }

    /// Disjoint meshes produce only default values.
    #[test]
    fn transfer3d_no_overlap() {
        let source = source3d_with_field(nd::array![7.0].into_dyn());
        let target = RegularUMeshBuilder::new()
            .add_axis(vec![2.0, 3.0])
            .add_axis(vec![2.0, 3.0])
            .add_axis(vec![2.0, 3.0])
            .build();
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let field = op.apply(&field_view_3d(&source), FieldNature::Intensive, 99.0);
        assert!(field.0[&ElementType::HEX8].iter().all(|&v| v == 99.0));
    }

    /// A tetrahedral source cell clips the target cells along the diagonal plane.
    #[test]
    fn transfer3d_tetrahedral_source() {
        let mut source = UMesh::new(
            nd::ArcArray2::from_shape_vec(
                (4, 3),
                vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            )
            .unwrap(),
        );
        source.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([(
            ElementType::TET4,
            nd::array![7.0].into_dyn(),
        )]));
        source.update_field("f", field.into_shared());

        let target = me::make_imesh_3d(4);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let out = op.apply(&field_view_3d(&source), FieldNature::Intensive, 0.0);
        let arr = &out.0[&ElementType::HEX8];
        for (k, &v) in arr.iter().enumerate() {
            let (i, j, l) = (k % 4, (k / 4) % 4, k / 16);
            let s = i + j + l;
            if s <= 1 {
                assert!(
                    (v - 7.0).abs() < 1e-12,
                    "cell ({i}, {j}, {l}) fully covered: {v}"
                );
            } else if s >= 4 {
                assert_eq!(v, 0.0, "cell ({i}, {j}, {l}) should be outside");
            } else {
                assert!(
                    (0.0..7.0).contains(&v),
                    "cell ({i}, {j}, {l}) partially covered: {v}"
                );
            }
        }
    }

    /// Source cells of several 3D element types are correctly addressed by the overlap matrix.
    #[test]
    fn transfer3d_mixed_source_types() {
        let corners: Vec<[f64; 3]> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
        ];
        let coords: Vec<f64> = corners.iter().copied().flatten().collect();

        let mut source =
            UMesh::new(nd::ArcArray2::from_shape_vec((12, 3), coords.clone()).unwrap());
        source.add_regular_block(
            ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        source.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[8, 9, 10, 11]]).to_shared(),
            None,
        );
        let field = FieldOwnedD::new(BTreeMap::from([
            (ElementType::HEX8, nd::array![10.0].into_dyn()),
            (ElementType::TET4, nd::array![20.0].into_dyn()),
        ]));
        source.update_field("f", field.into_shared());

        // Target cells exactly matching each region: an HEX8 cell that overlaps the source cube
        // (volume 1) and a TET4 cell that overlaps the source tetrahedron (volume 1/6).
        let mut target = UMesh::new(nd::ArcArray2::from_shape_vec((12, 3), coords).unwrap());
        target.add_regular_block(
            ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        target.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[8, 9, 10, 11]]).to_shared(),
            None,
        );

        let src_view = field_view_3d(&source);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());

        let ext = op.apply(&src_view, FieldNature::Extensive, 0.0);
        assert!((ext.0[&ElementType::HEX8][0] - 10.0).abs() < 1e-12);
        assert!((ext.0[&ElementType::TET4][0] - 20.0 / 6.0).abs() < 1e-12);

        let int = op.apply(&src_view, FieldNature::Intensive, 0.0);
        assert!((int.0[&ElementType::HEX8][0] - 10.0).abs() < 1e-12);
        assert!((int.0[&ElementType::TET4][0] - 20.0).abs() < 1e-12);
    }

    /// The same 3D operator can evaluate several fields without rebuilding the precompute.
    #[test]
    fn transfer3d_reuse() {
        let mut source = me::make_imesh_3d(1);
        let f1 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::array![7.0].into_dyn(),
        )]));
        let f2 = FieldOwnedD::new(BTreeMap::from([(
            ElementType::HEX8,
            nd::array![3.0].into_dyn(),
        )]));
        source.update_field("f1", f1.into_shared());
        source.update_field("f2", f2.into_shared());
        let target = me::make_imesh_3d(2);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let r1 = op.apply(
            &source.field("f1", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        let r2 = op.apply(
            &source.field("f2", Some(Dimension::D3)).unwrap(),
            FieldNature::Intensive,
            0.0,
        );
        assert!(
            r1.0[&ElementType::HEX8]
                .iter()
                .all(|&v| (v - 7.0).abs() < 1e-10)
        );
        assert!(
            r2.0[&ElementType::HEX8]
                .iter()
                .all(|&v| (v - 3.0).abs() < 1e-10)
        );
    }

    /// `apply_update` stores the transferred field on the 3D target mesh.
    #[test]
    fn transfer3d_apply_update() {
        let source = source3d_with_field(nd::array![7.0].into_dyn());
        let mut target = me::make_imesh_3d(2);
        let op = ConservativeP0Transfer::new(&source.view(), &target.view());
        let old = op.apply_update(
            &mut target,
            "transferred",
            &field_view_3d(&source),
            FieldNature::Intensive,
            0.0,
        );
        assert!(old.is_none());
        let field = target.field("transferred", Some(Dimension::D3)).unwrap();
        assert!(
            field.0[&ElementType::HEX8]
                .iter()
                .all(|&v| (v - 7.0).abs() < 1e-10)
        );
    }

    /// A lower-dimensional source mesh in 3D space fails with a clear message.
    #[test]
    #[should_panic(expected = "Source mesh should be full-dimensional")]
    fn transfer3d_source_not_full_dimensional_panics() {
        let mut source = UMesh::new(nd::ArcArray2::from_shape_vec((4, 3), vec![0.0; 12]).unwrap());
        source.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );
        let target = me::make_imesh_3d(1);
        let _ = ConservativeP0Transfer::new(&source.view(), &target.view());
    }

    /// Feeding meshes with different space dimensions fails with a clear message.
    #[test]
    #[should_panic(expected = "same space dimension")]
    fn transfer_space_dim_mismatch_panics() {
        let source = me::make_imesh_2d(1);
        let target = me::make_imesh_3d(1);
        let _ = ConservativeP0Transfer::new(&source.view(), &target.view());
    }

    /// An empty source mesh fails with a clear message.
    #[test]
    #[should_panic(expected = "Source mesh should not be empty")]
    fn transfer_empty_source_panics() {
        let source = UMesh::new(nd::ArcArray2::from_shape_vec((0, 2), vec![]).unwrap());
        let target = me::make_imesh_2d(2);
        let _ = ConservativeP0Transfer::new(&source.view(), &target.view());
    }

    /// A lower-dimensional target mesh fails with a clear message.
    #[test]
    #[should_panic(expected = "Target mesh should be full-dimensional")]
    fn transfer_target_not_full_dimensional_panics() {
        let source = me::make_imesh_2d(1);
        let mut target =
            UMesh::new(nd::ArcArray2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap());
        target.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        let _ = ConservativeP0Transfer::new(&source.view(), &target.view());
    }
}
