//! Coordinate transforms, mesh duplication and concatenation.
//!
//! Everything is built on a single [`Transform`] value: a homogeneous 4x4 matrix.
//! Simple operations (translation, rotation, scaling, mirroring) are convenience
//! constructors that produce such a matrix; any composition is a matrix product,
//! so arbitrarily complex transformations can be assembled out of simple ones.
//!
//! Applying a transform maps every node coordinate once and keeps all topology,
//! fields, families and groups untouched (a "coordinates-only" operation, like
//! VTK cell data).
//!
//! # Simple usage
//!
//! ```rust,ignore
//! let moved = mesh.translate(&[1.0, 0.0, 0.0])?;
//! let rotated = mesh.rotate(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)?;
//! let mirrored = mesh.mirror(&[0.0, 0.0, 1.0])?;
//! ```
//!
//! # Composed usage
//!
//! ```rust,ignore
//! let tr = Transform::translation(&[1.0, 0.0, 0.0])?
//!     * Transform::rotation(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)?
//!     * Transform::scaling(&[2.0, 2.0, 2.0])?;
//! let moved = mesh.transform(&tr)?;
//! ```
//!
//! # Conventions
//!
//! - Coordinates are column vectors: `p' = M * [p; 1]` with `p = [x, y, z, 1]`.
//! - `Transform::rotation` takes an angle in **radians** and rotates around an axis
//!   through the origin; [`Transform::rotation_about`] centers the rotation anywhere.
//! - `a * b` is the matrix product of the two transforms ("apply `b` first");
//!   [`Transform::then`] chains in the opposite, left-to-right reading.
//! - 1D and 2D meshes embed naturally in 3D: a transform applied to a lower
//!   dimensional mesh must leave the extra axes at rest (e.g. a 2D mesh only
//!   accepts rotations about the z-axis). Transforms that would move a lower
//!   dimensional mesh out of its plane/line are rejected with an error.
//!
//! Meshes may be joined without intersection checks with [`aggregate`] / [`concat`],
//! and duplicated into several copies with [`duplicate`].

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::mesh::{
    ArcGroups, Connectivity, ConnectivityView, Dimension, ElementBlock, ElementType, UMesh,
    UMeshView,
};
use nalgebra as na;
use ndarray as nd;

/// Tolerance under which a value is treated as exactly `0` in validation checks.
const EPS: f64 = 1e-12;

// ----------------------------------------------------------------------------
// Transform
// ----------------------------------------------------------------------------

/// An affine transform encoded as a homogeneous 4x4 matrix (row-major, last row
/// `[0, 0, 0, 1]`), applied to column coordinates: `p' = M * [p; 1]`.
///
/// This is a value type: transforms are composed with `*` (matrix product) or
/// chained with [`Transform::then`], and applied to a mesh with
/// [`Transformable::transform`].
#[derive(Clone, Debug, PartialEq)]
pub struct Transform {
    mat: nd::Array2<f64>,
}

/// Pads a 1-, 2- or 3-element slice to a fixed-length 3-vector, filling the
/// missing trailing components with `default`.
fn pad(v: &[f64], default: f64) -> Result<[f64; 3], String> {
    if v.iter().any(|value| !value.is_finite()) {
        return Err("Vector components must be finite.".to_string());
    }
    match v {
        [x] => Ok([*x, default, default]),
        [x, y] => Ok([*x, *y, default]),
        [x, y, z] => Ok([*x, *y, *z]),
        _ => Err(format!("Expected 1, 2 or 3 components, got {}.", v.len())),
    }
}

fn normalized_direction(v: [f64; 3], name: &str) -> Result<na::Vector3<f64>, String> {
    let vector = na::Vector3::from_row_slice(&v);
    let scale = v.iter().map(|value| value.abs()).fold(0.0, f64::max);
    if scale == 0.0 {
        return Err(format!("Zero-length {name}."));
    }
    let scaled = vector / scale;
    Ok(scaled / scaled.norm())
}

/// Builds a homogeneous 4x4 matrix out of a `d`-dimensional linear part and a
/// translation, keeping the extra axes (and the homogeneous row) at rest.
fn embed(d: usize, lin: &nd::Array2<f64>, t: [f64; 3]) -> Transform {
    let mut mat = nd::Array2::eye(4);
    mat.slice_mut(nd::s![..d, ..d]).assign(lin);
    for i in 0..d {
        mat[[i, 3]] = t[i];
    }
    Transform { mat }
}

impl Transform {
    /// The identity transform.
    pub fn identity() -> Self {
        Transform {
            mat: nd::Array2::eye(4),
        }
    }

    /// A transform from an explicit 4x4 homogeneous matrix. Errors if the shape
    /// is not 4x4 or the last row is not `[0, 0, 0, 1]`.
    pub fn from_matrix(mat: nd::ArrayView2<f64>) -> Result<Self, String> {
        if mat.shape() != [4, 4] {
            return Err(format!(
                "Expected a 4x4 homogeneous matrix, got {:?}.",
                mat.shape()
            ));
        }
        if mat.iter().any(|value| !value.is_finite()) {
            return Err("Transform matrix components must be finite.".to_string());
        }
        let last = [mat[[3, 0]], mat[[3, 1]], mat[[3, 2]], mat[[3, 3]]];
        if last.iter().take(3).any(|v| v.abs() > EPS) || (last[3] - 1.0).abs() > EPS {
            return Err("The last matrix row must be [0, 0, 0, 1].".to_string());
        }
        Ok(Transform {
            mat: mat.to_owned(),
        })
    }

    /// Yields the 4x4 homogeneous matrix backing this transform.
    pub fn matrix(&self) -> nd::ArrayView2<'_, f64> {
        self.mat.view()
    }

    /// Translation by a 1-, 2- or 3-component vector.
    pub fn translation(v: &[f64]) -> Result<Self, String> {
        let [tx, ty, tz] = pad(v, 0.0)?;
        let lin = nd::Array2::eye(3);
        Ok(embed(3, &lin, [tx, ty, tz]))
    }

    /// Non-uniform scaling by 1-, 2- or 3-component factors (missing axes are
    /// left unchanged, i.e. scaled by 1).
    pub fn scaling(factors: &[f64]) -> Result<Self, String> {
        let [sx, sy, sz] = pad(factors, 1.0)?;
        let lin = nd::Array2::from_diag(&nd::array![sx, sy, sz]);
        Ok(embed(3, &lin, [0.0, 0.0, 0.0]))
    }

    /// Rotation by `angle` (radians) around an axis through the origin.
    ///
    /// The axis is padded so that 1- and 2-component slices (e.g. `[0, 0, 1]`)
    /// work as expected.
    pub fn rotation(axis: &[f64], angle: f64) -> Result<Self, String> {
        let [ax, ay, az] = pad(axis, 0.0)?;
        if !angle.is_finite() {
            return Err("Rotation angle must be finite.".to_string());
        }
        let u = normalized_direction([ax, ay, az], "rotation axis")?;
        // Rodrigues: R = cos(θ)·I + sin(θ)·[u]_× + (1 - cos(θ))·u·uᵀ
        let c = angle.cos();
        let s = angle.sin();
        let k = na::Matrix3::new(0.0, -u.z, u.y, u.z, 0.0, -u.x, -u.y, u.x, 0.0);
        let rot = c * na::Matrix3::identity() + s * k + (1.0 - c) * (u * u.transpose());
        let lin = nd::Array2::from_shape_fn((3, 3), |(r, c)| rot[(r, c)]);
        Ok(embed(3, &lin, [0.0, 0.0, 0.0]))
    }

    /// Rotation by `angle` (radians) around an axis passing through `center`.
    pub fn rotation_about(center: &[f64], axis: &[f64], angle: f64) -> Result<Self, String> {
        let [cx, cy, cz] = pad(center, 0.0)?;
        let t = Self::translation(center)?;
        let r = Self::rotation(axis, angle)?;
        let back = Self::translation(&[-cx, -cy, -cz])?;
        Ok(t * r * back)
    }

    /// Reflection (symmetry) through a plane passing through the origin with the
    /// given normal.
    pub fn reflection(normal: &[f64]) -> Result<Self, String> {
        let [nx, ny, nz] = pad(normal, 0.0)?;
        let n = normalized_direction([nx, ny, nz], "reflection normal")?;
        let lin = na::Matrix3::identity() - 2.0 * (n * n.transpose());
        let lin = nd::Array2::from_shape_fn((3, 3), |(r, c)| lin[(r, c)]);
        Ok(embed(3, &lin, [0.0, 0.0, 0.0]))
    }

    /// Reflection through a plane passing through `point` with the given normal.
    pub fn reflection_about(point: &[f64], normal: &[f64]) -> Result<Self, String> {
        let [x, y, z] = pad(point, 0.0)?;
        let t = Self::translation(point)?;
        let r = Self::reflection(normal)?;
        let back = Self::translation(&[-x, -y, -z])?;
        Ok(t * r * back)
    }

    /// Composes this transform with `after`: the result applies `self` first,
    /// then `after`. This is the "left-to-right" reading of a sequence.
    pub fn then(&self, after: &Transform) -> Transform {
        Transform {
            mat: after.mat.dot(&self.mat),
        }
    }

    /// The inverse affine transform, or an error if the linear part is singular.
    pub fn inverse(&self) -> Result<Self, String> {
        if self.mat.iter().any(|value| !value.is_finite()) {
            return Err("Transform matrix components must be finite.".to_string());
        }
        let lin = na::Matrix3::from_fn(|r, c| self.mat[[r, c]]);
        let lin_inv = lin
            .try_inverse()
            .ok_or_else(|| "Transform is not invertible (singular linear part).".to_string())?;
        let t = na::Vector3::new(self.mat[[0, 3]], self.mat[[1, 3]], self.mat[[2, 3]]);
        let t_inv = -lin_inv * t;
        if t_inv.iter().any(|value| !value.is_finite())
            || lin_inv.iter().any(|value| !value.is_finite())
        {
            return Err("Transform inverse is not finite.".to_string());
        }
        let lin_inv = nd::Array2::from_shape_fn((3, 3), |(r, c)| lin_inv[(r, c)]);
        Ok(embed(3, &lin_inv, [t_inv[0], t_inv[1], t_inv[2]]))
    }
}

impl std::ops::Mul for Transform {
    type Output = Transform;
    fn mul(self, rhs: Transform) -> Transform {
        Transform {
            mat: self.mat.dot(&rhs.mat),
        }
    }
}

impl std::ops::Mul<&Transform> for &Transform {
    type Output = Transform;
    fn mul(self, rhs: &Transform) -> Transform {
        Transform {
            mat: self.mat.dot(&rhs.mat),
        }
    }
}

// ----------------------------------------------------------------------------
// Coordinate mapping
// ----------------------------------------------------------------------------

/// Applies a homogeneous matrix to all coordinates of a mesh.
///
/// For meshes with fewer than 3 coordinates per node, the matrix is required to
/// keep the extra axes at rest (see module-level docs).
fn map_coords(
    coords: nd::ArrayView2<f64>,
    mat: &nd::Array2<f64>,
) -> Result<nd::Array2<f64>, String> {
    let d = coords.ncols();
    if coords.iter().any(|value| !value.is_finite()) {
        return Err("Mesh coordinates must be finite.".to_string());
    }
    if mat.iter().any(|value| !value.is_finite()) {
        return Err("Transform matrix components must be finite.".to_string());
    }
    let result = match d {
        3 => Ok(apply_affine(coords, mat, 3)),
        2 => {
            check_extra_axes_at_rest(mat, 2)?;
            Ok(apply_affine(coords, mat, 2))
        }
        1 => {
            check_extra_axes_at_rest(mat, 1)?;
            Ok(apply_affine(coords, mat, 1))
        }
        0 => Err("Cannot apply a transform to a mesh with 0 coordinates per node.".to_string()),
        c => Err(format!("Unsupported space dimension {c}.")),
    }?;
    if result.iter().any(|value| !value.is_finite()) {
        return Err("Transformed coordinates are not finite.".to_string());
    }
    Ok(result)
}

/// Verifies that rows `d..3` of the matrix leave the extra coordinates `d..3`
/// of any input point unchanged (i.e. they never move a lower-dimensional mesh
/// out of its plane/line).
fn check_extra_axes_at_rest(mat: &nd::Array2<f64>, d: usize) -> Result<(), String> {
    for r in d..3 {
        for c in 0..d {
            if mat[[r, c]].abs() > EPS || mat[[r, 3]].abs() > EPS {
                return Err(
                    "This transform moves the mesh out of its plane/line: rows beyond the \
                     space dimension must leave the extra coordinates unchanged."
                        .to_string(),
                );
            }
        }
    }
    Ok(())
}

/// `p' = L * p + t` with `L = mat[..d, ..d]` and `t = mat[..d, 3]`.
fn apply_affine(coords: nd::ArrayView2<f64>, mat: &nd::Array2<f64>, d: usize) -> nd::Array2<f64> {
    let lin = mat.slice(nd::s![..d, ..d]);
    let t = mat.slice(nd::s![..d, 3]);
    coords.dot(&lin.t()) + t
}

// ----------------------------------------------------------------------------
// Coordinate-level operations
// ----------------------------------------------------------------------------

/// Returns an owned mesh with the same topology, fields, families and groups as
/// `mesh` but a brand-new coordinate array given by the user.
///
/// This is the escape hatch for user-provided coordinates (e.g. an analytic
/// deformation): anything that produces a same-shaped array works.
pub fn set_coords(mesh: &UMeshView, coords: nd::ArrayView2<f64>) -> Result<UMesh, String> {
    if coords.shape() != mesh.coords().shape() {
        return Err(format!(
            "New coordinates shape {:?} do not match the mesh coordinates shape {:?}.",
            coords.shape(),
            mesh.coords().shape()
        ));
    }
    if coords.iter().any(|value| !value.is_finite()) {
        return Err("New coordinates must be finite.".to_string());
    }
    let mesh = mesh.to_owned_with_coords(coords.to_owned().into_shared());
    mesh.validate_structure()?;
    Ok(mesh)
}

/// Returns an owned mesh whose coordinates are `f(mesh.coords())`.
///
/// This accepts arbitrary user-provided transformations; the returned array must
/// have exactly the same shape as the input coordinates.
pub fn transform_coords<F>(mesh: &UMeshView, f: F) -> Result<UMesh, String>
where
    F: Fn(nd::ArrayView2<f64>) -> nd::Array2<f64>,
{
    let new_coords = f(mesh.coords());
    if new_coords.shape() != mesh.coords().shape() {
        return Err(format!(
            "The coordinate function returned shape {:?}; expected {:?}.",
            new_coords.shape(),
            mesh.coords().shape()
        ));
    }
    if new_coords.iter().any(|value| !value.is_finite()) {
        return Err("The coordinate function returned non-finite coordinates.".to_string());
    }
    let mesh = mesh.to_owned_with_coords(new_coords.into_shared());
    mesh.validate_structure()?;
    Ok(mesh)
}

/// Applies a [`Transform`] to all node coordinates of `mesh`, keeping topology,
/// fields, families and groups untouched.
pub fn transform(mesh: &UMeshView, tr: &Transform) -> Result<UMesh, String> {
    let new_coords = map_coords(mesh.coords(), &tr.mat)?.into_shared();
    let result = mesh.to_owned_with_coords(new_coords);
    result.validate_structure()?;
    Ok(result)
}

// ----------------------------------------------------------------------------
// Concatenation
// ----------------------------------------------------------------------------

/// Joins several non-intersecting meshes sharing the same space dimension into a
/// single mesh: coordinates are concatenated and each element block becomes the
/// union of the per-mesh blocks of the same element type.
///
/// - Nodes of each contribution are shifted by the number of preceding nodes.
/// - Fields of a given (name, element type) must be present on every mesh that
///   carries that element type, with identical trailing dimensions; values are
///   concatenated element-wise.
/// - Groups are rebuilt element-wise: an element of the result belongs to the
///   same groups as the source element.
///
/// This is a topology-agnostic operation: meshes are *not* checked for
/// intersection or shared boundaries.
pub fn aggregate(meshes: &[UMeshView]) -> Result<UMesh, String> {
    aggregate_with_family_mode(meshes, FamilyMode::Relabel)
}

fn aggregate_with_family_mode(
    meshes: &[UMeshView],
    family_mode: FamilyMode,
) -> Result<UMesh, String> {
    if meshes.is_empty() {
        return Err("aggregate requires at least one mesh.".to_string());
    }
    let space_dim = meshes[0].coords().ncols();
    for (i, m) in meshes.iter().enumerate() {
        if m.coords().ncols() != space_dim {
            return Err(format!(
                "Space dimension mismatch: mesh 0 has space dimension {space_dim}, mesh {i} \
                 has {}.",
                m.coords().ncols()
            ));
        }
    }

    let coords_views: Vec<nd::ArrayView2<f64>> = meshes.iter().map(|m| m.coords()).collect();
    let coords = nd::concatenate(nd::Axis(0), &coords_views).map_err(|e| e.to_string())?;
    let mut out = UMesh::new(coords.into_shared());

    let mut node_offsets: Vec<usize> = Vec::with_capacity(meshes.len());
    let mut acc = 0;
    for m in meshes {
        node_offsets.push(acc);
        acc += m.coords().nrows();
    }

    let field_names = aggregate_field_names(meshes)?;
    let etypes: BTreeSet<ElementType> = meshes
        .iter()
        .flat_map(|m| m.element_types().copied())
        .collect();
    for et in etypes {
        aggregate_block(
            &mut out,
            meshes,
            &node_offsets,
            et,
            &field_names,
            family_mode,
        )?;
    }
    out.validate_structure()?;
    Ok(out)
}

fn aggregate_block(
    out: &mut UMesh,
    meshes: &[UMeshView],
    node_offsets: &[usize],
    et: ElementType,
    field_names: &BTreeMap<Dimension, BTreeSet<String>>,
    family_mode: FamilyMode,
) -> Result<(), String> {
    let source: Vec<Option<&crate::mesh::ElementBlockView>> =
        meshes.iter().map(|m| m.block(et)).collect();
    let first = source
        .iter()
        .flatten()
        .next()
        .expect("element type comes from the meshes");

    let fields = aggregate_fields(meshes, et, field_names)?;
    let (families, groups) = aggregate_families(meshes, et, family_mode);

    match &first.connectivity {
        ConnectivityView::Regular(first_conn) => {
            let mut total = 0;
            for b in source.iter().flatten() {
                let ConnectivityView::Regular(c) = &b.connectivity else {
                    return Err(format!(
                        "Element type {et:?} is regular in some meshes and polygonal in others."
                    ));
                };
                if c.ncols() != first_conn.ncols() {
                    return Err(format!(
                        "Element type {et:?} has incompatible connectivity width {} vs {}.",
                        c.ncols(),
                        first_conn.ncols()
                    ));
                }
                total += c.nrows();
            }
            let mut conn = nd::Array2::zeros((total, first_conn.ncols()));
            let mut row = 0;
            for (b, &off) in source.iter().zip(node_offsets) {
                if let Some(b) = b {
                    let ConnectivityView::Regular(c) = &b.connectivity else {
                        unreachable!("already checked")
                    };
                    let mut dst = conn.slice_mut(nd::s![row..row + c.nrows(), ..]);
                    for (mut drow, src) in dst.rows_mut().into_iter().zip(c.rows()) {
                        for (dd, &s) in drow.iter_mut().zip(src.iter()) {
                            *dd = s + off;
                        }
                    }
                    row += c.nrows();
                }
            }
            let block = ElementBlock::new_with_metadata(
                et,
                Connectivity::Regular(conn.into_shared()),
                families,
                fields,
                ArcGroups(Arc::new(groups)),
            );
            out.insert_block(block);
        }
        ConnectivityView::Poly(_) => {
            let mut data: Vec<usize> = Vec::new();
            let mut offsets: Vec<usize> = Vec::new();
            for (b, &off) in source.iter().zip(node_offsets) {
                if let Some(b) = b {
                    let ConnectivityView::Poly(c) = &b.connectivity else {
                        return Err(format!(
                            "Element type {et:?} is regular in some meshes and polygonal in \
                             others."
                        ));
                    };
                    let base = data.len();
                    data.extend(c.data.iter().map(|&v| v + off));
                    offsets.extend(c.offsets.iter().map(|&o| base + o));
                }
            }
            let block = ElementBlock::new_with_metadata(
                et,
                Connectivity::new_poly(
                    nd::Array1::from_vec(data).into_shared(),
                    nd::Array1::from_vec(offsets).into_shared(),
                ),
                families,
                fields,
                ArcGroups(Arc::new(groups)),
            );
            out.insert_block(block);
        }
    }
    Ok(())
}

fn aggregate_field_names(
    meshes: &[UMeshView],
) -> Result<BTreeMap<Dimension, BTreeSet<String>>, String> {
    let mut names_by_dim = BTreeMap::<Dimension, BTreeSet<String>>::new();
    for mesh in meshes {
        for (&et, block) in mesh.blocks() {
            names_by_dim
                .entry(et.dimension())
                .or_default()
                .extend(block.fields.keys().cloned());
        }
    }

    for (&dim, names) in &names_by_dim {
        for name in names {
            let mut reference = None;
            for mesh in meshes {
                for (&et, block) in mesh.blocks() {
                    if et.dimension() != dim {
                        continue;
                    }
                    let field = block.fields.get(name).ok_or_else(|| {
                        format!(
                            "Field '{name}' is missing on element type {et:?} of dimension {dim:?} \
                             in one of the aggregated meshes."
                        )
                    })?;
                    if field.ndim() == 0 {
                        return Err(format!(
                            "Field '{name}' on element type {et:?} must have an element axis."
                        ));
                    }
                    if field.shape()[0] != block.len() {
                        return Err(format!(
                            "Field '{name}' on element type {et:?} has {} values for {} elements.",
                            field.shape()[0],
                            block.len()
                        ));
                    }
                    let trailing = field.shape()[1..].to_vec();
                    if let Some(expected) = &reference {
                        if *expected != trailing {
                            return Err(format!(
                                "Field '{name}' has incompatible trailing shapes across element \
                                 types of dimension {dim:?}."
                            ));
                        }
                    } else {
                        reference = Some(trailing);
                    }
                }
            }
        }
    }
    Ok(names_by_dim)
}

/// Concatenates the values of every field present on the requested topological
/// dimension, requiring one compatible schema on every block of that dimension.
fn aggregate_fields(
    meshes: &[UMeshView],
    et: ElementType,
    field_names: &BTreeMap<Dimension, BTreeSet<String>>,
) -> Result<BTreeMap<String, nd::ArcArray<f64, nd::IxDyn>>, String> {
    let mut fields = BTreeMap::new();
    let names = field_names
        .get(&et.dimension())
        .cloned()
        .unwrap_or_default();
    for name in names {
        let mut parts: Vec<nd::ArrayViewD<f64>> = Vec::new();
        let mut reference: Option<Vec<usize>> = None;
        for mesh in meshes {
            if let Some(block) = mesh.block(et) {
                let field = block
                    .fields
                    .get(&name)
                    .ok_or_else(|| format!("Field '{name}' is missing on element type {et:?}."))?;
                if let Some(expected) = &reference {
                    if field.shape()[1..] != *expected {
                        return Err(format!(
                            "Field '{name}' has incompatible trailing shape {:?} (expected \
                             {:?}).",
                            &field.shape()[1..],
                            expected
                        ));
                    }
                } else {
                    reference = Some(field.shape()[1..].to_vec());
                }
                parts.push(field.view());
            }
        }
        let array = nd::concatenate(nd::Axis(0), &parts).map_err(|e| e.to_string())?;
        fields.insert(name.clone(), array.into_shared());
    }
    Ok(fields)
}

#[derive(Clone, Copy)]
enum FamilyMode {
    Relabel,
    Preserve,
}

fn aggregate_families(
    meshes: &[UMeshView],
    et: ElementType,
    mode: FamilyMode,
) -> (nd::ArcArray1<usize>, BTreeMap<String, BTreeSet<usize>>) {
    let mut families: Vec<usize> = Vec::new();
    let mut groups: BTreeMap<String, BTreeSet<usize>> = BTreeMap::new();
    let mut family_offset = 0;
    for mesh in meshes {
        if let Some(block) = mesh.block(et) {
            let shift = match mode {
                FamilyMode::Relabel => family_offset,
                FamilyMode::Preserve => 0,
            };
            families.extend(block.families().iter().map(|&family| family + shift));
            for (name, family_ids) in block.groups().iter() {
                groups
                    .entry(name.clone())
                    .or_default()
                    .extend(family_ids.iter().map(|&family| family + shift));
            }
            if matches!(mode, FamilyMode::Relabel) {
                family_offset =
                    family_offset + block.families().iter().copied().max().unwrap_or(0) + 1;
            }
        }
    }
    (nd::Array1::from_vec(families).into_shared(), groups)
}

/// Joins exactly two non-intersecting meshes; see [`aggregate`].
pub fn concat(a: &UMeshView, b: &UMeshView) -> Result<UMesh, String> {
    let views: [UMeshView; 2] = [a.view(), b.view()];
    aggregate(&views)
}

pub fn from_mesh(
    source: &UMeshView,
    coords: Option<nd::ArrayView2<'_, f64>>,
    selected: Option<&BTreeSet<ElementType>>,
) -> Result<UMesh, String> {
    if let Some(selected) = selected {
        if selected.is_empty() {
            return Err("Element type selection cannot be empty.".to_string());
        }
        if let Some(element_type) = selected
            .iter()
            .find(|element_type| source.block(**element_type).is_none())
        {
            return Err(format!(
                "Element type {element_type:?} is not present in the source mesh."
            ));
        }
    }

    let coords = match coords {
        Some(coords) => {
            if coords.shape() != source.coords().shape() {
                return Err(format!(
                    "Coordinates shape {:?} does not match source mesh shape {:?}.",
                    coords.shape(),
                    source.coords().shape()
                ));
            }
            coords.to_owned().into_shared()
        }
        None => source.coords().to_owned().into_shared(),
    };
    let mesh = source.to_owned_with_coords_and_types(coords, selected);
    mesh.validate_structure()?;
    Ok(mesh)
}

// ----------------------------------------------------------------------------
// Duplication
// ----------------------------------------------------------------------------

/// Returns a mesh made of `n` copies of `mesh`, each further transformed by the
/// powers `step`, `step^2`, ..., `step^(n-1)` of `step` (the first copy is
/// untransformed). This produces evenly spaced or repeated patterns: pass a pure
/// [`Transform::translation`] for a row of identical copies, or a composed
/// translation-and-rotation for a screw-like repetition.
///
/// Every copy keeps the fields, families and groups of the original mesh.
pub fn duplicate(mesh: &UMeshView, step: &Transform, n: usize) -> Result<UMesh, String> {
    if n == 0 {
        return Err("duplicate requires at least one copy (n >= 1).".to_string());
    }
    let mut current = Transform::identity();
    let mut copies: Vec<UMesh> = Vec::with_capacity(n);
    for _ in 0..n {
        copies.push(transform(mesh, &current)?);
        current = current.then(step);
    }
    let views: Vec<UMeshView> = copies.iter().map(|m| m.view()).collect();
    aggregate_with_family_mode(&views, FamilyMode::Preserve)
}

// ----------------------------------------------------------------------------
// Transformable
// ----------------------------------------------------------------------------

/// Convenience methods for mesh coordinate transforms.
///
/// All methods return a new mesh (the input is never mutated) that shares its
/// topology, fields, families and groups with the input. Angles are in radians.
pub trait Transformable {
    /// Applies an arbitrary [`Transform`].
    fn transform(&self, tr: &Transform) -> Result<UMesh, String>;

    /// Translation by a 1-, 2- or 3-component vector.
    fn translate(&self, v: &[f64]) -> Result<UMesh, String> {
        self.transform(&Transform::translation(v)?)
    }

    /// Non-uniform scaling by 1-, 2- or 3-component factors.
    fn scale(&self, factors: &[f64]) -> Result<UMesh, String> {
        self.transform(&Transform::scaling(factors)?)
    }

    /// Uniform scaling.
    fn scale_uniform(&self, factor: f64) -> Result<UMesh, String> {
        self.transform(&Transform::scaling(&[factor, factor, factor])?)
    }

    /// Rotation by `angle` (radians) around an axis through the origin.
    fn rotate(&self, axis: &[f64], angle: f64) -> Result<UMesh, String> {
        self.transform(&Transform::rotation(axis, angle)?)
    }

    /// Rotation by `angle` (radians) around an axis through `center`.
    fn rotate_about(&self, center: &[f64], axis: &[f64], angle: f64) -> Result<UMesh, String> {
        self.transform(&Transform::rotation_about(center, axis, angle)?)
    }

    /// Reflection (symmetry) through a plane through the origin with the given
    /// normal.
    fn mirror(&self, normal: &[f64]) -> Result<UMesh, String> {
        self.transform(&Transform::reflection(normal)?)
    }

    /// Reflection through a plane through `point` with the given normal.
    fn mirror_about(&self, point: &[f64], normal: &[f64]) -> Result<UMesh, String> {
        self.transform(&Transform::reflection_about(point, normal)?)
    }

    /// `n` copies of this mesh with successive transforms `step`, `step^2`, ...
    fn duplicate(&self, step: &Transform, n: usize) -> Result<UMesh, String>;
}

impl Transformable for UMesh {
    fn transform(&self, tr: &Transform) -> Result<UMesh, String> {
        let new_coords = map_coords(self.coords(), &tr.mat)?.into_shared();
        let result = self.with_coords(new_coords);
        result.validate_structure()?;
        Ok(result)
    }

    /// `n` copies of this mesh with successive transforms `step`, `step^2`, ...
    fn duplicate(&self, step: &Transform, n: usize) -> Result<UMesh, String> {
        duplicate(&self.view(), step, n)
    }
}

impl Transformable for UMeshView<'_> {
    fn transform(&self, tr: &Transform) -> Result<UMesh, String> {
        let new_coords = map_coords(self.coords(), &tr.mat)?.into_shared();
        let result = self.to_owned_with_coords(new_coords);
        result.validate_structure()?;
        Ok(result)
    }

    /// `n` copies of this mesh with successive transforms `step`, `step^2`, ...
    fn duplicate(&self, step: &Transform, n: usize) -> Result<UMesh, String> {
        duplicate(self, step, n)
    }
}

// ----------------------------------------------------------------------------
// In-place helpers
// ----------------------------------------------------------------------------

impl UMesh {
    /// Replaces the coordinates of this mesh in place, keeping topology, fields,
    /// families and groups untouched. The new coordinates must have exactly the
    /// same shape as the current ones.
    pub fn set_coordinates(&mut self, coords: nd::ArrayView2<f64>) -> Result<(), String> {
        if coords.shape() != self.coords().shape() {
            return Err(format!(
                "New coordinates shape {:?} do not match the mesh coordinates shape {:?}.",
                coords.shape(),
                self.coords().shape()
            ));
        }
        if coords.iter().any(|value| !value.is_finite()) {
            return Err("New coordinates must be finite.".to_string());
        }
        self.validate_structure()?;
        let mut own = std::mem::take(&mut self.coords).into_owned();
        own.assign(&coords);
        self.coords = own.into_shared();
        Ok(())
    }

    /// Applies a [`Transform`] in place (updates the coordinates of this mesh,
    /// keeping topology, fields, families and groups untouched).
    pub fn transform_coordinates(&mut self, tr: &Transform) -> Result<(), String> {
        let new_coords = map_coords(self.coords(), &tr.mat)?;
        self.validate_structure()?;
        let mut own = std::mem::take(&mut self.coords).into_owned();
        own.assign(&new_coords);
        self.coords = own.into_shared();
        Ok(())
    }
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples::make_mesh_2d_quad;

    fn quad() -> UMesh {
        make_mesh_2d_quad()
    }

    #[test]
    fn test_from_mesh_preserves_source_and_selects_blocks() {
        let source = crate::mesh_examples::make_mesh_2d_multi();
        let original = source.coords().to_owned();
        let copy = from_mesh(&source.view(), None, None).unwrap();
        assert_eq!(copy.coords(), original);
        assert_eq!(copy.element_types().count(), source.element_types().count());

        let changed =
            nd::arr2(&[[9.0, 9.0], [8.0, 8.0], [7.0, 7.0], [6.0, 6.0], [5.0, 5.0]]).into_shared();
        let changed_copy = from_mesh(&source.view(), Some(changed.view()), None).unwrap();
        assert_eq!(changed_copy.coords(), changed);
        assert_eq!(source.coords(), original);

        let selected = BTreeSet::from([ElementType::QUAD4]);
        let selected_copy = from_mesh(&source.view(), None, Some(&selected)).unwrap();
        assert_eq!(
            selected_copy.element_types().collect::<Vec<_>>(),
            vec![&ElementType::QUAD4]
        );
        assert!(selected_copy.block(ElementType::SEG2).is_none());
    }

    #[test]
    fn test_from_mesh_rejects_invalid_structure() {
        let mut bad = UMesh::new(nd::arr2(&[[0.0, 0.0], [1.0, 0.0]]).into_shared());
        bad.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).into_shared(),
            None,
        );
        assert!(from_mesh(&bad.view(), None, None).is_err());

        let nonfinite = UMesh::new(nd::arr2(&[[f64::NAN, 0.0]]).into_shared());
        assert!(from_mesh(&nonfinite.view(), None, None).is_err());

        let mut bad_poly = UMesh::new(nd::arr2(&[[0.0, 0.0], [1.0, 0.0]]).into_shared());
        bad_poly.add_poly_block(
            ElementType::PGON,
            nd::arr1(&[0, 1]).into_shared(),
            nd::arr1(&[3]).into_shared(),
            None,
        );
        assert!(from_mesh(&bad_poly.view(), None, None).is_err());
    }

    #[test]
    fn test_duplicate_preserves_exact_family_ids() {
        let mut source = crate::mesh_examples::make_imesh_2d(2);
        let mut ids = crate::mesh::ElementIds::new();
        ids.add_block(ElementType::QUAD4, vec![0, 2]);
        source.set_groups(BTreeMap::from([("selected".to_string(), ids)]));
        let original = source
            .block(ElementType::QUAD4)
            .unwrap()
            .families()
            .to_vec();
        let duplicated = duplicate(
            &source.view(),
            &Transform::translation(&[10.0, 0.0]).unwrap(),
            2,
        )
        .unwrap();
        let families = duplicated
            .block(ElementType::QUAD4)
            .unwrap()
            .families()
            .to_vec();
        assert_eq!(families.len(), original.len() * 2);
        assert_eq!(&families[..original.len()], original.as_slice());
        assert_eq!(&families[original.len()..], original.as_slice());
    }

    #[test]
    fn test_transform_rejects_nonfinite_and_handles_large_directions() {
        assert!(Transform::translation(&[f64::NAN]).is_err());
        assert!(
            Transform::from_matrix(
                nd::arr2(&[
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [f64::NAN, 0.0, 0.0, 1.0],
                ])
                .into_shared()
                .view()
            )
            .is_err()
        );
        let rotation = Transform::rotation(&[1.0e300, 1.0e300, 0.0], 0.7);
        assert!(rotation.is_ok());
        assert!(
            rotation
                .unwrap()
                .matrix()
                .iter()
                .all(|value| value.is_finite())
        );
        let reflection = Transform::reflection(&[1.0e300, 1.0e300, 0.0]);
        assert!(reflection.is_ok());
        let huge = quad().scale_uniform(f64::MAX).unwrap();
        assert!(huge.scale_uniform(2.0).is_err());
    }

    #[test]
    fn test_rotation_90_around_z() {
        let m = quad();
        let r = m
            .rotate(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
            .unwrap();
        let c = r.coords();
        // (1, 0) -> (0, 1)
        assert!((c[[1, 0]] - 0.0).abs() < 1e-12 && (c[[1, 1]] - 1.0).abs() < 1e-12);
        // (0, 1) -> (-1, 0)
        assert!((c[[2, 0]] + 1.0).abs() < 1e-12 && (c[[2, 1]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_translation_and_scaling() {
        let m = quad();
        let t = m.translate(&[10.0, 20.0]).unwrap();
        assert!((t.coords()[[1, 0]] - 11.0).abs() < 1e-12);
        assert!((t.coords()[[1, 1]] - 20.0).abs() < 1e-12);
        let s = m.scale(&[2.0, 2.0]).unwrap();
        assert!((s.coords()[[3, 0]] - 2.0).abs() < 1e-12);
        assert!((s.coords()[[3, 1]] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_2d_rejects_out_of_plane_rotation() {
        let m = quad();
        assert!(
            m.rotate(&[1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2)
                .is_err()
        );
        assert!(m.translate(&[0.0, 0.0, 0.5]).is_err());
        assert!(m.translate(&[1.0, 2.0]).is_ok());
    }

    #[test]
    fn test_reflection() {
        let m = quad();
        let s = m.mirror(&[0.0, 0.0, 1.0]).unwrap();
        // z-plane reflection is the 2D identity on (x, y)
        let c = s.coords();
        for i in 0..4 {
            assert!((c[[i, 0]] - m.coords()[[i, 0]]).abs() < 1e-12);
            assert!((c[[i, 1]] - m.coords()[[i, 1]]).abs() < 1e-12);
        }
        let s = m.mirror(&[1.0, 0.0, 0.0]).unwrap();
        let c = s.coords();
        // x -> -x
        for i in 0..4 {
            assert!((c[[i, 0]] + m.coords()[[i, 0]]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_composition_order() {
        let m = quad();
        // translate by (1, 0) *then* rotate 90° about z
        let tr = Transform::translation(&[1.0, 0.0])
            .unwrap()
            .then(&Transform::rotation(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).unwrap());
        // point (1, 0) -> translated (2, 0) -> rotated (0, 2)
        let moved = m.transform(&tr).unwrap();
        assert!((moved.coords()[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((moved.coords()[[1, 1]] - 2.0).abs() < 1e-12);
        // matrix product applies rhs first: (rotate then translate) should differ
        let prod = Transform::translation(&[1.0, 0.0]).unwrap()
            * Transform::rotation(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).unwrap();
        let moved = m.transform(&prod).unwrap();
        // point (1, 0) -> rotated (0, 1) -> translated (1, 1)
        assert!((moved.coords()[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((moved.coords()[[1, 1]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_inverse() {
        let coords = nd::Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.5],
        )
        .unwrap();
        let m = UMesh::new(coords.into_shared());
        let tr = Transform::translation(&[1.0, 2.0, 3.0]).unwrap()
            * Transform::rotation(&[0.0, 1.0, 0.0], 0.7).unwrap()
            * Transform::scaling(&[2.0, 1.0, 0.5]).unwrap();
        let back = tr.inverse().unwrap();
        let restored = m.transform(&tr).unwrap().transform(&back).unwrap();
        for i in 0..4 {
            for j in 0..3 {
                assert!((restored.coords()[[i, j]] - m.coords()[[i, j]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn test_aggregate_two_quads() {
        let a = quad();
        let b = quad().translate(&[0.0, 5.0, 0.0]).unwrap();
        let joined = concat(&a.view(), &b.view()).unwrap();
        assert_eq!(joined.coords().nrows(), 8);
        let blk = joined.block(ElementType::QUAD4).unwrap();
        let Connectivity::Regular(conn) = &blk.connectivity else {
            panic!()
        };
        // second quad nodes are shifted by 4
        assert_eq!(conn.row(1).to_vec(), vec![4, 5, 7, 6]);
    }

    #[test]
    fn test_aggregate_is_error_for_empty() {
        assert!(aggregate(&[]).is_err());
    }

    #[test]
    fn test_aggregate_space_dim_mismatch() {
        let a = quad();
        let b = crate::mesh_examples::make_mesh_3d_seg2();
        assert!(aggregate(&[a.view(), b.view()]).is_err());
    }

    #[test]
    fn test_aggregate_multi_poly_and_groups() {
        let mut a = crate::mesh_examples::make_mesh_2d_multi();
        let mut ids = crate::mesh::ElementIds::new();
        ids.add_block(ElementType::PGON, vec![0]);
        ids.add_block(ElementType::SEG2, vec![0]);
        a.set_groups(BTreeMap::from([("groupA".to_string(), ids)]));

        let b = a.translate(&[0.0, 5.0, 0.0]).unwrap();
        let joined = concat(&a.view(), &b.view()).unwrap();

        assert_eq!(joined.coords().nrows(), 10);
        assert_eq!(joined.block(ElementType::SEG2).unwrap().len(), 4);
        assert_eq!(joined.block(ElementType::QUAD4).unwrap().len(), 2);
        assert_eq!(joined.block(ElementType::PGON).unwrap().len(), 2);
        // PGON connectivity shifted for the second copy
        let pgon = joined.block(ElementType::PGON).unwrap();
        let Connectivity::Poly(idx) = &pgon.connectivity else {
            panic!("PGON must be stored as poly connectivity")
        };
        let sub: Vec<&[usize]> = idx.iter().collect();
        assert_eq!(sub, vec![&[0, 1, 4, 3, 2][..], &[5, 6, 9, 8, 7][..]]);
        // groups survive the concatenation on both copies
        let pgon = joined.block(ElementType::PGON).unwrap();
        let members = pgon.group_elements_local("groupA");
        assert!(members.contains(&0) && members.contains(&1));
        let seg2 = joined.block(ElementType::SEG2).unwrap();
        let members = seg2.group_elements_local("groupA");
        assert!(members.contains(&0) && members.contains(&1));
    }

    #[test]
    fn test_duplicate() {
        let m = quad();
        let dup = m
            .duplicate(&Transform::translation(&[0.0, 3.0, 0.0]).unwrap(), 3)
            .unwrap();
        let blk = dup.block(ElementType::QUAD4).unwrap();
        let Connectivity::Regular(conn) = &blk.connectivity else {
            panic!()
        };
        assert_eq!(conn.nrows(), 3);
        // copy 2 is at y=6
        let y = dup.coords()[[4 + 4, 1]];
        assert!((y - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_duplicate_zero() {
        let m = quad();
        assert!(m.duplicate(&Transform::identity(), 0).is_err());
    }

    #[test]
    fn test_set_coords_validation() {
        let m = quad();
        let bad = nd::Array2::zeros((3, 2));
        assert!(set_coords(&m.view(), bad.view()).is_err());
        let good = nd::Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        let s = set_coords(&m.view(), good.view()).unwrap();
        assert!(s.coords()[[2, 1]] == 1.0);
    }

    #[test]
    fn test_transform_coords_custom() {
        let m = quad();
        let w = transform_coords(&m.view(), |c| {
            let mut out = c.to_owned();
            for mut row in out.rows_mut() {
                row[0] = row[0].powi(2);
            }
            out
        })
        .unwrap();
        assert!((w.coords()[[1, 0]] - 1.0).abs() < 1e-12);
        assert!((w.coords()[[2, 0]] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_inplace_set_coordinates() {
        let mut m = quad();
        let new = nd::Array2::from_shape_vec((4, 2), vec![1.0; 8]).unwrap();
        m.set_coordinates(new.view()).unwrap();
        assert!(m.coords()[[0, 0]] == 1.0);
        let bad = nd::Array2::zeros((3, 2));
        assert!(m.set_coordinates(bad.view()).is_err());
    }

    #[test]
    fn test_inplace_transform_coordinates() {
        let mut m = quad();
        m.transform_coordinates(&Transform::translation(&[1.0, 0.0, 0.0]).unwrap())
            .unwrap();
        assert!((m.coords()[[1, 0]] - 2.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------------
    // About-a-point variants
    // ------------------------------------------------------------------------

    #[test]
    fn test_rotation_about_center() {
        let m = quad();
        // rotate the quad by pi about its center: p -> 2*c - p
        let r = m
            .rotate_about(&[0.5, 0.5], &[0.0, 0.0, 1.0], std::f64::consts::PI)
            .unwrap();
        let c = r.coords();
        for i in 0..4 {
            assert!((c[[i, 0]] - (1.0 - m.coords()[[i, 0]])).abs() < 1e-12);
            assert!((c[[i, 1]] - (1.0 - m.coords()[[i, 1]])).abs() < 1e-12);
        }
        // 90° about a corner pivots the opposite corner to the other axis
        let r = m
            .rotate_about(&[0.0, 0.0], &[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
            .unwrap();
        // (1, 0) -> (0, 1)
        assert!((r.coords()[[1, 0]] - 0.0).abs() < 1e-12);
        assert!((r.coords()[[1, 1]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mirror_about_point() {
        let m = quad();
        let s = m.mirror_about(&[0.5, 0.5], &[1.0, 0.0, 0.0]).unwrap();
        // mirroring through x = 0.5 maps x -> 1 - x
        let c = s.coords();
        for i in 0..4 {
            assert!((c[[i, 0]] - (1.0 - m.coords()[[i, 0]])).abs() < 1e-12);
        }
    }

    #[test]
    fn test_2d_rotate_about_rejects_out_of_plane() {
        let m = quad();
        assert!(
            m.rotate_about(&[0.5, 0.5], &[1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2)
                .is_err()
        );
    }

    // ------------------------------------------------------------------------
    // Constructor and validation branches
    // ------------------------------------------------------------------------

    #[test]
    fn test_from_matrix_validation() {
        // wrong last row
        let bad = nd::array![
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0]
        ];
        assert!(Transform::from_matrix(bad.view()).is_err());
        // non-4x4 shape
        let bad = nd::Array2::eye(3);
        assert!(Transform::from_matrix(bad.view()).is_err());
        // valid matrix works
        let good = nd::Array2::eye(4);
        assert!(Transform::from_matrix(good.view()).is_ok());
    }

    #[test]
    fn test_pad_errors() {
        assert!(Transform::translation(&[]).is_err());
        assert!(Transform::translation(&[1.0, 2.0, 3.0, 4.0]).is_err());
        assert!(Transform::scaling(&[1.0, 2.0, 3.0, 4.0]).is_err());
        assert!(Transform::rotation(&[1.0, 2.0, 3.0, 4.0], 1.0).is_err());
    }

    #[test]
    fn test_zero_axis_normal_errors() {
        assert!(Transform::rotation(&[0.0, 0.0, 0.0], 1.0).is_err());
        assert!(Transform::reflection(&[0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn test_inverse_singular() {
        let tr = Transform::scaling(&[2.0, 0.0, 1.0]).unwrap();
        assert!(tr.inverse().is_err());
    }

    #[test]
    fn test_scaling_pads_missing_axes() {
        // 1-component scaling leaves the other axes unchanged
        let tr = Transform::scaling(&[2.0]).unwrap();
        let m = quad();
        let s = m.transform(&tr).unwrap();
        assert!((s.coords()[[3, 0]] - 2.0).abs() < 1e-12);
        assert!((s.coords()[[3, 1]] - 1.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------------
    // 1D meshes (the x-axis line)
    // ------------------------------------------------------------------------

    #[test]
    fn test_1d_translate_and_rotate() {
        let m = crate::mesh_examples::make_mesh_3d_seg2();
        let t = m.translate(&[1.0]).unwrap();
        assert!((t.coords()[[1, 0]] - 2.0).abs() < 1e-12);
        // moving out of the line is rejected
        assert!(m.translate(&[0.0, 1.0]).is_err());
        assert!(m.translate(&[0.0, 0.0, 1.0]).is_err());
        // a rotation about the line direction (x-axis) leaves a 1D mesh unchanged...
        let r = m
            .rotate(&[1.0, 0.0, 0.0], std::f64::consts::FRAC_PI_2)
            .unwrap();
        for i in 0..3 {
            assert!((r.coords()[[i, 0]] - m.coords()[[i, 0]]).abs() < 1e-12);
        }
        // ...while a rotation about the z-axis would lift the line into the plane
        assert!(
            m.rotate(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2)
                .is_err()
        );
    }

    #[test]
    fn test_aggregate_1d() {
        let a = crate::mesh_examples::make_mesh_3d_seg2();
        let b = a.translate(&[5.0, 0.0, 0.0]).unwrap();
        let joined = concat(&a.view(), &b.view()).unwrap();
        assert_eq!(joined.coords().nrows(), 6);
        let blk = joined.block(ElementType::SEG2).unwrap();
        let Connectivity::Regular(conn) = &blk.connectivity else {
            panic!()
        };
        // second copy's nodes are shifted by 3: rows 2 and 3
        assert_eq!(conn.row(2).to_vec(), vec![3, 4]);
        assert_eq!(conn.row(3).to_vec(), vec![4, 5]);
    }

    // ------------------------------------------------------------------------
    // 3D meshes with a real block
    // ------------------------------------------------------------------------

    #[test]
    fn test_3d_rotate_about_arbitrary_axis() {
        let m = crate::mesh_examples::make_mesh_3d_tet();
        // 90° rotation about the unit axis (1, 1, 0)/sqrt(2): node (1,0,0) -> (0.5, 0.5, -1/sqrt(2))
        let r = m
            .rotate(&[1.0, 1.0, 0.0], std::f64::consts::FRAC_PI_2)
            .unwrap();
        let expected = 1.0 / std::f64::consts::SQRT_2;
        assert!((r.coords()[[1, 0]] - 0.5).abs() < 1e-9);
        assert!((r.coords()[[1, 1]] - 0.5).abs() < 1e-9);
        assert!((r.coords()[[1, 2]] + expected).abs() < 1e-9);
        // node (0, 1, 0) -> (0.5, 0.5, 1/sqrt(2))
        assert!((r.coords()[[2, 0]] - 0.5).abs() < 1e-9);
        assert!((r.coords()[[2, 1]] - 0.5).abs() < 1e-9);
        assert!((r.coords()[[2, 2]] - expected).abs() < 1e-9);
        // rigid motion: the TET4 block is unchanged
        assert_eq!(r.block(ElementType::TET4).unwrap().len(), 1);
    }

    #[test]
    fn test_3d_translate_scale_and_mirror() {
        let m = crate::mesh_examples::make_mesh_3d_tet();
        let t = m.translate(&[1.0, 2.0, 3.0]).unwrap();
        assert!((t.coords()[[3, 2]] - 4.0).abs() < 1e-12);
        let s = m.scale(&[2.0, 0.5, 1.0]).unwrap();
        assert!((s.coords()[[1, 0]] - 2.0).abs() < 1e-12);
        let g = m.mirror(&[0.0, 0.0, 1.0]).unwrap();
        assert!((g.coords()[[3, 2]] + 1.0).abs() < 1e-12);
    }

    // ------------------------------------------------------------------------
    // Aggregate error branches and semantics
    // ------------------------------------------------------------------------

    /// Helper: an owned QUAD4 mesh shifted by `shift` with a `temperature` field.
    fn mesh_with_field(value: f64, shift: f64) -> UMesh {
        let coords = nd::Array2::from_shape_vec(
            (4, 2),
            vec![shift, 0.0, shift + 1.0, 0.0, shift, 1.0, shift + 1.0, 1.0],
        )
        .unwrap();
        let mut m = UMesh::new(coords.into());
        let conn = nd::arr2(&[[0, 1, 3, 2]]).into_shared();
        let fields = BTreeMap::from([(
            "temperature".to_string(),
            nd::Array::from_shape_vec((1, 1), vec![value])
                .unwrap()
                .into_dyn()
                .into_shared(),
        )]);
        m.add_regular_block(ElementType::QUAD4, conn, Some(fields));
        m
    }

    #[test]
    fn test_aggregate_fields_values() {
        let a = mesh_with_field(21.5, 0.0);
        let b = mesh_with_field(31.5, 5.0);
        let joined = concat(&a.view(), &b.view()).unwrap();
        let blk = joined.block(ElementType::QUAD4).unwrap();
        let f = blk.fields.get("temperature").unwrap();
        let vals: Vec<f64> = f.iter().copied().collect();
        assert_eq!(vals, vec![21.5, 31.5]);
    }

    #[test]
    fn test_aggregate_missing_field_error() {
        let a = mesh_with_field(21.5, 0.0);
        let b = quad();
        let err = concat(&a.view(), &b.view()).unwrap_err();
        assert!(err.contains("'temperature'"), "{err}");
    }

    #[test]
    fn test_aggregate_field_shape_mismatch_error() {
        let a = mesh_with_field(21.5, 0.0);
        let b = quad().translate(&[5.0, 0.0]).unwrap();
        let fields = BTreeMap::from([(
            "temperature".to_string(),
            nd::Array::from_shape_vec((1, 2), vec![31.5, 32.5])
                .unwrap()
                .into_dyn()
                .into_shared(),
        )]);
        let conn = nd::arr2(&[[0, 1, 3, 2]]).into_shared();
        let b = {
            let coords = b.coords().to_owned().into_shared();
            let mut m = UMesh::new(coords);
            m.add_regular_block(ElementType::QUAD4, conn, Some(fields));
            m
        };
        let err = concat(&a.view(), &b.view()).unwrap_err();
        assert!(err.contains("incompatible trailing shape"), "{err}");
    }

    #[test]
    fn test_aggregate_single_mesh() {
        let m = quad();
        let joined = aggregate(&[m.view()]).unwrap();
        assert_eq!(joined.coords().nrows(), 4);
        assert_eq!(joined.block(ElementType::QUAD4).unwrap().len(), 1);
    }

    #[test]
    fn test_aggregate_disjoint_etypes() {
        let a = quad();
        let coords =
            nd::Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0]).unwrap();
        let mut b = UMesh::new(coords.into());
        b.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]).into_shared(),
            None,
        );
        let joined = aggregate(&[a.view(), b.view()]).unwrap();
        assert_eq!(joined.block(ElementType::QUAD4).unwrap().len(), 1);
        assert_eq!(joined.block(ElementType::SEG2).unwrap().len(), 2);
        // SEG2 nodes are shifted by the 4 quad nodes
        let blk = joined.block(ElementType::SEG2).unwrap();
        let Connectivity::Regular(conn) = &blk.connectivity else {
            panic!()
        };
        assert_eq!(conn.row(1).to_vec(), vec![5, 6]);
    }

    #[test]
    fn test_aggregate_family_relabeling() {
        let coords_a = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut a = UMeshView::new(coords_a.view());
        let conn_a = nd::arr2(&[[0, 1, 3, 2]]);
        let fam_a = nd::array![0usize];
        a.add_regular_block(ElementType::QUAD4, conn_a.view(), Some(fam_a.view()));
        let coords_b = nd::array![[2.0, 0.0], [3.0, 0.0], [2.0, 1.0], [3.0, 1.0]];
        let mut b = UMeshView::new(coords_b.view());
        let conn_b = nd::arr2(&[[0, 1, 3, 2]]);
        let fam_b = nd::array![2usize];
        b.add_regular_block(ElementType::QUAD4, conn_b.view(), Some(fam_b.view()));
        let joined = aggregate(&[a, b]).unwrap();
        // family 2 of the second mesh is relabeled by max_family+1 = 1 -> 3
        assert_eq!(
            joined
                .block(ElementType::QUAD4)
                .unwrap()
                .families()
                .to_vec(),
            vec![0, 3]
        );
    }

    #[test]
    fn test_aggregate_regular_vs_poly_error() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut a = UMeshView::new(coords.view());
        let conn_a = nd::arr2(&[[0, 1, 2, 0]]);
        let fam = nd::array![0usize];
        a.add_regular_block(ElementType::QUAD4, conn_a.view(), Some(fam.view()));
        let mut b = UMeshView::new(coords.view());
        let conn_b = nd::Array1::from(vec![0, 1, 2]);
        let off_b = nd::Array1::from(vec![0, 3]);
        b.add_poly_block(
            ElementType::QUAD4,
            conn_b.view(),
            off_b.view(),
            Some(fam.view()),
        );
        let err = aggregate(&[a, b]).unwrap_err();
        assert!(
            err.contains("regular") && err.contains("polygonal"),
            "{err}"
        );
    }

    #[test]
    fn test_aggregate_connectivity_width_mismatch() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let fam = nd::array![0usize];
        let mut a = UMeshView::new(coords.view());
        let conn_a = nd::arr2(&[[0, 1, 2, 0]]);
        a.add_regular_block(ElementType::QUAD4, conn_a.view(), Some(fam.view()));
        let mut b = UMeshView::new(coords.view());
        // same element type, different connectivity width
        let conn_b = nd::arr2(&[[0, 1, 2]]);
        b.add_regular_block(ElementType::QUAD4, conn_b.view(), Some(fam.view()));
        let err = aggregate(&[a, b]).unwrap_err();
        assert!(err.contains("incompatible connectivity width"), "{err}");
    }

    // ------------------------------------------------------------------------
    // Duplicate
    // ------------------------------------------------------------------------

    #[test]
    fn test_duplicate_rotated_copies() {
        let m = quad();
        let step = Transform::rotation(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).unwrap();
        let dup = m.duplicate(&step, 4).unwrap();
        assert_eq!(dup.coords().nrows(), 16);
        // node 1 of each copy: (1,0) under successive 90° rotations
        let expected = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]];
        for (k, exp) in expected.iter().enumerate() {
            assert!((dup.coords()[[4 * k + 1, 0]] - exp[0]).abs() < 1e-9);
            assert!((dup.coords()[[4 * k + 1, 1]] - exp[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_duplicate_composed_step() {
        let m = quad();
        // screw: translate by (1, 0) then rotate by 90°
        let step = Transform::translation(&[1.0, 0.0])
            .unwrap()
            .then(&Transform::rotation(&[0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2).unwrap());
        let dup = m.duplicate(&step, 3).unwrap();
        let blk = dup.block(ElementType::QUAD4).unwrap();
        let Connectivity::Regular(conn) = &blk.connectivity else {
            panic!()
        };
        assert_eq!(conn.nrows(), 3);
        assert_eq!(dup.coords().nrows(), 12);
        // reference point (1, 0): copy0 (1,0); step: (1,0)->(2,0)->(0,2); step^2: (0,2)->(1,2)->(-2,1)
        let exp = [[1.0, 0.0], [0.0, 2.0], [-2.0, 1.0]];
        for (k, e) in exp.iter().enumerate() {
            assert!((dup.coords()[[4 * k + 1, 0]] - e[0]).abs() < 1e-9);
            assert!((dup.coords()[[4 * k + 1, 1]] - e[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn test_duplicate_preserves_fields_and_groups() {
        let mut ids = crate::mesh::ElementIds::new();
        ids.add_block(ElementType::QUAD4, vec![0]);

        let mut m = mesh_with_field(21.5, 0.0);
        m.set_groups(BTreeMap::from([("heated".to_string(), ids)]));

        let step = Transform::translation(&[0.0, 3.0, 0.0]).unwrap();
        let dup = m.duplicate(&step, 2).unwrap();
        // field payloads are concatenated per copy
        let blk = dup.block(ElementType::QUAD4).unwrap();
        let vals: Vec<f64> = blk
            .fields
            .get("temperature")
            .unwrap()
            .iter()
            .copied()
            .collect();
        assert_eq!(vals, vec![21.5, 21.5]);
        // group survives on every copy
        let members = blk.group_elements_local("heated");
        assert!(members.contains(&0) && members.contains(&1));
    }

    #[test]
    fn test_duplicate_single() {
        let m = quad();
        let dup = m
            .duplicate(&Transform::translation(&[1.0, 0.0]).unwrap(), 1)
            .unwrap();
        assert_eq!(dup.coords().nrows(), 4);
        assert_eq!(dup.block(ElementType::QUAD4).unwrap().len(), 1);
        assert!((dup.coords()[[1, 0]] - 1.0).abs() < 1e-12);
    }
}
