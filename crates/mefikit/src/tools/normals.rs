//! Surface normals for mesh cells.
//!
//! Computes unit outward normals for 2D cells embedded in 3D space (a 3-vector per cell)
//! and for 1D cells embedded in 2D space (an in-plane 2-vector per cell).

use crate::element_traits::ElementGeo;
use crate::geometry::newell_normal;
use crate::mesh::ElementType;
use crate::mesh::{Dimension, UMeshView};

use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Resolves the element dimension on which normals are defined.
///
/// With `None` this is the hypersurface, `space_dimension() - 1`. With `Some(d)` the given
/// dimension is honored strictly; panics if it is not a valid hypersurface dimension of the
/// space (normals are undefined on volume cells in 3D or surfaces in 2D).
fn target_normal_dim(mesh: &UMeshView, dim: Option<Dimension>) -> Dimension {
    match dim {
        Some(d) => {
            let space = mesh.space_dimension();
            let hypersurface = Dimension::try_from(space - 1)
                .expect("space dimension minus one is a valid element dimension");
            if d != hypersurface {
                panic!(
                    "normals are only defined on the hypersurface of the space (dimension \
                     {:?}), not on dimension {:?} elements",
                    hypersurface, d
                );
            }
            d
        }
        None => Dimension::try_from(mesh.space_dimension() - 1)
            .expect("space dimension minus one is a valid element dimension"),
    }
}

/// Computes the unit normal of each element in the mesh.
///
/// Returns a map of element types to arrays of normal vectors, either `[n, 3]` for 2D
/// cells in 3D space or `[n, 2]` for 1D cells in 2D space.
///
/// Normals are defined on the hypersurface elements: those of dimension
/// `space_dimension() - 1` (e.g. the QUAD4 boundary faces of a volume mesh, or the
/// surface of a pure surface mesh). Volume cells are therefore never selected by default.
///
/// When `dim` is `None` the target dimension derives from the space dimension so that a
/// mesh holding both volume cells and their boundary faces yields normals on the boundary
/// faces. When `dim` is `Some(d)` the given dimension is honored strictly; panics if the
/// requested dimension is not a hypersurface dimension of the space (normals are
/// undefined there).
pub fn normals(mesh: &UMeshView, dim: Option<Dimension>) -> BTreeMap<ElementType, nd::Array2<f64>> {
    let target = target_normal_dim(mesh, dim);
    mesh.par_blocks()
        .filter(|(et, _)| et.dimension() == target)
        .map(|(&k, v)| {
            let vecs: Vec<[f64; 3]> = v
                .par_iter(mesh.coords.view())
                .map(|e| surface_normal(&e, mesh.space_dimension()))
                .collect();
            let ncomp = if mesh.space_dimension() == 3 { 3 } else { 2 };
            let data: Vec<f64> = vecs.into_iter().flat_map(|x| x[..ncomp].to_vec()).collect();
            (
                k,
                nd::Array2::from_shape_vec((v.len(), ncomp), data)
                    .expect("normal count matches component count"),
            )
        })
        .collect()
}

/// Computes the unit `x` component of each element's normal in the mesh.
pub fn nx(mesh: &UMeshView, dim: Option<Dimension>) -> BTreeMap<ElementType, nd::Array1<f64>> {
    normal_component(mesh, dim, 0)
}

/// Computes the unit `y` component of each element's normal in the mesh.
pub fn ny(mesh: &UMeshView, dim: Option<Dimension>) -> BTreeMap<ElementType, nd::Array1<f64>> {
    normal_component(mesh, dim, 1)
}

/// Computes the unit `z` component of each element's normal in the mesh.
pub fn nz(mesh: &UMeshView, dim: Option<Dimension>) -> BTreeMap<ElementType, nd::Array1<f64>> {
    normal_component(mesh, dim, 2)
}

fn normal_component(
    mesh: &UMeshView,
    dim: Option<Dimension>,
    component: usize,
) -> BTreeMap<ElementType, nd::Array1<f64>> {
    let target = target_normal_dim(mesh, dim);
    mesh.par_blocks()
        .filter(|(et, _)| et.dimension() == target)
        .map(|(&k, v)| {
            let vals: Vec<f64> = v
                .par_iter(mesh.coords.view())
                .map(|e| {
                    let n = surface_normal(&e, mesh.space_dimension());
                    n[component]
                })
                .collect();
            (k, nd::Array1::from_shape_vec((v.len(),), vals).unwrap())
        })
        .collect()
}

/// Computes the unit surface normal of a single element.
///
/// For a 2D cell in 3D space this is the (normalized) Newell normal of its facet; for a
/// 1D cell in 2D space it is the in-plane normal obtained by rotating the segment
/// direction by 90° counter-clockwise. Returns a 3-vector whose first two components
/// carry the in-plane normal when the space is 2D.
fn surface_normal<'a, E>(e: &E, space_dim: usize) -> [f64; 3]
where
    E: ElementGeo<'a> + ?Sized,
{
    match space_dim {
        3 => {
            let points: Vec<[f64; 3]> = e.coords3().copied().collect();
            let mut n = newell_normal(&points);
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if len > 0.0 {
                for c in n.iter_mut() {
                    *c /= len;
                }
            }
            n
        }
        2 => {
            let a = e.coord2_ref(0);
            let b = e.coord2_ref(1);
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = (dx * dx + dy * dy).sqrt();
            if len > 0.0 {
                // 90° counter-clockwise rotation of the tangent.
                [-dy / len, dx / len, 0.0]
            } else {
                [0.0, 0.0, 0.0]
            }
        }
        _ => panic!("surface normals are only defined in 2D or 3D space, got {space_dim}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;
    use crate::prelude as mf;
    use approx::*;
    use ndarray as nd;

    fn quad_in_3d() -> mf::UMesh {
        let coords = nd::Array2::from_shape_vec(
            (4, 3),
            vec![
                0.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, //
                0.0, 1.0, 0.0, //
                1.0, 1.0, 0.0, //
            ],
        )
        .unwrap();
        let mut mesh = mf::UMesh::new(coords.into());
        mesh.add_regular_block(
            mf::ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );
        mesh
    }

    fn seg_in_2d() -> mf::UMesh {
        // A vertical segment (0,0)-(0,1) in 2D: in-plane normal must be (-1, 0).
        let coords = nd::Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 1.0]).unwrap();
        let mut mesh = mf::UMesh::new(coords.into());
        mesh.add_regular_block(mf::ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        mesh
    }

    #[test]
    fn test_normals_2d_cells_in_3d() {
        // Planar quad lying in z=0 -> unit normal (0, 0, 1).
        let mesh = quad_in_3d();
        let normals = normals(&mesh.view(), None);
        let n = normals.get(&ElementType::QUAD4).unwrap();
        assert_eq!(n.shape(), &[1, 3]);
        assert_abs_diff_eq!(n[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(n[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(n[[0, 2]], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_normal_components_3d() {
        let mesh = quad_in_3d();
        let qx = nx(&mesh.view(), None);
        let qy = ny(&mesh.view(), None);
        let qz = nz(&mesh.view(), None);
        assert_abs_diff_eq!(qx[&ElementType::QUAD4][0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(qy[&ElementType::QUAD4][0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(qz[&ElementType::QUAD4][0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_normals_1d_cells_in_2d() {
        // Vertical segment -> in-plane normal index ordering keeps the 2-vector (-1, 0).
        let mesh = seg_in_2d();
        let normals = normals(&mesh.view(), None);
        let n = normals.get(&ElementType::SEG2).unwrap();
        assert_eq!(n.shape(), &[1, 2]);
        assert_abs_diff_eq!(n[[0, 0]], -1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(n[[0, 1]], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_normals_on_boundary_not_volume() {
        // Regression: a mesh holding both a HEX8 volume and a QUAD4 boundary face must
        // yield normals on the QUAD4 (space_dim - 1) block, never on the HEX8 volume.
        let coords = nd::Array2::from_shape_vec(
            (9, 3),
            vec![
                0.0, 0.0, 0.0, // 0
                1.0, 0.0, 0.0, // 1
                1.0, 1.0, 0.0, // 2
                0.0, 1.0, 0.0, // 3
                0.0, 0.0, 1.0, // 4
                1.0, 0.0, 1.0, // 5
                1.0, 1.0, 1.0, // 6
                0.0, 1.0, 1.0, // 7
                0.0, 0.0, 2.0, // 8 (extra node for the top-quad boundary face)
            ],
        )
        .unwrap();
        let mut mesh = mf::UMesh::new(coords.into());
        // Unit cube.
        mesh.add_regular_block(
            mf::ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );
        // A QUAD4 boundary face lying in the plane z = 1 with normal (0, 0, 1).
        mesh.add_regular_block(
            mf::ElementType::QUAD4,
            nd::arr2(&[[4, 5, 6, 7]]).to_shared(),
            None,
        );

        let n = normals(&mesh.view(), None);
        assert!(
            !n.contains_key(&ElementType::HEX8),
            "volume cells must not get normals"
        );
        let quad = n.get(&ElementType::QUAD4).unwrap();
        assert_eq!(quad.shape(), &[1, 3]);
        assert_abs_diff_eq!(quad[[0, 0]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(quad[[0, 1]], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(quad[[0, 2]], 1.0, epsilon = 1e-12);

        let qz = nz(&mesh.view(), None);
        assert!(!qz.contains_key(&ElementType::HEX8));
        assert_abs_diff_eq!(qz[&ElementType::QUAD4][0], 1.0, epsilon = 1e-12);
    }
}
