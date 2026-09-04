//! Convert any mesh to a polygonal/polyhedral mesh.
//!
//! This module converts all elements of a `UMeshView` into their poly equivalents:
//! - 0D elements (VERTEX) stay unchanged.
//! - 1D elements become SPLINE.
//! - 2D elements become PGON.
//! - 3D elements become PHED with face-based connectivity.
//!
//! The inverse operation can be used to simplify some cells from a mesh.

use crate::element_traits::ElementTopo;
use crate::mesh::{Dimension, UMesh, UMeshView};

/// Converts all elements in the mesh to their poly equivalents.
///
/// Returns a new `UMesh` where:
/// - 0D elements (VERTEX) are preserved as-is.
/// - 1D elements (SEG2, SEG3, SEG4) become SPLINE.
/// - 2D elements (TRI3, TRI6, TRI7, QUAD4, QUAD8, QUAD9) become PGON.
/// - 3D elements (TET4, TET10, HEX8, HEX21) become PHED.
/// - Already-poly elements (SPLINE, PGON, PHED) are preserved.
///
/// Fields and groups are not transferred.
pub fn polyze(mesh: &UMeshView) -> UMesh {
    let mut new_mesh = UMesh::new(mesh.coords().to_shared());

    for elem in mesh.elements() {
        let (poly_et, poly_conn) = elem.to_poly();
        new_mesh.add_element(poly_et, &poly_conn, None);
    }

    new_mesh
}

/// Converts poly elements back to their regular equivalents.
///
/// - SPLINE cannot be converted (ambiguous).
/// - PGON with 3 nodes becomes TRI3, with 4 nodes becomes QUAD4.
/// - PHED with 4 triangular faces becomes TET4, with 6 quadrilateral faces becomes HEX8.
/// - Already-regular elements are copied unchanged.
///
/// Returns `Err` on the first element that cannot be converted.
pub fn unpolyze(mesh: &UMeshView) -> Result<UMesh, String> {
    let mut new_mesh = UMesh::new(mesh.coords().to_shared());

    for elem in mesh.elements() {
        let (et, conn) = elem.from_poly()?;
        new_mesh.add_element(et, &conn, None);
    }

    Ok(new_mesh)
}

/// Converts elements of a specific dimension to their poly equivalents.
///
/// Elements not of the target dimension are copied as-is (regular blocks stay
/// regular, poly blocks stay poly). Only elements matching `dim` are converted.
pub fn polyze_dim(mesh: &UMeshView, dim: Dimension) -> UMesh {
    let mut new_mesh = UMesh::new(mesh.coords().to_shared());

    for elem in mesh.elements() {
        if elem.element_type.dimension() == dim {
            let (poly_et, poly_conn) = elem.to_poly();
            new_mesh.add_element(poly_et, &poly_conn, None);
        } else {
            new_mesh.add_element(elem.element_type, elem.connectivity, None);
        }
    }

    new_mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, UMesh};
    use ndarray as nd;

    #[test]
    fn test_polyze_empty_mesh() {
        let coords = nd::ArcArray2::zeros((0, 2));
        let mesh = UMesh::new(coords);
        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 0);
    }

    #[test]
    fn test_polyze_seg2() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 2);
        assert!(
            poly_mesh
                .element_types()
                .all(|&et| et == ElementType::SPLINE)
        );
    }

    #[test]
    fn test_polyze_tri3() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 2]]).to_shared(), None);

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 1);
        assert!(poly_mesh.element_types().all(|&et| et == ElementType::PGON));
    }

    #[test]
    fn test_polyze_quad4() {
        let coords =
            nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 1);
        assert!(poly_mesh.element_types().all(|&et| et == ElementType::PGON));
    }

    #[test]
    fn test_polyze_tet4() {
        let coords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 1);
        assert!(poly_mesh.element_types().all(|&et| et == ElementType::PHED));
    }

    #[test]
    fn test_polyze_hex8() {
        let coords = nd::ArcArray2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 1);
        assert!(poly_mesh.element_types().all(|&et| et == ElementType::PHED));
    }

    #[test]
    fn test_polyze_mixed_mesh() {
        let coords = nd::ArcArray2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 3]]).to_shared(),
            None,
        );
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );
        mesh.add_element(ElementType::PGON, &[0, 1, 4, 3, 2], None);

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 4);
        // All should be poly: 2 SPLINE + 1 PGON + 1 PGON
        let types: Vec<ElementType> = poly_mesh.element_types().copied().collect();
        assert!(types.contains(&ElementType::SPLINE));
        assert!(types.contains(&ElementType::PGON));
        assert!(!types.contains(&ElementType::SEG2));
        assert!(!types.contains(&ElementType::QUAD4));
    }

    #[test]
    fn test_polyze_vertex_unchanged() {
        let coords = nd::ArcArray2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::VERTEX, nd::arr2(&[[0], [1]]).to_shared(), None);

        let poly_mesh = polyze(&mesh.view());
        assert_eq!(poly_mesh.num_elements(), 2);
        assert!(
            poly_mesh
                .element_types()
                .all(|&et| et == ElementType::VERTEX)
        );
    }

    #[test]
    fn test_polyze_coords_preserved() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.5, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 2]]).to_shared(), None);

        let poly_mesh = polyze(&mesh.view());
        // Coordinates should be shared (same pointer)
        assert_eq!(poly_mesh.coords(), mesh.coords());
    }

    #[test]
    fn test_polyze_dim_selective() {
        let coords = nd::ArcArray2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 3]]).to_shared(),
            None,
        );
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );

        // Only polyze 2D elements
        let poly_mesh = polyze_dim(&mesh.view(), Dimension::D2);
        assert_eq!(poly_mesh.num_elements(), 3);
        // SEG2 elements should remain as SEG2
        assert!(poly_mesh.element_types().any(|&et| et == ElementType::SEG2));
        // QUAD4 should become PGON
        assert!(poly_mesh.element_types().any(|&et| et == ElementType::PGON));
        assert!(
            !poly_mesh
                .element_types()
                .any(|&et| et == ElementType::QUAD4)
        );
    }

    // ===== unpolyze tests =====

    #[test]
    fn test_unpolyze_empty_mesh() {
        let coords = nd::ArcArray2::zeros((0, 2));
        let mesh = UMesh::new(coords);
        let result = unpolyze(&mesh.view()).unwrap();
        assert_eq!(result.num_elements(), 0);
    }

    #[test]
    fn test_unpolyze_tri3_roundtrip() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 2]]).to_shared(), None);

        let poly_mesh = polyze(&mesh.view());
        let unpoly_mesh = unpolyze(&poly_mesh.view()).unwrap();

        assert_eq!(unpoly_mesh.num_elements(), 1);
        let types: Vec<ElementType> = unpoly_mesh.element_types().copied().collect();
        assert_eq!(types, vec![ElementType::TRI3]);
    }

    #[test]
    fn test_unpolyze_quad4_roundtrip() {
        let coords =
            nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        let unpoly_mesh = unpolyze(&poly_mesh.view()).unwrap();

        assert_eq!(unpoly_mesh.num_elements(), 1);
        let types: Vec<ElementType> = unpoly_mesh.element_types().copied().collect();
        assert_eq!(types, vec![ElementType::QUAD4]);
    }

    #[test]
    fn test_unpolyze_tet4_roundtrip() {
        let coords = nd::ArcArray2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::TET4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        let unpoly_mesh = unpolyze(&poly_mesh.view()).unwrap();

        assert_eq!(unpoly_mesh.num_elements(), 1);
        let types: Vec<ElementType> = unpoly_mesh.element_types().copied().collect();
        assert_eq!(types, vec![ElementType::TET4]);
    }

    #[test]
    fn test_unpolyze_hex8_roundtrip() {
        let coords = nd::ArcArray2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::HEX8,
            nd::arr2(&[[0, 1, 2, 3, 4, 5, 6, 7]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        let unpoly_mesh = unpolyze(&poly_mesh.view()).unwrap();

        assert_eq!(unpoly_mesh.num_elements(), 1);
        let types: Vec<ElementType> = unpoly_mesh.element_types().copied().collect();
        assert_eq!(types, vec![ElementType::HEX8]);
    }

    #[test]
    fn test_unpolyze_mixed_mesh() {
        let coords = nd::ArcArray2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 3]]).to_shared(),
            None,
        );
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );
        mesh.add_element(ElementType::PGON, &[0, 1, 4, 3, 2], None);

        let poly_mesh = polyze(&mesh.view());
        let result = unpolyze(&poly_mesh.view());

        // SPLINE cannot be converted, so this should error.
        assert!(result.is_err());
    }

    #[test]
    fn test_unpolyze_spline_error() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 2.0, 0.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        assert!(
            poly_mesh
                .element_types()
                .all(|&et| et == ElementType::SPLINE)
        );

        let result = unpolyze(&poly_mesh.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_unpolyze_coords_preserved() {
        let coords =
            nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
                .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 2, 3]]).to_shared(),
            None,
        );

        let poly_mesh = polyze(&mesh.view());
        let unpoly_mesh = unpolyze(&poly_mesh.view()).unwrap();

        assert_eq!(unpoly_mesh.coords(), mesh.coords());
    }

    #[test]
    fn test_unpolyze_pure_regular_mesh() {
        let coords =
            nd::ArcArray2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::TRI3, nd::arr2(&[[0, 1, 2]]).to_shared(), None);

        // unpolyze on a mesh with no poly elements should just copy it.
        let result = unpolyze(&mesh.view()).unwrap();
        assert_eq!(result.num_elements(), 1);
        let types: Vec<ElementType> = result.element_types().copied().collect();
        assert_eq!(types, vec![ElementType::TRI3]);
    }
}
