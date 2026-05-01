//! Integration tests for mefikit library
//!
//! These tests focus on end-to-end functionality and interactions between modules.

use mefikit::prelude::*;
use mefikit::mesh::{ElementIds, ElementId, FieldBase};
use ndarray as nd;
use std::collections::BTreeMap;
use approx::assert_abs_diff_eq;

// Helper functions (similar to mesh_examples but available in integration tests)
fn make_mesh_2d_quad() -> UMesh {
    let coords =
        nd::ArcArray2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
            .unwrap();
    let mut mesh = UMesh::new(coords);
    mesh.add_regular_block(
        ElementType::QUAD4,
        nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
        None,
    );
    mesh
}

fn make_mesh_2d_multi() -> UMesh {
    let coords = nd::Array2::from_shape_vec(
        (5, 2),
        vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.5, 0.5],
    )
    .unwrap();
    let mut mesh = UMesh::new(coords.into());
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
    mesh.add_element(ElementType::PGON, &[0, 1, 4, 3, 2], None, None);
    mesh
}

fn make_imesh_2d(n: usize) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build()
}

fn make_imesh_3d(n: usize) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .add_axis((0..=n).map(|k| (k as f64) / (n as f64)).collect())
        .build()
}

mod mesh_creation {
    use super::*;

    #[test]
    fn create_empty_mesh() {
        let coords = nd::ArcArray2::from_shape_vec((0, 2), vec![]).unwrap();
        let mesh = UMesh::new(coords);
        assert_eq!(mesh.num_elements(), 0);
        assert_eq!(mesh.topological_dimension(), None);
    }

    #[test]
    fn create_1d_mesh_with_segments() {
        let coords = nd::Array2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1]]).to_shared(),
            None,
        );
        assert_eq!(mesh.num_elements(), 1);
        assert_eq!(mesh.num_elements_of_dim(Dimension::D1), 1);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D1));
    }

    #[test]
    fn create_2d_mesh_with_quads() {
        let mesh = make_mesh_2d_quad();
        assert_eq!(mesh.num_elements(), 1);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D2));
        assert_eq!(mesh.space_dimension(), 2);
    }

    #[test]
    fn create_3d_mesh_with_tets() {
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
        assert_eq!(mesh.num_elements(), 1);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D3));
        assert_eq!(mesh.space_dimension(), 3);
    }

    #[test]
    fn create_mixed_dimension_mesh() {
        let mesh = make_mesh_2d_multi();
        assert_eq!(mesh.num_elements(), 4);
        assert_eq!(mesh.num_elements_of_dim(Dimension::D1), 2);
        assert_eq!(mesh.num_elements_of_dim(Dimension::D2), 2);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D2));
    }

    #[test]
    fn create_polygon_element() {
        let coords = nd::ArcArray2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_element(ElementType::PGON, &[0, 1, 2, 3], None, None);
        assert_eq!(mesh.num_elements(), 1);
        let elements: Vec<_> = mesh.elements().collect();
        assert_eq!(elements[0].connectivity, &[0, 1, 2, 3]);
    }
}

mod mesh_iteration {
    use super::*;

    #[test]
    fn iterate_all_elements() {
        let mesh = make_mesh_2d_multi();
        let elements: Vec<_> = mesh.elements().collect();
        assert_eq!(elements.len(), 4);
    }

    #[test]
    fn iterate_by_dimension() {
        let mesh = make_mesh_2d_multi();
        let dim1_elements: Vec<_> = mesh.elements_of_dim(Dimension::D1).collect();
        let dim2_elements: Vec<_> = mesh.elements_of_dim(Dimension::D2).collect();
        assert_eq!(dim1_elements.len(), 2);
        assert_eq!(dim2_elements.len(), 2);
    }

    #[test]
    fn iterate_element_types() {
        let mesh = make_mesh_2d_multi();
        let types: Vec<_> = mesh.element_types().collect();
        assert_eq!(types.len(), 3);
        assert!(types.contains(&&ElementType::SEG2));
        assert!(types.contains(&&ElementType::QUAD4));
        assert!(types.contains(&&ElementType::PGON));
    }

    #[test]
    fn iterate_blocks() {
        let mesh = make_mesh_2d_multi();
        let blocks: Vec<_> = mesh.blocks().collect();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn get_specific_element() {
        let mesh = make_mesh_2d_multi();
        let elem = mesh.element(ElementId::new(ElementType::QUAD4, 0));
        assert_eq!(elem.element_type, ElementType::QUAD4);
        assert_eq!(elem.connectivity, &[0, 1, 3, 2]);
    }

    #[test]
    fn used_nodes() {
        let mesh = make_mesh_2d_multi();
        let used = mesh.used_nodes();
        assert_eq!(used.len(), 5);
        assert_eq!(used, vec![0, 1, 2, 3, 4]);
    }
}

mod mesh_views {
    use super::*;

    #[test]
    fn create_and_use_view() {
        let mesh = make_imesh_2d(10);
        let view = mesh.view();
        assert_eq!(view.num_elements(), mesh.num_elements());
    }

    #[test]
    fn view_to_owned() {
        let mesh = make_mesh_2d_quad();
        let view = mesh.view();
        let owned = view.to_shared();
        assert_eq!(owned.num_elements(), mesh.num_elements());
    }

    #[test]
    fn view_preserves_coordinates() {
        let mesh = make_mesh_2d_quad();
        let view = mesh.view();
        assert_eq!(view.coords().shape(), mesh.coords().shape());
    }
}

mod element_types {
    use super::*;

    #[test]
    fn dimension_of_element_types() {
        assert_eq!(ElementType::VERTEX.dimension(), Dimension::D0);
        assert_eq!(ElementType::SEG2.dimension(), Dimension::D1);
        assert_eq!(ElementType::TRI3.dimension(), Dimension::D2);
        assert_eq!(ElementType::QUAD4.dimension(), Dimension::D2);
        assert_eq!(ElementType::TET4.dimension(), Dimension::D3);
        assert_eq!(ElementType::HEX8.dimension(), Dimension::D3);
        assert_eq!(ElementType::PGON.dimension(), Dimension::D2);
        assert_eq!(ElementType::PHED.dimension(), Dimension::D3);
        assert_eq!(ElementType::SPLINE.dimension(), Dimension::D1);
    }

    #[test]
    fn regularity_of_element_types() {
        assert_eq!(ElementType::SEG2.regularity(), Regularity::Regular);
        assert_eq!(ElementType::QUAD4.regularity(), Regularity::Regular);
        assert_eq!(ElementType::PGON.regularity(), Regularity::Poly);
        assert_eq!(ElementType::PHED.regularity(), Regularity::Poly);
        assert_eq!(ElementType::SPLINE.regularity(), Regularity::Poly);
    }

    #[test]
    fn num_nodes_of_regular_elements() {
        assert_eq!(ElementType::VERTEX.num_nodes(), Some(1));
        assert_eq!(ElementType::SEG2.num_nodes(), Some(2));
        assert_eq!(ElementType::SEG3.num_nodes(), Some(3));
        assert_eq!(ElementType::TRI3.num_nodes(), Some(3));
        assert_eq!(ElementType::TRI6.num_nodes(), Some(6));
        assert_eq!(ElementType::QUAD4.num_nodes(), Some(4));
        assert_eq!(ElementType::QUAD8.num_nodes(), Some(8));
        assert_eq!(ElementType::QUAD9.num_nodes(), Some(9));
        assert_eq!(ElementType::TET4.num_nodes(), Some(4));
        assert_eq!(ElementType::TET10.num_nodes(), Some(10));
        assert_eq!(ElementType::HEX8.num_nodes(), Some(8));
        assert_eq!(ElementType::HEX21.num_nodes(), Some(21));
    }

    #[test]
    fn num_nodes_of_poly_elements() {
        assert_eq!(ElementType::PGON.num_nodes(), None);
        assert_eq!(ElementType::PHED.num_nodes(), None);
        assert_eq!(ElementType::SPLINE.num_nodes(), None);
    }

    #[test]
    fn element_id_creation_and_access() {
        let id = ElementId::new(ElementType::TRI3, 5);
        assert_eq!(id.element_type(), ElementType::TRI3);
        assert_eq!(id.index(), 5);
    }

    #[test]
    fn element_id_ordering() {
        let id1 = ElementId::new(ElementType::SEG2, 0);
        let id2 = ElementId::new(ElementType::TRI3, 0);
        let id3 = ElementId::new(ElementType::TRI3, 1);
        assert!(id1 < id2);
        assert!(id2 < id3);
    }
}

mod connectivity {
    use super::*;

    #[test]
    fn regular_connectivity_access() {
        let mesh = make_mesh_2d_quad();
        let conn = mesh.regular_connectivity(ElementType::QUAD4).unwrap();
        assert_eq!(conn.shape(), &[1, 4]);
        assert_eq!(conn[[0, 0]], 0);
        assert_eq!(conn[[0, 1]], 1);
        assert_eq!(conn[[0, 2]], 3);
        assert_eq!(conn[[0, 3]], 2);
    }

    #[test]
    fn regular_connectivity_error_on_poly() {
        let mesh = make_mesh_2d_multi();
        let result = mesh.regular_connectivity(ElementType::PGON);
        assert!(result.is_err());
    }

    #[test]
    fn poly_connectivity_access() {
        let mesh = make_mesh_2d_multi();
        let (data, offsets) = mesh.poly_connectivity(ElementType::PGON).unwrap();
        assert!(data.len() > 0);
        assert!(offsets.len() > 0);
    }

    #[test]
    fn poly_connectivity_error_on_regular() {
        let mesh = make_mesh_2d_quad();
        let result = mesh.poly_connectivity(ElementType::QUAD4);
        assert!(result.is_err());
    }
}

mod fields {
    use super::*;

    #[test]
    fn field_nonexistent_returns_none() {
        let mesh = make_mesh_2d_quad();
        let field = mesh.field("nonexistent", None);
        assert!(field.is_none());
    }

    #[test]
    fn update_and_retrieve_field() {
        let mut mesh = make_mesh_2d_quad();
        // First, add a field using update_field (this will insert if not exists)
        let field_data = nd::ArcArray::from_shape_vec(nd::IxDyn(&[1]), vec![42.0]).unwrap();
        let mut field_map = BTreeMap::new();
        field_map.insert(ElementType::QUAD4, field_data);
        let field_base = FieldBase::new(field_map);
        mesh.update_field("temperature", field_base, Some(Dimension::D2));
        let field = mesh.field("temperature", Some(Dimension::D2));
        assert!(field.is_some());
    }

    #[test]
    fn iterate_fields() {
        let mut mesh = make_mesh_2d_quad();
        let field_data = nd::ArcArray::from_shape_vec(nd::IxDyn(&[1]), vec![42.0]).unwrap();
        let mut field_map = BTreeMap::new();
        field_map.insert(ElementType::QUAD4, field_data);
        let field_base = FieldBase::new(field_map);
        mesh.update_field("temperature", field_base, Some(Dimension::D2));
        let field_names: Vec<_> = mesh.fields().map(|(name, _)| name).collect();
        assert_eq!(field_names.len(), 1);
        assert_eq!(field_names[0], "temperature");
    }

    #[test]
    fn remove_field() {
        let mut mesh = make_mesh_2d_quad();
        let field_data = nd::ArcArray::from_shape_vec(nd::IxDyn(&[1]), vec![42.0]).unwrap();
        let mut field_map = BTreeMap::new();
        field_map.insert(ElementType::QUAD4, field_data);
        let field_base = FieldBase::new(field_map);
        mesh.update_field("temperature", field_base, Some(Dimension::D2));
        let removed = mesh.remove_field("temperature", Some(Dimension::D2));
        assert!(removed.is_some());
        let field = mesh.field("temperature", Some(Dimension::D2));
        assert!(field.is_none());
    }
}

mod element_operations {
    use super::*;

    #[test]
    fn element_connectivity_equals() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem1 = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        let elem2 = Element::new(
            1,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        assert!(elem1.connectivity_equals(&elem2));
    }

    #[test]
    fn element_connectivity_not_equals() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn1 = &[0, 1, 2];
        let conn2 = &[2, 1, 0];
        let groups = BTreeMap::new();
        let family = 0;
        let elem1 = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn1,
            ElementType::TRI3,
        );
        let elem2 = Element::new(
            1,
            coords.view(),
            None,
            &family,
            &groups,
            conn2,
            ElementType::TRI3,
        );
        assert!(!elem1.connectivity_equals(&elem2));
    }

    #[test]
    fn element_groups_empty_by_default() {
        let coords = nd::array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let conn = &[0, 1, 2];
        let groups = BTreeMap::new();
        let family = 0;
        let elem = Element::new(
            0,
            coords.view(),
            None,
            &family,
            &groups,
            conn,
            ElementType::TRI3,
        );
        assert!(elem.groups().is_empty());
        assert!(!elem.in_group("test"));
    }
}

mod mesh_modification {
    use super::*;

    #[test]
    fn add_element_to_existing_block() {
        let mut mesh = make_mesh_2d_quad();
        let new_id = mesh.add_element(ElementType::QUAD4, &[0, 1, 3, 2], None, None);
        assert_eq!(mesh.num_elements(), 2);
        assert_eq!(new_id.element_type(), ElementType::QUAD4);
        assert_eq!(new_id.index(), 1);
    }

    #[test]
    fn add_element_creates_new_block() {
        let mut mesh = make_mesh_2d_quad();
        let new_id = mesh.add_element(ElementType::TRI3, &[0, 1, 2], None, None);
        assert_eq!(mesh.num_elements(), 2);
        assert_eq!(new_id.element_type(), ElementType::TRI3);
        assert_eq!(mesh.element_types().count(), 2);
    }

    #[test]
    #[should_panic(expected = "Connectivity length does not match")]
    fn add_element_wrong_connectivity_length() {
        let mut mesh = make_mesh_2d_quad();
        mesh.add_element(ElementType::TRI3, &[0, 1], None, None);
    }

    #[test]
    fn append_coordinate() {
        let mut mesh = make_mesh_2d_quad();
        let old_num_coords = mesh.coords().nrows();
        mesh.append_coord(nd::arr1(&[2.0, 0.0]).view()).unwrap();
        assert_eq!(mesh.coords().nrows(), old_num_coords + 1);
    }

    #[test]
    fn append_coordinates() {
        let mut mesh = make_mesh_2d_quad();
        let old_num_coords = mesh.coords().nrows();
        let new_coords = nd::arr2(&[[2.0, 0.0], [2.0, 1.0]]);
        mesh.append_coords(new_coords.view()).unwrap();
        assert_eq!(mesh.coords().nrows(), old_num_coords + 2);
    }
}

mod structured_grids {
    use super::*;

    #[test]
    fn create_2d_structured_grid() {
        let mesh = make_imesh_2d(5);
        assert!(mesh.num_elements() > 0);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D2));
    }

    #[test]
    fn create_3d_structured_grid() {
        let mesh = make_imesh_3d(5);
        assert!(mesh.num_elements() > 0);
        assert_eq!(mesh.topological_dimension(), Some(Dimension::D3));
    }

    #[test]
    fn structured_grid_2d_element_count() {
        let n = 5;
        let mesh = make_imesh_2d(n);
        let expected_elements = n * n;
        assert_eq!(mesh.num_elements(), expected_elements);
    }

    #[test]
    fn structured_grid_3d_element_count() {
        let n = 5;
        let mesh = make_imesh_3d(n);
        let expected_elements = n * n * n;
        assert_eq!(mesh.num_elements(), expected_elements);
    }
}

mod extract_mesh {
    use super::*;

    #[test]
    fn extract_single_element() {
        let mesh = make_mesh_2d_multi();
        let ids: ElementIds = vec![ElementId::new(ElementType::QUAD4, 0)].into_iter().collect();
        let extracted = mesh.extract(&ids, false);
        assert_eq!(extracted.num_elements(), 1);
        assert!(extracted.element_types().any(|&et| et == ElementType::QUAD4));
    }

    #[test]
    fn extract_multiple_elements_same_type() {
        let mesh = make_mesh_2d_multi();
        let ids: ElementIds = vec![
            ElementId::new(ElementType::SEG2, 0),
            ElementId::new(ElementType::SEG2, 1),
        ].into_iter().collect();
        let extracted = mesh.extract(&ids, false);
        assert_eq!(extracted.num_elements(), 2);
        assert_eq!(extracted.num_elements_of_dim(Dimension::D1), 2);
    }

    #[test]
    fn extract_multiple_elements_different_types() {
        let mesh = make_mesh_2d_multi();
        let ids: ElementIds = vec![
            ElementId::new(ElementType::SEG2, 0),
            ElementId::new(ElementType::QUAD4, 0),
        ].into_iter().collect();
        let extracted = mesh.extract(&ids, false);
        assert_eq!(extracted.num_elements(), 2);
        assert_eq!(extracted.element_types().count(), 2);
    }

    #[test]
    fn extract_nonexistent_element() {
        let mesh = make_mesh_2d_quad();
        let ids: ElementIds = vec![ElementId::new(ElementType::TRI3, 0)].into_iter().collect();
        let extracted = mesh.extract(&ids, false);
        assert_eq!(extracted.num_elements(), 0);
    }
}

mod selection {
    use super::*;
    use sel;

    #[test]
    fn select_all_elements() {
        let mesh = make_mesh_2d_multi();
        // Create an "all" selection using a wildcard approach - select by existing types
        let ids = mesh.select_ids(sel::types(vec![ElementType::SEG2, ElementType::QUAD4, ElementType::PGON, ElementType::TRI3, ElementType::TET4, ElementType::HEX8]));
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn select_by_element_type() {
        let mesh = make_mesh_2d_multi();
        let ids = mesh.select_ids(sel::types(vec![ElementType::SEG2]));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn select_by_dimension() {
        let mesh = make_mesh_2d_multi();
        let ids = mesh.select_ids(sel::dimensions(vec![Dimension::D2]));
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn select_by_node_ids() {
        let mesh = make_mesh_2d_multi();
        let ids = mesh.select_ids(sel::nids(vec![0, 1], false));
        assert!(ids.len() > 0);
    }
}

mod geometric_operations {
    use super::*;

    #[test]
    fn measure_quad_area() {
        let mesh = make_mesh_2d_quad();
        let measures = measure(mesh.view(), None);
        let quad_measures = measures.get(&ElementType::QUAD4).unwrap();
        assert_abs_diff_eq!(quad_measures[0], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn measure_multiple_quads() {
        // Create a mesh with only regular elements (QUAD4) for measure
        let mut mesh = make_mesh_2d_quad();
        mesh.add_element(ElementType::QUAD4, &[0, 1, 3, 2], None, None);
        let measures = measure(mesh.view(), None);
        let quad_measures = measures.get(&ElementType::QUAD4).unwrap();
        let total: f64 = quad_measures.iter().sum();
        assert!(total > 0.0);
        assert_abs_diff_eq!(total, 2.0, epsilon = 1e-10);
    }
}

mod snap_operations {
    use super::*;

    #[test]
    fn snap_coordinates() {
        let subject_coords = nd::Array2::from_shape_vec(
            (3, 2),
            vec![0.0, 0.0, 1.01, 0.0, 0.0, 1.01],
        )
        .unwrap();
        let mut subject = UMesh::new(subject_coords.clone().into());
        subject.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1], [1, 2]]).to_shared(),
            None,
        );

        let reference_coords = nd::Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.0, 1.0, 0.0],
        )
        .unwrap();
        let mut reference = UMesh::new(reference_coords.clone().into());
        reference.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1]]).to_shared(),
            None,
        );

        snap(&mut subject, reference.view(), 0.02);
        assert_abs_diff_eq!(subject.coords()[[1, 0]], 1.0, epsilon = 1e-6);
    }
}

mod element_block_operations {
    use super::*;

    #[test]
    fn block_len() {
        let mesh = make_mesh_2d_multi();
        let seg_block = mesh.block(ElementType::SEG2).unwrap();
        assert_eq!(seg_block.len(), 2);
    }

    #[test]
    fn block_element_access() {
        let mesh = make_mesh_2d_quad();
        let block = mesh.block(ElementType::QUAD4).unwrap();
        let elem = block.get(0, mesh.coords());
        assert_eq!(elem.connectivity, &[0, 1, 3, 2]);
    }

    #[test]
    fn block_iter() {
        let mesh = make_mesh_2d_multi();
        let block = mesh.block(ElementType::SEG2).unwrap();
        let elements: Vec<_> = block.iter(mesh.coords()).collect();
        assert_eq!(elements.len(), 2);
    }

    #[test]
    fn block_has_family() {
        let mesh = make_mesh_2d_quad();
        let block = mesh.block(ElementType::QUAD4).unwrap();
        assert_eq!(block.families.len(), 1);
    }
}

mod edge_cases {
    use super::*;

    #[test]
    fn empty_mesh_iteration() {
        let coords = nd::ArcArray2::from_shape_vec((0, 2), vec![]).unwrap();
        let mesh = UMesh::new(coords);
        let elements: Vec<_> = mesh.elements().collect();
        assert!(elements.is_empty());
    }

    #[test]
    fn single_element_mesh() {
        let mesh = make_mesh_2d_quad();
        assert_eq!(mesh.num_elements(), 1);
        let elements: Vec<_> = mesh.elements().collect();
        assert_eq!(elements.len(), 1);
    }

    #[test]
    fn mesh_with_duplicate_nodes() {
        let coords = nd::ArcArray2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_element(ElementType::PGON, &[0, 1, 4, 3, 2], None, None);
        let used = mesh.used_nodes();
        assert_eq!(used.len(), 5);
    }
}

mod mesh_info {
    use super::*;

    #[test]
    fn space_dimension() {
        let mesh = make_mesh_2d_quad();
        assert_eq!(mesh.space_dimension(), 2);
    }

    #[test]
    fn num_elements() {
        let mesh = make_mesh_2d_multi();
        assert_eq!(mesh.num_elements(), 4);
    }

    #[test]
    fn element_blocks_count() {
        let mesh = make_mesh_2d_multi();
        assert_eq!(mesh.element_types().count(), 3);
    }
}
