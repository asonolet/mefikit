#[cfg(test)]
mod tests {
    
    use ndarray::{array};
    use mefikit::PointCloud;

    #[test]
    fn test_new_pointcloud() {
        let coords = array![[0.0, 1.0], [2.0, 3.0]];
        let pc = PointCloud::new(coords.clone(), None, None);

        assert_eq!(pc.coords().shape(), &[2, 2]);
        assert_eq!(pc.node_families().len(), 2);
        assert!(pc.node_groups().is_empty());
        assert!(pc.node_data().is_empty());
    }

    #[test]
    fn test_add_group() {
        let coords = array![[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]];
        let mut pc = PointCloud::new(coords, None, None);

        let group_indices = array![0, 2];
        pc.add_group("boundary".to_string(), group_indices.clone());

        assert!(pc.node_groups().contains_key("boundary"));
        assert_eq!(pc.node_groups()["boundary"], group_indices);
    }

    #[test]
    fn test_append_node() {
        let coords = array![[0.0, 0.0]];
        let mut pc = PointCloud::new(coords, None, None);

        let new_coord = array![1.0, 1.0];
        pc.append(new_coord.clone(), 1, None);

        assert_eq!(pc.coords().shape(), &[2, 2]);
        assert_eq!(pc.coords().row(1), new_coord.view());
        assert_eq!(pc.node_families()[1], 1);
    }

    #[test]
    fn test_pop_node() {
        let coords = array![[1.0, 2.0], [3.0, 4.0]];
        let mut pc = PointCloud::new(coords, None, None);

        let (popped, fam, _) = pc.pop(1, true);
        assert_eq!(popped, array![3.0, 4.0]);
        assert_eq!(fam, Some(0)); // default family
    }

}

