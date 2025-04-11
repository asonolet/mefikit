use ndarray::{Array1, Array2, ArrayD};
use std::collections::HashMap;

pub struct PointCloud {
    coords: Array2<f64>, // 2D array for coordinates (n_nodes x dim)
    node_families: Array1<usize>, // 1D array for families per node
    node_groups: HashMap<String, Array1<usize>>, // group name to node indices
    node_data: HashMap<String, ArrayD<f64>>, // node metadata
}

impl PointCloud {
    pub fn new(
        coords: Array2<f64>,
        node_groups: Option<HashMap<String, Array1<usize>>>,
        node_data: Option<HashMap<String, ArrayD<f64>>>,
    ) -> Self {
        let n_nodes = coords.nrows();
        let mut pc = Self {
            coords,
            node_families: Array1::zeros(n_nodes),
            node_groups: HashMap::new(),
            node_data: node_data.unwrap_or_default(),
        };

        if let Some(groups) = node_groups {
            for (name, node_ids) in groups {
                pc.add_group(name, node_ids);
            }
        }

        pc
    }


    // Getter methods for private fields
    pub fn coords(&self) -> &Array2<f64> {
        &self.coords
    }

    pub fn node_families(&self) -> &Array1<usize> {
        &self.node_families
    }

    pub fn node_groups(&self) -> &HashMap<String, Array1<usize>> {
        &self.node_groups
    }

    pub fn node_data(&self) -> &HashMap<String, ArrayD<f64>> {
        &self.node_data
    }

    pub fn add_group(&mut self, grpname: String, node_ids: Array1<usize>) {
        self.node_groups.insert(grpname, node_ids);
    }


    pub fn append(&mut self, coord: Array1<f64>, fam: usize, node_data: Option<HashMap<String, ArrayD<f64>>>) {
        self.coords.append(ndarray::Axis(0), coord.insert_axis(ndarray::Axis(0)).view()).unwrap();
        self.node_families.append(ndarray::Axis(0), Array1::from(vec![fam]).view()).unwrap();
        if let Some(data) = node_data {
            for (k, v) in data {
                self.node_data.insert(k, v);
            }
        }
    }

    pub fn pop(&mut self, i: usize, get_info: bool) -> (Array1<f64>, Option<usize>, Option<HashMap<String, ArrayD<f64>>>) {
        let node = self.coords.row(i).to_owned();
        let fam = self.node_families[i];

        // Note: real removal not implemented, only demonstration
        let node_info = if get_info {
            Some(
                self.node_data
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect(),
            )
        } else {
            None
        };

        (node, Some(fam), node_info)
    }
}
