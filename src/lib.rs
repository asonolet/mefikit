use ndarray::{s, concatenate, Axis, Array1, Array2, ArrayD};
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

    pub fn insert(
        &mut self,
        i: usize,
        coord: Array1<f64>,
        fam: usize,
        // node_data: Option<HashMap<String, ArrayD<f64>>>,
    ) {
        // --- Insert into coords ---
        let coord_row = coord.insert_axis(Axis(0));
        let before = self.coords.slice(s![..i, ..]).to_owned();
        let after = self.coords.slice(s![i.., ..]).to_owned();
        self.coords = concatenate![Axis(0), before, coord_row, after];

        // --- Insert into node_families ---
        let mut families = self.node_families.to_vec();
        families.insert(i, fam);
        self.node_families = Array1::from(families);

        // --- Insert into node_data ---
        // for (name, data) in &mut self.node_data {
        //     let shape = data.shape();
        //     let mut slices: Vec<_> = (0..shape.len()).map(|_| s![..]).collect();

        //     // Extract before and after slices
        //     let before = data.slice(&slices[..]).slice_axis(Axis(0), s![..i]);
        //     let after = data.slice(&slices[..]).slice_axis(Axis(0), s![i..]);

        //     // Determine what to insert: user data or zeros
        //     let insert_value = if let Some(new_data) = &node_data {
        //         if let Some(val) = new_data.get(name) {
        //             val.clone().into_dyn()
        //         } else {
        //             ArrayD::zeros(IxDyn(&shape[1..]))
        //         }
        //     } else {
        //         ArrayD::zeros(IxDyn(&shape[1..]))
        //     };

        //     // Expand insert_value to have shape [1, D1, D2, ...]
        //     let insert_value = insert_value.insert_axis(Axis(0));

        //     // Re-stack along axis 0
        //     let new_data = stack(Axis(0), &[before, insert_value.view(), after]).unwrap();
        //     *data = new_data;
        // }

        // // --- If new keys exist in user input, add them ---
        // if let Some(new_fields) = node_data {
        //     for (name, new_val) in new_fields {
        //         if !self.node_data.contains_key(&name) {
        //             // Insert default 0s for existing nodes
        //             let shape = new_val.shape();
        //             let mut full_shape = vec![self.coords.nrows() - 1];
        //             full_shape.extend_from_slice(&shape[1..]);
        //             let mut new_array = ArrayD::zeros(full_shape.into());

        //             // Insert the given value at index i
        //             let insert_val = new_val.insert_axis(Axis(0));
        //             let before = new_array.slice_axis(Axis(0), s![..i]);
        //             let after = new_array.slice_axis(Axis(0), s![i..]);
        //             let stacked = stack(Axis(0), &[before, insert_val.view(), after]).unwrap();
        //             self.node_data.insert(name, stacked);
        //         }
        //     }
        // }
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
