use ndarray::{Array1, Array2, ArrayD, ArrayViewD, Axis};
use std::collections::HashMap;
use std::collections::HashSet;

use crate::mesh_element::ElementType;

pub trait ElementBlockLike {
    fn len(&self) -> usize;
    fn params(&self) -> &HashMap<String, f64>;
    fn fields(&self) -> &HashMap<String, ArrayD<f64>>;
    fn families(&self) -> &Array1<usize>;
    fn groups(&self) -> &HashMap<String, HashSet<usize>>;
    fn iter_elements<'a>(
        &'a self,
        coords: &'a Array2<f64>,
    ) -> Box<dyn Iterator<Item = MeshElementView<'a>> + 'a>;
    fn compo_type(&self) -> ElementType;
    fn element_fields(&self, i: usize) -> HashMap<String, ArrayViewD<f64>> {
        self.fields()
            .iter()
            .map(|(k, v)| (k.clone(), v.index_axis(Axis(0), i)))
            .collect()
    }
    fn element_family_and_groups(&self, i: usize) -> (usize, HashSet<String>) {
        let families = self.families();
        let groups = self.groups();

        let family = families[i];
        let group_names = groups
            .iter()
            .filter_map(|(name, fset)| {
                if fset.contains(&family) {
                    Some(name.clone())
                } else {
                    None
                }
            })
            .collect();

        (family, group_names)
    }
}
