use rustc_hash::FxHashSet;

use super::selection::SelectedView;
use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType, UMeshView};

#[derive(Clone, Debug)]
pub enum ElementSelection {
    Types(Vec<ElementType>),
    InIds(ElementIds),
    Dimensions(Vec<Dimension>),
}

impl ElementSelection {
    pub fn select_types<'a>(types: &[ElementType], mut sel: ElementIdsSet) -> ElementIdsSet {
        let sel_types: Vec<_> = sel.keys().collect();
        let types_to_match: FxHashSet<_> = types.iter().collect();
        for k in sel_types {
            if !types_to_match.contains(&k) {
                sel.remove_type(k);
            }
        }
        sel
    }
    pub fn select_dimensions<'a>(dims: &[Dimension], mut sel: ElementIdsSet) -> ElementIdsSet {
        let mut key_toremove = Vec::new();
        for k in sel.keys() {
            if !dims.contains(&k.dimension()) {
                key_toremove.push(k);
            }
        }
        for k in key_toremove {
            sel.remove_type(k);
        }
        sel
    }
    pub fn select_ids<'a>(ids: ElementIds, mut sel: ElementIdsSet) -> ElementIdsSet {
        let ids: ElementIdsSet = ids.into();
        sel.intersection(&ids);
        sel
    }
}
