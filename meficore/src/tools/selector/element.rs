use rustc_hash::FxHashSet;

use super::selection::SelectedView;
use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType};

#[derive(Clone, Debug)]
pub enum ElementSelection {
    Types(Vec<ElementType>),
    InIds(ElementIds),
    Dimensions(Vec<Dimension>),
}

impl ElementSelection {
    pub fn select_types<'a>(
        types: &[ElementType],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        let SelectedView(view, mut sel) = selection;
        let sel_types: Vec<_> = sel.keys().collect();
        let types_to_match: FxHashSet<_> = types.iter().collect();
        for k in sel_types {
            if !types_to_match.contains(&k) {
                sel.remove_type(k);
            }
        }
        SelectedView(view, sel)
    }
    pub fn select_dimensions<'a>(
        dims: &[Dimension],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        let SelectedView(view, mut sel) = selection;
        let mut key_toremove = Vec::new();
        for k in sel.keys() {
            if !dims.contains(&k.dimension()) {
                key_toremove.push(k);
            }
        }
        for k in key_toremove {
            sel.remove_type(k);
        }
        SelectedView(view, sel)
    }
    pub fn select_ids<'a>(ids: ElementIds, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, mut index) = selection;
        let ids: ElementIdsSet = ids.into();
        index.intersection(&ids);
        SelectedView(view, index)
    }
}
