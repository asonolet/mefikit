use rustc_hash::FxHashSet;
use std::collections::BTreeMap;

use super::SelectedView;
use crate::mesh::{Dimension, ElementId, ElementIds, ElementType};

type ElementIdsSet = BTreeMap<ElementType, FxHashSet<usize>>;

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
        for k in types {
            sel.remove(k);
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
                key_toremove.push(*k);
            }
        }
        for k in key_toremove {
            sel.remove(&k);
        }
        SelectedView(view, sel)
    }
    pub fn select_ids<'a>(ids: &ElementIds, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, index) = selection;
        // TODO: zip on keys and intersect on Ids
        SelectedView(view, index)
    }
}
