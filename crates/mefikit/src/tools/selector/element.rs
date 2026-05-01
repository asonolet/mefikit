use rustc_hash::FxHashSet;

use crate::mesh::{Dimension, ElementIds, ElementIdsSet, ElementType};

#[derive(Clone, Debug)]
pub enum ElementSelection {
    Types(Vec<ElementType>),
    InIds(ElementIds),
    Dimensions(Vec<Dimension>),
}

impl ElementSelection {
    pub fn select_types(types: &[ElementType], mut sel: ElementIdsSet) -> ElementIdsSet {
        let sel_types: Vec<_> = sel.keys().collect();
        let types_to_match: FxHashSet<_> = types.iter().collect();
        for k in sel_types {
            if !types_to_match.contains(&k) {
                sel.remove_type(k);
            }
        }
        sel
    }
    pub fn select_dimensions(dims: &[Dimension], mut sel: ElementIdsSet) -> ElementIdsSet {
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
    pub fn select_ids(ids: ElementIds, mut sel: ElementIdsSet) -> ElementIdsSet {
        let ids: ElementIdsSet = ids.into();
        sel.intersection(&ids);
        sel
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementType;

    #[test]
    fn test_select_types() {
        let mut ids = ElementIds::new();
        ids.add(ElementType::QUAD4, 0);
        ids.add(ElementType::SEG2, 0);
        ids.add(ElementType::SEG2, 1);

        let sel: ElementIdsSet = ids.into();
        let result = ElementSelection::select_types(&[ElementType::QUAD4], sel);
        assert!(result.contains_type(ElementType::QUAD4));
        assert!(!result.contains_type(ElementType::SEG2));
    }

    #[test]
    fn test_select_dimensions() {
        let mut ids = ElementIds::new();
        ids.add(ElementType::QUAD4, 0);
        ids.add(ElementType::SEG2, 0);

        let sel: ElementIdsSet = ids.into();
        let result = ElementSelection::select_dimensions(&[Dimension::D1], sel);
        assert!(result.contains_type(ElementType::SEG2));
        assert!(!result.contains_type(ElementType::QUAD4));
    }

    #[test]
    fn test_select_ids() {
        let mut ids = ElementIds::new();
        ids.add(ElementType::QUAD4, 0);
        ids.add(ElementType::QUAD4, 1);
        ids.add(ElementType::SEG2, 0);

        let mut filter_ids = ElementIds::new();
        filter_ids.add(ElementType::QUAD4, 0);

        let sel: ElementIdsSet = ids.into();
        let result = ElementSelection::select_ids(filter_ids, sel);
        assert_eq!(result.len(), 1);
    }
}
