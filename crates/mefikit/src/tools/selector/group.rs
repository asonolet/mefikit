use crate::mesh::{ElementIdsSet, ElementLike, UMeshView};

#[derive(Clone, Debug)]
pub enum GroupSelection {
    IncludeGroup(String),
    ExcludeGroup(String),
    IncludeFamily(usize),
    ExcludeFamily(usize),
}

impl GroupSelection {
    pub fn include_group(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| view.element(eid).in_group(group))
            .collect()
    }
    pub fn exclude_group(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| !view.element(eid).in_group(group))
            .collect()
    }
    pub fn include_family(family: usize, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| *view.element(eid).family == family)
            .collect()
    }
    pub fn exclude_family(family: usize, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| *view.element(eid).family != family)
            .collect()
    }
}
