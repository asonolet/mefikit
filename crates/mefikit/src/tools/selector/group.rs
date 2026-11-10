use crate::mesh::{ElementIdsSet, UMeshView};

#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum GroupSelection {
    IncludeGroup(String),
    ExcludeGroup(String),
}

impl GroupSelection {
    pub fn include_group(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| view.in_group(eid, group))
            .collect()
    }
    pub fn exclude_group(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| !view.in_group(eid, group))
            .collect()
    }
}
