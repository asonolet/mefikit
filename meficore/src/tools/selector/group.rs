use crate::mesh::{ElementIdsSet, ElementLike, UMeshView};
use crate::tools::sel::SelectedView;

#[derive(Clone, Debug)]
pub enum GroupSelection {
    IncludeGroup(String),
    ExcludeGroup(String),
    IncludeFamily(usize),
    ExcludeFamily(usize),
}

impl GroupSelection {
    pub fn include_group<'a>(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| view.element(eid).in_group(group))
            .collect()
    }
    pub fn exclude_group<'a>(group: &str, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| !view.element(eid).in_group(group))
            .collect()
    }
    pub fn include_family<'a>(
        family: usize,
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| *view.element(eid).family == family)
            .collect()
    }
    pub fn exclude_family<'a>(
        family: usize,
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        sel.into_iter()
            .filter(|&eid| *view.element(eid).family != family)
            .collect()
    }
}
