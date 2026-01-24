use crate::mesh::ElementLike;
use crate::tools::sel::SelectedView;

#[derive(Clone, Debug)]
pub enum GroupSelection {
    IncludeGroup(String),
    ExcludeGroup(String),
    IncludeFamily(usize),
    ExcludeFamily(usize),
}

impl GroupSelection {
    pub fn include_group<'a>(group: &str, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, sel) = selection;
        let index = sel
            .into_iter()
            .filter(|&eid| view.element(eid).in_group(group))
            .collect();
        SelectedView(view, index)
    }
    pub fn exclude_group<'a>(group: &str, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, sel) = selection;
        let index = sel
            .into_iter()
            .filter(|&eid| !view.element(eid).in_group(group))
            .collect();
        SelectedView(view, index)
    }
    pub fn include_family<'a>(family: usize, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, sel) = selection;
        let index = sel
            .into_iter()
            .filter(|&eid| *view.element(eid).family == family)
            .collect();
        SelectedView(view, index)
    }
    pub fn exclude_family<'a>(family: usize, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(view, sel) = selection;
        let index = sel
            .into_iter()
            .filter(|&eid| *view.element(eid).family != family)
            .collect();
        SelectedView(view, index)
    }
}
