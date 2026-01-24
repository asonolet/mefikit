#[derive(Clone, Debug)]
pub enum GroupSelection {
    IncludeGroups(Vec<String>),
    ExcludeGroups(Vec<String>),
    IncludeFamilies(Vec<usize>),
    ExcludeFamilies(Vec<usize>),
}
