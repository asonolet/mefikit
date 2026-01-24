use std::sync::Arc;

use super::selection::{Select, SelectedView, Selection};

use crate::mesh::ElementIdsSet;

#[derive(Copy, Clone, Debug)]
pub enum BooleanOp {
    And,
    Or,
    Xor,
    Diff,
}

#[derive(Clone, Debug)]
pub struct BinarayExpr {
    pub operator: BooleanOp,
    pub left: Arc<Selection>,
    pub right: Arc<Selection>,
}

#[derive(Clone, Debug)]
pub struct NotExpr(pub Arc<Selection>);

impl BinarayExpr {
    pub fn and_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        if self.left.weight() < self.right.weight() {
            let selection = self.left.select(selection);
            self.right.select(selection)
        } else {
            let selection = self.right.select(selection);
            self.left.select(selection)
        }
    }
    pub fn or_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let SelectedView(mview, mut sel1) = self.left.select(selection.clone());
        let SelectedView(_, sel2) = self.right.select(selection);
        sel1.union(&sel2);
        SelectedView(mview, sel1)
    }
    pub fn xor_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let SelectedView(mview, mut sel1) = self.left.select(selection.clone());
        let SelectedView(_, sel2) = self.right.select(selection);
        sel1.symmetric_difference(&sel2);
        SelectedView(mview, sel1)
    }
    pub fn diff_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let SelectedView(mview, mut sel1) = self.left.select(selection.clone());
        let SelectedView(_, sel2) = self.right.select(selection);
        sel1.difference(&sel2);
        SelectedView(mview, sel1)
    }
}

impl NotExpr {
    pub fn not_select<'a>(&self, selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(mview, mut sel0) = selection;
        let all_ids: ElementIdsSet = ElementIdsSet(
            mview
                .blocks()
                .map(|(k, v)| (*k, (0..v.len()).collect()))
                .collect(),
        );
        let SelectedView(mview, sel) = self.0.select(SelectedView(mview, all_ids));
        // let mut not_sel = all_ids;
        // not_sel.difference(&sel);
        // sel0.intersection(&not_sel);
        sel0.difference(&sel);
        SelectedView(mview, sel0)
    }
}
