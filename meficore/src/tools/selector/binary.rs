use std::sync::Arc;

use super::selection::{Select, SelectedView, Selection};

#[derive(Copy, Clone, Debug)]
pub enum BooleanOp {
    Eq,
    And,
    Or,
    Xor,
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
}
