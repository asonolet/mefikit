use std::sync::Arc;

use super::selection::{Select, SelectedView, Selection};

#[derive(Copy, Clone)]
pub enum BooleanOp {
    Eq,
    And,
    Or,
    Xor,
}

pub struct BinarayExpr<const N: usize> {
    pub operator: BooleanOp,
    pub left: Arc<Selection<N>>,
    pub right: Arc<Selection<N>>,
}

pub struct NotExpr<const N: usize>(pub Arc<Selection<N>>);

impl<const N: usize> BinarayExpr<N> {
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
        // TODO: spawn one thread per selection
        let selection1 = self.left.select(selection.clone());
        let _selection2 = self.right.select(selection);
        //TODO: return the union of both
        selection1
    }
}
