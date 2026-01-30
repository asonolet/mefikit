use std::{sync::Arc, thread};

use super::selection::{Select, SelectedView, Selection};

use crate::mesh::{ElementIdsSet, UMeshView};

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
    pub fn and_select<'a>(&'a self, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet {
        if self.left.weight() < self.right.weight() {
            let selection = self.left.select(view, sel);
            self.right.select(view, selection)
        } else {
            let selection = self.right.select(view, sel);
            self.left.select(view, selection)
        }
    }
    pub fn or_select<'a>(&'a self, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let mut sel1 = self.left.select(view, sel.clone());
        let sel2 = self.right.select(view, sel);
        sel1.union(&sel2);
        sel1
    }
    pub fn xor_select<'a>(&'a self, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let mut sel1 = self.left.select(view, sel.clone());
        let sel2 = self.right.select(view, sel);
        sel1.symmetric_difference(&sel2);
        sel1
    }
    pub fn diff_select<'a>(&'a self, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet {
        // TODO: spawn one thread per selection so that they are computed in parallel
        let mut sel1 = self.left.select(view, sel.clone());
        let sel2 = self.right.select(view, sel);
        sel1.difference(&sel2);
        sel1
    }
}

impl NotExpr {
    pub fn not_select<'a>(
        &'a self,
        view: &'a UMeshView<'a>,
        mut sel: ElementIdsSet,
    ) -> ElementIdsSet {
        let all_ids: ElementIdsSet = ElementIdsSet(
            view.blocks()
                .map(|(k, v)| (*k, (0..v.len()).collect()))
                .collect(),
        );
        let not_sel = self.0.select(view, all_ids);
        // let mut not_sel = all_ids;
        // not_sel.difference(&sel);
        // sel0.intersection(&not_sel);
        sel.difference(&not_sel);
        sel
    }
}
