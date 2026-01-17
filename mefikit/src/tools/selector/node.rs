pub enum NodeSelection<const N: usize> {
    InBBox {
        all: bool,
        min: [f64; N],
        max: [f64; N],
    }, // Axis aligned BBox
    InSphere {
        all: bool,
        center: [f64; N],
        r2: f64,
    }, // center and rayon
    InIds {
        all: bool,
        ids: Vec<usize>,
    },
}

impl<const N: usize> NodeSelection<N> {
    // fn all_in<F0>(self, f: F0, selection: SelectedView) -> SelectedView
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //      let SelectedView(mview, index) = selection;
    //     let index = index
    //         .into_par_iter()
    //         .filter(|&e_id| mview.element(e_id).coords().all(&f))
    //         .collect();

    //      SelectedView(mview, index)
    // }

    // fn any_in<F0>(self, f: F0, selection: SelectedView) -> SelectedView
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //     let SelectedView(mview, index) = selection;
    //     let index = index
    //         .into_iter()
    //         .flat_map(|(t, eids)| std::iter::repeat(t).zip(eids))
    //         .filter(|(&t, &eid)| mview.element(ElementId::new(t, eid)).coords().any(&f))
    //         .collect();
    //     SelectedView(mview, index)

    // }

    // pub fn in_shape<F0>(self, f: F0) -> Self
    // where
    //     F0: Fn(&[f64]) -> bool + Sync,
    // {
    //     if self.state.all_nodes {
    //         self.all_in(f)
    //     } else {
    //         self.any_in(f)
    //     }
    // }

    // pub fn in_sphere(self, p0: &[f64; 3], r: f64) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 3);
    //         geo::in_sphere(
    //             x.try_into().expect("Coords should have 3 components."),
    //             p0,
    //             r,
    //         )
    //     })
    // }

    // pub fn in_bbox(self, p0: &[f64; 3], p1: &[f64; 3]) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 3);
    //         geo::in_aa_bbox(
    //             x.try_into().expect("Coords should have 3 components."),
    //             p0,
    //             p1,
    //         )
    //     })
    // }

    // pub fn in_rectangle(self, p0: &[f64; 2], p1: &[f64; 2]) -> Self {
    //     self.in_shape(|x| {
    //         debug_assert_eq!(x.len(), 2);
    //         geo::in_aa_rectangle(
    //             x.try_into().expect("Coords should have 2 components."),
    //             p0,
    //             p1,
    //         )
    //     })
    // }

    // fn any_id_in(self, nodes_ids: &[usize]) -> Self {
    //     let index = if nodes_ids.len() < 50 {
    //         self.index
    //             .into_iter()
    //             .filter(|&e_id| {
    //                 nodes_ids
    //                     .iter()
    //                     .any(|n| self.umesh.element(e_id).connectivity().contains(n))
    //             })
    //             .collect()
    //     } else {
    //         let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
    //         nodes_ids.sort_unstable();

    //         self.index
    //             .into_iter()
    //             .filter(|&e_id| {
    //                 self.umesh
    //                     .element(e_id)
    //                     .connectivity()
    //                     .iter()
    //                     .any(|n| nodes_ids.binary_search(n).is_ok())
    //             })
    //             .collect()
    //     };
    // }

    // fn all_id_in(self, nodes_ids: &[usize]) -> Self {
    //     let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();

    //     let index = self
    //         .index
    //         .into_par_iter()
    //         .filter(|&e_id| {
    //             self.umesh
    //                 .element(e_id)
    //                 .connectivity()
    //                 .iter()
    //                 .all(|n| nodes_ids.contains(n))
    //         })
    //         .collect();
    // }

    // pub fn id_in(self, nodes_ids: &[usize]) -> Self {
    //     let all = self.state.all_nodes;
    //     if all {
    //         self.all_id_in(nodes_ids)
    //     } else {
    //         self.any_id_in(nodes_ids)
    //     }
    // }
}
