use rustc_hash::FxHashSet;

use super::selection::SelectedView;
use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;
use crate::mesh::ElementLike;

#[derive(Clone, Debug)]
pub enum NodeSelection {
    BBox {
        all: bool,
        min: [f64; 3],
        max: [f64; 3],
    }, // Axis aligned BBox
    Rect {
        all: bool,
        min: [f64; 2],
        max: [f64; 2],
    }, // Axis aligned BBox
    Sphere {
        all: bool,
        center: [f64; 3],
        r2: f64,
    }, // center and rayon
    Circle {
        all: bool,
        center: [f64; 2],
        r2: f64,
    }, // center and rayon
    Ids {
        all: bool,
        ids: Vec<usize>,
    },
}

impl NodeSelection {
    fn all_in<F0>(f: F0, selection: SelectedView) -> SelectedView
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let SelectedView(mview, index) = selection;
        let index = index
            .into_iter()
            .filter(|&e_id| mview.element(e_id).coords().all(&f))
            .collect();

        SelectedView(mview, index)
    }

    fn any_in<F0>(f: F0, selection: SelectedView) -> SelectedView
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        let SelectedView(mview, index) = selection;
        let index = index
            .into_iter()
            .filter(|&eid| mview.element(eid).coords().any(&f))
            .collect();
        SelectedView(mview, index)
    }
    fn in_shape<F0>(all: bool, f: F0, selection: SelectedView) -> SelectedView
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        if all {
            Self::all_in(f, selection)
        } else {
            Self::any_in(f, selection)
        }
    }
    pub fn in_sphere<'a>(
        all: bool,
        p0: &[f64; 3],
        r: f64,
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_shape(
            all,
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_sphere(
                    x.try_into().expect("Coords should have N components."),
                    p0,
                    r,
                )
            },
            selection,
        )
    }

    pub fn in_bbox<'a>(
        all: bool,
        p0: &[f64; 3],
        p1: &[f64; 3],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_shape(
            all,
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_aa_bbox(
                    x.try_into().expect("Coords should have 3 components."),
                    p0,
                    p1,
                )
            },
            selection,
        )
    }
    pub fn in_circle<'a>(
        all: bool,
        p0: &[f64; 2],
        r: f64,
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_shape(
            all,
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_circle(
                    x.try_into().expect("Coords should have N components."),
                    p0,
                    r,
                )
            },
            selection,
        )
    }

    pub fn in_rectangle<'a>(
        all: bool,
        p0: &[f64; 2],
        p1: &[f64; 2],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_shape(
            all,
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_aa_rectangle(
                    x.try_into().expect("Coords should have 2 components."),
                    p0,
                    p1,
                )
            },
            selection,
        )
    }
    fn any_id_in<'a>(nodes_ids: &[usize], selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(mview, index) = selection;
        let index = if nodes_ids.len() < 50 {
            index
                .into_iter()
                .filter(|&e_id| {
                    nodes_ids
                        .iter()
                        .any(|n| mview.element(e_id).connectivity().contains(n))
                })
                .collect()
        } else {
            let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
            nodes_ids.sort_unstable();

            index
                .into_iter()
                .filter(|&e_id| {
                    mview
                        .element(e_id)
                        .connectivity()
                        .iter()
                        .any(|n| nodes_ids.binary_search(n).is_ok())
                })
                .collect()
        };
        SelectedView(mview, index)
    }

    fn all_id_in<'a>(nodes_ids: &[usize], selection: SelectedView<'a>) -> SelectedView<'a> {
        let SelectedView(mview, index) = selection;
        let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();

        let index = index
            .into_iter()
            .filter(|&e_id| {
                mview
                    .element(e_id)
                    .connectivity()
                    .iter()
                    .all(|n| nodes_ids.contains(n))
            })
            .collect();
        SelectedView(mview, index)
    }

    pub fn id_in<'a>(
        all: bool,
        nodes_ids: &[usize],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        if all {
            Self::all_id_in(nodes_ids, selection)
        } else {
            Self::any_id_in(nodes_ids, selection)
        }
    }
}
