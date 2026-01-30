use rustc_hash::FxHashSet;

use super::selection::SelectedView;
use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;
use crate::mesh::ElementIds;
use crate::mesh::ElementIdsSet;
use crate::mesh::ElementLike;
use crate::mesh::UMeshView;

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
        r: f64,
    }, // center and rayon
    Circle {
        all: bool,
        center: [f64; 2],
        r: f64,
    }, // center and rayon
    Ids {
        all: bool,
        ids: Vec<usize>,
    },
}

impl NodeSelection {
    fn all_in<F0>(f: F0, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        sel.into_iter()
            .filter(|&e_id| view.element(e_id).coords().all(&f))
            .collect()
    }

    fn any_in<F0>(f: F0, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        sel.into_iter()
            .filter(|&eid| view.element(eid).coords().any(&f))
            .collect()
    }
    fn in_shape<F0>(all: bool, f: F0, view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64]) -> bool + Sync,
    {
        if all {
            Self::all_in(f, view, sel)
        } else {
            Self::any_in(f, view, sel)
        }
    }
    pub fn in_sphere<'a>(
        all: bool,
        p0: &[f64; 3],
        r: f64,
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
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
            view,
            sel,
        )
    }

    pub fn in_bbox<'a>(
        all: bool,
        p0: &[f64; 3],
        p1: &[f64; 3],
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
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
            view,
            sel,
        )
    }
    pub fn in_circle<'a>(
        all: bool,
        p0: &[f64; 2],
        r: f64,
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
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
            view,
            sel,
        )
    }

    pub fn in_rectangle<'a>(
        all: bool,
        p0: &[f64; 2],
        p1: &[f64; 2],
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
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
            view,
            sel,
        )
    }
    fn any_id_in<'a>(nodes_ids: &[usize], view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        if nodes_ids.len() < 50 {
            sel.into_iter()
                .filter(|&e_id| {
                    nodes_ids
                        .iter()
                        .any(|n| view.element(e_id).connectivity().contains(n))
                })
                .collect()
        } else {
            let mut nodes_ids: Vec<usize> = nodes_ids.to_vec();
            nodes_ids.sort_unstable();

            sel.into_iter()
                .filter(|&e_id| {
                    view.element(e_id)
                        .connectivity()
                        .iter()
                        .any(|n| nodes_ids.binary_search(n).is_ok())
                })
                .collect()
        }
    }

    fn all_id_in<'a>(nodes_ids: &[usize], view: &UMeshView, sel: ElementIdsSet) -> ElementIdsSet {
        let nodes_ids: FxHashSet<usize> = nodes_ids.iter().cloned().collect();

        let index = sel
            .into_iter()
            .filter(|&e_id| {
                view.element(e_id)
                    .connectivity()
                    .iter()
                    .all(|n| nodes_ids.contains(n))
            })
            .collect();
        index
    }

    pub fn id_in<'a>(
        all: bool,
        nodes_ids: &[usize],
        view: &UMeshView,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        if all {
            Self::all_id_in(nodes_ids, view, sel)
        } else {
            Self::any_id_in(nodes_ids, view, sel)
        }
    }
}
