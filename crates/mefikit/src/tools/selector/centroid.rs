use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;
use crate::mesh::{ElementIdsSet, UMeshView};

#[derive(Clone, Debug)]
pub enum CentroidSelection {
    BBox { min: [f64; 3], max: [f64; 3] }, // Axis aligned BBox
    Rect { min: [f64; 2], max: [f64; 2] }, // Axis aligned BBox
    Sphere { center: [f64; 3], r2: f64 },  // center and rayon
    Circle { center: [f64; 2], r2: f64 },  // center and rayon
}

impl CentroidSelection {
    fn in_2d<'a, F0>(f: F0, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64; 2]) -> bool + Sync,
    {
        sel.into_iter()
            .filter(|&e_id| f(&view.element(e_id).centroid2()))
            .collect()
    }
    fn in_3d<'a, F0>(f: F0, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64; 3]) -> bool + Sync,
    {
        sel.into_iter()
            .filter(|&e_id| f(&view.element(e_id).centroid3()))
            .collect()
    }

    pub fn in_sphere<'a>(
        p0: &[f64; 3],
        r: f64,
        view: &'a UMeshView<'a>,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        Self::in_3d(
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_sphere(x, p0, r)
            },
            view,
            sel,
        )
    }
    pub fn in_bbox<'a>(
        p0: &[f64; 3],
        p1: &[f64; 3],
        view: &'a UMeshView<'a>,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        Self::in_3d(
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_aa_bbox(x, p0, p1)
            },
            view,
            sel,
        )
    }

    pub fn in_circle<'a>(
        p0: &[f64; 2],
        r: f64,
        view: &'a UMeshView<'a>,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        Self::in_2d(
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_circle(x, p0, r)
            },
            view,
            sel,
        )
    }

    pub fn in_rectangle<'a>(
        p0: &[f64; 2],
        p1: &[f64; 2],
        view: &'a UMeshView<'a>,
        sel: ElementIdsSet,
    ) -> ElementIdsSet {
        Self::in_2d(
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_aa_rectangle(x, p0, p1)
            },
            view,
            sel,
        )
    }
}
