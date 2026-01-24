use super::selection::SelectedView;
use crate::element_traits::ElementGeo;
use crate::element_traits::is_in as geo;

pub enum CentroidSelection {
    BBox { min: [f64; 3], max: [f64; 3] }, // Axis aligned BBox
    Rect { min: [f64; 2], max: [f64; 2] }, // Axis aligned BBox
    Sphere { center: [f64; 3], r2: f64 },  // center and rayon
    Circle { center: [f64; 2], r2: f64 },  // center and rayon
}

impl CentroidSelection {
    fn in_2d<F0>(f: F0, selection: SelectedView) -> SelectedView
    where
        F0: Fn(&[f64; 2]) -> bool + Sync,
    {
        let SelectedView(mview, index) = selection;
        let index = index
            .into_iter()
            .filter(|&e_id| f(&mview.element(e_id).centroid2()))
            .collect();

        SelectedView(mview, index)
    }
    fn in_3d<F0>(f: F0, selection: SelectedView) -> SelectedView
    where
        F0: Fn(&[f64; 3]) -> bool + Sync,
    {
        let SelectedView(mview, index) = selection;
        let index = index
            .into_iter()
            .filter(|&e_id| f(&mview.element(e_id).centroid3()))
            .collect();

        SelectedView(mview, index)
    }

    pub fn in_sphere<'a>(p0: &[f64; 3], r: f64, selection: SelectedView<'a>) -> SelectedView<'a> {
        Self::in_3d(
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_sphere(x, p0, r)
            },
            selection,
        )
    }
    pub fn in_bbox<'a>(
        p0: &[f64; 3],
        p1: &[f64; 3],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_3d(
            |x| {
                debug_assert_eq!(x.len(), 3);
                geo::in_aa_bbox(x, p0, p1)
            },
            selection,
        )
    }

    pub fn in_circle<'a>(p0: &[f64; 2], r: f64, selection: SelectedView<'a>) -> SelectedView<'a> {
        Self::in_2d(
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_circle(x, p0, r)
            },
            selection,
        )
    }

    pub fn in_rectangle<'a>(
        p0: &[f64; 2],
        p1: &[f64; 2],
        selection: SelectedView<'a>,
    ) -> SelectedView<'a> {
        Self::in_2d(
            |x| {
                debug_assert_eq!(x.len(), 2);
                geo::in_aa_rectangle(x, p0, p1)
            },
            selection,
        )
    }
}
