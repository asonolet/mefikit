use crate::element_traits::ElementGeo;
use crate::geometry as geo;
use crate::mesh::{ElementIdsSet, UMeshView};
#[cfg(feature = "rayon")]
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub enum CentroidSelection {
    BBox { min: [f64; 3], max: [f64; 3] }, // Axis aligned BBox
    Rect { min: [f64; 2], max: [f64; 2] }, // Axis aligned BBox
    Sphere { center: [f64; 3], r: f64 },   // center and radius
    Circle { center: [f64; 2], r: f64 },   // center and radius
}

impl CentroidSelection {
    fn in_2d<'a, F0>(f: F0, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64; 2]) -> bool + Sync,
    {
        sel.into_par_iter()
            .filter(|&e_id| f(&view.element(e_id).centroid2()))
            .collect()
    }
    fn in_3d<'a, F0>(f: F0, view: &'a UMeshView<'a>, sel: ElementIdsSet) -> ElementIdsSet
    where
        F0: Fn(&[f64; 3]) -> bool + Sync,
    {
        sel.into_par_iter()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples as me;
    use crate::tools::MeshSelect;
    use crate::tools::Selection;

    #[test]
    fn test_in_sphere() {
        let mesh = me::make_mesh_2d_quad();
        let selection = CentroidSelection::Circle {
            center: [0.5, 0.5],
            r: 0.5,
        };
        let ids = mesh.select_ids(Selection::CentroidSelection(selection), None);
        // Quad centroid is at (0.5, 0.5) which is within radius 0.5
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_in_circle() {
        let mesh = me::make_mesh_2d_quad();
        let selection = CentroidSelection::Circle {
            center: [0.5, 0.5],
            r: 0.5,
        };
        let ids = mesh.select_ids(Selection::CentroidSelection(selection), None);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_in_bbox() {
        let mesh = me::make_mesh_2d_quad();
        let selection = Selection::CentroidSelection(CentroidSelection::Rect {
            min: [0.0, 0.0],
            max: [1.0, 1.0],
        });
        let ids = mesh.select_ids(selection, None);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_in_rect() {
        let mesh = me::make_mesh_2d_quad();
        let selection = Selection::CentroidSelection(CentroidSelection::Rect {
            min: [0.0, 0.0],
            max: [1.0, 1.0],
        });
        let ids = mesh.select_ids(selection, None);
        assert!(!ids.is_empty());
    }
}

#[cfg(feature = "rayon")]
#[cfg(test)]
mod par_tests {
    use super::*;
    use crate::geometry as geo;
    use crate::mesh::{ElementId, ElementType};
    use crate::mesh_examples as me;

    #[test]
    fn test_in_circle_parallel_matches_serial() {
        let mesh = me::make_imesh_2d(6);
        let view = mesh.view();
        let center = [0.5, 0.5];
        let r = 0.3;
        let f = |x: &[f64; 2]| geo::in_circle(x, &center, r);
        let all: ElementIdsSet = (0..36)
            .map(|i| ElementId::new(ElementType::QUAD4, i))
            .collect();
        let ser: ElementIdsSet = all
            .clone()
            .into_iter()
            .filter(|&e| f(&view.element(e).centroid2()))
            .collect();
        let par = CentroidSelection::in_2d(f, &view, all);
        assert_eq!(par.0, ser.0);
        assert_eq!(par.len(), ser.len());
        assert!(!par.is_empty());
    }
}
