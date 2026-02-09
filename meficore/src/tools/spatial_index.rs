use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use nalgebra::{Point2, Vector2};
use rayon::iter::ParallelIterator;

use crate::mesh::{ElementId, ElementIds, ElementLike, UMeshView};
use crate::prelude::ElementGeo;

pub struct ElementBbox<const D: usize> {
    aabb: Aabb<f32, D>,
    index: usize,
    eid: ElementId,
}

impl<const D: usize> Bounded<f32, D> for ElementBbox<D> {
    fn aabb(&self) -> Aabb<f32, D> {
        self.aabb.clone()
    }
}

impl<const D: usize> BHShape<f32, D> for ElementBbox<D> {
    fn set_bh_node_index(&mut self, index: usize) {
        self.index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.index
    }
}

type EBox2 = ElementBbox<2>;
type EBox3 = ElementBbox<3>;

pub struct SpatialIndex<const D: usize> {
    elems: Vec<ElementBbox<D>>,
    bvh: Bvh<f32, D>,
}

impl SpatialIndex<2> {
    fn in_bounds(&self, min: [f64; 2], max: [f64; 2]) -> ElementIds {
        let bbox = Aabb::with_bounds(
            [min[0] as f32, min[1] as f32].into(),
            [max[0] as f32, max[1] as f32].into(),
        );
        self.bvh
            .traverse(&bbox, &self.elems)
            .into_iter()
            .map(|eb| eb.eid)
            .collect()
    }
}

impl SpatialIndex<3> {
    fn in_bounds(&self, min: [f64; 3], max: [f64; 3]) -> ElementIds {
        let bbox = Aabb::with_bounds(
            [min[0] as f32, min[1] as f32, min[2] as f32].into(),
            [max[0] as f32, max[1] as f32, max[2] as f32].into(),
        );
        self.bvh
            .traverse(&bbox, &self.elems)
            .into_iter()
            .map(|eb| eb.eid)
            .collect()
    }
}

type SpIdx2 = SpatialIndex<2>;
type SpIdx3 = SpatialIndex<3>;

pub trait SpatiallyIndexable {
    fn bvh2(&self) -> SpIdx2;
    fn bvh3(&self) -> SpIdx3;
}

impl SpatiallyIndexable for UMeshView<'_> {
    fn bvh2(&self) -> SpIdx2 {
        assert_eq!(self.space_dimension(), 2);
        let mut ebox2: Vec<_> = self
            .par_elements()
            .map(|e| {
                let [min, max] = e.bounds2();
                EBox2 {
                    aabb: Aabb::with_bounds(
                        [min[0] as f32, min[1] as f32].into(),
                        [max[0] as f32, max[1] as f32].into(),
                    ),
                    index: 0,
                    eid: e.id(),
                }
            })
            .collect();
        #[cfg(feature = "rayon")]
        let bvh = Bvh::build_par(&mut ebox2);
        #[cfg(not(feature = "rayon"))]
        let bvh = Bvh::build(&mut ebox2);
        SpIdx2 { elems: ebox2, bvh }
    }

    fn bvh3(&self) -> SpIdx3 {
        assert_eq!(self.space_dimension(), 3);
        let mut ebox3: Vec<_> = self
            .par_elements()
            .map(|e| {
                let [min, max] = e.bounds3();
                EBox3 {
                    aabb: Aabb::with_bounds(
                        [min[0] as f32, min[1] as f32, min[2] as f32].into(),
                        [max[0] as f32, max[1] as f32, max[2] as f32].into(),
                    ),
                    index: 0,
                    eid: e.id(),
                }
            })
            .collect();
        #[cfg(feature = "rayon")]
        let bvh = Bvh::build_par(&mut ebox3);
        #[cfg(not(feature = "rayon"))]
        let bvh = Bvh::build(&mut ebox3);
        SpIdx3 { elems: ebox3, bvh }
    }
}
