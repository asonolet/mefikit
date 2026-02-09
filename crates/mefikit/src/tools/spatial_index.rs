use bvh::aabb::{Aabb, Bounded};
use bvh::bounding_hierarchy::{BHShape, BoundingHierarchy};
use bvh::bvh::Bvh;
use nalgebra::{Point2, Vector2};
use rayon::iter::ParallelIterator;

use crate::mesh::{ElementId, ElementLike, UMeshView};
use crate::prelude::ElementGeo;

struct SpatialIndex {}

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
type Bvh2 = Bvh<f32, 2>;
type Bvh3 = Bvh<f32, 3>;

pub trait SpatiallyIndexable {
    fn bvh2(&self) -> (Vec<EBox2>, Bvh2);
    fn bvh3(&self) -> (Vec<EBox3>, Bvh3);
}

impl SpatiallyIndexable for UMeshView<'_> {
    fn bvh2(&self) -> (Vec<EBox2>, Bvh2) {
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
        (ebox2, bvh)
    }

    fn bvh3(&self) -> (Vec<EBox3>, Bvh3) {
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
        (ebox3, bvh)
    }
}
