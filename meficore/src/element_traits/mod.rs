mod element_geo;
mod element_topo;
pub mod is_in;
pub mod measures;
pub mod seg_intersect;
mod seg_intersect2;
mod symmetry;
mod utils;

pub use element_geo::ElementGeo;
pub use element_topo::ElementTopo;
pub use seg_intersect2::{Intersection, Intersections, PointId, intersect_seg_seg};
pub use utils::SortedVecKey;
