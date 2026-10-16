//! Geometric measurements for mesh elements.
//!
//! Computes element measures (length, area, volume) and stores them as fields.

use crate::element_traits::ElementGeo;
use crate::mesh::ElementLike;
use crate::mesh::ElementType;
use crate::mesh::FieldOwned;
use crate::mesh::UMesh;
use crate::mesh::{Dimension, UMeshView};

use ndarray as nd;
#[cfg(feature = "rayon")]
use rayon::prelude::*;
use std::collections::BTreeMap;

/// Computes the geometric center of each element in the mesh.
///
/// Returns a map of element types to arrays of measure values.
pub fn centroids(
    mesh: &UMeshView,
    dim: Option<Dimension>,
) -> BTreeMap<ElementType, nd::Array2<f64>> {
    let dim = dim.unwrap_or_else(|| mesh.topological_dimension().unwrap());
    mesh
        .par_blocks()
        .filter(|(et, _)| et.dimension() == dim)
        .map(|(&k, v)| {
            (
                k,
                match mesh.space_dimension() {
                    1 => {
                        let a: Vec<f64> = v.par_iter(mesh.coords.view())
                            .map(|e| vertex_mean_component(&e, 0))
                            .collect();
                        nd::Array2::from_shape_vec(
                            (v.len(), 1),
                            a
                        ).unwrap()
                    }
                    2 => {
                        let a: Vec<[f64; 2]> = v.par_iter(mesh.coords.view())
                            .map(|e| e.centroid2())
                            .collect();
                        nd::Array2::from_shape_vec(
                        (v.len(), 2),
                        a.into_iter().flat_map(|x| x.into_iter()).collect()
                    ).unwrap()},
                    3 => {
                        let a: Vec<[f64; 3]> = v.par_iter(mesh.coords.view())
                            .map(|e| e.centroid3())
                            .collect();
                        nd::Array2::from_shape_vec(
                        (v.len(), 3),
                        a.into_iter().flat_map(|x| x.into_iter()).collect()
                    ).unwrap()},
                    c => panic!( "{c} is not a valid space dimension. Space (coordinates) dimension must be 0, 1, 2 ou 3.")
                }
            )
        })
    .collect()
}

/// The `axis` component of each element's vertex centroid in a 1D space (there is no
/// `centroid1`, so the mean is taken directly over the vertex block).
fn vertex_mean_component<'a>(elem: &impl ElementLike<'a>, axis: usize) -> f64 {
    let n = elem.num_vertices();
    elem.coords().take(n).map(|c| c[axis]).sum::<f64>() / n as f64
}

/// The `axis` component of each element's vertex centroid: [`ElementGeo::centroid2`] or
/// [`ElementGeo::centroid3`] in 2D/3D space, matching [`centroids`].
fn center_component(
    mesh: &UMeshView,
    dim: Option<Dimension>,
    axis: usize,
) -> BTreeMap<ElementType, nd::Array1<f64>> {
    let dim = dim.unwrap_or_else(|| mesh.topological_dimension().unwrap());
    mesh.par_blocks()
        .filter(|(et, _)| et.dimension() == dim)
        .map(|(&k, v)| {
            let a: Vec<f64> = v
                .par_iter(mesh.coords.view())
                .map(|e| match mesh.space_dimension() {
                    2 => e.centroid2()[axis],
                    3 => e.centroid3()[axis],
                    1 => vertex_mean_component(&e, axis),
                    c => panic!(
                        "{c} is not a valid space dimension. Space (coordinates) dimension must \
                         be 0, 1, 2 ou 3."
                    ),
                })
                .collect();
            (k, nd::Array1::from_shape_vec((v.len(),), a).unwrap())
        })
        .collect()
}

/// Computes the x coord of each element center in the mesh.
///
/// Returns a map of element types to arrays of x components of the vertex centroid.
pub fn x_center(
    mesh: &UMeshView,
    dim: Option<Dimension>,
) -> BTreeMap<ElementType, nd::Array1<f64>> {
    center_component(mesh, dim, 0)
}

/// Computes the y coord of each element center in the mesh.
///
/// Returns a map of element types to arrays of y components of the vertex centroid.
pub fn y_center(
    mesh: &UMeshView,
    dim: Option<Dimension>,
) -> BTreeMap<ElementType, nd::Array1<f64>> {
    center_component(mesh, dim, 1)
}

/// Computes the z coord of each element center in the mesh.
///
/// Returns a map of element types to arrays of z components of the vertex centroid.
pub fn z_center(
    mesh: &UMeshView,
    dim: Option<Dimension>,
) -> BTreeMap<ElementType, nd::Array1<f64>> {
    center_component(mesh, dim, 2)
}

/// Trait for computing and storing element measures as fields.
pub trait Centroidable {
    /// Computes element measures and returns them as a field.
    fn centroid(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix2>;
    fn x(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1>;
    fn y(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1>;
    fn z(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1>;
}

pub trait CentroidableMut {
    /// Computes element measures and stores them as a named field in the mesh.
    fn centroid_update(&mut self, dim: Option<Dimension>);
}

impl Centroidable for UMesh {
    fn centroid(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix2> {
        FieldOwned::new(centroids(&self.view(), dim))
    }

    fn x(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1> {
        FieldOwned::new(x_center(&self.view(), dim))
    }

    fn y(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1> {
        FieldOwned::new(y_center(&self.view(), dim))
    }

    fn z(&self, dim: Option<Dimension>) -> FieldOwned<nd::Ix1> {
        FieldOwned::new(z_center(&self.view(), dim))
    }
}

impl CentroidableMut for UMesh {
    fn centroid_update(&mut self, dim: Option<Dimension>) {
        let f = FieldOwned::new(centroids(&self.view(), dim));
        self.update_field("Centroids", f.into_shared().into_dyn());
    }
}

impl Centroidable for UMeshView<'_> {
    fn centroid(&self, dim: Option<Dimension>) -> FieldOwned<ndarray::Ix2> {
        FieldOwned::new(centroids(self, dim))
    }

    fn x(&self, dim: Option<Dimension>) -> FieldOwned<ndarray::Ix1> {
        FieldOwned::new(x_center(self, dim))
    }

    fn y(&self, dim: Option<Dimension>) -> FieldOwned<ndarray::Ix1> {
        FieldOwned::new(y_center(self, dim))
    }

    fn z(&self, dim: Option<Dimension>) -> FieldOwned<ndarray::Ix1> {
        FieldOwned::new(z_center(self, dim))
    }
}
