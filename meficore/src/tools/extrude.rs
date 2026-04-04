use crate::mesh::{ElementType, UMesh, UMeshView};

use ndarray::{self as nd, IntoNdProducer};

/// This is the most simple extrusion method.
///
/// A new axis is added with coords from along.
fn extrude_coords(coords: nd::ArrayView2<'_, f64>, along: &[f64]) -> nd::Array2<f64> {
    let new_axis = nd::Array::zeros((coords.nrows(), 1));
    let coord0 = nd::concatenate(nd::Axis(1), &[coords, new_axis.view()]).unwrap();
    let dim = coord0.ncols();

    let mut new_coords: Vec<_> = Vec::with_capacity(along.len());
    for &val in along {
        let mut coord_i = coord0.clone();
        coord_i.slice_mut(nd::s![.., dim - 1]).fill(val);
        new_coords.push(coord_i);
    }
    let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
    nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
}

/// This allows extrusion along a new axis with parallel but staggered planes.
///
/// A new axis is added with coords from along. The x (and potentially y) coords take an offset
/// from the along array. The offset starts at 0., so if along does not starts at [0., 0., z0], but
/// at [x0, y0, z0], x0 is removed from all Axis(0) and y0 is removed from all Axis(1).
fn extrude_coords_parallel(
    coords: nd::ArrayView2<'_, f64>,
    along: nd::ArrayView2<'_, f64>,
) -> nd::Array2<f64> {
    let new_axis = nd::Array::zeros((coords.nrows(), 1));
    let coord0 = nd::concatenate(nd::Axis(1), &[coords, new_axis.view()]).unwrap();
    let dim = coord0.ncols();

    let mut along_centered = along.to_owned();
    // Remove x offset
    along_centered
        .slice_mut(nd::s![.., 0])
        .mapv_inplace(|x| x - along[[0, 0]]);
    if dim > 2 {
        // Remove y offset
        along_centered
            .slice_mut(nd::s![.., 1])
            .mapv_inplace(|y| y - along[[0, 1]]);
    }

    let mut new_coords: Vec<_> = Vec::with_capacity(along.nrows());
    for val in along_centered.rows() {
        let mut coord_i = coord0.clone();
        coord_i.slice_mut(nd::s![.., dim - 1]).fill(val[dim - 1]);
        coord_i
            .slice_mut(nd::s![.., 0])
            .mapv_inplace(|x| x + val[0]);
        if dim > 2 {
            coord_i
                .slice_mut(nd::s![.., 1])
                .mapv_inplace(|x| x + val[1]);
        }
        new_coords.push(coord_i);
    }
    let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
    nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
}

/// This allows curvilinear extrusion along a new axis with non parallel planes.
///
/// A new axis is added with coords from along. The x (and potentially y) coords take an offset and
/// a rotation from the along array. The offset starts at 0., so if along does not starts at [0.,
/// 0., z0], but at [x0, y0, z0], x0 is removed from all Axis(0) and y0 is removed from all
/// Axis(1). The rotation is computed with a 3 points fit. The last plane is normal to the last
/// vector (formed with two last coords from the along array).
///
/// Warning: This method can lead to self intersecting meshes.
fn extrude_coords_curvilinear(
    coords: nd::ArrayView2<'_, f64>,
    along: nd::ArrayView2<'_, f64>,
) -> nd::Array2<f64> {
    // Center is the barycenter of the nodes used as rotation center, this is arbitrary
    let dim = along.ncols();
    assert_eq!(dim - 1, coords.ncols());
    let z0 = along[[0, dim]];
    let new_axis = nd::Array::from_elem((coords.nrows(), 1), z0);
    let coords = nd::concatenate(nd::Axis(1), &[coords, new_axis.view()]).unwrap();
    let center = coords.mean_axis(nd::Axis(0)).unwrap();

    let origin = along.slice(nd::s![0, ..]).to_owned();
    let offsets_vecs = along.to_owned() - &origin; // z starts at 0.0
    // first offsets_vecs row should be full zeros

    // Compute normals
    // 1. Compute vectors
    let dir_vec =
        offsets_vecs.slice(nd::s![1.., ..]).to_owned() - offsets_vecs.slice(nd::s![..-1, ..]);
    let normal_vecs = dir_vec.slice(nd::s![1.., ..]).to_owned() - dir_vec.slice(nd::s![..-1, ..]);
    let first_vec = dir_vec.slice(nd::s![0..1, ..]).to_owned();
    let last_vec = dir_vec.slice(nd::s![-1.., ..]).to_owned();

    let normal_vecs = nd::concatenate![nd::Axis(0), first_vec, normal_vecs, last_vec];

    let mut new_coords: Vec<_> = Vec::with_capacity(along.nrows());
    for (offset, normal) in offsets_vecs.rows().into_iter().zip(normal_vecs.rows()) {
        let mut coord_i = coords.clone();
        todo!("Translate and rotate coord_i with center, offset and normal.");
        new_coords.push(coord_i);
    }
    let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
    nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
}

// fn extrude_coords_along_normal(coords: nd::ArrayView2<'_, f64>, along: &[f64]) -> nd::Array2<f64> {
//     let new_axis = nd::Array::zeros((coords.nrows(), 1));
//     let coord0 = nd::concatenate(nd::Axis(1), &[coords, new_axis.view()]).unwrap();
//     let dim = coord0.ncols();
//
//     let mut along_centered = along.to_owned();
//     // Remove x offset
//     along_centered
//         .slice_mut(nd::s![.., 0])
//         .mapv_inplace(|x| x - along[[0, 0]]);
//     if dim > 2 {
//         // Remove y offset
//         along_centered
//             .slice_mut(nd::s![.., 1])
//             .mapv_inplace(|y| y - along[[0, 1]]);
//     }
//
//     let mut new_coords: Vec<_> = Vec::with_capacity(along.shape()[0]);
//     for val in along.rows() {
//         let mut coord_i = coord0.clone();
//         coord_i.slice_mut(nd::s![.., dim - 1]).fill(val[dim - 1]);
//         coord_i
//             .slice_mut(nd::s![.., 0])
//             .mapv_inplace(|x| x + val[0]);
//         if dim > 2 {
//             coord_i
//                 .slice_mut(nd::s![.., 1])
//                 .mapv_inplace(|x| x + val[1]);
//         }
//         new_coords.push(coord_i);
//     }
//     let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
//     nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
// }

fn extrude_dup_connectivity(mesh: UMeshView, et: ElementType, n: usize) -> nd::Array2<usize> {
    let old_connectivity = mesh.regular_connectivity(et).unwrap();
    let n_nodes = mesh.coords().nrows();
    let old_elem_size = old_connectivity.ncols();
    let old_nb_elem = old_connectivity.nrows();
    let mut new_connectivity: nd::Array2<usize> =
        nd::Array::zeros((n * old_nb_elem, 2 * old_elem_size));
    for (i, elem) in old_connectivity.rows().into_iter().enumerate() {
        for k in 0..n {
            let new_elem_id: usize = i + old_nb_elem * k;
            let conn_inf = &elem + k * n_nodes;
            let conn_sup = &elem + (k + 1) * n_nodes;

            new_connectivity
                .slice_mut(nd::s![new_elem_id, ..old_elem_size])
                .assign(&conn_inf);
            new_connectivity
                .slice_mut(nd::s![new_elem_id, old_elem_size..])
                .assign(&conn_sup);
        }
    }
    new_connectivity
}

fn extrude_inv_connectivity(mesh: UMeshView, et: ElementType, n: usize) -> nd::Array2<usize> {
    let old_connectivity = mesh.regular_connectivity(et).unwrap();
    let n_nodes = mesh.coords().nrows();
    let old_elem_size = old_connectivity.ncols();
    let old_nb_elem = old_connectivity.nrows();
    let mut new_connectivity: nd::Array2<usize> =
        nd::Array::zeros((n * old_nb_elem, 2 * old_elem_size));
    for (i, elem) in old_connectivity.rows().into_iter().enumerate() {
        for k in 0..n {
            let new_elem_id: usize = i + old_nb_elem * k;
            let conn_inf = &elem + k * n_nodes;
            let conn_sup = &elem + (k + 1) * n_nodes;

            new_connectivity
                .slice_mut(nd::s![new_elem_id, ..old_elem_size])
                .assign(&conn_inf);
            new_connectivity
                .slice_mut(nd::s![new_elem_id, old_elem_size..;-1])
                .assign(&conn_sup);
        }
    }
    new_connectivity
}

fn extrude_connectivity(
    mesh: UMeshView,
    n: usize,
    new_coords: ndarray::ArrayBase<ndarray::OwnedRepr<f64>, ndarray::Dim<[usize; 2]>, f64>,
) -> UMesh {
    use ElementType::*;
    let mut extruded_mesh = UMesh::new(new_coords.into_shared());
    let etypes = mesh.blocks().map(|(et, _)| et);
    for &et in etypes {
        match et {
            VERTEX => extruded_mesh.add_regular_block(
                SEG2,
                extrude_dup_connectivity(mesh.view(), et, n).into_shared(),
                None,
            ),
            SEG2 => extruded_mesh.add_regular_block(
                QUAD4,
                extrude_inv_connectivity(mesh.view(), et, n).into_shared(),
                None,
            ),
            QUAD4 => extruded_mesh.add_regular_block(
                HEX8,
                extrude_dup_connectivity(mesh.view(), et, n).into_shared(),
                None,
            ),
            _ => todo!("Extrusion of {et:?} is not implemented yet"),
        };
    }
    extruded_mesh
}

pub fn extrude(mesh: UMeshView, along: &[f64]) -> UMesh {
    if along.is_empty() {
        return mesh.to_shared();
    }
    let new_coords = extrude_coords(mesh.coords(), along);
    if along.len() == 1 {
        let mut extruded_mesh = mesh.to_shared();
        extruded_mesh.coords = new_coords.into_shared();
        return extruded_mesh;
    }
    extrude_connectivity(mesh, along.len() - 1, new_coords)
}

pub fn extrude_parallel(mesh: UMeshView, along: nd::ArrayView2<'_, f64>) -> UMesh {
    if along.is_empty() {
        return mesh.to_shared();
    }
    let new_coords = extrude_coords_parallel(mesh.coords(), along);
    if along.nrows() == 1 {
        let mut extruded_mesh = mesh.to_shared();
        extruded_mesh.coords = new_coords.into_shared();
        return extruded_mesh;
    }
    extrude_connectivity(mesh, along.len() - 1, new_coords)
}

pub trait Extrudable {
    fn extrude(&self, along: &[f64]) -> UMesh;
    fn extrude_curv(&self, along: nd::ArrayView2<'_, f64>) -> UMesh;
    fn extrude_parallel(&self, along: nd::ArrayView2<'_, f64>) -> UMesh;
    fn extrude_grow_normal_dir(&self, along: &[f64]) -> UMesh;
    fn extrude_grow_with_focal(&self, along: &[f64], focal: f64, normal: &[f64]);
}
