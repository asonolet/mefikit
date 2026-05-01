use crate::mesh::{ElementType, UMesh, UMeshView};

use ndarray::{self as nd, ArrayView1, s};

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
        coord_i.slice_mut(s![.., dim - 1]).fill(val);
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
        .slice_mut(s![.., 0])
        .mapv_inplace(|x| x - along[[0, 0]]);
    if dim > 2 {
        // Remove y offset
        along_centered
            .slice_mut(s![.., 1])
            .mapv_inplace(|y| y - along[[0, 1]]);
    }

    let mut new_coords: Vec<_> = Vec::with_capacity(along.nrows());
    for val in along_centered.rows() {
        let mut coord_i = coord0.clone();
        coord_i.slice_mut(s![.., dim - 1]).fill(val[dim - 1]);
        coord_i.slice_mut(s![.., 0]).mapv_inplace(|x| x + val[0]);
        if dim > 2 {
            coord_i.slice_mut(s![.., 1]).mapv_inplace(|x| x + val[1]);
        }
        new_coords.push(coord_i);
    }
    let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
    nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
}

/// This allows curvilinear extrusion along a new axis with non parallel planes.
/// This method is only valid in 3d.
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
    let z0 = along[[0, dim - 1]];
    let new_axis = nd::Array::from_elem((coords.nrows(), 1), z0);
    let coords = nd::concatenate(nd::Axis(1), &[coords, new_axis.view()]).unwrap();

    // TODO: implement the method for 2d extrusion
    let zero = nd::arr2(&[[0., 0., 0.]]);
    let offsets_vecs = nd::concatenate![
        nd::Axis(0),
        zero,
        &along.slice(s![1.., ..]) - &along.slice(s![..-1, ..]),
    ]; // z starts at 0.0

    // Compute normals
    // 1. Compute vectors
    // For all except first and last, the normal vector is computed from previous and next offsets
    // positions (centered)
    let normal_vecs = &along.slice(s![2.., ..]) - &along.slice(s![..-2, ..]);
    let oz = nd::arr2(&[[0., 0., 1.]]);
    let first_vec = &along.slice(s![1..2, ..]) - &along.slice(s![..1, ..]);
    let last_vec = &along.slice(s![-1.., ..]) - &along.slice(s![-2..-1, ..]);

    let normal_vecs = nd::concatenate![nd::Axis(0), oz, first_vec, normal_vecs, last_vec];

    let mut new_coords: Vec<_> = Vec::with_capacity(along.nrows());
    let mut last_coords = coords.clone();

    for (offset, z_win) in offsets_vecs
        .rows()
        .into_iter()
        .zip(normal_vecs.axis_windows(nd::Axis(0), 2))
    {
        let z_prev = z_win.row(0).to_owned();
        let z_new = z_win.row(1).to_owned();
        let center = last_coords.mean_axis(nd::Axis(0)).unwrap();
        let rotated_coords = rotate_mesh(last_coords.view(), center.view(), z_prev, z_new) + offset;
        last_coords = rotated_coords.clone();
        new_coords.push(rotated_coords);
    }
    let new_coords_views: Vec<_> = new_coords.iter().map(|c| c.view()).collect();
    nd::concatenate(nd::Axis(0), &new_coords_views).unwrap()
}

use ndarray::Array1;

fn orthogonal(a: nd::ArrayView1<f64>) -> Array1<f64> {
    if a[0].abs() < a[1].abs() && a[0].abs() < a[2].abs() {
        let ox = Array1::from(vec![1.0, 0.0, 0.0]);
        cross(a, ox.view())
    } else if a[1].abs() < a[2].abs() {
        let oy = Array1::from(vec![0.0, 1.0, 0.0]);
        cross(a, oy.view())
    } else {
        let oz = Array1::from(vec![0.0, 0.0, 1.0]);
        cross(a, oz.view())
    }
}

fn normalize(mut v: nd::ArrayViewMut1<f64>) {
    let n = v.dot(&v).sqrt();
    v /= n;
}

fn rotate_mesh(
    coords: nd::ArrayView2<f64>,
    center: nd::ArrayView1<f64>,
    z_prev: nd::Array1<f64>,
    z_new: nd::Array1<f64>,
) -> nd::Array2<f64> {
    let mut result = coords.to_owned();

    // a = (0,0,1)
    let mut a = z_prev;
    normalize(a.view_mut());

    // normaliser b
    let mut b = z_new;
    normalize(b.view_mut());

    let c = a.dot(&b);

    // produit vectoriel
    let v = cross(a.view(), b.view());

    if c > 0.999999 {
        // identité
        return result;
    }

    if c < -0.999999 {
        // rotation 180°
        let u = orthogonal(a.view());

        for mut row in result.axis_iter_mut(nd::Axis(0)) {
            let mut x = &row - &center;
            let dot = x.dot(&u);
            x = &x - &(2.0 * dot * &u);
            row.assign(&(x + center));
        }
        return result;
    }

    let denom = 1.0 + c;

    for mut row in result.axis_iter_mut(nd::Axis(0)) {
        let mut x = &row - &center;

        let vx = cross(v.view(), x.view());
        let vvx = cross(v.view(), vx.view());

        x = &x + &vx + &(vvx / denom);

        row.assign(&(x + center));
    }

    result
}

fn cross(a: ArrayView1<f64>, b: ArrayView1<f64>) -> Array1<f64> {
    Array1::from(vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ])
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
    extrude_connectivity(mesh, along.nrows() - 1, new_coords)
}

pub fn extrude_curv(mesh: UMeshView, along: nd::ArrayView2<'_, f64>) -> UMesh {
    if along.is_empty() {
        return mesh.to_shared();
    }
    let new_coords = extrude_coords_curvilinear(mesh.coords(), along);
    if along.nrows() == 1 {
        let mut extruded_mesh = mesh.to_shared();
        extruded_mesh.coords = new_coords.into_shared();
        return extruded_mesh;
    }
    extrude_connectivity(mesh, along.nrows() - 1, new_coords)
}

pub trait Extrudable {
    fn extrude(&self, along: &[f64]) -> UMesh;
    fn extrude_curv(&self, along: nd::ArrayView2<'_, f64>) -> UMesh;
    fn extrude_parallel(&self, along: nd::ArrayView2<'_, f64>) -> UMesh;
    // fn extrude_grow_normal_dir(&self, along: &[f64]) -> UMesh;
    // fn extrude_grow_with_focal(&self, along: &[f64], focal: f64, normal: &[f64]);
}

impl Extrudable for UMeshView<'_> {
    fn extrude(&self, along: &[f64]) -> UMesh {
        extrude(self.clone(), along)
    }

    fn extrude_parallel(&self, along: ndarray::ArrayView2<'_, f64>) -> UMesh {
        extrude_parallel(self.clone(), along)
    }

    fn extrude_curv(&self, along: ndarray::ArrayView2<'_, f64>) -> UMesh {
        extrude_curv(self.clone(), along)
    }
}

impl Extrudable for UMesh {
    fn extrude(&self, along: &[f64]) -> UMesh {
        extrude(self.view(), along)
    }

    fn extrude_parallel(&self, along: ndarray::ArrayView2<'_, f64>) -> UMesh {
        extrude_parallel(self.view(), along)
    }

    fn extrude_curv(&self, along: ndarray::ArrayView2<'_, f64>) -> UMesh {
        extrude_curv(self.view(), along)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ElementType, UMesh};
    use ndarray as nd;

    #[test]
    fn test_extrude_1d_to_2d() {
        let coords = nd::ArcArray2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1]]).to_shared(),
            None,
        );
        let extruded = mesh.extrude(&[0.0, 1.0]);
        assert_eq!(extruded.space_dimension(), 2);
        assert!(extruded.num_elements() > 0);
    }

    #[test]
    fn test_extrude_2d_to_3d() {
        let coords = nd::ArcArray2::from_shape_vec(
            (4, 2),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::QUAD4,
            nd::arr2(&[[0, 1, 3, 2]]).to_shared(),
            None,
        );
        let extruded = mesh.extrude(&[0.0, 1.0]);
        assert_eq!(extruded.space_dimension(), 3);
        assert!(extruded.num_elements() > 0);
    }

    #[test]
    fn test_extrude_empty_along() {
        let coords = nd::ArcArray2::from_shape_vec((2, 1), vec![0.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(
            ElementType::SEG2,
            nd::arr2(&[[0, 1]]).to_shared(),
            None,
        );
        let extruded = mesh.extrude(&[]);
        assert_eq!(extruded.num_elements(), mesh.num_elements());
    }

    #[test]
    fn test_extrude_coords_2d() {
        let coords = nd::arr2(&[[0.0], [1.0]]);
        let along = [0.0, 1.0];
        let new_coords = extrude_coords(coords.view(), &along);
        // Should have 2 (original) * 2 (along length) = 4 rows
        assert_eq!(new_coords.nrows(), 4);
        assert_eq!(new_coords.ncols(), 2); // Original 1D + 1 new dimension
    }

    #[test]
    fn test_extrude_coords_3d() {
        let coords = nd::arr2(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
        let along = [0.0, 1.0];
        let new_coords = extrude_coords(coords.view(), &along);
        // Should have 4 (original) * 2 (along length) = 8 rows
        assert_eq!(new_coords.nrows(), 8);
        assert_eq!(new_coords.ncols(), 3); // Original 2D + 1 new dimension
    }
}
