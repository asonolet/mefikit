use crate::mesh::{ElementType, UMesh, UMeshView};

use ndarray as nd;

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

pub fn extrude(mesh: UMeshView, along: &[f64]) -> UMesh {
    use ElementType::*;
    if along.is_empty() {
        return mesh.to_shared();
    }
    let new_coords = extrude_coords(mesh.coords(), along);
    if along.len() == 1 {
        let mut extruded_mesh = mesh.to_shared();
        extruded_mesh.coords = new_coords.into_shared();
        return extruded_mesh;
    }
    let mut extruded_mesh = UMesh::new(new_coords.into_shared());
    let etypes = mesh.blocks().map(|(et, _)| et);
    for &et in etypes {
        match et {
            VERTEX => extruded_mesh.add_regular_block(
                SEG2,
                extrude_dup_connectivity(mesh.view(), et, along.len() - 1).into_shared(),
            ),
            SEG2 => extruded_mesh.add_regular_block(
                QUAD4,
                extrude_inv_connectivity(mesh.view(), et, along.len() - 1).into_shared(),
            ),
            QUAD4 => extruded_mesh.add_regular_block(
                HEX8,
                extrude_dup_connectivity(mesh.view(), et, along.len() - 1).into_shared(),
            ),
            _ => todo!("Extrusion of {et:?} is not implemented yet"),
        };
    }
    extruded_mesh
}
