use crate::ElementLike;
use crate::umesh::ElementType;
use crate::{UMesh, UMeshView};
use ndarray as nd;
use ndarray::prelude::*;
use std::path::Path;
use vtkio::model::*;

fn to_vtk_cell(et: ElementType) -> CellType {
    use ElementType::*;
    match et {
        VERTEX => CellType::Vertex,
        SEG2 => CellType::Line,
        TRI3 => CellType::Triangle,
        PGON => CellType::Polygon,
        QUAD4 => CellType::Quad,
        TET4 => CellType::Tetra,
        HEX8 => CellType::Hexahedron,
        PHED => CellType::Polyhedron,
        _ => panic!("Unsupported element type for VTK: {:?}", et),
    }
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    let coords: Vec<f64> = match mesh.coords().shape()[1] {
        1 => mesh
            .coords()
            .outer_iter()
            .flat_map(|x1| {
                let mut x3 = vec![0.0; 3];
                x3[0] = x1[0];
                x3
            })
            .collect(),
        2 => mesh
            .coords()
            .outer_iter()
            .flat_map(|x2| {
                let mut x3 = vec![0.0; 3];
                x3[0] = x2[0];
                x3[1] = x2[1];
                x3
            })
            .collect(),
        // Grr this is also copy
        3 => mesh
            .coords()
            .as_slice()
            .expect("Layout should be contiguous")
            .into(),
        _ => panic!("Only 3D meshes are supported for VTK export"),
    };
    let connectivity: Vec<u64> = mesh
        .elements()
        .flat_map(|x| x.connectivity().to_vec())
        .map(|x| x as u64)
        .collect();

    let offsets: Vec<u64> = mesh
        .elements()
        .scan(0, |state, elem| {
            let count = elem.connectivity().len() as u64;
            *state += count;
            Some(*state)
        })
        .collect();

    let types: Vec<CellType> = mesh
        .elements()
        .map(|x| to_vtk_cell(x.element_type()))
        .collect();

    let vtk = Vtk {
        version: Version::new((4, 1)),
        byte_order: ByteOrder::BigEndian,
        title: String::from("Test VTK File"),
        file_path: Some(path.into()),
        data: DataSet::inline(UnstructuredGridPiece {
            points: coords.into(),
            cells: Cells {
                cell_verts: VertexNumbers::XML {
                    connectivity,
                    offsets,
                },
                types,
            },
            data: Attributes::new(),
        }),
    };
    Ok(vtk.export(path)?)
}

fn to_element_type(cell_type: CellType) -> ElementType {
    use CellType::*;
    match cell_type {
        Vertex => ElementType::VERTEX,
        Line => ElementType::SEG2,
        Triangle => ElementType::TRI3,
        Polygon => ElementType::PGON,
        Quad => ElementType::QUAD4,
        Tetra => ElementType::TET4,
        Hexahedron => ElementType::HEX8,
        Polyhedron => ElementType::PHED,
        _ => panic!("Unsupported cell type for VTK: {:?}", cell_type),
    }
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let vtk = Vtk::import(path)?;
    let data = vtk.data;
    todo!();
    // let pieces = match data {
    //     DataSet::UnstructuredGrid { pieces, .. } => pieces,
    //     _ => return Err("Only unstructured grid pieces are supported".into()),
    // };
    // let piece = &pieces[0];
    // let data = match *piece {
    //     Piece::Inline(data) => data,
    //     _ => return Err("Only inline unstructured grid pieces are supported".into()),
    // };

    // let points: Vec<f64> = data.points.into_vec().unwrap();
    // let mut mesh = UMesh::new(Array2::from_shape_vec((points.len() / 3, 3), points)?);

    // for (cell_verts, cell_type) in data.cells.cell_verts {
    //     let et = to_element_type(cell_type);
    //     mesh.add_regular_block(et, cell_verts.connectivity().into());
    // }
    //
    // Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::UMesh;
    use ndarray as nd;
    use ndarray::Array2;
    use std::path::PathBuf;

    fn make_test_2d_mesh() -> UMesh {
        let coords =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let mut mesh = UMesh::new(coords);
        mesh.add_regular_block(ElementType::QUAD4, nd::arr2(&[[0, 1, 3, 2]]));
        mesh
    }

    #[test]
    fn test_write_vtk() {
        let path = PathBuf::from("test.vtk");
        let mesh = make_test_2d_mesh();
        assert!(write(&path, mesh.view()).is_ok());
        std::fs::remove_file(path).unwrap(); // Clean up the test file
    }
}
