use crate::ElementLike;
use crate::ElementType;
use crate::{UMesh, UMeshView};
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
        _ => panic!("Unsupported element type for VTK: {et:?}"),
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
        version: Version::XML { major: 1, minor: 0 },
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
        _ => panic!("Unsupported cell type for VTK: {cell_type:?}"),
    }
}

fn extract_connectivity(connectivity: &[u64], offsets: &[u64], i: usize) -> Vec<usize> {
    let lower_bound = if i > 0 { offsets[i - 1] as usize } else { 0 };
    let higher_bound = offsets[i] as usize;
    let mut cell_connectivity = Vec::with_capacity(higher_bound - lower_bound);
    for k in lower_bound..higher_bound {
        cell_connectivity.push(connectivity[k] as usize);
    }
    cell_connectivity
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let vtk = Vtk::import(path)?;
    let pieces = if let DataSet::UnstructuredGrid { pieces, .. } = vtk.data {
        pieces
    } else {
        panic!("Wrong vtk data type");
    };

    // If piece is already inline, this just returns a piece data clone.
    let piece = pieces[0]
        .load_piece_data(None)
        .expect("Failed to load piece data");

    let points: Vec<f64> = piece.points.into_vec().unwrap();
    let mut mesh = UMesh::new(Array2::from_shape_vec((points.len() / 3, 3), points)?.into());
    let (connectivity, offsets) = piece.cells.cell_verts.into_xml();
    let cell_type = piece.cells.types;

    // TODO: for efficiency I could preallocate the connectivities vectors
    for i in 0..cell_type.len() {
        let cell_connectivity =
            extract_connectivity(connectivity.as_slice(), offsets.as_slice(), i);
        mesh.add_element(
            to_element_type(cell_type[i]),
            cell_connectivity.as_slice(),
            None,
            None,
        );
    }

    Ok(mesh)
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
        let mut mesh = UMesh::new(coords.into());
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

    #[test]
    fn test_read_vtk() {
        let path = PathBuf::from("test2.vtk");
        let mesh = make_test_2d_mesh();
        assert!(write(&path, mesh.view()).is_ok());
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap(); // Clean up the test file
        // This is not equal because of the coords dimension issue
        // assert_eq!(mesh.coords, mesh2.coords);
        for (e1, e2) in mesh.elements().zip(mesh2.elements()) {
            assert_eq!(e1.connectivity, e2.connectivity);
        }
    }
}
