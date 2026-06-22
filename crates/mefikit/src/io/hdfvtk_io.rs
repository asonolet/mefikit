use crate::mesh::{ElementLike, ElementType, UMesh, UMeshView};
use hdf5_metno::{
    File,
    types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode},
};
use ndarray::{Array1, Array2, arr1, s};
use std::path::Path;

fn el_to_usize(code: usize) -> Result<ElementType, Box<dyn std::error::Error>> {
     match code {
         1 => Ok(ElementType::VERTEX),
         3 => Ok(ElementType::SEG2),
         5 => Ok(ElementType::TRI3),
         7 => Ok(ElementType::PGON),
         9 => Ok(ElementType::QUAD4),
         10 => Ok(ElementType::TET4),
         12 => Ok(ElementType::HEX8),
         42 => Ok(ElementType::PHED),
         other => Err(format!("Unsupported HdfVtkElementType code {other}").into()),
     }
}

fn usize_to_el(el_type: ElementType) -> Result<usize, Box<dyn std::error::Error>> {
    match el_type {
        ElementType::VERTEX => Ok(1),
        ElementType::SEG2 => Ok(3),
        ElementType::TRI3 => Ok(5),
        ElementType::PGON => Ok(7),
        ElementType::QUAD4 => Ok(9),
        ElementType::TET4 => Ok(10),
        ElementType::HEX8 => Ok(12),
        ElementType::PHED => Ok(42),
        other => Err(format!("Unsupported ElementType {other:?}").into()),
    }
}

fn handle_unstructured(block: &hdf5_metno::Group) -> Result<UMesh, Box<dyn std::error::Error>> {
    // read data from file
    let points: Array2<f64> = block.dataset("Points")?.read()?;
    let offsets: Array1<usize> = block.dataset("Offsets")?.read()?;
    let conn: Array1<i64> = block.dataset("Connectivity")?.read()?;
    let types: Array1<usize> = block.dataset("Types")?.read()?;

    // transform data into mesh
    let mut mesh = UMesh::new(points.into());
    for i in 0..types.len() {
        let start = offsets[i];
        let end = offsets[i + 1];
        let el_type = el_to_usize(types[i])?;
        let cell_conn: Vec<usize> = conn
            .slice(s![start..end])
            .iter()
            .map(|&x| x as usize)
            .collect();
        mesh.add_element(el_type, &cell_conn, None, None);
    }
    Ok(mesh)
}

fn read_type_attr(group: &hdf5_metno::Group) -> Result<String, Box<dyn std::error::Error>> {
    let attr = group.attr("Type")?;
    let dtype = attr.dtype()?;
    let desc = dtype.to_descriptor()?;

    match desc {
        TypeDescriptor::VarLenUnicode => {
            let s: VarLenUnicode = attr.read_scalar()?;
            Ok(s.to_string())
        }
        TypeDescriptor::VarLenAscii => {
            let s: VarLenAscii = attr.read_scalar()?;
            Ok(s.to_string())
        }
        TypeDescriptor::FixedAscii(_) => {
            let s: FixedAscii<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        TypeDescriptor::FixedUnicode(_) => {
            let s: FixedUnicode<64> = attr.read_scalar()?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        other => Err(format!("Unexpected string type: {other:?}").into()),
    }
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let vtk = file.group("VTKHDF").map_err(|_| "Not a VTKHDF file")?;

    match read_type_attr(&vtk)?.as_str() {
        "UnstructuredGrid" => return handle_unstructured(&vtk),
        "PartitionedDataSetCollection" | "MultiBlockDataSet" => {
            for name in vtk.member_names()? {
                let block = vtk.group(name.as_str())?;
                dbg!(&block);
                let Ok(_) = block.attr("Type") else { continue };
                match read_type_attr(&block)?.as_str() {
                    "UnstructuredGrid" => return handle_unstructured(&block),
                    _ => continue,
                }
            }
        }
        _ => {}
    }
    Err(format!("No VTKHDF group found in {}", path.display()).into())
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    // create file
    let file = File::create(path)?;
    // create VTKHDF group
    let vtk = file.create_group("VTKHDF")?;

    // add type UnstructuredGrid attr
    vtk.new_attr::<FixedAscii<16>>()
        .shape(())
        .create("Type")?
        .write_scalar(&FixedAscii::<16>::from_ascii("UnstructuredGrid").unwrap())?;

    // add version
    vtk.new_attr::<i64>()
        .shape([2])
        .create("Version")?
        .write(&arr1(&[2i64, 0]))?;

    // collect from mesh view
    let coords: Array2<f64> = mesh.coords().to_owned();

    // destructure elements of UMesh
    let mut types: Vec<u8> = Vec::new();
    let mut offsets: Vec<usize> = vec![0];
    let mut connectivity: Vec<usize> = Vec::new();

    for el in mesh.elements() {
        let conn = el.connectivity();
        types.push(usize_to_el(el.element_type())? as u8);
        connectivity.extend_from_slice(conn);
        offsets.push(connectivity.len());
    }

    // write datasets
    // coords
    vtk.new_dataset::<f64>()
        .shape(coords.shape())
        .create("Points")?
        .write(&coords)?;
    // types
    vtk.new_dataset::<u8>()
        .shape([types.len()])
        .create("Types")?
        .write(&Array1::from(types))?;
    // offsets
    vtk.new_dataset::<usize>()
        .shape([offsets.len()])
        .create("Offsets")?
        .write(&Array1::from(offsets))?;
    // connectivity
    vtk.new_dataset::<usize>()
        .shape([connectivity.len()])
        .create("Connectivity")?
        .write(&Array1::from(connectivity))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples as me;
    use std::path::PathBuf;

    // #[test]
    // fn test_read_hdfvtk() {
    //     let path = PathBuf::from(concat!(
    //         env!("CARGO_MANIFEST_DIR"),
    //         "/../../tests/Box1.vtkhdf"
    //     ));
    //     let mesh = read(&path).unwrap();
    //     assert_eq!(mesh.coords().nrows(), 13);
    //     assert_eq!(mesh.num_elements(), 54);
    // }

    #[test]
    fn test_write_hdfvtk() {
        let path = PathBuf::from("test_write.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        assert!(write(&path, mesh.view()).is_ok());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_roundtrip_hdfvtk() {
        let path = PathBuf::from("test_roundtrip.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        assert!(write(&path, mesh.view()).is_ok());
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for (e1, e2) in mesh.elements().zip(mesh2.elements()) {
            assert_eq!(e1.connectivity, e2.connectivity);
        }
    }
}
