use super::elements_mapping::ElementsMapping;
use super::error::MefikitIOError;
use super::hdf_utils::read_group_attr;
use crate::mesh::{ElementLike, ElementType, UMesh, UMeshView};
use hdf5_metno::{
    File,
    Group,
    types::FixedAscii,
};
use ndarray::{Array1, Array2, arr1, s};
use std::path::Path;

// VTKHDF reuses the standard VTK cell type codes.
const VTKHDF_MAPPING: ElementsMapping = ElementsMapping::new(
    "VTKHDF",
    &[
        (1, ElementType::VERTEX),
        (3, ElementType::SEG2),
        (5, ElementType::TRI3),
        (7, ElementType::PGON),
        (9, ElementType::QUAD4),
        (10, ElementType::TET4),
        (12, ElementType::HEX8),
        (42, ElementType::PHED),
    ],
);

fn handle_unstructured(block: &Group) -> Result<UMesh, MefikitIOError> {
    let points: Array2<f64> = block
        .dataset("Points")
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?
        .read()
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?;
    let offsets: Array1<usize> = block
        .dataset("Offsets")
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?
        .read()
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?;
    let conn: Array1<i64> = block
        .dataset("Connectivity")
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?
        .read()
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?;
    let types: Array1<usize> = block
        .dataset("Types")
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?
        .read()
        .map_err(|e| MefikitIOError::Parse(e.to_string()))?;

    let mut mesh = UMesh::new(points.into());
    for i in 0..types.len() {
        let start = offsets[i];
        let end = offsets[i + 1];
        let el_type = VTKHDF_MAPPING.to_element(types[i] as u32)?;
        let cell_conn: Vec<usize> = conn
            .slice(s![start..end])
            .iter()
            .map(|&x| x as usize)
            .collect();
        mesh.add_element(el_type, &cell_conn, None, None);
    }
    Ok(mesh)
}

pub fn read(path: &Path) -> Result<UMesh, MefikitIOError> {
    let file = File::open(path).map_err(|e| MefikitIOError::Parse(e.to_string()))?;
    let vtk = file
        .group("VTKHDF")
        .map_err(|_| MefikitIOError::MalformedFile("Not a VTKHDF file".to_string()))?;
    eprintln!("I survived");

    match read_group_attr(&vtk, "Type")?.as_str() {
        "UnstructuredGrid" => return handle_unstructured(&vtk),
        "PartitionedDataSetCollection" | "MultiBlockDataSet" => {
            for name in vtk
                .member_names()
                .map_err(|e| MefikitIOError::Parse(e.to_string()))?
            {
                let block = vtk
                    .group(name.as_str())
                    .map_err(|e| MefikitIOError::Parse(e.to_string()))?;
                let Ok(_) = block.attr("Type") else { continue };
                match read_group_attr(&block, "Type")?.as_str() {
                    "UnstructuredGrid" => return handle_unstructured(&block),
                    _ => continue,
                }
            }
        }
        _ => {}
    }
    Err(MefikitIOError::MalformedFile(format!(
        "Cannot read {}: Group should be of attribute `UnstructuredGrid`,
        `PartitionedDataSetCollection` or `MultiblockDataSet`",
        path.display()
    )))
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
    let file = File::create(path).map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    let vtk = file
        .create_group("VTKHDF")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;

    vtk.new_attr::<FixedAscii<16>>()
        .shape(())
        .create("Type")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write_scalar(
            &FixedAscii::<16>::from_ascii("UnstructuredGrid")
                .map_err(|e| MefikitIOError::Encode(e.to_string()))?,
        )
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;

    vtk.new_attr::<i64>()
        .shape([2])
        .create("Version")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write(&arr1(&[2i64, 0]))
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;

    // VTKHDF requires Points to always have 3 components. Pad lower-dimensional
    // meshes (e.g. 2D) with zeros so the dataset is N x 3.
    let src = mesh.coords();
    let n_points = src.nrows();
    let dim = src.ncols().min(3);
    let mut coords: Array2<f64> = Array2::zeros((n_points, 3));
    coords
        .slice_mut(s![.., ..dim])
        .assign(&src.slice(s![.., ..dim]));

    let mut types: Vec<u8> = Vec::new();
    // VTKHDF stores Connectivity and Offsets as Int64.
    let mut offsets: Vec<i64> = vec![0];
    let mut connectivity: Vec<i64> = Vec::new();

    for el in mesh.elements() {
        let conn = el.connectivity();
        let code = VTKHDF_MAPPING.to_code(el.element_type()).ok_or_else(|| {
            MefikitIOError::Encode(format!(
                "Unsupported ElementType for VTKHDF: {:?}",
                el.element_type()
            ))
        })?;
        types.push(code as u8);
        connectivity.extend(conn.iter().map(|&x| x as i64));
        offsets.push(connectivity.len() as i64);
    }

    // A single, non-partitioned dataset: each "NumberOf*" array has one entry.
    let n_cells = types.len() as i64;
    let n_conn_ids = connectivity.len() as i64;
    for (name, value) in [
        ("NumberOfPoints", n_points as i64),
        ("NumberOfCells", n_cells),
        ("NumberOfConnectivityIds", n_conn_ids),
    ] {
        vtk.new_dataset::<i64>()
            .shape([1])
            .create(name)
            .map_err(|e| MefikitIOError::Encode(e.to_string()))?
            .write(&arr1(&[value]))
            .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    }

    vtk.new_dataset::<f64>()
        .shape(coords.shape())
        .create("Points")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write(&coords)
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    vtk.new_dataset::<u8>()
        .shape([types.len()])
        .create("Types")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write(&Array1::from(types))
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    vtk.new_dataset::<i64>()
        .shape([offsets.len()])
        .create("Offsets")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write(&Array1::from(offsets))
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
    vtk.new_dataset::<i64>()
        .shape([connectivity.len()])
        .create("Connectivity")
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?
        .write(&Array1::from(connectivity))
        .map_err(|e| MefikitIOError::Encode(e.to_string()))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_examples as me;
    use std::path::PathBuf;

    #[test]
    fn test_read_hdfvtk() {
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/single_hex8.vtkhdf"
        ));
        let mesh = read(&path).unwrap();
    }

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

    /// Checks that the file written matches the VTKHDF UnstructuredGrid spec:
    /// the mandatory `NumberOf*` datasets are present, `Points` has 3 columns,
    /// and the integer datasets use Int64.
    #[test]
    fn test_write_is_vtkhdf_compliant() {
        let path = PathBuf::from("test_compliant.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        write(&path, mesh.view()).unwrap();

        let file = File::open(&path).unwrap();
        let vtk = file.group("VTKHDF").unwrap();

        // Mandatory single-entry summary datasets.
        let n_points: Array1<i64> = vtk.dataset("NumberOfPoints").unwrap().read().unwrap();
        let n_cells: Array1<i64> = vtk.dataset("NumberOfCells").unwrap().read().unwrap();
        let n_conn: Array1<i64> = vtk
            .dataset("NumberOfConnectivityIds")
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(n_points.as_slice().unwrap(), &[5]);
        assert_eq!(n_cells.as_slice().unwrap(), &[4]);
        assert_eq!(n_conn.as_slice().unwrap(), &[13]);

        // Points must always have 3 components, even for a 2D mesh.
        let points = vtk.dataset("Points").unwrap();
        assert_eq!(points.shape(), vec![5, 3]);

        // Connectivity / Offsets are Int64.
        assert_eq!(
            vtk.dataset("Connectivity").unwrap().dtype().unwrap().size(),
            8
        );
        assert_eq!(vtk.dataset("Offsets").unwrap().dtype().unwrap().size(), 8);

        std::fs::remove_file(path).unwrap();
    }
}
