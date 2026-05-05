use hdf5_metno::{File, types::FixedAscii};
use ndarray::{Array2, Array1, s};
use std::path::Path;
use crate::mesh::{ElementType, UMesh};

impl TryFrom<usize> for ElementType {
    type Error = Box<dyn std::error::Error>;
    
    fn try_from(code: usize) -> Result<Self, Self::Error> {
        match code {
            1  => Ok(ElementType::VERTEX),
            3  => Ok(ElementType::SEG2),
            5  => Ok(ElementType::TRI3),
            7  => Ok(ElementType::PGON),
            9  => Ok(ElementType::QUAD4),
            10 => Ok(ElementType::TET4),
            12 => Ok(ElementType::HEX8),
            42 => Ok(ElementType::PHED),
            other => Err(format!("Unsupported VTK cell type: {other}").into()),
        }
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
        let el_type = ElementType::try_from(types[i])?;
        let cell_conn: Vec<usize> = conn
            .slice(s![start..end])
            .iter()
            .map(|&x| x as usize)
            .collect();
        mesh.add_element(el_type, &cell_conn, None, None);
    }
    Ok(mesh)
}

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let vtkhdf_group = file
        .group("VTKHDF").map_err(|_| "Not a VTKHDF file")?;
    let groups = vtkhdf_group.member_names()?;

    // Match over block type and get the first one matching the caracteristics
    // Multiple blocks iteration is not supported here but you can extract 
    // from a multiblock vtkhdf file
    for group in groups {
        let block = vtkhdf_group.group(group.as_str())?;
        let kind: FixedAscii<64> = block.attr("Type")?.read_scalar()?;
        match kind.as_str().trim_end_matches('\0') {
            "UnstructuredGrid" => return handle_unstructured(&block),
            _ => continue,
        }
    }
    Err(format!("No VTKHDF group found in {}", path.display()).into())
}

pub fn write() {}