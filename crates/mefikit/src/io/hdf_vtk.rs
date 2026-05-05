use hdf5_metno::{File, types::FixedAscii};
use ndarray::{Array2, Array1, arr1, s};
use std::path::Path;
use crate::mesh::{ElementLike, ElementType, UMesh, UMeshView};

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

impl ElementType {
    pub fn into_vtk_u8(self) -> u8 {
        match self {
            ElementType::VERTEX => 1,
            ElementType::SEG2   => 3,
            ElementType::TRI3   => 5,
            ElementType::PGON   => 7,
            ElementType::QUAD4  => 9,
            ElementType::TET4   => 10,
            ElementType::HEX8   => 12,
            ElementType::PHED   => 42,
            other => panic!("Unsupported ElementType {other:?}")
        }
    }
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
    let mut offsets: Vec<usize> = Vec::new();
    let mut connectivity: Vec<usize> = Vec::new();

    for el in mesh.elements() {
        let conn = el.connectivity();
        types.push(ElementType::into_vtk_u8(el.element_type()));
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