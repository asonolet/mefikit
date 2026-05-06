use crate::mesh::{UMesh, UMeshView};
use std::path::Path;

mod serde_io;
pub mod vtk_io;
mod hdfvtk_io;
mod cgns_io;
// mod med; // for later

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::read_json(path),
        "yaml" | "yml" => serde_io::read_yaml(path),
        "vtk" | "vtu" => vtk_io::read(path),
        "vtkhdf" | "h5" | "hdf5" => hdfvtk_io::read(path),
        "cgns" => todo!(),
        _ => Err(format!("Unsupported file extension: {path:?}").into()),
    }
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::write_json(path, mesh),
        "yaml" | "yml" => serde_io::write_yaml(path, mesh),
        "vtk" | "vtu" => vtk_io::write(path, mesh),
        "vtkhdf" | "h5" | "hdf5" => hdfvtk_io::write(path, mesh),
        _ => Err(format!("Unsupported file extension: {path:?}").into()),
    }
}
