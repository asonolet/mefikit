use crate::{UMesh, UMeshView};
use std::path::Path;

// mod serde_io;
mod vtk_io;
// mod med; // for later

pub fn read(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        // "json" => serde_io::read_json(path),
        // "yaml" | "yml" => serde_io::read_yaml(path),
        _ => Err(format!("Unsupported file extension: {:?}", path).into()),
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
        // "json" => serde_io::write_json(path, mesh),
        // "yaml" | "yml" => serde_io::write_yaml(path, mesh),
        "vtk" | "vtu" => vtk_io::write(path, mesh),
        _ => Err(format!("Unsupported file extension: {:?}", path).into()),
    }
}
