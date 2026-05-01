//! Mesh I/O operations for reading and writing mesh files.
//!
//! Supports JSON, YAML, and VTK/VTU formats.

use crate::mesh::{UMesh, UMeshView};
use std::path::Path;

mod serde_io;
mod vtk_io;

/// Reads a mesh from the given file path.
///
/// The file format is determined by the file extension.
/// Supported formats: JSON, YAML, VTK, VTU.
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
        _ => Err(format!("Unsupported file extension: {path:?}").into()),
    }
}

/// Writes a mesh to the given file path.
///
/// The file format is determined by the file extension.
/// Supported formats: JSON, YAML, VTK, VTU.
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
        _ => Err(format!("Unsupported file extension: {path:?}").into()),
    }
}
