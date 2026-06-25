//! Mesh I/O operations for reading and writing mesh files.
//!
//! Supports JSON, YAML, and VTK/VTU formats.

use crate::mesh::{UMesh, UMeshView};
use std::path::Path;

mod error;
mod hdfvtk_io;
mod serde_io;
mod vtk_io;

pub use error::MefikitIOError;

/// Reads a mesh from the given file path.
///
/// The file format is determined by the file extension.
/// Supported formats: JSON, YAML, VTK, VTU.
pub fn read(path: &Path) -> Result<UMesh, MefikitIOError> {
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
        _ => Err(MefikitIOError::UnsupportedFileExtension(format!(
            "{path:?}"
        ))),
    }
}

/// Writes a mesh to the given file path.
///
/// The file format is determined by the file extension.
/// Supported formats: JSON, YAML, VTK, VTU.
pub fn write(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
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
        _ => Err(MefikitIOError::UnsupportedFileExtension(format!(
            "{path:?}"
        ))),
    }
}
