use super::error::MefikitIOError;
use crate::mesh::{UMesh, UMeshView};
use std::fs::File;
use std::path::Path;

pub fn read_json(path: &Path) -> Result<UMesh, MefikitIOError> {
    let file = File::open(path)?;
    serde_json::from_reader(file).map_err(|e| MefikitIOError::Parse(e.to_string()))
}

pub fn write_json(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, &mesh).map_err(|e| MefikitIOError::Encode(e.to_string()))
}

pub fn read_yaml(path: &Path) -> Result<UMesh, MefikitIOError> {
    let file = File::open(path)?;
    serde_yaml::from_reader(file).map_err(|e| MefikitIOError::Parse(e.to_string()))
}

pub fn write_yaml(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
    let file = File::create(path)?;
    serde_yaml::to_writer(file, &mesh).map_err(|e| MefikitIOError::Encode(e.to_string()))
}
