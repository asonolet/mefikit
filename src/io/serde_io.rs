use crate::UMesh;
use std::fs::File;
use std::path::Path;

pub fn read_json(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mesh = serde_json::from_reader(file)?;
    Ok(mesh)
}

pub fn write_json(path: &Path, mesh: &UMesh) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    serde_json::to_writer_pretty(file, mesh)?;
    Ok(())
}

pub fn read_yaml(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mesh = serde_yaml::from_reader(file)?;
    Ok(mesh)
}

pub fn write_yaml(path: &Path, mesh: &UMesh) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(path)?;
    serde_yaml::to_writer(file, mesh)?;
    Ok(())
}
