use crate::UMesh;
use std::path::Path;

mod serde_io;
// mod med; // for later

pub fn load_mesh(path: &Path) -> Result<UMesh, Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::read_json(path),
        "yaml" | "yml" => serde_io::read_yaml(path),
        _ => Err(format!("Unsupported file extension: {:?}", path).into()),
    }
}

pub fn save_mesh(path: &Path, mesh: &UMesh) -> Result<(), Box<dyn std::error::Error>> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "json" => serde_io::write_json(path, mesh),
        "yaml" | "yml" => serde_io::write_yaml(path, mesh),
        _ => Err(format!("Unsupported file extension: {:?}", path).into()),
    }
}
