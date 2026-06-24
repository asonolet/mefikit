use thiserror::Error;

#[derive(Error, Debug)]
pub enum MefikitIOError {
    #[error("Unsupported file extension: {0}")]
    UnsupportedFileExtension(String),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serde JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Serde YAML error: {0}")]
    Yaml(#[from] serde_yaml::Error),
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5_metno::Error),
    #[error("Shape error: {0}")]
    NdArray(#[from] ndarray::ShapeError),
    #[cfg(feature = "io")]
    #[error("VTK error: {0}")]
    Vtk(#[from] vtkio::Error),
    #[error("Malformed file: {0}")]
    MalformedFile(String),
}
