use thiserror::Error;

#[derive(Error, Debug)]
pub enum MefikitIOError {
    #[error("Unsupported file extension: {0}")]
    UnsupportedFileExtension(String),
    #[error("HDF5 error: {0}")]
    Hdf(#[from] hdf5_metno::Error),
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("Encode error: {0}")]
    Encode(String),
    #[error("Invalid data layout: {0}")]
    InvalidLayout(#[from] ndarray::ShapeError),
    #[error("Malformed file: {0}")]
    MalformedFile(String),
    /// The in-memory mesh cannot be represented in the target format (e.g. an
    /// element type with no equivalent in the format, or inconsistent
    /// polyhedral topology when exporting to CGNS NFACE_n/NGON_n).
    #[error("Invalid mesh for export: {0}")]
    InvalidMesh(String),
    /// A raw HDF5 C-API call (used for CGNS-specific null-terminated string
    /// attributes) returned a failure status.
    #[error("HDF5 C-API error: {0}")]
    Hdf5Sys(String),
}
