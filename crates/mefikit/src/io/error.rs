use thiserror::Error;

#[derive(Error, Debug)]
pub enum MefikitIOError {
    #[error("Unsupported file extension: {0}")]
    UnsupportedFileExtension(String),
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
}
