use super::error::MefikitIOError;
use crate::mesh::ElementType;

#[derive(Debug)]
pub struct UnknownElement {
    pub format: &'static str,
    pub code: u32,
}

impl From<UnknownElement> for MefikitIOError {
    fn from(e: UnknownElement) -> Self {
        MefikitIOError::MalformedFile(format!(
            "Unsupported {} element code {}",
            e.format, e.code
        ))
    }
}

/// Bidirectional mapping between a file format's integer element codes and
/// mefikit [`ElementType`]s. Each file format owns one `const` instance built
/// from its own code table (see `VTK_MAPPING`, `VTKHDF_MAPPING`, `CGNS_MAPPING`).
pub struct ElementsMapping {
    name: &'static str,
    table: &'static [(u32, ElementType)],
}

impl ElementsMapping {
    pub const fn new(name: &'static str, table: &'static [(u32, ElementType)]) -> Self {
        Self { name, table }
    }

    pub fn to_element(&self, code: u32) -> Result<ElementType, UnknownElement> {
        self.table
            .iter()
            .find(|(c, _)| *c == code)
            .map(|(_, e)| *e)
            .ok_or(UnknownElement {
                format: self.name,
                code,
            })
    }

    pub fn to_code(&self, elem: ElementType) -> Option<u32> {
        self.table
            .iter()
            .find(|(_, e)| *e == elem)
            .map(|(c, _)| *c)
    }
}
