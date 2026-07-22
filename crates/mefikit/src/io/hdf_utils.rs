use super::error::MefikitIOError;
use hdf5_metno::Group;
use hdf5_metno::types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode};

/// libhdf5 is linked statically without `--enable-threadsafe`, so its internal
/// API-context stack (`H5CX`) is a plain global rather than thread-local. Two
/// threads inside libhdf5 at once corrupt it, which shows up as an assertion
/// failure in `H5CX_get_vec_size` or a plain SIGSEGV. Cargo runs tests in
/// parallel, so *every* test that touches HDF5 - CGNS or VTKHDF - must
/// serialize through this one lock. Poison-tolerant so a panicking test does
/// not wedge the rest.
#[cfg(test)]
static HDF5_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquires the process-wide HDF5 test lock. Hold the guard for the whole
/// duration of any test that opens, reads or writes an HDF5 file.
#[cfg(test)]
pub fn hdf5_test_guard() -> std::sync::MutexGuard<'static, ()> {
    HDF5_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn read_group_attr(group: &hdf5_metno::Group, name: &str) -> Result<String, MefikitIOError> {
    let attr = group.attr(name).map_err(MefikitIOError::Hdf)?;
    let dtype = attr.dtype().map_err(MefikitIOError::Hdf)?;
    let desc = dtype.to_descriptor().map_err(MefikitIOError::Hdf)?;

    match desc {
        TypeDescriptor::VarLenUnicode => {
            let s: VarLenUnicode = attr.read_scalar().map_err(MefikitIOError::Hdf)?;
            Ok(s.to_string())
        }
        TypeDescriptor::VarLenAscii => {
            let s: VarLenAscii = attr.read_scalar().map_err(MefikitIOError::Hdf)?;
            Ok(s.to_string())
        }
        TypeDescriptor::FixedAscii(_) => {
            let s: FixedAscii<64> = attr.read_scalar().map_err(MefikitIOError::Hdf)?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        TypeDescriptor::FixedUnicode(_) => {
            let s: FixedUnicode<64> = attr.read_scalar().map_err(MefikitIOError::Hdf)?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        other => Err(MefikitIOError::MalformedFile(format!(
            "Unexpected string type: {other:?}"
        ))),
    }
}

pub fn read_string_data(group: &Group) -> Result<String, MefikitIOError> {
    let s: String = group
        .dataset(" data")?
        .as_reader()
        .read_1d::<i8>()?
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8 as char)
        .collect();
    Ok(s.trim().to_string())
}

pub fn read_index_array(group: &Group) -> Result<Vec<i64>, MefikitIOError> {
    let ds = group.dataset(" data")?;
    let type_str = read_group_attr(group, "type")?;

    let values: Vec<i64> = match type_str.as_str() {
        "I4" => ds
            .as_reader()
            .read_dyn::<i32>()?
            .iter()
            .map(|&x| x as i64)
            .collect(),
        "I8" => {
            ds.as_reader()
                .read_dyn::<i64>()?
                .into_raw_vec_and_offset()
                .0
        }
        _ => {
            return Err(MefikitIOError::MalformedFile(format!(
                "Unexpected index array type: {type_str}"
            )));
        }
    };

    Ok(values)
}
