use hdf5_metno::types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5_metno::Group;
use ndarray::{Array1, Array2};

use super::error::MefikitIOError;

pub fn read_group_attr(group: &hdf5_metno::Group, name: &str) -> Result<String, MefikitIOError> {
    let attr = group
        .attr(name)
        .map_err(|e| MefikitIOError::Hdf(e))?;
    let dtype = attr
        .dtype()
        .map_err(|e| MefikitIOError::Hdf(e))?;
    let desc = dtype
        .to_descriptor()
        .map_err(|e| MefikitIOError::Hdf(e))?;

    match desc {
        TypeDescriptor::VarLenUnicode => {
            let s: VarLenUnicode = attr
                .read_scalar()
                .map_err(|e| MefikitIOError::Hdf(e))?;
            Ok(s.to_string())
        }
        TypeDescriptor::VarLenAscii => {
            let s: VarLenAscii = attr
                .read_scalar()
                .map_err(|e| MefikitIOError::Hdf(e))?;
            Ok(s.to_string())
        }
        TypeDescriptor::FixedAscii(_) => {
            let s: FixedAscii<64> = attr
                .read_scalar()
                .map_err(|e| MefikitIOError::Hdf(e))?;
            Ok(s.as_str().trim_end_matches('\0').to_string())
        }
        TypeDescriptor::FixedUnicode(_) => {
            let s: FixedUnicode<64> = attr
                .read_scalar()
                .map_err(|e| MefikitIOError::Hdf(e))?;
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
        "I4" => {
            ds.as_reader().read_dyn::<i32>()?
                .iter()
                .map(|&x| x as i64)
                .collect()
        }
        "I8" => {
            ds.as_reader().read_dyn::<i64>()?
                .into_raw_vec_and_offset().0
        }
        _ => {
            return Err(MefikitIOError::MalformedFile(format!(
                "Unexpected index array type: {type_str}"
            )))
        }
    };
    
    Ok(values)
}