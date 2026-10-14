use super::cell_index::{
    BlockLayout, field_from_global_rows, flatten_field_for_dim, global_ids_of_dim,
};
use super::error::MefikitIOError;
use super::hdf_utils::read_group_attr;
use super::vtk_io::VTK_MAPPING;
use crate::mesh::{Dimension, ElementLike, UMesh, UMeshView};
use hdf5_metno::{File, Group, types::FixedAscii};
use ndarray::{Array1, Array2, ArrayD, arr1, s};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// HDF5 names are hierarchical: a `/` in a field or group name would be
/// interpreted as a path separator, so reject such names up front.
fn validate_dataset_name(name: &str) -> Result<(), MefikitIOError> {
    if name.is_empty() || name.contains('/') {
        return Err(MefikitIOError::Encode(format!(
            "field/group name {name:?} is not a valid VTKHDF dataset name"
        )));
    }
    Ok(())
}

fn write_field_attribute(
    dataset: &hdf5_metno::Dataset,
    flat: &ArrayD<f64>,
) -> Result<(), MefikitIOError> {
    // VTKHDF uses the optional `Attribute` on a data array to tell consumers how
    // to interpret it. Only scalars and 3-component vectors have a canonical
    // meaning; anything else is left untagged.
    let attribute = match flat.shape() {
        [_] => Some("Scalars"),
        [_, 1] => Some("Scalars"),
        [_, 3] => Some("Vectors"),
        _ => None,
    };
    if let Some(attribute) = attribute {
        let ascii = FixedAscii::<8>::from_ascii(attribute)
            .map_err(|e| MefikitIOError::Encode(e.to_string()))?;
        dataset
            .new_attr::<FixedAscii<8>>()
            .shape(())
            .create("Attribute")?
            .write_scalar(&ascii)?;
    }
    Ok(())
}

/// Writes cell-centered fields into the VTKHDF `CellData` group.
///
/// VTKHDF has no notion of topological dimension for cell data, so a flat
/// `CellData` array can only represent fields of a mesh with a single
/// topological dimension. On mixed-dimension meshes fields are skipped (and
/// this limitation is documented).
fn write_fields(vtk: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let dimensions: BTreeSet<Dimension> = mesh.element_types().map(|et| et.dimension()).collect();
    if dimensions.len() != 1 {
        return Ok(());
    }
    let dim = *dimensions.iter().next().unwrap();

    let mut fields: Vec<(String, Dimension)> = mesh
        .fields()
        .filter_map(|(name, field)| field.dimension().map(|dim| (name, dim)))
        .collect();
    fields.sort();
    fields.retain(|(_, field_dim)| *field_dim == dim);

    if fields.is_empty() {
        return Ok(());
    }

    let cell_data = vtk.create_group("CellData")?;
    for (name, _) in fields {
        validate_dataset_name(&name)?;
        let flat = flatten_field_for_dim(mesh, &name, dim).ok_or_else(|| {
            MefikitIOError::Encode(format!("field {name:?} cannot be flattened at {dim:?}"))
        })?;
        let dataset = cell_data
            .new_dataset::<f64>()
            .shape(flat.shape())
            .create(name.as_str())?;
        dataset.write(&flat)?;
        write_field_attribute(&dataset, &flat)?;
    }
    Ok(())
}

/// Writes element groups into the VTKHDF `GroupData` group.
///
/// `GroupData` is a mefikit extension: VTKHDF has no standard group/subset
/// concept. Each group becomes an Int64 dataset of 0-based global cell indices
/// (using the canonical [`BlockLayout`] numbering).
fn write_groups(vtk: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let mut group_names = mesh.group_names();
    group_names.sort();

    if group_names.is_empty() {
        return Ok(());
    }

    let layout = BlockLayout::new(mesh);
    let group_data = vtk.create_group("GroupData")?;
    for name in group_names {
        validate_dataset_name(&name)?;
        let ids = layout.group_global_ids(mesh, &name);
        group_data
            .new_dataset::<i64>()
            .shape([ids.len()])
            .create(name.as_str())?
            .write(&Array1::from(ids))?;
    }
    Ok(())
}

/// Reads cell-centered fields from the VTKHDF `CellData` group.
///
/// Only supported for meshes with a single topological dimension (the same
/// restriction as the writer); otherwise the fields are skipped.
fn read_fields(
    block: &Group,
    mesh: &mut UMesh,
    layout: &BlockLayout,
) -> Result<(), MefikitIOError> {
    let Ok(cell_data) = block.group("CellData") else {
        return Ok(());
    };

    let dimensions: BTreeSet<Dimension> = mesh.element_types().map(|et| et.dimension()).collect();
    if dimensions.len() != 1 {
        return Ok(());
    }
    let dim = *dimensions.iter().next().unwrap();
    let globals = global_ids_of_dim(layout, dim);

    for name in cell_data.member_names()? {
        let Ok(dataset) = cell_data.dataset(&name) else {
            continue;
        };
        let Ok(data): Result<ArrayD<f64>, _> = dataset.read() else {
            continue;
        };
        let Ok(field) = field_from_global_rows(mesh, layout, dim, &data.view(), &globals) else {
            continue;
        };
        mesh.update_field(&name, field);
    }
    Ok(())
}

/// Reads element groups from the VTKHDF `GroupData` group.
fn read_groups(
    block: &Group,
    mesh: &mut UMesh,
    layout: &BlockLayout,
) -> Result<(), MefikitIOError> {
    let Ok(group_data) = block.group("GroupData") else {
        return Ok(());
    };

    let mut groups: BTreeMap<String, crate::mesh::ElementIds> = BTreeMap::new();
    for name in group_data.member_names()? {
        let Ok(dataset) = group_data.dataset(&name) else {
            continue;
        };
        let Ok(ids): Result<Array1<i64>, _> = dataset.read() else {
            continue;
        };
        let element_ids = layout.element_ids_from_globals(ids.iter().copied());
        if !element_ids.is_empty() {
            groups.insert(name, element_ids);
        }
    }
    if !groups.is_empty() {
        mesh.set_groups(groups);
    }
    Ok(())
}

fn handle_unstructured(block: &Group) -> Result<UMesh, MefikitIOError> {
    let points: Array2<f64> = block.dataset("Points")?.read()?;
    let offsets: Array1<usize> = block.dataset("Offsets")?.read()?;
    let conn: Array1<i64> = block.dataset("Connectivity")?.read()?;
    let types: Array1<usize> = block.dataset("Types")?.read()?;

    let mut mesh = UMesh::new(points.into());
    for i in 0..types.len() {
        let start = offsets[i];
        let end = offsets[i + 1];
        let el_type = VTK_MAPPING.to_element(types[i] as u32)?;
        let cell_conn: Vec<usize> = conn
            .slice(s![start..end])
            .iter()
            .map(|&x| x as usize)
            .collect();
        mesh.add_element(el_type, &cell_conn, None);
    }

    let layout = BlockLayout::new(&mesh);
    read_fields(block, &mut mesh, &layout)?;
    read_groups(block, &mut mesh, &layout)?;

    Ok(mesh)
}

pub fn read(path: &Path) -> Result<UMesh, MefikitIOError> {
    let file = File::open(path)?;
    let vtk = file
        .group("VTKHDF")
        .map_err(|_| MefikitIOError::MalformedFile("Not a VTKHDF file".to_string()))?;

    match read_group_attr(&vtk, "Type")?.as_str() {
        "UnstructuredGrid" => return handle_unstructured(&vtk),
        "PartitionedDataSetCollection" | "MultiBlockDataSet" => {
            for name in vtk.member_names()? {
                let block = vtk.group(name.as_str())?;
                let Ok(_) = block.attr("Type") else { continue };
                match read_group_attr(&block, "Type")?.as_str() {
                    "UnstructuredGrid" => return handle_unstructured(&block),
                    _ => continue,
                }
            }
        }
        _ => {}
    }
    Err(MefikitIOError::MalformedFile(format!(
        "Cannot read {}: Group should be of attribute `UnstructuredGrid`,
        `PartitionedDataSetCollection` or `MultiblockDataSet`",
        path.display()
    )))
}

pub fn write(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
    let file = File::create(path)?;
    let vtk = file.create_group("VTKHDF")?;

    vtk.new_attr::<FixedAscii<16>>()
        .shape(())
        .create("Type")?
        .write_scalar(
            &FixedAscii::<16>::from_ascii("UnstructuredGrid")
                .map_err(|e| MefikitIOError::Encode(e.to_string()))?,
        )?;

    vtk.new_attr::<i64>()
        .shape([2])
        .create("Version")?
        .write(&arr1(&[2i64, 0]))?;

    // VTKHDF requires Points to always have 3 components. Pad lower-dimensional
    // meshes (e.g. 2D) with zeros so the dataset is N x 3.
    let src = mesh.coords();
    let n_points = src.nrows();
    let dim = src.ncols().min(3);
    let mut coords: Array2<f64> = Array2::zeros((n_points, 3));
    coords
        .slice_mut(s![.., ..dim])
        .assign(&src.slice(s![.., ..dim]));

    let mut types: Vec<u8> = Vec::new();
    // VTKHDF stores Connectivity and Offsets as Int64.
    let mut offsets: Vec<i64> = vec![0];
    let mut connectivity: Vec<i64> = Vec::new();

    for el in mesh.elements() {
        let conn = el.connectivity();
        let code = VTK_MAPPING.to_code(el.element_type()).ok_or_else(|| {
            MefikitIOError::Encode(format!(
                "Unsupported ElementType for VTKHDF: {:?}",
                el.element_type()
            ))
        })?;
        types.push(code as u8);
        connectivity.extend(conn.iter().map(|&x| x as i64));
        offsets.push(connectivity.len() as i64);
    }

    // A single, non-partitioned dataset: each "NumberOf*" array has one entry.
    let n_cells = types.len() as i64;
    let n_conn_ids = connectivity.len() as i64;
    for (name, value) in [
        ("NumberOfPoints", n_points as i64),
        ("NumberOfCells", n_cells),
        ("NumberOfConnectivityIds", n_conn_ids),
    ] {
        vtk.new_dataset::<i64>()
            .shape([1])
            .create(name)?
            .write(&arr1(&[value]))?;
    }

    vtk.new_dataset::<f64>()
        .shape(coords.shape())
        .create("Points")?
        .write(&coords)?;
    vtk.new_dataset::<u8>()
        .shape([types.len()])
        .create("Types")?
        .write(&Array1::from(types))?;
    vtk.new_dataset::<i64>()
        .shape([offsets.len()])
        .create("Offsets")?
        .write(&Array1::from(offsets))?;
    vtk.new_dataset::<i64>()
        .shape([connectivity.len()])
        .create("Connectivity")?
        .write(&Array1::from(connectivity))?;

    write_fields(&vtk, &mesh)?;
    write_groups(&vtk, &mesh)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::hdf_utils::hdf5_test_guard;
    use crate::mesh::ElementType;
    use crate::mesh::FieldBase;
    use crate::mesh_examples as me;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[test]
    fn test_read_hdfvtk() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/single_hex8.vtkhdf"
        ));
        let _mesh = read(&path).unwrap();
    }

    #[test]
    fn test_write_hdfvtk() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("test_write.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        assert!(write(&path, mesh.view()).is_ok());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_roundtrip_hdfvtk() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("test_roundtrip.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        assert!(write(&path, mesh.view()).is_ok());
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        for (e1, e2) in mesh.elements().zip(mesh2.elements()) {
            assert_eq!(e1.connectivity, e2.connectivity);
        }
    }

    /// A mesh with a single topological dimension (only QUAD4 cells), needed to
    /// exercise the `CellData` field path.
    fn field_mesh() -> UMesh {
        me::make_imesh_2d(2)
    }

    #[test]
    fn test_roundtrip_hdfvtk_fields() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("test_roundtrip_fields.vtkhdf");

        let mut mesh = field_mesh();
        let mut blocks = BTreeMap::new();
        for (et, block) in mesh.blocks() {
            let n = block.len();
            let values: Vec<f64> = (0..n).map(|i| (i as f64) * 2.0).collect();
            blocks.insert(
                *et,
                ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), values)
                    .unwrap()
                    .into_shared(),
            );
        }
        mesh.update_field("temp", FieldBase(blocks));

        write(&path, mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        let field = mesh2.field("temp", None).unwrap();
        for (et, original) in mesh.blocks() {
            let n = original.len();
            let written = &field.0[et];
            for i in 0..n {
                assert_eq!(written[[i]], (i as f64) * 2.0);
            }
        }
    }

    #[test]
    fn test_roundtrip_hdfvtk_groups() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("test_roundtrip_groups.vtkhdf");

        // A mixed-dimension mesh: the group spans SEG2 (D1) and the D2 cells,
        // which is exactly what the global-id encoding has to preserve.
        let mut mesh = me::make_mesh_2d_multi();
        let mut element_ids = crate::mesh::ElementIds::new();
        for (et, block) in mesh.blocks() {
            let indices: Vec<usize> = (0..block.len()).step_by(2).collect();
            if !indices.is_empty() {
                element_ids.add_block(*et, indices);
            }
        }
        let mut groups = BTreeMap::new();
        groups.insert("left".to_string(), element_ids);
        mesh.set_groups(groups);

        write(&path, mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        assert_eq!(mesh.group_names(), mesh2.group_names());
        assert_eq!(
            mesh.group_elements("left").iter().count(),
            mesh2.group_elements("left").iter().count()
        );
        for id in mesh.group_elements("left").iter() {
            assert!(mesh2.in_group(id, "left"));
        }
    }

    /// Checks that the file written matches the VTKHDF UnstructuredGrid spec:
    /// the mandatory `NumberOf*` datasets are present, `Points` has 3 columns,
    /// and the integer datasets use Int64.
    #[test]
    fn test_write_is_vtkhdf_compliant() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("test_compliant.vtkhdf");
        let mesh = me::make_mesh_2d_multi();
        write(&path, mesh.view()).unwrap();

        let file = File::open(&path).unwrap();
        let vtk = file.group("VTKHDF").unwrap();

        // Mandatory single-entry summary datasets.
        let n_points: Array1<i64> = vtk.dataset("NumberOfPoints").unwrap().read().unwrap();
        let n_cells: Array1<i64> = vtk.dataset("NumberOfCells").unwrap().read().unwrap();
        let n_conn: Array1<i64> = vtk
            .dataset("NumberOfConnectivityIds")
            .unwrap()
            .read()
            .unwrap();
        assert_eq!(n_points.as_slice().unwrap(), &[5]);
        assert_eq!(n_cells.as_slice().unwrap(), &[4]);
        assert_eq!(n_conn.as_slice().unwrap(), &[13]);

        // Points must always have 3 components, even for a 2D mesh.
        let points = vtk.dataset("Points").unwrap();
        assert_eq!(points.shape(), vec![5, 3]);

        // Connectivity / Offsets are Int64.
        assert_eq!(
            vtk.dataset("Connectivity").unwrap().dtype().unwrap().size(),
            8
        );
        assert_eq!(vtk.dataset("Offsets").unwrap().dtype().unwrap().size(), 8);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn repro_single_block_group() {
        let _guard = hdf5_test_guard();
        let path = PathBuf::from("repro_single_block_group.vtkhdf");
        let coords =
            ndarray::Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect::<Vec<_>>())
                .unwrap();
        let mut mesh = UMesh::new(coords.into_shared());
        mesh.add_element(ElementType::QUAD4, &[0, 1, 2, 3], None);
        mesh.add_element(ElementType::QUAD4, &[1, 2, 3, 4], None);
        mesh.add_element(ElementType::QUAD4, &[4, 5, 6, 7], None);
        mesh.add_element(ElementType::QUAD4, &[5, 6, 7, 8], None);

        let mut element_ids = crate::mesh::ElementIds::new();
        element_ids.add_block(ElementType::QUAD4, vec![0, 2]);
        let mut groups = BTreeMap::new();
        groups.insert("left".to_string(), element_ids);
        mesh.set_groups(groups);

        write(&path, mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(path).unwrap();

        eprintln!(
            "entries: {:?}",
            BlockLayout::new(&mesh).entries().collect::<Vec<_>>()
        );
        eprintln!("src ids: {:?}", mesh.group_elements("left"));
        eprintln!("back ids: {:?}", mesh2.group_elements("left"));
    }
}
