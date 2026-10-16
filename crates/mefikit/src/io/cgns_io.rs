use super::cell_index::{
    BlockLayout, field_from_global_rows, flatten_field_for_dim, global_ids_of_dim,
};
use super::elements_mapping::ElementsMapping;
use super::error::MefikitIOError;
use super::hdf_utils::{read_group_attr, read_index_array, read_string_data};
use crate::mesh::{Dimension, ElementIds, ElementLike, ElementType, UMesh, UMeshView};
use hdf5_metno::types::FixedAscii;
use hdf5_metno::{File, Group};
use ndarray::{Array1, Array2, ArrayD, arr1, arr2};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

// The cgns module is responsible for reading and writing CGNS files. It is strictly limited to cgns hdf format files since it uses hdf5-metno as an interface to the hdf5 library.
// future versions with cgns general support as a feature will be implemented

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CgnsBaseDim {
    pub cell_dim: usize,
    pub phys_dim: usize,
}

impl TryFrom<&Group> for CgnsBaseDim {
    type Error = MefikitIOError;

    fn try_from(base: &Group) -> Result<Self, Self::Error> {
        let data: Vec<i32> = base
            .dataset(" data")?
            .as_reader()
            .read_dyn::<i32>()?
            .into_raw_vec_and_offset()
            .0;

        if data.len() < 2 {
            return Err(MefikitIOError::MalformedFile(
                "CGNSBase_t data must have at least 2 elements".to_string(),
            ));
        }

        let base_data = Self {
            cell_dim: data[0] as usize,
            phys_dim: data[1] as usize,
        };

        base_data.validate()?;
        Ok(base_data)
    }
}

impl CgnsBaseDim {
    pub fn validate(&self) -> Result<(), MefikitIOError> {
        match (self.cell_dim, self.phys_dim) {
            (3, 3) | (2, 3) | (2, 2) | (1, 1) => Ok(()),
            other => Err(MefikitIOError::Parse(format!(
                "Unsupported dimension combo {other:?}"
            ))),
        }
    }
}

// CGNS element type codes (see the CGNS SIDS ElementType_t enumeration). Poly
// types (NGON_n/NFACE_n) have variable stride and are handled separately; their
// node count comes from `ElementType::num_nodes()` returning `None`.
const CGNS_MAPPING: ElementsMapping = ElementsMapping::new(
    "CGNS",
    &[
        (2, ElementType::VERTEX),
        (3, ElementType::SEG2),
        (5, ElementType::TRI3),
        (7, ElementType::QUAD4),
        (10, ElementType::TET4),
        (17, ElementType::HEX8),
        (22, ElementType::PGON),
        (23, ElementType::PHED),
    ],
);

fn cgns_label(group: &Group) -> Result<String, MefikitIOError> {
    let attr = group.attr(" label").or_else(|_| group.attr("label"))?;
    let label: String = attr
        .as_reader()
        .read_scalar::<FixedAscii<64>>()?
        .to_string();
    Ok(label.trim().trim_matches('\0').to_string())
}

fn cgns_name(group: &Group) -> Result<String, MefikitIOError> {
    read_group_attr(group, "name")
}

fn find_first_child_with_label(group: &Group, label: &str) -> Result<Group, MefikitIOError> {
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else {
            continue;
        };
        let Ok(lbl) = cgns_label(&child) else {
            continue;
        };
        if lbl == label {
            return Ok(child);
        }
    }
    Err(MefikitIOError::Parse(format!(
        "no child with label '{label}' in '{}'",
        group.name()
    )))
}

fn children_with_label(group: &Group, label: &str) -> Result<Vec<Group>, MefikitIOError> {
    let mut out = Vec::new();
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else {
            continue;
        };
        let Ok(lbl) = cgns_label(&child) else {
            continue;
        };
        if lbl == label {
            out.push(child);
        }
    }
    Ok(out)
}

fn read_coordinates(
    zone: &Group,
    phys_dim: usize,
) -> Result<ndarray::ArcArray2<f64>, MefikitIOError> {
    let gc = find_first_child_with_label(zone, "GridCoordinates_t")?;
    let names = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    let columns: Vec<Vec<f64>> = (0..phys_dim)
        .map(|i| {
            // CoordinateX/Y/Z are groups, data lives in their " data" dataset
            let coord_group = gc.group(names[i])?;
            let ds = coord_group.dataset(" data")?;

            // handle both R4 and R8 — check "type" attribute
            let type_attr = coord_group
                .attr("type")
                .and_then(|a| {
                    use hdf5_metno::types::FixedAscii;
                    a.as_reader().read_scalar::<FixedAscii<8>>()
                })
                .map(|s| s.to_string())
                .unwrap_or_else(|_| "R8".to_string());

            let values: Vec<f64> = if type_attr.trim_matches('\0').starts_with("R4") {
                ds.as_reader()
                    .read_1d::<f32>()?
                    .iter()
                    .map(|&v| v as f64)
                    .collect()
            } else {
                ds.as_reader().read_1d::<f64>()?.to_vec()
            };

            Ok(values)
        })
        .collect::<Result<_, MefikitIOError>>()?;

    let n = columns[0].len();
    let mut coords = ndarray::Array2::<f64>::zeros((n, phys_dim));
    for (col_idx, col_data) in columns.iter().enumerate() {
        coords
            .column_mut(col_idx)
            .iter_mut()
            .zip(col_data)
            .for_each(|(dst, &src)| *dst = src);
    }

    Ok(coords.into_shared())
}

// // inline to speed up the check since we'll be doing it for every element
// #[inline]
// fn is_ngon(cgns_code: i32) -> bool {
//     cgns_code == 22
// }

// // Return true if element describes a cell-face section (NFACE_n)
// #[inline]
// fn is_nfaces(cgns_code: i32) -> bool {
//     cgns_code == 23
// }

// Elements_t stores [ElementType, ElementSizeBoundary] in its " data" dataset;
// the first value is the CGNS element type code.
fn read_element_type(element: &Group) -> Result<i32, MefikitIOError> {
    let data: Vec<i32> = element
        .dataset(" data")?
        .as_reader()
        .read_dyn::<i32>()?
        .into_raw_vec_and_offset()
        .0;
    data.first().copied().ok_or_else(|| {
        MefikitIOError::MalformedFile(format!("Elements_t '{}' has empty data", element.name()))
    })
}

fn read_element_range(element: &Group) -> Result<[i64; 2], MefikitIOError> {
    let range_group = element.group("ElementRange")?;
    let values = read_index_array(&range_group)?;
    Ok([values[0], values[1]])
}

fn read_element_connectivity(element: &Group) -> Result<Vec<i64>, MefikitIOError> {
    let conn_group = element.group("ElementConnectivity")?;
    read_index_array(&conn_group)
}

fn read_phed_connectivity(element: &Group) -> Result<Vec<i64>, MefikitIOError> {
    let conn_group = element.group("ElementConnectivity")?;
    // PHED can contain negative values so substract here is irrelevant
    read_index_array(&conn_group)
}

fn read_element_offsets(element: &Group) -> Result<Option<Vec<i64>>, MefikitIOError> {
    let Ok(offset_group) = element.group("ElementStartOffset") else {
        return Ok(None);
    };
    let values = read_index_array(&offset_group)?;
    Ok(Some(values))
}

// CGNS defines two "polyhedral" element types with variable connectivity
// length: NGON_n and NFACE_n, where n is the number of nodes per face. They
// are encoded with cgns_code 22 and 23, respectively, and their connectivity
// is stored as a length-prefixed list of node indices: [n_nodes, v0, v1, ...,
// vn, n_nodes, v0, ...]. We need to handle these separately from the regular
// fixed-stride elements.
// ElementStartOffset = [0, 4, 9, 13, ...]
//                       ↑  ↑  ↑   ↑
//                       |  |  |   cell 3 starts at index 13
//                       |  |  cell 2 starts at index 9
//                       |  cell 1 starts at index 4
//                       cell 0 starts at index 0

// ElementConnectivity = [n0 n1 n2 n3 | n0 n1 n2 n3 n4 | n0 n1 n2 n3 | ...]
//                        ←— cell 0 —→  ←——— cell 1 ———→  ←— cell 2 —→
fn read_elements(mesh: &mut UMesh, zone: &Group) -> Result<(), MefikitIOError> {
    let el_group = children_with_label(zone, "Elements_t")?;

    // --- first pass: collect PGON and PHED raw data ---
    let mut pgon_offsets: Option<Vec<i64>> = None;
    let mut pgon_conn: Option<Vec<i64>> = None;
    let mut phed_offsets: Option<Vec<i64>> = None;
    let mut phed_conn: Option<Vec<i64>> = None;

    for element in &el_group {
        let code = read_element_type(element)?;
        // Unsupported sections are skipped with a warning rather than aborting.
        let element_type = match CGNS_MAPPING.to_element(code as u32) {
            Ok(et) => et,
            Err(_) => {
                eprintln!("warning: unsupported CGNS element type {code}, section skipped");
                continue;
            }
        };
        match element_type {
            ElementType::PGON => {
                let conn = read_element_connectivity(element)?;
                let offsets = read_element_offsets(element)?.ok_or_else(|| {
                    MefikitIOError::MalformedFile("PGON missing ElementStartOffset".to_string())
                })?;
                let range = read_element_range(element)?;
                let n_cells = (range[1] - range[0] + 1) as usize;
                for i in 0..n_cells {
                    let start = offsets[i] as usize;
                    let end = offsets[i + 1] as usize;
                    // CGNS indices are 1-based; mefikit connectivity is 0-based
                    // (indexes directly into `coords`), so subtract 1.
                    let nodes: Vec<usize> =
                        conn[start..end].iter().map(|&v| (v as usize) - 1).collect();
                    mesh.add_element(ElementType::PGON, &nodes, None);
                }
                pgon_offsets = Some(offsets);
                pgon_conn = Some(conn);
            }
            ElementType::PHED => {
                phed_conn = Some(read_phed_connectivity(element)?);
                phed_offsets = Some(read_element_offsets(element)?.ok_or_else(|| {
                    MefikitIOError::MalformedFile("PHED missing ElementStartOffset".to_string())
                })?);
            }
            other => {
                let range = read_element_range(element)?;
                let conn = read_element_connectivity(element)?;
                let n_cells = (range[1] - range[0] + 1) as usize;
                let nodes_per_cell = other.num_nodes().ok_or_else(|| {
                    MefikitIOError::MalformedFile(format!(
                        "CGNS element type {other:?} has no fixed node count"
                    ))
                })?;
                for i in 0..n_cells {
                    let start = i * nodes_per_cell;
                    let end = start + nodes_per_cell;
                    // CGNS indices are 1-based; convert to mefikit's 0-based
                    // connectivity by subtracting 1.
                    let cell: Vec<usize> =
                        conn[start..end].iter().map(|&v| (v as usize) - 1).collect();
                    mesh.add_element(other, &cell, None);
                }
            }
        }
    }

    // --- second pass: resolve PHED using PGON ---
    if let (Some(p_off), Some(p_conn), Some(f_off), Some(f_conn)) =
        (phed_offsets, phed_conn, pgon_offsets, pgon_conn)
    {
        let n_cells = p_off.len() - 1;
        for i in 0..n_cells {
            let start = p_off[i] as usize;
            let end = p_off[i + 1] as usize;

            let mut cell_nodes: Vec<usize> = Vec::new();

            for &face_ref in &p_conn[start..end] {
                let reversed = face_ref < 0;
                let face_index = (face_ref.unsigned_abs() as usize) - 1;

                let node_start = f_off[face_index] as usize;
                let node_end = f_off[face_index + 1] as usize;

                // Face
                let face: Vec<usize> = f_conn[node_start..node_end]
                    .iter()
                    .map(|&node_id| (node_id as usize) - 1) // 0-based
                    .collect();

                // Orientation
                if reversed {
                    cell_nodes.extend(face.iter().rev());
                } else {
                    cell_nodes.extend(face.iter());
                }
                cell_nodes.push(usize::MAX); // PHED separator
            }

            mesh.add_element(ElementType::PHED, &cell_nodes, None);
        }
    }

    Ok(())
}

/// Returns the unique topological dimension of the cells referenced by
/// `globals`, or `None` if the list is empty or mixes dimensions.
fn dim_of_globals(layout: &BlockLayout, globals: &[usize]) -> Option<Dimension> {
    let mut dim: Option<Dimension> = None;
    for &global in globals {
        let (element_type, _) = layout.global_to_type_local(global)?;
        let current = element_type.dimension();
        match dim {
            None => dim = Some(current),
            Some(existing) if existing == current => {}
            _ => return None,
        }
    }
    dim
}

/// Reads cell-centered `FlowSolution_t` nodes into `mesh`.
///
/// Only `GridLocation = CellCenter` solutions are supported (mefikit fields are
/// cell-centered). A solution without a `PointList` applies to every cell of the
/// zone, which only has a single topological dimension in the supported case.
fn read_fields(zone: &Group, mesh: &mut UMesh, layout: &BlockLayout) -> Result<(), MefikitIOError> {
    for solution in children_with_label(zone, "FlowSolution_t")? {
        let location = match solution.group("GridLocation") {
            Ok(grid_location) => read_string_data(&grid_location).unwrap_or_default(),
            // CGNS defaults GridLocation_t to Vertex when absent.
            Err(_) => "Vertex".to_string(),
        };
        if location != "CellCenter" {
            continue;
        }

        let point_list: Option<Vec<usize>> = match solution.group("PointList") {
            Ok(point_list) => Some(
                read_index_array(&point_list)?
                    .into_iter()
                    .filter(|&v| v >= 1)
                    .map(|v| (v - 1) as usize)
                    .collect(),
            ),
            Err(_) => None,
        };

        for data_array in children_with_label(&solution, "DataArray_t")? {
            let name = cgns_name(&data_array)?;
            let data: ArrayD<f64> = data_array.dataset(" data")?.as_reader().read_dyn::<f64>()?;
            // CGNS cell-centered solutions are scalar: skip multi-component data.
            if data.ndim() != 1 {
                continue;
            }

            let globals: Vec<usize> = match &point_list {
                Some(list) => list.clone(),
                None => {
                    // No PointList: the solution covers every cell of the zone,
                    // so it is only unambiguous for a single-dimension mesh.
                    let dims: BTreeSet<Dimension> =
                        mesh.element_types().map(|et| et.dimension()).collect();
                    if dims.len() != 1 {
                        continue;
                    }
                    global_ids_of_dim(layout, *dims.iter().next().unwrap())
                }
            };

            let Some(dim) = dim_of_globals(layout, &globals) else {
                continue;
            };
            let Ok(field) = field_from_global_rows(mesh, layout, dim, &data.view(), &globals)
            else {
                continue;
            };
            mesh.update_field(&name, field);
        }
    }
    Ok(())
}

/// Reads cell-centered `ZoneSubRegion_t` nodes into element groups.
fn read_groups(zone: &Group, mesh: &mut UMesh, layout: &BlockLayout) -> Result<(), MefikitIOError> {
    let mut groups: BTreeMap<String, ElementIds> = BTreeMap::new();

    for subregion in children_with_label(zone, "ZoneSubRegion_t")? {
        let name = cgns_name(&subregion)?;
        if name.is_empty() {
            continue;
        }
        let Ok(point_list) = subregion.group("PointList") else {
            continue;
        };
        let globals = read_index_array(&point_list)?
            .into_iter()
            .filter(|&v| v >= 1)
            .map(|v| v - 1);
        let ids = layout.element_ids_from_globals(globals);
        if ids.is_empty() {
            continue;
        }
        match groups.entry(name) {
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(ids);
            }
            std::collections::btree_map::Entry::Occupied(mut slot) => {
                for (element_type, indices) in ids.iter_blocks() {
                    slot.get_mut().add_block(*element_type, indices.clone());
                }
            }
        }
    }

    // `add_to_group` (not `set_groups`) preserves the exact cell indices: on a
    // freshly built mesh every cell shares family 0, and `set_groups` would map
    // the indices onto that family and thereby pull in every other cell too.
    for (name, element_ids) in &groups {
        mesh.add_to_group(name, element_ids);
    }
    Ok(())
}

// DISCLAIMER: the Family and BC connectivity are not handled in this version, but the code is
// structured to allow for future implementation of these features. The current implementation
// focuses on reading the mesh geometry and element connectivity from CGNS files, specifically
// handling unstructured meshes with PGON and PHED elements.
pub fn read(path: &Path) -> Result<UMesh, MefikitIOError> {
    let f = File::open(path)?;
    let base = find_first_child_with_label(&f.as_group()?, "CGNSBase_t")?;
    let cgns_dim = CgnsBaseDim::try_from(&base)?;

    let zone = find_first_child_with_label(&base, "Zone_t")?;

    let z_type = read_string_data(&find_first_child_with_label(&zone, "ZoneType_t")?)?;
    if z_type != "Unstructured" {
        return Err(MefikitIOError::Parse(format!(
            "unsupported zone type: {z_type}"
        )));
    }

    let coords = read_coordinates(&zone, cgns_dim.phys_dim)?;
    let mut mesh = UMesh::new(coords);

    read_elements(&mut mesh, &zone)?;

    let layout = BlockLayout::new(&mesh);
    read_fields(&zone, &mut mesh, &layout)?;
    read_groups(&zone, &mut mesh, &layout)?;

    // Future implementation: read families and boundary conditions
    Ok(mesh)
}

// ── write primitives ──────────────────────────────────────────────────────────
//
// ── CGNS node attribute helpers ──────────────────────────────────────────────
// Fixed-length, null-padded ASCII string attributes are required by CGNS SIDS
// for the "name", "label", and "type" attributes on every node (see CGNS SIDS
// §2.3.1 and §2.4.1).  `FixedAscii<N>` from hdf5-metno produces the correct
// on-disk format (H5T_STR_NULLPAD, which is byte-identical to NULLTERM for
// fixed-length strings).

fn write_str_attr<const N: usize>(
    group: &Group,
    name: &str,
    value: &str,
) -> Result<(), MefikitIOError> {
    let ascii = FixedAscii::<N>::from_ascii(value)
        .map_err(|e| MefikitIOError::Encode(format!("invalid attribute '{name}': {e}")))?;
    group
        .new_attr::<FixedAscii<N>>()
        .shape(())
        .create(name)?
        .write_scalar(&ascii)?;
    Ok(())
}

fn write_node_attrs(
    group: &Group,
    name: &str,
    label: &str,
    type_str: &str,
    flags: i32,
) -> Result<(), MefikitIOError> {
    write_str_attr::<33>(group, "label", label)?;
    write_str_attr::<33>(group, "name", name)?;
    write_str_attr::<3>(group, "type", type_str)?;
    group
        .new_attr::<i32>()
        .shape([1])
        .create("flags")?
        .write(&arr1(&[flags]))?;
    Ok(())
}

fn write_root_attrs(file: &File) -> Result<(), MefikitIOError> {
    let root = file.as_group()?;
    write_str_attr::<33>(&root, "label", "Root Node of HDF5 File")?;
    write_str_attr::<33>(&root, "name", "HDF5 MotherNode")?;
    write_str_attr::<3>(&root, "type", "MT")?;
    Ok(())
}

fn write_c1_data(group: &Group, value: &str) -> Result<(), MefikitIOError> {
    let bytes: Vec<i8> = value.bytes().map(|b| b as i8).collect();
    group
        .new_dataset::<i8>()
        .shape([bytes.len()])
        .create(" data")?
        .write(&Array1::from(bytes))?;
    Ok(())
}

// Writes a CGNS PointList (IndexArray_t) holding 1-based cell indices. For
// unstructured zones (index_dim = 1) the library stores the list as an
// `[n, index_dim]` dataset, i.e. one point per row, so the physical shape is
// `[n, 1]`.
fn write_point_list(parent: &Group, globals_0based: &[usize]) -> Result<(), MefikitIOError> {
    let node = parent.create_group("PointList")?;
    write_node_attrs(&node, "PointList", "IndexArray_t", "I8", 1)?;
    let values: Vec<i64> = globals_0based.iter().map(|&g| g as i64 + 1).collect();
    let data = Array2::from_shape_vec((values.len(), 1), values)?;
    node.new_dataset::<i64>()
        .shape(data.shape())
        .create(" data")?
        .write(&data)?;
    Ok(())
}

// ── write sub-functions ───────────────────────────────────────────────────────

fn write_version(file: &File) -> Result<(), MefikitIOError> {
    let node = file.create_group("CGNSLibraryVersion")?;
    write_node_attrs(&node, "CGNSLibraryVersion", "CGNSLibraryVersion_t", "R4", 0)?;
    node.new_dataset::<f32>()
        .shape([1])
        .create(" data")?
        .write(&arr1(&[4.0_f32]))?; // CPEX0031 offsets require CGNS >= 4.0
    Ok(())
}

fn write_base(file: &File, cell_d: i32, phys_d: i32) -> Result<Group, MefikitIOError> {
    let base = file.create_group("Base")?;
    write_node_attrs(&base, "Base", "CGNSBase_t", "I4", 1)?;
    base.new_dataset::<i32>()
        .shape([2])
        .create(" data")?
        .write(&arr1(&[cell_d, phys_d]))?;
    Ok(base)
}

fn write_zone(base: &Group, n_vertices: usize, n_cells: usize) -> Result<Group, MefikitIOError> {
    let zone = base.create_group("Zone1")?;
    write_node_attrs(&zone, "Zone1", "Zone_t", "I8", 1)?;
    // shape [3, 1] — matches real CGNS files, required by vtkCGNSReader.
    zone.new_dataset::<i32>()
        .shape([3, 1])
        .create(" data")?
        .write(&arr2(&[[n_vertices as i32], [n_cells as i32], [0_i32]]))?;

    let zt = zone.create_group("ZoneType")?;
    write_node_attrs(&zt, "ZoneType", "ZoneType_t", "C1", 1)?;
    write_c1_data(&zt, "Unstructured")?;

    Ok(zone)
}

fn write_coords(zone: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let gc = zone.create_group("GridCoordinates")?;
    write_node_attrs(&gc, "GridCoordinates", "GridCoordinates_t", "MT", 1)?;

    let coords = mesh.coords();
    let n = coords.nrows();
    let names = ["CoordinateX", "CoordinateY", "CoordinateZ"];

    for col in 0..mesh.space_dimension() {
        let data: Vec<f64> = (0..n).map(|i| coords[[i, col]]).collect();
        let coord_node = gc.create_group(names[col])?;
        write_node_attrs(&coord_node, names[col], "DataArray_t", "R8", 0)?;
        coord_node
            .new_dataset::<f64>()
            .shape([n])
            .create(" data")?
            .write(&Array1::from(data))?;
    }
    Ok(())
}

fn write_conn_and_offset(
    section: &Group,
    conn: &[i64],
    offsets: &[i64],
) -> Result<(), MefikitIOError> {
    let conn_node = section.create_group("ElementConnectivity")?;
    write_node_attrs(&conn_node, "ElementConnectivity", "DataArray_t", "I8", 1)?;
    conn_node
        .new_dataset::<i64>()
        .shape([conn.len()])
        .create(" data")?
        .write(&Array1::from(conn.to_vec()))?;

    let off_node = section.create_group("ElementStartOffset")?;
    write_node_attrs(&off_node, "ElementStartOffset", "DataArray_t", "I8", 1)?;
    off_node
        .new_dataset::<i64>()
        .shape([offsets.len()])
        .create(" data")?
        .write(&Array1::from(offsets.to_vec()))?;
    Ok(())
}

// Sign of an NFACE reference: +1 if `face` has the same cyclic orientation as
// the canonical PGON face `canon`, -1 if reversed. Faces with < 3 nodes carry
// no orientation, so return +1.
fn face_orientation(canon: &[usize], face: &[usize]) -> i64 {
    let n = canon.len();
    if n < 3 {
        return 1;
    }
    let pos = match face.iter().position(|&x| x == canon[0]) {
        Some(p) => p,
        None => return 1,
    };
    if face[(pos + 1) % n] == canon[1] {
        1
    } else {
        -1
    }
}

fn write_elements(zone: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    // Pre-pass: assign each block a contiguous, 1-based [start, end] element range.
    let mut ranges: Vec<(ElementType, i64, i64)> = Vec::new();
    let mut start = 1_i64;
    for (et, block) in mesh.blocks() {
        let n = block.len() as i64;
        ranges.push((*et, start, start + n - 1));
        start += n;
    }

    // Build the NGON face-index map from the PGON block:
    //   sorted-node-set -> (global cgns face index, canonical node order)
    let mut face_map: HashMap<Vec<usize>, (i64, Vec<usize>)> = HashMap::new();
    if let Some(&(_, pgon_start, _)) = ranges.iter().find(|(et, _, _)| *et == ElementType::PGON) {
        let block = mesh.block(ElementType::PGON).unwrap();
        for (i, el) in block.iter(mesh.coords()).enumerate() {
            let conn = el.connectivity().to_vec();
            let mut key = conn.clone();
            key.sort_unstable();
            face_map.insert(key, (pgon_start + i as i64, conn));
        }
    }

    for (et, r_start, r_end) in ranges.iter().copied() {
        let block = mesh.block(et).unwrap();
        let cgns_code = CGNS_MAPPING.to_code(et).ok_or_else(|| {
            MefikitIOError::InvalidMesh(format!("unsupported ElementType for CGNS export: {et:?}"))
        })? as i32;
        let section_name = format!("{et:?}");

        let section = zone.create_group(&section_name)?;
        write_node_attrs(&section, &section_name, "Elements_t", "I4", 1)?;
        section
            .new_dataset::<i32>()
            .shape([2])
            .create(" data")?
            .write(&arr1(&[cgns_code, 0_i32]))?;

        let er = section.create_group("ElementRange")?;
        write_node_attrs(&er, "ElementRange", "IndexRange_t", "I8", 1)?;
        er.new_dataset::<i64>()
            .shape([2])
            .create(" data")?
            .write(&arr1(&[r_start, r_end]))?;

        match et {
            ElementType::PGON => {
                let mut conn: Vec<i64> = Vec::new();
                let mut offsets: Vec<i64> = vec![0];
                for el in block.iter(mesh.coords()) {
                    for &node in el.connectivity() {
                        conn.push(node as i64 + 1); // CGNS is 1-based
                    }
                    offsets.push(conn.len() as i64);
                }
                write_conn_and_offset(&section, &conn, &offsets)?;
            }
            ElementType::PHED => {
                let mut conn: Vec<i64> = Vec::new();
                let mut offsets: Vec<i64> = vec![0];
                for el in block.iter(mesh.coords()) {
                    for face in el.connectivity().split(|&x| x == usize::MAX) {
                        if face.is_empty() {
                            continue; // trailing separator
                        }
                        let mut key = face.to_vec();
                        key.sort_unstable();
                        let (idx, canon) = face_map.get(&key).ok_or_else(|| {
                            MefikitIOError::InvalidMesh(
                                "PHED face is not present in the PGON block".to_string(),
                            )
                        })?;
                        conn.push(face_orientation(canon, face) * idx);
                    }
                    offsets.push(conn.len() as i64);
                }
                write_conn_and_offset(&section, &conn, &offsets)?;
            }
            _ => {
                let mut conn: Vec<i64> = Vec::new();
                for el in block.iter(mesh.coords()) {
                    for &node in el.connectivity() {
                        conn.push(node as i64 + 1); // CGNS is 1-based
                    }
                }
                let conn_node = section.create_group("ElementConnectivity")?;
                write_node_attrs(&conn_node, "ElementConnectivity", "DataArray_t", "I8", 1)?;
                conn_node
                    .new_dataset::<i64>()
                    .shape([conn.len()])
                    .create(" data")?
                    .write(&Array1::from(conn))?;
            }
        }
    }
    Ok(())
}

// Writes cell-centered fields as `FlowSolution_t` nodes, one per topological
// dimension. The solution is `GridLocation = CellCenter`; a `PointList` is only
// emitted for mixed-dimension meshes (where one dimension's cells are a subset
// of the zone). Each field becomes an `R8` `DataArray_t` shaped [n].
//
// CGNS requires the `DataArray_t` of a `CellCenter` solution to be one
// dimensional (its length is the number of selected cells), so multi-component
// fields cannot be represented and are skipped.
fn write_fields(zone: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let mut fields: Vec<(String, Dimension)> = mesh
        .fields()
        .filter(|(_, field)| ndarray::Dimension::ndim(&field.dim()) == 0)
        .filter_map(|(name, field)| field.dimension().map(|dim| (name, dim)))
        .collect();
    fields.sort();

    if fields.is_empty() {
        return Ok(());
    }

    let dimensions: BTreeSet<Dimension> = mesh.element_types().map(|et| et.dimension()).collect();
    let mixed = dimensions.len() > 1;
    let layout = BlockLayout::new(mesh);

    let field_dimensions: BTreeSet<Dimension> = fields.iter().map(|(_, dim)| *dim).collect();
    for dim in field_dimensions {
        let node_name = format!("FlowSolution_{}", u8::from(dim));
        let solution = zone.create_group(&node_name)?;
        write_node_attrs(&solution, &node_name, "FlowSolution_t", "MT", 1)?;

        let grid_location = solution.create_group("GridLocation")?;
        write_node_attrs(&grid_location, "GridLocation", "GridLocation_t", "C1", 1)?;
        write_c1_data(&grid_location, "CellCenter")?;

        if mixed {
            write_point_list(&solution, &global_ids_of_dim(&layout, dim))?;
        }

        for (name, field_dim) in &fields {
            if *field_dim != dim {
                continue;
            }
            let flat = flatten_field_for_dim(mesh, name, dim).ok_or_else(|| {
                MefikitIOError::Encode(format!("field {name:?} cannot be flattened at {dim:?}"))
            })?;
            let data_array = solution.create_group(name)?;
            write_node_attrs(&data_array, name, "DataArray_t", "R8", 1)?;
            data_array
                .new_dataset::<f64>()
                .shape(flat.shape())
                .create(" data")?
                .write(&flat)?;
        }
    }
    Ok(())
}

// Writes element groups as cell-centered `ZoneSubRegion_t` nodes. Each group
// becomes a node named after the group carrying its cells as a 1-based
// PointList; `reg_dim` is the highest topological dimension present in the group.
fn write_groups(zone: &Group, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let mut group_names = mesh.group_names();
    group_names.sort();

    if group_names.is_empty() {
        return Ok(());
    }

    let layout = BlockLayout::new(mesh);
    for name in group_names {
        if name.is_empty() || name.contains('/') {
            return Err(MefikitIOError::Encode(format!(
                "group name {name:?} is not a valid CGNS node name"
            )));
        }

        let subregion = zone.create_group(&name)?;
        write_node_attrs(&subregion, &name, "ZoneSubRegion_t", "I4", 1)?;

        let element_ids = mesh.group_elements(&name);
        let reg_dim = element_ids
            .iter_blocks()
            .map(|(element_type, _)| u8::from(element_type.dimension()))
            .max()
            .unwrap_or(u8::from(
                mesh.topological_dimension().unwrap_or(Dimension::D3),
            ));
        subregion
            .new_dataset::<i32>()
            .shape([1])
            .create(" data")?
            .write(&arr1(&[reg_dim as i32]))?;

        let globals: Vec<usize> = layout
            .group_global_ids(mesh, &name)
            .into_iter()
            .map(|global| global as usize)
            .collect();
        write_point_list(&subregion, &globals)?;

        let grid_location = subregion.create_group("GridLocation")?;
        write_node_attrs(&grid_location, "GridLocation", "GridLocation_t", "C1", 1)?;
        write_c1_data(&grid_location, "CellCenter")?;
    }
    Ok(())
}

// ── entry point ───────────────────────────────────────────────────────────────

// Writes `mesh` as a CGNS/HDF5 file. One section per element block: regular
// types become fixed-stride `Elements_t` sections, PGON becomes NGON_n and PHED
// becomes NFACE_n (CPEX0031, with `ElementStartOffset`). Family_t / ZoneBC_t are
// intentionally not written (mirrors the reader's coverage).
pub fn write_cgns(path: &Path, mesh: UMeshView) -> Result<(), MefikitIOError> {
    let file = File::create(path)?;

    // Root group attributes.
    write_root_attrs(&file)?;

    // " format" — 15 bytes null-padded.
    {
        let mut fmt = [0i8; 15];
        for (i, &b) in b"IEEE_LITTLE_32".iter().enumerate() {
            fmt[i] = b as i8;
        }
        file.new_dataset::<i8>()
            .shape([15])
            .create(" format")?
            .write(&Array1::from(fmt.to_vec()))?;
    }

    // " hdf5version" — 33 bytes null-padded.
    {
        let mut ver = [0i8; 33];
        for (i, &b) in b"HDF5 Version 1.10.6".iter().enumerate() {
            ver[i] = b as i8;
        }
        file.new_dataset::<i8>()
            .shape([33])
            .create(" hdf5version")?
            .write(&Array1::from(ver.to_vec()))?;
    }

    // CGNSLibraryVersion_t group.
    write_version(&file)?;

    // `cell_dim` is the topological dimension of the volume elements (2 or 3);
    // `phys_dim` is the number of coordinate arrays in GridCoordinates_t. Both
    // are required by vtkCGNSReader and become the two i32 values in the Base
    // group's " data" dataset.
    let top_dim = mesh.topological_dimension().unwrap_or(Dimension::D3);
    let cell_dim = u8::from(top_dim) as i32;
    let phys_dim = mesh.space_dimension() as i32;
    let base = write_base(&file, cell_dim, phys_dim)?;

    // Zone size array is [n_vertices, n_cells, 0] where n_cells counts volume
    // elements only (boundary patches excluded) and the trailing 0 is the
    // "boundary vertex count" (0 = unspecified).
    let n_cells = mesh.num_elements_of_dim(top_dim);
    let n_vertices = mesh.coords().nrows();
    let zone = write_zone(&base, n_vertices, n_cells)?;

    write_coords(&zone, &mesh)?;
    write_elements(&zone, &mesh)?;
    write_fields(&zone, &mesh)?;
    write_groups(&zone, &mesh)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::hdf_utils::hdf5_test_guard;
    use crate::mesh::FieldBase;
    use crate::mesh_examples as me;
    use std::process::Command;

    fn fixture() -> std::path::PathBuf {
        std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/particles_example.cgns"
        ))
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(name)
    }

    // Runs cgnscheck if it is installed. Returns None if the binary is missing
    // (so the test degrades to a no-op instead of failing on machines without
    // the CGNS tools), otherwise (exit code, combined stdout+stderr).
    fn cgnscheck(path: &std::path::Path) -> Option<(i32, String)> {
        let out = Command::new("cgnscheck")
            .arg("-v")
            .arg(path)
            .output()
            .ok()?;
        let code = out.status.code().unwrap_or(-1);
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        Some((code, text))
    }

    #[test]
    fn read_particles_has_pgon_and_phed() {
        let _guard = hdf5_test_guard();
        let mesh = read(&fixture()).unwrap();

        let n = mesh.coords().nrows();
        assert!(n > 0, "mesh must have coordinates");

        let pgon = mesh.block(ElementType::PGON).expect("PGON block present");
        let phed = mesh.block(ElementType::PHED).expect("PHED block present");
        assert!(pgon.len() > 0, "PGON block must be non-empty");
        assert!(phed.len() > 0, "PHED block must be non-empty");

        // PGON connectivity is stored 0-based, so every node index is < n.
        let max_pgon = pgon
            .iter(mesh.coords())
            .flat_map(|e| e.connectivity().to_vec())
            .max()
            .unwrap();
        assert!(max_pgon < n, "PGON must be 0-based (max {max_pgon} < {n})");

        // PHED cells are face-separated node lists ending with usize::MAX.
        let first = phed.iter(mesh.coords()).next().unwrap();
        let conn = first.connectivity();
        assert!(
            conn.contains(&usize::MAX),
            "PHED connectivity must contain usize::MAX face separators"
        );
        assert_eq!(
            *conn.last().unwrap(),
            usize::MAX,
            "PHED cell connectivity must end with a face separator"
        );
    }

    #[test]
    fn write_roundtrip_reread_matches() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_roundtrip_reread.cgns");
        let _ = std::fs::remove_file(&dst);

        let orig = read(&fixture()).unwrap();
        write_cgns(&dst, orig.view()).unwrap();
        let back = read(&dst).unwrap();

        assert_eq!(orig.coords().nrows(), back.coords().nrows());
        assert_eq!(orig.space_dimension(), back.space_dimension());
        assert_eq!(
            back.block(ElementType::PGON).unwrap().len(),
            orig.block(ElementType::PGON).unwrap().len()
        );
        assert_eq!(
            back.block(ElementType::PHED).unwrap().len(),
            orig.block(ElementType::PHED).unwrap().len()
        );

        // The first PHED cell's face-node structure survives the round-trip.
        let a = orig
            .block(ElementType::PHED)
            .unwrap()
            .iter(orig.coords())
            .next()
            .unwrap()
            .connectivity()
            .to_vec();
        let b = back
            .block(ElementType::PHED)
            .unwrap()
            .iter(back.coords())
            .next()
            .unwrap()
            .connectivity()
            .to_vec();
        assert_eq!(
            a, b,
            "first PHED cell connectivity must match after round-trip"
        );

        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn write_roundtrip_passes_cgnscheck() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_roundtrip_check.cgns");
        let _ = std::fs::remove_file(&dst);

        let mesh = read(&fixture()).unwrap();
        write_cgns(&dst, mesh.view()).unwrap();

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping HDF/CGNS compliance check"),
            Some((code, text)) => {
                assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}");
                assert!(
                    !text.to_lowercase().contains("error"),
                    "cgnscheck reported errors:\n{text}"
                );
            }
        }
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn write_regular_hex_passes_cgnscheck() {
        let _guard = hdf5_test_guard();
        // Hand-built unit-cube HEX8. CGNS HEX_8 node order: bottom quad CCW then
        // top quad CCW.
        let coords = ndarray::arr2(&[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ])
        .into_shared();
        let mut mesh = UMesh::new(coords);
        mesh.add_element(ElementType::HEX8, &[0, 1, 2, 3, 4, 5, 6, 7], None);

        let dst = tmp("mefikit_cgns_regular_hex.cgns");
        let _ = std::fs::remove_file(&dst);
        write_cgns(&dst, mesh.view()).unwrap();

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping HDF/CGNS compliance check"),
            Some((code, text)) => {
                assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}");
                assert!(
                    !text.to_lowercase().contains("error"),
                    "cgnscheck reported errors:\n{text}"
                );
            }
        }
        let _ = std::fs::remove_file(&dst);
    }

    fn field_blocks<F>(
        mesh: &UMesh,
        mut value: F,
    ) -> FieldBase<ndarray::OwnedArcRepr<f64>, ndarray::IxDyn>
    where
        F: FnMut(Dimension, usize) -> f64,
    {
        let mut blocks = std::collections::BTreeMap::new();
        for (element_type, block) in mesh.blocks() {
            let n = block.len();
            let values: Vec<f64> = (0..n).map(|i| value(element_type.dimension(), i)).collect();
            blocks.insert(
                *element_type,
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), values)
                    .unwrap()
                    .into_shared(),
            );
        }
        FieldBase(blocks)
    }

    #[test]
    fn roundtrip_cgns_fields_single_dim() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_fields_single.cgns");
        let _ = std::fs::remove_file(&dst);

        let mut mesh = me::make_imesh_2d(2);
        mesh.update_field("pressure", field_blocks(&mesh, |_, i| i as f64 + 0.5));

        write_cgns(&dst, mesh.view()).unwrap();
        let back = read(&dst).unwrap();

        let field = back.field("pressure", None).unwrap();
        for (element_type, block) in mesh.blocks() {
            for i in 0..block.len() {
                assert_eq!(field.0[element_type][[i]], i as f64 + 0.5);
            }
        }

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping compliance check"),
            Some((code, text)) => assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}"),
        }
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn roundtrip_cgns_skips_vector_fields() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_vector.cgns");
        let _ = std::fs::remove_file(&dst);

        // CGNS cell-centered solutions must be scalar, so a 3-component field is
        // skipped on write and does not round-trip.
        let mut mesh = me::make_imesh_2d(2);
        let mut blocks = std::collections::BTreeMap::new();
        for (element_type, block) in mesh.blocks() {
            let n = block.len();
            let values: Vec<f64> = (0..n * 3).map(|i| i as f64).collect();
            blocks.insert(
                *element_type,
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n, 3]), values)
                    .unwrap()
                    .into_shared(),
            );
        }
        mesh.update_field("velocity", FieldBase(blocks));

        write_cgns(&dst, mesh.view()).unwrap();
        let back = read(&dst).unwrap();
        assert!(back.field("velocity", None).is_none());

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping compliance check"),
            Some((code, text)) => assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}"),
        }
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn roundtrip_cgns_fields_mixed_dim() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_fields_mixed.cgns");
        let _ = std::fs::remove_file(&dst);

        // `make_mesh_2d_multi` mixes SEG2 (D1) with QUAD4/PGON (D2), so the D2
        // solution must carry a PointList selecting only the D2 cells.
        let mut mesh = me::make_mesh_2d_multi();
        let mut blocks = std::collections::BTreeMap::new();
        for (element_type, block) in mesh.blocks() {
            if element_type.dimension() != Dimension::D2 {
                continue;
            }
            let n = block.len();
            let values: Vec<f64> = (0..n).map(|i| i as f64 + 1.0).collect();
            blocks.insert(
                *element_type,
                ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[n]), values)
                    .unwrap()
                    .into_shared(),
            );
        }
        mesh.update_field("pressure", FieldBase(blocks));

        write_cgns(&dst, mesh.view()).unwrap();
        let back = read(&dst).unwrap();

        let field = back.field("pressure", Some(Dimension::D2)).unwrap();
        for (element_type, block) in mesh.blocks() {
            if element_type.dimension() != Dimension::D2 {
                continue;
            }
            for i in 0..block.len() {
                assert_eq!(field.0[element_type][[i]], i as f64 + 1.0);
            }
        }

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping compliance check"),
            Some((code, text)) => assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}"),
        }
        let _ = std::fs::remove_file(&dst);
    }

    #[test]
    fn roundtrip_cgns_groups() {
        let _guard = hdf5_test_guard();
        let dst = tmp("mefikit_cgns_groups.cgns");
        let _ = std::fs::remove_file(&dst);

        let mut mesh = me::make_mesh_2d_multi();
        let mut element_ids = crate::mesh::ElementIds::new();
        for (element_type, block) in mesh.blocks() {
            let indices: Vec<usize> = (0..block.len()).step_by(2).collect();
            if !indices.is_empty() {
                element_ids.add_block(*element_type, indices);
            }
        }
        mesh.add_to_group("left", &element_ids);

        write_cgns(&dst, mesh.view()).unwrap();
        let back = read(&dst).unwrap();

        assert_eq!(mesh.group_names(), back.group_names());
        assert_eq!(
            mesh.group_elements("left").iter().count(),
            back.group_elements("left").iter().count()
        );
        for id in mesh.group_elements("left").iter() {
            assert!(back.in_group(id, "left"));
        }

        match cgnscheck(&dst) {
            None => eprintln!("cgnscheck not installed — skipping compliance check"),
            Some((code, text)) => assert_eq!(code, 0, "cgnscheck failed (exit {code}):\n{text}"),
        }
        let _ = std::fs::remove_file(&dst);
    }

    // #[test]
    // fn strpad_nullterm_vs_nullpad_produce_identical_bytes() {
    //     use hdf5_metno::types::FixedAscii;

    //     let _guard = hdf5_test_guard();
    //     let path = tmp("mefikit_strpad_compare.h5");
    //     let _ = std::fs::remove_file(&path);
    //     let file = File::create(&path).unwrap();

    //     // ---- group written with unsafe C API: H5T_STR_NULLTERM ----
    //     let grp_nt = file.create_group("nullterm").unwrap();
    //     {
    //         use hdf5_metno_sys::h5a::{H5Aclose, H5Acreate2, H5Awrite};
    //         use hdf5_metno_sys::h5p::H5P_DEFAULT;
    //         use hdf5_metno_sys::h5s::{H5S_class_t, H5Sclose, H5Screate};
    //         use hdf5_metno_sys::h5t::{
    //             H5T_C_S1, H5T_cset_t, H5T_str_t, H5Tclose, H5Tcopy, H5Tset_cset, H5Tset_size,
    //             H5Tset_strpad,
    //         };
    //         use std::ffi::CString;

    //         unsafe {
    //             let tid = H5Tcopy(*H5T_C_S1);
    //             H5Tset_size(tid, 16);
    //             H5Tset_strpad(tid, H5T_str_t::H5T_STR_NULLTERM);
    //             H5Tset_cset(tid, H5T_cset_t::H5T_CSET_ASCII);
    //             let sid = H5Screate(H5S_class_t::H5S_SCALAR);
    //             let name = CString::new("attr").unwrap();
    //             let aid = H5Acreate2(
    //                 grp_nt.id(),
    //                 name.as_ptr(),
    //                 tid,
    //                 sid,
    //                 H5P_DEFAULT,
    //                 H5P_DEFAULT,
    //             );
    //             let mut buf = [0u8; 16];
    //             buf[..4].copy_from_slice(b"test");
    //             H5Awrite(aid, tid, buf.as_ptr() as *const _);
    //             H5Aclose(aid);
    //             H5Sclose(sid);
    //             H5Tclose(tid);
    //         }
    //     }

    //     // ---- group written with safe hdf5-metno API: H5T_STR_NULLPAD ----
    //     let grp_np = file.create_group("nullpad").unwrap();
    //     grp_np
    //         .new_attr::<FixedAscii<16>>()
    //         .shape(())
    //         .create("attr")
    //         .unwrap()
    //         .write_scalar(&FixedAscii::<16>::from_ascii("test").unwrap())
    //         .unwrap();

    //     // ---- read both back and compare raw bytes ----
    //     let raw_nt: FixedAscii<16> = grp_nt.attr("attr").unwrap().read_scalar().unwrap();
    //     let raw_np: FixedAscii<16> = grp_np.attr("attr").unwrap().read_scalar().unwrap();
    //     assert_eq!(raw_nt.as_str(), raw_np.as_str());
    //     assert_eq!(raw_nt.as_bytes(), raw_np.as_bytes());

    //     let _ = std::fs::remove_file(&path);
    // }
}
