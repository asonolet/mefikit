use hdf5_metno::{File, Group};
use crate::mesh::{UMesh, ElementType};
use std::path::Path;
use hdf5_metno::types::FixedAscii;
use super::error::MefikitIOError;
use super::hdf_utils::{
    read_index_array,
    read_string_data,
};

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
            other  => Err(MefikitIOError::Parse(format!("Unsupported dimension combo {other:?}"))),
        }
    }
}

struct CgnsElementInfo {
    element_type: ElementType,
    nodes_per_cell: Option<usize>,  // None for poly
}

fn cgns_element_info(code: i32) -> Option<CgnsElementInfo> {
    match code {
        2  => Some(CgnsElementInfo { element_type: ElementType::VERTEX, nodes_per_cell: Some(1) }),
        3  => Some(CgnsElementInfo { element_type: ElementType::SEG2,   nodes_per_cell: Some(2) }),
        5  => Some(CgnsElementInfo { element_type: ElementType::TRI3,   nodes_per_cell: Some(3) }),
        7  => Some(CgnsElementInfo { element_type: ElementType::QUAD4,  nodes_per_cell: Some(4) }),
        10 => Some(CgnsElementInfo { element_type: ElementType::TET4,   nodes_per_cell: Some(4) }),
        17 => Some(CgnsElementInfo { element_type: ElementType::HEX8,   nodes_per_cell: Some(8) }),
        22 => Some(CgnsElementInfo { element_type: ElementType::PGON,   nodes_per_cell: None    }),
        23 => Some(CgnsElementInfo { element_type: ElementType::PHED,   nodes_per_cell: None    }),
        other => {
            eprintln!("warning: unsupported CGNS element type {other}, section skipped");
            None
        }
    }
}

fn nodes_per_cgns_code(code: i32) -> Option<usize> {
    match code {
        2  => Some(1),
        3  => Some(2),
        5  => Some(3),
        7  => Some(4),
        10 => Some(4),
        17 => Some(8),
        _  => None, // poly types have variable stride — handled separately
    }
}

pub fn cgns_label(group: &Group) -> Result<String, MefikitIOError> {
    let attr = group.attr(" label").or_else(|_| group.attr("label"))?;
    let label: String = attr
        .as_reader()
        .read_scalar::<FixedAscii<64>>()?
        .to_string();
    Ok(label.trim().trim_matches('\0').to_string())
}

fn find_first_child_with_label(
    group: &Group,
    label: &str,
) -> Result<Group, MefikitIOError > {
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
        if lbl == label {
            return Ok(child);
        }
    }
    Err(MefikitIOError::Parse(format!("no child with label '{label}' in '{}'", group.name()).into()))
}

fn children_with_label(
    group: &Group,
    label: &str,
) -> Result<Vec<Group>, MefikitIOError> {
    let mut out = Vec::new();
    for name in group.member_names()? {
        let Ok(child) = group.group(&name) else { continue };
        let Ok(lbl) = cgns_label(&child) else { continue };
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
                ds.as_reader()
                    .read_1d::<f64>()?
                    .to_vec()
            };

            Ok(values)
        })
        .collect::<Result<_, MefikitIOError>>()?;

    let n = columns[0].len();
    let mut coords = ndarray::Array2::<f64>::zeros((n, phys_dim));
    for (col_idx, col_data) in columns.iter().enumerate() {
        coords.column_mut(col_idx)
            .iter_mut()
            .zip(col_data)
            .for_each(|(dst, &src)| *dst = src);
    }
    
    Ok(coords.into_shared())
}

// inline to speed up the check since we'll be doing it for every element
#[inline]
fn is_ngon(cgns_code: i32) -> bool {
    cgns_code == 22
}

// Return true if element describes a cell-face section (NFACE_n)
#[inline]
fn is_nfaces(cgns_code: i32) -> bool {
    cgns_code == 23
}

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
    let mut pgon_conn:    Option<Vec<i64>> = None;
    let mut phed_offsets: Option<Vec<i64>> = None;
    let mut phed_conn:    Option<Vec<i64>> = None;

    for element in &el_group {
        let type_info = cgns_element_info(read_element_type(element)?);
        if let Some(info) = type_info {
            match info.element_type {
                ElementType::PGON => {
                    let conn    = read_element_connectivity(element)?;
                    let offsets = read_element_offsets(element)?.ok_or_else(|| {
                        MefikitIOError::MalformedFile("PGON missing ElementStartOffset".to_string())
                    })?;
                    let range = read_element_range(element)?;
                    let n_cells = (range[1] - range[0] + 1) as usize;
                    for i in 0..n_cells {
                        let start = offsets[i] as usize;
                        let end   = offsets[i + 1] as usize;
                        let nodes: Vec<usize> = conn[start..end]
                            .iter().map(|&v| v as usize).collect();
                        mesh.add_element(ElementType::PGON, &nodes, None, None);
                    }
                    pgon_offsets = Some(offsets);
                    pgon_conn    = Some(conn);
                }
                ElementType::PHED => {
                    phed_conn    = Some(read_phed_connectivity(element)?);
                    phed_offsets = Some(read_element_offsets(element)?.ok_or_else(|| {
                        MefikitIOError::MalformedFile("PHED missing ElementStartOffset".to_string())
                    })?);
                }
                other => {
                    let range = read_element_range(element)?;
                    let conn  = read_element_connectivity(element)?;
                    let n_cells = (range[1] - range[0] + 1) as usize;
                    let nodes_per_cell = info.nodes_per_cell.unwrap();
                    for i in 0..n_cells {
                        let start = i * nodes_per_cell;
                        let end   = start + nodes_per_cell;
                        let cell: Vec<usize> = conn[start..end]
                            .iter().map(|&v| v as usize).collect();
                        mesh.add_element(info.element_type, &cell, None, None);
                    }
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
            let end   = p_off[i + 1] as usize;

            let mut cell_nodes: Vec<usize> = Vec::new();

            for &face_ref in &p_conn[start..end] {
                let _reversed  = face_ref < 0;
                let face_index = (face_ref.unsigned_abs() as usize) - 1;

                let node_start = f_off[face_index] as usize;
                let node_end   = f_off[face_index + 1] as usize;

                for &node_id in &f_conn[node_start..node_end] {
                    let coord_index = (node_id as usize) - 1;
                    cell_nodes.push(coord_index);
                }
            }

            mesh.add_element(ElementType::PHED, &cell_nodes, None, None);
        }
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
        return Err(MefikitIOError::Parse(format!("unsupported zone type: {z_type}")));
    }

   let coords = read_coordinates(&zone, cgns_dim.phys_dim)?;
   let mut mesh = UMesh::new(coords);

   read_elements(&mut mesh, &zone)?;

   // Future implementation: read families and boundary conditions
    Ok(mesh)
}