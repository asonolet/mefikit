use super::error::MefikitIOError;
use crate::mesh::ConnectivityView;
use crate::mesh::ElementBlock;
use crate::mesh::ElementBlockView;
use crate::mesh::ElementType;
use crate::mesh::IntoElementBlockEntry;
use crate::mesh::Regularity;
use crate::mesh::UMesh;
use crate::mesh::UMeshView;

use hdf5_metno::types::FixedAscii;
use hdf5_metno::types::VarLenAscii;
use hdf5_metno::{File, Group};
use ndarray::prelude::*;
use std::collections::BTreeMap;
use std::path::Path;

impl ElementType {
    pub fn med_name(self) -> &'static str {
        match self {
            ElementType::VERTEX => "PO1",
            ElementType::SEG2 => "SE2",
            ElementType::SEG3 => "SE3",

            ElementType::TRI3 => "TR3",
            ElementType::TRI6 => "TR6",
            ElementType::TRI7 => "TR7",

            ElementType::QUAD4 => "QU4",
            ElementType::QUAD8 => "QU8",
            ElementType::QUAD9 => "QU9",

            ElementType::TET4 => "TE4",
            ElementType::TET10 => "T10",

            ElementType::HEX8 => "HE8",
            // ElementType::HEX21     => "H20",
            // ElementType::Hexa27     => "H27",

            // ElementType::Pyramid5   => "PY5",
            // ElementType::Pyramid13  => "P13",

            // ElementType::Wedge6     => "PE6",
            // ElementType::Wedge15    => "P15",
            ElementType::PGON => "POG",
            ElementType::PHED => "POE",
            _ => todo!(),
        }
    }
}

pub fn write(path: impl AsRef<std::path::Path>, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    let file = hdf5_metno::File::create(path)?;

    write_general_info(&file)?;
    write_mesh(&file, mesh)?;
    write_fields(&file, mesh)?;

    Ok(())
}

fn write_general_info(file: &File) -> hdf5_metno::Result<()> {
    let info = file.create_group("INFOS_GENERALES")?;

    write_scalar_attr(&info, "MAJ", 3i64)?;
    write_scalar_attr(&info, "MIN", 0i64)?;
    write_scalar_attr(&info, "REL", 0i64)?;

    Ok(())
}

fn write_scalar_attr<T: hdf5_metno::H5Type>(
    obj: &hdf5_metno::Location,
    name: &str,
    value: T,
) -> hdf5_metno::Result<()> {
    obj.new_attr::<T>().create(name)?.write_scalar(&value)
}

fn write_mesh(file: &File, mesh: &UMeshView) -> hdf5_metno::Result<()> {
    let ensemble = file.create_group("ENS_MAA")?;
    let med_mesh = ensemble.create_group("mesh")?;

    let dim = mesh.coords().shape()[1];

    write_scalar_attr(&med_mesh, "DIM", dim as i64)?;
    write_scalar_attr(&med_mesh, "ESP", dim as i64)?;
    write_scalar_attr(&med_mesh, "REP", 0i64)?;

    // Empty for now, as in the Python reference when no units are supplied.
    write_fixed_bytes_attr::<1>(&med_mesh, "UNT", b"")?;
    write_fixed_bytes_attr::<1>(&med_mesh, "UNI", b"")?;

    write_scalar_attr(&med_mesh, "SRT", 1i64)?;

    // "X", "Y", "Z", each occupying 16 characters.
    let names = ["X", "Y", "Z"];
    let mut nom = Vec::with_capacity(dim * 16);

    for name in names.iter().take(dim) {
        let bytes = name.as_bytes();
        nom.extend_from_slice(bytes);
        nom.resize(nom.len() + 16 - bytes.len(), b' ');
    }

    write_fixed_bytes_attr::<48>(&med_mesh, "NOM", &nom)?;

    write_fixed_bytes_attr::<25>(&med_mesh, "DES", b"Mesh created with mefikit")?;

    write_scalar_attr(&med_mesh, "TYP", 0i64)?;

    let timestep = med_mesh.create_group("-0000000000000000001-0000000000000000001")?;

    write_scalar_attr(&timestep, "CGT", 1i64)?;
    write_scalar_attr(&timestep, "NDT", -1i64)?;
    write_scalar_attr(&timestep, "NOR", -1i64)?;
    write_scalar_attr(&timestep, "PDT", -1.0f64)?;

    // Explicitly create CHA, even though it is empty for now.

    write_nodes(&timestep, &mesh.coords())?;

    let mai = timestep.create_group("MAI")?;
    write_scalar_attr(&mai, "CGT", 1i64)?;

    for (et, block) in mesh.blocks() {
        write_block(&mai, et, &block.view())?;
    }

    write_families(file, mesh)?;

    Ok(())
}

fn write_nodes(timestep: &Group, coords: &ArrayView2<f64>) -> hdf5_metno::Result<()> {
    let noe = timestep.create_group("NOE")?;

    write_scalar_attr(&noe, "CGT", 1i64)?;
    write_scalar_attr(&noe, "CGS", 1i64)?;

    write_fixed_bytes_attr::<23>(&noe, "PFL", b"MED_NO_PROFILE_INTERNAL")?;

    let coo_data: Vec<f64> = {
        let (n, dim) = coords.dim();
        let mut out = Vec::with_capacity(n * dim);

        for d in 0..dim {
            for n in 0..n {
                out.push(coords[[n, d]]);
            }
        }

        out
    };

    let coo = noe
        .new_dataset_builder()
        .with_data(&coo_data)
        .create("COO")?;

    write_scalar_attr(&coo, "CGT", 1i64)?;
    write_scalar_attr(&coo, "NBR", coords.nrows() as i64)?;

    Ok(())
}

impl ElementType {
    fn med_permutation(self) -> Option<&'static [usize]> {
        match self {
            // VTK→MED: swap nodes 2↔3 (self-inverse, used on both read and write paths).
            ElementType::TET4 => Some(&[0, 1, 3, 2]),
            ElementType::TET10 => Some(&[0, 1, 3, 2, 4, 8, 7, 6, 5, 9]),
            // VTK→MED: MED node 0 is top-left-front, VTK node 0 is bottom-left-front.
            // Maps VTK bottom(0,1,2,3) → MED bottom(4,5,6,7), VTK top(4,5,6,7) → MED top(0,1,2,3).
            // Self-inverse.
            ElementType::HEX8 => Some(&[4, 5, 6, 7, 0, 1, 2, 3]),
            _ => None,
        }
    }
}

fn reorder_connectivity(conn: &ArrayView2<usize>, permutation: Option<&[usize]>) -> Array2<u64> {
    let Some(p) = permutation else {
        return conn.mapv(|x| x as u64);
    };

    let mut out = Array2::<u64>::zeros((conn.nrows(), p.len()));

    for (new_j, &old_j) in p.iter().enumerate() {
        out.column_mut(new_j)
            .assign(&conn.column(old_j).mapv(|x| x as u64));
    }

    out
}

fn write_fam_dataset(
    parent: &Group,
    families: ndarray::ArrayView1<'_, usize>,
) -> hdf5_metno::Result<()> {
    if families.iter().all(|&f| f == 0) {
        return Ok(());
    }
    let fam_data: Vec<i64> = families.iter().map(|&f| f as i64).collect();
    let n = fam_data.len() as i64;
    let ds = parent
        .new_dataset_builder()
        .with_data(&fam_data)
        .create("FAM")?;
    write_scalar_attr(&ds, "CGT", 1i64)?;
    write_scalar_attr(&ds, "NBR", n)?;
    Ok(())
}

impl ElementType {
    /// Numeric MED geometry-type code written as the "GEO" attribute on every
    /// cell group. Some MED tools reject a cell group without it. Codes follow
    /// MED's convention: dim*100 + n_nodes (point = 1); polygons/polyhedra use
    /// 400/500. Same values as the reference meshio `med_geo_code` map.
    fn med_geo_code(self) -> i64 {
        match self {
            ElementType::VERTEX => 1,
            ElementType::SEG2 => 102,
            ElementType::SEG3 => 103,
            ElementType::TRI3 => 203,
            ElementType::TRI6 => 206,
            ElementType::TRI7 => 207,
            ElementType::QUAD4 => 204,
            ElementType::QUAD8 => 208,
            ElementType::QUAD9 => 209,
            ElementType::TET4 => 304,
            ElementType::TET10 => 310,
            ElementType::HEX8 => 308,
            ElementType::PGON => 400,
            ElementType::PHED => 500,
            _ => todo!(),
        }
    }
}

fn write_regular(
    mai: &Group,
    element_type: ElementType,
    regular: &ElementBlockView,
) -> hdf5_metno::Result<()> {
    let conn = match &regular.connectivity {
        ConnectivityView::Regular(pc) => pc,
        _ => panic!("A Regular block must contain a regular connectivity."),
    };
    let med_type = element_type.med_name();

    let group = mai.create_group(med_type)?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;
    write_scalar_attr(&group, "GEO", element_type.med_geo_code())?;
    write_fixed_bytes_attr::<23>(&group, "PFL", b"MED_NO_PROFILE_INTERNAL")?;
    let med_conn = reorder_connectivity(conn, element_type.med_permutation());

    let mut flat = Vec::with_capacity(med_conn.len());

    // MED expects column-major ordering.
    for j in 0..med_conn.ncols() {
        for i in 0..med_conn.nrows() {
            flat.push(med_conn[[i, j]] + 1);
        }
    }

    let nod = group.new_dataset_builder().with_data(&flat).create("NOD")?;

    write_scalar_attr(&nod, "CGT", 1i64)?;
    write_scalar_attr(&nod, "NBR", med_conn.nrows() as i64)?;

    write_fam_dataset(&group, regular.families())?;

    Ok(())
}

fn write_families(file: &File, mesh: &UMeshView) -> hdf5_metno::Result<()> {
    let fas = file.create_group("FAS")?;
    let families = fas.create_group("mesh")?;

    let family_zero = families.create_group("FAMILLE_ZERO")?;
    write_scalar_attr(&family_zero, "NUM", 0i64)?;

    // Collect the reverse map: family_id → set of group names, across all blocks.
    let mut fam_to_names: BTreeMap<usize, BTreeMap<String, ()>> = BTreeMap::new();

    for (_et, block) in mesh.blocks() {
        for (group_name, family_ids) in block.groups() {
            for &fid in family_ids {
                if fid == 0 {
                    continue;
                }
                fam_to_names
                    .entry(fid)
                    .or_default()
                    .insert(group_name.clone(), ());
            }
        }
    }

    if fam_to_names.is_empty() {
        return Ok(());
    }

    let elem = families.create_group("ELEME")?;

    for (&fid, names_map) in &fam_to_names {
        let names: Vec<&str> = names_map.keys().map(|s| s.as_str()).collect();
        let gname = format!("FAM_{fid}_");
        let fam_group = elem.create_group(&gname)?;
        write_scalar_attr(&fam_group, "NUM", fid as i64)?;

        if names.is_empty() {
            continue;
        }

        let gro = fam_group.create_group("GRO")?;
        write_scalar_attr(&gro, "NBR", names.len() as i64)?;

        // NOM: (n_groups, 80) array of signed i8, space-padded (meshio/MED convention).
        let n_groups = names.len();
        let mut buf2d = Array2::<i8>::from_elem((n_groups, 80), 0x20);
        for (i, name) in names.iter().enumerate() {
            let name_bytes = name.as_bytes();
            let len = name_bytes.len().min(80);
            for (j, &b) in name_bytes[..len].iter().enumerate() {
                buf2d[[i, j]] = b as i8;
            }
        }

        let nom_ds = gro.new_dataset_builder().with_data(&buf2d).create("NOM")?;

        let _ = nom_ds;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Field I/O
// ---------------------------------------------------------------------------

/// MED fixed-width for component names (16 chars each).
const MED_NOM_LEN: usize = 16;

/// Write element-centered fields into the CHA group.
///
/// Layout per field:
/// ```text
/// CHA/<field_name>/
///   (attrs: MAI="mesh", TYP=6, NCO=<n_comp>, NOM=<16-char padded names>, UNI="", UNT="")
///   0000000000000000000100000000000000000001/      ← static step: NDT=1, NOR=1
///     MAI.<MED_TYPE>/
///       (attrs: GAU="", PFL="MED_NO_PROFILE_INTERNAL")
///       MED_NO_PROFILE_INTERNAL/
///         (attrs: NBR=<n_cells>, NGA=1, GAU="")
///         CO                                        ← f64, Fortran-order flattened
/// ```
fn write_fields(file: &File, mesh: &UMeshView) -> Result<(), MefikitIOError> {
    // Collect per-field-name: { ElementType → (ArrayViewD<f64>, n_cells) }.
    // We iterate blocks to discover all field names and their per-type data.
    let mut field_map: BTreeMap<String, BTreeMap<ElementType, (usize, Vec<f64>)>> = BTreeMap::new();

    for (et, block) in mesh.blocks() {
        for (name, arr) in &block.fields {
            let n_cells = block.len();
            // Flatten to contiguous f64 vec.
            let data: Vec<f64> = arr.as_slice().unwrap().to_vec();
            field_map
                .entry(name.clone())
                .or_default()
                .insert(*et, (n_cells, data));
        }
    }

    if field_map.is_empty() {
        return Ok(());
    }

    let cha = file.create_group("CHA")?;
    let step_key = "0000000000000000000100000000000000000001";

    for (field_name, type_map) in &field_map {
        // Infer n_components from the first type entry.
        let (_first_et, &(n_first_cells, ref first_data)) = type_map.iter().next().unwrap();
        // Infer n_components from shape: if data.len() / n_cells == 1 → scalar.
        let n_components = first_data.len().checked_div(n_first_cells).unwrap_or(1);

        let field_grp = cha.create_group(field_name)?;

        // Attributes on the field group.
        write_scalar_attr(&field_grp, "TYP", 6i64)?; // MED_FLOAT64
        write_scalar_attr(&field_grp, "NCO", n_components as i64)?;
        write_fixed_bytes_attr::<8>(&field_grp, "MAI", b"mesh")?;
        write_fixed_bytes_attr::<8>(&field_grp, "UNI", b"")?;
        write_fixed_bytes_attr::<8>(&field_grp, "UNT", b"")?;

        // Component names: V1, V2, ... or use the single name for scalars.
        let nom_bytes = build_field_nom(field_name, n_components);
        let nom_len = nom_bytes.len();
        // NOM is an attribute on the field group (16 chars per component).
        write_field_nom_attr(&field_grp, "NOM", &nom_bytes, nom_len)?;

        // Time-step group.
        let step_grp = field_grp.create_group(step_key)?;
        write_scalar_attr(&step_grp, "NDT", 1i64)?;
        write_scalar_attr(&step_grp, "NOR", 1i64)?;
        write_scalar_attr(&step_grp, "PDT", 0.0f64)?;
        write_scalar_attr(&step_grp, "RDT", -1i64)?;
        write_scalar_attr(&step_grp, "ROR", -1i64)?;

        // One MAI.<type> subgroup per element type.
        for (&et, &(n_cells, ref data)) in type_map {
            let med_type = et.med_name();
            let mai_type_name = format!("MAI.{med_type}");
            let mai_type_grp = step_grp.create_group(&mai_type_name)?;

            write_fixed_bytes_attr::<1>(&mai_type_grp, "GAU", b"")?;
            write_fixed_bytes_attr::<23>(&mai_type_grp, "PFL", b"MED_NO_PROFILE_INTERNAL")?;

            let prof_grp = mai_type_grp.create_group("MED_NO_PROFILE_INTERNAL")?;
            write_scalar_attr(&prof_grp, "NBR", n_cells as i64)?;
            write_scalar_attr(&prof_grp, "NGA", 1i64)?;
            write_fixed_bytes_attr::<1>(&prof_grp, "GAU", b"")?;

            // CO dataset: shape (n_cells, n_components) stored in Fortran (column-major) order.
            let co_data: Vec<f64> = if n_components == 1 {
                data.clone()
            } else {
                // data is (n_cells, n_components) in C order → transpose for Fortran order.
                let arr = Array2::from_shape_vec((n_cells, n_components), data.clone())?;
                let t = arr.reversed_axes();
                t.iter().copied().collect()
            };

            let co_ds = prof_grp
                .new_dataset_builder()
                .with_data(&co_data)
                .create("CO")?;
            write_scalar_attr(&co_ds, "CGT", 1i64)?;
        }
    }

    Ok(())
}

/// Build NOM bytes for a field. For scalar (n_comp=1) the name is field_name
/// padded to 16 chars. For multi-component, each component gets V1, V2, ... padded.
fn build_field_nom(field_name: &str, n_components: usize) -> Vec<u8> {
    if n_components == 1 {
        let mut buf = vec![b' '; MED_NOM_LEN];
        let name_bytes = field_name.as_bytes();
        let len = name_bytes.len().min(MED_NOM_LEN);
        buf[..len].copy_from_slice(&name_bytes[..len]);
        buf
    } else {
        let mut buf = Vec::with_capacity(n_components * MED_NOM_LEN);
        for c in 1..=n_components {
            let label = format!("V{c}");
            let label_bytes = label.as_bytes();
            let mut slot = vec![b' '; MED_NOM_LEN];
            let len = label_bytes.len().min(MED_NOM_LEN);
            slot[..len].copy_from_slice(&label_bytes[..len]);
            buf.extend_from_slice(&slot);
        }
        buf
    }
}

/// Write the field component-name attribute. MED stores it on the field group
/// (as `attrs["NOM"]`, a fixed-size char array of `16 * n_components`) — not as
/// a dataset. The hdf5-metno API requires a concrete `FixedAscii<N>` size at
/// compile time, but `n_components` is runtime, so we write a variable-length
/// ASCII string attribute instead, which MED readers accept.
fn write_field_nom_attr(
    grp: &Group,
    name: &str,
    nom_bytes: &[u8],
    _total_len: usize,
) -> Result<(), MefikitIOError> {
    let value = VarLenAscii::from_ascii(nom_bytes)
        .map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    grp.new_attr::<VarLenAscii>()
        .create(name)?
        .write_scalar(&value)?;

    Ok(())
}

#[allow(unused)]
fn write_bytes_attr(group: &Group, name: &str, value: &[u8]) -> hdf5_metno::Result<()> {
    let value =
        VarLenAscii::from_ascii(value).map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    group
        .new_attr::<VarLenAscii>()
        .create(name)?
        .write_scalar(&value)?;

    Ok(())
}

fn write_fixed_bytes_attr<const N: usize>(
    group: &Group,
    name: &str,
    value: &[u8],
) -> hdf5_metno::Result<()> {
    let value = FixedAscii::<N>::from_ascii(value)
        .map_err(|e| hdf5_metno::Error::Internal(e.to_string()))?;

    group
        .new_attr::<FixedAscii<N>>()
        .create(name)?
        .write_scalar(&value)
}

fn write_polygon(mai: &Group, poly: &ElementBlockView) -> hdf5_metno::Result<()> {
    let poly_conn = match &poly.connectivity {
        ConnectivityView::Poly(pc) => pc,
        _ => panic!("A PGON block must contain a poly connectivity."),
    };
    let group = mai.create_group("POG")?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;
    write_scalar_attr(&group, "GEO", 400i64)?;

    // MED NOD is 1-based.
    let nod: Vec<u64> = poly_conn.data.iter().map(|&x| x as u64 + 1).collect();

    let nod_ds = group.new_dataset_builder().with_data(&nod).create("NOD")?;

    write_scalar_attr(&nod_ds, "CGT", 1i64)?;
    write_scalar_attr(&nod_ds, "NBR", nod.len() as i64)?;

    // Mefikit:
    //
    // offset = [3, 7, 10]
    //
    // MED:
    //
    // INN = [1, 4, 8, 11]
    //
    let mut inn = Vec::with_capacity(poly_conn.offsets.len() + 1);

    inn.push(1u64);

    for &end in poly_conn.offsets.iter() {
        inn.push(end as u64 + 1);
    }

    let inn_ds = group.new_dataset_builder().with_data(&inn).create("INN")?;

    write_scalar_attr(&inn_ds, "CGT", 1i64)?;
    write_scalar_attr(&inn_ds, "NBR", inn.len() as i64)?;

    write_fam_dataset(&group, poly.families())?;

    Ok(())
}

fn write_polyhedron(mai: &Group, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    let poly_conn = match &block.connectivity {
        ConnectivityView::Poly(pc) => pc,
        _ => panic!("A PHED block must contain a poly connectivity."),
    };
    let group = mai.create_group("POE")?;

    write_scalar_attr(&group, "CGT", 1i64)?;
    write_scalar_attr(&group, "CGS", 1i64)?;
    write_scalar_attr(&group, "GEO", 500i64)?;
    write_fixed_bytes_attr::<23>(&group, "PFL", b"MED_NO_PROFILE_INTERNAL")?;

    // Mefikit data layout (per element):
    //   [face0_n0, face0_n1, ..., usize::MAX, face1_n0, ..., usize::MAX, ...]
    //
    // MED POE layout:
    //   NOD = [face0_n0, face0_n1, ..., face1_n0, ...]  (1-based, flat)
    //   INN = [1, end_of_face0+1, end_of_face1+1, ...]  (1-based cumulative-end, n_faces+1)
    //   IFN = [1, end_of_face0_of_poly0+1, ...]         (1-based cumulative-end, n_poly+1)

    let mut nod: Vec<u64> = Vec::new();
    let mut inn: Vec<u64> = vec![1]; // leading sentinel
    let mut ifn: Vec<u64> = vec![1]; // leading sentinel

    // let n_poly = poly_conn.offsets.len();

    for elem_conn in poly_conn {
        // Split element connectivity by usize::MAX to get faces.
        let mut face_count = 0;
        for run in elem_conn.split(|&v| v == usize::MAX) {
            if run.is_empty() {
                continue;
            }
            // Convert 0-based node IDs to 1-based for MED.
            nod.extend(run.iter().map(|&n| (n + 1) as u64));
            // INN tracks cumulative end position in NOD (1-based).
            inn.push(nod.len() as u64 + 1);
            face_count += 1;
        }
        // IFN tracks cumulative end position in faces (1-based).
        ifn.push(ifn.last().unwrap() + face_count as u64);
    }

    let nod_ds = group.new_dataset_builder().with_data(&nod).create("NOD")?;
    write_scalar_attr(&nod_ds, "CGT", 1i64)?;
    write_scalar_attr(&nod_ds, "NBR", nod.len() as i64)?;

    let inn_ds = group.new_dataset_builder().with_data(&inn).create("INN")?;
    write_scalar_attr(&inn_ds, "CGT", 1i64)?;
    write_scalar_attr(&inn_ds, "NBR", inn.len() as i64)?;

    let ifn_ds = group.new_dataset_builder().with_data(&ifn).create("IFN")?;
    write_scalar_attr(&ifn_ds, "CGT", 1i64)?;
    write_scalar_attr(&ifn_ds, "NBR", ifn.len() as i64)?;

    write_fam_dataset(&group, block.families())?;

    Ok(())
}

fn write_block(mai: &Group, et: &ElementType, block: &ElementBlockView) -> hdf5_metno::Result<()> {
    match et.regularity() {
        Regularity::Regular => {
            write_regular(mai, *et, block)?;
        }

        Regularity::Poly => match et {
            ElementType::PGON => {
                write_polygon(mai, block)?;
            }

            ElementType::PHED => {
                write_polyhedron(mai, block)?;
            }

            _ => {
                return Err(hdf5_metno::Error::Internal(format!(
                    "PolyConnectivity is not supported \
                                for {:?}",
                    et
                )));
            }
        },
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

fn med_name_to_element_type(name: &str) -> Option<ElementType> {
    match name {
        "PO1" => Some(ElementType::VERTEX),
        "SE2" => Some(ElementType::SEG2),
        "SE3" => Some(ElementType::SEG3),
        "TR3" => Some(ElementType::TRI3),
        "TR6" => Some(ElementType::TRI6),
        "TR7" => Some(ElementType::TRI7),
        "QU4" => Some(ElementType::QUAD4),
        "QU8" => Some(ElementType::QUAD8),
        "QU9" => Some(ElementType::QUAD9),
        "TE4" => Some(ElementType::TET4),
        "T10" => Some(ElementType::TET10),
        "HE8" => Some(ElementType::HEX8),
        "POG" | "POG2" => Some(ElementType::PGON),
        "POE" => Some(ElementType::PHED),
        _ => None,
    }
}

fn read_scalar_attr_i64(loc: &hdf5_metno::Location, name: &str) -> Result<i64, MefikitIOError> {
    let val = loc.attr(name)?.read_scalar::<i64>()?;
    Ok(val)
}

type Fas = (BTreeMap<i64, Vec<String>>, BTreeMap<i64, Vec<String>>);

/// Parse FAS/<mesh_name>/NOEUD and FAS/<mesh_name>/ELEME into
/// { family_id → [group_name, ...] }.
fn read_fas(fas_root: &Group, mesh_name: &str) -> Result<Fas, MefikitIOError> {
    let mesh_fas = match fas_root.group(mesh_name) {
        Ok(g) => g,
        Err(_) => return Ok((BTreeMap::new(), BTreeMap::new())),
    };

    let node_fams = read_fas_subgroup(&mesh_fas, "NOEUD");
    let elem_fams = read_fas_subgroup(&mesh_fas, "ELEME");

    Ok((node_fams, elem_fams))
}

fn read_fas_subgroup(parent: &Group, name: &str) -> BTreeMap<i64, Vec<String>> {
    let sub = match parent.group(name) {
        Ok(g) => g,
        Err(_) => return BTreeMap::new(),
    };

    let mut result = BTreeMap::new();

    for entry_name in sub.member_names().unwrap_or_default() {
        let fam_group = match sub.group(&entry_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let num = match fam_group.attr("NUM") {
            Ok(a) => a.read_scalar::<i64>().unwrap_or(0),
            Err(_) => continue,
        };

        let mut names = Vec::new();

        if let Ok(gro) = fam_group.group("GRO")
            && let Ok(nom_ds) = gro.dataset("NOM")
        {
            // NOM is an array of 80-byte char slots.
            let raw: Array2<i8> = nom_ds.read().unwrap_or_else(|_| Array2::zeros((0, 80)));
            for row in raw.rows() {
                let name_str: String = row
                    .iter()
                    .map(|&b| b as u8)
                    .take_while(|&b| b != 0 && b != b' ')
                    .map(|b| b as char)
                    .collect();
                if !name_str.is_empty() {
                    names.push(name_str);
                }
            }
        }

        result.insert(num, names);
    }

    result
}

fn read_nodal_data(timestep: &Group, dim: usize) -> Result<(usize, Array2<f64>), MefikitIOError> {
    let noe = timestep.group("NOE")?;
    let coo: Array1<f64> = noe.dataset("COO")?.read()?;
    let n_points = coo.len() / dim;
    // MED stores COO column-major: [x0,x1,...,xN, y0,y1,...,yN, ...]
    // Reshape as (dim, n_points) then transpose to get (n_points, dim).
    let raw = Array2::from_shape_vec((dim, n_points), coo.to_vec())?;
    let coords = raw.reversed_axes().as_standard_layout().to_owned();
    Ok((n_points, coords))
}

type RegBlock = (Array2<usize>, Option<Array1<i64>>);

fn read_regular_block(
    mai: &Group,
    med_type_name: &str,
    et: ElementType,
) -> Result<RegBlock, MefikitIOError> {
    let grp = mai.group(med_type_name)?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let n_nodes = et.num_nodes().ok_or_else(|| {
        MefikitIOError::MalformedFile("element type has variable node count".into())
    })?;
    let n_cells = nod.len() / n_nodes;

    // Column-major reshape, then 1-based → 0-based.
    let flat: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED stores (n_cells, n_nodes) in column-major: reshape as (n_nodes, n_cells) then transpose.
    let conn = Array2::from_shape_vec((n_nodes, n_cells), flat)?;
    let conn = conn.reversed_axes().as_standard_layout().to_owned();

    // Apply inverse MED node permutation.
    let conn = match et.med_permutation() {
        Some(perm) => {
            let mut out = Array2::zeros((conn.nrows(), perm.len()));
            for (new_j, &old_j) in perm.iter().enumerate() {
                out.column_mut(new_j).assign(&conn.column(old_j));
            }
            out
        }
        None => conn,
    };

    let fam = read_fam_array(&grp)?;

    Ok((conn, fam))
}

fn read_fam_array(grp: &Group) -> Result<Option<Array1<i64>>, MefikitIOError> {
    Ok(grp.dataset("FAM").ok().and_then(|ds| ds.read().ok()))
}

type PolyBlock = (Array1<usize>, Array1<usize>, Option<Array1<i64>>);

fn read_polygon_block(mai: &Group, med_type_name: &str) -> Result<PolyBlock, MefikitIOError> {
    let grp = mai.group(med_type_name)?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let inn: Array1<i64> = grp.dataset("INN")?.read()?;

    // 1-based → 0-based.
    let data: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED INN has n+1 entries for n polygons: [1, end1+1, ..., endN+1].
    // Subtract 1 then drop leading 0 to get mefikit's n-entry cumulative-end offsets.
    let offsets: Vec<usize> = inn.iter().skip(1).map(|&x| (x - 1) as usize).collect();

    let fam = read_fam_array(&grp)?;

    Ok((Array1::from_vec(data), Array1::from_vec(offsets), fam))
}

fn read_polyhedron_block(mai: &Group) -> Result<PolyBlock, MefikitIOError> {
    let grp = mai.group("POE")?;
    let nod: Array1<i64> = grp.dataset("NOD")?.read()?;
    let inn: Array1<i64> = grp.dataset("INN")?.read()?;
    let ifn: Array1<i64> = grp.dataset("IFN")?.read()?;

    // 1-based → 0-based for all three arrays.
    let nod: Vec<usize> = nod.iter().map(|&x| (x - 1) as usize).collect();
    // MED INN has n_faces+1 entries (leading 1-based sentinel). Convert to 0-based cumulative.
    let inn: Vec<usize> = inn.iter().map(|&x| (x - 1) as usize).collect();
    // MED IFN has n_poly+1 entries (leading 1-based sentinel). Convert to 0-based cumulative.
    let ifn: Vec<usize> = ifn.iter().map(|&x| (x - 1) as usize).collect();

    let n_poly = ifn.len() - 1;

    // Flatten each polyhedron's faces into a single contiguous slice,
    // inserting usize::MAX sentinels between faces (mefikit convention).
    let mut data = Vec::new();
    let mut offsets = Vec::with_capacity(n_poly);

    for p in 0..n_poly {
        let face_start = ifn[p];
        let face_end = ifn[p + 1];
        for f_idx in face_start..face_end {
            let node_start = inn[f_idx];
            let node_end = inn[f_idx + 1];
            data.extend_from_slice(&nod[node_start..node_end]);
            data.push(usize::MAX);
        }
        offsets.push(data.len());
    }

    let fam = read_fam_array(&grp)?;

    Ok((Array1::from_vec(data), Array1::from_vec(offsets), fam))
}

/// Build group_name → element_index map from FAS family data and per-element FAM array.
fn build_groups_from_fas(
    fam_id_to_names: &BTreeMap<i64, Vec<String>>,
    fam_array: &Array1<i64>,
) -> BTreeMap<String, Vec<usize>> {
    let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();

    for (elem_idx, &fam_id) in fam_array.iter().enumerate() {
        if fam_id == 0 {
            continue;
        }
        if let Some(names) = fam_id_to_names.get(&fam_id) {
            for name in names {
                groups.entry(name.clone()).or_default().push(elem_idx);
            }
        }
    }

    groups
}

/// Assign families/groups to a block and insert it into the mesh.
fn insert_block(
    mesh: &mut UMesh,
    mut block: ElementBlock,
    fam: &Option<Array1<i64>>,
    elem_fam_map: &BTreeMap<i64, Vec<String>>,
) {
    if let Some(fam_arr) = fam {
        let groups = build_groups_from_fas(elem_fam_map, fam_arr);
        block.set_families(fam_arr.mapv(|x| x as usize).into_shared());
        if !groups.is_empty() {
            block.set_groups_internal(groups);
        }
    }
    let (key, wrapped) = block.into_entry();
    mesh.element_blocks.entry(key).or_insert(wrapped);
}

/// Read a MED file into a `UMesh`.
pub fn read(path: impl AsRef<Path>) -> Result<UMesh, MefikitIOError> {
    let file = File::open(path)?;

    // Navigate to mesh ensemble.
    let ensemble = file.group("ENS_MAA")?;
    let mesh_names = ensemble.member_names()?;
    if mesh_names.is_empty() {
        return Err(MefikitIOError::MalformedFile(
            "No meshes found in MED file".into(),
        ));
    }
    if mesh_names.len() > 1 {
        return Err(MefikitIOError::MalformedFile(format!(
            "MED file contains {} meshes; only single-mesh files are supported",
            mesh_names.len()
        )));
    }
    let mesh_name = &mesh_names[0];
    let med_mesh = ensemble.group(mesh_name)?;

    // Spatial dimension.
    let dim = read_scalar_attr_i64(&med_mesh, "ESP")? as usize;

    // Navigate to time-step level. If NOE is not directly under med_mesh,
    // descend into the sole time-step sub-group.
    let data_level = if med_mesh.group("NOE").is_ok() {
        med_mesh.clone()
    } else {
        let steps = med_mesh.member_names()?;
        if steps.is_empty() {
            return Err(MefikitIOError::MalformedFile(
                "No time-step groups found in mesh".into(),
            ));
        }
        if steps.len() > 1 {
            return Err(MefikitIOError::MalformedFile(
                "Multiple time-step groups found; only single step is supported".into(),
            ));
        }
        med_mesh.group(&steps[0])?
    };

    // Read coordinates.
    let (_n_points, coords) = read_nodal_data(&data_level, dim)?;

    let mut mesh = UMesh::new(coords.into_shared());

    // Parse FAS for family/group information.
    let fas_root = file.group("FAS").ok();
    let (_node_fam_map, elem_fam_map) = match &fas_root {
        Some(root) => read_fas(root, mesh_name)?,
        None => (BTreeMap::new(), BTreeMap::new()),
    };

    // Node families (stored in point_data convention; we skip for now since
    // mefikit doesn't have a direct per-node family slot on UMesh itself).

    // Read element blocks.
    let mai = data_level.group("MAI")?;

    for med_type_name in mai.member_names()? {
        let et = match med_name_to_element_type(&med_type_name) {
            Some(et) => et,
            None => continue, // Skip unknown element types.
        };

        match et {
            ElementType::PGON => {
                let (data, offsets, fam) = read_polygon_block(&mai, &med_type_name)?;
                let block =
                    ElementBlock::new_poly(et, data.into_shared(), offsets.into_shared(), None);
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
            ElementType::PHED => {
                let (data, offsets, fam) = read_polyhedron_block(&mai)?;
                let block =
                    ElementBlock::new_poly(et, data.into_shared(), offsets.into_shared(), None);
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
            _ => {
                let (conn, fam) = read_regular_block(&mai, &med_type_name, et)?;
                let block = ElementBlock::new_regular(et, conn.into_shared(), None, None);
                insert_block(&mut mesh, block, &fam, &elem_fam_map);
            }
        }
    }

    read_fields(&file, &mut mesh)?;

    Ok(mesh)
}

/// Read element-centered fields from the CHA group into the mesh.
fn read_fields(file: &File, mesh: &mut UMesh) -> Result<(), MefikitIOError> {
    let cha = match file.group("CHA") {
        Ok(g) => g,
        Err(_) => return Ok(()),
    };

    for field_name in cha.member_names().unwrap_or_default() {
        let field_grp = match cha.group(&field_name) {
            Ok(g) => g,
            Err(_) => continue,
        };

        let n_co = field_grp
            .attr("NCO")
            .ok()
            .and_then(|a| a.read_scalar::<i64>().ok())
            .unwrap_or(1) as usize;

        // Find the first time-step group (any child that is a group).
        let step_grp = match field_grp
            .member_names()
            .ok()
            .and_then(|names| names.iter().find(|n| field_grp.group(n).is_ok()).cloned())
            .and_then(|name| field_grp.group(&name).ok())
        {
            Some(g) => g,
            None => continue,
        };

        // Iterate MAI.<type> support groups.
        for member in step_grp.member_names().unwrap_or_default() {
            if !member.starts_with("MAI.") {
                continue;
            }

            let med_type = &member[4..];
            let et = match med_name_to_element_type(med_type) {
                Some(et) => et,
                None => continue,
            };

            let mai_grp = match step_grp.group(&member) {
                Ok(g) => g,
                Err(_) => continue,
            };

            let prof = match mai_grp.group("MED_NO_PROFILE_INTERNAL") {
                Ok(g) => g,
                Err(_) => continue,
            };

            let n_cells = prof
                .attr("NBR")
                .ok()
                .and_then(|a| a.read_scalar::<i64>().ok())
                .unwrap_or(0) as usize;

            let co: Array1<f64> = match prof.dataset("CO") {
                Ok(ds) => ds.read().unwrap_or_default(),
                Err(_) => continue,
            };

            // MED stores (n_cells, n_components) in Fortran (column-major) order.
            // For scalar (n_co=1), CO is a flat 1D array — no reshape needed.
            // For vector, CO is column-major flattened: reshape as (n_comp, n_cells)
            // then transpose → (n_cells, n_comp).
            let arr = if n_co == 1 {
                co.into_dyn()
            } else {
                let raw = Array2::from_shape_vec((n_co, n_cells), co.to_vec())?;
                raw.reversed_axes()
                    .as_standard_layout()
                    .to_owned()
                    .into_dyn()
            };

            // Set the field on the block.
            if let Some(block) = mesh.element_blocks.get_mut(&et) {
                block.fields.insert(field_name.clone(), arr.into_shared());
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::ElementIds;
    use crate::mesh::ElementLike;
    use crate::mesh_examples as me;
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    fn assert_mesh_eq(a: &UMesh, b: &UMesh) {
        assert_eq!(a.coords(), b.coords());
        let elems1: Vec<_> = a
            .elements()
            .map(|e| (e.element_type(), e.connectivity.to_vec()))
            .collect();
        let elems2: Vec<_> = b
            .elements()
            .map(|e| (e.element_type(), e.connectivity.to_vec()))
            .collect();
        assert_eq!(elems1.len(), elems2.len());
        for ((et1, c1), (et2, c2)) in elems1.iter().zip(elems2.iter()) {
            assert_eq!(et1, et2);
            assert_eq!(c1, c2);
        }

        // Compare family ID multisets per element type. The family ordering may
        // differ when blocks merge (e.g. polygon3+polygon5 → single PGON on
        // write), but the multiset of family IDs per type must match.
        let fams_a = collect_families_by_type(a);
        let fams_b = collect_families_by_type(b);
        let mut types_a: Vec<_> = fams_a.keys().collect();
        let mut types_b: Vec<_> = fams_b.keys().collect();
        types_a.sort();
        types_b.sort();
        assert_eq!(types_a, types_b, "element types differ between meshes");
        for (et, fa) in &fams_a {
            let fb = fams_b.get(et).expect("missing block type in b");
            let mut sa = fa.clone();
            let mut sb = fb.clone();
            sa.sort();
            sb.sort();
            assert_eq!(sa, sb, "family ID multisets differ for {et:?}");
        }

        // Compare groups per element type (group→family-id-sets).
        let groups_a = collect_groups_by_type(a);
        let groups_b = collect_groups_by_type(b);
        let mut gtypes_a: Vec<_> = groups_a.keys().collect();
        let mut gtypes_b: Vec<_> = groups_b.keys().collect();
        gtypes_a.sort();
        gtypes_b.sort();
        assert_eq!(gtypes_a, gtypes_b, "element types with groups differ");
        for (et, ga) in &groups_a {
            let gb = groups_b.get(et).expect("missing block type in b");
            assert_eq!(ga, gb, "groups differ for {et:?}");
        }
    }

    /// Collect families per element type (concatenated across blocks of the same type).
    fn collect_families_by_type(mesh: &UMesh) -> BTreeMap<ElementType, Vec<usize>> {
        let mut result: BTreeMap<ElementType, Vec<usize>> = BTreeMap::new();
        for (et, block) in mesh.blocks() {
            result
                .entry(*et)
                .or_default()
                .extend(block.families().iter());
        }
        result
    }

    /// Collect groups per element type (merged across blocks of the same type).
    fn collect_groups_by_type(
        mesh: &UMesh,
    ) -> BTreeMap<ElementType, BTreeMap<String, BTreeSet<usize>>> {
        let mut result: BTreeMap<ElementType, BTreeMap<String, BTreeSet<usize>>> = BTreeMap::new();
        for (et, block) in mesh.blocks() {
            let entry = result.entry(*et).or_default();
            for (name, fids) in block.groups() {
                entry.entry(name.clone()).or_default().extend(fids);
            }
        }
        result
    }

    #[test]
    fn test_roundtrip_med_2d() {
        let path = PathBuf::from("test_roundtrip_med_2d.med");
        let mesh = me::make_mesh_2d_multi();
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_3d() {
        let path = PathBuf::from("test_roundtrip_med_3d.med");
        let mesh = me::make_imesh_3d(2);
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_phed() {
        let path = PathBuf::from("test_roundtrip_med_phed.med");

        let coords = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());

        let conn: Vec<usize> = vec![
            0,
            1,
            2,
            3,
            usize::MAX,
            4,
            7,
            6,
            5,
            usize::MAX,
            0,
            4,
            5,
            1,
            usize::MAX,
            2,
            6,
            7,
            3,
            usize::MAX,
            0,
            3,
            7,
            4,
            usize::MAX,
            1,
            5,
            6,
            2,
            usize::MAX,
        ];
        mesh.add_element(ElementType::PHED, &conn, None);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_groups() {
        let path = PathBuf::from("test_roundtrip_med_groups.med");
        let mesh = me::make_imesh_3d(2);
        let mut mesh = mesh;

        let mut wall_ids = ElementIds::new();
        wall_ids.add_block(ElementType::TET4, vec![0, 1]);
        mesh.add_to_group("wall", &wall_ids);

        let mut inlet_ids = ElementIds::new();
        inlet_ids.add_block(ElementType::TET4, vec![2]);
        mesh.add_to_group("inlet", &inlet_ids);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_med_groups_overlapping() {
        let path = PathBuf::from("test_roundtrip_med_groups_overlapping.med");
        let mesh = me::make_imesh_3d(2);
        let mut mesh = mesh;

        let mut ids1 = ElementIds::new();
        ids1.add_block(ElementType::TET4, vec![0, 1, 2]);
        mesh.add_to_group("region_a", &ids1);

        let mut ids2 = ElementIds::new();
        ids2.add_block(ElementType::TET4, vec![2, 3, 4]);
        mesh.add_to_group("region_b", &ids2);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_field_scalar() {
        let path = PathBuf::from("test_roundtrip_field_scalar.med");
        let mut mesh = me::make_imesh_3d(2);
        let n_hex = mesh.block(ElementType::HEX8).unwrap().len();

        let vals: Vec<f64> = (0..n_hex).map(|i| i as f64 * 1.5).collect();
        let arr = Array1::from_vec(vals).into_dyn().into_shared();
        let field_map: BTreeMap<_, _> = BTreeMap::from([(ElementType::HEX8, arr)]);
        mesh.update_field("temperature", crate::mesh::FieldArcD::new(field_map));

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let f1 = mesh.field("temperature", None).unwrap();
        let f2 = mesh2.field("temperature", None).unwrap();
        assert_eq!(
            f1.0.keys().collect::<Vec<_>>(),
            f2.0.keys().collect::<Vec<_>>()
        );
        for (et, v1) in &f1.0 {
            let v2 = f2.0.get(et).unwrap();
            assert_eq!(v1, v2, "field values differ for {et:?}");
        }
    }

    #[test]
    fn test_roundtrip_field_vector() {
        let path = PathBuf::from("test_roundtrip_field_vector.med");
        let mut mesh = me::make_imesh_3d(2);
        let n_hex = mesh.block(ElementType::HEX8).unwrap().len();

        let mut data = Array2::zeros((n_hex, 3));
        for i in 0..n_hex {
            data[[i, 0]] = i as f64;
            data[[i, 1]] = i as f64 * 2.0;
            data[[i, 2]] = i as f64 * 3.0;
        }
        let arr = data.into_dyn().into_shared();
        let field_map: BTreeMap<_, _> = BTreeMap::from([(ElementType::HEX8, arr)]);
        mesh.update_field("velocity", crate::mesh::FieldArcD::new(field_map));

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let f1 = mesh.field("velocity", None).unwrap();
        let f2 = mesh2.field("velocity", None).unwrap();
        assert_eq!(
            f1.0.keys().collect::<Vec<_>>(),
            f2.0.keys().collect::<Vec<_>>()
        );
        for (et, v1) in &f1.0 {
            let v2 = f2.0.get(et).unwrap();
            assert_eq!(v1, v2, "field values differ for {et:?}");
        }
    }

    #[test]
    fn test_roundtrip_field_multi_component() {
        let path = PathBuf::from("test_roundtrip_field_multi.med");
        let mut mesh = me::make_imesh_3d(2);
        let n_hex = mesh.block(ElementType::HEX8).unwrap().len();

        mesh.update_field(
            "temperature",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::HEX8,
                Array1::from_vec((0..n_hex).map(|i| i as f64 * 0.1).collect())
                    .into_dyn()
                    .into_shared(),
            )])),
        );

        let mut vel = Array2::zeros((n_hex, 3));
        for i in 0..n_hex {
            vel[[i, 0]] = i as f64;
            vel[[i, 1]] = i as f64 * 2.0;
            vel[[i, 2]] = i as f64 * 3.0;
        }
        mesh.update_field(
            "velocity",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::HEX8,
                vel.into_dyn().into_shared(),
            )])),
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        // Verify scalar
        let f1 = mesh.field("temperature", None).unwrap();
        let f2 = mesh2.field("temperature", None).unwrap();
        for (et, v1) in &f1.0 {
            let v2 = f2.0.get(et).unwrap();
            assert_eq!(v1, v2, "temperature differs for {et:?}");
        }

        // Verify vector
        let f1 = mesh.field("velocity", None).unwrap();
        let f2 = mesh2.field("velocity", None).unwrap();
        for (et, v1) in &f1.0 {
            let v2 = f2.0.get(et).unwrap();
            assert_eq!(v1, v2, "velocity differs for {et:?}");
        }
    }

    #[test]
    fn test_med_export_structure() {
        // Structural check that the written MED file is consumable by external
        // tools: every mesh cell group must carry a numeric GEO attribute, and
        // every CHA field must expose NOM as an *attribute* (not a dataset).
        // tools: every mesh cell group must carry a numeric GEO attribute, and
        // every CHA field must expose NOM as an *attribute* (not a dataset).
        let path = PathBuf::from("test_med_export_structure.med");
        let mut mesh = me::make_mesh_2d_multi(); // SEG2 + QUAD4 + PGON
        let n_quad = mesh.block(ElementType::QUAD4).unwrap().len();
        let n_pgon = mesh.block(ElementType::PGON).unwrap().len();
        // update_field requires the field on every element type of the field's
        // dimension (here D2: QUAD4 + PGON).
        mesh.update_field(
            "pressure",
            crate::mesh::FieldArcD::new(BTreeMap::from([
                (
                    ElementType::QUAD4,
                    Array1::from_vec((0..n_quad).map(|i| i as f64 * 2.0).collect())
                        .into_dyn()
                        .into_shared(),
                ),
                (
                    ElementType::PGON,
                    Array1::from_vec((0..n_pgon).map(|i| i as f64 * 2.0).collect())
                        .into_dyn()
                        .into_shared(),
                ),
            ])),
        );

        write(&path, &mesh.view()).unwrap();

        let file = File::open(&path).unwrap();
        let ensemble = file.group("ENS_MAA").unwrap();
        let med_mesh = ensemble
            .group(&ensemble.member_names().unwrap()[0])
            .unwrap();
        let mai = med_mesh
            .group("-0000000000000000001-0000000000000000001")
            .unwrap()
            .group("MAI")
            .unwrap();

        // Every cell group must carry a numeric MED geometry-type code (GEO).
        let mut types: Vec<String> = mai
            .member_names()
            .unwrap()
            .into_iter()
            .filter(|n| mai.group(n).is_ok())
            .collect();
        types.sort();
        assert_eq!(types, vec!["POG", "QU4", "SE2"]);
        for med_type in ["SE2", "QU4", "POG"] {
            let grp = mai.group(med_type).unwrap();
            let geo: i64 = grp.attr("GEO").unwrap().read_scalar().unwrap();
            assert!(geo > 0, "GEO missing/invalid for {med_type}");
        }

        // The CHA field must have NOM as an attribute, not a dataset.
        let cha = file.group("CHA").unwrap();
        let field = cha.group("pressure").unwrap();
        assert!(field.attr("NOM").is_ok(), "field NOM must be an attribute");
        assert!(
            field.dataset("NOM").is_err(),
            "field NOM must not be a dataset"
        );

        // Round-trip sanity, then cleanup.
        let mesh2 = read(&path).unwrap();
        let f2 = mesh2.field("pressure", None).unwrap();
        let mut keys: Vec<_> = f2.0.keys().cloned().collect();
        keys.sort_by_key(|k| format!("{k:?}"));
        assert_eq!(
            keys,
            vec![ElementType::PGON, ElementType::QUAD4],
            "field must round-trip on both D2 blocks"
        );
        std::fs::remove_file(&path).unwrap();
    }

    /// Build a cube mesh [0,1]^3 (8 corner nodes) with a HEX8 block plus,
    /// sharing those nodes, TET4 / QUAD4 / TRI3 / SEG2 / VERTEX blocks.
    fn make_mixed_3d_mesh() -> UMesh {
        let coords = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(
            ElementType::HEX8,
            arr2(&[[0, 1, 3, 2, 4, 5, 7, 6]]).to_shared(),
            None,
        );
        mesh.add_regular_block(ElementType::TET4, arr2(&[[0, 1, 2, 4]]).to_shared(), None);
        mesh.add_regular_block(ElementType::QUAD4, arr2(&[[0, 1, 5, 4]]).to_shared(), None);
        mesh.add_regular_block(ElementType::TRI3, arr2(&[[0, 1, 5]]).to_shared(), None);
        mesh.add_regular_block(ElementType::SEG2, arr2(&[[0, 1]]).to_shared(), None);
        mesh.add_regular_block(ElementType::VERTEX, arr2(&[[7]]).to_shared(), None);
        mesh
    }

    fn make_mixed_2d_mesh() -> UMesh {
        // Unit square + one hanging vertex, meshed with QUAD4 + TRI3 + SEG2 + VERTEX.
        let coords = Array2::from_shape_vec(
            (5, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(ElementType::QUAD4, arr2(&[[0, 1, 2, 3]]).to_shared(), None);
        mesh.add_regular_block(ElementType::TRI3, arr2(&[[0, 1, 4]]).to_shared(), None);
        mesh.add_regular_block(ElementType::SEG2, arr2(&[[0, 1]]).to_shared(), None);
        mesh.add_regular_block(ElementType::VERTEX, arr2(&[[4]]).to_shared(), None);
        mesh
    }

    #[test]
    fn test_roundtrip_mixed_3d() {
        let path = PathBuf::from("test_roundtrip_mixed_3d.med");
        let mesh = make_mixed_3d_mesh();
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_mixed_2d() {
        let path = PathBuf::from("test_roundtrip_mixed_2d.med");
        let mesh = make_mixed_2d_mesh();
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_pgon_2d_mixed() {
        let path = PathBuf::from("test_roundtrip_pgon_2d_mixed.med");
        // PGON (pentagon) + TRI3 + SEG2 + VERTEX in a 2D space.
        let coords = Array2::from_shape_vec(
            (6, 2),
            vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.5, 1.5],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(ElementType::TRI3, arr2(&[[0, 1, 4]]).to_shared(), None);
        mesh.add_regular_block(ElementType::SEG2, arr2(&[[0, 1]]).to_shared(), None);
        mesh.add_regular_block(ElementType::VERTEX, arr2(&[[5]]).to_shared(), None);
        mesh.add_poly_block(
            ElementType::PGON,
            arr1(&[0, 1, 2, 3, 4]).to_shared(),
            arr1(&[5]).to_shared(),
            None,
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_seg2_1d() {
        let path = PathBuf::from("test_roundtrip_seg2_1d.med");
        let mesh = me::make_mesh_3d_seg2();
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_vertex_only() {
        let path = PathBuf::from("test_roundtrip_vertex_only.med");
        let coords = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(
            ElementType::VERTEX,
            arr2(&[[0], [1], [2], [3]]).to_shared(),
            None,
        );
        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_fields_all_types() {
        // A scalar field covering every regular block type (each dimension) and
        // a vector field on D2 must round-trip.
        let path = PathBuf::from("test_roundtrip_fields_all_types.med");
        let mut mesh = make_mixed_3d_mesh();

        let mut scalar_map = BTreeMap::new();
        for et in [ElementType::HEX8, ElementType::TET4] {
            let n = mesh.block(et).unwrap().len();
            scalar_map.insert(
                et,
                Array1::from_vec((0..n).map(|i| i as f64 + 0.5).collect())
                    .into_dyn()
                    .into_shared(),
            );
        }
        mesh.update_field("pressure", crate::mesh::FieldArcD::new(scalar_map));

        let mut vec_map = BTreeMap::new();
        for et in [ElementType::QUAD4, ElementType::TRI3] {
            let n = mesh.block(et).unwrap().len();
            let mut data = Array2::zeros((n, 2));
            for i in 0..n {
                data[[i, 0]] = i as f64;
                data[[i, 1]] = i as f64 * 2.0;
            }
            vec_map.insert(et, data.into_dyn().into_shared());
        }
        mesh.update_field("velocity", crate::mesh::FieldArcD::new(vec_map));

        // D1 field on SEG2 and D0 field on VERTEX (PO1) blocks.
        let n_seg = mesh.block(ElementType::SEG2).unwrap().len();
        mesh.update_field(
            "seg_stress",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::SEG2,
                Array1::from_vec((0..n_seg).map(|i| i as f64 + 7.0).collect())
                    .into_dyn()
                    .into_shared(),
            )])),
        );
        let n_vert = mesh.block(ElementType::VERTEX).unwrap().len();
        mesh.update_field(
            "vertex_temp",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::VERTEX,
                Array1::from_vec((0..n_vert).map(|i| i as f64 + 1.0).collect())
                    .into_dyn()
                    .into_shared(),
            )])),
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        for (name, dim) in [
            ("pressure", 3usize),
            ("velocity", 2),
            ("seg_stress", 1),
            ("vertex_temp", 0),
        ] {
            let f1 = mesh.field(name, Some(mf_dim(dim))).unwrap();
            let f2 = mesh2.field(name, Some(mf_dim(dim))).unwrap();
            assert_eq!(
                f1.0.keys().collect::<Vec<_>>(),
                f2.0.keys().collect::<Vec<_>>(),
                "field {name}"
            );
            for (et, v1) in &f1.0 {
                let v2 = f2.0.get(et).unwrap();
                assert_eq!(v1, v2, "field {name} differs for {et:?}");
            }
        }
    }

    #[test]
    fn test_roundtrip_field_on_vertex_block() {
        // A vector field carried by a VERTEX (PO1) element block must round-trip:
        // written as MAI.PO1 and read back onto the VERTEX block, preserving the
        // (n, n_components) column-major ordering.
        let path = PathBuf::from("test_roundtrip_field_vertex.med");
        let coords = Array2::from_shape_vec(
            (4, 3),
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_regular_block(
            ElementType::VERTEX,
            arr2(&[[0], [1], [2], [3]]).to_shared(),
            None,
        );

        let n = mesh.block(ElementType::VERTEX).unwrap().len();
        let mut data = Array2::zeros((n, 2));
        for i in 0..n {
            data[[i, 0]] = i as f64 * 3.0;
            data[[i, 1]] = i as f64 * 5.0;
        }
        mesh.update_field(
            "my_vertex_field",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::VERTEX,
                data.into_dyn().into_shared(),
            )])),
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let v1 = mesh
            .block(ElementType::VERTEX)
            .unwrap()
            .fields
            .get("my_vertex_field")
            .unwrap();
        let v2 = mesh2
            .block(ElementType::VERTEX)
            .unwrap()
            .fields
            .get("my_vertex_field")
            .unwrap();
        assert_eq!(v1, v2, "vertex-block field round-trip");
    }

    fn mf_dim(d: usize) -> crate::mesh::Dimension {
        match d {
            0 => crate::mesh::Dimension::D0,
            1 => crate::mesh::Dimension::D1,
            2 => crate::mesh::Dimension::D2,
            _ => crate::mesh::Dimension::D3,
        }
    }

    #[test]
    fn test_roundtrip_field_on_phed_block() {
        let path = PathBuf::from("test_roundtrip_field_phed.med");
        let coords = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        let conn: Vec<usize> = vec![
            0,
            1,
            2,
            3,
            usize::MAX,
            4,
            7,
            6,
            5,
            usize::MAX,
            0,
            4,
            5,
            1,
            usize::MAX,
            2,
            6,
            7,
            3,
            usize::MAX,
            0,
            3,
            7,
            4,
            usize::MAX,
            1,
            5,
            6,
            2,
            usize::MAX,
        ];
        mesh.add_element(ElementType::PHED, &conn, None);
        mesh.update_field(
            "pressure",
            crate::mesh::FieldArcD::new(BTreeMap::from([(
                ElementType::PHED,
                Array1::from_vec(vec![1.5]).into_dyn().into_shared(),
            )])),
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        let v1 = mesh
            .block(ElementType::PHED)
            .unwrap()
            .fields
            .get("pressure")
            .unwrap();
        let v2 = mesh2
            .block(ElementType::PHED)
            .unwrap()
            .fields
            .get("pressure")
            .unwrap();
        assert_eq!(v1, v2, "PHED-block field round-trip");
    }

    #[test]
    fn test_roundtrip_pgon_in_3d_space() {
        // A 2D polygon used as the top-level element of a mesh embedded in a 3D
        // space (all coordinates live in the z=0 plane).
        let path = PathBuf::from("test_roundtrip_pgon_3d.med");
        let coords = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.5, 0.5, 0.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());
        mesh.add_poly_block(
            ElementType::PGON,
            arr1(&[0, 1, 2, 3, 4]).to_shared(),
            arr1(&[5]).to_shared(),
            None,
        );

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_pgon_and_phed() {
        // A volume of PHED with a 2D PGON surface in the same 3D space.
        let path = PathBuf::from("test_roundtrip_pgon_phed.med");
        let coords = Array2::from_shape_vec(
            (8, 3),
            vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0,
                0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
            ],
        )
        .unwrap();
        let mut mesh = UMesh::new(coords.into());

        // Hexahedron as a polyhedron block.
        let phed_conn: Vec<usize> = vec![
            0,
            1,
            2,
            3,
            usize::MAX,
            4,
            7,
            6,
            5,
            usize::MAX,
            0,
            4,
            5,
            1,
            usize::MAX,
            2,
            6,
            7,
            3,
            usize::MAX,
            0,
            3,
            7,
            4,
            usize::MAX,
            1,
            5,
            6,
            2,
            usize::MAX,
        ];
        mesh.add_element(ElementType::PHED, &phed_conn, None);

        // Bottom face as a polygon.
        mesh.add_element(ElementType::PGON, &[0, 1, 2, 3], None);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }

    #[test]
    fn test_roundtrip_real_phed_data() {
        // Round-trip through the real (Salome/medcoupling) polyhedron file:
        // read the external file, write it back with mefikit, read again. The
        // two mefikit reads must be identical.
        let src = PathBuf::from("../../tests/data/mesh_27.med");
        if !src.exists() {
            return;
        }
        let path = PathBuf::from("test_roundtrip_real_phed_data.med");
        let mesh = read(&src).unwrap();
        let ph = mesh
            .block(ElementType::PHED)
            .expect("mesh_27 must contain a PHED block");
        assert!(ph.len() > 0);

        write(&path, &mesh.view()).unwrap();
        let mesh2 = read(&path).unwrap();
        std::fs::remove_file(&path).unwrap();
        assert_mesh_eq(&mesh, &mesh2);
    }
}
