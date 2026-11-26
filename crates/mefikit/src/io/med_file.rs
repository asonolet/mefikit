//! High-level multi-mesh MED file container.
//!
//! The low-level HDF5 layout lives in [`super::med_io`]; this module adds named
//! meshes, per-mesh fields, time steps, metadata inspection and partial access.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use hdf5_metno::{File, Group};

use super::error::MefikitIOError;
use super::hdf_utils::read_group_attr;
use super::med_io;
use crate::mesh::{Dimension, UMesh};

/// One time-step selection for a field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepSelection {
    /// Every step.
    All,
    /// One zero-based step index; negative indices count from the end.
    Index(i64),
    /// Several zero-based step indices; negative indices count from the end.
    Indices(Vec<i64>),
}

/// Selection for the fields of one mesh.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshSelection {
    /// Mesh, all its fields and all steps.
    All,
    /// Mesh and only the listed fields. An empty map loads no fields.
    Fields(BTreeMap<String, StepSelection>),
}

/// Selection for a whole MED file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MedSelection {
    /// Mesh name → selection. An empty map loads no meshes.
    pub meshes: BTreeMap<String, MeshSelection>,
}

/// One field step in a MED file.
#[derive(Clone, Debug, PartialEq)]
pub struct MedStepInfo {
    /// Zero-based index in the file's step order.
    pub index: usize,
    /// MED iteration number.
    pub iteration: i64,
    /// MED order number.
    pub order: i64,
    /// MED physical time.
    pub time: f64,
}

/// Metadata for one mesh in a MED file.
#[derive(Clone, Debug, PartialEq)]
pub struct MedMeshInfo {
    /// Mesh name.
    pub name: String,
    /// Coordinate space dimension.
    pub space_dimension: usize,
    /// Number of coordinate nodes.
    pub num_nodes: usize,
    /// Number of cells at the file's top level.
    pub num_cells: usize,
    /// Number of cells per MED element type name.
    pub element_types: BTreeMap<String, usize>,
    /// MED element type names not supported by mefikit.
    pub unsupported_element_types: Vec<String>,
    /// Element group/family names.
    pub groups: Vec<String>,
}

/// Metadata for one field in a MED file.
#[derive(Clone, Debug, PartialEq)]
pub struct MedFieldInfo {
    /// Field name (unique in the file).
    pub name: String,
    /// Owning mesh name.
    pub mesh_name: String,
    /// Inferred topological dimension, when the field uses a supported element type.
    pub dimension: Option<Dimension>,
    /// Number of components.
    pub components: usize,
    /// Time steps, in ascending MED iteration order.
    pub steps: Vec<MedStepInfo>,
}

/// Metadata-only view of a MED file.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MedFileStructure {
    /// Named meshes.
    pub meshes: BTreeMap<String, MedMeshInfo>,
    /// Named fields.
    pub fields: BTreeMap<String, MedFieldInfo>,
}

/// A multi-mesh MED file in memory.
#[derive(Clone, Debug, Default)]
pub struct MedFile {
    /// Meshes by name. Fields belong to their owning `UMesh`.
    pub meshes: BTreeMap<String, UMesh>,
}

/// How to write a [`MedFile`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MedWriteMode {
    /// Create a new file; fail if it already exists.
    Create,
    /// Replace an existing file, or create it.
    Overwrite,
    /// Open an existing file and update only the entries present in memory.
    Update,
}

impl MedFile {
    /// Reads a MED file, optionally applying a partial selection.
    pub fn read(
        path: impl AsRef<Path>,
        selection: Option<&MedSelection>,
    ) -> Result<Self, MefikitIOError> {
        let file = File::open(path)?;
        let structure = inspect_file(&file)?;
        let mut meshes = BTreeMap::new();

        let names: Vec<String> = match selection {
            None => structure.meshes.keys().cloned().collect(),
            Some(selection) => selection.meshes.keys().cloned().collect(),
        };

        for mesh_name in names {
            let mesh_selection = selection
                .and_then(|selection| selection.meshes.get(&mesh_name))
                .cloned();
            let mesh = read_mesh_from_file(&file, &structure, &mesh_name, mesh_selection.as_ref())?;
            meshes.insert(mesh_name, mesh);
        }

        Ok(Self { meshes })
    }

    /// Reads one named mesh, optionally applying a partial field selection.
    ///
    /// `selection` follows the per-mesh rules of [`MedSelection`]: `None` loads
    /// every field, while `MeshSelection::Fields` with an empty map loads only
    /// the mesh geometry and groups.
    pub fn read_mesh(
        path: impl AsRef<Path>,
        mesh_name: &str,
        selection: Option<&MeshSelection>,
    ) -> Result<UMesh, MefikitIOError> {
        let file = File::open(path)?;
        let structure = inspect_file(&file)?;
        read_mesh_from_file(&file, &structure, mesh_name, selection)
    }

    /// Reads metadata only, without loading any mesh or field arrays.
    pub fn inspect(path: impl AsRef<Path>) -> Result<MedFileStructure, MefikitIOError> {
        let file = File::open(path)?;
        inspect_file(&file)
    }

    /// Writes this container to a MED file.
    pub fn write(&self, path: impl AsRef<Path>, mode: MedWriteMode) -> Result<(), MefikitIOError> {
        let meshes: Vec<(&str, &UMesh)> = self
            .meshes
            .iter()
            .map(|(name, mesh)| (name.as_str(), mesh))
            .collect();
        write_named_meshes(path, mode, &meshes)
    }
}

/// Writes named meshes without taking ownership of them.
///
/// This is the borrowed counterpart of [`MedFile::write`], used by bindings
/// that keep their meshes in foreign objects.
pub fn write_named_meshes(
    path: impl AsRef<Path>,
    mode: MedWriteMode,
    meshes: &[(&str, &UMesh)],
) -> Result<(), MefikitIOError> {
    validate_field_names(meshes)?;
    for &(_, mesh) in meshes {
        super::validate_field_coverage(&mesh.view())?;
    }
    let path = path.as_ref();
    match mode {
        MedWriteMode::Create => {
            if path.exists() {
                return Err(MefikitIOError::Io(std::io::Error::new(
                    std::io::ErrorKind::AlreadyExists,
                    format!("{path:?} already exists"),
                )));
            }
            let file = File::create(path)?;
            med_io::write_general_info(&file)?;
            write_meshes(&file, meshes)
        }
        MedWriteMode::Overwrite => {
            let file = File::create(path)?;
            med_io::write_general_info(&file)?;
            write_meshes(&file, meshes)
        }
        MedWriteMode::Update => {
            let file = if path.exists() {
                File::open_rw(path)?
            } else {
                File::create(path)?
            };
            if file.group("INFOS_GENERALES").is_err() {
                med_io::write_general_info(&file)?;
            }
            write_meshes(&file, meshes)
        }
    }
}

fn read_mesh_from_file(
    file: &File,
    structure: &MedFileStructure,
    mesh_name: &str,
    selection: Option<&MeshSelection>,
) -> Result<UMesh, MefikitIOError> {
    if !structure.meshes.contains_key(mesh_name) {
        return Err(MefikitIOError::MalformedFile(format!(
            "MED file has no mesh named {mesh_name:?}"
        )));
    }

    let field_steps = match selection {
        None | Some(MeshSelection::All) => None,
        Some(MeshSelection::Fields(fields)) => {
            let mut steps = BTreeMap::new();
            for (field_name, step_selection) in fields {
                let info = structure.fields.get(field_name).ok_or_else(|| {
                    MefikitIOError::MalformedFile(format!(
                        "MED file has no field named {field_name:?}"
                    ))
                })?;
                if info.mesh_name != mesh_name {
                    return Err(MefikitIOError::MalformedFile(format!(
                        "field {field_name:?} belongs to mesh {:?}, not {mesh_name:?}",
                        info.mesh_name
                    )));
                }
                steps.insert(
                    field_name.clone(),
                    resolve_steps(step_selection, info.steps.len())?,
                );
            }
            Some(steps)
        }
    };

    let mut mesh = med_io::read_mesh_base(file, mesh_name)?;
    med_io::read_fields_named(file, mesh_name, &mut mesh, field_steps.as_ref())?;
    Ok(mesh)
}

/// Writes every mesh and its fields into an open file.
fn write_meshes(file: &File, meshes: &[(&str, &UMesh)]) -> Result<(), MefikitIOError> {
    for &(mesh_name, mesh) in meshes {
        // Replace only this mesh's geometry and families; other meshes are untouched.
        if let Ok(ensemble) = file.group("ENS_MAA") {
            let _ = ensemble.unlink(mesh_name);
        }
        if let Ok(fas) = file.group("FAS") {
            let _ = fas.unlink(mesh_name);
        }
        med_io::write_mesh_named(file, mesh_name, &mesh.view())?;
        med_io::write_fields_named(file, mesh_name, &mesh.view())?;
    }
    Ok(())
}

/// Enforces the MED file-level field-name namespace.
fn validate_field_names(meshes: &[(&str, &UMesh)]) -> Result<(), MefikitIOError> {
    let mut seen: BTreeMap<String, (String, Dimension)> = BTreeMap::new();
    for &(mesh_name, mesh) in meshes {
        for (field_name, dimension) in mesh_field_names(mesh) {
            if let Some((previous_mesh, previous_dim)) =
                seen.insert(field_name.clone(), (mesh_name.to_string(), dimension))
            {
                if previous_mesh == mesh_name {
                    return Err(MefikitIOError::InvalidMesh(format!(
                        "field name {field_name:?} is used at dimensions {previous_dim:?} and \
                         {dimension:?} of mesh {mesh_name:?}; MED stores one support per field"
                    )));
                }
                return Err(MefikitIOError::InvalidMesh(format!(
                    "field name {field_name:?} is used on both {previous_mesh:?} and {mesh_name:?}; \
                     MED field names are file-global"
                )));
            }
        }
    }
    Ok(())
}

fn resolve_steps(
    selection: &StepSelection,
    total: usize,
) -> Result<BTreeSet<usize>, MefikitIOError> {
    let mut result = BTreeSet::new();
    let mut push = |index: i64| -> Result<(), MefikitIOError> {
        let resolved = if index < 0 {
            total as i64 + index
        } else {
            index
        };
        if resolved < 0 || resolved as usize >= total {
            return Err(MefikitIOError::MalformedFile(format!(
                "step index {index} is out of range for {total} steps"
            )));
        }
        result.insert(resolved as usize);
        Ok(())
    };
    match selection {
        StepSelection::All => result.extend(0..total),
        StepSelection::Index(index) => push(*index)?,
        StepSelection::Indices(indices) => {
            for index in indices {
                push(*index)?;
            }
        }
    }
    Ok(result)
}

fn mesh_field_names(mesh: &UMesh) -> BTreeSet<(String, Dimension)> {
    let mut names = BTreeSet::new();
    for (element_type, block) in mesh.blocks() {
        for name in block.fields.keys() {
            let name = match crate::tools::timeseries::FieldKey::parse(name) {
                Some(key) => key.base,
                None => name.clone(),
            };
            names.insert((name, element_type.dimension()));
        }
    }
    names
}

fn inspect_file(file: &File) -> Result<MedFileStructure, MefikitIOError> {
    let ensemble = file.group("ENS_MAA")?;
    let fas_root = file.group("FAS").ok();
    let mut structure = MedFileStructure::default();

    for mesh_name in ensemble.member_names()? {
        let med_mesh = ensemble.group(&mesh_name)?;
        let space_dimension = med_io::read_scalar_attr_i64(&med_mesh, "ESP")? as usize;
        let data_level = mesh_data_level(&med_mesh)?;
        let num_nodes = match data_level.group("NOE").and_then(|noe| noe.dataset("COO")) {
            Ok(ds) => ds.shape().iter().product::<usize>() / space_dimension.max(1),
            Err(_) => 0,
        };

        let mut element_types = BTreeMap::new();
        let mut unsupported_element_types = Vec::new();
        let mut num_cells = 0usize;
        if let Ok(mai) = data_level.group("MAI") {
            for type_name in mai.member_names()? {
                let count = element_count(&mai, &type_name)?;
                num_cells += count;
                if med_io::med_name_to_element_type(&type_name).is_some() {
                    element_types.insert(type_name, count);
                } else {
                    unsupported_element_types.push(type_name);
                }
            }
        }

        let mut groups = BTreeSet::new();
        if let Some(root) = &fas_root {
            // `UMesh` only carries element groups; node families are not loaded.
            let (_node_fams, elem_fams) = med_io::read_fas(root, &mesh_name)?;
            for names in elem_fams.values() {
                groups.extend(names.iter().cloned());
            }
        }

        structure.meshes.insert(
            mesh_name.clone(),
            MedMeshInfo {
                name: mesh_name.clone(),
                space_dimension,
                num_nodes,
                num_cells,
                element_types,
                unsupported_element_types,
                groups: groups.into_iter().collect(),
            },
        );
    }

    let cha = match file.group("CHA") {
        Ok(group) => group,
        Err(_) => return Ok(structure),
    };

    for field_name in cha.member_names().unwrap_or_default() {
        let field_grp = cha.group(&field_name)?;
        let mesh_name = read_group_attr(&field_grp, "MAI")
            .unwrap_or_default()
            .trim()
            .to_string();
        let components = field_grp
            .attr("NCO")
            .ok()
            .and_then(|attr| attr.read_scalar::<i64>().ok())
            .unwrap_or(1) as usize;

        let mut steps = Vec::new();
        for step_name in field_grp.member_names().unwrap_or_default() {
            let Ok(step_grp) = field_grp.group(&step_name) else {
                continue;
            };
            let iteration = step_grp
                .attr("NDT")
                .ok()
                .and_then(|attr| attr.read_scalar::<i64>().ok())
                .unwrap_or(-1);
            let order = step_grp
                .attr("NOR")
                .ok()
                .and_then(|attr| attr.read_scalar::<i64>().ok())
                .unwrap_or(-1);
            let time = step_grp
                .attr("PDT")
                .ok()
                .and_then(|attr| attr.read_scalar::<f64>().ok())
                .unwrap_or(0.0);
            steps.push((iteration, order, time));
        }
        steps.sort_by_key(|(iteration, order, _)| (*iteration, *order));
        let steps: Vec<MedStepInfo> = steps
            .into_iter()
            .enumerate()
            .map(|(index, (iteration, order, time))| MedStepInfo {
                index,
                iteration,
                order,
                time,
            })
            .collect();

        let dimension = field_dimension(&field_grp);
        if structure
            .fields
            .insert(
                field_name.clone(),
                MedFieldInfo {
                    name: field_name,
                    mesh_name,
                    dimension,
                    components,
                    steps,
                },
            )
            .is_some()
        {
            return Err(MefikitIOError::MalformedFile(
                "duplicate field name in MED file".into(),
            ));
        }
    }

    Ok(structure)
}

fn mesh_data_level(med_mesh: &Group) -> Result<Group, MefikitIOError> {
    if med_mesh.group("NOE").is_ok() {
        return Ok(med_mesh.clone());
    }
    let steps = med_mesh.member_names()?;
    let Some(step) = steps.first() else {
        return Err(MefikitIOError::MalformedFile(
            "No time-step groups found in mesh".into(),
        ));
    };
    Ok(med_mesh.group(step)?)
}

fn element_count(mai: &Group, type_name: &str) -> Result<usize, MefikitIOError> {
    let group = mai.group(type_name)?;
    if let Ok(ifn) = group.dataset("IFN") {
        return Ok(ifn.shape().iter().product::<usize>().saturating_sub(1));
    }
    if let Ok(inn) = group.dataset("INN") {
        return Ok(inn.shape().iter().product::<usize>().saturating_sub(1));
    }
    let nod = group.dataset("NOD")?;
    let n_nodes = nod.shape().iter().product::<usize>();
    match med_type_num_nodes(type_name) {
        Some(nodes_per_cell) if nodes_per_cell > 0 => Ok(n_nodes / nodes_per_cell),
        _ => Ok(n_nodes),
    }
}

/// Node count for MED element type names, including types mefikit cannot read.
fn med_type_num_nodes(name: &str) -> Option<usize> {
    med_io::med_name_to_element_type(name)
        .and_then(|element_type| element_type.num_nodes())
        .or(match name {
            "PY5" => Some(5),
            "P13" => Some(13),
            "H20" => Some(20),
            "H27" => Some(27),
            _ => None,
        })
}

fn field_dimension(field_grp: &Group) -> Option<Dimension> {
    for step_name in field_grp.member_names().ok()? {
        let Ok(step_grp) = field_grp.group(&step_name) else {
            continue;
        };
        for member in step_grp.member_names().ok()? {
            if let Some(med_type) = member.strip_prefix("MAI.")
                && let Some(et) = med_io::med_name_to_element_type(med_type)
            {
                return Some(et.dimension());
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::hdf_utils::hdf5_test_guard;
    use crate::mesh::{ElementIds, ElementType, FieldArcD};
    use crate::mesh_examples as me;
    use crate::tools::timeseries::FieldKey;
    use ndarray as nd;

    fn set_field(mesh: &mut UMesh, element_type: ElementType, name: &str, data: nd::ArrayD<f64>) {
        let field = FieldArcD::new(BTreeMap::from([(element_type, data.into_shared())]));
        mesh.update_field(name, field);
    }

    fn make_3d_mesh() -> UMesh {
        let mut mesh = me::make_imesh_3d(1);
        set_field(
            &mut mesh,
            ElementType::HEX8,
            "pressure",
            nd::arr1(&[3.0]).into_dyn(),
        );
        set_field(
            &mut mesh,
            ElementType::HEX8,
            "velocity",
            nd::arr2(&[[1.0, 2.0]]).into_dyn(),
        );
        for (step, time, value) in [(0u64, 0.0, 1.0), (1, 0.5, 2.0)] {
            let name = FieldKey::new("T", step, time).name();
            set_field(
                &mut mesh,
                ElementType::HEX8,
                &name,
                nd::arr1(&[value]).into_dyn(),
            );
        }
        let mut wall = ElementIds::new();
        wall.add_block(ElementType::HEX8, vec![0]);
        mesh.add_to_group("wall", &wall);
        mesh
    }

    fn make_2d_mesh() -> UMesh {
        let mut mesh = me::make_mesh_2d_quad();
        set_field(
            &mut mesh,
            ElementType::QUAD4,
            "density",
            nd::arr1(&[5.0]).into_dyn(),
        );
        let mut boundary = ElementIds::new();
        boundary.add_block(ElementType::QUAD4, vec![0]);
        mesh.add_to_group("boundary", &boundary);
        mesh
    }

    fn make_container() -> MedFile {
        MedFile {
            meshes: BTreeMap::from([
                ("fluid".to_string(), make_3d_mesh()),
                ("solid".to_string(), make_2d_mesh()),
            ]),
        }
    }

    fn write_temp(container: &MedFile) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("container.med");
        container.write(&path, MedWriteMode::Overwrite).unwrap();
        (dir, path)
    }

    #[test]
    fn inspect_reports_meshes_fields_and_steps() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let structure = MedFile::inspect(&path).unwrap();

        let fluid = &structure.meshes["fluid"];
        assert_eq!(fluid.space_dimension, 3);
        assert_eq!(fluid.num_nodes, 8);
        assert_eq!(fluid.num_cells, 1);
        assert_eq!(
            fluid.element_types,
            BTreeMap::from([("HE8".to_string(), 1)])
        );
        assert!(fluid.unsupported_element_types.is_empty());
        assert_eq!(fluid.groups, vec!["wall".to_string()]);

        let solid = &structure.meshes["solid"];
        assert_eq!(solid.space_dimension, 2);
        assert_eq!(solid.num_nodes, 4);
        assert_eq!(solid.groups, vec!["boundary".to_string()]);

        let temperature = &structure.fields["T"];
        assert_eq!(temperature.mesh_name, "fluid");
        assert_eq!(temperature.dimension, Some(Dimension::D3));
        assert_eq!(temperature.components, 1);
        assert_eq!(temperature.steps.len(), 2);
        assert_eq!(temperature.steps[0].iteration, 0);
        assert_eq!(temperature.steps[1].time, 0.5);

        let velocity = &structure.fields["velocity"];
        assert_eq!(velocity.components, 2);
        assert_eq!(velocity.steps.len(), 1);

        let density = &structure.fields["density"];
        assert_eq!(density.mesh_name, "solid");
        assert_eq!(density.dimension, Some(Dimension::D2));
    }

    #[test]
    fn read_all_meshes_and_fields() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let loaded = MedFile::read(&path, None).unwrap();
        assert_eq!(loaded.meshes.keys().collect::<Vec<_>>(), ["fluid", "solid"]);

        let fluid = &loaded.meshes["fluid"];
        assert_eq!(fluid.num_elements(), 1);
        assert!(fluid.has_group("wall"));
        let pressure = fluid.field("pressure", Some(Dimension::D3)).unwrap();
        assert_eq!(pressure.0[&ElementType::HEX8], nd::arr1(&[3.0]).into_dyn());
        let velocity = fluid.field("velocity", Some(Dimension::D3)).unwrap();
        assert_eq!(
            velocity.0[&ElementType::HEX8],
            nd::arr2(&[[1.0, 2.0]]).into_dyn()
        );

        let (_, steps) = fluid.resolve_time_series("T", Some(Dimension::D3)).unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].time, 0.0);
        assert_eq!(steps[1].time, 0.5);
    }

    #[test]
    fn read_selection_selects_meshes_fields_and_steps() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let selection = MedSelection {
            meshes: BTreeMap::from([(
                "fluid".to_string(),
                MeshSelection::Fields(BTreeMap::from([
                    ("T".to_string(), StepSelection::Index(-1)),
                    ("pressure".to_string(), StepSelection::All),
                ])),
            )]),
        };
        let loaded = MedFile::read(&path, Some(&selection)).unwrap();
        assert_eq!(loaded.meshes.keys().collect::<Vec<_>>(), ["fluid"]);
        let fluid = &loaded.meshes["fluid"];
        assert!(fluid.field("pressure", Some(Dimension::D3)).is_some());
        assert!(fluid.field("velocity", Some(Dimension::D3)).is_none());
        let (_, steps) = fluid.resolve_time_series("T", Some(Dimension::D3)).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].time, 0.5);

        let selection = MedSelection {
            meshes: BTreeMap::from([("fluid".to_string(), MeshSelection::Fields(BTreeMap::new()))]),
        };
        let loaded = MedFile::read(&path, Some(&selection)).unwrap();
        let fluid = &loaded.meshes["fluid"];
        assert_eq!(fluid.num_elements(), 1);
        assert!(fluid.has_group("wall"));
        assert_eq!(fluid.fields().count(), 0);

        let selection = MedSelection {
            meshes: BTreeMap::from([(
                "fluid".to_string(),
                MeshSelection::Fields(BTreeMap::from([(
                    "T".to_string(),
                    StepSelection::Indices(vec![-2, -1]),
                )])),
            )]),
        };
        let loaded = MedFile::read(&path, Some(&selection)).unwrap();
        let (_, steps) = loaded.meshes["fluid"]
            .resolve_time_series("T", Some(Dimension::D3))
            .unwrap();
        assert_eq!(steps.len(), 2);

        let loaded = MedFile::read(&path, Some(&MedSelection::default())).unwrap();
        assert!(loaded.meshes.is_empty());
    }

    #[test]
    fn read_rejects_invalid_selections() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let unknown_mesh = MedSelection {
            meshes: BTreeMap::from([("nope".to_string(), MeshSelection::All)]),
        };
        assert!(MedFile::read(&path, Some(&unknown_mesh)).is_err());

        let unknown_field = MedSelection {
            meshes: BTreeMap::from([(
                "fluid".to_string(),
                MeshSelection::Fields(BTreeMap::from([(
                    "density".to_string(),
                    StepSelection::All,
                )])),
            )]),
        };
        assert!(MedFile::read(&path, Some(&unknown_field)).is_err());

        let wrong_mesh = MedSelection {
            meshes: BTreeMap::from([(
                "solid".to_string(),
                MeshSelection::Fields(BTreeMap::from([(
                    "density".to_string(),
                    StepSelection::Index(1),
                )])),
            )]),
        };
        assert!(MedFile::read(&path, Some(&wrong_mesh)).is_err());
    }

    #[test]
    fn read_mesh_loads_one_mesh() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let fluid = MedFile::read_mesh(&path, "fluid", None).unwrap();
        assert_eq!(fluid.num_elements(), 1);
        assert!(fluid.field("pressure", Some(Dimension::D3)).is_some());

        let solid_only = MedFile::read_mesh(
            &path,
            "solid",
            Some(&MeshSelection::Fields(BTreeMap::new())),
        )
        .unwrap();
        assert_eq!(solid_only.num_elements(), 1);
        assert_eq!(solid_only.fields().count(), 0);
        assert!(solid_only.has_group("boundary"));
    }

    #[test]
    fn update_writes_only_loaded_entries() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        let selection = MedSelection {
            meshes: BTreeMap::from([(
                "solid".to_string(),
                MeshSelection::Fields(BTreeMap::from([(
                    "density".to_string(),
                    StepSelection::All,
                )])),
            )]),
        };
        let mut loaded = MedFile::read(&path, Some(&selection)).unwrap();
        let solid = loaded.meshes.get_mut("solid").unwrap();
        set_field(
            solid,
            ElementType::QUAD4,
            "density",
            nd::arr1(&[9.0]).into_dyn(),
        );
        loaded.write(&path, MedWriteMode::Update).unwrap();

        let reread = MedFile::read(&path, None).unwrap();
        let density = reread.meshes["solid"]
            .field("density", Some(Dimension::D2))
            .unwrap();
        assert_eq!(density.0[&ElementType::QUAD4], nd::arr1(&[9.0]).into_dyn());
        let fluid = &reread.meshes["fluid"];
        assert!(fluid.field("pressure", Some(Dimension::D3)).is_some());
        let (_, steps) = fluid.resolve_time_series("T", Some(Dimension::D3)).unwrap();
        assert_eq!(steps.len(), 2);
    }

    #[test]
    fn update_creates_missing_file() {
        let _guard = hdf5_test_guard();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("new.med");
        let container = make_container();
        container.write(&path, MedWriteMode::Update).unwrap();

        let reread = MedFile::read(&path, None).unwrap();
        assert_eq!(reread.meshes.len(), 2);
    }

    #[test]
    fn create_refuses_existing_file() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);
        assert!(container.write(&path, MedWriteMode::Create).is_err());
    }

    #[test]
    fn duplicate_field_names_are_rejected() {
        let _guard = hdf5_test_guard();
        let mut container = make_container();
        let solid = container.meshes.get_mut("solid").unwrap();
        set_field(
            solid,
            ElementType::QUAD4,
            "pressure",
            nd::arr1(&[7.0]).into_dyn(),
        );
        let dir = tempfile::tempdir().unwrap();
        let err = container
            .write(dir.path().join("dup.med"), MedWriteMode::Overwrite)
            .unwrap_err();
        assert!(err.to_string().contains("file-global"), "{err}");
    }

    #[test]
    fn duplicate_field_name_across_dimensions_is_rejected() {
        let _guard = hdf5_test_guard();
        let mut mesh = me::make_mesh_2d_quad();
        mesh.add_regular_block(ElementType::SEG2, nd::arr2(&[[0, 1]]).to_shared(), None);
        set_field(
            &mut mesh,
            ElementType::QUAD4,
            "p",
            nd::arr1(&[1.0]).into_dyn(),
        );
        set_field(
            &mut mesh,
            ElementType::SEG2,
            "p",
            nd::arr1(&[1.0, 2.0]).into_dyn(),
        );
        let container = MedFile {
            meshes: BTreeMap::from([("mixed".to_string(), mesh)]),
        };
        let dir = tempfile::tempdir().unwrap();
        let err = container
            .write(dir.path().join("dup_dim.med"), MedWriteMode::Overwrite)
            .unwrap_err();
        assert!(err.to_string().contains("dimensions"), "{err}");
    }

    #[test]
    fn inspect_and_read_report_unsupported_element_types() {
        let _guard = hdf5_test_guard();
        let container = make_container();
        let (_dir, path) = write_temp(&container);

        {
            let file = File::open_rw(&path).unwrap();
            let med_mesh = file.group("ENS_MAA").unwrap().group("fluid").unwrap();
            let step_name = med_mesh.member_names().unwrap()[0].clone();
            let mai = med_mesh.group(&step_name).unwrap().group("MAI").unwrap();
            let pyramid = mai.create_group("PY5").unwrap();
            let nod: nd::Array1<i64> = nd::arr1(&[1, 2, 3, 4, 5]);
            let dataset = pyramid
                .new_dataset_builder()
                .with_data(&nod)
                .create("NOD")
                .unwrap();
            super::super::med_io::write_scalar_attr(&pyramid, "CGT", 1i64).unwrap();
            super::super::med_io::write_scalar_attr(&dataset, "CGT", 1i64).unwrap();
            super::super::med_io::write_scalar_attr(&dataset, "NBR", 1i64).unwrap();
        }

        let structure = MedFile::inspect(&path).unwrap();
        let fluid = &structure.meshes["fluid"];
        assert_eq!(fluid.unsupported_element_types, vec!["PY5".to_string()]);
        assert_eq!(fluid.num_cells, 2);

        let loaded = MedFile::read(&path, None).unwrap();
        assert_eq!(loaded.meshes["fluid"].num_elements(), 1);
    }
}
