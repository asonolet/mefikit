//! `UMesh` construction and field access for C++.

use std::collections::{BTreeMap, BTreeSet};

use cxx::CxxVector;
use ndarray as nd;

use mefikit::mesh::ElementType as CoreElementType;
use mefikit::prelude as mf;

use crate::ffi::bridge::ElementType;
use crate::ffi::{Error, UMesh};

impl UMesh {
    /// Creates an empty mesh from row-major `(n_nodes, space_dim)` coordinates.
    pub fn from_coords(
        coords: &CxxVector<f64>,
        n_nodes: usize,
        space_dim: usize,
    ) -> Result<Box<Self>, Error> {
        if !(1..=3).contains(&space_dim) {
            return Err(Error::InvalidArgument(format!(
                "space_dim must be 1, 2 or 3, got {space_dim}"
            )));
        }
        let expected = n_nodes
            .checked_mul(space_dim)
            .ok_or_else(|| Error::InvalidArgument("coordinate count overflows usize".to_owned()))?;
        if coords.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "coordinates hold {} values but n_nodes * space_dim = {expected}",
                coords.len()
            )));
        }
        let array = nd::Array2::from_shape_vec((n_nodes, space_dim), coords.as_slice().to_vec())?;
        Ok(Box::new(Self {
            mesh: mf::UMesh::new(array.into_shared()),
            pending_fields: BTreeMap::new(),
        }))
    }

    /// Adds a fixed-node-count element block.
    pub fn add_regular_block(
        &mut self,
        element_type: ElementType,
        conn: &CxxVector<usize>,
        n_elements: usize,
    ) -> Result<(), Error> {
        let element_type = element_type.to_core()?;
        let n_nodes = element_type.num_nodes().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{element_type:?} has a variable node count; use add_poly_block"
            ))
        })?;
        self.reject_duplicate_block(element_type)?;
        let expected = n_elements.checked_mul(n_nodes).ok_or_else(|| {
            Error::InvalidArgument("connectivity size overflows usize".to_owned())
        })?;
        if conn.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "{element_type:?} connectivity holds {} indices but {n_elements} elements x \
                 {n_nodes} nodes = {expected}",
                conn.len()
            )));
        }
        let conn = nd::Array2::from_shape_vec((n_elements, n_nodes), conn.as_slice().to_vec())?;
        self.mesh
            .add_regular_block(element_type, conn.into_shared(), None);
        Ok(())
    }

    /// Adds a variable-node-count element block.
    pub fn add_poly_block(
        &mut self,
        element_type: ElementType,
        conn: &CxxVector<usize>,
        offsets: &CxxVector<usize>,
    ) -> Result<(), Error> {
        let element_type = element_type.to_core()?;
        if element_type.regularity() != mf::Regularity::Poly {
            return Err(Error::InvalidArgument(format!(
                "{element_type:?} has a fixed node count; use add_regular_block"
            )));
        }
        self.reject_duplicate_block(element_type)?;
        if offsets.is_empty() {
            return Err(Error::InvalidArgument(
                "a poly block needs at least one offset".to_owned(),
            ));
        }
        let offsets = offsets.as_slice();
        if offsets[offsets.len() - 1] != conn.len() {
            return Err(Error::InvalidArgument(format!(
                "the last poly offset is {} but the connectivity holds {} indices",
                offsets[offsets.len() - 1],
                conn.len()
            )));
        }
        let conn = nd::Array1::from_vec(conn.as_slice().to_vec()).into_shared();
        let offsets = nd::Array1::from_vec(offsets.to_vec()).into_shared();
        self.mesh.add_poly_block(element_type, conn, offsets, None);
        Ok(())
    }

    /// Sets field `name` on one element type, keeping the other blocks of that
    /// field at the same topological dimension.
    pub fn set_field(
        &mut self,
        name: &str,
        element_type: ElementType,
        n_components: usize,
        values: &CxxVector<f64>,
    ) -> Result<(), Error> {
        if name.is_empty() {
            return Err(Error::InvalidArgument(
                "a field name may not be empty".to_owned(),
            ));
        }
        if n_components == 0 {
            return Err(Error::InvalidArgument(format!(
                "field '{name}' declares 0 components"
            )));
        }
        let element_type = element_type.to_core()?;
        let n_elements = self
            .mesh
            .block(element_type)
            .ok_or_else(|| Error::InvalidArgument(format!("mesh has no {element_type:?} block")))?
            .len();
        let expected = n_elements.checked_mul(n_components).ok_or_else(|| {
            Error::InvalidArgument(format!("field '{name}' size overflows usize"))
        })?;
        if values.len() != expected {
            return Err(Error::InvalidArgument(format!(
                "field '{name}' on {element_type:?} holds {} values but {n_elements} elements x \
                 {n_components} components = {expected}",
                values.len()
            )));
        }

        let array = nd::Array::from_shape_vec(
            nd::IxDyn(&[n_elements, n_components]),
            values.as_slice().to_vec(),
        )?
        .into_shared();
        let dimension = element_type.dimension();
        let trailing_shape = array.shape()[1..].to_vec();

        let mut blocks: BTreeMap<CoreElementType, nd::ArcArray<f64, nd::IxDyn>> = BTreeMap::new();
        if let Some(existing) = self.mesh.field(name, Some(dimension)) {
            for (other_type, other_array) in existing.0 {
                if other_array.shape()[1..] != trailing_shape {
                    return Err(Error::InvalidArgument(format!(
                        "field '{name}' has {} components on {other_type:?} but \
                         {n_components} on {element_type:?}",
                        other_array.shape()[1..].iter().product::<usize>()
                    )));
                }
                blocks.insert(other_type, other_array.to_owned().into_shared());
            }
        }
        if let Some(pending) = self.pending_fields.get(name) {
            for (other_type, other_array) in pending {
                if other_type.dimension() != dimension {
                    continue;
                }
                if other_array.shape()[1..] != trailing_shape {
                    return Err(Error::InvalidArgument(format!(
                        "field '{name}' has {} components on {other_type:?} but \
                         {n_components} on {element_type:?}",
                        other_array.shape()[1..].iter().product::<usize>()
                    )));
                }
                blocks.insert(*other_type, other_array.clone());
            }
        }
        blocks.insert(element_type, array);

        let expected: BTreeSet<CoreElementType> = self
            .mesh
            .element_types()
            .filter(|other| other.dimension() == dimension)
            .copied()
            .collect();
        let provided: BTreeSet<CoreElementType> = blocks.keys().copied().collect();
        if provided == expected {
            self.mesh.update_field(name, mf::FieldArcD::new(blocks));
            self.pending_fields.remove(name);
        } else {
            self.pending_fields.insert(name.to_owned(), blocks);
        }
        Ok(())
    }

    /// Returns field `name` on one element type, row-major.
    pub fn field_values(&self, name: &str, element_type: ElementType) -> Result<Vec<f64>, Error> {
        let element_type = element_type.to_core()?;
        let block = self
            .mesh
            .block(element_type)
            .ok_or_else(|| Error::InvalidArgument(format!("mesh has no {element_type:?} block")))?;
        let array = block.fields.get(name).ok_or_else(|| {
            Error::InvalidArgument(format!(
                "mesh has no field named '{name}' on {element_type:?}"
            ))
        })?;
        Ok(array.iter().copied().collect())
    }

    /// Number of cells carried by one element type.
    #[must_use]
    pub fn num_elements(&self, element_type: ElementType) -> usize {
        element_type
            .to_core()
            .ok()
            .and_then(|element_type| self.mesh.block(element_type))
            .map_or(0, |block| block.len())
    }

    fn reject_duplicate_block(&self, element_type: CoreElementType) -> Result<(), Error> {
        if self.mesh.block(element_type).is_some() {
            return Err(Error::InvalidArgument(format!(
                "the mesh already has a {element_type:?} block"
            )));
        }
        Ok(())
    }
}
