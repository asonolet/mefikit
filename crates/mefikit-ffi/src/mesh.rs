//! `UMesh` methods: building a mesh from C++ and reading data back out of it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use ndarray as nd;
use ndarray::IxDyn;

use mefikit::mesh::ElementType as CoreElementType;
use mefikit::prelude as mf;

use crate::ffi::bridge::{Dimension, ElementType, FieldBlock, FieldInfo};
use crate::ffi::{Error, UMesh};

/// A field array as stored inside an owned `mefikit::UMesh`: contiguous and
/// reference counted, which is what makes the zero-copy `field_values` possible.
type OwnedField = nd::ArcArray<f64, IxDyn>;

impl UMesh {
    /// Creates an empty mesh from a row-major `(n_nodes, space_dim)` array.
    ///
    /// `coords.len()` must be exactly `n_nodes * space_dim`.
    pub fn from_coords(
        coords: &[f64],
        n_nodes: usize,
        space_dim: usize,
    ) -> Result<Box<Self>, Error> {
        let array = nd::Array2::from_shape_vec((n_nodes, space_dim), coords.to_vec())?;
        Ok(Box::new(Self(mf::UMesh::new(array.into_shared()))))
    }

    /// Adds a fixed-node-count block from a row-major
    /// `(n_elements, num_nodes(element_type))` connectivity table.
    pub fn add_regular_block(
        &mut self,
        element_type: ElementType,
        conn: &[usize],
        n_elements: usize,
    ) -> Result<(), Error> {
        let element_type = element_type.to_core()?;
        let nodes_per_element = element_type.num_nodes().ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{element_type:?} has a variable node count, use add_poly_block"
            ))
        })?;
        let conn = nd::Array2::from_shape_vec((n_elements, nodes_per_element), conn.to_vec())?;
        self.0
            .add_regular_block(element_type, conn.into_shared(), None);
        Ok(())
    }

    /// Adds a variable-node-count block: a flat `conn` node list plus one
    /// cumulative end index per element in `offsets`.
    pub fn add_poly_block(
        &mut self,
        element_type: ElementType,
        conn: &[usize],
        offsets: &[usize],
    ) -> Result<(), Error> {
        self.0.add_poly_block(
            element_type.to_core()?,
            nd::Array1::from(conn.to_vec()).into_shared(),
            nd::Array1::from(offsets.to_vec()).into_shared(),
            None,
        );
        Ok(())
    }

    /// Checks the mesh is internally consistent.
    pub fn validate_structure(&self) -> Result<(), Error> {
        self.0.validate_structure().map_err(Error::InvalidArgument)
    }

    /// Number of nodes.
    #[must_use]
    pub fn n_nodes(&self) -> usize {
        self.0.coords().nrows()
    }

    /// Total number of elements, over all element types.
    #[must_use]
    pub fn n_elements(&self) -> usize {
        self.0.num_elements()
    }

    /// Number of elements carried by one element type, 0 if it has no block.
    #[must_use]
    pub fn n_elements_of(&self, element_type: ElementType) -> usize {
        // An element type this build does not know about trivially has no block.
        element_type
            .to_core()
            .ok()
            .and_then(|core| self.0.block(core))
            .map_or(0, |block| block.len())
    }

    /// Dimensionality of the coordinates: 1, 2 or 3.
    #[must_use]
    pub fn space_dimension(&self) -> usize {
        self.0.space_dimension()
    }

    /// True for a mesh with no element block yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.topological_dimension().is_none()
    }

    /// Highest topological dimension present, `D0` for an empty mesh.
    #[must_use]
    pub fn topological_dimension(&self) -> Dimension {
        self.0
            .topological_dimension()
            .map_or(Dimension::D0, Dimension::from)
    }

    /// Element types present, in ascending order.
    #[must_use]
    pub fn element_types(&self) -> Vec<ElementType> {
        self.0
            .element_types()
            .map(|e| ElementType::from(*e))
            .collect()
    }

    /// Sets the field `name` from one block per element type.
    ///
    /// Every block must sit at the same topological dimension, and the set of
    /// element types supplied must be exactly the mesh's blocks at that
    /// dimension — `mefikit` fields are per-dimension, not per-element-type.
    pub fn set_field(
        &mut self,
        name: &str,
        blocks: &[FieldBlock],
        values: &[f64],
    ) -> Result<(), Error> {
        let first = blocks.first().ok_or_else(|| {
            Error::InvalidArgument("set_field needs at least one field block".to_owned())
        })?;

        let dimension = first.element_type.to_core()?.dimension();
        for block in blocks {
            let element_type = block.element_type.to_core()?;
            if element_type.dimension() != dimension {
                return Err(Error::InvalidArgument(format!(
                    "field '{name}' mixes dimensions: {element_type:?} is D{} but the first \
                     block is D{}",
                    u8::from(element_type.dimension()),
                    u8::from(dimension)
                )));
            }
        }

        let present: BTreeSet<CoreElementType> = self
            .0
            .element_types()
            .filter(|e| e.dimension() == dimension)
            .copied()
            .collect();
        let provided: BTreeSet<CoreElementType> = blocks
            .iter()
            .map(|b| b.element_type.to_core())
            .collect::<Result<_, _>>()?;
        if present != provided {
            return Err(Error::InvalidArgument(format!(
                "field '{name}' covers {provided:?} but the mesh has {present:?} at dimension {}",
                u8::from(dimension)
            )));
        }

        let mut map: BTreeMap<CoreElementType, OwnedField> = BTreeMap::new();
        for block in blocks {
            let element_type = block.element_type.to_core()?;
            if block.n_components == 0 {
                return Err(Error::InvalidArgument(format!(
                    "field '{name}' on {element_type:?} declares 0 components"
                )));
            }
            let n_elements = self.n_elements_of(block.element_type);
            let expected = n_elements * block.n_components;
            if block.len != expected {
                return Err(Error::InvalidArgument(format!(
                    "field '{name}' on {element_type:?} supplies {} values but {} elements x {} \
                     components were expected",
                    block.len, n_elements, block.n_components
                )));
            }
            let end = block.offset.checked_add(block.len).ok_or_else(|| {
                Error::InvalidArgument(format!("field '{name}' block range overflows"))
            })?;
            let data = values.get(block.offset..end).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "field '{name}' block reads values[{}..{end}] but only {} were given",
                    block.offset,
                    values.len()
                ))
            })?;

            let shape = IxDyn(&[n_elements, block.n_components]);
            map.insert(
                element_type,
                nd::Array::from_shape_vec(shape, data.to_vec())?.into_shared(),
            );
        }

        self.0.update_field(name, mf::FieldArcD::new(map));
        Ok(())
    }

    /// Convenience wrapper around [`set_field`](Self::set_field) for meshes with
    /// a single block at the relevant dimension.
    pub fn set_field_uniform(
        &mut self,
        name: &str,
        element_type: ElementType,
        n_components: usize,
        values: &[f64],
    ) -> Result<(), Error> {
        let core_type = element_type.to_core()?;
        if self.0.block(core_type).is_none() {
            return Err(Error::InvalidArgument(format!(
                "mesh has no {element_type:?} block"
            )));
        }
        let dimension = core_type.dimension();
        let blocks_at_dim = self
            .0
            .element_types()
            .filter(|e| e.dimension() == dimension)
            .count();
        if blocks_at_dim != 1 {
            return Err(Error::InvalidArgument(format!(
                "set_field_uniform needs a single block at dimension {} but the mesh has \
                 {blocks_at_dim}; use set_field instead",
                u8::from(dimension)
            )));
        }

        let block = FieldBlock {
            element_type,
            n_components,
            offset: 0,
            len: values.len(),
        };
        self.set_field(name, std::slice::from_ref(&block), values)
    }

    /// Shape of field `name` on element type `element_type`.
    pub fn field_info(&self, name: &str, element_type: ElementType) -> Result<FieldInfo, Error> {
        let array = self.field_array(name, element_type)?;
        let (n_elements, n_components) = if array.ndim() == 0 {
            (1, 1)
        } else {
            (array.shape()[0], array.shape()[1..].iter().product())
        };
        Ok(FieldInfo {
            n_elements,
            n_components,
        })
    }

    /// Zero-copy row-major view of field `name` on element type `element_type`.
    ///
    /// # Safety
    ///
    /// The returned slice aliases the mesh's own storage. The caller must keep
    /// the mesh alive for as long as it uses the slice, and must not call any
    /// method taking `&mut self` in the meantime.
    pub unsafe fn field_values<'a>(
        &'a self,
        name: &str,
        element_type: ElementType,
    ) -> Result<&'a [f64], Error> {
        let array = self.field_array(name, element_type)?;
        array
            .as_slice()
            .ok_or_else(|| Error::InvalidArgument(format!("field '{name}' is not contiguous")))
    }

    /// Names of every field carried by the mesh, sorted.
    #[must_use]
    pub fn field_names(&self) -> Vec<String> {
        self.0
            .fields()
            .map(|(name, _)| name)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Reads a mesh. The format follows the file extension.
    pub fn read(path: &str) -> Result<Box<Self>, Error> {
        Ok(Box::new(Self(mf::read(Path::new(path))?)))
    }

    /// Writes a mesh, in the format matching the file extension.
    pub fn write(&self, path: &str) -> Result<(), Error> {
        mf::write(Path::new(path), self.0.view()).map_err(Error::Io)
    }

    fn field_array(&self, name: &str, element_type: ElementType) -> Result<&OwnedField, Error> {
        self.0
            .block(element_type.to_core()?)
            .ok_or_else(|| Error::InvalidArgument(format!("mesh has no {element_type:?} block")))?
            .fields
            .get(name)
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "mesh has no field named '{name}' on {element_type:?}"
                ))
            })
    }
}
