//! Mesh manipulation tools and algorithms.
//!
//! This module provides various utilities for mesh operations including:
//! - Connected component analysis
//! - Mesh cracking (splitting shared nodes/faces)
//! - Mesh extrusion (raising dimension)
//! - Field expressions and evaluation
//! - Structured grid generation
//! - Mesh overlay operations
//! - Geometric measurements
//! - Neighbor computation
//! - Element selection
//! - Node snapping

/// Centroids of meshes.
pub mod centroids;
/// Connected component analysis for meshes.
pub mod connected_components;
/// Crack along shared faces/nodes to separate mesh regions.
///
/// # Entrée
///
/// - soit des couples d'id de cellule, soit un maillage de faces
/// - soit la consigne est de vérifier que la séparation est possible, soit on continue sans séparer
///
/// # Identification des faces/noeuds
///
/// - on identifie tous les couples cellule-cellule à séparer (facile, SortedNodes)
/// - on identifie tous les noeuds appartenant à la frontière
///
/// # Parcours des noeuds à la frontière
///
/// - on construit le sous maillage qui contient les cellules adjacentes
/// - on construit le graph c2c de ce maillage
/// - on coupe les arrêtes qui sont dans la liste des arrêtes à séparer
/// - on calcule le nombre de compo connexe du petit graphe
/// - on duplique le noeud autant de fois qu'il y a de compo connexe, on le remplace dans chaque
///   compo connexe par sa nouvelle valeur
/// - on créé un vecteur de tuples pour marquer le remplacement
///
/// # Elements de dimension inférieure
///
/// - pour tous les noeuds dupliqués je récupère les éléments de dimension inférieure
pub mod crack;
/// Mesh extrusion to build a higher-dimensional mesh.
///
/// This module builds a mesh of one dimension higher than the input mesh by extruding it.
/// Duplicated nodes are allowed, both in the original mesh and the 1d mesh.
pub mod extrude;
/// Field expression evaluation and manipulation.
pub mod fieldexpr;
/// Structured grid generation utilities.
pub mod grid;
/// Geometric measurement utilities for meshes.
pub mod measure;
/// Neighbor computation for mesh elements.
pub mod neighbours;
/// Boolean-like overlay operations on 2D meshes.
pub mod overlay;
/// Element and node selection utilities.
pub mod selector;
/// Node snapping to merge nearby nodes.
pub mod snap;
pub mod spatial_index;

pub use centroids::*;
pub use connected_components::*;
pub use crack::*;
pub use extrude::*;
pub use grid::*;
pub use measure::*;
pub use neighbours::*;
pub use overlay::*;
pub use selector::*;
pub use snap::*;
