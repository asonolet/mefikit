pub mod connected_components;
/// Crack along
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
/// This module builds a mesh of one dimension higher than the input mesh by extuding it.
/// Duplicated nodes are allowed, both in the original mesh and the 1d mesh.
pub mod extrude;
pub mod fieldexpr;
pub mod grid;
/// Module for intersecting meshes.
///
/// In this context, intersections operations can be separated in the following cases:
/// - 1d + 1d
///   - cut: Intersection of 1D elements with other 1D elements producing 0D elements.
///   - cut_add: Intersecting 1D elements with 1D elements producing 1D mesh conformized to both
///     meshes.
/// - 2d + 1d
///   - cut_edges: Intersecting 2D elements with 1D elements and producing 2D mesh which contains
///     intersections nodes in the connectivity. This new 2D mesh has the same number of elements as
///     the original 2D mesh.
///   - cute_faces: Intersecting 2D elements with 1D elements and producing 2D mesh by cutting the
///     2D mesh with the 1D mesh. The new 2D mesh may contains more elements, and conformizes with
///     the 1D mesh.
/// - 2d + 2d
///   - cut_union: Intersecting both meshes fully.
///   - cut_intersect: Only keeping the domain covered by both meshes and intersecting them.
///   - cut_xor: Only keeping the domain covered by one of the meshes and intersecting them.
///
/// The input meshes do not need to be clean (i.e. they can have unmerged nodes). They need to be
/// conformed (i.e. no overlapping elements).
/// In all cases, the operation gives a "conformized without merging nodes" mesh. The user can
/// choose to merge nodes after the operation if needed.
///
/// Note: The implementation of these operations is not trivial. The main difficulty is to
/// manage non conformities and numerical precision issues. The implementation should be robust
/// and handle these issues gracefully.
pub mod intersect;
pub mod measure;
pub mod neighbours;
pub mod selector;
pub mod snap;

pub use connected_components::*;
pub use crack::*;
pub use extrude::*;
pub use grid::*;
pub use measure::*;
pub use neighbours::*;
pub use selector::*;
pub use snap::*;
