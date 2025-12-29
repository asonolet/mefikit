/// This algorithm duplicates some nodes in order to break connectivites between some cells.
/// Crack along

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
/// - on duplique le noeud autant de fois qu'il y a de compo connexe, on le remplace dans chaque compo connexe par sa nouvelle valeur
/// - on créé un vecteur de tuples pour marquer le remplacement
/// 
/// # Elements de dimension inférieure
/// 
/// - pour tous les noeuds dupliqués je récupère les éléments de dimension inférieure

use crate::mesh::{UMesh, UMeshView};
use crate::tools::neighbours::compute_sub_to_elem;


pub fn crack(mesh: UMesh, cut: UMeshView) -> UMesh {
    // First extract the vicinity of the cut
    let nodes = cut.used_nodes();
    let near_mesh = Selector::new(&mesh).nodes(false).id_in(nodes.as_slice()).select();
    // o(mesh cells), it is not fully optimal. In best case I should be able to only consider the vicinity of the crack as it is the only part I am interested in.
    let (submesh, f2c) = compute_sub_to_elem(&near_mesh, None, None);
    let cut_ids = find_equals(submesh.view(), cut.view());
    
}