use super::umesh::Dimension::*;
use super::umesh::{Element, ElementId, ElementLike};
use super::umesh::{UMesh, UMeshView};

use std::collections::HashMap;

use ndarray::prelude::*;
use rstar::AABB;
use rstar::primitives::{GeomWithData, Line};

struct Segment(GeomWithData<Line<[f64; 2]>, (ElementId, UndirectedEdge)>);

impl Segment {
    pub fn new(p1: [f64; 2], p2: [f64; 2], eid: ElementId, n1: usize, n2: usize) -> Self {
        Self(GeomWithData::new(
            Line { from: p1, to: p2 },
            (eid, UndirectedEdge::new(n1, n2)),
        ))
    }
    fn from(el: &Element) -> Self {
        let p1: &[f64] = el.coords().index_axis(Axis(0), 0).to_slice().unwrap();
        let p2: &[f64] = el.coords().index_axis(Axis(0), 1).to_slice().unwrap();
        Self::new(
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            el.id(),
            el.connectivity[0],
            el.connectivity[1],
        )
    }
}

fn to_aabb(el: &Element) -> AABB<[f64; 2]> {
    todo!()
}

fn intersect_seg_seg(seg1: &Element, seg2: &Element) -> Option<[f64; 2]> {
    let p1 = [
        *seg1.coords().get((0, 0)).unwrap(),
        *seg1.coords().get((0, 1)).unwrap(),
    ];
    let p2 = [
        *seg1.coords().get((1, 0)).unwrap(),
        *seg1.coords().get((1, 1)).unwrap(),
    ];
    let p3 = [
        *seg2.coords().get((0, 0)).unwrap(),
        *seg2.coords().get((0, 1)).unwrap(),
    ];
    let p4 = [
        *seg2.coords().get((1, 0)).unwrap(),
        *seg2.coords().get((1, 1)).unwrap(),
    ];
    todo!()
}

fn colinear_seg_seg(seg1: &Element, seg2: &Element) -> Option<([f64; 2], [f64; 2])> {
    let p1 = [
        *seg1.coords().get((0, 0)).unwrap(),
        *seg1.coords().get((0, 1)).unwrap(),
    ];
    let p2 = [
        *seg1.coords().get((1, 0)).unwrap(),
        *seg1.coords().get((1, 1)).unwrap(),
    ];
    let p3 = [
        *seg2.coords().get((0, 0)).unwrap(),
        *seg2.coords().get((0, 1)).unwrap(),
    ];
    let p4 = [
        *seg2.coords().get((1, 0)).unwrap(),
        *seg2.coords().get((1, 1)).unwrap(),
    ];
    todo!()
}

enum Intersection {
    /// The point variant corresponds to the case when two segments intersects in one point. This
    /// point can be an edge point.
    Point(ElementId, usize),

    /// This variant corresponds to the case when two segments share a common part with non null
    /// length because they are colinear. The two points returned corresponds to this common
    /// segment. Those points can correspond to edge points of one or both segments.
    Segment(ElementId, usize, usize),
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct DirectedEdge(usize, usize);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
struct UndirectedEdge(usize, usize);

impl UndirectedEdge {
    fn new(a: usize, b: usize) -> Self {
        if a < b { Self(a, b) } else { Self(b, a) }
    }
}

impl From<DirectedEdge> for UndirectedEdge {
    fn from(e: DirectedEdge) -> Self {
        Self::new(e.0, e.1)
    }
}

// impl From<UndirectedEdge> for DirectedEdge {
//     fn from(e: UndirectedEdge) -> Self {
//         Self(e.0, e.1)
//     }
// }

/// Cette méthode permet de découper un maillage 2d potentiellement non conforme avec un maillage
/// de segments propres (sans noeuds non fusionnés).
pub fn intersect_2dmesh_1dtool_mesh(mesh: UMeshView, tool_mesh: UMesh) -> UMesh {
    // tool_mesh.merge_nodes();
    // tool_mesh.prepend_coords(mesh.coordinates().clone());

    // build R*-tree with 1d tool mesh
    let segs: Vec<_> = tool_mesh
        .elements()
        .map(|seg| Segment::from(&seg).0)
        .collect();
    let tree = rstar::RTree::bulk_load(segs);
    // find intersections and create two maps: polygon map with new intersection points and segments map
    // There is a little preparation on tool_mesh that should be done:
    // - nodes should be merged.
    // - coords from mesh should be prepended
    // Maintenant je travaille uniquement en terme de connectivité, j'utilise la table de
    // coordonnées suivante pour les identifiants de noeuds:
    // Table de mesh + table de tool_mesh après merge + nouveaux noeuds différents
    // Je créé la liste des segments de tool mesh découpés (avec les noeuds d'intersection)
    let mut e2int: HashMap<UndirectedEdge, Vec<Intersection>> = HashMap::new();
    for el in mesh.elements() {
        let segs_in_elem: Vec<_> = tree
            .locate_in_envelope_intersecting(&to_aabb(&el))
            .collect();
        for (_, edge) in el.subentities(None).unwrap() {
            // TODO: edge could be SEG3!
            let edge = DirectedEdge(edge[0], edge[1]);
            let uedge = UndirectedEdge::from(edge.clone());
            let intersections: &_ = e2int.entry(uedge).or_insert_with(|| {
                for seg in &segs_in_elem {
                    // Calcul des intersections avec edge
                    // Une intersection est soit un Point, soit un Segment
                }
                vec![]
            });
            // Je cherches les intersections de uedge déjà calculées dans le dict
            // Si ce n'est pas déjà calculé je calcule
        }
        // Je trouve tous les polygones créés à partir des intersections
        // Je traite le cas des points à fusionner par élément. Cela permet de traiter des mesh qui
        // ont des noeuds non mergés.
    }
    // Je supprime de tool_mesh les segments des extrémités (un seul noeuds) jusqu'à tomber soit
    // sur un point d'intersection, soit sur un point commun avec mesh. Ainsi je ne garde pas les
    // "bouts qui dépassent" de mesh_tool.
    // Ensuite je créé les polygones en fermant les boucles. Pour suivre une boucle la règle est la
    // suivante:
    // - Je prend un segment de mesh, je le marque, le parcours dans un certain sens, puis je
    //   cherche tous les segments qui partagent l'extrémité à laquelle j'arrive
    // - Je retiens le segment qui tourne le plus dans le sens trigo (calcul d'angle)
    // - Je continue jusqu'à fermer le polygone, que j'ajoute comme nouvel élément dans le mesh
    // final.
    todo!()
}
