use super::umesh::Dimension::*;
use super::umesh::{Element, ElementId, ElementLike, ElementType};
use super::umesh::{UMesh, UMeshView};

struct BBox(f64, f64, f64, f64);

impl BBox {
    fn from_seg2(seg: &Element) -> Self {
        let p1 = [
            *seg.coords().get((0, 0)).unwrap(),
            *seg.coords().get((0, 1)).unwrap(),
        ];
        let p2 = [
            *seg.coords().get((1, 0)).unwrap(),
            *seg.coords().get((1, 1)).unwrap(),
        ];
        Self(
            f64::min(p1[0], p2[0]),
            f64::min(p1[1], p2[1]),
            f64::max(p1[0], p2[0]),
            f64::max(p1[1], p2[1]),
        )
    }

    pub fn from_elem(e: &Element) -> Self {
        use ElementType::*;
        match e.element_type() {
            SEG2 => Self::from_seg2(e),
            _ => todo!(),
        }
    }

    pub fn x_min(&self) -> f64 {
        return self.0;
    }

    pub fn x_max(&self) -> f64 {
        return self.2;
    }

    pub fn y_min(&self) -> f64 {
        return self.1;
    }

    pub fn y_max(&self) -> f64 {
        return self.3;
    }

    pub fn intersects(&self, other: &Self) -> bool {
        if self.x_min() > other.x_max() {
            return false;
        }
        if self.x_max() < other.x_min() {
            return false;
        }
        if self.y_min() > other.y_max() {
            return false;
        }
        if self.y_max() < other.y_min() {
            return false;
        }
        true
    }
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
    Point(ElementId, ElementId, [f64; 2]),
    /// This variant corresponds to the case when two segments share a common part with non null
    /// length because they are colinear. The two points returned corresponds to this common
    /// segment. Those points can correspond to edge points of one or both segments.
    Segment(ElementId, ElementId, [f64; 2], [f64; 2]),
}

fn intersect_seg_mesh1d(seg: Element, m1d: UMeshView) -> Vec<Intersection> {
    let bbox1 = BBox::from_elem(&seg);
    let mut res = Vec::new();
    // TODO: optimize using BVH
    for e in m1d.elements() {
        let bbox2 = BBox::from_elem(&e);
        if !bbox1.intersects(&bbox2) {
            continue;
        }
        // TODO: check if colinear
        let colinearity = colinear_seg_seg(&seg, &e);
        if let Some(col) = colinearity {
            res.push(Intersection::Segment(seg.id(), e.id(), col.0, col.1));
            continue;
        }
        // else compute intersection
        let intersection = intersect_seg_seg(&seg, &e);
        if let Some(int) = intersection {
            res.push(Intersection::Point(seg.id(), e.id(), int));
        }
    }
    res
}

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
struct DirectedEdge(usize, usize);

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
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
pub fn intersect_2dmesh_1dtool_mesh(mesh: UMeshView, tool_mesh: UMeshView) -> UMesh {
    // find intersections and create two maps: polygon map with new intersection points and segments map
    let (m1d, _mesh_grph) = mesh.compute_submesh(Some(D2), None);
    // There is a little preparation on tool_mesh that should be done:
    // - nodes should be merged.
    // - coords from mesh should be prepended
    let mut intersections = Vec::new();
    for seg in tool_mesh.elements() {
        // Find intersections with m1d (can be optimized using BVH)
        intersections.append(&mut intersect_seg_mesh1d(seg, m1d.view()));
    }
    // Maintenant je travaille uniquement en terme de connectivité, j'utilise la table de
    // coordonnées suivante pour les identifiants de noeuds:
    // Table de mesh + table de tool_mesh après merge + nouveaux noeuds différents
    // Je créé la liste des segments de tool mesh découpés (avec les noeuds d'intersection)
    for el in mesh.elements() {
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
