use super::geometry::measure as mes;
use super::umesh::Dimension::*;
use super::umesh::{Element, ElementId, ElementLike};
use super::umesh::{UMesh, UMeshView};

use std::collections::HashMap;

use ndarray::prelude::*;
use rstar::primitives::{GeomWithData, Line};
use rstar::{AABB, RTree};

const EPSLION_L: f64 = 1e-12;
const EPSLION_THETA: f64 = 1e-4;
const EPSILON_NN: f64 = 2.0 * EPSLION_L / EPSLION_THETA;
const EPSILON_NN2: f64 = EPSILON_NN * EPSILON_NN;

struct Segment(GeomWithData<Line<[f64; 2]>, ElementId>);

impl Segment {
    pub fn new(p1: [f64; 2], p2: [f64; 2], eid: ElementId) -> Self {
        Self(GeomWithData::new(Line { from: p1, to: p2 }, eid))
    }
    fn from(el: &Element) -> Self {
        let p1: &[f64] = el.coords().index_axis(Axis(0), 0).to_slice().unwrap();
        let p2: &[f64] = el.coords().index_axis(Axis(0), 1).to_slice().unwrap();
        Self::new([p1[0], p1[1]], [p2[0], p2[1]], el.id())
    }
}

fn to_aabb(el: &Element) -> AABB<[f64; 2]> {
    AABB::from_points(
        el.coords()
            .axis_iter(Axis(0))
            .map(|e| e.to_slice().unwrap()[..2].try_into().unwrap()),
    )
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

enum Intersection {
    /// The point variant corresponds to the case when two segments intersects in one point. This
    /// point can be an edge point.
    Point(ElementId, usize),

    /// This variant corresponds to the case when two segments share a common part with non null
    /// length because they are colinear. The two points returned corresponds to this common
    /// segment. Those points can correspond to edge points of one or both segments.
    Segment(ElementId, usize, usize),
}

enum IntersectionCoords {
    CommonNode(usize, usize),
    Seg1TangentNode(usize, [f64; 2]),
    Seg2TangentNode(usize, [f64; 2]),
    ColinearityNSeg1NSeg2(usize, usize),
    ColinearityNSeg1NSeg1(usize, usize),
    ColinearityNSeg2NSeg2(usize, usize),
    Point([f64; 2]),
    None,
}

/// Les règles de fusions sont les suivantes (et servent à éviter de créer des éléments dégénérés:
/// seg1 est la référence. Si un noeud doit être fusionné, c'est un noeud de seg1 qui est utilisé
/// si possible.
/// 1. Fusion de deux noeuds: 2 * EPSLION_NN est considéré plus petit que toutes les arrètes de mesh et
///    tool. Si c'est bien le cas la fusion se passe bien.
/// 2. Fusion d'un noeud de seg2 tangent à seg1: plutôt que de calculer une intersection toute
///    petite, EPSILON_L est utilisé pour déterminer si un point de seg2 est sur seg1. La
///    projection orthogonale est utilisée pour garantir la symmétrie du résultat. Un cas
///    particulier dégénéré peut apparaitre si deux segments très proches de mesh sont présents
///    dans cette zone. C'est pour cela que EPSILON_NN est calculé à partir de EPSILON_L et d'un
///    angle minimal entre deux segments, EPSILON_THETA. La relation n'étant pas exacte, cet angle
///    est indicatif. Ce point doit être dupliqué pour éviter de le partager entre deux arrètes
///    proches. Ainsi dans ce cas je renvoie le numéro du point proche de seg2 et les coordonnées.
/// 3. Fusion d'un noeud de seg1 tangent à seg2: même chose, si ce n'est que je n'ai pas besoin de
///    créer de nouveau point, j'utilise le point de seg1 que j'insererais plus tard dans seg2.
/// 4. Cas colinéaires
/// 5. Intersection classique
/// 6. Pas d'intersection
fn intersect_seg_seg(seg1: &Element, seg2: &Element) -> Option<IntersectionCoords> {
    let seg1_nodes = seg1.connectivity();
    let seg2_nodes = seg2.connectivity();
    let p1 = [seg1.coords()[[0, 0]], seg1.coords()[[0, 1]]];
    let p2 = [seg1.coords()[[1, 0]], seg1.coords()[[1, 1]]];
    let p3 = [seg2.coords()[[0, 0]], seg2.coords()[[0, 1]]];
    let p4 = [seg2.coords()[[1, 0]], seg2.coords()[[1, 1]]];

    let d1 = [p2[0] - p1[0], p2[1] - p1[1]];
    let d2 = [p4[0] - p3[0], p4[1] - p3[1]];

    let denom = d1[0] * d2[1] - d1[1] * d2[0];
    let epsilon = 1e-12;

    // Check for colinearity or parallelism
    if denom.abs() < epsilon {
        // Colinear: check for overlap
        let is_p1_on_seg2 = point_on_segment(p1, p3, p4, epsilon);
        let is_p2_on_seg2 = point_on_segment(p2, p3, p4, epsilon);
        let is_p3_on_seg1 = point_on_segment(p3, p1, p2, epsilon);
        let is_p4_on_seg1 = point_on_segment(p4, p1, p2, epsilon);

        if is_p1_on_seg2 && is_p2_on_seg2 {
            return Some(IntersectionCoords::ColinearityNSeg1NSeg1(
                seg1_nodes[0],
                seg1_nodes[1],
            ));
        }
        if is_p3_on_seg1 && is_p4_on_seg1 {
            return Some(IntersectionCoords::ColinearityNSeg2NSeg2(
                seg2_nodes[0],
                seg2_nodes[1],
            ));
        }
        if is_p1_on_seg2 && is_p3_on_seg1 {
            return Some(IntersectionCoords::ColinearityNSeg1NSeg2(
                seg1_nodes[0],
                seg2_nodes[0],
            ));
        }
        if is_p2_on_seg2 && is_p4_on_seg1 {
            return Some(IntersectionCoords::ColinearityNSeg1NSeg2(
                seg1_nodes[1],
                seg2_nodes[1],
            ));
        }
        return Some(IntersectionCoords::None);
    }

    // Parametric intersection
    let dx = p3[0] - p1[0];
    let dy = p3[1] - p1[1];
    let t = (dx * d2[1] - dy * d2[0]) / denom;
    let u = (dx * d1[1] - dy * d1[0]) / denom;

    // Check if intersection is within both segments
    if t >= -epsilon && t <= 1.0 + epsilon && u >= -epsilon && u <= 1.0 + epsilon {
        let intersection = [p1[0] + t * d1[0], p1[1] + t * d1[1]];

        // Endpoint merging
        for (i, &node1) in seg1_nodes.iter().enumerate() {
            if nearly_equal(intersection, p1, epsilon) {
                for (j, &node2) in seg2_nodes.iter().enumerate() {
                    if nearly_equal(p1, p3, epsilon) {
                        return Some(IntersectionCoords::CommonNode(node1, node2));
                    }
                    if nearly_equal(p1, p4, epsilon) {
                        return Some(IntersectionCoords::CommonNode(node1, node2));
                    }
                }
                return Some(IntersectionCoords::Seg1TangentNode(node1, intersection));
            }
            if nearly_equal(intersection, p2, epsilon) {
                for (j, &node2) in seg2_nodes.iter().enumerate() {
                    if nearly_equal(p2, p3, epsilon) {
                        return Some(IntersectionCoords::CommonNode(node1, node2));
                    }
                    if nearly_equal(p2, p4, epsilon) {
                        return Some(IntersectionCoords::CommonNode(node1, node2));
                    }
                }
                return Some(IntersectionCoords::Seg1TangentNode(node1, intersection));
            }
        }
        for (j, &node2) in seg2_nodes.iter().enumerate() {
            if nearly_equal(intersection, p3, epsilon) || nearly_equal(intersection, p4, epsilon) {
                return Some(IntersectionCoords::Seg2TangentNode(node2, intersection));
            }
        }
        return Some(IntersectionCoords::Point(intersection));
    }

    Some(IntersectionCoords::None)
}

// Helper functions
fn nearly_equal(a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps
}

fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    let cross = ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])).abs();
    if cross > eps {
        return false;
    }
    let dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1]);
    if dot < -eps {
        return false;
    }
    let len_sq = (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2);
    if dot > len_sq + eps {
        return false;
    }
    true
}

/// In the general case, there could be multiple intersections beween 1d elements (ex SEG2 and
/// SEG3).
///
/// The implementation should be completly symmetric for it to be correct.
fn intersect_1d_elems(e1: &Element, e2: &Element) -> Vec<IntersectionCoords> {
    todo!()
}

/// Cette méthode permet de découper un maillage 2d potentiellement non conforme avec un maillage
/// de segments propres (sans noeuds non fusionnés).
pub fn intersect_2dmesh_1dtool_mesh(mesh: UMeshView, tool_mesh: UMesh) -> Result<UMesh, String> {
    // tool_mesh.merge_nodes();
    let tool_mesh = tool_mesh.prepend_coords(mesh.coords());

    let (m1d, mgraph) = mesh.compute_submesh(Some(D2), None);

    // build R*-tree with 1d tool mesh
    let segs: Vec<_> = tool_mesh
        .elements()
        .map(|seg| Segment::from(&seg).0)
        .collect();
    let cutter_tree = RTree::bulk_load(segs);

    // Maintenant je travaille uniquement en terme de connectivité, j'utilise la table de
    // coordonnées suivante pour les identifiants de noeuds:
    // Table de mesh + table de tool_mesh après merge + nouveaux noeuds différents
    // Je créé la liste des segments de tool mesh découpés (avec les noeuds d'intersection)
    let mut e2int: HashMap<ElementId, Vec<Intersection>> = HashMap::new();
    for el in mesh.elements_of_dim(D2) {
        let segs_in_elem: Vec<_> = cutter_tree
            .locate_in_envelope_intersecting(&to_aabb(&el))
            .collect();
        for (_, _, &eid) in mgraph.edges(el.id()) {
            // TODO: edge could be SEG3!
            let edge = m1d.get_element(eid);
            let intersections: &_ = e2int.entry(eid).or_insert_with(|| {
                let mut intersections = Vec::new();
                for seg in &segs_in_elem {
                    let seg_elem = tool_mesh.get_element(seg.data);
                    // Calcul des intersections avec edge
                    // Une intersection est soit un Point, soit un Segment
                    let intersection_coords = intersect_1d_elems(&edge, &seg_elem);
                    let mut intersection_nodes: Vec<Intersection> =
                        Vec::with_capacity(intersection_coords.len());
                    for int in intersection_coords {
                        // TODO: if int is a new coord, append it to the tool_mesh coords, else retrieve its id, searching
                        // first into edge nodes, then into seg_elem
                        // push the new Intersection to intersection_nodes
                    }
                    intersections.append(&mut intersection_nodes);
                }
                intersections
            });
        }
        // Je trouve tous les polygones créés à partir des intersections
        // Je traite le cas des points à fusionner par élément. Cela permet de traiter des mesh qui
        // ont des noeuds non mergés.
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
        //
    }
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a mock Element with two nodes and coordinates
    fn mock_seg(coords: [[f64; 2]; 2]) -> UMesh {
        // Replace with your actual Element constructor
        let mut umesh = UMesh::new(
            ArcArray::from_shape_vec(
                [2, 2],
                vec![coords[0][0], coords[0][1], coords[1][0], coords[1][1]],
            )
            .unwrap(),
        );
        umesh.add_element(crate::ElementType::SEG2, &[0, 1], None, None);
        umesh
    }

    #[test]
    fn test_classic_intersection() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 1.0]]);
        let seg2 = mock_seg([[0.0, 1.0], [1.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let result = intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0));
        assert!(matches!(result, Some(IntersectionCoords::Point(_))));
    }

    #[test]
    fn test_tangency_at_endpoint() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [1.0, 1.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let result = intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0));
        assert!(matches!(result, Some(IntersectionCoords::CommonNode(1, 2))));
    }

    #[test]
    fn test_colinear_overlap() {
        let seg1 = mock_seg([[0.0, 0.0], [2.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [3.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let result = intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0));
        assert!(matches!(
            result,
            Some(IntersectionCoords::ColinearityNSeg1NSeg2(_, _))
                | Some(IntersectionCoords::ColinearityNSeg1NSeg1(_, _))
                | Some(IntersectionCoords::ColinearityNSeg2NSeg2(_, _))
        ));
    }

    #[test]
    fn test_no_intersection() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[2.0, 1.0], [2.0, 2.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let result = intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0));
        assert!(matches!(result, Some(IntersectionCoords::None)));
    }

    #[test]
    fn test_endpoint_merging() {
        let seg1 = mock_seg([[0.0, 0.0], [1.0, 0.0]]);
        let seg2 = mock_seg([[1.0, 0.0], [2.0, 0.0]]);
        let e0 = ElementId::new(crate::ElementType::SEG2, 0);
        let result = intersect_seg_seg(&seg1.get_element(e0), &seg2.get_element(e0));
        assert!(matches!(
            result,
            Some(IntersectionCoords::CommonNode(1, 2))
                | Some(IntersectionCoords::Seg1TangentNode(1, _))
                | Some(IntersectionCoords::Seg2TangentNode(2, _))
        ));
    }
}
