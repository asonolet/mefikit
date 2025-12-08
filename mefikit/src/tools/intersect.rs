use crate::geometry::ElementGeo;
use crate::mesh::Dimension::*;
use crate::mesh::UMesh;
use crate::mesh::{Element, ElementId, ElementLike, ElementType};
use crate::tools::compute_neighbours;

use std::collections::HashMap;

use arrayvec::ArrayVec;
use rstar::RTree;
use rstar::primitives::{GeomWithData, Line};

/// A wrapper struct representing a geometric line segment with associated element ID data.
///
/// The segment is defined by a `Line<[f64; 2]>` geometry and an `ElementId` for identification.
struct Segment(GeomWithData<Line<[f64; 2]>, ElementId>);

impl Segment {
    pub fn new(p1: [f64; 2], p2: [f64; 2], eid: ElementId) -> Self {
        Self(GeomWithData::new(Line { from: p1, to: p2 }, eid))
    }
    fn from(el: &Element) -> Self {
        let p1: [f64; 2] = *el.coord2_ref(0);
        let p2: [f64; 2] = *el.coord2_ref(1);
        Self::new(p1, p2, el.id())
    }
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
fn intersect_1d_elems(
    e1: &Element,
    e2: &Element,
    next_node_index: usize,
) -> (ArrayVec<usize, 4>, ArrayVec<usize, 4>, Option<[f64; 2]>) {
    match (e1.element_type(), e2.element_type()) {
        (ElementType::SEG2, ElementType::SEG2) => {
            let seg1_nodes = [e1.connectivity()[0], e1.connectivity()[1]];
            let seg2_nodes = [e2.connectivity()[0], e2.connectivity()[1]];
            let p1 = *e1.coord2_ref(0);
            let p2 = *e1.coord2_ref(1);
            let p3 = *e2.coord2_ref(0);
            let p4 = *e2.coord2_ref(1);
            todo!()
            // intersect_seg_seg(seg1_nodes, seg2_nodes, p1, p2, p3, p4, next_node_index)
        }
        _ => todo!(),
    }
}

/// Cette méthode permet de découper un maillage 2d potentiellement non conforme avec un maillage
/// de segments propres (sans noeuds non fusionnés).
pub fn cut_2d_mesh_with_1d_mesh(mesh: &UMesh, tool_mesh: UMesh) -> Result<UMesh, String> {
    // tool_mesh.merge_nodes();
    let tool_mesh = tool_mesh.prepend_coords(mesh.coords());

    let (m1d, mgraph) = compute_neighbours(mesh, Some(D2), None);

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
    let mut e2int: HashMap<ElementId, Vec<[usize; 2]>> = HashMap::new();
    for el in mesh.elements_of_dim(D2) {
        let segs_in_elem: Vec<_> = cutter_tree
            .locate_in_envelope_intersecting(&el.to_aabb2())
            .collect();
        for (_, _, &eid) in mgraph.edges(el.id()) {
            // TODO: edge could be SEG3!
            let edge = m1d.get_element(eid);
            let intersections: &_ = e2int.entry(eid).or_insert_with(|| {
                let intersections = Vec::new();
                for seg in &segs_in_elem {
                    let seg_elem = tool_mesh.get_element(seg.data);
                    // Calcul des intersections avec edge
                    // Une intersection est soit un Point, soit un Segment
                    let intersection_coords = intersect_1d_elems(&edge, &seg_elem, 4);
                    todo!()
                }
                intersections
            });
        }
    }
    todo!()
}
