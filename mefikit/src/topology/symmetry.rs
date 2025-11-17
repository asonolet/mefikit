use crate::ElementLike;
use rustc_hash::FxHashSet as HashSet;

pub trait ElementEquality<'a>: ElementLike<'a> {
    fn strict_equality(&self, other: &Self) -> bool {
        if self.connectivity().len() != other.connectivity().len() {
            return false;
        }
        self.connectivity()
            .iter()
            .zip(other.connectivity().iter())
            .all(|(a, b)| a == b)
    }
    fn node_set_equality(&self, other: &Self) -> bool {
        let self_nodes: HashSet<usize> = self.connectivity().iter().copied().collect();
        let other_nodes: HashSet<usize> = other.connectivity().iter().copied().collect();
        self_nodes == other_nodes
    }
    // fn strict_equivalence(&self, other: &Self) -> bool {
    //     if self.connectivity().len() != other.connectivity().len() {
    //         return false;
    //     }
    //     for symmetry in symmetries {
    //         let mut is_equivalent = true;
    //         for (i, &node_idx) in symmetry.iter().enumerate() {
    //             if self.connectivity()[i] != other.connectivity()[node_idx] {
    //                 is_equivalent = false;
    //                 break;
    //             }
    //         }
    //         if is_equivalent {
    //             return true;
    //         }
    //     }
    //     false
    // }
}

// define_symmetries! {
//     QUAD4 => {
//         order: 4,
//         rotations: [
//             [0, 1, 2, 3],
//             [1, 2, 3, 0],
//             [2, 3, 0, 1],
//             [3, 0, 1, 2],
//         ]
//     },
//     HEX8 => {
//         order: 8,
//         rotations: [
//             [0, 1, 2, 3, 4, 5, 6, 7],
//             [1, 2, 3, 0, 5, 6, 7, 4],
//             [2, 3, 0, 1, 6, 7, 4, 5],
//             [3, 0, 1, 2, 7, 4, 5, 6],
//         ]
//     }
// }
