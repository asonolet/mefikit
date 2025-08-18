pub trait ElementSymmetry {
    const ROTATIONS: &'static [&'static [usize]];
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
