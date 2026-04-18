use nalgebra::Point2;
use nalgebra::{self as na, Vector2};

#[derive(Copy, Debug, PartialEq, Clone, PartialOrd)]
pub enum Intersection {
    Existing(PointId),
    New([f64; 2]),
}

#[derive(Copy, Debug, PartialEq, Clone, PartialOrd)]
pub enum PointId {
    P1,
    P2,
    P3,
    P4,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Intersections {
    None,
    One(Intersection),      // classic one intersection
    Two([Intersection; 2]), // Two intersections only possible with arc
    Segment([PointId; 2]),  // Shared arc/seg, between two existing points
}

#[inline(always)]
fn cross_prod2(v1: Vector2<f64>, v2: Vector2<f64>) -> f64 {
    v1[0] * v2[1] - v1[1] * v2[0]
}

pub fn intersect_seg_seg(
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    p4: Point2<f64>,
) -> Intersections {
    let v1 = p2 - p1;
    let v2 = p4 - p3;

    let cross12 = cross_prod2(v1, v2);
    let scale = v1[0]
        .abs()
        .max(v1[1].abs())
        .max(v2[0].abs())
        .max(v2[1].abs())
        .max(1.0);
    // As this error is used on cross or dot product, it should scale with the length of the
    // segments.
    let eps = 64.0 * scale * f64::EPSILON;

    // If one of the edges is degenerated, there is no intersection. This is simplist, but there
    // should be no degenerated segments in a proper mesh.
    if (v2.norm_squared() < eps) || (v1.norm_squared() < eps) {
        return Intersections::None;
    }
    let v3 = p3 - p1;
    let cross31 = cross_prod2(v3, v1);

    if cross12.abs() < eps {
        if cross31.abs() > eps {
            // Segments are // but do not cross
            Intersections::None
        } else {
            colinear_seg_intersection(p1, p2, p3, p4)
        }
    } else {
        let cross32 = cross_prod2(v3, v2);
        let t = cross32 / cross12;
        let u = cross31 / cross12;
        let intersection = p1 + t * v1;
        if !((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)) {
            Intersections::None
        } else if (p1 - intersection).norm_squared() < eps {
            Intersections::One(Intersection::Existing(PointId::P1))
        } else if (p2 - intersection).norm_squared() < eps {
            Intersections::One(Intersection::Existing(PointId::P2))
        } else if (p3 - intersection).norm_squared() < eps {
            Intersections::One(Intersection::Existing(PointId::P3))
        } else if (p4 - intersection).norm_squared() < eps {
            Intersections::One(Intersection::Existing(PointId::P4))
        } else {
            Intersections::One(Intersection::New(intersection.into()))
        }
    }
}

fn colinear_seg_intersection(
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    p4: Point2<f64>,
) -> Intersections {
    let or = na::Point2::origin();
    let o = or + ((p1 - or) + (p2 - or) + (p3 - or) + (p4 - or)) / 4.0;
    let dir = p2 - p1;
    let ts = if dir[0] > dir[1] {
        [
            (p1[0] - o[0]) / dir[0],
            (p2[0] - o[0]) / dir[0],
            (p3[0] - o[0]) / dir[0],
            (p4[0] - o[0]) / dir[0],
        ]
    } else {
        [
            (p1[1] - o[1]) / dir[1],
            (p2[1] - o[1]) / dir[1],
            (p3[1] - o[1]) / dir[1],
            (p4[1] - o[1]) / dir[1],
        ]
    };
    use PointId::*;
    let mut ord = [0, 1, 2, 3];
    ord.sort_unstable_by(|&i, &j| ts[i].partial_cmp(&ts[j]).unwrap());
    const PS: [PointId; 4] = [P1, P2, P3, P4];
    let ps = [PS[ord[0]], PS[ord[1]], PS[ord[2]], PS[ord[3]]];
    match ps {
        [P1, P2, _, _] | [_, _, P1, P2] => Intersections::None,
        [P1, a, P2, _] => Intersections::Segment([a, P2]),
        [P1, a, b, P2] => Intersections::Segment([a, b]),
        [_, P1, a, P2] => Intersections::Segment([P1, a]),
        _ => {
            panic!(
                "This situation should not be possible as P1 is before P2 along the P2 - P1 vec."
            )
        }
    }
}

// /// Computes the squared distance of point p to the line defined by origin and dir.
// fn compute_distance2_to_line(o: Point2<f64>, dir: Vector2<f64>, p: Point2<f64>) -> f64 {
//     let op = p - o;
//     let len2 = dir.norm_squared();
//     let eps = 32.0 * f64::EPSILON.powi(2);
//     if len2 < eps {
//         return op.norm_squared();
//     }
//     let t = op.dot(&dir) / len2;
//     let proj = o + t * dir;
//
//     (p - proj).norm_squared()
// }

#[cfg(test)]
mod tests {
    use super::*;
    fn points_close(p1: [f64; 2], p2: [f64; 2], scale: f64) -> bool {
        let p1 = Point2::from(p1);
        let p2 = Point2::from(p2);
        let eps = 64.0 * scale * f64::EPSILON;
        let v = p1 - p2;
        v.norm_squared() <= eps
    }

    fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2], scale: f64) -> bool {
        let p = Point2::from(p);
        let a = Point2::from(a);
        let b = Point2::from(b);
        let ap = p - a;
        let bp = p - b;
        let ab = b - a;

        let cross = ap[0] * bp[1] - ap[1] * bp[0];

        // This error depends on scale^2 because it comes from the computation of the intersection.
        let eps = 64.0 * scale * scale * f64::EPSILON;

        if cross.abs() > eps {
            return false;
        }

        let dot = ap.dot(&ab);

        if dot < 0.0 {
            return false;
        }

        let len_sq = ab.norm_squared();

        dot <= len_sq
    }

    fn id_to_point(
        id: PointId,
        p1: [f64; 2],
        p2: [f64; 2],
        p3: [f64; 2],
        p4: [f64; 2],
    ) -> [f64; 2] {
        match id {
            PointId::P1 => p1,
            PointId::P2 => p2,
            PointId::P3 => p3,
            PointId::P4 => p4,
        }
    }
    use proptest::prelude::*;

    fn pos_vector() -> impl Strategy<Value = [f64; 2]> {
        prop::array::uniform2(1.0f64..1e6)
    }

    fn arb_point() -> impl Strategy<Value = [f64; 2]> {
        prop::array::uniform2(-1e6f64..1e6)
    }

    fn scale(p1: [f64; 2], p2: [f64; 2], p3: [f64; 2], p4: [f64; 2]) -> f64 {
        let v1 = [p2[0] - p1[0], p2[1] - p1[1]];
        let v2 = [p4[0] - p3[0], p4[1] - p3[1]];
        v1[0]
            .abs()
            .max(v1[1].abs())
            .max(v2[0].abs())
            .max(v2[1].abs())
            .max(1.0)
    }

    fn is_symmetric(int1: Intersections, int2: Intersections, scale: f64) -> bool {
        match (int1, int2) {
            (Intersections::None, Intersections::None) => true,
            (
                Intersections::One(Intersection::Existing(a)),
                Intersections::One(Intersection::Existing(b)),
            ) => a == b,
            (
                Intersections::One(Intersection::New(p1)),
                Intersections::One(Intersection::New(p2)),
            ) => points_close(p1, p2, scale),
            (Intersections::Segment([a1, a2]), Intersections::Segment([a3, a4])) => {
                ((a1 == a3) && (a2 == a4)) || ((a1 == a4) && (a2 == a3))
            }
            _ => false,
        }
    }

    proptest! {
        #[test]
        fn intersection_is_symmetric(
            p1 in arb_point(),
            p2 in arb_point(),
            p3 in arb_point(),
            p4 in arb_point(),
        ) {
            let r1 = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
            let r2 = intersect_seg_seg(p3.into(), p4.into(), p1.into(), p2.into());
            let scale = scale(p1, p2, p3, p4);

            prop_assert!(is_symmetric(r1, r2, scale))
        }
    }
    proptest! {
        #[test]
        fn intersection_point_lies_on_both_segments(
            p1 in arb_point(),
            p2 in arb_point(),
            p3 in arb_point(),
            p4 in arb_point(),
        ) {
            let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
            let scale = scale(p1, p2, p3, p4);

            if let Intersections::One(int) = res {
                let p = match int {
                    Intersection::Existing(id) =>
                        id_to_point(id, p1, p2, p3, p4),
                    Intersection::New(p) => p,
                };

                prop_assert!(point_on_segment(p, p1, p2, scale));
                prop_assert!(point_on_segment(p, p3, p4, scale));
            }
        }
    }
    proptest! {
        #[test]
        fn segment_intersection_is_collinear(
            p1 in arb_point(),
            p2 in arb_point(),
            p3 in arb_point(),
            p4 in arb_point(),
        ) {
            let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
            let scale = scale(p1, p2, p3, p4);

            if let Intersections::Segment([a, b]) = res {
                let pa = id_to_point(a, p1, p2, p3, p4);
                let pb = id_to_point(b, p1, p2, p3, p4);

                prop_assert!(point_on_segment(pa, p1, p2, scale));
                prop_assert!(point_on_segment(pa, p3, p4, scale));
                prop_assert!(point_on_segment(pb, p1, p2, scale));
                prop_assert!(point_on_segment(pb, p3, p4, scale));
            }
        }
    }
    proptest! {
        #[test]
        fn segment_with_common_point_intersect(
            p1 in arb_point(),
            p2 in arb_point(),
            p3 in arb_point(),
        ) {
            let res = intersect_seg_seg(p1.into(), p2.into(), p1.into(), p3.into());
            matches!(res, Intersections::One(_) | Intersections::Segment(_));
        }
    }
    proptest! {
        #[test]
        fn parallel_segments_do_not_intersect(
            p in arb_point(),
            v in pos_vector(),
        ) {
            let p1 = p;
            let p2 = [p[0] + v[0], p[1]];
            let p3 = [p[0], p[1] + v[1]];
            let p4 = [p[0] + v[0], p[1] + v[1]];

            let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
            prop_assert_eq!(res, Intersections::None);
        }
    }

    #[test]
    fn test_1separated_seg_do_not_intersect() {
        let p1 = [0.0, 0.0];
        let p2 = [1.0, 0.0];
        let p3 = [0.0, 10.0];
        let p4 = [1.0, 10.0];

        let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
        assert_eq!(res, Intersections::None);
    }

    #[test]
    fn test_2both_seg() {
        let p1 = [-730405.1992762072, 0.0];
        let p2 = [0.0, 0.0];
        let p3 = [0.0, -805700.7903997403];
        let p4 = [-65840.31658583878, 990202.9211195839];
        let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
        let scale = scale(p1, p2, p3, p4);

        if let Intersections::One(int) = res {
            let p = match int {
                Intersection::Existing(id) => id_to_point(id, p1, p2, p3, p4),
                Intersection::New(p) => p,
            };
            println!("{p:?}");

            assert!(point_on_segment(p, p1, p2, scale));
            assert!(point_on_segment(p, p3, p4, scale));
        }
    }

    #[test]
    fn test_3both_seg() {
        let p1 = [809224.6799957141, -235808.925127813];
        let p2 = [-901882.3353661865, 570278.1374180546];
        let p3 = [270416.52282510285, 15443.5513155427];
        let p4 = [269544.0152902226, 23150.45387644443];
        let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());
        let scale = scale(p1, p2, p3, p4);

        if let Intersections::One(int) = res {
            let p = match int {
                Intersection::Existing(id) => id_to_point(id, p1, p2, p3, p4),
                Intersection::New(p) => p,
            };
            println!("{p:?}");

            assert!(point_on_segment(p, p1, p2, scale));
            assert!(point_on_segment(p, p3, p4, scale));
        }
    }
}
