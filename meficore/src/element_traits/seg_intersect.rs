use nalgebra as na;
use nalgebra::Point2;

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
    One(Intersection),
    Segment([PointId; 2]),
}

pub fn intersect_seg_seg(
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    p4: Point2<f64>,
) -> Intersections {
    let v1 = p2 - p1;
    let v2 = p4 - p3;

    let alpha = v1[0] * v2[1] - v1[1] * v2[0];
    let scale = v1[0]
        .abs()
        .max(v1[1].abs())
        .max(v2[0].abs())
        .max(v2[1].abs())
        .max(1.0);
    let eps = 64.0 * scale * f64::EPSILON;
    let eps_squared = eps * eps;

    // If one of the edges is degenerated, there is no intersection
    if (v2.norm_squared() < eps_squared) | (v1.norm_squared() < eps_squared) {
        return Intersections::None;
    }

    if alpha.abs() < eps {
        colinear_seg_intersection(p1, p2, p3, p4, eps)
    } else {
        let dx = p3[0] - p1[0];
        let dy = p3[1] - p1[1];
        let t = (dx * v2[1] - dy * v2[0]) / alpha;
        let u = (dx * v1[1] - dy * v1[0]) / alpha;
        let intersection1 = p1 + t * v1;
        let intersection2 = p3 + u * v2;
        let intersection = intersection1 + (intersection2 - intersection1) / 2.0;
        if !((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)) {
            Intersections::None
        } else if (p1 - intersection).norm_squared() < eps_squared {
            Intersections::One(Intersection::Existing(PointId::P1))
        } else if (p2 - intersection).norm_squared() < eps_squared {
            Intersections::One(Intersection::Existing(PointId::P2))
        } else if (p3 - intersection).norm_squared() < eps_squared {
            Intersections::One(Intersection::Existing(PointId::P3))
        } else if (p4 - intersection).norm_squared() < eps_squared {
            Intersections::One(Intersection::Existing(PointId::P4))
        } else {
            dbg!(t);
            dbg!(u);
            Intersections::One(Intersection::New(intersection.into()))
        }
    }
}

fn colinear_seg_intersection(
    p1: Point2<f64>,
    p2: Point2<f64>,
    p3: Point2<f64>,
    p4: Point2<f64>,
    eps: f64,
) -> Intersections {
    // Check if points are aligned.
    let d1 = p3 - p2;
    let d2 = p4 - p1;
    let det = d1[0] * d2[1] - d1[1] * d2[0];
    if det.abs() > eps {
        return Intersections::None;
    }
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

    fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
        let p = Point2::from(p);
        let a = Point2::from(a);
        let b = Point2::from(b);
        let ap = p - a;
        let bp = p - b;
        let ab = b - a;

        let cross = dbg!(ap[0] * bp[1] - ap[1] * bp[0]);

        let scale = ab[0].abs().max(ab[1].abs()).max(1.0);
        let eps = dbg!(64.0 * scale.powi(2) * f64::EPSILON);

        if cross.abs() > eps {
            return false;
        }

        let dot = dbg!(ap.dot(&ab));

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

    fn arb_point() -> impl Strategy<Value = [f64; 2]> {
        prop::array::uniform2(-1e6f64..1e6)
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

            match (r1, r2) {
                (Intersections::None, Intersections::None) => {}
                (Intersections::One(_), Intersections::One(_)) => {}
                (Intersections::Segment(_), Intersections::Segment(_)) => {}
                _ => panic!("asymmetric result"),
            }
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

            if let Intersections::One(int) = res {
                let p = match int {
                    Intersection::Existing(id) =>
                        id_to_point(id, p1, p2, p3, p4),
                    Intersection::New(p) => p,
                };

                prop_assert!(point_on_segment(p, p1, p2));
                prop_assert!(point_on_segment(p, p3, p4));
            }
        }
    }
    proptest! {
        #[test]
        fn existing_intersection_is_endpoint(
            p1 in arb_point(),
            p2 in arb_point(),
            p3 in arb_point(),
            p4 in arb_point(),
        ) {
            let res = intersect_seg_seg(p1.into(), p2.into(), p3.into(), p4.into());

            if let Intersections::One(Intersection::Existing(id)) = res {
                let p = id_to_point(id, p1, p2, p3, p4);

                prop_assert!(
                    p == p1 || p == p2 || p == p3 || p == p4
                );
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

            if let Intersections::Segment([a, b]) = res {
                let pa = id_to_point(a, p1, p2, p3, p4);
                let pb = id_to_point(b, p1, p2, p3, p4);

                prop_assert!(point_on_segment(pa, p1, p2));
                prop_assert!(point_on_segment(pa, p3, p4));
                prop_assert!(point_on_segment(pb, p1, p2));
                prop_assert!(point_on_segment(pb, p3, p4));
            }
        }
    }
    proptest! {
        #[test]
        fn separated_segments_do_not_intersect(
            p in arb_point(),
        ) {
            let p1 = p;
            let p2 = [p[0] + 1.0, p[1]];
            let p3 = [p[0], p[1] + 10.0];
            let p4 = [p[0] + 1.0, p[1] + 10.0];

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

        if let Intersections::One(int) = res {
            let p = match int {
                Intersection::Existing(id) => id_to_point(id, p1, p2, p3, p4),
                Intersection::New(p) => p,
            };
            println!("{p:?}");

            assert!(point_on_segment(p, p1, p2));
            assert!(point_on_segment(p, p3, p4));
        }
    }
}
