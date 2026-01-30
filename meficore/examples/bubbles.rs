use mefikit::{
    prelude as mf,
    tools::{Descendable, MeshSelect},
};
use ndarray as nd;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let xmax = 5.0;
    let ymax = 2.0;
    let r = 0.17;
    let nb = 15;
    let nr = 12.5;

    let nx = (xmax / r * nr) as usize;
    let ny = (ymax / r * nr) as usize;
    let nelems = nx * ny * ny;
    println!("nx: {nx}");
    println!("ny: {ny}");
    println!("nz: {ny}");
    println!("Number of elements: {nelems}");

    let xc = nd::array![
        3.34975968, 0.42080595, 1.19687701, 1.02917264, 0.9897215, 3.9543604, 4.47278769,
        1.45883669, 3.99005626, 4.31689995, 2.56044232, 1.31153504, 4.01096584, 1.16613541,
        3.62523646
    ];
    let yc = nd::array![
        0.58576054, 0.78208879, 0.3230594, 0.69742258, 0.51198892, 0.32282671, 0.27949664,
        0.49854072, 0.55459826, 0.29166307, 0.17983065, 0.48094793, 0.6506406, 0.77627632,
        0.58285244
    ];
    let zc = nd::array![
        0.7753009, 0.74069557, 0.3139743, 0.7416441, 0.65229628, 0.35339109, 0.69604875,
        0.74104633, 0.36762901, 0.51784778, 0.21718129, 0.55493735, 0.32701822, 0.67487601,
        0.28459688
    ];

    let mut sphere_union = mf::sel::sphere([xc[0], yc[0], zc[0]], r);
    for i in 1..nb {
        sphere_union = sphere_union | mf::sel::sphere([xc[i], yc[i], zc[i]], r);
    }

    let now = time::Instant::now();
    let volumes = mf::RegularUMeshBuilder::new()
        .add_axis((0..=nx).map(|i| i as f64 / xmax).collect::<Vec<f64>>())
        .add_axis((0..=ny).map(|i| i as f64 / ymax).collect::<Vec<f64>>())
        .add_axis((0..=ny).map(|i| i as f64 / ymax).collect::<Vec<f64>>())
        .build();

    let t0 = now.elapsed().as_secs_f64();
    let _box_boundaries = volumes.boundaries(None, None);
    let t1 = now.elapsed().as_secs_f64();
    let (_, inner_bubbles) = volumes.select(sphere_union);
    let interface = inner_bubbles.boundaries(None, None);
    let _cracked = mf::crack(volumes, interface.view());
    let _bubble_groups = mf::compute_connected_components(&inner_bubbles, None, None);

    let t_tot = now.elapsed().as_secs_f64();
    let t_dc = t1 - t0;
    println!("Op1: {t_dc:}s");
    println!("Total elapsed time per op: {t_tot:}s");

    Ok(())
}
