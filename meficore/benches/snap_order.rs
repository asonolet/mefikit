use mefikit::prelude as mf;
use scaling::bench_scaling_gen;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // mf::write(Path::new("mesh.vtk"), mesh.view())?;

    let benched_snap = bench_scaling_gen(
        |n| {
            let m1 = (n as f64).powf(1. / 3.) as usize;
            let m2 = m1;
            let m3 = n / (m1 * m2);
            let x1: Vec<f64> = (0..=m1).map(|i| i as f64 / (m1 as f64)).collect();
            let x2: Vec<f64> = (0..=m2).map(|i| i as f64 / (m2 as f64)).collect();
            let x3: Vec<f64> = (0..=m3).map(|i| i as f64 / (m3 as f64)).collect();
            let mesh1 = mf::RegularUMeshBuilder::new()
                .add_axis(x1)
                .add_axis(x2)
                .add_axis(x3)
                .build();
            let mesh2 = mesh1.clone();
            (mesh1, mesh2)
        },
        |(m1, m2)| mf::snap::snap(m1, m2.view(), 1e-12),
        10000,
    );
    println!("snapping bench: {benched_snap}");
    Ok(())
}
