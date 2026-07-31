use mefikit::prelude as mf;
use std::path::Path;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 60;
    let mesh1 = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    let n = 70;
    let mesh2 = mf::RegularUMeshBuilder::new()
        .add_axis(
            (0..=n)
                .map(|i| i as f64 / (2.0 * n as f64))
                .collect::<Vec<f64>>(),
        )
        .add_axis(
            (0..=n)
                .map(|i| i as f64 / (2.0 * n as f64))
                .collect::<Vec<f64>>(),
        )
        .build();
    // mf::write(Path::new("mesh.vtk"), mesh.view())?;

    println!("Start: snapping");
    let now = time::Instant::now();
    let cutted = mf::intersect::intersect_2d2d(mesh1, mesh2);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("cutted.vtk"), cutted.view())?;
    println!("End:   building cutted mesh in {ttot}s");
    Ok(())
}
