use mefikit::prelude as mf;
use mf::NodeDuplicates;
use std::path::Path;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 60;
    let mut mesh1 = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    let mesh2 = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    // mf::write(Path::new("mesh.vtk"), mesh.view())?;

    println!("Start: snapping");
    let now = time::Instant::now();
    mesh1.snap_on(mesh2.view(), 1e-12);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("snapped.vtk"), mesh1.view())?;
    println!("End:   building snappededmesh in {ttot}s");
    Ok(())
}
