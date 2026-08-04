use mefikit::prelude as mf;
use mefikit::prelude::Overlayable;
use std::path::Path;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 60;
    let mesh1 = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    let n = 70;
    let mut mesh2 = mf::RegularUMeshBuilder::new()
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
    let mut coords = mesh2.coords_mut();
    coords += 0.75;

    println!("Start: imprint");
    let now = time::Instant::now();
    let cutted = mesh1.overlay(mesh2.clone(), mf::OverlayOperation::Imprint);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    println!("End:   building imprinted mesh in {ttot}s");
    mf::write(Path::new("imprint.vtk"), cutted.view())?;
    println!("Start: union");
    let now = time::Instant::now();
    let cutted = mesh1.overlay(mesh2.clone(), mf::OverlayOperation::Union);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    println!("End:   building union mesh in {ttot}s");
    mf::write(Path::new("union.vtk"), cutted.view())?;
    println!("Start: difference");
    let now = time::Instant::now();
    let cutted = mesh1.overlay(mesh2.clone(), mf::OverlayOperation::Difference);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    println!("End:   building difference mesh in {ttot}s");
    mf::write(Path::new("difference.vtk"), cutted.view())?;
    let now = time::Instant::now();
    let cutted = mesh1.overlay(mesh2.clone(), mf::OverlayOperation::Intersection);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    println!("End:   building intersection mesh in {ttot}s");
    mf::write(Path::new("intersection.vtk"), cutted.view())?;
    let now = time::Instant::now();
    let cutted = mesh1.overlay(mesh2.clone(), mf::OverlayOperation::SymmetricDifference);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    println!("End:   building SymmetricDifference mesh in {ttot}s");
    mf::write(Path::new("symmetric_difference.vtk"), cutted.view())?;
    Ok(())
}
