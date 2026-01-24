use mefikit::prelude as mf;
use mefikit::prelude::MeshSelect;
use std::path::Path;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 60;
    println!("Start: building mesh");
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    // mf::write(Path::new("mesh.vtk"), mesh.view())?;
    println!("End:   building mesh");

    println!("Start: building descending_mesh");
    let descending_mesh = mf::compute_descending(&mesh, None, None);
    // mf::write(Path::new("descending_mesh.vtk"), descending_mesh.view())?;
    println!("End:   building descending_mesh");

    println!("Start: building partdescending_mesh");
    let (_, partdescending_mesh) =
        descending_mesh.select(mf::sel::nsphere([0.5, 0.5, 0.5], 0.5, false));
    // mf::write(Path::new("partdescending_mesh.vtk"), partdescending_mesh.view())?;
    println!("End:   building partdescending_mesh");

    println!("Start: building crackedmesh");
    let now = time::Instant::now();
    let cracked = mf::crack::crack(mesh, partdescending_mesh.view());
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("cracked.vtk"), cracked.view())?;
    println!("End:   building crackedmesh in {ttot}s");
    Ok(())
}
