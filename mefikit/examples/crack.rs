use mefikit::prelude as mf;
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

    println!("Start: building submesh");
    let submesh = mf::compute_submesh(&mesh, None, None);
    // mf::write(Path::new("submesh.vtk"), submesh.view())?;
    println!("End:   building submesh");

    println!("Start: building partsubmesh");
    let partsubmesh = mf::Selector::new(&submesh)
        .nodes(false)
        .in_sphere(&[0.5, 0.5, 0.5], 0.5)
        .select();
    // mf::write(Path::new("partsubmesh.vtk"), partsubmesh.view())?;
    println!("End:   building partsubmesh");

    println!("Start: building crackedmesh");
    let now = time::Instant::now();
    let cracked = mf::crack::crack(mesh, partsubmesh.view());
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("cracked.vtk"), cracked.view())?;
    println!("End:   building crackedmesh in {ttot}s");
    Ok(())
}
