use mefikit::prelude as mf;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 20;
    println!("Start: building mesh");
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    // mf::write(&Path::new("mesh.vtk"), mesh.view())?;
    println!("End:   building mesh");

    println!("Start: building submesh");
    let submesh = mf::compute_submesh(&mesh, None, None);
    // mf::write(&Path::new("submesh.vtk"), submesh.view())?;
    println!("End:   building submesh");

    println!("Start: building partsubmesh");
    let partsubmesh = mf::Selector::new(&submesh)
        .nodes(false)
        .in_sphere(&[0.5, 0.5, 0.5], 0.5)
        .select();
    // mf::write(&Path::new("partsubmesh.vtk"), partsubmesh.view())?;
    println!("End:   building partsubmesh");

    println!("Start: building crackedmesh");
    let cracked = mf::crack::crack(mesh, partsubmesh.view());
    // mf::write(&Path::new("cracked.vtk"), cracked.view())?;
    println!("End:   building crackedmesh");
    Ok(())
}
