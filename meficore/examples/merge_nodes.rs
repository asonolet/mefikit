use mefikit::prelude as mf;
use mefikit::tools::selector::MeshSelect;
use std::path::Path;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 60;
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64 / (n as f64)).collect::<Vec<f64>>())
        .build();
    // mf::write(Path::new("mesh.vtk"), mesh.view())?;
    let descending_mesh = mf::compute_descending(&mesh, None, None);
    let (_, descending_mesh) = descending_mesh.select(mf::sel::sphere([0.5, 0.5, 0.5], 0.5));

    let cracked = mf::crack(mesh, descending_mesh.view());

    println!("Start: merge_nodes");
    let now = time::Instant::now();
    let merged = mf::merge_nodes(cracked, 1e-12);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("snapped.vtk"), merged.view())?;
    println!("End:   building merged mesh in {ttot}s");
    Ok(())
}
