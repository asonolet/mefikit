use mefikit::prelude as mf;
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
    let submesh = mf::compute_submesh(&mesh, None, None);
    let submesh = mf::Selector::new(&submesh)
        .centroids()
        .in_sphere(&[0.5, 0.5, 0.5], 0.5)
        .select();

    let cracked = mf::crack(mesh, submesh.view());

    println!("Start: merge_nodes");
    let now = time::Instant::now();
    let merged = mf::merge_nodes(cracked, 1e-12);
    let elapsed = now.elapsed();
    let ttot = elapsed.as_secs_f64();
    mf::write(Path::new("snapped.vtk"), merged.view())?;
    println!("End:   building merged mesh in {ttot}s");
    Ok(())
}
