use mefikit::prelude as mf;
use mefikit::tools::selector::MeshSelect;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a regular UMesh with specified axes
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .build();

    println!("Selecting in sphere.");
    let now = time::Instant::now();
    for _ in 0..10 {
        let _ = mesh.select(mf::sel::sphere([0.5, 0.5, 0.5], 0.5));
    }
    let t_tot = now.elapsed().as_secs_f64() * 100.0;

    println!("Total elapsed time per op: {t_tot:}ms");
    // Return Ok to indicate successful execution
    Ok(())
}
