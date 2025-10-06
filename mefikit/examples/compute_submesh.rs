use mefikit as mf;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    println!("Creating mesh");
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .build();
    println!("Computing submesh");
    let now = time::Instant::now();
    for _ in 0..10 {
        let _ = mf::topo::compute_submesh(&mesh, None, None);
    }
    let t_tot = now.elapsed().as_secs_f64() * 100.0;

    println!("Total elapsed time per op: {t_tot:}ms");

    Ok(())
}
