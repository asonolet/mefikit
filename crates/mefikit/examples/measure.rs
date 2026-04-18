use mefikit::prelude as mf;
use std::time;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=1000).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .add_axis((0..=1000).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .build();
    let mut t_tot = 0.0;
    let n = 100;
    for _ in 0..n {
        let now = time::Instant::now();
        let _ = mf::measure(mesh.view(), None);
        let elapsed = now.elapsed();
        t_tot += elapsed.as_secs_f64();
    }
    t_tot *= 1000.0 / (n as f64);
    println!("Total elapsed time per op: {t_tot:}ms");
    Ok(())
}
