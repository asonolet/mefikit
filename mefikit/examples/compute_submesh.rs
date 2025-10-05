use mefikit as mf;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    println!("Creating mesh");
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .add_axis((0..=10).map(|i| i as f64 / 10.0).collect::<Vec<f64>>())
        .build();
    println!("Computing submesh");
    let submesh = mf::topo::compute_submesh(&mesh, None, None);

    println!("Writing submesh");
    mf::write(&Path::new("examples/submesh.vtk"), submesh.view())
}
