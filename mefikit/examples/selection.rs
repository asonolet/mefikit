use mefikit as mf;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a regular UMesh with specified axes
    let mesh = mf::RegularUMeshBuilder::new()
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .add_axis((0..=100).map(|i| i as f64 / 100.0).collect::<Vec<f64>>())
        .build();

    println!("Selecting in sphere.");
    let mesh_sel = mf::Selector::new(mesh.view())
        .centroids()
        .in_sphere(&[0.5, 0.5, 0.5], 0.5)
        .select();

    println!("Writing mesh selected in sphere");
    mf::write(&Path::new("examples/mesh_sphere.vtk"), mesh_sel.view())
    // Return Ok to indicate successful execution
}
