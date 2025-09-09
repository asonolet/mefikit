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
    let (submesh, _) = mesh.compute_submesh(None, None);
    println!("Selecting in sphere.");
    let mesh_sel = mf::Selector::new(submesh.view())
        .centroids()
        .in_sphere(&[0.5, 0.5, 0.5], 0.5)
        .select();

    println!("Writing submesh selected in sphere");
    mf::write(&Path::new("examples/submesh_sphere.vtk"), mesh_sel.view())
}
