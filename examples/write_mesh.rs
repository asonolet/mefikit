use mefikit::RegularUMeshBuilder;
use mefikit::io;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let builder = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .add_axis(vec![0.0, 10.0, 20.0, 30.0])
        .add_axis(vec![0.0, 100.0, 200.0]);
    let mesh = builder.build();

    io::write(Path::new("examples/out.vtk"), mesh.view())?;
    println!("Mesh saved to out.vtk");

    // save_mesh(Path::new("examples/out.yaml"), &mesh)?;
    // println!("Mesh saved to out.yaml");

    // let umesh2 = load_mesh(Path::new("examples/out.json"))?;

    // println!("Mesh loaded from out.json");
    // println!("{umesh2:?}");

    Ok(())
}
