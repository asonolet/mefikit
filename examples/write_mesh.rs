use mefikit::RegularUMeshBuilder;
use mefikit::io::{save_mesh, load_mesh};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let builder = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0])
        .add_axis(vec![0.0, 10.0])
        .add_axis(vec![0.0, 100.0]);
    let mesh = builder.build();

    save_mesh(Path::new("examples/out.json"), &mesh)?;
    println!("Mesh saved to out.json");

    save_mesh(Path::new("examples/out.yaml"), &mesh)?;
    println!("Mesh saved to out.yaml");

    let umesh2 = load_mesh(Path::new("examples/out.json"))?;
    println!("Mesh loaded from out.json {umesh2:?}");
    assert_eq!(umesh2.coords().shape(), &[12, 3]);
    assert_eq!(umesh2.element_blocks().len(), 1);

    Ok(())
}
