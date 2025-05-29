use mefikit::RegularUMeshBuilder;
use mefikit::io::save_mesh;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let builder = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0])
        .add_axis(vec![0.0, 10.0])
        .add_axis(vec![0.0, 100.0]);
    let mesh = builder.build();

    save_mesh(Path::new("out.json"), &mesh)?;

    println!("Mesh saved to out.json");
    Ok(())
}
