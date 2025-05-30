use mefikit::RegularUMeshBuilder;
use mefikit::compute_subentities;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let builder = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .add_axis(vec![0.0, 10.0, 20.0, 30.0]);
    let mesh = builder.build();
    let (submesh, _, _) = compute_subentities(&mesh, None);

    println!("{mesh:?}");
    println!("{submesh:?}");

    Ok(())
}
