use mefikit::RegularUMeshBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let mesh = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0])
        .add_axis(vec![0.0, 5.0])
        .add_axis(vec![0.0, 10.0])
        .build();
    println!("{mesh:?}");
    Ok(())
}
