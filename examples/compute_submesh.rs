use mefikit::compute_submesh;
use mefikit::RegularUMeshBuilder;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let mesh = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0])
        .add_axis(vec![0.0, 10.0])
        .build();
    let (submesh, _, _) = compute_submesh(&mesh, None);

    Ok(())
}
