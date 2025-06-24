use mefikit::RegularUMeshBuilder;
use mefikit::UMesh;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a regular UMesh with specified axes
    let mesh: UMesh = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0, 6.0]) // First axis with three points
        .add_axis(vec![0.0, 10.0, 20.0, 30.0]) // Second axis with two points
        .add_axis(vec![0.0, 100.0, 200.0, 300.0]) // Third axis with two points
        .build();

    // Compute selecion ids
    let sel = mesh.select_ids();
    // sel.nodes(true).is_in_sphere(
    //     &[2.0, 10.0, 100.0], // Center of the sphere
    //     1.0,               // Radius of the sphere
    // ).index;

    // Return Ok to indicate successful execution
    Ok(())
}
