use mefikit::RegularUMeshBuilder;
use mefikit::io;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    let mesh = RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0])
        .add_axis(vec![0.0, 5.0])
        .add_axis(vec![0.0, 10.0])
        .build();
    let (submesh, _, _) = mesh.compute_submesh(None);

    io::write(&Path::new("examples/submesh.vtk"), submesh.view())
}
