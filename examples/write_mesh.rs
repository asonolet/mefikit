use mefikit as mf;
use ndarray as nd;
use ndarray::prelude::*;
use std::path::Path;

fn regular_hexa() -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .add_axis(vec![0.0, 10.0, 20.0, 30.0])
        .add_axis(vec![0.0, 100.0, 200.0])
        .build()
}

fn regular_quad() -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .add_axis(vec![0.0, 10.0, 20.0, 30.0])
        .build()
}

fn regular_line() -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis(vec![0.0, 1.0, 2.0, 3.0, 4.0])
        .build()
}

fn triangles_mesh() -> mf::UMesh {
    let mut mesh = mf::UMesh::new(arr2(&[[0., 0.], [1., 0.], [1., 1.], [0., 1.]]));
    mesh.add_regular_block(mf::ElementType::TRI3, nd::arr2(&[[0, 1, 3], [1, 2, 3]]));
    mesh
}

fn tetra_mesh() -> mf::UMesh {
    let mut mesh = mf::UMesh::new(arr2(&[
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]));
    mesh.add_regular_block(
        mf::ElementType::TET4,
        nd::arr2(&[[0, 1, 3, 4], [1, 2, 3, 4]]),
    );
    mesh
}

fn triangle_tetra_mesh() -> mf::UMesh {
    let mut mesh = mf::UMesh::new(arr2(&[
        [0., 0., 0.],
        [1., 0., 0.],
        [1., 1., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
    ]));
    mesh.add_regular_block(mf::ElementType::TRI3, nd::arr2(&[[0, 1, 3], [1, 2, 3]]));
    mesh.add_regular_block(
        mf::ElementType::TET4,
        nd::arr2(&[[0, 1, 3, 4], [1, 2, 3, 4]]),
    );
    mesh
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Dummy mesh

    mf::io::write(Path::new("examples/hexa.vtu"), regular_hexa().view())?;
    println!("Mesh saved to hexa.vtu");

    mf::io::write(Path::new("examples/quad.vtu"), regular_quad().view())?;
    println!("Mesh saved to quad.vtu");

    mf::io::write(Path::new("examples/line.vtu"), regular_line().view())?;
    println!("Mesh saved to line.vtu");

    mf::io::write(Path::new("examples/triangles.vtu"), triangles_mesh().view())?;
    println!("Mesh saved to triangles.vtu");

    mf::io::write(Path::new("examples/tetra.vtu"), tetra_mesh().view())?;
    println!("Mesh saved to tetra.vtu");

    mf::io::write(
        Path::new("examples/triangles_tetra.vtu"),
        triangle_tetra_mesh().view(),
    )?;
    println!("Mesh saved to triangles_tetra.vtu");

    // save_mesh(Path::new("examples/out.yaml"), &mesh)?;
    // println!("Mesh saved to out.yaml");

    // let umesh2 = load_mesh(Path::new("examples/out.json"))?;

    // println!("Mesh loaded from out.json");
    // println!("{umesh2:?}");

    Ok(())
}
