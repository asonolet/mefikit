use gungraun::{
    Callgrind, FlamegraphConfig, LibraryBenchmarkConfig, library_benchmark,
    library_benchmark_group, main,
};
use mefikit::prelude as mf;
use std::hint::black_box;

fn regular_grid(n: usize) -> mf::UMesh {
    mf::RegularUMeshBuilder::new()
        .add_axis((0..=n).map(|i| i as f64).collect::<Vec<f64>>())
        .add_axis((0..=n).map(|i| i as f64).collect::<Vec<f64>>())
        .build()
}

#[library_benchmark]
#[bench::short(regular_grid(4))]
#[bench::long(regular_grid(32))]
fn bench_submesh(mesh: mf::UMesh) {
    black_box(mf::compute_submesh(&mesh, None, None));
}

library_benchmark_group!(
    name = bench_umesh_group;
    benchmarks = bench_submesh
);

main!(
    config = LibraryBenchmarkConfig::default()
        .tool(Callgrind::default()
            .flamegraph(FlamegraphConfig::default())
        );
    library_benchmark_groups = bench_umesh_group
);
