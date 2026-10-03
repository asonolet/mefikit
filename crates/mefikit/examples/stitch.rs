//! Profiling driver for [`mefikit::tools::stitch`].
//!
//! Runs a few representative assemblies in a loop so that `cargo flamegraph` has enough samples:
//!
//! ```bash
//! cargo flamegraph --profile flame --example stitch
//! ```

use std::time::Instant;

use mefikit::mesh::UMesh;
use mefikit::tools::{RegularUMeshBuilder, stitch};

fn block(xs: &[f64], ys: &[f64], zs: &[f64]) -> UMesh {
    RegularUMeshBuilder::new()
        .add_axis(xs.to_vec())
        .add_axis(ys.to_vec())
        .add_axis(zs.to_vec())
        .build()
}

fn axis(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    (0..=n)
        .map(|i| lo + (hi - lo) * i as f64 / n as f64)
        .collect()
}

/// A wall of blocks side by side: many small interface regions.
fn wall(k: usize, n: usize) -> Vec<UMesh> {
    (0..k)
        .map(|i| {
            block(
                &axis(i as f64, i as f64 + 1.0, n),
                &axis(0.0, 1.0, n),
                &axis(0.0, 1.0, 2 * n),
            )
        })
        .collect()
}

/// A volume split into eight blocks refined differently: a few large interface regions.
fn eight_blocks(n: usize) -> Vec<UMesh> {
    let mut meshes = Vec::new();
    for k in 0..2 {
        for j in 0..2 {
            for i in 0..2 {
                let m = if (i + j + k) % 2 == 0 { n } else { 2 * n };
                meshes.push(block(
                    &axis(i as f64, i as f64 + 1.0, m),
                    &axis(j as f64, j as f64 + 1.0, m),
                    &axis(k as f64, k as f64 + 1.0, m),
                ));
            }
        }
    }
    meshes
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(12);
    let reps: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);

    let cases: [(&str, Vec<UMesh>); 2] =
        [("wall_4", wall(4, n)), ("eight_blocks", eight_blocks(n))];
    for (name, meshes) in cases {
        let cells: usize = meshes.iter().map(|m| m.num_elements()).sum();
        let views: Vec<_> = meshes.iter().map(|m| m.view()).collect();
        let now = Instant::now();
        for _ in 0..reps {
            std::hint::black_box(stitch(&views, 1e-9).unwrap());
        }
        let elapsed = now.elapsed().as_secs_f64() / reps as f64;
        println!("{name}: {cells} cells in {elapsed:.6}s");
    }
}
