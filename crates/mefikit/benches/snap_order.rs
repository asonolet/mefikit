use mefikit::prelude as mf;

fn make_snap_input(n: usize) -> (mf::UMesh, mf::UMesh) {
    let m1 = (n as f64).powf(1. / 3.) as usize;
    let m2 = m1;
    let m3 = n / (m1 * m2);
    let x1: Vec<f64> = (0..=m1).map(|i| i as f64 / (m1 as f64)).collect();
    let x2: Vec<f64> = (0..=m2).map(|i| i as f64 / (m2 as f64)).collect();
    let x3: Vec<f64> = (0..=m3).map(|i| i as f64 / (m3 as f64)).collect();
    let mesh1 = mf::RegularUMeshBuilder::new()
        .add_axis(x1)
        .add_axis(x2)
        .add_axis(x3)
        .build();
    let mesh2 = mesh1.clone();
    (mesh1, mesh2)
}

#[scaling::bench_scaling(nmin = 10000, make_input = make_snap_input)]
fn snap_order(input: &mut (mf::UMesh, mf::UMesh)) {
    mf::snap::snap(&mut input.0, &input.1.view(), 1e-12);
}

fn main() -> Result<(), scaling::RegistrationError> {
    scaling::Config::default().run_and_print()
}
