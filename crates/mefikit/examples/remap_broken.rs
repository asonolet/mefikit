use mefikit::prelude as mf;
use std::path::Path;

#[cfg(feature = "io")]
fn main() -> Result<(), mf::MefikitIOError> {
    use mefikit::tools::{Measurable, Transfer};

    let mesh_file = Path::new("/home/catA/as259691/Codes/mefikit/tmp/mesh_27.med");
    let mut m = mf::read(mesh_file)?;
    let remap = mf::transfer::ConservativeP0Transfer::new(&m.view(), &m.view());
    m.measure_update("Measure", None);
    let f1 = m.field("Measure", None).unwrap();
    let f2 = remap.apply(&f1, mf::FieldNature::Intensive, 0.0);
    Ok(())
}
