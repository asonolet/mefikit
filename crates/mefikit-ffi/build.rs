use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");

    cxx_build::bridge("src/ffi.rs")
        .std("c++17")
        .compile("mefikit-cxxbridge");

    // cxx writes its generated header into a build-script-hash directory
    // (target/<profile>/build/mefikit-ffi-<hash>/out/cxxbridge/include/...), which nothing
    // outside cargo can predict. Publish the public headers and the cxx runtime source in
    // one stable tree, next to the libmefikit_ffi library itself, so that the CMake
    // integration and the docs only ever mention target/<profile>/cxxbridge.
    let root = public_cxxbridge_dir();
    let include = root.join("include");
    let out = Path::new(&out_dir).join("cxxbridge");
    let generated_h = out
        .join("include/mefikit-ffi/src/ffi.rs.h")
        .canonicalize()
        .expect("cxx generated ffi.rs.h");
    let generated_cc = out
        .join("sources/mefikit-ffi/src/ffi.rs.cc")
        .canonicalize()
        .expect("cxx generated ffi.rs.cc");

    stage(
        generated_h,
        include.join("mefikit-ffi/src/ffi.rs.h"),
        "cxx's generated header",
    );
    stage(
        Path::new("include/mefikit/mefikit.hpp").to_path_buf(),
        include.join("mefikit/mefikit.hpp"),
        "the umbrella header",
    );

    // cxx splits its C++ side in two, and both halves were compiled into the Rust
    // library where C++ mangled symbols are not exported:
    //   * ffi.rs.cc, the generated glue declaring every function of the bridge,
    //   * cxx.cc, the out-of-line part of cxx's runtime (rust::String, rust::Error,
    //     Slice's internals, ...).
    // Two spellings of the same runtime header: `rust/cxx.h` is what our umbrella
    // header includes, `cxx.h` is what the staged cxx.cc reaches through its own
    // relative `#include "../include/cxx.h"`.
    let cxx_h = out
        .join("include/rust/cxx.h")
        .canonicalize()
        .expect("cxx stages rust/cxx.h");
    stage(
        cxx_h.clone(),
        include.join("rust/cxx.h"),
        "cxx's C++ runtime header",
    );
    stage(
        include
            .join("rust/cxx.h")
            .canonicalize()
            .expect("rust/cxx.h is staged"),
        include.join("cxx.h"),
        "cxx's C++ runtime header",
    );

    // A C++ consumer therefore has to compile its own copy of both; staging the exact
    // files that were linked in keeps the two sides in step.
    stage(
        generated_cc,
        root.join("src/ffi.rs.cc"),
        "cxx's generated glue",
    );
    stage(
        cxx_runtime(&cxx_h),
        root.join("src/cxx.cc"),
        "cxx's C++ runtime",
    );

    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=include/mefikit/mefikit.hpp");
    println!("cargo:public_include={}", include.display());
}

fn stage(from: PathBuf, to: PathBuf, what: &str) {
    if let Some(parent) = to.parent() {
        std::fs::create_dir_all(parent).expect("can create the cxxbridge tree");
    }
    std::fs::copy(&from, &to)
        .unwrap_or_else(|e| panic!("cannot stage {what} to {}: {e}", to.display()));
}

/// `<target>/<profile>/cxxbridge`, derived from `OUT_DIR` so it needs no
/// `CARGO_TARGET_DIR` bookkeeping on our side.
fn public_cxxbridge_dir() -> PathBuf {
    let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR is set by cargo"));
    // OUT_DIR = <target>/<profile>/build/<pkg>-<hash>/out
    let profile_dir = out_dir
        .ancestors()
        .nth(3)
        .expect("OUT_DIR has at least four components");
    profile_dir.join("cxxbridge")
}

/// Path of `src/cxx.cc` in the cxx crate that `cxx_h` was taken from.
///
/// The header cxx stages into `OUT_DIR` is a symlink into the vendored cxx crate,
/// so resolving it names that exact version: the runtime source staged next to the
/// generated glue is always the one that was linked into the Rust library, and not
/// whichever version happens to be newest in the registry.
fn cxx_runtime(cxx_h: &Path) -> PathBuf {
    let crate_dir = cxx_h
        .ancestors()
        .nth(2)
        .expect("cxx.h lives two levels below the cxx crate root");
    let source = crate_dir.join("src/cxx.cc");
    assert!(
        source.is_file(),
        "cxx's C++ runtime is missing at {}",
        source.display()
    );
    source
}
