//! Compiles the cxx bridge and publishes the C++ artifacts in a stable tree.
//!
//! `cxx-build` writes the generated header and glue into a cargo hash directory
//! under `OUT_DIR`. This script copies them, together with cxx's runtime source
//! and our umbrella header, to `target/<profile>/cxxbridge`, which is what the
//! CMake example (and downstream users) can rely on.

use std::path::{Path, PathBuf};

fn main() {
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR is set by cargo");

    cxx_build::bridge("src/ffi.rs")
        .std("c++17")
        .compile("mefikit-cxxbridge");

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
