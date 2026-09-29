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
    // A C++ consumer therefore has to compile its own copy of both; staging the exact
    // files that were linked in keeps the two sides in step.
    stage(
        generated_cc,
        root.join("src/ffi.rs.cc"),
        "cxx's generated glue",
    );
    stage(
        locate_cxx_runtime(),
        root.join("src/cxx.cc"),
        "cxx's C++ runtime",
    );
    // Two spellings of the same runtime header: `rust/cxx.h` is what our umbrella
    // header includes, `cxx.h` is what the staged cxx.cc reaches through its own
    // relative `#include "../include/cxx.h"`.
    stage(
        out.join("include/rust/cxx.h")
            .canonicalize()
            .expect("cxx stages rust/cxx.h"),
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

/// Finds `src/cxx.cc` inside the vendored `cxx` crate in the cargo registry.
///
/// cxx publishes the path of its header to build scripts through
/// `DEP_CXXBRIDGE1_HEADER`, but not the path of its runtime source, and that
/// variable is only set for crates that depend on cxx as a *build* dependency.
/// Going through the registry is what every cxx CMake integration does anyway;
/// doing it once here means users never have to.
fn locate_cxx_runtime() -> PathBuf {
    let cargo_home = std::env::var_os("CARGO_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cargo")))
        .expect("CARGO_HOME or HOME is set");

    let registry_src = cargo_home.join("registry/src");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&registry_src)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .flat_map(|entry| {
            std::fs::read_dir(entry.path())
                .into_iter()
                .flatten()
                .filter_map(|inner| inner.ok())
                .map(|inner| inner.path())
        })
        .filter(|dir| {
            dir.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("cxx-"))
        })
        .map(|dir| dir.join("src/cxx.cc"))
        .filter(|path| path.is_file())
        .collect();
    // Highest version wins; the plain string sort is right for cxx's version scheme.
    candidates.sort();

    candidates.pop().unwrap_or_else(|| {
        panic!(
            "cannot find cxx's C++ runtime (cxx.cc) under {}.\n\
             mefikit-ffi needs it so C++ consumers can link rust::String, rust::Error and \
             friends. This usually means the cargo registry is not where CARGO_HOME points.",
            registry_src.display()
        )
    })
}
