# Vendored dependencies (`vendor/`)

`vendor/lzma-sys/` is **generated and version-controlled**: it is produced by
`tools/update_liblzma_sys.sh` from the pinned upstream crates.io archive and
committed as-is. To bump the pinned version, update the constants in the
script, run it, and commit the regenerated tree.

```
tools/update_liblzma_sys.sh        # (re)generate vendor/lzma-sys
tools/update_liblzma_sys.sh --verify   # confirm the tree matches upstream
```

## What is `vendor/lzma-sys`?

A vendored build of the `liblzma-sys` crate, version 0.4.9, from the alive
upstream project

- repository: <https://github.com/Portable-Network-Archive/liblzma-rs>
  (a maintained fork of <https://github.com/alexcrichton/xz2-rs>, which is
  where the historical `lzma-sys` crate that vtkio consumes lives),
- crates.io archive: `https://static.crates.io/crates/liblzma-sys/liblzma-sys-0.4.9.crate`,
- SHA-256: `7c7e3581f367a78d7b7e7ae948d023310556f0cfc13156c2e4e00e25616492b9`.

It is shipped under the historical `lzma-sys` name/version (0.1.20) because
vtkio's `xz2 ^0.1.18` requires that package name, while the maintained
implementation lives under the `liblzma-sys` name.

## Why

mefikit's `io` feature links `vtkio` (a stale 0.7.0-rc), which depends on
`xz2 0.1.7` → `lzma-sys`, and `lzma-sys` declares `links = "lzma"`. mefikit's
`gpu`/`gpu-cpu` features link `cubecl`, whose `tracel-llvm-bundler` (a build
dependency of `cubecl-llvm`) depends on `liblzma` → `liblzma-sys`, which also
declares `links = "lzma"`. Cargo forbids two packages declaring the same
`links` value in one build graph, so the two cannot coexist from crates.io.

The fix is applied on the frozen side (vtkio's `lzma-sys`): the modern,
maintained `liblzma-sys` implementation is vendored under the `lzma-sys`
package name with its `links` renamed, so the conflict disappears while
cubecl's `liblzma-sys` stays completely pristine on crates.io.

`[patch.crates-io]` in the workspace root points `lzma-sys` at this directory.

## Local modifications (relative to the upstream archive)

1. `Cargo.toml`: package `name` `"liblzma-sys"` → `"lzma-sys"`.
2. `Cargo.toml`: package `version` `"0.4.9"` → `"0.1.20"` (resolution target
   for vtkio's `xz2 ^0.1.18`).
3. `Cargo.toml`: `links = "lzma"` → `links = "lzma-vtkio"` (ends the
   collision; nothing on the vtkio side reads `DEP_LZMA_*`, so the rename is
   build-safe).
4. `Cargo.toml`: `default = ["bindgen"]` → `default = ["bindgen", "static"]`
   (compile the bundled xz rather than depending on a system `liblzma` via
   pkg-config; `bindgen` here only selects the *committed* bindings in
   `src/bindgen.rs`, so no libclang is needed at build time).
5. `Cargo.toml`: `[lib] name` `"liblzma_sys"` → `"lzma_sys"` (the crate name
   that `xz2` references with `use lzma_sys`).

Everything else is byte-for-byte the upstream archive (minus `.cargo*`/`Cargo.lock`).

## License

The vendored contents are the upstream crate's own code under its dual
`LICENSE-APACHE` / `LICENSE-MIT` (kept in the vendored tree).
