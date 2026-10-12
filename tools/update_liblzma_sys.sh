#!/usr/bin/env bash
# Generate or verify `vendor/lzma-sys`, the git-ignored vendored copy of the
# `lzma-sys` crate (this repo's rename of the upstream `liblzma-sys` 0.4.9).
#
# The vendored tree is regenerated from the pinned crates.io archive so it stays
# out of version control: vendor/lzma-sys is gitignored and must be materialised
# once after a fresh clone (and after each upstream bump pinned here) before any
# `cargo` command. See vendor/README.md for the rationale and provenance.
#
# Usage (from the repository root):
#   tools/update_liblzma_sys.sh            (re)generate vendor/lzma-sys
#   tools/update_liblzma_sys.sh --verify   check the on-disk tree matches upstream
#
# Requires: curl, tar, sha256sum, sed.

set -euo pipefail

SRC_NAME="liblzma-sys"
SRC_VERSION="0.4.9"
SRC_SHA256="7c7e3581f367a78d7b7e7ae948d023310556f0cfc13156c2e4e00e25616492b9"
# The package name/version/links we ship it under, for vtkio's `xz2 ^0.1.18`.
DST_NAME="lzma-sys"
DST_VERSION="0.1.20"
DST_LINKS="lzma-vtkio"

URL="https://static.crates.io/crates/${SRC_NAME}/${SRC_NAME}-${SRC_VERSION}.crate"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DST_DIR="$ROOT/vendor/lzma-sys"
# Files copied byte-for-byte from the archive (Cargo.toml is re-created from the
# archive copy with the documented edits below; Cargo.lock/.cargo* are dropped).
UPLOAD_FILES=(build.rs Cargo.toml.orig LICENSE-APACHE LICENSE-MIT README.md \
    config.h src wasm-shim xz)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

fetch_and_extract() {
    curl -fsSL --retry 3 -o "$WORK/archive.crate" "$URL"
    local got
    got="$(sha256sum "$WORK/archive.crate" | cut -d' ' -f1)"
    if [ "$got" != "$SRC_SHA256" ]; then
        echo "error: SHA-256 mismatch for ${SRC_NAME}-${SRC_VERSION}.crate" >&2
        echo "  expected: $SRC_SHA256" >&2
        echo "  got:      $got" >&2
        echo "  (re-pin SRC_VERSION/SRC_SHA256 in this script if upstream released again)" >&2
        return 1
    fi
    tar xzf "$WORK/archive.crate" -C "$WORK"
    SRC_DIR="$WORK/${SRC_NAME}-${SRC_VERSION}"
}

# The one-off local transformation documented in vendor/README.md.
apply_edits() {
    sed -i \
        -e "s/^name = \"${SRC_NAME}\"\$/name = \"$DST_NAME\"/" \
        -e "s/^version = \"${SRC_VERSION}\"\$/version = \"$DST_VERSION\"/" \
        -e "s/^links = \"lzma\"\$/links = \"$DST_LINKS\"/" \
        -e 's/^default = \["bindgen"\]$/default = ["bindgen", "static"]/' \
        -e "s/^name = \"${SRC_NAME//-/_}\"\$/name = \"${DST_NAME//-/_}\"/" \
        "$1"
    # Fail loudly if any of the expected edits did not take effect.
    grep -q "^name = \"$DST_NAME\"\$" "$1"
    grep -q "^version = \"$DST_VERSION\"\$" "$1"
    grep -q "^links = \"$DST_LINKS\"\$" "$1"
    grep -q '^default = \["bindgen", "static"\]$' "$1"
    grep -q "^name = \"${DST_NAME//-/_}\"\$" "$1"
}

check_edits() {
    [ -f "$1" ] || { echo "error: $1 missing" >&2; return 1; }
    grep -q "^name = \"$DST_NAME\"\$" "$1" \
        && grep -q "^version = \"$DST_VERSION\"\$" "$1" \
        && grep -q "^links = \"$DST_LINKS\"\$" "$1" \
        && grep -q '^default = \["bindgen", "static"\]$' "$1" \
        && grep -q "^name = \"${DST_NAME//-/_}\"\$" "$1"
}

if [ "${1:-}" = "--verify" ]; then
    fetch_and_extract
    local_status=0
    for f in "${UPLOAD_FILES[@]}"; do
        if ! diff -rq "$SRC_DIR/$f" "$DST_DIR/$f" >/dev/null 2>&1; then
            echo "error: $DST_DIR/$f differs from the upstream archive" >&2
            local_status=1
        fi
    done
    if ! check_edits "$DST_DIR/Cargo.toml"; then
        echo "error: $DST_DIR/Cargo.toml is missing the expected edits" >&2
        local_status=1
    fi
    if [ $local_status -eq 0 ]; then
        echo "ok: vendor/lzma-sys matches upstream ${SRC_NAME}-${SRC_VERSION} (modulo documented edits)"
    fi
    exit $local_status
fi

fetch_and_extract
rm -rf "$DST_DIR"
mkdir -p "$DST_DIR"
for f in "${UPLOAD_FILES[@]}"; do
    cp -a "$SRC_DIR/$f" "$DST_DIR/$f"
done
cp -a "$SRC_DIR/Cargo.toml" "$DST_DIR/Cargo.toml"
apply_edits "$DST_DIR/Cargo.toml"
echo "ok: vendor/lzma-sys (re)generated from ${SRC_NAME}-${SRC_VERSION}"
echo "    see vendor/README.md for details"
