//! How much of the input the bindings check before handing it to mefikit.
//!
//! mefikit is written for Rust callers and states what it assumes about its
//! input as `assert!`s: a node index past the end of the coordinate array, a
//! poly offset table that does not match the connectivity, a non-finite
//! coordinate. Reached from C++ those abort the process, so the bindings check
//! them first and return an error instead.
//!
//! Two of those checks walk the whole buffer, which for a million-node mesh is
//! an extra pass over tens of megabytes. The copies mefikit needs anyway are not
//! the expensive part, so a caller that already trusts its input can turn the
//! scans off with [`set_checks`](crate::bridge::set_checks) or the
//! `MEFIKIT_FFI_CHECKS` environment variable and keep everything else.
//!
//! Nothing in `Checks::Fast` is skipped that the core would catch for free.
//! What it skips is exactly the work whose cost grows with the size of the mesh,
//! and skipping it turns a later panic into a wrong answer rather than an error.

use std::sync::atomic::{AtomicU8, Ordering};

use crate::ffi::Error;
use crate::ffi::bridge::Checks;

/// What a [`Checks`] value from C++ turned into.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a discriminant this build does not know
/// about. Falling back to a level instead would mean a C++ caller who passed
/// something invalid got a setting they did not ask for.
fn to_level(repr: u8) -> Result<u8, Error> {
    if repr == Checks::Fast.repr {
        Ok(FAST)
    } else if repr == Checks::Full.repr {
        Ok(FULL)
    } else {
        Err(Error::InvalidArgument(format!(
            "Checks discriminant {repr} is not known to this mefikit build"
        )))
    }
}

/// `UNSET` until the environment has been read, then `FAST` or `FULL`.
static LEVEL: AtomicU8 = AtomicU8::new(UNSET);
const UNSET: u8 = 0;
const FAST: u8 = 1;
const FULL: u8 = 2;

/// Whether the linear scans over coordinates and connectivity should run.
///
/// The environment is consulted once, the first time a check asks, so setting
/// the variable after startup has no effect; use
/// [`set_checks`](crate::bridge::set_checks) to change it at runtime.
#[must_use]
pub(crate) fn heavy_checks() -> bool {
    match LEVEL.load(Ordering::Relaxed) {
        FAST => return false,
        FULL => return true,
        _ => {}
    }
    let level = from_env();
    LEVEL.store(level, Ordering::Relaxed);
    level == FULL
}

/// Records the level chosen for the rest of the process.
///
/// # Errors
///
/// As [`to_level`].
pub(crate) fn set(checks: Checks) -> Result<(), Error> {
    LEVEL.store(to_level(checks.repr)?, Ordering::Relaxed);
    Ok(())
}

/// The level in force, reading the environment on first use.
#[must_use]
pub(crate) fn get() -> Checks {
    if heavy_checks() {
        Checks::Full
    } else {
        Checks::Fast
    }
}

/// An unreadable or unrecognised value keeps the safe default: a typo should
/// not quietly trade away the errors a caller is relying on.
fn from_env() -> u8 {
    from_value(std::env::var("MEFIKIT_FFI_CHECKS").ok().as_deref())
}

fn from_value(value: Option<&str>) -> u8 {
    match value.map(str::trim).map(str::to_ascii_lowercase).as_deref() {
        Some("fast" | "off" | "0" | "none") => FAST,
        _ => FULL,
    }
}

/// Puts the level back to undecided, so tests do not leak one into the next.
#[cfg(test)]
pub(crate) fn reset() {
    LEVEL.store(UNSET, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test rather than several: the level is process-wide state, and
    /// `cargo test` runs the tests in a binary concurrently.
    #[test]
    fn the_level_round_trips_and_ignores_what_it_does_not_know() {
        reset();
        assert!(heavy_checks(), "the safe default is to check everything");
        assert!(get().repr == Checks::Full.repr, "expected Full");

        set(Checks::Fast).unwrap();
        assert!(!heavy_checks());
        assert!(get().repr == Checks::Fast.repr, "expected Fast");

        set(Checks::Full).unwrap();
        assert!(heavy_checks());
        assert!(get().repr == Checks::Full.repr, "expected Full");

        // cxx enums are open, so a C++ caller can pass a discriminant this
        // build has never heard of. It has to be an error, not a silent default.
        let err = to_level(200).unwrap_err();
        assert!(
            matches!(&err, Error::InvalidArgument(m) if m.contains("200")),
            "unexpected error: {err}"
        );
        // A rejected value must leave the previous level alone.
        assert!(get().repr == Checks::Full.repr, "expected Full");

        for fast in ["fast", "off", "0", "none", "FAST", " off "] {
            assert_eq!(from_value(Some(fast)), FAST, "{fast:?} should mean fast");
        }
        for full in ["", "full", "on", "1", "true", "maybe", "fastest"] {
            assert_eq!(from_value(Some(full)), FULL, "{full:?} should mean full");
        }
        assert_eq!(from_value(None), FULL, "unset should mean full");
    }
}
