//! `uni-bake sbdb` (M1 Task A9): JPL Small-Body Database query results → `comets.ron`, `asteroids.bin` + `asteroid_names.txt`.
//!
//! A9 defines and parses its options.
//! A0b stub; A9 implements.

use std::ffi::OsString;

/// Runs `uni-bake sbdb` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A9")
}
