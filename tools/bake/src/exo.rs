//! `uni-bake exo` (M1 Task A7): NASA Exoplanet Archive `pscomppars` CSV → `data/baked/exoplanets.ron`.
//!
//! Planned command line (the plan's; A7 owns and parses it):
//! `uni-bake exo --input data/cache/exoplanets/pscomppars.csv --stars data/baked/athyg_m10.bin --names data/baked/athyg_names.ron --out data/baked/exoplanets.ron`.
//! A0b stub; A7 implements.

use std::ffi::OsString;

/// Runs `uni-bake exo` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A7")
}
