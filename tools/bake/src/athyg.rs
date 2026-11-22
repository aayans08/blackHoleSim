//! `uni-bake athyg` (M1 Task A7): AT-HYG v4.0 subset (V ≤ 10) → `data/baked/athyg_m10.bin` + `athyg_names.ron`.
//!
//! Planned command line (the plan's; A7 owns and parses it):
//! `uni-bake athyg --input data/cache/athyg-m10/athyg_40_reduced_m10.csv.gz --out data/baked/athyg_m10.bin --names data/baked/athyg_names.ron`.
//! A0b stub; A7 implements.

use std::ffi::OsString;

/// Runs `uni-bake athyg` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A7")
}
