//! `uni-bake horizons` (M1 Task A9): JPL Horizons spot values → `data/golden/horizons_spots.ron` + `sky_spots.ron`.
//!
//! Planned command line (the plan's; A9 owns and parses it):
//! `uni-bake horizons --plan <built-in list> --out data/golden/horizons_spots.ron --sky data/golden/sky_spots.ron`.
//! A0b stub; A9 implements.

use std::ffi::OsString;

/// Runs `uni-bake horizons` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A9")
}
