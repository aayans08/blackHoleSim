//! `uni-bake rings` (M1 Task A8): Saturn's ring alpha PNG → the radial profile `data/baked/saturn_rings.ron`.
//!
//! Planned command line (the plan's; A8 owns and parses it):
//! `uni-bake rings --input data/cache/sss-8k/8k_saturn_ring_alpha.png --inner-km <r> --outer-km <r> --out data/baked/saturn_rings.ron`.
//! A0b stub; A8 implements.

use std::ffi::OsString;

/// Runs `uni-bake rings` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A8")
}
