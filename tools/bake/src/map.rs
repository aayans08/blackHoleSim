//! `uni-bake map` (M1 Task A8): one surface map (JPEG/PNG/TIFF) → a `UNITEX1` BC mip chain plus its 2k uncompressed fallback.
//!
//! Planned command line (the plan's; A8 owns and parses it):
//! `uni-bake map --input <jpg|png|tif> --body Earth --naif 399 --kind albedo --lon0 -180 --geometric-albedo 0.367 [--brdf lambert|lunar-lambert] [--clouds <map>] --out data/baked/maps/earth_albedo.big`.
//! A0b stub; A8 implements.

use std::ffi::OsString;

/// Runs `uni-bake map` with the options after the subcommand name.
pub fn run(_options: &[OsString]) -> anyhow::Result<()> {
    todo!("A8")
}
