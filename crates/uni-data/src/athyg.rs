//! AT-HYG v4.0 stars with V ≤ 10 (D. Nash, astronexus; CC BY-SA 4.0) (M1 Task A7).
//! `uni-bake athyg` writes the file. A0b stub; A7 implements.
//!
//! Binary format v1, little-endian: magic `b"UNIATHY\0"`, `u32` version 1, `u32` count, then
//! `count` × 28-byte [`Star3`] records.

use std::path::Path;

use crate::DataError;

/// One star as stored on disk (B4 builds its own vec4-only GPU record).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Star3 {
    /// Position [pc], ICRS, origin SSB, at [`STAR3_EPOCH_JD_TDB`].
    pub pos_pc: [f32; 3],
    /// Absolute magnitude, Johnson V (Tycho rows converted).
    pub abs_mag_v: f32,
    /// f16 pairs: (vx, vy), (vz, B−V); velocities [km/s], ICRS.
    pub vel_bv: [u32; 2],
    /// Low 22 bits: AT-HYG id (or `ExoSystem` index for archive hosts); flag bits above.
    pub id_flags: u32,
}

/// Size of one [`Star3`] record [bytes].
pub const STAR3_SIZE: usize = 28;
/// Epoch of [`Star3::pos_pc`] [JD TDB]. AT-HYG: "epoch + equinox 2000.0" (version-info.md).
pub const STAR3_EPOCH_JD_TDB: f64 = 2451545.0;
/// The star has a proper name or Bayer/Flamsteed designation in `athyg_names.ron`.
pub const FLAG_NAMED: u32 = 1 << 22;
/// The star hosts known exoplanets.
pub const FLAG_EXO_HOST: u32 = 1 << 23;
/// AT-HYG computed no Cartesian velocity (missing distance or proper motion): stored as 0.
pub const FLAG_NO_VELOCITY: u32 = 1 << 24;
/// Radial velocity missing: tangential motion only.
pub const FLAG_NO_RV: u32 = 1 << 25;
/// Built by uni-sim from the NASA Exoplanet Archive (review I11); never in the baked file.
pub const FLAG_FROM_ARCHIVE: u32 = 1 << 26;

impl Star3 {
    /// Velocity [km/s], ICRS.
    pub fn velocity_kms(&self) -> glam::Vec3 {
        todo!("A7")
    }

    /// Johnson B−V colour index.
    pub fn bv(&self) -> f32 {
        todo!("A7")
    }

    /// AT-HYG id (low 22 bits of `id_flags`).
    pub fn athyg_id(&self) -> u32 {
        todo!("A7")
    }

    /// Flag bits of `id_flags`.
    pub fn flags(&self) -> u32 {
        todo!("A7")
    }

    pub fn is_named(&self) -> bool {
        todo!("A7")
    }

    pub fn is_exo_host(&self) -> bool {
        todo!("A7")
    }
}

/// Reads a baked `athyg_m10.bin`. Missing → [`DataError::NotFound`]; wrong magic, version or
/// length → [`DataError::BadFormat`].
pub fn load_athyg(_path: &Path) -> Result<Vec<Star3>, DataError> {
    todo!("A7")
}

/// Writes `s` as a baked `athyg_m10.bin`, replacing any existing file.
pub fn write_athyg(_path: &Path, _s: &[Star3]) -> Result<(), DataError> {
    todo!("A7")
}
