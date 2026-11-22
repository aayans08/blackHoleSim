//! Comets (`comets.ron`) and asteroids (`asteroids.bin` + `asteroid_names.txt`) from the JPL
//! Small-Body Database (M1 Task A9). A0b stub; A9 implements.

use std::path::Path;

use crate::DataError;

/// One comet's osculating elements (J2000 ecliptic).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Comet {
    pub name: String,
    pub spkid: i32,
    pub q_au: f64,
    pub e: f64,
    pub i_deg: f64,
    pub node_deg: f64,
    pub argp_deg: f64,
    pub tp_jd_tdb: f64,
    pub epoch_jd_tdb: f64,
    /// Total-magnitude parameters (m = M1 + 5 log₁₀Δ + K1 log₁₀r).
    pub m1: Option<f64>,
    pub k1: Option<f64>,
    pub diameter_km: Option<f64>,
}

/// Reads `comets.ron`.
pub fn load_comets(_path: &Path) -> Result<Vec<Comet>, DataError> {
    todo!("A9")
}

/// One asteroid's osculating elements (J2000 ecliptic) and photometry, as stored on disk.
///
/// The plan calls this record 72 B, but the fields as listed are 7 × f64 + 4 × f32 + 2 × u32
/// = 80 B with no padding (A0b report; the lead settles it with A9).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AsteroidRecord {
    pub a_au: f64,
    pub e: f64,
    pub i_deg: f64,
    pub node_deg: f64,
    pub argp_deg: f64,
    pub m0_deg: f64,
    pub epoch_jd_tdb: f64,
    pub h: f32,
    pub g: f32,
    /// NaN when unknown.
    pub diameter_km: f32,
    /// NaN when unknown.
    pub albedo: f32,
    pub spkid: u32,
    /// Line of the name in `asteroid_names.txt`.
    pub name_index: u32,
}

/// Reads `asteroids.bin` (magic `b"UNIAST1\0"`) and its names file.
pub fn load_asteroids(
    _bin: &Path,
    _names: &Path,
) -> Result<(Vec<AsteroidRecord>, Vec<String>), DataError> {
    todo!("A9")
}
