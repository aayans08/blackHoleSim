//! JPL Horizons spot values (DE440/441) used as test goldens: `data/golden/horizons_spots.ron`
//! (state vectors) and `sky_spots.ron` (observer tables) (M1 Task A9, review I9).
//! A0b stub; A9 implements.

use std::path::Path;

use crate::DataError;

/// One geometric state vector, ICRF.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct EphemSpot {
    /// Horizons target (NAIF id).
    pub target: i32,
    /// Horizons centre (NAIF id).
    pub center: i32,
    pub jd_tdb: f64,
    pub pos_km: [f64; 3],
    pub vel_km_s: [f64; 3],
}

/// One observer-table row (astrometric; Horizons quantities 1, 9, 19, 20, 24).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SkySpot {
    pub target: String,
    /// NAIF id of the body at whose centre the observer sits (Horizons `CENTER='500@<id>'`).
    pub observer: i32,
    pub jd_utc: f64,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub delta_au: f64,
    pub apmag: Option<f64>,
    pub r_au: Option<f64>,
    pub phase_deg: Option<f64>,
}

/// Reads `horizons_spots.ron`.
pub fn load_ephem_spots(_path: &Path) -> Result<Vec<EphemSpot>, DataError> {
    todo!("A9")
}

/// Reads `sky_spots.ron`.
pub fn load_sky_spots(_path: &Path) -> Result<Vec<SkySpot>, DataError> {
    todo!("A9")
}
