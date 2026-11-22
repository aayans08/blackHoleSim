//! Planetary-satellite mean elements and physical parameters from JPL SSD
//! (https://ssd.jpl.nasa.gov/sats/elem/, /sats/phys_par/) as `sat_elements.ron` (M1 Task A9).
//! A0b stub; A9 implements.

use std::path::Path;

use crate::DataError;

/// One moon's mean elements on its local Laplace plane, in the source's units.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SatRecord {
    pub naif: i32,
    pub name: String,
    pub planet_naif: i32,
    pub a_km: f64,
    pub e: f64,
    pub argp_deg: f64,
    pub m_deg: f64,
    pub i_deg: f64,
    pub node_deg: f64,
    pub n_deg_per_day: f64,
    pub p_apsides_yr: f64,
    pub p_node_yr: f64,
    pub epoch_jd_tdb: f64,
    pub laplace_ra_deg: f64,
    pub laplace_dec_deg: f64,
    pub radius_km: Option<f64>,
    pub gm_km3_s2: Option<f64>,
}

/// Reads `sat_elements.ron`.
pub fn load_sat_elements(_path: &Path) -> Result<Vec<SatRecord>, DataError> {
    todo!("A9")
}
