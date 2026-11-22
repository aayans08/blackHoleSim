//! Exoplanet systems from the NASA Exoplanet Archive `pscomppars` table (`exoplanets.ron`)
//! (M1 Task A7). A0b stub; A7 implements.

use std::path::Path;

use crate::DataError;

/// One planet.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExoPlanet {
    pub name: String,
    pub a_au: Option<f64>,
    pub e: Option<f64>,
    pub i_deg: Option<f64>,
    /// Archive `pl_orblper` (A7 records whether it is ω of the planet or of the star).
    pub argp_deg: Option<f64>,
    pub period_d: Option<f64>,
    pub t_transit_jd: Option<f64>,
    pub mass_mjup: Option<f64>,
    pub radius_rjup: Option<f64>,
    pub teq_k: Option<f64>,
    /// Discovery method.
    pub method: String,
    /// Discovery year.
    pub year: u16,
}

/// One host star and its planets.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ExoSystem {
    pub host: String,
    /// Index of the matched star in `athyg_m10.bin`; `None` if the host is not in AT-HYG.
    pub star_index: Option<u32>,
    pub ra_deg: f64,
    pub dec_deg: f64,
    pub dist_pc: f64,
    /// Archive `sy_vmag`.
    pub v_mag: Option<f64>,
    pub teff_k: Option<f64>,
    pub radius_rsun: Option<f64>,
    pub planets: Vec<ExoPlanet>,
}

/// Reads `exoplanets.ron`.
pub fn load_exoplanets(_path: &Path) -> Result<Vec<ExoSystem>, DataError> {
    todo!("A7")
}
