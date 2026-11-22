//! Moons from JPL SSD mean elements on the local Laplace plane
//! (https://ssd.jpl.nasa.gov/sats/elem/) (M1 Task A2). JPL states these are not for precise
//! ephemerides; the info card labels them. A0b stub; A2 implements.

use glam::DVec3;
use uni_core::Epoch;

use crate::NaifId;

/// One moon's mean elements, in SI-friendly units (uni-sim maps `uni_data::SatRecord` to this).
#[derive(Clone, Debug, PartialEq)]
pub struct SatElements {
    pub naif: NaifId,
    pub name: String,
    /// NAIF id of the planet (body centre) it orbits.
    pub planet: NaifId,
    /// Semi-major axis [m].
    pub a_m: f64,
    pub e: f64,
    /// Argument of periapsis [deg].
    pub argp_deg: f64,
    /// Mean anomaly at `epoch` [deg].
    pub m_deg: f64,
    /// Inclination to the Laplace plane [deg].
    pub i_deg: f64,
    /// Ascending node on the Laplace plane [deg].
    pub node_deg: f64,
    /// Mean motion [deg/day].
    pub n_deg_per_day: f64,
    /// Apsidal precession period [yr].
    pub p_apsides_yr: f64,
    /// Nodal precession period [yr].
    pub p_node_yr: f64,
    pub epoch: Epoch,
    /// Laplace-plane pole, ICRS right ascension [deg].
    pub laplace_ra_deg: f64,
    /// Laplace-plane pole, ICRS declination [deg].
    pub laplace_dec_deg: f64,
}

/// Position relative to the planet's centre [m], ICRS axes; `gm_planet` [m³/s²].
pub fn planetocentric_icrs(_el: &SatElements, _gm_planet: f64, _t: Epoch) -> DVec3 {
    todo!("A2")
}
