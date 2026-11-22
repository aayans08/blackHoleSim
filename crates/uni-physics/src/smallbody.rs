//! Comet and asteroid orbits from catalogue elements (J2000 ecliptic) (M1 Task A3).
//! A0b stub; A3 implements.

use glam::DVec3;
use uni_core::Epoch;

use crate::kepler::Conic;

/// Heliocentric conic from perihelion elements (q [AU], angles [deg], T_p [JD TDB]).
pub fn conic_from_perihelion(
    _q_au: f64,
    _e: f64,
    _i_deg: f64,
    _node_deg: f64,
    _argp_deg: f64,
    _tp_jd_tdb: f64,
) -> Conic {
    todo!("A3")
}

/// Heliocentric conic from mean-anomaly elements (a [AU], angles [deg], epoch [JD TDB]).
pub fn conic_from_mean_anomaly(
    _a_au: f64,
    _e: f64,
    _i_deg: f64,
    _node_deg: f64,
    _argp_deg: f64,
    _m0_deg: f64,
    _epoch_jd_tdb: f64,
) -> Conic {
    todo!("A3")
}

/// Heliocentric position [m] and velocity [m/s] in ICRS axes of a conic whose elements are in
/// the J2000 ecliptic.
pub fn heliocentric_state_icrs(_c: &Conic, _t: Epoch) -> (DVec3, DVec3) {
    todo!("A3")
}
