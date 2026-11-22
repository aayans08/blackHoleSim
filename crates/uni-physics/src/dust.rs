//! Comet dust tails: Finson & Probstein 1968 (ApJ 154, 327) syndynes and synchrones, with β
//! from Burns, Lamy & Soter 1979 (Icarus 40, 1) (M1 Task A3). A0b stub; A3 implements.

use glam::DVec3;
use uni_core::Epoch;

use crate::kepler::Conic;

/// The grid of grains to emit.
#[derive(Clone, Copy, Debug)]
pub struct DustParams {
    pub beta_min: f64,
    /// Below 1 (repulsive β ≥ 1 is excluded).
    pub beta_max: f64,
    pub n_beta: u32,
    pub max_age_days: f64,
    pub n_age: u32,
}

/// One grain at the observation time.
#[derive(Clone, Copy, Debug)]
pub struct DustGrain {
    /// Position relative to the nucleus [m], ICRS axes.
    pub pos_rel_nucleus_m: DVec3,
    /// Radiation-pressure to gravity ratio.
    pub beta: f64,
    /// Time since emission [s].
    pub age_s: f64,
    /// Relative brightness weight (production × size distribution).
    pub weight: f64,
}

/// Grains of `comet` at `t_obs`; `production(r_h_au)` is the relative dust production at
/// heliocentric distance r_h.
pub fn finson_probstein(
    _comet: &Conic,
    _t_obs: Epoch,
    _p: &DustParams,
    _production: &dyn Fn(f64) -> f64,
) -> Vec<DustGrain> {
    todo!("A3")
}

/// β = 5.7×10⁻⁵ Q_pr / (ρ[g cm⁻³] · s[cm]) for a grain of `radius_m` and `density_kg_m3`.
pub fn beta_for_grain(_radius_m: f64, _density_kg_m3: f64, _q_pr: f64) -> f64 {
    todo!("A3")
}
