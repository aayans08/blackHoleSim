//! Two-body conics (M1 Task A1): elliptic and hyperbolic Kepler solvers, universal-variable
//! propagation and conversion between elements and state vectors. A0b stub; A1 implements.

use glam::{DMat3, DVec3};
use uni_core::Epoch;

/// A two-body orbit. Angles are in the conic's reference frame (whatever frame the state
/// vectors are in); any e ≥ 0.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conic {
    /// Periapsis distance [m].
    pub q_m: f64,
    /// Eccentricity.
    pub e: f64,
    /// Inclination [rad].
    pub i_rad: f64,
    /// Longitude of the ascending node [rad].
    pub node_rad: f64,
    /// Argument of periapsis [rad].
    pub argp_rad: f64,
    /// Time of periapsis passage.
    pub tp: Epoch,
    /// Gravitational parameter of the central body [m³/s²].
    pub mu_m3s2: f64,
}

impl Conic {
    /// From classical elements with mean anomaly `m0` [rad] at `epoch` (e < 1); `a_m` [m],
    /// angles [rad], `mu` [m³/s²].
    #[allow(clippy::too_many_arguments)]
    pub fn from_mean_anomaly(
        _a_m: f64,
        _e: f64,
        _i: f64,
        _node: f64,
        _argp: f64,
        _m0: f64,
        _epoch: Epoch,
        _mu: f64,
    ) -> Self {
        todo!("A1")
    }

    /// The osculating conic of state `r` [m], `v` [m/s] at `t` (review I9: used by C5, A4,
    /// A1's Pluto and C1).
    pub fn from_state(_r: DVec3, _v: DVec3, _mu: f64, _t: Epoch) -> Self {
        todo!("A1")
    }

    /// Position [m] and velocity [m/s] at `t`, in the reference frame; any e ≥ 0.
    pub fn state(&self, _t: Epoch) -> (DVec3, DVec3) {
        todo!("A1")
    }

    /// Orbital period [s]; `None` for e ≥ 1.
    pub fn period_s(&self) -> Option<f64> {
        todo!("A1")
    }
}

/// Eccentric anomaly E [rad] for mean anomaly `m_rad`, e < 1 (E − e sin E = M).
pub fn solve_kepler(_m_rad: f64, _e: f64) -> f64 {
    todo!("A1")
}

/// Hyperbolic anomaly H for mean anomaly `m`, e > 1 (e sinh H − H = M).
pub fn solve_kepler_hyperbolic(_m: f64, _e: f64) -> f64 {
    todo!("A1")
}

/// Propagates state `r0` [m], `v0` [m/s] by `dt_s` with universal variables (Stumpff C, S);
/// any conic.
pub fn propagate_universal(_r0: DVec3, _v0: DVec3, _mu: f64, _dt_s: f64) -> (DVec3, DVec3) {
    todo!("A1")
}

/// Rotation from the perifocal frame (x to periapsis, z along the angular momentum) to the
/// reference frame: R_z(node) · R_x(i) · R_z(argp).
pub fn perifocal_to_reference(_i: f64, _node: f64, _argp: f64) -> DMat3 {
    todo!("A1")
}
