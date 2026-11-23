//! Orbit lines (Task C5; Task C0 wrote the type and the `exo_orbit_points` stub).
//!
//! 256-point polylines of the current osculating conic (`Conic::from_state`), projected in f64
//! from camera-relative positions and split at the near plane, 1 pt grey at 35 % alpha
//! (selected: cyan).
//!
//! **Exoplanet orbits** (ruling 13, review m9) use the sky-plane frame of Winn 2010
//! ("Transits and Occultations", eqs. 1–2): +Z toward the observer (from the host toward the
//! Sun), X along the node line at position angle Ω from celestial north, Y completing the frame;
//! z = r sin(ω + f) sin i, so the planet transits at f = 90° − ω. Lead ruling B.2: the archive's
//! `argp_deg` is taken as the PLANET's argument of periastron (the archive does not say whose;
//! card note `theme::NOTE_PERIASTRON_CONVENTION`); ω = 90° when it is missing or e < 0.01, so the
//! planet is at f = 0 at `t_transit_jd`. The archive has no Ω, so every M1 exoplanet orbit is
//! drawn with Ω = 0° and dashed (with `theme::NOTE_NODE_UNKNOWN`), and i = 90° when the
//! inclination is missing (`theme::NOTE_INCLINATION_UNKNOWN`). The solid case is tested on a
//! synthetic system.

use glam::{DVec3, Vec2};

use crate::theme::LabelTone;

/// One orbit polyline, in output points.
#[derive(Clone, Debug)]
pub struct OrbitLine {
    pub pts: Vec<Vec2>,
    /// Drawn dashed: the orientation is assumed (exoplanets: Ω or i unknown).
    pub dashed: bool,
    pub tone: LabelTone,
}

/// `n` points [m] of planet `planet` of `sys`'s orbit around its host, relative to the host,
/// ICRS axes, with the sky-plane +Z along `-host_icrs_dir` (toward the Sun); and whether its
/// orientation is assumed (module docs).
pub fn exo_orbit_points(
    sys: &uni_data::ExoSystem,
    planet: usize,
    host_icrs_dir: DVec3,
    n: usize,
) -> (Vec<DVec3>, bool) {
    let _ = (sys, planet, host_icrs_dir, n);
    todo!("C5")
}
