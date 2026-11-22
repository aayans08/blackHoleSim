//! Pluto system barycentre (Horizons target 9) from Meeus, *Astronomical Algorithms* 2nd ed.,
//! ch. 37 (periodic terms, 1885–2099); a two-body conic outside that window (M1 Task A1,
//! review I1). A0b stub; A1 implements.

use glam::DVec3;
use uni_core::Epoch;

/// Validity window of Meeus ch. 37 [JD TDB]: 1885-01-01 – 2099-12-31.
pub const VALID_JD_TDB: (f64, f64) = (2409542.5, 2488068.5);

/// Heliocentric ecliptic J2000 longitude l, latitude b [rad] and distance r [m]; `None`
/// outside [`VALID_JD_TDB`].
pub fn heliocentric_ecliptic_j2000(_t: Epoch) -> Option<(f64, f64, f64)> {
    todo!("A1")
}

/// Heliocentric position [m] and velocity [m/s] in ICRS axes; two-body outside the window.
pub fn heliocentric_state_icrs(_t: Epoch) -> (DVec3, DVec3) {
    todo!("A1")
}

/// Whether `t` is inside [`VALID_JD_TDB`].
pub fn in_window(_t: Epoch) -> bool {
    todo!("A1")
}
