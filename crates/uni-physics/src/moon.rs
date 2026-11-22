//! The Moon from Meeus ch. 47 (truncated ELP-2000/82, Chapront-Touzé & Chapront) (M1 Task A2).
//! A0b stub; A2 implements.

use glam::DVec3;
use uni_core::Epoch;

/// Geocentric ecliptic longitude λ, latitude β [rad] and distance Δ [m], mean equinox of date.
pub fn ecliptic_of_date(_t: Epoch) -> (f64, f64, f64) {
    todo!("A2")
}

/// Geocentric position [m] in ICRS axes.
pub fn geocentric_icrs(_t: Epoch) -> DVec3 {
    todo!("A2")
}
