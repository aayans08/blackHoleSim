//! Ecliptic and equatorial frame rotations (M1 Task A1). A0b stub; A1 implements.

use glam::{DMat3, DVec3};
use uni_core::Epoch;

/// Mean obliquity of the ecliptic at J2000, ε₀ = 84381.448″ (IAU 1976; Lieske et al. 1977,
/// A&A 58, 1).
pub const OBLIQUITY_J2000_RAD: f64 = 84_381.448 * std::f64::consts::PI / (180.0 * 3600.0);

/// Rotates a vector from the J2000 ecliptic frame to ICRS (about +x by ε₀).
pub fn ecliptic_j2000_to_icrs(_v: DVec3) -> DVec3 {
    todo!("A1")
}

/// Precession from the mean equator and equinox of date to J2000 (IAU 1976 ζ_A, z_A, θ_A).
pub fn precession_to_j2000(_t: Epoch) -> DMat3 {
    todo!("A1")
}

/// ICRS position [m] of ecliptic-of-date longitude `lon`, latitude `lat` [rad] at `dist_m`.
pub fn ecliptic_of_date_to_icrs(_lon: f64, _lat: f64, _dist_m: f64, _t: Epoch) -> DVec3 {
    todo!("A1")
}
