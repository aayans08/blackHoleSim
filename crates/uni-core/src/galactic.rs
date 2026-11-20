//! ICRS ↔ Galactic coordinate conversion (Hipparcos, ESA 1997, Vol. 1 §1.5.3).

use glam::DVec3;

/// ICRS → Galactic rotation (ESA 1997, Vol. 1 §1.5.3).
/// Rows are the galactic x, y, z axes expressed in ICRS.
pub const ICRS_TO_GALACTIC: [[f64; 3]; 3] = [
    [-0.054_875_560_416_215_4, -0.873_437_090_234_885_0, -0.483_835_015_548_713_2],
    [ 0.494_109_427_875_583_7, -0.444_829_629_960_011_2,  0.746_982_244_497_218_9],
    [-0.867_666_149_019_004_7, -0.198_076_373_431_201_5,  0.455_983_776_175_066_9],
];

/// Rotates an ICRS vector into Galactic axes.
pub fn icrs_to_galactic(v: DVec3) -> DVec3 {
    todo!()
}

/// Unit vector in ICRS for right ascension and declination [rad].
pub fn radec_to_icrs(ra_rad: f64, dec_rad: f64) -> DVec3 {
    todo!()
}
