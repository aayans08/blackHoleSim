//! ICRS ↔ Galactic coordinate conversion (Hipparcos, ESA 1997, Vol. 1 §1.5.3).

use glam::DVec3;

/// ICRS → Galactic rotation (ESA 1997, Vol. 1 §1.5.3).
/// Rows are the galactic x, y, z axes expressed in ICRS.
pub const ICRS_TO_GALACTIC: [[f64; 3]; 3] = [
    [
        -0.054_875_560_416_215_4,
        -0.873_437_090_234_885_0,
        -0.483_835_015_548_713_2,
    ],
    [
        0.494_109_427_875_583_7,
        -0.444_829_629_960_011_2,
        0.746_982_244_497_218_9,
    ],
    [
        -0.867_666_149_019_004_7,
        -0.198_076_373_431_201_5,
        0.455_983_776_175_066_9,
    ],
];

/// Rotates an ICRS vector into Galactic axes.
pub fn icrs_to_galactic(v: DVec3) -> DVec3 {
    let [gx, gy, gz] = ICRS_TO_GALACTIC.map(DVec3::from_array);
    DVec3::new(gx.dot(v), gy.dot(v), gz.dot(v))
}

/// Unit vector in ICRS for right ascension and declination [rad].
pub fn radec_to_icrs(ra_rad: f64, dec_rad: f64) -> DVec3 {
    let (sin_ra, cos_ra) = ra_rad.sin_cos();
    let (sin_dec, cos_dec) = dec_rad.sin_cos();
    DVec3::new(cos_dec * cos_ra, cos_dec * sin_ra, sin_dec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DMat3;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn galactic_center_direction() {
        // Row 0 is the galactic +x axis (towards the Galactic Centre) in ICRS.
        let gx = DVec3::from_array(ICRS_TO_GALACTIC[0]);
        let ra = gx.y.atan2(gx.x).to_degrees().rem_euclid(360.0);
        let dec = gx.z.asin().to_degrees();
        assert!(
            (ra - 266.4050).abs() < 1e-3 && (dec + 28.9362).abs() < 1e-3,
            "RA {ra} Dec {dec}"
        );
        let (ra, dec) = (266.4050_f64.to_radians(), (-28.9362_f64).to_radians());
        let gc = icrs_to_galactic(radec_to_icrs(ra, dec));
        assert!(gc.abs_diff_eq(DVec3::X, 2e-5), "{gc}");
    }

    #[test]
    fn north_galactic_pole_direction() {
        // The NGP at RA 192.85948°, Dec +27.12825° defines the frame (ESA 1997, Vol. 1 §1.5.3).
        let (ra, dec) = (192.859_48_f64.to_radians(), 27.128_25_f64.to_radians());
        let ngp = icrs_to_galactic(radec_to_icrs(ra, dec));
        assert!(ngp.abs_diff_eq(DVec3::Z, 1e-12), "{ngp}");
    }

    #[test]
    fn matrix_is_a_rotation() {
        let m = DMat3::from_cols_array_2d(&ICRS_TO_GALACTIC);
        assert!((m * m.transpose()).abs_diff_eq(DMat3::IDENTITY, 1e-12));
        assert!((m.determinant() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn radec_axes() {
        assert!(radec_to_icrs(0.0, 0.0).abs_diff_eq(DVec3::X, 1e-15));
        assert!(radec_to_icrs(FRAC_PI_2, 0.0).abs_diff_eq(DVec3::Y, 1e-15));
        assert!(radec_to_icrs(1.0, FRAC_PI_2).abs_diff_eq(DVec3::Z, 1e-15));
    }
}
