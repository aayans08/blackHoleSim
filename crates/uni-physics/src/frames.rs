//! Ecliptic and equatorial frame rotations (M1 Task A1).
//!
//! The J2000 mean equator and equinox are taken as ICRS: the 23 mas frame bias between them
//! (IERS Conventions 2010, §5.4.4) is below the error of every theory that uses these
//! rotations (Standish, Meeus), so it is ignored.

use std::f64::consts::PI;

use glam::{DMat3, DVec3};
use uni_core::Epoch;

/// One arcsecond [rad].
const ARCSEC: f64 = PI / (180.0 * 3600.0);

/// Mean obliquity of the ecliptic at J2000, ε₀ = 84381.448″ (IAU 1976; Lieske et al. 1977,
/// A&A 58, 1).
pub const OBLIQUITY_J2000_RAD: f64 = 84_381.448 * std::f64::consts::PI / (180.0 * 3600.0);

/// Rotates a vector from the J2000 ecliptic frame to ICRS (about +x by ε₀).
pub fn ecliptic_j2000_to_icrs(v: DVec3) -> DVec3 {
    DMat3::from_rotation_x(OBLIQUITY_J2000_RAD) * v
}

/// Precession from the mean equator and equinox of date to J2000 (IAU 1976 ζ_A, z_A, θ_A).
///
/// Lieske et al. 1977 (A&A 58, 1), as in Meeus, *Astronomical Algorithms* 2nd ed., (21.3)
/// with J2000 as the starting epoch, t in Julian centuries TDB:
/// ζ_A = 2306.2181″t + 0.30188″t² + 0.017998″t³, z_A = 2306.2181″t + 1.09468″t² +
/// 0.018203″t³, θ_A = 2004.3109″t − 0.42665″t² − 0.041833″t³. J2000 → date is
/// R₃(−z_A)·R₂(θ_A)·R₃(−ζ_A) (frame rotations); this returns its transpose.
///
/// Validity: the polynomials are a fit for a few centuries around J2000 and their error grows
/// with t³; far from J2000 the matrix stays a finite rotation but is no longer precession.
/// A2 uses it only inside the Moon theory's window. The upgrade path is the long-term
/// precession of Vondrák, Capitaine & Wallace 2011 (A&A 534, A22), valid over ±200 000 yr
/// (review m21).
pub fn precession_to_j2000(t: Epoch) -> DMat3 {
    let t = t.julian_centuries();
    let zeta = (2306.2181 + (0.30188 + 0.017998 * t) * t) * t * ARCSEC;
    let z = (2306.2181 + (1.09468 + 0.018203 * t) * t) * t * ARCSEC;
    let theta = (2004.3109 - (0.42665 + 0.041833 * t) * t) * t * ARCSEC;
    // (R_z(z)·R_y(−θ)·R_z(ζ))ᵀ in active rotations.
    DMat3::from_rotation_z(-zeta) * DMat3::from_rotation_y(theta) * DMat3::from_rotation_z(-z)
}

/// ICRS position [m] of ecliptic-of-date longitude `lon`, latitude `lat` [rad] at `dist_m`.
///
/// Mean ecliptic and equinox of date (no nutation) → mean equator of date by the mean
/// obliquity ε(t) → J2000 by [`precession_to_j2000`]. Same validity as that function.
pub fn ecliptic_of_date_to_icrs(lon: f64, lat: f64, dist_m: f64, t: Epoch) -> DVec3 {
    let (sin_l, cos_l) = lon.sin_cos();
    let (sin_b, cos_b) = lat.sin_cos();
    let ecliptic = dist_m * DVec3::new(cos_b * cos_l, cos_b * sin_l, sin_b);
    let equator_of_date = DMat3::from_rotation_x(mean_obliquity_rad(t)) * ecliptic;
    precession_to_j2000(t) * equator_of_date
}

/// Mean obliquity of date, ε = 84381.448″ − 46.8150″T − 0.00059″T² + 0.001813″T³
/// (IAU 1976; Lieske et al. 1977; Meeus (22.2)).
fn mean_obliquity_rad(t: Epoch) -> f64 {
    let t = t.julian_centuries();
    (84_381.448 + (-46.8150 + (-0.00059 + 0.001813 * t) * t) * t) * ARCSEC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ra_dec_deg(v: DVec3) -> (f64, f64) {
        let v = v.normalize();
        (
            v.y.atan2(v.x).to_degrees().rem_euclid(360.0),
            v.z.asin().to_degrees(),
        )
    }

    fn unit(ra_deg: f64, dec_deg: f64) -> DVec3 {
        let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
        DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
    }

    /// ICRS → J2000 ecliptic (the inverse rotation, −ε₀ about +x).
    fn icrs_to_ecliptic(v: DVec3) -> DVec3 {
        DMat3::from_rotation_x(-OBLIQUITY_J2000_RAD) * v
    }

    /// The north ecliptic pole is at RA 270°, Dec 90° − ε₀ = 66.560 709°.
    #[test]
    fn ecliptic_pole_in_icrs() {
        let (ra, dec) = ra_dec_deg(ecliptic_j2000_to_icrs(DVec3::Z));
        assert!((ra - 270.0).abs() < 1e-6, "{ra}");
        assert!((dec - 66.560_709).abs() < 1e-6, "{dec}");
        // The equinox direction is shared by both frames.
        assert_eq!(ecliptic_j2000_to_icrs(DVec3::X), DVec3::X);
    }

    /// The mean pole of date at J2100, carried back to J2000, is θ_A(T = 1) = 2004.3109″ −
    /// 0.42665″ − 0.041833″ = 2003.8423″ from the J2000 pole (Lieske et al. 1977; Meeus 21.2).
    #[test]
    fn precession_pole_one_century() {
        let j2100 = Epoch::from_days_since_j2000(36_525.0);
        let pole = precession_to_j2000(j2100) * DVec3::Z;
        let angle = pole.angle_between(DVec3::Z) / ARCSEC;
        assert!((angle - 2003.8423).abs() < 0.01, "{angle}″");
        assert_eq!(precession_to_j2000(Epoch::J2000), DMat3::IDENTITY);
    }

    /// Meeus, *Astronomical Algorithms* 2nd ed., Example 21.b (θ Persei, J2000 → 2028 Nov
    /// 13.19 TD = JDE 2462088.69): α₀ = 41.054063°, δ₀ = 49.227750° (proper motion applied)
    /// precess to α = 41.547214°, δ = 49.348483°.
    #[test]
    fn precession_meeus_21b() {
        let t = Epoch::from_days_since_j2000(2_462_088.69 - 2_451_545.0);
        let of_date = precession_to_j2000(t).transpose() * unit(41.054063, 49.227750);
        let (ra, dec) = ra_dec_deg(of_date);
        assert!((ra - 41.547214).abs() < 1e-6, "{ra}");
        assert!((dec - 49.348483).abs() < 1e-6, "{dec}");
        // The inverse takes it back.
        let back = precession_to_j2000(t) * of_date;
        assert!((back - unit(41.054063, 49.227750)).length() < 1e-15);
    }

    /// At J2000 the ecliptic of date is the J2000 ecliptic. At J2100 the equinox of date lies
    /// at J2000-ecliptic longitude −p_A = −5030.2077″ and the ecliptic of date is tilted
    /// π_A = 46.9699″ from the J2000 ecliptic (Meeus 21.5, 21.6; Lieske et al. 1977).
    #[test]
    fn ecliptic_of_date_equinox_and_pole() {
        let (lon, lat, d) = (1.234, -0.3, 3.0e8);
        let at_j2000 = ecliptic_of_date_to_icrs(lon, lat, d, Epoch::J2000);
        let direct = ecliptic_j2000_to_icrs(
            d * DVec3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()),
        );
        assert!((at_j2000 - direct).length() < 1e-15 * d);

        let j2100 = Epoch::from_days_since_j2000(36_525.0);
        let equinox = icrs_to_ecliptic(ecliptic_of_date_to_icrs(0.0, 0.0, 1.0, j2100));
        let lon_arcsec = equinox.y.atan2(equinox.x) / ARCSEC;
        let p_a = 5029.0966 + 1.11113 - 0.000006;
        assert!((lon_arcsec + p_a).abs() < 0.01, "{lon_arcsec}″");
        let pole = icrs_to_ecliptic(ecliptic_of_date_to_icrs(
            0.0,
            std::f64::consts::FRAC_PI_2,
            1.0,
            j2100,
        ));
        let tilt = pole.angle_between(DVec3::Z) / ARCSEC;
        let pi_a = 47.0029 - 0.03302 + 0.000060;
        assert!((tilt - pi_a).abs() < 0.01, "{tilt}″");
        // Distance is preserved.
        assert!((ecliptic_of_date_to_icrs(0.3, 0.2, d, j2100).length() / d - 1.0).abs() < 1e-15);
    }
}
