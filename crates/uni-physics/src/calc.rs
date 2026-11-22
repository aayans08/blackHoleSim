//! Small orbital calculators for the HUD and the world (M1 Task A3).
//!
//! Newtonian point-mass formulas (Newton 1687, *Principia*); SI units. Radii and distances must
//! be positive (r = 0 gives ∞ or NaN).

use uni_core::C;

/// Surface gravity GM/r² [m/s²].
///
/// Gravity alone: no centrifugal term from the body's rotation.
pub fn local_g(gm: f64, r_m: f64) -> f64 {
    gm / (r_m * r_m)
}

/// Escape speed √(2GM/r) [m/s].
///
/// Zero total energy (vis-viva with a → ∞).
pub fn escape_speed(gm: f64, r_m: f64) -> f64 {
    (2.0 * gm / r_m).sqrt()
}

/// Circular orbital speed √(GM/r) [m/s].
///
/// Vis-viva v² = GM(2/r − 1/a) with a = r.
pub fn circular_speed(gm: f64, r_m: f64) -> f64 {
    (gm / r_m).sqrt()
}

/// Hill radius a(1 − e)∛(m/(3M)) [m] of a body of mass `m` orbiting `m_central`.
///
/// The distance from the body to its L1/L2 points for m ≪ M (Murray & Dermott 1999, *Solar
/// System Dynamics*, eq. 3.75), evaluated at pericentre a(1 − e), where the sphere is smallest
/// (Hamilton & Burns 1992, Icarus 96, 43).
pub fn hill_radius(a_m: f64, e: f64, m: f64, m_central: f64) -> f64 {
    a_m * (1.0 - e) * (m / (3.0 * m_central)).cbrt()
}

/// Light travel time over `d_m` [s].
///
/// d/c with c = 299 792 458 m/s, exact by the SI definition of the metre (CGPM 1983).
pub fn light_time_s(d_m: f64) -> f64 {
    d_m / C
}

#[cfg(test)]
mod tests {
    use uni_core::{AU, GM_SUN};

    use super::*;

    /// Earth's GM [m³/s²] (IERS Conventions 2010, Table 1.1; EGM2008).
    const GM_EARTH: f64 = 3.986_004_418e14;

    #[test]
    fn escape_speed_earth() {
        assert!((escape_speed(3.986004418e14, 6.3781e6) - 11_180.0).abs() < 5.0);
        // √2 × the circular speed at the same radius, by definition.
        let r = 7.0e6;
        let ratio = escape_speed(GM_EARTH, r) / circular_speed(GM_EARTH, r);
        assert!((ratio - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn hill_radius_earth() {
        assert!((hill_radius(AU, 0.0, 5.972e24, 1.989e30) / 1.497e9 - 1.0).abs() < 0.01);
        // Measured at pericentre: an eccentric orbit shrinks it by (1 − e).
        let circular = hill_radius(AU, 0.0, 5.972e24, 1.989e30);
        assert!((hill_radius(AU, 0.5, 5.972e24, 1.989e30) / circular - 0.5).abs() < 1e-15);
    }

    #[test]
    fn surface_gravity_and_orbit_speed() {
        // Earth's mean radius 6371.0 km (IUGG): 9.820 m/s² from gravity alone (no rotation).
        assert!((local_g(GM_EARTH, 6.371e6) - 9.820).abs() < 1e-3);
        // Earth's mean orbital speed, √(GM☉/1 au) = 29.78 km/s.
        assert!((circular_speed(GM_SUN, AU) - 29_784.7).abs() < 0.1);
    }

    #[test]
    fn light_time_one_au() {
        // τ_A = 499.004 783 836 s (IAU 2009 System of Astronomical Constants), the light time
        // for 1 au with both the au and c exact.
        assert!((light_time_s(AU) - 499.004_783_836).abs() < 1e-9);
        assert_eq!(light_time_s(0.0), 0.0);
    }
}
