//! SI physical constants and unit helpers (CODATA 2018 / IAU 2012 B2 / IAU 2015 B3).

/// Speed of light in vacuum [m/s]. Exact by SI definition (CODATA 2018).
pub const C: f64 = 299_792_458.0;
/// Newtonian constant of gravitation [m³ kg⁻¹ s⁻²] (CODATA 2018).
pub const G: f64 = 6.674_30e-11;
/// Nominal solar mass parameter GM☉ [m³/s²] (IAU 2015 Resolution B3, TDB-compatible).
pub const GM_SUN: f64 = 1.327_124_400_41e20;
/// Solar mass [kg], derived as GM☉ / G (IAU 2015 B3 with CODATA 2018 G).
pub const M_SUN: f64 = GM_SUN / G;
/// Astronomical unit [m]. Exact by definition (IAU 2012 Resolution B2).
pub const AU: f64 = 1.495_978_707e11;
/// Parsec [m] = 648000/π au (IAU 2012 Resolution B2; restated in IAU 2015 Resolution B2).
pub const PC: f64 = 3.085_677_581_491_367_3e16;
/// Light-year [m]: the distance light travels in vacuum in one Julian year of 365.25 days of
/// 86400 s, i.e. c × 31 557 600 s (IAU Style Manual, Wilkins 1989, Trans. IAU XXB; the Julian
/// year is from the IAU 1976 System of Astronomical Constants).
pub const LY: f64 = 9.460_730_472_580_8e15;
/// Day [s]: 86400 SI seconds, the astronomical unit of time used for Julian dates and the
/// Julian year (IAU 1976 System of Astronomical Constants; IAU Style Manual, Wilkins 1989).
pub const DAY: f64 = 86_400.0;

/// Schwarzschild radius r_s = 2GM/c² [m] for a mass in kg (Schwarzschild 1916).
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    2.0 * gravitational_radius(mass_kg)
}

/// Gravitational radius r_g = GM/c² [m] for a mass in kg; the unit "M" of black-hole maths.
/// It is the mass in geometrised units (G = c = 1; Misner, Thorne & Wheeler 1973,
/// *Gravitation*, W. H. Freeman) and half the Schwarzschild radius (Schwarzschild 1916).
pub fn gravitational_radius(mass_kg: f64) -> f64 {
    G * mass_kg / (C * C)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schwarzschild_radius_of_sun() {
        assert!((schwarzschild_radius(M_SUN) - 2953.25).abs() < 0.1);
    }

    #[test]
    fn gravitational_radius_of_sun() {
        // GM☉/c² = 1476.625 m (IAU 2015 B3 GM☉ with the exact c).
        assert!((gravitational_radius(M_SUN) - 1476.625).abs() < 1e-3);
    }

    #[test]
    fn light_year_is_c_times_julian_year() {
        assert_eq!(LY, C * 365.25 * DAY);
    }

    #[test]
    fn parsec_is_648000_over_pi_au() {
        assert!((PC / (AU * 648_000.0 / std::f64::consts::PI) - 1.0).abs() < 1e-15);
    }
}
