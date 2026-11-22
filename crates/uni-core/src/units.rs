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
/// Nominal solar radius R☉ [m] (IAU 2015 Resolution B3).
pub const R_SUN_M: f64 = 6.957e8;
/// Nominal solar luminosity L☉ [W] (IAU 2015 Resolution B3).
pub const L_SUN_W: f64 = 3.828e26;
/// Nominal solar effective temperature [K] (IAU 2015 Resolution B3).
pub const TEFF_SUN_K: f64 = 5772.0;
/// Absolute visual magnitude of the Sun, M_V = m_V − 5 log₁₀(1 au / 10 pc) with apparent
/// m_V = −26.74 (Cox 2000, *Allen's Astrophysical Quantities*, 4th ed.), the same Sun as
/// `color::star_illuminance_lux`, which gives 1.26e5 lx for it.
pub const M_V_SUN: f64 = 4.83;
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
    fn solar_nominal_values_are_consistent() {
        // IAU 2015 B3 chose the nominal values so that L = 4π R² σ T⁴ (Stefan–Boltzmann law;
        // σ = 5.670 374 419e-8 W m⁻² K⁻⁴, CODATA 2018) holds to their rounding.
        let sigma = 5.670_374_419e-8;
        let l = 4.0 * std::f64::consts::PI * R_SUN_M * R_SUN_M * sigma * TEFF_SUN_K.powi(4);
        assert!((l / L_SUN_W - 1.0).abs() < 1e-5, "{l:e}");
        // M_V = m_V − 5 log₁₀(d / 10 pc) with m_V = −26.74 seen from d = 1 au.
        let m_v = -26.74 - 5.0 * (AU / (10.0 * PC)).log10();
        assert!((M_V_SUN - m_v).abs() < 0.005, "{m_v}");
    }

    #[test]
    fn parsec_is_648000_over_pi_au() {
        assert!((PC / (AU * 648_000.0 / std::f64::consts::PI) - 1.0).abs() < 1e-15);
    }
}
