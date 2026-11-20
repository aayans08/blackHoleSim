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
/// Light-year [m] = c × 365.25 days (IAU Julian year).
pub const LY: f64 = 9.460_730_472_580_8e15;
/// Day [s] (SI day of 86400 s, as used by Julian dates).
pub const DAY: f64 = 86_400.0;

/// Schwarzschild radius r_s = 2GM/c² [m] for a mass in kg (Schwarzschild 1916).
pub fn schwarzschild_radius(mass_kg: f64) -> f64 {
    todo!()
}

/// Gravitational radius r_g = GM/c² [m] for a mass in kg; the unit "M" of black-hole maths.
pub fn gravitational_radius(mass_kg: f64) -> f64 {
    todo!()
}
