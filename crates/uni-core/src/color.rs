//! Colour science: Planck blackbody → CIE XYZ → linear sRGB, B−V → temperature, star illuminance.

use glam::{DVec3, Vec3};

/// Blackbody chromaticity as CIE 1931 2° XYZ normalised to Y = 1
/// (CMF from Wyman, Sloan & Shirley 2013 multi-lobe fit).
pub fn blackbody_xyz(temp_k: f64) -> DVec3 {
    todo!()
}

/// CIE XYZ → linear sRGB with the IEC 61966-2-1 D65 matrix.
pub fn xyz_to_linear_srgb(xyz: DVec3) -> DVec3 {
    todo!()
}

/// Blackbody colour in linear sRGB with Y = 1; negative components clamped to 0.
pub fn blackbody_linear_srgb(temp_k: f64) -> Vec3 {
    todo!()
}

/// Effective temperature [K] from B−V colour index (Ballesteros 2012, EPL 97 34008).
pub fn bv_to_teff(bv: f64) -> f64 {
    todo!()
}

/// Illuminance [lux] at the observer from a star of apparent V magnitude: 10^(−0.4 (m + 13.99)).
/// Zero point: an m_V = 0 star gives 2.54 × 10⁻⁶ lx outside the atmosphere (Allen 1973,
/// *Astrophysical Quantities*, 3rd ed., Athlone Press, p. 197); −2.5 log₁₀(2.54e-6) = 13.99.
pub fn star_illuminance_lux(v_mag: f64) -> f64 {
    todo!()
}
