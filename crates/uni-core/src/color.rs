//! Colour science: Planck blackbody → CIE XYZ → linear sRGB, B−V → temperature, star illuminance.

use glam::{DVec3, Vec3};

use crate::units::C;

/// Planck constant [J s], exact in the SI since 2019 (CODATA 2018).
const H: f64 = 6.626_070_15e-34;
/// Boltzmann constant [J/K], exact in the SI since 2019 (CODATA 2018).
const K_B: f64 = 1.380_649e-23;
/// Second radiation constant c₂ = hc/k_B [m K] (CODATA 2018).
const C2: f64 = H * C / K_B;

/// Sample range for the Planck × CMF sum: 380–780 nm in 1 nm steps.
const LAMBDA_MIN_NM: usize = 380;
const LAMBDA_SAMPLES: usize = 401;

/// Blackbody chromaticity as CIE 1931 2° XYZ normalised to Y = 1
/// (CMF from Wyman, Sloan & Shirley 2013 multi-lobe fit).
///
/// Planck's law is summed against the CMFs from 380 to 780 nm in 1 nm steps, as CIE 15:2004
/// computes tristimulus values. The sum runs in log space relative to its largest sample, so
/// it stays finite from ~1 K to 10³⁰ K. Non-positive, infinite or NaN temperatures give zero.
pub fn blackbody_xyz(temp_k: f64) -> DVec3 {
    let lambda_nm = |i: usize| (LAMBDA_MIN_NM + i) as f64;
    let ln_b: [f64; LAMBDA_SAMPLES] =
        std::array::from_fn(|i| ln_planck_shape(lambda_nm(i) * 1e-9, temp_k));
    let peak = ln_b.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let xyz: DVec3 = ln_b
        .iter()
        .enumerate()
        .map(|(i, &ln)| (ln - peak).exp() * cie_1931_cmf(lambda_nm(i)))
        .sum();
    // A bad temperature leaves every sample NaN or ±∞, so there is nothing finite to normalise.
    if xyz.is_finite() && xyz.y > 0.0 {
        xyz / xyz.y
    } else {
        DVec3::ZERO
    }
}

/// ln of Planck's spectral radiance B_λ(T) = (2hc²/λ⁵) / (exp(c₂/λT) − 1) (Planck 1901), without
/// the constant 2hc², which cancels when normalising to Y = 1.
fn ln_planck_shape(lambda_m: f64, temp_k: f64) -> f64 {
    let x = C2 / (lambda_m * temp_k);
    // ln(eˣ − 1) as x + ln(1 − e⁻ˣ) for large x, which cannot overflow.
    let ln_exp_m1 = if x < 1.0 {
        x.exp_m1().ln()
    } else {
        x + (-(-x).exp()).ln_1p()
    };
    -5.0 * lambda_m.ln() - ln_exp_m1
}

/// CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at `lambda_nm`, from the multi-lobe
/// piecewise-Gaussian fit of Wyman, Sloan & Shirley 2013, "Simple Analytic Approximations to
/// the CIE XYZ Color Matching Functions", JCGT 2(2).
fn cie_1931_cmf(lambda_nm: f64) -> DVec3 {
    // One lobe: a·exp(−t²/2), with t = (λ − μ)·w₁ below μ and (λ − μ)·w₂ above.
    let lobe = |a: f64, mu: f64, w1: f64, w2: f64| {
        let t = (lambda_nm - mu) * if lambda_nm < mu { w1 } else { w2 };
        a * (-0.5 * t * t).exp()
    };
    DVec3::new(
        lobe(0.362, 442.0, 0.0624, 0.0374)
            + lobe(1.056, 599.8, 0.0264, 0.0323)
            + lobe(-0.065, 501.1, 0.0490, 0.0382),
        lobe(0.821, 568.8, 0.0213, 0.0247) + lobe(0.286, 530.9, 0.0613, 0.0322),
        lobe(1.217, 437.0, 0.0845, 0.0278) + lobe(0.681, 459.0, 0.0385, 0.0725),
    )
}

/// CIE XYZ → linear sRGB with the IEC 61966-2-1 D65 matrix.
pub fn xyz_to_linear_srgb(xyz: DVec3) -> DVec3 {
    DVec3::new(
        3.2406 * xyz.x - 1.5372 * xyz.y - 0.4986 * xyz.z,
        -0.9689 * xyz.x + 1.8758 * xyz.y + 0.0415 * xyz.z,
        0.0557 * xyz.x - 0.2040 * xyz.y + 1.0570 * xyz.z,
    )
}

/// Blackbody colour in linear sRGB with Y = 1; negative components clamped to 0.
pub fn blackbody_linear_srgb(temp_k: f64) -> Vec3 {
    xyz_to_linear_srgb(blackbody_xyz(temp_k))
        .max(DVec3::ZERO)
        .as_vec3()
}

/// B−V range passed to the temperature formula, and the temperature for a missing colour (the
/// lead's ruling for the BSC5 bake, M0 Task 7; moved here in M1, review T3).
const BV_MIN: f64 = -0.4;
const BV_MAX: f64 = 2.0;
/// Teff [K] for a non-finite B−V: close to the Sun's nominal 5772 K (IAU 2015 B3), near white.
const UNKNOWN_BV_TEFF_K: f64 = 5800.0;

/// Effective temperature [K] from B−V colour index (Ballesteros 2012, EPL 97 34008).
///
/// T = 4600 K · [1/(0.92 (B−V) + 1.7) + 1/(0.92 (B−V) + 0.62)]. The formula diverges at
/// B−V = −0.674, so B−V is first clamped to [−0.4, 2.0] (2.0 gives about 3170 K; only very red
/// stars such as carbon stars and Miras reach it). A non-finite B−V gives 5800 K.
pub fn bv_to_teff(bv: f64) -> f64 {
    if !bv.is_finite() {
        return UNKNOWN_BV_TEFF_K;
    }
    let bv = bv.clamp(BV_MIN, BV_MAX);
    4600.0 * (1.0 / (0.92 * bv + 1.7) + 1.0 / (0.92 * bv + 0.62))
}

/// Illuminance [lux] at the observer from a star of apparent V magnitude: 10^(−0.4 (m + 13.99)).
/// Zero point: an m_V = 0 star gives 2.54 × 10⁻⁶ lx outside the atmosphere (Allen 1973,
/// *Astrophysical Quantities*, 3rd ed., Athlone Press, p. 197); −2.5 log₁₀(2.54e-6) = 13.99.
pub fn star_illuminance_lux(v_mag: f64) -> f64 {
    10_f64.powf(-0.4 * (v_mag + 13.99))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CIE 1931 (x, y) on the Planckian locus, from Krystek's rational approximation of the
    /// CIE 1960 (u, v) locus (Krystek 1985, "An algorithm to calculate correlated colour
    /// temperature", Color Res. Appl. 10(1), 38–40; valid 1000–15000 K to about 1e-4 in u and v),
    /// converted with x = 3u / (2u − 8v + 4), y = 2v / (2u − 8v + 4) (CIE 1960 UCS, CIE 15:2004).
    fn planckian_locus_xy(t: f64) -> (f64, f64) {
        let u = (0.860_117_757 + 1.541_182_54e-4 * t + 1.286_412_12e-7 * t * t)
            / (1.0 + 8.424_202_35e-4 * t + 7.081_451_63e-7 * t * t);
        let v = (0.317_398_726 + 4.228_062_45e-5 * t + 4.204_816_91e-8 * t * t)
            / (1.0 - 2.897_418_16e-5 * t + 1.614_560_53e-7 * t * t);
        let d = 2.0 * u - 8.0 * v + 4.0;
        (3.0 * u / d, 2.0 * v / d)
    }

    fn chromaticity(temp_k: f64) -> (f64, f64) {
        let c = blackbody_xyz(temp_k);
        (c.x / c.element_sum(), c.y / c.element_sum())
    }

    #[test]
    fn blackbody_6504_is_white() {
        // D65 (x 0.3127, y 0.3290) lies above the Planckian locus, which at 6504 K is at
        // (0.3135, 0.3235) (Krystek 1985). A blackbody there is therefore faintly magenta in sRGB:
        // the exact locus point gives R/G 1.061 and B/G 1.054, and this fit gives 1.056 and 1.050.
        // Both exceed the brief's 0.05 bound, so no correct blackbody passes it. 0.07 still pins
        // "near white", and the tint must point the physical way (R and B above G). Colour
        // accuracy is pinned by `planckian_locus_chromaticity` and `illuminant_a_chromaticity`.
        let c = blackbody_linear_srgb(6504.0);
        assert!(
            (c.x / c.y - 1.0).abs() < 0.07 && (c.z / c.y - 1.0).abs() < 0.07,
            "{c}"
        );
        assert!(c.x > c.y && c.z > c.y, "{c}");
    }

    #[test]
    fn planckian_locus_chromaticity() {
        // A blackbody lies on the Planckian locus by definition, so the locus, not D65, is the
        // reference for its chromaticity. 6500 K sits at (0.3135, 0.3235) (Krystek 1985).
        let (x, y) = chromaticity(6500.0);
        assert!(
            (x - 0.3135).abs() < 0.002 && (y - 0.3235).abs() < 0.002,
            "x {x} y {y}"
        );
        // From 3000 K up the multi-lobe fit stays within 0.002 of the locus (worst 0.0013, at
        // 3000 K). It drifts further below about 2500 K (0.011 at 1000 K); 2856 K is pinned by
        // `illuminant_a_chromaticity`.
        for t in [3000.0, 4000.0, 5000.0, 6500.0, 8000.0, 10_000.0, 15_000.0] {
            let ((x, y), (lx, ly)) = (chromaticity(t), planckian_locus_xy(t));
            assert!(
                (x - lx).abs() < 0.002 && (y - ly).abs() < 0.002,
                "{t} K: ({x}, {y}) vs locus ({lx}, {ly})"
            );
        }
    }

    #[test]
    fn blackbody_ordering() {
        let a = blackbody_linear_srgb(3000.0);
        assert!(a.x > a.y && a.y > a.z);
        let b = blackbody_linear_srgb(20000.0);
        assert!(b.z > b.x);
    }

    #[test]
    fn illuminant_a_chromaticity() {
        // CIE standard illuminant A is a ≈2856 K Planckian radiator at x 0.44757, y 0.40745
        // (CIE 15:2004). The multi-lobe fit over 380–780 nm lands within 0.002.
        let (x, y) = chromaticity(2856.0);
        assert!(
            (x - 0.44757).abs() < 0.002 && (y - 0.40745).abs() < 0.002,
            "x {x} y {y}"
        );
        assert_eq!(blackbody_xyz(2856.0).y, 1.0);
    }

    #[test]
    fn d65_white_maps_to_unit_rgb() {
        // D65 white point at Y = 1 (IEC 61966-2-1).
        let rgb = xyz_to_linear_srgb(DVec3::new(0.950_47, 1.0, 1.088_83));
        assert!(rgb.abs_diff_eq(DVec3::ONE, 1e-3), "{rgb}");
    }

    #[test]
    fn blackbody_extremes_stay_finite() {
        for t in [1.0, 30.0, 1e9, 1e30] {
            let c = blackbody_xyz(t);
            assert!(c.is_finite() && c.y == 1.0, "{t} K -> {c}");
        }
        assert_eq!(blackbody_xyz(0.0), DVec3::ZERO);
        assert_eq!(blackbody_xyz(-5.0), DVec3::ZERO);
        assert_eq!(blackbody_linear_srgb(f64::NAN), Vec3::ZERO);
    }

    #[test]
    fn sun_teff_from_bv() {
        assert!((bv_to_teff(0.65) - 5778.4).abs() < 1.0);
    }

    #[test]
    fn bv_to_teff_clamps() {
        assert_eq!(bv_to_teff(5.0), bv_to_teff(2.0));
        assert_eq!(bv_to_teff(-3.0), bv_to_teff(-0.4));
        // Past the formula's pole at −0.674 it would otherwise go negative or infinite.
        assert!(bv_to_teff(-0.674).is_finite() && bv_to_teff(-0.674) > 0.0);
        assert_eq!(bv_to_teff(f64::NAN), 5800.0);
        assert_eq!(bv_to_teff(f64::INFINITY), 5800.0);
        assert_eq!(bv_to_teff(f64::NEG_INFINITY), 5800.0);
        // Inside the range the formula is applied unchanged.
        assert_eq!(
            bv_to_teff(0.65),
            4600.0 * (1.0 / (0.92 * 0.65 + 1.7) + 1.0 / (0.92 * 0.65 + 0.62))
        );
    }

    #[test]
    fn star_lux_golden() {
        assert!((star_illuminance_lux(0.0) / 2.5351e-6 - 1.0).abs() < 1e-3);
        assert!((star_illuminance_lux(-26.74) / 1.2589e5 - 1.0).abs() < 1e-3);
    }
}
