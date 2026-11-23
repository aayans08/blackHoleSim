//! Photometry (M1 Task A3): asteroid H, G magnitudes (Bowell et al. 1989), comet total
//! magnitudes, Lambert and lunar-Lambert (McEwen 1991) reflectance, disk illuminance, the
//! planets' observed phase curves (Mallama & Hilton 2018) as a correction to those BRDFs, and
//! star radii from bolometric corrections (Torres 2010).
//!
//! Radiance convention (shared with `body.wgsl`): a surface of normal reflectance ρ lit by an
//! illuminance E☉ at normal incidence has radiance L = ρ · [`reflectance`](μ₀, μ) · E☉/π, where
//! μ₀ and μ are the cosines of the incidence and emission angles.
//!
//! [`rho_from_geometric_albedo`] was implemented by the lead's A0b before A3 ran (A8's bake calls
//! it); A3 keeps it and adds the negative-albedo guard.

use std::f64::consts::{FRAC_PI_2, PI};

use uni_core::L_SUN_W;

use crate::NaifId;

/// A surface reflectance model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Brdf {
    Lambert,
    /// Lunar-Lambert with limb-darkening weight `l` (1 = Lommel–Seeliger, 0 = Lambert).
    LunarLambert {
        l: f64,
    },
}

/// The H, G phase function (Bowell et al. 1989) at phase angle `alpha_rad`.
///
/// Φ = (1 − G)Φ₁ + GΦ₂ with Φᵢ = exp(−Aᵢ tan^Bᵢ(α/2)), A = (3.33, 1.87), B = (0.63, 1.22):
/// the two-parameter magnitude system adopted by IAU Commission 20 in 1985 (Bowell et al. 1989,
/// "Application of photometric models to asteroids", *Asteroids II*, 524), in its exponential
/// approximation (without the small-phase blending term of the full form). The phase angle is
/// unsigned and capped at π, where Φ = 0. Φ(0) = 1 exactly for every G.
pub fn hg_phase(alpha_rad: f64, g: f64) -> f64 {
    let t = (0.5 * alpha_rad.abs().clamp(0.0, PI)).tan();
    let phi1 = (-3.33 * t.powf(0.63)).exp();
    let phi2 = (-1.87 * t.powf(1.22)).exp();
    // (1 − G)Φ₁ + GΦ₂, written so that Φ₁ = Φ₂ = 1 gives exactly 1.
    phi1 + g * (phi2 - phi1)
}

/// Asteroid apparent V magnitude from H, G at heliocentric `r_au`, observer `delta_au` and
/// phase `alpha_rad`.
///
/// V = H + 5 log₁₀(rΔ) − 2.5 log₁₀Φ(α) (Bowell et al. 1989, *Asteroids II*, 524). At α = π the
/// body is unlit and V = +∞.
pub fn asteroid_v(h: f64, g: f64, r_au: f64, delta_au: f64, alpha_rad: f64) -> f64 {
    h + 5.0 * (r_au * delta_au).log10() - 2.5 * hg_phase(alpha_rad, g).log10()
}

/// Comet total magnitude m = M1 + 5 log₁₀Δ + K1 log₁₀r (JPL SBDB convention).
///
/// The total (nucleus + coma) magnitude law whose parameters M1, K1 JPL's Small-Body Database
/// lists for comets.
pub fn comet_total_v(m1: f64, k1: f64, r_au: f64, delta_au: f64) -> f64 {
    m1 + 5.0 * delta_au.log10() + k1 * r_au.log10()
}

/// ρ-normalised reflectance factor at incidence cosine `mu0` and emission cosine `mu`; the CPU
/// twin of `body.wgsl`.
///
/// Lunar-Lambert (McEwen 1991, Icarus 92, 298): 2Lμ₀/(μ₀ + μ) + (1 − L)μ₀. L = 1 is
/// Lommel–Seeliger, L = 0 (and [`Brdf::Lambert`]) is Lambert, μ₀. A point that is unlit
/// (μ₀ ≤ 0) or faces away from the observer (μ ≤ 0) reflects nothing toward it: 0. NaN in gives
/// NaN out.
pub fn reflectance(brdf: Brdf, mu0: f64, mu: f64) -> f64 {
    if mu0 <= 0.0 || mu <= 0.0 {
        return 0.0;
    }
    match brdf {
        Brdf::Lambert => mu0,
        Brdf::LunarLambert { l } => 2.0 * l * mu0 / (mu0 + mu) + (1.0 - l) * mu0,
    }
}

/// Normal reflectance ρ that gives V geometric albedo `p_v`: Lambert ρ = 1.5 p; lunar-Lambert
/// p = ρ[L + 2(1 − L)/3]; capped at 1 (review I4).
///
/// Implemented by the lead's A0b (fix round 1, review I3) ahead of A3, because A8's bake
/// (`albedo_scale`) calls it before A3 runs. A3 owns this module and keeps or extends it.
///
/// Derivation: at phase 0 (μ₀ = μ) the lunar-Lambert reflectance factor
/// ρ[2Lμ₀/(μ₀ + μ) + (1 − L)μ₀] (McEwen 1991, Icarus 92, 298) is ρ[L + (1 − L)μ]. The geometric
/// albedo is its mean over the projected disk, where ⟨μ⟩ = 2/3, so p = ρ[L + 2(1 − L)/3]. Lambert
/// is L = 0, giving the classical Lambert-sphere p = 2ρ/3 (Russell 1916, ApJ 43, 173). A NaN or
/// negative `p_v` names no real surface and gives NaN (no silent default); 0 gives 0.
pub fn rho_from_geometric_albedo(brdf: Brdf, p_v: f64) -> f64 {
    if p_v < 0.0 {
        return f64::NAN;
    }
    let l = match brdf {
        Brdf::Lambert => 0.0,
        Brdf::LunarLambert { l } => l,
    };
    let rho = p_v / (l + 2.0 * (1.0 - l) / 3.0);
    // Cap at 1 (review I4: Venus, p_V = 0.689, would need ρ = 1.03; the cap costs 0.036 mag).
    if rho > 1.0 { 1.0 } else { rho }
}

/// Lambert-sphere phase function Φ(α) = [sin α + (π − α) cos α]/π.
///
/// The disk-integrated brightness of a Lambert sphere relative to α = 0 (Russell 1916, ApJ 43,
/// 173). The phase angle is unsigned and capped at π, where Φ = 0.
pub fn lambert_phase(alpha_rad: f64) -> f64 {
    let a = alpha_rad.abs().clamp(0.0, PI);
    (a.sin() + (PI - a) * a.cos()) / PI
}

/// Samples per axis of [`disk_illuminance_lux`]'s grid: 256² in all (brief: ≥ 256²).
const DISK_SAMPLES: usize = 256;

/// Illuminance [lx] at `dist_m` from a sphere of `radius_m` with normal reflectance `albedo`,
/// lit by `sun_lux` at phase `phase_rad` (numerical integral over the lit disk).
///
/// E = (E☉/π)(R/d)² ρ ∬ R(μ₀, μ) μ dΩ over the visible lit hemisphere: the sum over the disk of
/// the radiance `body.wgsl` writes (see the module's radiance convention) times each surface
/// element's solid angle, so it is the CPU twin of what the bodies pass puts on screen.
///
/// Grid: photometric latitude θ (pole perpendicular to the scattering plane) and longitude φ
/// (from the sub-observer point toward the Sun). Then μ = cos θ cos φ, μ₀ = cos θ cos(φ − α),
/// and the visible lit region is exactly φ ∈ [α − π/2, π/2] at every θ, so the limb and the
/// terminator are grid edges rather than staircases. 256 × 256 midpoint samples, one reflectance
/// evaluation each, keep the Lambert sphere within 1e-5 of its closed form even for the
/// α = 150° crescent (`disk_flux_matches_point_flux` prints it), where a square 256² pixel grid
/// over the disk is 0.19 % off.
///
/// Far field: the disk is integrated in parallel projection, so the result is ∝ (R/d)² and is
/// meant for d ≫ R (the point ↔ disk handover, a few pixels across). Linear in `albedo` and
/// `sun_lux`. The phase angle is unsigned and capped at π (dark). Cost: 65 536 reflectance
/// evaluations per call.
pub fn disk_illuminance_lux(
    brdf: Brdf,
    albedo: f64,
    radius_m: f64,
    sun_lux: f64,
    phase_rad: f64,
    dist_m: f64,
) -> f64 {
    let alpha = phase_rad.abs().clamp(0.0, PI);
    let n = DISK_SAMPLES as f64;
    let d_theta = PI / n;
    let phi_start = alpha - FRAC_PI_2;
    let d_phi = (FRAC_PI_2 - phi_start) / n;
    let mut cos_theta = [0.0; DISK_SAMPLES];
    for (i, c) in cos_theta.iter_mut().enumerate() {
        *c = (-FRAC_PI_2 + (i as f64 + 0.5) * d_theta).cos();
    }
    let mut sum = 0.0;
    for j in 0..DISK_SAMPLES {
        let phi = phi_start + (j as f64 + 0.5) * d_phi;
        let (cos_phi, cos_phi_sun) = (phi.cos(), (phi - alpha).cos());
        let mut column = 0.0;
        for &ct in &cos_theta {
            let mu = ct * cos_phi;
            // Surface element dΩ = cos θ dθ dφ; its projected area toward the observer adds μ.
            column += reflectance(brdf, ct * cos_phi_sun, mu) * mu * ct;
        }
        sum += column;
    }
    let integral = sum * d_theta * d_phi;
    let ratio = radius_m / dist_m;
    albedo * sun_lux / PI * ratio * ratio * integral
}

/// Lommel–Seeliger sphere phase function Φ(α) = 1 − sin(α/2) tan(α/2) ln cot(α/4), relative to
/// α = 0 (Fairbairn 2005, JRASC 99, 92). `a` is unsigned and at most π, where Φ = 0.
fn lommel_seeliger_phase(a: f64) -> f64 {
    if a == 0.0 {
        return 1.0; // the 0 · ∞ limit of the closed form
    }
    if a >= PI {
        return 0.0;
    }
    let h = 0.5 * a;
    (1.0 - h.sin() * h.tan() * (1.0 / (0.25 * a).tan()).ln()).max(0.0)
}

/// Disk-integrated phase function of a sphere with this BRDF, relative to α = 0 (closed form).
///
/// The lunar-Lambert reflectance (McEwen 1991, Icarus 92, 298) is linear in L: L times
/// Lommel–Seeliger plus (1 − L) times Lambert. Their disk integrals at α = 0 are ρ and (2/3)ρ
/// (see [`rho_from_geometric_albedo`]), so Φ = [L·Φ_LS + (1 − L)(2/3)·Φ_L] / [L + (1 − L)(2/3)],
/// with Φ_L = [`lambert_phase`] (Russell 1916) and Φ_LS = 1 − sin(α/2) tan(α/2) ln cot(α/4)
/// (Fairbairn 2005, JRASC 99, 92). Then
/// `disk_illuminance_lux(brdf, ρ, R, E☉, α, d)` = ρ·[L + 2(1 − L)/3]·E☉·(R/d)²·Φ(α) to the
/// integral's accuracy (`disk_phase_function_matches_integral`), which lets a point path
/// evaluate a body's flux in O(1) instead of 65 536 reflectance samples. The phase angle is
/// unsigned and capped at π (Φ = 0); NaN gives NaN.
pub fn disk_phase_function(brdf: Brdf, phase_rad: f64) -> f64 {
    let a = phase_rad.abs().clamp(0.0, PI);
    if a.is_nan() {
        return f64::NAN;
    }
    let l = match brdf {
        Brdf::Lambert => 0.0,
        Brdf::LunarLambert { l } => l,
    };
    // Weights: each part's disk integral at α = 0 per unit ρ.
    let (w_ls, w_lambert) = (l, (1.0 - l) * 2.0 / 3.0);
    (w_ls * lommel_seeliger_phase(a) + w_lambert * lambert_phase(a)) / (w_ls + w_lambert)
}

/// Mallama & Hilton 2018 (Astron. Comput. 25, 10; arXiv:1808.01973), the planetary magnitude
/// laws of *The Astronomical Almanac* and of JPL Horizons' APmag: V(1, α) [mag] of planet
/// `naif`'s globe at unit distances from the Sun and observer, α in degrees, or `None` for a
/// body without a law. Coefficients transcribed from the paper's Eq. 2–17 (all angular
/// coefficients in degrees); A3 checked them against Horizons' APmag at the 2026-11-20 sky
/// spots (`mallama_hilton_matches_horizons_apmag`).
///
/// - Mercury, Eq. 2: sixth-order polynomial.
/// - Venus, Eq. 3 for α ≤ 163.7°, Eq. 4 above (the forward-scattering brightening; a
///   0.007 mag step at the join, as published).
/// - Earth, Eq. 5 (Tinetti et al. 2006's realistic-cloud model with V(1,0) = −3.99).
/// - Mars, Eq. 6 for α ≤ 50°, Eq. 7 above (the mean of Mercury's and Earth's dimming), without
///   the sub-Earth-longitude and L_s terms (not modelled here; 0.018 mag at the 2026-11-20
///   spot).
/// - Jupiter, Eq. 8 for α ≤ 12°, Eq. 9 above (Mayorga et al. 2016, Cassini green filter).
/// - Saturn's globe without its rings, Eq. 11 for α ≤ 6°, Eq. 12 above. The rings are a
///   separate body in the renderer, so the ringed law (Eq. 10) is not used.
/// - Uranus, Eq. 15 without its −8.4 × 10⁻⁴ φ′ sub-latitude term (≤ 0.07 mag, φ′ ≤ 82°).
/// - Neptune, Eq. 17 (the post-2000 V(1,0) = −7.00 of Eq. 16).
fn mallama_hilton_v1(naif: NaifId, a: f64) -> Option<f64> {
    // a₀ + a₁α + a₂α² + …
    let poly = |x: f64, c: &[f64]| c.iter().rev().fold(0.0, |acc, &k| acc * x + k);
    let v = match naif {
        199 => poly(
            a,
            &[
                -0.613,
                6.3280e-2,
                -1.6336e-3,
                3.3644e-5,
                -3.4265e-7,
                1.6893e-9,
                -3.0334e-12,
            ],
        ),
        299 if a <= 163.7 => poly(a, &[-4.384, -1.044e-3, 3.687e-4, -2.814e-6, 8.938e-9]),
        299 => poly(a, &[236.05828, -2.81914, 8.39034e-3]),
        399 => poly(a, &[-3.99, -1.060e-3, 2.054e-4]),
        499 if a <= 50.0 => poly(a, &[-1.601, 0.02267, -0.0001302]),
        499 => poly(a, &[-0.367, -0.02573, 0.0003445]),
        599 if a <= 12.0 => poly(a, &[-9.395, -3.7e-4, 6.16e-4]),
        // Eq. 9 takes the phase in units of 180°.
        599 => {
            -9.428 - 2.5 * poly(a / 180.0, &[1.0, -1.507, -0.363, -0.062, 2.809, -1.876]).log10()
        }
        699 if a <= 6.0 => poly(a, &[-8.95, -3.7e-4, 6.16e-4]),
        699 => poly(a, &[-8.94, 2.446e-4, 2.672e-4, -1.505e-6, 4.767e-9]),
        // Uranus and Neptune: the phase terms are applied at every α. Mallama & Hilton's reference
        // code (as ported in Skyfield's `magnitudelib`) applies them only beyond geocentric
        // limits of 3.1° (Uranus) and 1.9° (Neptune); inside those the smooth law differs by
        // at most 0.02 mag (A3 review r1 item 3, from memory of the port, not checked against
        // the paper), which is the criterion for keeping the smooth law for rendering.
        799 => poly(a, &[-7.110, 6.587e-3, 1.045e-4]),
        899 => poly(a, &[-7.00, 7.944e-3, 9.617e-5]),
        _ => return None,
    };
    Some(v)
}

/// The largest phase angle [deg] that each [`mallama_hilton_v1`] law covers; beyond it
/// [`planet_phase_correction`] holds its value there. From the paper: Mercury's observations
/// span 2.1°–169.5° (§4.1), Venus's Eq. 4 ends at 179°, Jupiter's Eq. 9 is "not trustworthy
/// beyond ~130°", Saturn's Eq. 12 is valid to 150°, Uranus's Eq. 15 to 154°, Neptune's Eq. 17 to
/// 133°. The paper gives no limit for Earth's Eq. 5 or Mars's Eq. 7 (built from Mercury's and
/// Earth's curves); A3 holds both beyond 170°, near Mercury's observed limit.
fn mallama_hilton_max_deg(naif: NaifId) -> Option<f64> {
    match naif {
        199 => Some(169.5),
        299 => Some(179.0),
        399 | 499 => Some(170.0),
        599 => Some(130.0),
        699 => Some(150.0),
        799 => Some(154.0),
        899 => Some(133.0),
        _ => None,
    }
}

/// Observed-phase-curve correction for planet `naif` drawn with `brdf`: the factor that makes
/// the BRDF's disk-integrated flux at phase `phase_rad` follow Mallama & Hilton 2018.
///
/// C(α) = 10^(−0.4[V(1, α) − V(1, 0)]) / Φ_brdf(α), with V(1, ·) the paper's law for the planet
/// (see `mallama_hilton_v1`) and Φ_brdf = [`disk_phase_function`]. C(0) = 1, so the geometric
/// albedo calibration ([`rho_from_geometric_albedo`]) is untouched; only the phase dependence
/// changes. Lambert and lunar-Lambert spheres cannot reproduce the observed curves (A3 review
/// r0 I1): at the 2026-11-20 spots they put Mercury 1.78 mag too bright at α = 82°, Mars
/// 0.47 mag too bright at 38° and Venus 0.54 mag too faint at 130°. Multiplying by C brings all
/// seven planets within 0.1 mag of Horizons (`planet_magnitudes_vs_horizons`).
///
/// Use: multiply the point flux (`disk_illuminance_lux`, or the closed form in
/// [`disk_phase_function`]'s doc) by C, and the disk's radiance by the same C at every pixel,
/// so the two paths still agree at the handover. The radiance distribution across the disk
/// keeps the BRDF's shape; only its total follows the observed curve. Beyond the law's range
/// (`mallama_hilton_max_deg`) C is held at its value at the limit, so the BRDF's own shape
/// carries the deeper crescent. Bodies without a law (the Moon, Pluto, moons, small bodies)
/// get 1. The phase angle is unsigned; NaN gives NaN.
pub fn planet_phase_correction(naif: NaifId, brdf: Brdf, phase_rad: f64) -> f64 {
    if phase_rad.is_nan() {
        return f64::NAN;
    }
    let (Some(max_deg), Some(v0)) = (mallama_hilton_max_deg(naif), mallama_hilton_v1(naif, 0.0))
    else {
        return 1.0;
    };
    let a_deg = phase_rad.abs().to_degrees().min(max_deg);
    let Some(v) = mallama_hilton_v1(naif, a_deg) else {
        return 1.0;
    };
    10_f64.powf(-0.4 * (v - v0)) / disk_phase_function(brdf, a_deg.to_radians())
}

/// Torres 2010 (AJ 140, 1158), Table 1: Flower's (1996, ApJ 469, 355) BC_V polynomial in
/// x = log₁₀ T_eff, as corrected by Torres; coefficients a, b, c, … of a + bx + cx² + …
/// (checked against the paper, arXiv:1008.3913). The fit puts the Sun at BC_V = −0.080.
const BC_COOL: [f64; 4] = [
    -0.190_537_291_496_456e5,
    0.155_144_866_764_412e5,
    -0.421_278_819_301_717e4,
    0.381_476_328_422_343e3,
];
/// Table 1, 3.70 < log T_eff < 3.90.
const BC_MID: [f64; 5] = [
    -0.370_510_203_809_015e5,
    0.385_672_629_965_804e5,
    -0.150_651_486_316_025e5,
    0.261_724_637_119_416e4,
    -0.170_623_810_323_864e3,
];
/// Table 1, log T_eff > 3.90.
const BC_HOT: [f64; 6] = [
    -0.118_115_450_538_963e6,
    0.137_145_973_583_929e6,
    -0.636_233_812_100_225e5,
    0.147_412_923_562_646e5,
    -0.170_587_278_406_872e4,
    0.788_731_721_804_990e2,
];

/// T_eff range [K] in which the polynomial is used; outside it the edge value is.
///
/// The cool end: Torres notes the corrections "break down completely for the M dwarfs", and the
/// cubic runs away below about 3000 K (BC −24 at 2000 K). The hot end: O3 V stars, the hottest
/// main-sequence type, have T_eff ≈ 45–50 kK (Martins, Schaerer & Hillier 2005, A&A 436, 1049),
/// and the polynomial turns back up above 60.6 kK (its minimum, BC −4.76).
const BC_TEFF_MIN_K: f64 = 3000.0;
const BC_TEFF_MAX_K: f64 = 50_000.0;

/// Bolometric correction BC_V at effective temperature `teff_k` (Torres 2010, AJ 140, 1158).
///
/// Flower's polynomial as corrected by Torres (three segments split at log T_eff = 3.70 and
/// 3.90, used as printed, with their 0.022 and 0.003 mag steps), with T_eff held to
/// 3000–50 000 K. A
/// non-positive or NaN T_eff gives NaN. Approximate below about 3900 K (M stars).
pub fn bolometric_correction_v(teff_k: f64) -> f64 {
    if teff_k.is_nan() || teff_k <= 0.0 {
        return f64::NAN;
    }
    let x = teff_k.clamp(BC_TEFF_MIN_K, BC_TEFF_MAX_K).log10();
    let coefficients: &[f64] = if x < 3.70 {
        &BC_COOL
    } else if x < 3.90 {
        &BC_MID
    } else {
        &BC_HOT
    };
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// Bolometric absolute magnitude of the nominal Sun (IAU 2015 Resolution B2: M_bol = 4.74 for
/// L☉ = 3.828 × 10²⁶ W).
const M_BOL_SUN: f64 = 4.74;

/// Stefan–Boltzmann constant σ [W m⁻² K⁻⁴] (CODATA 2018).
const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;

/// Photosphere radius [m] from absolute V magnitude and effective temperature.
///
/// L = L☉ · 10^(−0.4(M_V + BC_V − M_bol☉)) with BC_V from [`bolometric_correction_v`] and
/// M_bol☉ = 4.74 (IAU 2015 B2), then R = √(L/(4πσT⁴)) (Stefan–Boltzmann). The Sun
/// (M_V 4.83, 5772 K) comes out 0.4 % under R☉ because Flower's scale has BC_V☉ = −0.08, so
/// M_bol = 4.75. T_eff itself is not clamped; only the BC is (see `bolometric_correction_v`).
pub fn star_radius_m(abs_mag_v: f64, teff_k: f64) -> f64 {
    let m_bol = abs_mag_v + bolometric_correction_v(teff_k);
    let luminosity_w = L_SUN_W * 10_f64.powf(-0.4 * (m_bol - M_BOL_SUN));
    let t2 = teff_k * teff_k;
    (luminosity_w / (4.0 * PI * STEFAN_BOLTZMANN * t2 * t2)).sqrt()
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use uni_core::{AU, M_V_SUN, PC, R_SUN_M, TEFF_SUN_K, star_illuminance_lux};

    use super::*;

    /// Solar illuminance [lx] at `r_au`, on `star_illuminance_lux`'s zero point: the Sun's
    /// apparent V at 1 au is M_V☉ + 5 log₁₀(1 au / 10 pc) = −26.74 (uni-core's M_V_SUN source).
    fn sun_lux_at(r_au: f64) -> f64 {
        star_illuminance_lux(M_V_SUN + 5.0 * (AU / (10.0 * PC)).log10()) / (r_au * r_au)
    }

    const LUNAR_LAMBERT: Brdf = Brdf::LunarLambert { l: 0.9 };

    #[test]
    fn hg_zero_phase() {
        assert!(
            (asteroid_v(3.33, 0.12, 2.0, 1.0, 0.0) - (3.33 + 5.0 * 2f64.log10())).abs() < 1e-12
        );
        // Φ(0) = 1 for any G; Φ falls with phase and reaches 0 at α = π (V → +∞).
        for g in [-0.1, 0.0, 0.15, 0.5, 1.0] {
            assert_eq!(hg_phase(0.0, g), 1.0);
            assert!(hg_phase(0.3, g) < hg_phase(0.1, g));
            assert_eq!(hg_phase(PI, g), 0.0);
        }
        // Larger G (Φ₂ weighted more) gives a flatter phase curve.
        assert!(hg_phase(0.5, 0.5) > hg_phase(0.5, 0.15));
    }

    /// Values of Bowell et al.'s Φ = (1 − G)Φ₁ + GΦ₂ worked out by hand from the published
    /// A = (3.33, 1.87), B = (0.63, 1.22): at α = 20°, tan 10° = 0.176327, Φ₁ = 0.327614,
    /// Φ₂ = 0.798418, so Φ(G = 0.15) = 0.398244 (0.9996 mag below opposition).
    #[test]
    fn hg_phase_bowell_values() {
        let phi = hg_phase(20f64.to_radians(), 0.15);
        assert!((phi - 0.398_244_407_7).abs() < 1e-9, "{phi}");
        assert!((hg_phase(60f64.to_radians(), 0.25) - 0.167_143_351_8).abs() < 1e-9);
        // V = H + 5 log₁₀(rΔ) − 2.5 log₁₀Φ: H 3.34, G 0.12 at r 2.77 au, Δ 1.9 au, α 20°.
        let v = asteroid_v(3.34, 0.12, 2.77, 1.9, 20f64.to_radians());
        assert!((v - 7.985_000_622).abs() < 1e-8, "{v}");
        // The phase angle is unsigned: −α is α.
        assert_eq!(hg_phase(-0.4, 0.15), hg_phase(0.4, 0.15));
    }

    #[test]
    fn comet_total_magnitude_law() {
        // m = M1 + 5 log₁₀Δ + K1 log₁₀r: 1P/Halley's SBDB M1 5.5, K1 8.0 at r = 2 au, Δ = 0.5 au.
        let m = comet_total_v(5.5, 8.0, 2.0, 0.5);
        assert!((m - (5.5 + 5.0 * 0.5f64.log10() + 8.0 * 2f64.log10())).abs() < 1e-12);
        assert_eq!(comet_total_v(5.5, 8.0, 1.0, 1.0), 5.5);
    }

    #[test]
    fn lambert_phase_values() {
        assert!((lambert_phase(0.0) - 1.0).abs() < 1e-12);
        assert!((lambert_phase(FRAC_PI_2) - 1.0 / PI).abs() < 1e-12);
        assert!(lambert_phase(PI).abs() < 1e-15);
    }

    #[test]
    fn lunar_lambert_limits() {
        for (mu0, mu) in [(0.3, 0.7), (0.9, 0.2), (1.0, 1.0), (0.05, 0.99)] {
            // L = 0 is Lambert: the reflectance factor is μ₀ (per unit ρ).
            assert!((reflectance(Brdf::LunarLambert { l: 0.0 }, mu0, mu) - mu0).abs() < 1e-15);
            assert_eq!(reflectance(Brdf::Lambert, mu0, mu), mu0);
            // L = 1 is Lommel–Seeliger, 2μ₀/(μ₀ + μ).
            let ls = reflectance(Brdf::LunarLambert { l: 1.0 }, mu0, mu);
            assert!((ls - 2.0 * mu0 / (mu0 + mu)).abs() < 1e-15);
        }
        // L = 1 at μ₀ = μ is 1 per unit ρ.
        for mu in [0.1, 0.5, 1.0] {
            assert!((reflectance(Brdf::LunarLambert { l: 1.0 }, mu, mu) - 1.0).abs() < 1e-15);
        }
        // Unlit (μ₀ ≤ 0) or facing away (μ ≤ 0): no light.
        for brdf in [Brdf::Lambert, LUNAR_LAMBERT] {
            assert_eq!(reflectance(brdf, -0.1, 0.5), 0.0);
            assert_eq!(reflectance(brdf, 0.0, 0.5), 0.0);
            assert_eq!(reflectance(brdf, 0.5, -0.1), 0.0);
            assert_eq!(reflectance(brdf, 0.5, 0.0), 0.0);
        }
    }

    /// Folds in A0b's `rho_from_geometric_albedo_brief_values` (the values A8 expects, review I4).
    #[test]
    fn rho_from_geometric_albedo_cases() {
        // Lambert: ρ = 1.5 p.
        assert!((rho_from_geometric_albedo(Brdf::Lambert, 0.538) - 0.807).abs() < 1e-12);
        // Lunar-Lambert L = 0.9: ρ = p / (0.9 + 0.2/3) = p / 0.96667.
        let r = rho_from_geometric_albedo(LUNAR_LAMBERT, 0.12);
        assert!((r - 0.1241).abs() < 1e-4, "{r}");
        // Venus-like p = 0.689 > 2/3 would need ρ = 1.0335: capped at 1.
        assert_eq!(rho_from_geometric_albedo(Brdf::Lambert, 0.689), 1.0);
        // L = 0 is Lambert and L = 1 is Lommel–Seeliger (ρ = p).
        assert_eq!(
            rho_from_geometric_albedo(Brdf::LunarLambert { l: 0.0 }, 0.3),
            rho_from_geometric_albedo(Brdf::Lambert, 0.3)
        );
        assert!(
            (rho_from_geometric_albedo(Brdf::LunarLambert { l: 1.0 }, 0.3) - 0.3).abs() < 1e-15
        );
        // No silent replacement of a missing or impossible albedo.
        assert!(rho_from_geometric_albedo(Brdf::Lambert, f64::NAN).is_nan());
        assert!(rho_from_geometric_albedo(LUNAR_LAMBERT, -0.1).is_nan());
        assert_eq!(rho_from_geometric_albedo(Brdf::Lambert, 0.0), 0.0);

        // At α = 0 the disk integral with that ρ gives p·E☉·(R/d)² for both BRDFs.
        let (radius, dist, sun) = (2.4e6, 1.5e11, 1.2e5);
        for (brdf, p) in [(Brdf::Lambert, 0.538), (LUNAR_LAMBERT, 0.12)] {
            let rho = rho_from_geometric_albedo(brdf, p);
            let e = disk_illuminance_lux(brdf, rho, radius, sun, 0.0, dist);
            let expected = p * sun * (radius / dist).powi(2);
            assert!(
                (e / expected - 1.0).abs() < 1e-3,
                "{brdf:?}: {e} vs {expected}"
            );
        }
    }

    /// Review Focus 4: the disk integral (the CPU twin of `body.wgsl`) equals the point-source
    /// flux of a Lambert sphere, (2/3)ρ·E☉·(R/d)²·Φ(α), at every phase.
    #[test]
    fn disk_flux_matches_point_flux() {
        let (rho, radius, dist, sun) = (0.6, 1.0e6, 1.0e9, 1.0e5);
        for deg in [0.0_f64, 30.0, 90.0, 150.0] {
            let alpha = deg.to_radians();
            let disk = disk_illuminance_lux(Brdf::Lambert, rho, radius, sun, alpha, dist);
            let point = 2.0 / 3.0 * rho * sun * (radius / dist).powi(2) * lambert_phase(alpha);
            let rel = disk / point - 1.0;
            println!("α = {deg:5.1}°: disk / point − 1 = {rel:+.2e}");
            assert!(
                rel.abs() < 1e-3,
                "α = {deg}°: disk {disk}, point {point}, rel {rel:e}"
            );
        }
        // Fully dark at α = π, and the phase angle is unsigned.
        assert_eq!(
            disk_illuminance_lux(Brdf::Lambert, rho, radius, sun, PI, dist),
            0.0
        );
        let plus = disk_illuminance_lux(LUNAR_LAMBERT, rho, radius, sun, 0.8, dist);
        assert_eq!(
            disk_illuminance_lux(LUNAR_LAMBERT, rho, radius, sun, -0.8, dist),
            plus
        );
    }

    /// The lunar-Lambert branch at non-zero phase: with L = 1 (Lommel–Seeliger) the disk integral
    /// has the closed form Φ(α) = 1 − sin(α/2) tan(α/2) ln cot(α/4) relative to α = 0 (Fairbairn
    /// 2005, JRASC 99, 92), where it is ρ·E☉·(R/d)² (p = ρ for L = 1).
    #[test]
    fn lommel_seeliger_phase_law() {
        let ls = Brdf::LunarLambert { l: 1.0 };
        let (rho, radius, dist, sun) = (0.3, 1.0e6_f64, 1.0e9, 1.0e5);
        let full = rho * sun * (radius / dist).powi(2);
        for deg in [0.0_f64, 30.0, 90.0, 150.0] {
            let a = deg.to_radians();
            let phase = if a == 0.0 {
                1.0
            } else {
                1.0 - (a / 2.0).sin() * (a / 2.0).tan() * (1.0 / (a / 4.0).tan()).ln()
            };
            let disk = disk_illuminance_lux(ls, rho, radius, sun, a, dist);
            let rel = disk / (full * phase) - 1.0;
            assert!(rel.abs() < 1e-3, "α = {deg}°: rel {rel:e}");
        }
        // The disk flux scales linearly with ρ and E☉ and as (R/d)².
        let base = disk_illuminance_lux(LUNAR_LAMBERT, 0.2, 1.0e6, 1.0e5, 0.7, 1.0e9);
        let scaled = disk_illuminance_lux(LUNAR_LAMBERT, 0.4, 2.0e6, 3.0e5, 0.7, 4.0e9);
        assert!((scaled / base - 2.0 * 3.0 * 0.25).abs() < 1e-12);
    }

    /// A planet's Horizons spot and the BRDF B1 shades it with.
    struct PlanetCase {
        naif: i32,
        name: &'static str,
        brdf: Brdf,
        /// The brief's bound on |m − APmag| [mag].
        bound: f64,
    }

    /// Lambert for bodies with atmospheres and the giants, lunar-Lambert L = 0.9 for Mercury
    /// (plan B1); bounds 0.3 mag, 0.5 for Mercury, Venus and Saturn (brief, review I4).
    #[rustfmt::skip]
    const PLANETS: [PlanetCase; 7] = [
        PlanetCase { naif: 199, name: "Mercury", brdf: LUNAR_LAMBERT, bound: 0.5 },
        PlanetCase { naif: 299, name: "Venus", brdf: Brdf::Lambert, bound: 0.5 },
        PlanetCase { naif: 499, name: "Mars", brdf: Brdf::Lambert, bound: 0.3 },
        PlanetCase { naif: 599, name: "Jupiter", brdf: Brdf::Lambert, bound: 0.3 },
        PlanetCase { naif: 699, name: "Saturn", brdf: Brdf::Lambert, bound: 0.5 },
        PlanetCase { naif: 799, name: "Uranus", brdf: Brdf::Lambert, bound: 0.3 },
        PlanetCase { naif: 899, name: "Neptune", brdf: Brdf::Lambert, bound: 0.3 },
    ];

    /// A planet's geocentric Horizons sky spot on 2026-11-20: (APmag, r [au], Δ [au], α [deg]).
    fn horizons_spot(spots: &[uni_data::SkySpot], naif: i32) -> (f64, f64, f64, f64) {
        let target = naif.to_string();
        let spot = spots
            .iter()
            .find(|s| s.target == target && s.observer == 399)
            .unwrap_or_else(|| panic!("no geocentric sky spot for {target}"));
        let (Some(apmag), Some(r_au), Some(phase_deg)) = (spot.apmag, spot.r_au, spot.phase_deg)
        else {
            panic!("sky spot for {target} lacks APmag, r or phase");
        };
        (apmag, r_au, spot.delta_au, phase_deg)
    }

    fn load_spots() -> Vec<uni_data::SkySpot> {
        let path = uni_data::data_dir().join("golden").join("sky_spots.ron");
        uni_data::load_sky_spots(&path)
            .unwrap_or_else(|e| panic!("golden file missing: {} ({e})", path.display()))
    }

    /// Review I4 and A3 review r0 I1. The brief's BRDFs alone (Lambert; lunar-Lambert L 0.9 for
    /// Mercury) miss Horizons' APmag by −1.779 (Mercury, α 82°), +0.536 (Venus, 130°) and
    /// −0.466 mag (Mars, 38°): their phase curves are not the planets' observed ones (Mallama &
    /// Hilton 2018, which Horizons uses). With [`planet_phase_correction`] every planet meets
    /// the brief's bound (0.3 mag; 0.5 for Mercury, Venus and Saturn). Both columns are printed.
    #[test]
    fn planet_magnitudes_vs_horizons() {
        let spots = load_spots();
        let facts_path = uni_data::data_dir().join("baked").join("body_facts.ron");
        let facts = uni_data::load_body_facts(&facts_path)
            .unwrap_or_else(|e| panic!("facts file missing: {} ({e})", facts_path.display()));

        println!(
            "planet   BRDF     p_V    rho    r[au]   Δ[au]  α[deg]  APmag   BRDF−AP   C     m−AP  bound"
        );
        let mut failures = Vec::new();
        for case in &PLANETS {
            let target = case.naif.to_string();
            let (apmag, r_au, delta_au, phase_deg) = horizons_spot(&spots, case.naif);
            let fact = facts
                .iter()
                .find(|f| f.naif == case.naif)
                .unwrap_or_else(|| panic!("no body facts for {target}"));
            let p_v = fact
                .geometric_albedo
                .unwrap_or_else(|| panic!("no p_V for {target}"));
            let rho = rho_from_geometric_albedo(case.brdf, p_v);
            let e = disk_illuminance_lux(
                case.brdf,
                rho,
                fact.mean_radius_km * 1e3,
                sun_lux_at(r_au),
                phase_deg.to_radians(),
                delta_au * AU,
            );
            let e0 = star_illuminance_lux(0.0);
            let brdf_only = -2.5 * (e / e0).log10() - apmag;
            let c = planet_phase_correction(case.naif, case.brdf, phase_deg.to_radians());
            let diff = -2.5 * (e * c / e0).log10() - apmag;
            let brdf = match case.brdf {
                Brdf::Lambert => "Lambert",
                Brdf::LunarLambert { .. } => "LL 0.9",
            };
            println!(
                "{:<8} {brdf:<7} {p_v:5.3} {rho:6.4} {r_au:7.4} {delta_au:7.4} {phase_deg:7.3} {apmag:6.3} {brdf_only:+7.3} {c:6.3} {diff:+6.3}  {:.1}",
                case.name, case.bound
            );
            if !(diff.abs() <= case.bound) {
                failures.push(format!(
                    "{}: |m − APmag| = {:.3} > {}",
                    case.name,
                    diff.abs(),
                    case.bound
                ));
            }
        }
        assert!(failures.is_empty(), "{failures:#?}");
    }

    /// The transcription of Mallama & Hilton 2018 against Horizons' APmag (which uses it) at the
    /// same spots: V = V(1, α) + 5 log₁₀(rΔ). Mercury, Venus and Jupiter agree to ≤ 0.001 mag
    /// (bound 0.002). The others differ by what this module leaves out, each bound just above
    /// the measured gap: Mars's sub-Earth-longitude and L_s terms (−0.018); Saturn's rings
    /// (+0.085: APmag is brighter than the globe law, as the ringed Eq. 10 is at a ring opening
    /// near 7°, which this test does not compute; the 2018 paper says Horizons then reported the
    /// globe alone); Uranus's −8.4 × 10⁻⁴ φ′ sub-latitude term (+0.065, i.e. φ′ ≈ 77°); Neptune,
    /// whose APmag matches Eq. 16's V(1,0) without Eq. 17's phase term (+0.013, that term at
    /// 1.57°).
    #[test]
    fn mallama_hilton_matches_horizons_apmag() {
        let spots = load_spots();
        #[rustfmt::skip]
        let cases = [
            (199, 0.002), (299, 0.002), (499, 0.025), (599, 0.002),
            (699, 0.095), (799, 0.075), (899, 0.015),
        ];
        for (naif, tolerance) in cases {
            let (apmag, r_au, delta_au, phase_deg) = horizons_spot(&spots, naif);
            let v = mallama_hilton_v1(naif, phase_deg).unwrap() + 5.0 * (r_au * delta_au).log10();
            println!(
                "{naif}: V = {v:7.3}, APmag {apmag:7.3}, V − APmag = {:+.3}",
                v - apmag
            );
            assert!(
                (v - apmag).abs() <= tolerance,
                "{naif}: V {v} vs APmag {apmag}"
            );
        }
        // Mercury's law at 50°, summed by hand from Eq. 2: dimming 1.624 446 875 mag.
        let dimming = mallama_hilton_v1(199, 50.0).unwrap() - mallama_hilton_v1(199, 0.0).unwrap();
        assert!((dimming - 1.624_446_875).abs() < 1e-12, "{dimming}");
        // Jupiter's Eq. 9 at 90° (x = 0.5): −9.428 − 2.5 log₁₀(0.264 937 5) = −7.985 859 (by
        // hand).
        let jupiter = mallama_hilton_v1(599, 90.0).unwrap();
        assert!((jupiter + 7.985_858_6).abs() < 1e-6, "{jupiter}");
        assert_eq!(mallama_hilton_v1(301, 10.0), None);
    }

    /// The closed form of [`disk_phase_function`] against the numerical disk integral, for
    /// L ∈ {0, 0.5, 0.9, 1} and phases up to 170°.
    #[test]
    fn disk_phase_function_matches_integral() {
        for l in [0.0, 0.5, 0.9, 1.0] {
            let brdf = Brdf::LunarLambert { l };
            let norm = l + 2.0 * (1.0 - l) / 3.0;
            for deg in [0.0_f64, 10.0, 30.0, 60.0, 90.0, 120.0, 150.0, 170.0] {
                let a = deg.to_radians();
                let integral = disk_illuminance_lux(brdf, 1.0, 1.0, 1.0, a, 1.0) / norm;
                let closed = disk_phase_function(brdf, a);
                let rel = closed / integral - 1.0;
                assert!(rel.abs() < 1e-3, "L {l}, α {deg}°: {closed} vs {integral}");
            }
        }
        // Lambert is L = 0; Φ(0) = 1, Φ(π) = 0, unsigned, NaN propagates.
        for a in [0.0, 0.4, 1.3, 2.9] {
            assert_eq!(
                disk_phase_function(Brdf::Lambert, a),
                disk_phase_function(Brdf::LunarLambert { l: 0.0 }, a)
            );
            assert_eq!(
                disk_phase_function(LUNAR_LAMBERT, -a),
                disk_phase_function(LUNAR_LAMBERT, a)
            );
        }
        assert_eq!(disk_phase_function(LUNAR_LAMBERT, 0.0), 1.0);
        // At α = π only sin π's rounding (1.2e-16) is left.
        assert!(disk_phase_function(LUNAR_LAMBERT, PI).abs() < 1e-15);
        assert!(disk_phase_function(LUNAR_LAMBERT, PI - 1e-9) >= 0.0);
        assert!(disk_phase_function(Brdf::Lambert, f64::NAN).is_nan());
    }

    #[test]
    fn planet_phase_correction_properties() {
        let planets = [199, 299, 399, 499, 599, 699, 799, 899];
        for naif in planets {
            for brdf in [Brdf::Lambert, LUNAR_LAMBERT] {
                // C(0) = 1: the albedo calibration at opposition is untouched.
                assert!((planet_phase_correction(naif, brdf, 0.0) - 1.0).abs() < 1e-12);
                // Unsigned; finite and positive across the whole range, including past the
                // law's limit (held there) and at α = π.
                for deg in (0..=180).step_by(5).map(f64::from) {
                    let c = planet_phase_correction(naif, brdf, deg.to_radians());
                    assert!(c.is_finite() && c > 0.0, "{naif} {brdf:?} {deg}°: {c}");
                    assert_eq!(c, planet_phase_correction(naif, brdf, -deg.to_radians()));
                }
            }
        }
        // Held beyond the law's range: Jupiter's Eq. 9 stops at 130°.
        let at_limit = planet_phase_correction(599, Brdf::Lambert, 130f64.to_radians());
        assert_eq!(
            planet_phase_correction(599, Brdf::Lambert, 160f64.to_radians()),
            at_limit
        );
        // The piece joins are continuous to the paper's own steps: Venus 163.7° (0.007 mag),
        // Mars 50°, Jupiter 12°, Saturn 6° (≤ 0.001 mag).
        for (naif, join, step_mag) in [
            (299, 163.7, 0.008),
            (499, 50.0, 0.001),
            (599, 12.0, 0.001),
            (699, 6.0, 0.001),
        ] {
            let below =
                planet_phase_correction(naif, Brdf::Lambert, (join - 1e-9_f64).to_radians());
            let above =
                planet_phase_correction(naif, Brdf::Lambert, (join + 1e-9_f64).to_radians());
            let jump = 2.5 * (below / above).log10();
            assert!(jump.abs() < step_mag, "{naif} at {join}°: {jump} mag");
        }
        // Mercury at α 50° with lunar-Lambert L 0.9: 10^(−0.4 · 1.624 446 875) / Φ_LL(50°).
        let want = 10_f64.powf(-0.4 * 1.624_446_875)
            / disk_phase_function(LUNAR_LAMBERT, 50f64.to_radians());
        let c = planet_phase_correction(199, LUNAR_LAMBERT, 50f64.to_radians());
        assert!((c / want - 1.0).abs() < 1e-12, "{c} vs {want}");
        // Bodies without a law are left alone; NaN propagates.
        assert_eq!(planet_phase_correction(301, LUNAR_LAMBERT, 1.0), 1.0);
        assert_eq!(planet_phase_correction(999, LUNAR_LAMBERT, 1.0), 1.0);
        assert!(planet_phase_correction(199, LUNAR_LAMBERT, f64::NAN).is_nan());
    }

    #[test]
    fn sun_radius_from_magnitude() {
        assert!((star_radius_m(M_V_SUN, TEFF_SUN_K) / R_SUN_M - 1.0).abs() < 0.03);
    }

    #[test]
    fn bolometric_correction_flower_scale() {
        // Flower's polynomial gives the Sun BC_V = −0.08 (Torres 2010, Table 3).
        let bc_sun = bolometric_correction_v(TEFF_SUN_K);
        assert!((bc_sun + 0.08).abs() < 0.005, "{bc_sun}");
        // BC_V peaks near +0.02 for late-A/early-F stars and falls on both sides.
        assert!(bolometric_correction_v(8000.0) > bc_sun);
        assert!(bolometric_correction_v(4000.0) < -1.0);
        assert!(bolometric_correction_v(20_000.0) < -1.5);
        // Outside 3000–50 000 K the value at the edge is used: finite, never runaway (the
        // polynomial reaches BC −24 at 2000 K and turns back up above 60 600 K).
        assert_eq!(
            bolometric_correction_v(1000.0),
            bolometric_correction_v(3000.0)
        );
        assert_eq!(
            bolometric_correction_v(1.0e6),
            bolometric_correction_v(50_000.0)
        );
        // Inside the range the fit falls monotonically from its peak near 8000 K to 50 000 K.
        let mut last = bolometric_correction_v(8000.0);
        for t in (9..=50).map(|k| f64::from(k) * 1000.0) {
            let bc = bolometric_correction_v(t);
            assert!(bc < last, "BC_V({t} K) = {bc} is not below {last}");
            last = bc;
        }
        // Torres 2010 Table 1 at its segment boundaries (log T_eff = 3.70 and 3.90): the
        // published fit jumps by 0.022 and 0.003 mag there; both segments are used as printed.
        assert!((bolometric_correction_v(10f64.powf(3.70) * 0.999_999) + 0.2783).abs() < 1e-3);
        assert!((bolometric_correction_v(10f64.powf(3.70) * 1.000_001) + 0.3007).abs() < 1e-3);
        assert!(bolometric_correction_v(1000.0).is_finite());
        assert!(bolometric_correction_v(0.0).is_nan());
        assert!(bolometric_correction_v(-5.0).is_nan());
        // A hotter star of the same M_V is smaller (larger |BC| does not undo T⁻²).
        assert!(star_radius_m(0.0, 20_000.0) < star_radius_m(0.0, 6000.0));
    }
}
