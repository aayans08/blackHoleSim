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
/// - Saturn's globe without its rings, Eq. 11 for α ≤ 6°, Eq. 12 above. The renderer draws the
///   rings as their own pass, so the disk's correction is the globe's; the point path adds the
///   rings with the ringed law, Eq. 10 ([`saturn_ring_factor`]).
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

/// The H–G slope used where none was measured: G = 0.15, the value JPL's Small-Body Database
/// documents as its default and lists for such objects (the slope `uni-data`'s asteroid loader
/// and uni-sky's procedural belt use).
pub const DEFAULT_HG_SLOPE: f64 = 0.15;

/// The largest phase angle [deg] at which the Moon's law is evaluated; beyond it
/// [`phase_correction`] holds its value there. Krisciunas & Schaefer state no upper limit; the
/// hold keeps C bounded where the BRDF's crescent Φ_brdf → 0 (C would reach 7 at 178°), and
/// 170° is the hold A3 chose for Earth's and Mars's laws (`mallama_hilton_max_deg`).
const MOON_LAW_MAX_DEG: f64 = 170.0;

/// The observed disk-integrated phase law a body follows. The BRDF ([`Brdf`]) shapes the
/// radiance across a resolved disk; [`phase_correction`] scales its total to this law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseLaw {
    /// No observed law: the BRDF's own phase function (C = 1).
    Brdf,
    /// Mallama & Hilton 2018's law for planet `naif` (Saturn: the globe alone).
    Planet(NaifId),
    /// The Moon's observed law, Krisciunas & Schaefer 1991 Eq. 9 (from Allen 1976).
    Moon,
    /// The IAU H, G system (Bowell et al. 1989) with slope `g`.
    HG { g: f64 },
}

/// NAIF's asteroid id ranges (NAIF, "NAIF Integer ID codes" Required Reading, §Asteroids; the
/// same ranges as `uni_data::smallbody`): original schema numbered 2 000 001–2 999 999, extended
/// schema numbered 20 000 001–49 999 999 and provisional 50 000 001–99 999 999.
fn is_asteroid_naif(naif: NaifId) -> bool {
    (2_000_001..=2_999_999).contains(&naif) || (20_000_001..=99_999_999).contains(&naif)
}

impl PhaseLaw {
    /// The law body `naif` follows when nothing else is known about it: Mallama & Hilton for the
    /// eight planets, the lunar law for the Moon (301), H–G with [`DEFAULT_HG_SLOPE`] for an
    /// asteroid id (the caller passes the record's own slope where it has one), the BRDF for
    /// everything else (the other moons, Pluto, comets).
    pub fn for_naif(naif: NaifId) -> PhaseLaw {
        if mallama_hilton_max_deg(naif).is_some() {
            PhaseLaw::Planet(naif)
        } else if naif == 301 {
            PhaseLaw::Moon
        } else if is_asteroid_naif(naif) {
            PhaseLaw::HG {
                g: DEFAULT_HG_SLOPE,
            }
        } else {
            PhaseLaw::Brdf
        }
    }
}

/// The Moon's V(α) − V(0) [mag] at phase `a_deg` ≥ 0: 0.026α + 4 × 10⁻⁹α⁴, α in degrees.
///
/// Krisciunas & Schaefer 1991 (PASP 103, 1033) Eq. 9, m = −12.73 + 0.026|α| + 4 × 10⁻⁹α⁴ at
/// the mean distance, "see page 144 of Allen 1976" (*Astrophysical Quantities*, 3rd ed.); they
/// note it agrees at α = 0 with Lane & Irvine 1973 (AJ 78, 267) extrapolated, and that it leaves
/// out the opposition surge (the full Moon is ~35 % brighter for |α| < 7°), which is not
/// modelled here either.
fn moon_dimming_mag(a_deg: f64) -> f64 {
    0.026 * a_deg + 4.0e-9 * a_deg.powi(4)
}

/// Observed-phase-curve correction for a body following `law`, drawn with `brdf`: the factor
/// that makes the BRDF's disk-integrated flux at phase `phase_rad` follow the law.
///
/// C(α) = Φ_law(α) / Φ_brdf(α), with Φ_brdf = [`disk_phase_function`] and Φ_law the law's
/// brightness relative to α = 0:
/// - [`PhaseLaw::Planet`]: 10^(−0.4[V(1, α) − V(1, 0)]) with Mallama & Hilton 2018's V(1, ·)
///   (`mallama_hilton_v1`), held beyond the law's range (`mallama_hilton_max_deg`). Lambert and
///   lunar-Lambert spheres cannot reproduce the observed curves (A3 review r0 I1): at the
///   2026-11-20 spots they put Mercury 1.78 mag too bright at α = 82°, Mars 0.47 mag too
///   bright at 38° and Venus 0.54 mag too faint at 130°; with C all seven planets come within
///   0.1 mag of Horizons (`planet_magnitudes_vs_horizons`).
/// - [`PhaseLaw::Moon`]: 10^(−0.4 [`moon_dimming_mag`]) (Krisciunas & Schaefer 1991 Eq. 9),
///   held beyond 170° ([`MOON_LAW_MAX_DEG`]). Lunar-Lambert L 0.9 alone dims the Moon by
///   1.07 mag at α = 90°, the observed law by 2.60 mag (final review physics I3).
/// - [`PhaseLaw::HG`]: Bowell et al.'s Φ(α; G) ([`hg_phase`]) at every phase, as uni-sky draws
///   the fainter asteroids (`asteroid_v`), so the two agree at H = 10. Lunar-Lambert alone dims
///   an asteroid by 0.08 mag at α = 20°, H–G (G 0.15) by 1.00 mag.
/// - [`PhaseLaw::Brdf`]: 1.
///
/// C(0) = 1, so the geometric albedo calibration ([`rho_from_geometric_albedo`]) is untouched;
/// only the phase dependence changes. Use: multiply the point flux (`disk_illuminance_lux`, or
/// the closed form in [`disk_phase_function`]'s doc) by C, and the disk's radiance by the same C
/// at every pixel, so the two paths still agree at the handover. The radiance distribution
/// across the disk keeps the BRDF's shape; only its total follows the observed curve. At α = π
/// (Φ_brdf = 0, nothing lit is visible) an H–G or lunar law gives 0. The phase angle is
/// unsigned; NaN gives NaN.
pub fn phase_correction(law: PhaseLaw, brdf: Brdf, phase_rad: f64) -> f64 {
    if phase_rad.is_nan() {
        return f64::NAN;
    }
    let a_deg = phase_rad.abs().to_degrees().min(180.0);
    let ratio = |observed: f64, a_deg: f64| {
        let phi = disk_phase_function(brdf, a_deg.to_radians());
        if phi > 0.0 { observed / phi } else { 0.0 }
    };
    match law {
        PhaseLaw::Brdf => 1.0,
        PhaseLaw::Planet(naif) => {
            let (Some(max_deg), Some(v0)) =
                (mallama_hilton_max_deg(naif), mallama_hilton_v1(naif, 0.0))
            else {
                return 1.0;
            };
            let a = a_deg.min(max_deg);
            let Some(v) = mallama_hilton_v1(naif, a) else {
                return 1.0;
            };
            ratio(10_f64.powf(-0.4 * (v - v0)), a)
        }
        PhaseLaw::Moon => {
            let a = a_deg.min(MOON_LAW_MAX_DEG);
            ratio(10_f64.powf(-0.4 * moon_dimming_mag(a)), a)
        }
        PhaseLaw::HG { g } => ratio(hg_phase(a_deg.to_radians(), g), a_deg),
    }
}

/// [`phase_correction`] with the law of body `naif` ([`PhaseLaw::for_naif`]): Mallama & Hilton
/// for the planets, the lunar law for the Moon, H–G with the default slope for an asteroid id, 1
/// for every other body. The bodies pass (uni-planet) and the point path both call it, so the
/// point ↔ disk handover stays continuous; uni-sim's point path passes an asteroid's own slope
/// through [`phase_correction`] instead.
pub fn planet_phase_correction(naif: NaifId, brdf: Brdf, phase_rad: f64) -> f64 {
    phase_correction(PhaseLaw::for_naif(naif), brdf, phase_rad)
}

/// Mallama & Hilton 2018 Eq. 10: V(1, α) of Saturn *with its rings*, α in degrees, `beta_deg`
/// the effective ring inclination ([`saturn_effective_ring_tilt_deg`]):
/// V = −8.914 − 1.825 sin β + 0.026α − 0.378 sin β e^(−2.25α). The paper fits it (Mallama 2012)
/// for α < 6.5° and β < 27°; it is the ringed magnitude *The Astronomical Almanac* tabulates.
fn saturn_ringed_v1(a_deg: f64, beta_deg: f64) -> f64 {
    let s = beta_deg.to_radians().sin();
    -8.914 - 1.825 * s + 0.026 * a_deg - 0.378 * s * (-2.25 * a_deg).exp()
}

/// Eq. 10's range: α < 6.5°, β < 27° (Mallama & Hilton 2018 §3.6).
const SATURN_RING_LAW_MAX_ALPHA_DEG: f64 = 6.5;
const SATURN_RING_LAW_MAX_BETA_DEG: f64 = 27.0;

/// Eq. 10's effective ring inclination β [deg] from the planetocentric latitudes of the observer
/// (β_E) and of the Sun (β_S) above Saturn's ring (equatorial) plane, in degrees: (β_E β_S)^½
/// when they have the same sign, 0 when the Sun lights one face and the observer sees the other
/// (the rings are "backlit and very faint"; Mallama & Hilton 2018 §3.6). Unsigned.
pub fn saturn_effective_ring_tilt_deg(observer_lat_deg: f64, sun_lat_deg: f64) -> f64 {
    let product = observer_lat_deg * sun_lat_deg;
    if product > 0.0 { product.sqrt() } else { 0.0 }
}

/// The factor k ≥ 1 by which Saturn's rings brighten Saturn seen as a point: the ringed law
/// (Eq. 10) over the globe's law that [`planet_phase_correction`] gives (Eq. 11/12), at phase
/// `phase_rad` with the observer and the Sun at planetocentric latitudes `observer_lat_rad`,
/// `sun_lat_rad` above the ring plane (final review physics I2).
///
/// k = max(1, 10^(−0.4[V₁₀(α, β) − V_globe(α)])), with β = [`saturn_effective_ring_tilt_deg`]
/// held at 27° and α held at 6.5° beyond Eq. 10's range:
/// - The floor at 1: Eq. 10 at β = 0 is the globe with a 0.026 mag/deg phase slope that the
///   paper calls "unrealistically steep for the globe alone, … probably due to aliasing from the
///   much steeper slope of the ring system", and adopts Eq. 11 for the globe; so the rings'
///   share is never taken below zero. The floor acts only for β below 1° (α = 0) to 5.7°
///   (α = 6.5°), where Eq. 10 is 0.036–0.18 mag fainter than the globe law.
/// - Beyond α = 6.5° the paper has no ringed law ("not enough information available"); k is
///   held at its 6.5° value, as [`phase_correction`] holds each planet's law beyond its range,
///   so the rings keep their share of the globe's light. Approximate, and only reached away from
///   the Earth (Saturn's phase from the Earth never exceeds 6.4°).
///
/// At β = 26.7° (rings fully open) and α = 0, k = 2.40 (0.95 mag); at β 20°, α 5°, 1.55
/// (0.47 mag); backlit rings or edge-on rings give 1. NaN in gives NaN.
pub fn saturn_ring_factor(phase_rad: f64, observer_lat_rad: f64, sun_lat_rad: f64) -> f64 {
    if phase_rad.is_nan() || observer_lat_rad.is_nan() || sun_lat_rad.is_nan() {
        return f64::NAN;
    }
    let a = phase_rad
        .abs()
        .to_degrees()
        .min(SATURN_RING_LAW_MAX_ALPHA_DEG);
    let beta =
        saturn_effective_ring_tilt_deg(observer_lat_rad.to_degrees(), sun_lat_rad.to_degrees())
            .min(SATURN_RING_LAW_MAX_BETA_DEG);
    let Some(globe) = mallama_hilton_v1(699, a) else {
        return 1.0;
    };
    10_f64
        .powf(-0.4 * (saturn_ringed_v1(a, beta) - globe))
        .max(1.0)
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
    /// (plan B1); bounds 0.3 mag, 0.5 for Mercury and Venus (brief, review I4). Saturn's was 0.5
    /// for its rings until the point path drew them (final review physics I2); 0.3 now.
    #[rustfmt::skip]
    const PLANETS: [PlanetCase; 7] = [
        PlanetCase { naif: 199, name: "Mercury", brdf: LUNAR_LAMBERT, bound: 0.5 },
        PlanetCase { naif: 299, name: "Venus", brdf: Brdf::Lambert, bound: 0.5 },
        PlanetCase { naif: 499, name: "Mars", brdf: Brdf::Lambert, bound: 0.3 },
        PlanetCase { naif: 599, name: "Jupiter", brdf: Brdf::Lambert, bound: 0.3 },
        PlanetCase { naif: 699, name: "Saturn", brdf: Brdf::Lambert, bound: 0.3 },
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
    /// Hilton 2018, which Horizons uses). With [`planet_phase_correction`] (and Saturn's rings,
    /// [`saturn_ring_factor`], as the point path draws them) every planet meets the brief's
    /// bound (0.3 mag; 0.5 for Mercury and Venus). Both columns are printed.
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
            let mut c = planet_phase_correction(case.naif, case.brdf, phase_deg.to_radians());
            if case.naif == 699 {
                // The point path's rings (`saturn_ring_factor`), the tilts from this crate.
                let (_, _, _, beta_e, beta_s) = saturn_geometry("2026-11-20T00:00:00Z");
                c *= saturn_ring_factor(
                    phase_deg.to_radians(),
                    beta_e.to_radians(),
                    beta_s.to_radians(),
                );
            }
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
    /// (bound 0.002). The others differ by what this function leaves out, each bound just above
    /// the measured gap: Mars's sub-Earth-longitude and L_s terms (−0.018); Saturn's rings
    /// (+0.085: APmag is the ringed Eq. 10, which `saturn_with_rings_vs_mallama_hilton` matches
    /// to 0.001 mag at this spot; this is the globe law alone, the disk's correction);
    /// Uranus's −8.4 × 10⁻⁴ φ′ sub-latitude term (+0.065, i.e. φ′ ≈ 77°); Neptune,
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
        // Bodies without a law are left alone (Pluto, a moon other than ours, a comet's SPK-ID);
        // the Moon and asteroid ids have laws now (final review physics I3); NaN propagates.
        assert_eq!(planet_phase_correction(999, LUNAR_LAMBERT, 1.0), 1.0);
        assert_eq!(planet_phase_correction(401, LUNAR_LAMBERT, 1.0), 1.0);
        assert_eq!(planet_phase_correction(1_000_036, LUNAR_LAMBERT, 1.0), 1.0);
        assert!(planet_phase_correction(301, LUNAR_LAMBERT, 1.0) < 1.0);
        assert!(planet_phase_correction(2_000_001, LUNAR_LAMBERT, 1.0) < 1.0);
        assert!(planet_phase_correction(199, LUNAR_LAMBERT, f64::NAN).is_nan());
    }

    /// The Moon's magnitude at the mean distance against Krisciunas & Schaefer 1991 Eq. 9
    /// (Allen 1976 p. 144): V = −12.73 + 0.026|α| + 4 × 10⁻⁹α⁴, full (α = 0, without the
    /// opposition surge the law also leaves out) −12.73 and quarter (α = 90°) −10.13. The model
    /// is the one the renderer draws: lunar-Lambert L 0.9 with the fact sheet's p_V 0.12 and
    /// R 1737.4 km, at its semi-major axis 384 400 km and 1 au from the Sun, times C. Before the
    /// fix (C = 1) the quarter Moon was 1.47 mag too bright (final review physics I3).
    #[test]
    fn moon_full_and_quarter_vs_krisciunas_schaefer() {
        let facts_path = uni_data::data_dir().join("baked").join("body_facts.ron");
        let facts = uni_data::load_body_facts(&facts_path)
            .unwrap_or_else(|e| panic!("facts file missing: {} ({e})", facts_path.display()));
        let moon = facts.iter().find(|f| f.naif == 301).expect("Moon facts");
        let p_v = moon.geometric_albedo.expect("Moon p_V");
        let delta_m = moon.semi_major_au.expect("Moon a") * AU;
        let rho = rho_from_geometric_albedo(LUNAR_LAMBERT, p_v);
        let e0 = star_illuminance_lux(0.0);
        let vmag = |alpha_deg: f64, c: f64| {
            let a = alpha_deg.to_radians();
            let e = disk_illuminance_lux(
                LUNAR_LAMBERT,
                rho,
                moon.mean_radius_km * 1e3,
                sun_lux_at(1.0),
                a,
                delta_m,
            );
            -2.5 * (e * c / e0).log10()
        };
        for (alpha, cited) in [(0.0, -12.73), (90.0_f64, -10.13)] {
            let law = -12.73 + 0.026 * alpha + 4.0e-9 * alpha.powi(4);
            assert!((law - cited).abs() < 0.005, "{alpha}°: law {law}");
            let c = phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, alpha.to_radians());
            let (bare, drawn) = (vmag(alpha, 1.0), vmag(alpha, c));
            println!(
                "Moon α {alpha:5.1}°: BRDF alone {bare:7.3}, drawn {drawn:7.3}, Krisciunas & \
                 Schaefer {law:7.3} (C {c:.4})"
            );
            // p_V 0.12 gives −12.71 at α = 0, 0.02 mag from the law's −12.73.
            assert!((drawn - law).abs() < 0.05, "{alpha}°: {drawn} vs {law}");
        }
        // The BRDF alone: 1.07 mag of phase dimming at 90° against the law's 2.60.
        let bare_dimming = vmag(90.0, 1.0) - vmag(0.0, 1.0);
        assert!((bare_dimming - 1.072).abs() < 0.01, "{bare_dimming}");
        // Through the NAIF id, as the bodies pass and the point path call it.
        for deg in [5.0_f64, 45.0, 90.0, 150.0] {
            let a = deg.to_radians();
            assert_eq!(
                planet_phase_correction(301, LUNAR_LAMBERT, a),
                phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, a)
            );
        }
        // Held beyond 170°; bounded and positive everywhere.
        let held = phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, 170f64.to_radians());
        assert_eq!(
            phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, 178f64.to_radians()),
            held
        );
        for deg in (0..=180).map(f64::from) {
            let c = phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, deg.to_radians());
            assert!(c > 0.0 && c <= 1.0, "{deg}°: {c}");
        }
    }

    /// An asteroid's correction makes the lunar-Lambert flux follow Bowell et al.'s H–G law at
    /// every phase: C·Φ_LL = Φ_HG (final review physics I3). At α = 20° lunar-Lambert alone dims
    /// by 0.08 mag, H–G (G 0.15) by 1.00 mag; at 65° (the C2 belt view) by 0.60 against 2.30
    /// (G 0.15) or 2.37 (Ceres's G 0.12). An asteroid NAIF id without a known slope gets G 0.15.
    #[test]
    fn asteroid_correction_follows_h_g() {
        for g in [-0.04, 0.12, 0.15, 0.32, 0.5] {
            for deg in (0..=175).step_by(5).map(f64::from) {
                let a = deg.to_radians();
                let c = phase_correction(PhaseLaw::HG { g }, LUNAR_LAMBERT, a);
                let followed = c * disk_phase_function(LUNAR_LAMBERT, a);
                assert!(
                    (followed - hg_phase(a, g)).abs() < 1e-12,
                    "G {g} at {deg}°: {followed} vs {}",
                    hg_phase(a, g)
                );
                assert!(
                    c.is_finite() && (0.0..=1.0).contains(&c),
                    "G {g} {deg}°: {c}"
                );
            }
            // At α = π nothing lit is visible: 0, never NaN.
            assert_eq!(phase_correction(PhaseLaw::HG { g }, LUNAR_LAMBERT, PI), 0.0);
        }
        let mag = |x: f64| -2.5 * x.log10();
        let ll = |deg: f64| mag(disk_phase_function(LUNAR_LAMBERT, deg.to_radians()));
        let hg = |deg: f64, g: f64| mag(hg_phase(deg.to_radians(), g));
        println!(
            "dimming [mag]: 20° LL {:.3} HG(0.15) {:.3}; 65° LL {:.3} HG(0.15) {:.3} HG(0.12) {:.3}",
            ll(20.0),
            hg(20.0, 0.15),
            ll(65.0),
            hg(65.0, 0.15),
            hg(65.0, 0.12)
        );
        assert!((ll(20.0) - 0.08).abs() < 0.01 && (hg(20.0, 0.15) - 1.00).abs() < 0.005);
        assert!((ll(65.0) - 0.60).abs() < 0.01 && (hg(65.0, 0.12) - 2.37).abs() < 0.01);
        for naif in [2_000_001, 2_000_004, 20_000_001, 50_000_001] {
            assert_eq!(
                PhaseLaw::for_naif(naif),
                PhaseLaw::HG {
                    g: DEFAULT_HG_SLOPE
                },
                "{naif}"
            );
        }
        assert_eq!(PhaseLaw::for_naif(301), PhaseLaw::Moon);
        assert_eq!(PhaseLaw::for_naif(699), PhaseLaw::Planet(699));
        assert_eq!(PhaseLaw::for_naif(999), PhaseLaw::Brdf);
        assert_eq!(phase_correction(PhaseLaw::Brdf, LUNAR_LAMBERT, 1.0), 1.0);
        assert!(phase_correction(PhaseLaw::Moon, LUNAR_LAMBERT, f64::NAN).is_nan());
    }

    /// Saturn's position, the Earth's and the Sun's at `iso` (UTC) from the Ephemeris, and the
    /// WGCCRE pole: (r [au], Δ [au], α [deg], β_E [deg], β_S [deg]) of Mallama & Hilton's Eq. 10.
    fn saturn_geometry(iso: &str) -> (f64, f64, f64, f64, f64) {
        let t = uni_core::Epoch::parse_iso_utc(iso).expect("date");
        let e = crate::ephemeris::Ephemeris::new(Vec::new());
        let saturn = e.ssb_position(699, t).expect("Saturn");
        let to_sun = e.ssb_position(10, t).expect("Sun") - saturn;
        let to_earth = e.ssb_position(399, t).expect("Earth") - saturn;
        let pole = crate::rotation::pole_icrs(699, t).expect("Saturn pole");
        let lat = |v: glam::DVec3| v.normalize().dot(pole).asin().to_degrees();
        (
            to_sun.length() / AU,
            to_earth.length() / AU,
            to_sun.angle_between(to_earth).to_degrees(),
            lat(to_earth),
            lat(to_sun),
        )
    }

    /// Final review physics I2. Saturn with its rings (Mallama & Hilton 2018 Eq. 10) against the
    /// paper's own numbers (§4.6 and Table 5), with this crate's geometry (Standish, the WGCCRE
    /// pole):
    /// - Table 5's variations: 0.99 mag from the ring inclination (β 0 → 26.8°, α = 0) and 0.34
    ///   mag from the phase (α 0 → 6.3°, β 26.8°), each to the table's 0.01;
    /// - "The brightest magnitude in the time span examined is −0.55 on 2032-Dec-25", rings open
    ///   (|β| ≥ 20°): Eq. 10 at this crate's geometry within 0.02 mag (the paper's ephemeris and
    ///   rounding), and Saturn as the point path draws it (globe law × `saturn_ring_factor`)
    ///   within 0.05 (Eq. 11's globe at α ≈ 0 is 0.036 mag brighter than Eq. 10's β = 0 term);
    /// - "The faintest would be 1.17 on 2025-Apr-20 when the rings are nearly edge-on": Eq. 10
    ///   within 0.02; the point path is brighter there by the floor at 1 (Eq. 10's β → 0 globe
    ///   is 0.036 + 0.026α mag fainter than Eq. 11), stated and bounded (0.117 mag at α 3.3°);
    /// - Horizons' APmag at the 2026-11-20 golden spot (β 7.2°): the point within 0.01 mag
    ///   (measured +0.554 against +0.555), so Horizons now reports the ringed law.
    /// The globe-only law misses the brightest value by 0.90 mag, which is what the point path
    /// drew before the fix. Measured: 2032-12-25 Eq. 10 −0.548 (β 26.3°, α 0.1°), 2025-04-20
    /// +1.170 (rings backlit: β_E −1.5°, β_S +0.2°).
    #[test]
    fn saturn_with_rings_vs_mallama_hilton() {
        let incl = saturn_ringed_v1(0.0, 0.0) - saturn_ringed_v1(0.0, 26.8);
        let phase = saturn_ringed_v1(6.3, 26.8) - saturn_ringed_v1(0.0, 26.8);
        println!("Table 5: inclination {incl:.4} mag (0.99), phase {phase:.4} mag (0.34)");
        assert!((incl - 0.99).abs() <= 0.01, "{incl}");
        assert!((phase - 0.34).abs() <= 0.01, "{phase}");

        for (iso, cited, open) in [
            ("2032-12-25T00:00:00Z", -0.55, true),
            ("2025-04-20T00:00:00Z", 1.17, false),
        ] {
            let (r, delta, alpha, beta_e, beta_s) = saturn_geometry(iso);
            let beta = saturn_effective_ring_tilt_deg(beta_e, beta_s);
            let distance = 5.0 * (r * delta).log10();
            let eq10 = saturn_ringed_v1(alpha, beta) + distance;
            let globe = mallama_hilton_v1(699, alpha).unwrap() + distance;
            let k =
                saturn_ring_factor(alpha.to_radians(), beta_e.to_radians(), beta_s.to_radians());
            let point = globe - 2.5 * k.log10();
            println!(
                "{iso}: r {r:.4} Δ {delta:.4} α {alpha:.3}° β_E {beta_e:+.3}° β_S {beta_s:+.3}° \
                 β {beta:.3}°: Eq. 10 {eq10:+.3}, globe {globe:+.3}, point {point:+.3} (k {k:.4}), \
                 paper {cited:+.2}"
            );
            assert!(
                (eq10 - cited).abs() <= 0.02,
                "{iso}: Eq. 10 {eq10} vs {cited}"
            );
            if open {
                assert!(beta_e.abs() >= 20.0 && beta >= 20.0, "{iso}: rings open");
                assert!(
                    (point - cited).abs() <= 0.05,
                    "{iso}: point {point} vs {cited}"
                );
                assert!(
                    globe - cited > 0.8,
                    "{iso}: the globe alone is far too faint"
                );
            } else {
                // Edge-on: the floor keeps the globe; brighter than Eq. 10 by its β = 0 offset.
                let offset = 0.036 + 0.026 * alpha + 3.7e-4 * alpha - 6.16e-4 * alpha * alpha;
                assert_eq!(k, 1.0, "{iso}: β {beta}");
                assert!(
                    (cited - point - offset).abs() <= 0.03,
                    "{iso}: point {point}, paper {cited}, expected offset {offset}"
                );
            }
        }

        // Horizons' APmag at the 2026-11-20 golden spot (r, Δ, α from it; β from this crate).
        let (apmag, r, delta, alpha) = horizons_spot(&load_spots(), 699);
        let (_, _, _, beta_e, beta_s) = saturn_geometry("2026-11-20T00:00:00Z");
        let beta = saturn_effective_ring_tilt_deg(beta_e, beta_s);
        let distance = 5.0 * (r * delta).log10();
        let eq10 = saturn_ringed_v1(alpha, beta) + distance;
        let k = saturn_ring_factor(alpha.to_radians(), beta_e.to_radians(), beta_s.to_radians());
        let point = mallama_hilton_v1(699, alpha).unwrap() + distance - 2.5 * k.log10();
        println!(
            "2026-11-20 (Horizons spot): α {alpha:.3}° β_E {beta_e:+.3}° β_S {beta_s:+.3}° β \
             {beta:.3}°: Eq. 10 {eq10:+.3}, point {point:+.3}, APmag {apmag:+.3}"
        );
        // Horizons now reports the ringed law (the 2018 paper says it then gave the globe).
        assert!((point - apmag).abs() <= 0.01, "{point} vs APmag {apmag}");
    }

    /// The ring factor's shape: ≥ 1, 1 for backlit or edge-on rings, the floor, the holds
    /// beyond Eq. 10's α 6.5° and β 27°, continuity across them, unsigned phase, NaN.
    #[test]
    fn saturn_ring_factor_properties() {
        let rad = f64::to_radians;
        // Fully open (β 26.7°), α 0: 0.954 mag, k = 2.41; β 20°, α 5°: 0.472 mag.
        let open = saturn_ring_factor(0.0, rad(26.7), rad(26.7));
        assert!((2.5 * open.log10() - 0.954).abs() < 0.002, "{open}");
        let mid = saturn_ring_factor(rad(5.0), rad(20.0), rad(20.0));
        assert!((2.5 * mid.log10() - 0.472).abs() < 0.002, "{mid}");
        // Backlit (opposite faces) and edge-on: 1.
        assert_eq!(saturn_ring_factor(rad(3.0), rad(20.0), rad(-20.0)), 1.0);
        assert_eq!(saturn_ring_factor(rad(3.0), 0.0, rad(20.0)), 1.0);
        // The floor acts below β ≈ 0.94° at α = 0 and ≈ 5.74° at α = 6.5°.
        assert_eq!(saturn_ring_factor(0.0, rad(0.9), rad(0.9)), 1.0);
        assert!(saturn_ring_factor(0.0, rad(1.0), rad(1.0)) > 1.0);
        assert_eq!(saturn_ring_factor(rad(6.5), rad(5.7), rad(5.7)), 1.0);
        assert!(saturn_ring_factor(rad(6.5), rad(5.8), rad(5.8)) > 1.0);
        // Held beyond α 6.5° (continuous there) and β 27°.
        let at = saturn_ring_factor(rad(6.5), rad(25.0), rad(25.0));
        assert_eq!(saturn_ring_factor(rad(90.0), rad(25.0), rad(25.0)), at);
        let below = saturn_ring_factor(rad(6.5 - 1e-9), rad(25.0), rad(25.0));
        assert!((below / at - 1.0).abs() < 1e-9);
        assert_eq!(
            saturn_ring_factor(rad(2.0), rad(80.0), rad(26.0)),
            saturn_ring_factor(rad(2.0), rad(27.0), rad(27.0))
        );
        // ≥ 1 everywhere, unsigned in α and symmetric in the two latitudes.
        for a in [0.0, 1.0, 3.0, 6.0, 30.0, 120.0] {
            for b in [-26.0, -10.0, 0.0, 2.0, 15.0, 26.0] {
                let k = saturn_ring_factor(rad(a), rad(b), rad(b * 0.9));
                assert!(k >= 1.0 && k.is_finite(), "{a} {b}: {k}");
                assert_eq!(k, saturn_ring_factor(rad(-a), rad(b), rad(b * 0.9)));
                assert_eq!(k, saturn_ring_factor(rad(a), rad(b * 0.9), rad(b)));
            }
        }
        assert!(saturn_ring_factor(f64::NAN, 0.3, 0.3).is_nan());
        assert!(saturn_ring_factor(0.1, f64::NAN, 0.3).is_nan());
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
