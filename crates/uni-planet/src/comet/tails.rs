//! The dust and ion tails (Task C1).
//!
//! - **Dust** (Task C1L): `uni_physics::dust::finson_probstein` (Finson & Probstein 1968, ApJ 154,
//!   327) on `CometView::conic` (J2000 ecliptic, as it expects; lead ruling A.1: never rotate it
//!   to ICRS first), β ∈ [0.001, 0.9] in 16 log steps × ages up to 60 d in 64 steps = 1 024 grains
//!   ([`DUST_PARAMS`]), dust production ∝ r_h^(−K1/2.5) at emission (the comet's magnitude law).
//!   - *Light.* Each grain stands for the dust emitted in its cell of ln β × age. The dust's light
//!     production now is the coma's light divided by [`DUST_HEAD_TIME_S`], a time calibrated on
//!     C/2020 F3 (NEOWISE) from its measured dust production A(0)fρ (Aravind et al. 2025), its
//!     measured dust speed (Manzini et al. 2021) and the steady-outflow relation of A'Hearn et al.
//!     1984 (derivation on the constant). A grain carries that rate × its emission weight
//!     [`TailGrain::weight`] (production at emission relative to now × its size class's share ×
//!     the cell's age span) × the sunlight at its own heliocentric distance × a
//!     Henyey–Greenstein phase function with g = 0.6 ([`DUST_PHASE_G`]). The tail is therefore
//!     several times brighter than the coma just after perihelion (≈ 6 × for row 20's NEOWISE)
//!     and fainter on the way in; it depends on the viewing phase as the phase function does.
//!     The v0 ratio "dust = 0.5 × coma" is gone (C1L: it made NEOWISE's tail short).
//!   - *Shape.* The grains are drawn as elliptical Gaussian splats (Zwicker et al. 2001, "EWA
//!     Splatting", IEEE TVCG 8, 223), whose 3D covariance spans half a grid step toward each
//!     grid neighbour (so the syndynes and synchrones join into one smooth curved sheet) plus the
//!     spread of the grains' ejection speed, 1.11 km/s (Manzini et al. 2021) × age, which blurs
//!     the sheet as a real dust tail is blurred. Where that blur is wider than the grid, 2 × 2
//!     blocks of grains are merged into one splat with the same light and second moments
//!     ([`tail_grains`]): fewer, equally smooth splats, so less fill.
//! - **Ion:** straight along the aberrated anti-sunward direction û = normalize(400 km/s · r̂_h −
//!   v_comet) (Brandt 1968, in *Introduction to the Solar Wind*; ruling 8), 0.3 AU / r_h long (at
//!   most 1 AU), falling off exponentially along its length, a Gaussian 10⁵ km wide (FWHM)
//!   across it, CO⁺ blue (427 nm chromaticity, gamut-mapped, then desaturated 50 % toward white),
//!   0.2 × the coma's light (v0). Drawn as a ribbon that turns about its axis to face the camera,
//!   with the surface brightness of an optically thin cylinder ([`ion_tail_brightness`]); from
//!   afar its σ is widened to at least [`ION_MIN_SIGMA_PX`] with the light across it kept
//!   ([`ion_tail_drawn_brightness`], review C1 I2), so it never aliases into a beaded line.
//! - **Which comets, and how often** (ruling round 2, review I13): only the
//!   [`MAX_TAILED_COMETS`] most prominent comets on screen (brightest `total_v` from the camera)
//!   get tails ([`tailed_comets`]). A tail is recomputed when the epoch has moved more than
//!   [`TAIL_SIM_REFRESH_S`] of sim time since its last refresh but never more often than every
//!   [`TAIL_REFRESH_WALL_S`] of wall time ([`tail_schedule`]), and the grain work is spread over
//!   frames in chunks of whole β rows within ≤ [`TAIL_CPU_SHARE_MS`] of CPU per frame
//!   ([`TailManager`]). Refresh times are kept by `ObjectId`, because `Snapshot::comets` changes
//!   order and membership from frame to frame (ruling C.8).
//! - Only comets with r_h < 5 AU (`uni_sim::snapshot::COMET_ACTIVE_AU`) get tails, and only
//!   those with SBDB's M1 and K1 (no magnitude law, no light).
//!
//! Neither tail dims the sky behind it (optically thin), so neither writes `lens_mask`.

use std::collections::HashMap;

use glam::{DMat3, DVec3, Vec3};
use uni_core::{AU, Epoch};
use uni_physics::dust::{DustGrain, DustParams, finson_probstein};
use uni_physics::kepler::Conic;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{COMET_ACTIVE_AU, CometView};

use super::{MAX_TAILED_COMETS, TAIL_REFRESH_WALL_S, line_colour};

/// Solar-wind speed [m/s] for the ion tail's aberration (ruling 8; Brandt 1968).
pub const SOLAR_WIND_MPS: f64 = 400.0e3;
/// Ion-tail length at r_h = 1 AU [m]; it scales as 1/r_h (plan C1).
pub const ION_TAIL_LENGTH_AT_1AU_M: f64 = 0.3 * AU;
/// Longest ion tail drawn [m] (plan C1).
pub const ION_TAIL_MAX_LENGTH_M: f64 = AU;
/// Ion-tail width [m], the FWHM of its Gaussian cross-section (plan C1: 10⁵ km).
pub const ION_TAIL_WIDTH_M: f64 = 1.0e8;
/// The ion tail's light as a fraction of the coma's (plan C1, v0).
pub const ION_TAIL_LIGHT: f64 = 0.2;
/// e-foldings of the ion tail's exponential fall-off over its length (a choice: the brightness
/// at the tip is e^(−3) = 5 % of the base's before the profile is brought to 0 there).
pub const ION_TAIL_EFOLDS: f64 = 3.0;
/// Segments of the ion-tail ribbon (twin of `ION_SEGMENTS` in `comet.wgsl`).
pub const ION_SEGMENTS: u32 = 32;
/// The ribbon reaches this many σ of the cross-section on each side (light beyond: 5 × 10⁻⁴).
pub const ION_HALF_WIDTH_SIGMAS: f64 = 3.5;
/// The ion ribbon's smallest on-screen σ [px] (review C1 I2; the dust sprites' `SPRITE_MIN_PX`
/// in `comet.wgsl` is the same 0.75 px). From afar the 10⁵ km tail is narrower than a pixel (FWHM
/// under 1 px beyond ≈ 0.63 AU at 1080 lines and 60°), so the ribbon is drawn with
/// σ = max(σ_phys, 0.75 px) and a Gaussian renormalised to that σ: the light across the tail is
/// kept, only spread over more pixels. A Gaussian of σ ≥ 0.75 px sampled at pixel centres sums to
/// its integral within 2·e^(−2π²σ²) ≈ 3 × 10⁻⁵ (Poisson summation), so the line neither beads nor
/// breaks into dashes as it moves. Twin of `ION_MIN_SIGMA_PX` in `comet.wgsl`.
pub const ION_MIN_SIGMA_PX: f64 = 0.75;

/// The dust's "head time" τ [s]: the coma's luminous intensity divided by the light the dust
/// emitted per second at the current production carries per unit of the phase function
/// [`dust_phase`] (C1L). The light of a grain is (I_coma/τ) × w × (r_h/r)² × HG(θ)
/// ([`dust_grain_light`]); the drawn tail carries I_coma × Σw (r_h/r)² HG/τ.
///
/// Calibrated on C/2020 F3 (NEOWISE) on 2020-07-22, the first night of Aravind et al. 2025's
/// photometry (r_h 0.63 AU, Δ 0.69 AU, their Table; Earth–Sun 1.0160 AU, so phase α = 100.56°):
/// - the dust's light production is v·A(α)fρ·E☉/(2π): a steady 1/ρ dust coma flowing out at v
///   holds the light emitted over the last πρ/(2v) inside the aperture ρ, and its light there is
///   A(α)fρ·ρ·E☉/4 (A'Hearn et al. 1984, AJ 89, 579: the definition of Afρ);
/// - A(0)fρ = 34.99 m ([`NEOWISE_AFRHO0_M`]: 3 499 ± 7 cm, green continuum 526 nm, 10⁴ km,
///   phase-corrected to 0°; Aravind et al. 2025, arXiv:2507.21707, Table "Afρ"), back to the
///   observed phase with the Halley–Marcus composite they corrected with (Schleicher & Bair
///   2011, AJ 141, 177; Marcus 2007, ICQ 29, 39; table as in sbpy `phase_HalleyMarcus`):
///   Φ(100.56°) = 0.7103 ([`neowise_calibration`]);
/// - v = 1.11 km/s ([`DUST_EJECTION_SPEED_MPS`]);
/// - E☉ = 3.178 × 10⁵ lx at 0.63 AU; I_coma = 1.441 × 10¹⁵ cd from the comet's visual law
///   (`uni_sim::world::COMET_VISUAL_LAWS`: V 3.18 there, extrapolated along its K1 = 9.63, which
///   is Aravind et al.'s own slope over 0.63–1.61 AU);
/// - HG(0.6) at the scattering angle 180° − α = 0.5257.
///
/// τ = I_coma·HG/(v·A(α)fρ·E☉/2π) = 5.42 × 10⁵ s = 6.3 d; `neowise_head_time_is_derived`
/// recomputes it. It is one comet's value applied to all (a gas-rich comet's tail is drawn too
/// bright, a dustier one's too faint) and rests on one dust speed for all grain sizes; within a
/// factor of about 2 (card note `uni_ui::theme::NOTE_TAILS`: brightness approximate). The v0
/// ratio (tail = 0.5 × coma) it replaces drew NEOWISE's tail 12 × too faint.
pub const DUST_HEAD_TIME_S: f64 = 5.421e5;
/// NEOWISE's A(0)fρ on 2020-07-22 [m] (green continuum, ρ = 10⁴ km; Aravind et al. 2025).
pub const NEOWISE_AFRHO0_M: f64 = 34.99;
/// Dust ejection speed [m/s]: the mean deprojected expansion speed of C/2020 F3 (NEOWISE)'s
/// dust shells, 1.11 ± 0.08 km/s (Manzini et al. 2021, MNRAS, arXiv:2109.02039). It calibrates
/// [`DUST_HEAD_TIME_S`] and blurs the Finson–Probstein sheet: a grain emitted isotropically at
/// v has moved v·t from its zero-velocity position after an age t, a spread of v·t/√3 per axis
/// ([`dust_ejection_sigma_m`]). One speed for every size and r_h (approximate).
pub const DUST_EJECTION_SPEED_MPS: f64 = 1_110.0;
/// Henyey–Greenstein asymmetry of the dust's phase function (plan C1). Checked 2026-11-23: the
/// plan cites Marcus 2007 as "Icarus 191"; the paper is Marcus 2007, ICQ 29, 39, and its model is
/// a compound HG with a strong forward lobe (k = 0.95, as Li & Jewitt 2015, arXiv:1503.00387,
/// quote it; they fit g_f = 0.926, g_b = −0.584 for C/2010 X1). The single g = 0.6 lobe is the
/// plan's simplification: it under-predicts the forward-scattering surge near phase 180°.
pub const DUST_PHASE_G: f64 = 0.6;
/// The Halley–Marcus composite dust phase function Φ(α)/Φ(0) at α = 100° and 101° (Schleicher &
/// Bair 2011, AJ 141, 177; Marcus 2007, ICQ 29, 39; the 1° table as sbpy's
/// `phase_HalleyMarcus` holds it, read 2026-11-24): the curve Aravind et al. 2025 corrected
/// A(α)fρ to 0° with.
pub const HALLEY_MARCUS_100_101: [f64; 2] = [0.6987, 0.7196];
/// The NEOWISE calibration night (Aravind et al. 2025, Table "Afρ", 2020-07-22): r_h, Δ and the
/// Earth–Sun distance [AU] (the last from the Earth's orbit on that date).
pub const NEOWISE_CALIBRATION_AU: [f64; 3] = [0.63, 0.69, 1.0160];
/// C/2020 F3 (NEOWISE)'s SBDB SPK-ID, whose visual law calibrates [`DUST_HEAD_TIME_S`].
pub const NEOWISE_SPKID: i32 = 1_003_667;

/// [`DUST_HEAD_TIME_S`] from its inputs (its doc): returns (τ [s], the phase angle α [°] of the
/// calibration night, Φ_HM(α)). NaN when uni-sim has no visual law for NEOWISE.
pub fn neowise_calibration() -> (f64, f64, f64) {
    let [r_au, delta_au, earth_au] = NEOWISE_CALIBRATION_AU;
    // The Sun–comet–Earth angle from the triangle's three sides.
    let cos_alpha =
        (r_au * r_au + delta_au * delta_au - earth_au * earth_au) / (2.0 * r_au * delta_au);
    let alpha = cos_alpha.clamp(-1.0, 1.0).acos().to_degrees();
    let [p100, p101] = HALLEY_MARCUS_100_101;
    let phi = p100 + (p101 - p100) * (alpha - 100.0);
    let Some(law) = uni_sim::world::comet_visual_law(NEOWISE_SPKID) else {
        return (f64::NAN, alpha, phi);
    };
    let head_cd = super::coma::comet_total_intensity_cd(law.m1(), law.k1, r_au, delta_au);
    let r_m = r_au * AU;
    let sun_lux = super::sun_intensity_cd() / (r_m * r_m);
    // A'Hearn et al. 1984: the dust's light production v·A(α)fρ·E☉/(2π) [cd/s].
    let rate = DUST_EJECTION_SPEED_MPS * NEOWISE_AFRHO0_M * phi * sun_lux / std::f64::consts::TAU;
    // The scattering angle is 180° − α: cos θ = −cos α.
    let tau = head_cd * dust_phase(-cos_alpha) / rate;
    (tau, alpha, phi)
}

/// The Finson–Probstein grid: β ∈ [0.001, 0.9], 16 log steps (C1L ladder step 1: 32 → 16 rows,
/// 2 048 → 1 024 grains); ages to 60 d, 64 steps (plan C1). β ≤ 0.9 keeps NEOWISE's main dust
/// band, which Afghan et al. 2024 (PSJ 5, 264) bound at the β = 0.82 syndyne.
pub const DUST_PARAMS: DustParams = DustParams {
    beta_min: 0.001,
    beta_max: 0.9,
    n_beta: 16,
    max_age_days: 60.0,
    n_age: 64,
};
/// Grains (splat slots) per tail: 16 × 64. Merging leaves at most this many splats.
pub const GRAINS_PER_TAIL: usize = 1_024;
/// A grain's splat spans this many grid steps (σ) toward its β and age neighbours (a choice: a
/// Gaussian of σ = 0.5 step sampled every step sums to a flat sheet within 2·e^(−2π²/4) = 1.4 %,
/// Poisson summation).
pub const SPLAT_GRID_SIGMA: f64 = 0.5;
/// A 2 × 2 block of splats merges into one when every member lies within this many σ of the
/// block's ejection spread from their light-weighted mean ([`tail_grains`]; a choice: the
/// merged Gaussian keeps the block's light, mean and second moments).
pub const SPLAT_MERGE_SPREAD: f64 = 2.0;
/// Sprites are cut at this many σ (the profile is shifted to reach 0 there and renormalised;
/// C1L: 3 → 2, overlapping splats hide the cut, and the quads cover 4/9 of the area).
pub const SPRITE_SIGMAS: f64 = 2.0;
/// A tail is recomputed once the epoch has moved this far [s of sim time] from its last
/// refresh (plan C1: 1 h).
pub const TAIL_SIM_REFRESH_S: f64 = 3_600.0;
/// CPU share of the tail work per frame [ms] (plan C1, performance budget).
pub const TAIL_CPU_SHARE_MS: f64 = 1.0;
/// The plan's cost of one Kepler solve [µs] (review I13: "≈ 0.5 µs each"), the nominal rate the
/// schedule is tested at; the renderer measures the real rate.
pub const KEPLER_SOLVE_US: f64 = 0.5;
/// Kepler solves the nominal per-frame budget allows: 1 ms / 0.5 µs.
pub const NOMINAL_SOLVES_PER_FRAME: usize = 2_000;

/// CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at 427 nm, the CO⁺ comet-tail band (CIE 2018
/// data table, DOI 10.25039/CIE.DS.xvudnb9b; read 2026-11-23).
pub const CO_PLUS_CMF: [f64; 3] = [0.244_879_7, 0.008_908_72, 1.188_497_1];

/// Direction of the ion tail (unit, ICRS) from the Sun → nucleus geometry: `sun_rel` is the Sun
/// relative to the nucleus [m] (`CometView::sun_rel_m`), `helio_vel` the comet's heliocentric
/// velocity [m/s]. Anti-sunward, aberrated by the 400 km/s solar wind (module docs). Zero when
/// `sun_rel` is zero.
pub fn ion_tail_dir(sun_rel: DVec3, helio_vel: DVec3) -> DVec3 {
    let anti_sun = (-sun_rel).normalize_or_zero();
    if anti_sun == DVec3::ZERO {
        return DVec3::ZERO;
    }
    // The solar wind as the comet sees it: its radial velocity minus the comet's own.
    (SOLAR_WIND_MPS * anti_sun - helio_vel).normalize_or_zero()
}

/// The ion tail's length [m] at heliocentric distance `r_au`: 0.3 AU / r_h, at most 1 AU.
pub fn ion_tail_length_m(r_au: f64) -> f64 {
    (ION_TAIL_LENGTH_AT_1AU_M / r_au.max(0.0)).min(ION_TAIL_MAX_LENGTH_M)
}

/// The ion tail's surface brightness [cd/m²] on its ribbon at distance `s_m` along the axis from
/// the nucleus and `x_m` across it, for a tail of `length_m` carrying `intensity_cd`, seen at
/// angle ψ to its axis (`sin_psi`). The light per unit length falls as e^(−s/λ) − e^(−L/λ),
/// λ = L/[`ION_TAIL_EFOLDS`] (0 at the tip), the cross-section is a Gaussian of FWHM
/// [`ION_TAIL_WIDTH_M`], and the ribbon's brightness is divided by sin ψ (at least w/L) because an
/// optically thin cylinder's projected light is foreshortened into a shorter streak. Over the
/// ribbon ∫S sin ψ ds dx = I. CPU twin of the shader's; probed on the GPU.
pub fn ion_tail_brightness(
    s_m: f64,
    x_m: f64,
    length_m: f64,
    intensity_cd: f64,
    sin_psi: f64,
) -> f64 {
    ion_tail_drawn_brightness(s_m, x_m, length_m, intensity_cd, sin_psi, ion_sigma_m())
}

/// The ion tail as the ribbon draws it: [`ion_tail_brightness`] with the cross-section's σ
/// widened to `sigma_m` (≥ the physical σ: [`ion_drawn_sigma_m`]) and the Gaussian renormalised
/// to it, so the light across the tail, and ∫S sin ψ ds dx = I, are unchanged (review C1 I2).
/// The sin ψ cap keeps the physical width (the axial column is a property of the tail, not of
/// the pixel). 0 for σ ≤ 0. CPU twin of the shader's; probed on the GPU.
pub fn ion_tail_drawn_brightness(
    s_m: f64,
    x_m: f64,
    length_m: f64,
    intensity_cd: f64,
    sin_psi: f64,
    sigma_m: f64,
) -> f64 {
    if !(s_m >= 0.0 && s_m < length_m && sigma_m > 0.0) {
        return 0.0;
    }
    let lambda = length_m / ION_TAIL_EFOLDS;
    let tip = (-ION_TAIL_EFOLDS).exp();
    // ∫₀^L (e^(−s/λ) − e^(−L/λ)) ds.
    let norm = lambda * (1.0 - tip) - length_m * tip;
    let along = ((-s_m / lambda).exp() - tip) / norm;
    let t = x_m / sigma_m;
    let across = (-0.5 * t * t).exp() / (std::f64::consts::TAU.sqrt() * sigma_m);
    let sin_eff = sin_psi.max(ION_TAIL_WIDTH_M / length_m);
    intensity_cd * along * across / sin_eff
}

/// σ of the ion tail's Gaussian cross-section [m]: FWHM / (2√(2 ln 2)).
pub(crate) fn ion_sigma_m() -> f64 {
    ION_TAIL_WIDTH_M / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt())
}

/// The σ [m] the ion ribbon is drawn with where one pixel spans `pixel_m` metres (at the
/// ribbon's distance): the physical σ, but at least [`ION_MIN_SIGMA_PX`] pixels. CPU twin of the
/// shader's `ion_drawn_sigma`; probed on the GPU.
pub fn ion_drawn_sigma_m(pixel_m: f64) -> f64 {
    ion_sigma_m().max(ION_MIN_SIGMA_PX * pixel_m)
}

/// The ion tail's colour, linear Rec. 709 with luminance 1 (module docs).
pub fn ion_colour() -> Vec3 {
    0.5 * line_colour(CO_PLUS_CMF) + 0.5 * Vec3::ONE
}

/// Henyey & Greenstein 1941 (ApJ 93, 70) phase function with g = [`DUST_PHASE_G`], normalised
/// to a mean of 1 over the sphere, at scattering-angle cosine `cos_theta` (1 = forward). CPU twin
/// of the shader's; probed on the GPU.
pub fn dust_phase(cos_theta: f64) -> f64 {
    let g = DUST_PHASE_G;
    let denom = 1.0 + g * g - 2.0 * g * cos_theta.clamp(-1.0, 1.0);
    (1.0 - g * g) / (denom * denom.sqrt())
}

/// A grain's light per unit of the tail's emission rate [s] (C1L): its emission `weight` [s]
/// ([`TailGrain::weight`]) × the sunlight at the grain relative to the nucleus's, (r_h/r)², ×
/// HG(θ) ([`dust_phase`]), θ the scattering angle between the Sun → grain and grain → camera
/// directions (0 = forward). `offset` is the grain's offset from the nucleus, `nucleus_rel` the
/// nucleus relative to the camera and `sun_rel` the Sun relative to the nucleus [m]. 0 with the
/// camera within 1 m of the grain (the pass draws nothing there). × [`dust_emission_cd_per_s`]
/// (`TailGpu::sun.w`) it is the grain's luminous intensity [cd]. CPU twin of the shader's
/// `dust_grain_light`; probed on the GPU.
pub fn dust_grain_light(offset: DVec3, weight: f64, nucleus_rel: DVec3, sun_rel: DVec3) -> f64 {
    let rel = nucleus_rel + offset;
    let dist = rel.length();
    let helio = offset - sun_rel;
    let helio2 = helio.length_squared();
    if !(dist > 1.0 && helio2 > 0.0) {
        return 0.0;
    }
    // E☉ ∝ 1/r²: the grain's sunlight relative to the nucleus's.
    let lit = sun_rel.length_squared() / helio2;
    let cos_theta = helio.normalize().dot(-rel / dist);
    weight * lit * dust_phase(cos_theta)
}

/// The dust's light production [cd/s per unit of HG] for a coma of luminous intensity
/// `coma_cd` [cd]: I_coma/[`DUST_HEAD_TIME_S`] (C1L; `TailGpu::sun.w`). 0 for a non-positive or
/// non-finite coma.
pub fn dust_emission_cd_per_s(coma_cd: f64) -> f64 {
    if coma_cd > 0.0 && coma_cd.is_finite() {
        coma_cd / DUST_HEAD_TIME_S
    } else {
        0.0
    }
}

/// The spread [m, σ per axis] of grains emitted isotropically at [`DUST_EJECTION_SPEED_MPS`]
/// after `age_s`: v·t/√3 (the variance of a uniform direction is 1/3 per axis).
pub fn dust_ejection_sigma_m(age_s: f64) -> f64 {
    DUST_EJECTION_SPEED_MPS * age_s.max(0.0) / 3.0_f64.sqrt()
}

/// A lower-triangular factor L of a splat's covariance (Σ = L Lᵀ, Cholesky), what the GPU
/// record carries (C1L): projecting L instead of Σ keeps the ellipse's minor axis exact in f32
/// ([`splat_axes`]). A jitter of 10⁻¹² of the trace keeps a flat (singular) Σ factorable.
pub fn cov_factor(cov: DMat3) -> DMat3 {
    let eps = 1e-12 * (cov.x_axis.x + cov.y_axis.y + cov.z_axis.z).abs() + f64::MIN_POSITIVE;
    let a = |i: usize, j: usize| cov.col(j)[i] + if i == j { eps } else { 0.0 };
    let l00 = a(0, 0).max(eps).sqrt();
    let l10 = a(1, 0) / l00;
    let l20 = a(2, 0) / l00;
    let l11 = (a(1, 1) - l10 * l10).max(eps).sqrt();
    let l21 = (a(2, 1) - l20 * l10) / l11;
    let l22 = (a(2, 2) - l20 * l20 - l21 * l21).max(eps).sqrt();
    DMat3::from_cols(
        DVec3::new(l00, l10, l20),
        DVec3::new(0.0, l11, l21),
        DVec3::new(0.0, 0.0, l22),
    )
}

/// A splat's screen-plane ellipse (C1L): the 3D covariance L Lᵀ (`factor` = L, [`cov_factor`])
/// projected onto the plane spanned by the unit vectors `right` and `up` (the sprite's plane,
/// facing the camera), plus `floor_m`² on both axes (the smallest σ drawn, [`SPRITE_MIN_PX`]
/// pixels). With uᵢ = (right·Lᵢ, up·Lᵢ) the columns projected, the 2D covariance C has trace
/// Σ|uᵢ|² + 2f² and determinant Σ_{i<j}(uᵢ × uⱼ)² + f²Σ|uᵢ|² + f⁴ (Cauchy–Binet: a sum of
/// squares, so an elongated splat's minor axis does not cancel in f32); σ₁² = m + √(m² − det),
/// m = trace/2, σ₂² = det/σ₁². Returns (σ₁, σ₂, φ) [m, m, rad]: σ₁ ≥ σ₂ along the axes
/// cos φ·right + sin φ·up and −sin φ·right + cos φ·up, φ = ½ atan2(2C₁₂, C₁₁ − C₂₂). CPU twin of
/// the shader's `splat_axes` (same order of operations); probed on the GPU.
pub fn splat_axes(factor: DMat3, right: DVec3, up: DVec3, floor_m: f64) -> (f64, f64, f64) {
    let f2 = floor_m * floor_m;
    let u = [factor.x_axis, factor.y_axis, factor.z_axis].map(|c| (right.dot(c), up.dot(c)));
    let p = u[0].0 * u[0].0 + u[1].0 * u[1].0 + u[2].0 * u[2].0;
    let q = u[0].0 * u[0].1 + u[1].0 * u[1].1 + u[2].0 * u[2].1;
    let r = u[0].1 * u[0].1 + u[1].1 * u[1].1 + u[2].1 * u[2].1;
    let cross = |a: (f64, f64), b: (f64, f64)| a.0 * b.1 - a.1 * b.0;
    let (c01, c02, c12) = (cross(u[0], u[1]), cross(u[0], u[2]), cross(u[1], u[2]));
    let det = c01 * c01 + c02 * c02 + c12 * c12 + f2 * (p + r) + f2 * f2;
    let m = 0.5 * (p + r) + f2;
    let l1 = m + (m * m - det).max(0.0).sqrt();
    let l2 = det / l1;
    (l1.sqrt(), l2.sqrt(), 0.5 * (2.0 * q).atan2(p - r))
}

/// The smallest splat σ drawn [px] (twin of `SPRITE_MIN_PX` in `comet.wgsl`): a sub-pixel splat
/// neither aliases nor vanishes; its light is kept (C1 review I2's floor).
pub const SPRITE_MIN_PX: f64 = 0.75;

/// A grain sprite's normalised profile [1/m²] at distance `r` from its centre for width `sigma`:
/// a Gaussian cut at [`SPRITE_SIGMAS`]·σ, shifted to reach 0 there and renormalised so its
/// integral over the plane is 1. CPU twin of the shader's; probed on the GPU.
pub fn sprite_profile(r: f64, sigma: f64) -> f64 {
    if !(sigma > 0.0 && r.abs() < SPRITE_SIGMAS * sigma) {
        return 0.0;
    }
    let cut2 = SPRITE_SIGMAS * SPRITE_SIGMAS;
    let edge = (-0.5 * cut2).exp();
    // ∫ over the disk r < cut·σ of (e^(−r²/2σ²) − edge) dA = 2πσ²[1 − edge·(1 + cut²/2)].
    let norm = std::f64::consts::TAU * sigma * sigma * (1.0 - edge * (1.0 + 0.5 * cut2));
    let t = r / sigma;
    ((-0.5 * t * t).exp() - edge) / norm
}

/// Whether `view` can carry a coma and tails: active (r_h < 5 AU) with SBDB's M1 and K1.
pub(crate) fn is_active(view: &CometView) -> bool {
    view.r_au < COMET_ACTIVE_AU
        && view.m1.is_some()
        && view.k1.is_some()
        && view.total_v.is_some_and(f64::is_finite)
}

/// The comets that draw tails: indices into `comets` of the at most [`MAX_TAILED_COMETS`]
/// brightest (`total_v`, then id) active ones, brightest first. The caller passes only comets
/// whose bounds are on screen.
pub fn tailed_comets(comets: &[CometView]) -> Vec<usize> {
    let mut picked: Vec<usize> = (0..comets.len())
        .filter(|&i| is_active(&comets[i]))
        .collect();
    picked.sort_by(|&a, &b| {
        let v = |i: usize| comets[i].total_v.unwrap_or(f64::INFINITY);
        v(a).total_cmp(&v(b)).then(comets[a].id.cmp(&comets[b].id))
    });
    picked.truncate(MAX_TAILED_COMETS);
    picked
}

/// Which of `comets` (indices into it) get a tail refresh this frame at wall time `wall_s` [s],
/// given each comet's last refresh `last_refresh` [wall s] by id (module docs): the
/// [`tailed_comets`] whose last refresh is at least [`TAIL_REFRESH_WALL_S`] ago (or that have
/// none), never-refreshed first, then the longest-waiting. The renderer also skips those whose
/// tail is less than [`TAIL_SIM_REFRESH_S`] of sim time old, and works down the list within its
/// per-frame CPU share ([`TailManager`]).
pub fn tail_schedule(
    comets: &[CometView],
    last_refresh: &HashMap<ObjectId, f64>,
    wall_s: f64,
) -> Vec<usize> {
    let mut due: Vec<(usize, f64)> = tailed_comets(comets)
        .into_iter()
        .filter_map(|i| match last_refresh.get(&comets[i].id) {
            None => Some((i, f64::NEG_INFINITY)),
            Some(&t) if wall_s - t >= TAIL_REFRESH_WALL_S => Some((i, t)),
            Some(_) => None,
        })
        .collect();
    // Stable: equally old refreshes keep the brightness order.
    due.sort_by(|a, b| a.1.total_cmp(&b.1));
    due.into_iter().map(|(i, _)| i).collect()
}

/// The grains of β rows `rows` (indices 0..32 into [`DUST_PARAMS`]'s β grid) of `conic` at
/// `t_obs`, with dust production ∝ r_h^(−K1/2.5) (module docs). Row-major as `finson_probstein`
/// (β-major, ages within a row).
pub(crate) fn dust_rows(
    conic: &Conic,
    k1: f64,
    t_obs: Epoch,
    rows: std::ops::Range<u32>,
) -> Vec<DustGrain> {
    let n = DUST_PARAMS.n_beta;
    let (j0, j1) = (rows.start.min(n), rows.end.min(n));
    if j0 >= j1 {
        return Vec::new();
    }
    // The j-th β of the full grid, as `DustParams` spaces it (log-spaced, endpoints exact); a
    // sub-grid from β(j0) to β(j1 − 1) in j1 − j0 steps has the same ratio between rows.
    let beta = |j: u32| {
        let p = DUST_PARAMS;
        if j == 0 {
            p.beta_min
        } else if j + 1 == n {
            p.beta_max
        } else {
            p.beta_min * (p.beta_max / p.beta_min).powf(f64::from(j) / f64::from(n - 1))
        }
    };
    let p = DustParams {
        beta_min: beta(j0),
        beta_max: beta(j1 - 1),
        n_beta: j1 - j0,
        ..DUST_PARAMS
    };
    // Production ∝ r_h^(−K1/2.5): SBDB's total-magnitude law m = M1 + 5 log Δ + K1 log r_h
    // (plan C1).
    finson_probstein(conic, t_obs, &p, &|r_h| dust_production(r_h, k1))
}

/// One dust splat as the pass draws it (C1L): a grain of the Finson–Probstein grid, or a merged
/// 2 × 2 block of them ([`tail_grains`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TailGrain {
    /// Offset from the nucleus at the refresh epoch [m], ICRS (a merged block's light-weighted
    /// mean).
    pub(crate) offset_m: DVec3,
    /// The splat's 3D covariance [m²] (symmetric): half a grid step toward its neighbours, the
    /// ejection spread, and for a merged block its members' spread about the mean.
    pub(crate) cov_m2: DMat3,
    /// Emission weight [s]: the dust emitted in the grain's cell, as seconds of the current
    /// production, Σ over the grid of P(t_e)/P_now × (√β / Σ√β) × Δt (module docs). The pass
    /// draws [`dust_grain_light`] (weight × (r_h/r)² × HG) × [`dust_emission_cd_per_s`].
    pub(crate) weight: f64,
}

/// A computed dust tail.
#[derive(Clone, Debug)]
pub(crate) struct TailState {
    /// The epoch the grains were computed for.
    pub(crate) epoch: Epoch,
    pub(crate) grains: Vec<TailGrain>,
    /// Largest |offset| + [`SPRITE_SIGMAS`]·σ_max [m]: the tail's reach from the nucleus.
    pub(crate) reach_m: f64,
    /// Bumped at every completed refresh (the renderer re-uploads on a change).
    pub(crate) version: u64,
    /// Wall time [s] the comet last drew a tail (eviction).
    pub(crate) last_seen_wall_s: f64,
}

/// v vᵀ.
fn outer(v: DVec3) -> DMat3 {
    DMat3::from_cols(v * v.x, v * v.y, v * v.z)
}

/// A splat while blocks are merged: its [`TailGrain`] and the smallest ejection spread [m] of
/// the grains in it.
#[derive(Clone, Copy, Debug)]
struct Splat {
    grain: TailGrain,
    spread_m: f64,
}

/// The splats of a complete β × age grid of grains (`grains` in `finson_probstein`'s order, rows
/// of [`DUST_PARAMS`]`.n_age` ages) whose production law gives `production_now` at the refresh
/// epoch (module docs, C1L).
///
/// - **Weights:** a grain's `weight` (production at emission × √β) / (production now × Σ√β over
///   the β rows) × the age step [s]; they sum to the emission history's equivalent span
///   H = Σ_k P(t_e,k)/P_now·Δt (60 d at a constant production).
/// - **Covariance:** σ = [`SPLAT_GRID_SIGMA`] grid steps along the central differences to the β
///   and age neighbours (one-sided at the edges; the youngest grain's younger neighbour is the
///   nucleus), plus [`dust_ejection_sigma_m`] of its age on every axis.
/// - **Merging:** level by level, a 2 × 2 block of splats (each present) becomes one when every
///   member lies within [`SPLAT_MERGE_SPREAD`] × the block's smallest ejection spread of the
///   light-weighted mean: weight = Σ, offset = the weighted mean, covariance = the weighted mean
///   of (member covariance + its offset from the mean, outer product), so the light and its
///   first and second moments are kept.
pub(crate) fn tail_grains(grains: &[DustGrain], production_now: f64) -> Vec<TailGrain> {
    tail_grains_merged_within(grains, production_now, SPLAT_MERGE_SPREAD)
}

/// [`tail_grains`] with the merge threshold `merge_spread` (in ejection spreads; 0 keeps every
/// grain apart).
fn tail_grains_merged_within(
    grains: &[DustGrain],
    production_now: f64,
    merge_spread: f64,
) -> Vec<TailGrain> {
    let n_age = DUST_PARAMS.n_age as usize;
    let n_beta = grains.len() / n_age;
    if n_beta == 0 || !(production_now > 0.0 && production_now.is_finite()) {
        return Vec::new();
    }
    let at = |j: usize, k: usize| grains[j * n_age + k].pos_rel_nucleus_m;
    let sum_sqrt_beta: f64 = (0..n_beta).map(|j| grains[j * n_age].beta.sqrt()).sum();
    let dt = DUST_PARAMS.max_age_days * uni_core::DAY / DUST_PARAMS.n_age as f64;
    let norm = dt / (production_now * sum_sqrt_beta);
    let s2 = SPLAT_GRID_SIGMA * SPLAT_GRID_SIGMA;
    let mut level: Vec<Option<Splat>> = Vec::with_capacity(n_beta * n_age);
    for j in 0..n_beta {
        for k in 0..n_age {
            let g = &grains[j * n_age + k];
            let x = g.pos_rel_nucleus_m;
            let along_beta = match (j.checked_sub(1), j + 1 < n_beta) {
                (Some(jm), true) => 0.5 * (at(j + 1, k) - at(jm, k)),
                (Some(jm), false) => x - at(jm, k),
                (None, true) => at(j + 1, k) - x,
                (None, false) => DVec3::ZERO,
            };
            let younger = k.checked_sub(1).map_or(DVec3::ZERO, |km| at(j, km));
            let along_age = if k + 1 < n_age {
                0.5 * (at(j, k + 1) - younger)
            } else {
                x - younger
            };
            let spread = dust_ejection_sigma_m(g.age_s);
            let cov = (outer(along_beta) + outer(along_age)) * s2
                + DMat3::from_diagonal(DVec3::splat(spread * spread));
            let weight = g.weight * norm;
            level.push((weight.is_finite() && weight >= 0.0).then_some(Splat {
                grain: TailGrain {
                    offset_m: x,
                    cov_m2: cov,
                    weight,
                },
                spread_m: spread,
            }));
        }
    }
    let mut out = Vec::with_capacity(n_beta * n_age);
    let (mut rows, mut cols) = (n_beta, n_age);
    while rows >= 2 && cols >= 2 {
        let (rows2, cols2) = (rows / 2, cols / 2);
        let mut next: Vec<Option<Splat>> = vec![None; rows2 * cols2];
        for j in 0..rows2 {
            for k in 0..cols2 {
                let members: Vec<Splat> = [(0, 0), (1, 0), (0, 1), (1, 1)]
                    .iter()
                    .filter_map(|&(dj, dk)| level[(2 * j + dj) * cols + 2 * k + dk])
                    .collect();
                if members.len() == 4 {
                    next[j * cols2 + k] = merge(&members, merge_spread);
                }
            }
        }
        // The members of blocks that stayed apart are final.
        for j in 0..rows {
            for k in 0..cols {
                let merged =
                    j / 2 < rows2 && k / 2 < cols2 && next[(j / 2) * cols2 + k / 2].is_some();
                if let (Some(splat), false) = (level[j * cols + k], merged) {
                    out.push(splat.grain);
                }
            }
        }
        level = next;
        (rows, cols) = (rows2, cols2);
    }
    out.extend(level.into_iter().flatten().map(|s| s.grain));
    out
}

/// One splat for `members` when they lie within `merge_spread` × their smallest ejection
/// spread of their light-weighted mean (moments kept, [`tail_grains`]); `None` otherwise.
fn merge(members: &[Splat], merge_spread: f64) -> Option<Splat> {
    let weight: f64 = members.iter().map(|m| m.grain.weight).sum();
    if !(weight > 0.0) {
        return None;
    }
    let mean = members
        .iter()
        .map(|m| m.grain.offset_m * m.grain.weight)
        .sum::<DVec3>()
        / weight;
    let spread = members
        .iter()
        .map(|m| m.spread_m)
        .fold(f64::INFINITY, f64::min);
    let far = members
        .iter()
        .map(|m| (m.grain.offset_m - mean).length())
        .fold(0.0, f64::max);
    if !(far <= merge_spread * spread) {
        return None;
    }
    let cov = members
        .iter()
        .map(|m| (m.grain.cov_m2 + outer(m.grain.offset_m - mean)) * (m.grain.weight / weight))
        .fold(DMat3::ZERO, |a, c| a + c);
    Some(Splat {
        grain: TailGrain {
            offset_m: mean,
            cov_m2: cov,
            weight,
        },
        spread_m: spread,
    })
}

/// The largest σ [m] of a splat's covariance: an upper bound on its largest eigenvalue's root
/// (the largest row sum of |cov|, Gershgorin).
pub(crate) fn sigma_bound_m(cov: DMat3) -> f64 {
    let c = cov.abs();
    let rows = [
        c.x_axis.x + c.y_axis.x + c.z_axis.x,
        c.x_axis.y + c.y_axis.y + c.z_axis.y,
        c.x_axis.z + c.y_axis.z + c.z_axis.z,
    ];
    rows.into_iter().fold(0.0, f64::max).sqrt()
}

/// The reach of a tail of `grains` from the nucleus [m]: its farthest sprite edge.
pub(crate) fn tail_reach_m(grains: &[TailGrain]) -> f64 {
    grains
        .iter()
        .map(|g| g.offset_m.length() + SPRITE_SIGMAS * sigma_bound_m(g.cov_m2))
        .fold(0.0, f64::max)
}

/// The production law's value at heliocentric distance `r_au` for slope `k1`: r_h^(−K1/2.5)
/// (the factor `dust_rows` weights the emissions with).
pub(crate) fn dust_production(r_au: f64, k1: f64) -> f64 {
    r_au.powf(-k1 / 2.5)
}

/// A conservative reach [m] of a dust tail not computed yet, for culling: 1.5 × the drift
/// ½·β·g☉(r_h)·t² of the lightest grain (β = 0.9) at the oldest age (60 d), at most 3 AU.
/// Radiation pressure is β times the Sun's gravity g☉ = GM☉/r² (Burns, Lamy & Soter 1979, Icarus
/// 40, 1); a grain that drifts outward feels less, so ½βg☉t² bounds the drift; the factor 1.5
/// covers the orbital shear the formula leaves out.
pub(crate) fn dust_reach_estimate_m(r_au: f64) -> f64 {
    let r = r_au.max(1e-3) * AU;
    let t = DUST_PARAMS.max_age_days * uni_core::DAY;
    let drift = 0.5 * DUST_PARAMS.beta_max * uni_core::GM_SUN / (r * r) * t * t;
    (1.5 * drift).min(3.0 * AU)
}

/// What the tail work did in one frame.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct TailWork {
    /// Kepler solves this frame (grains plus the nucleus's emission states).
    pub(crate) solves: usize,
    /// Comets whose refresh started this frame.
    pub(crate) started: Vec<ObjectId>,
    /// Comets whose refresh completed this frame.
    pub(crate) completed: Vec<ObjectId>,
}

/// A refresh in progress: the grid is filled row by row over frames, all for one epoch.
#[derive(Clone, Debug)]
struct TailJob {
    id: ObjectId,
    epoch: Epoch,
    conic: Conic,
    k1: f64,
    sun_rel: DVec3,
    grains: Vec<DustGrain>,
    rows_done: u32,
}

/// The dust tails' CPU side (no GPU objects, so it is tested on its own): per-comet tails keyed
/// by id, the schedule's refresh times, and the refresh in progress.
#[derive(Debug, Default)]
pub(crate) struct TailManager {
    pub(crate) tails: HashMap<ObjectId, TailState>,
    pub(crate) last_refresh: HashMap<ObjectId, f64>,
    job: Option<TailJob>,
    next_version: u64,
}

/// A tail not drawn for this long [wall s] is dropped (memory; it is recomputed on return).
pub(crate) const TAIL_EVICT_WALL_S: f64 = 30.0;

impl TailManager {
    /// One frame of tail work for `comets` (the comets whose bounds are on screen) at `epoch`
    /// and wall time `wall_s`, within `max_solves` Kepler solves (at least one β row is always
    /// done while a refresh is due, so the work never stalls).
    pub(crate) fn update(
        &mut self,
        comets: &[CometView],
        epoch: Epoch,
        wall_s: f64,
        max_solves: usize,
    ) -> TailWork {
        let mut work = TailWork::default();
        let tailed = tailed_comets(comets);
        let is_tailed = |id: ObjectId| tailed.iter().any(|&i| comets[i].id == id);
        for &i in &tailed {
            if let Some(t) = self.tails.get_mut(&comets[i].id) {
                t.last_seen_wall_s = wall_s;
            }
        }
        // A comet that no longer draws a tail loses its refresh in progress.
        if self.job.as_ref().is_some_and(|job| !is_tailed(job.id)) {
            self.job = None;
        }
        let n_age = DUST_PARAMS.n_age as usize;
        loop {
            if self.job.is_none() {
                // Starting a refresh only pays once a row fits (or nothing was done yet).
                if work.solves > 0 && max_solves.saturating_sub(work.solves) < 2 * n_age {
                    break;
                }
                let stale = |id: ObjectId| {
                    self.tails
                        .get(&id)
                        .is_none_or(|t| epoch.seconds_since(t.epoch).abs() > TAIL_SIM_REFRESH_S)
                };
                let next = tail_schedule(comets, &self.last_refresh, wall_s)
                    .into_iter()
                    .find(|&i| stale(comets[i].id));
                let Some((view, k1)) = next.and_then(|i| Some((&comets[i], comets[i].k1?))) else {
                    break;
                };
                self.last_refresh.insert(view.id, wall_s);
                work.started.push(view.id);
                self.job = Some(TailJob {
                    id: view.id,
                    epoch,
                    conic: view.conic,
                    k1,
                    sun_rel: view.sun_rel_m,
                    grains: Vec::with_capacity(GRAINS_PER_TAIL),
                    rows_done: 0,
                });
            }
            let Some(job) = self.job.as_mut() else {
                break;
            };
            // Each chunk solves the nucleus once per age, then one grain per β row and age.
            let rows_left = DUST_PARAMS.n_beta - job.rows_done;
            let fit = max_solves.saturating_sub(work.solves).saturating_sub(n_age) / n_age;
            let rows = match (fit, work.solves) {
                (0, 0) => 1,
                (0, _) => break,
                (fit, _) => u32::try_from(fit).unwrap_or(u32::MAX).min(rows_left),
            };
            let range = job.rows_done..job.rows_done + rows;
            job.grains
                .extend(dust_rows(&job.conic, job.k1, job.epoch, range));
            job.rows_done += rows;
            work.solves += (rows as usize + 1) * n_age;
            if let Some(done) = self.job.take_if(|j| j.rows_done >= DUST_PARAMS.n_beta) {
                let production_now = dust_production(done.sun_rel.length() / AU, done.k1);
                let grains = tail_grains(&done.grains, production_now);
                self.next_version += 1;
                self.tails.insert(
                    done.id,
                    TailState {
                        epoch: done.epoch,
                        reach_m: tail_reach_m(&grains),
                        grains,
                        version: self.next_version,
                        last_seen_wall_s: wall_s,
                    },
                );
                work.completed.push(done.id);
            }
        }
        self.tails
            .retain(|_, t| wall_s - t.last_seen_wall_s <= TAIL_EVICT_WALL_S);
        let tails = &self.tails;
        self.last_refresh
            .retain(|id, t| tails.contains_key(id) || wall_s - *t <= TAIL_EVICT_WALL_S);
        work
    }
}

#[cfg(test)]
mod tests {
    use uni_core::DAY;
    use uni_physics::smallbody::{conic_from_perihelion, heliocentric_state_icrs};

    use super::*;
    use crate::comet::luminance;
    use crate::comet::test_support::{FOV_Y_RAD, HEIGHT_PX, comet_on, static_view};

    fn angle_deg(a: DVec3, b: DVec3) -> f64 {
        a.cross(b).length().atan2(a.dot(b)).to_degrees()
    }

    /// Ruling 8: a comet moving at 40 km/s perpendicular to the Sun line has its ion tail
    /// atan(40/400) = 5.711° off anti-sunward, on the side it is moving away from; at rest the
    /// tail is exactly anti-sunward.
    #[test]
    fn ion_tail_aberration() {
        let want = (40.0_f64 / 400.0).atan().to_degrees();
        assert!((want - 5.711).abs() < 1e-3);
        for sun_rel in [
            DVec3::new(-AU, 0.0, 0.0),
            DVec3::new(0.3, -0.5, 0.8).normalize() * 0.7 * AU,
        ] {
            let anti = -sun_rel.normalize();
            let side = anti.any_orthonormal_vector();
            let v = 40.0e3 * side;
            let u = ion_tail_dir(sun_rel, v);
            assert!((u.length() - 1.0).abs() < 1e-12);
            let got = angle_deg(u, anti);
            assert!((got - want).abs() < 0.01, "{got}° vs {want}°");
            assert!(u.dot(side) < 0.0, "the tail lags the motion");
            let rest = ion_tail_dir(sun_rel, DVec3::ZERO);
            assert!(angle_deg(rest, anti) < 1e-9, "{rest} vs {anti}");
        }
        assert_eq!(ion_tail_dir(DVec3::ZERO, DVec3::X), DVec3::ZERO);
        assert!((ion_tail_length_m(1.0) - 0.3 * AU).abs() < 1.0);
        assert!((ion_tail_length_m(0.6) - 0.5 * AU).abs() < 1.0);
        assert_eq!(ion_tail_length_m(0.1), AU);
    }

    /// The ion ribbon's profile integrates to the tail's light, ∫∫ S sin ψ ds dx = I, and is 0 at
    /// the tip and beyond either end.
    #[test]
    fn ion_tail_profile_integrates_to_its_light() {
        let (l, i) = (0.4 * AU, 2.0e14);
        for sin_psi in [1.0, 0.5, 0.1] {
            let (ns, nx) = (4_000, 400);
            let half = ION_HALF_WIDTH_SIGMAS * ION_TAIL_WIDTH_M / 2.354_820_045;
            let (ds, dx) = (l / ns as f64, 2.0 * half / nx as f64);
            let mut sum = 0.0;
            for a in 0..ns {
                for b in 0..nx {
                    let s = (a as f64 + 0.5) * ds;
                    let x = -half + (b as f64 + 0.5) * dx;
                    sum += ion_tail_brightness(s, x, l, i, sin_psi) * sin_psi * ds * dx;
                }
            }
            assert!(
                (sum / i - 1.0).abs() < 2e-3,
                "sin ψ {sin_psi}: {sum} of {i}"
            );
        }
        assert_eq!(ion_tail_brightness(l, 0.0, l, i, 1.0), 0.0);
        assert_eq!(ion_tail_brightness(-1.0, 0.0, l, i, 1.0), 0.0);
        assert_eq!(ion_tail_brightness(l + 1.0, 0.0, l, i, 1.0), 0.0);
        // Looking along the axis the brightness is capped at the axial column (sin ψ ≥ w/L).
        assert_eq!(
            ion_tail_brightness(1e9, 0.0, l, i, 0.0),
            ion_tail_brightness(1e9, 0.0, l, i, ION_TAIL_WIDTH_M / l)
        );
        let c = ion_colour();
        assert!((luminance(c) - 1.0).abs() < 1e-5, "{c}");
        assert!(c.min_element() > 0.0 && c.z > c.y && c.z > c.x, "{c}");
    }

    /// Review C1 I2: from afar the ion ribbon is drawn at least `ION_MIN_SIGMA_PX` wide, keeps
    /// the light across it, and the light of one screen row no longer depends on where the line
    /// falls between pixel centres. Without the floor, 2 AU away (σ = 0.13 px, the whole ribbon
    /// under one pixel) a row's light swung between 0 and several times its mean.
    #[test]
    fn ion_ribbon_has_a_pixel_floor() {
        // One pixel's side at unit distance: 1 080 lines, 60° (the fixtures' camera).
        let pixel_rad = 2.0 * (0.5 * FOV_Y_RAD).tan() / HEIGHT_PX;
        let phys = ion_sigma_m();
        let (length, intensity, sin_psi) = (0.3 * AU, 2.0e14, 0.8);
        let s = 0.1 * length;
        let drawn =
            |x: f64, sigma: f64| ion_tail_drawn_brightness(s, x, length, intensity, sin_psi, sigma);
        // Light per unit length: ∫ S dx over the ribbon's ±3.5σ (midpoint rule).
        let across = |sigma: f64| {
            let half = ION_HALF_WIDTH_SIGMAS * sigma;
            let n = 4_000;
            let dx = 2.0 * half / f64::from(n);
            (0..n)
                .map(|k| drawn(-half + (f64::from(k) + 0.5) * dx, sigma) * dx)
                .sum::<f64>()
        };
        // A row's light (the samples at pixel centres inside the ribbon, × pixel) as the line
        // moves by φ = 0 … 39/40 px across them: (max − min) / mean.
        let row_ripple = |sigma: f64, pixel: f64| {
            let half = ION_HALF_WIDTH_SIGMAS * sigma;
            let n = (half / pixel).ceil() as i64 + 1;
            let sums: Vec<f64> = (0..40)
                .map(|q| {
                    let phi = f64::from(q) / 40.0;
                    (-n..=n)
                        .map(|k| (k as f64 + phi) * pixel)
                        .filter(|x| x.abs() < half)
                        .map(|x| drawn(x, sigma) * pixel)
                        .sum::<f64>()
                })
                .collect();
            let max = sums.iter().copied().fold(f64::MIN, f64::max);
            let min = sums.iter().copied().fold(f64::MAX, f64::min);
            (max - min) * sums.len() as f64 / sums.iter().sum::<f64>()
        };
        let reference = across(phys);
        assert!(reference > 0.0);
        let mut report = Vec::new();
        for d_au in [0.05, 0.3, 0.63, 1.0, 1.9, 2.0, 5.0] {
            let pixel = d_au * AU * pixel_rad;
            let sigma = ion_drawn_sigma_m(pixel);
            assert_eq!(sigma, phys.max(ION_MIN_SIGMA_PX * pixel), "{d_au} AU");
            let fwhm_px = 2.354_820_045 * sigma / pixel;
            assert!(fwhm_px >= 2.354_820_045 * ION_MIN_SIGMA_PX * (1.0 - 1e-12));
            let kept = across(sigma) / reference;
            assert!(
                (kept - 1.0).abs() < 1e-9,
                "{d_au} AU: light across × {kept}"
            );
            let ripple = row_ripple(sigma, pixel);
            assert!(
                ripple < 5e-3,
                "{d_au} AU: a row's light ripples by {ripple}"
            );
            report.push(format!(
                "{d_au} AU: σ {:.2} px, ripple {ripple:.1e}",
                sigma / pixel
            ));
        }
        println!("ion ribbon with the floor: {}", report.join("; "));
        // The defect the floor fixes: the bare profile 2 AU away.
        let bare = row_ripple(phys, 2.0 * AU * pixel_rad);
        println!("without the floor at 2 AU: ripple {bare:.2}");
        assert!(bare > 1.0, "{bare}");
        // Up close the floor is inactive: the drawn profile is the physical one.
        assert_eq!(ion_drawn_sigma_m(1.0e3), phys);
        assert_eq!(
            drawn(1.0e7, phys),
            ion_tail_brightness(s, 1.0e7, length, intensity, sin_psi)
        );
        assert_eq!(drawn(0.0, 0.0), 0.0);
        // Widening lowers the peak by the same factor σ_phys/σ.
        let wide = 4.0 * phys;
        assert!((drawn(0.0, wide) / drawn(0.0, phys) - 0.25).abs() < 1e-12);
    }

    /// The HG phase function has mean 1 over the sphere and favours forward scattering; the
    /// sprite profile integrates to 1 and reaches 0 at its cut.
    #[test]
    fn dust_phase_and_sprite_are_normalised() {
        let n = 20_000;
        let mean = (0..n)
            .map(|k| dust_phase(-1.0 + 2.0 * (k as f64 + 0.5) / n as f64))
            .sum::<f64>()
            / n as f64;
        assert!((mean - 1.0).abs() < 1e-4, "{mean}");
        assert!(dust_phase(1.0) > 10.0 * dust_phase(-1.0));
        let sigma = 3.0e7;
        let m = 20_000;
        let h = SPRITE_SIGMAS * sigma / m as f64;
        let total: f64 = (0..m)
            .map(|k| {
                let r = (k as f64 + 0.5) * h;
                2.0 * std::f64::consts::PI * r * sprite_profile(r, sigma) * h
            })
            .sum();
        assert!((total - 1.0).abs() < 1e-6, "{total}");
        assert_eq!(sprite_profile(SPRITE_SIGMAS * sigma, sigma), 0.0);
        assert_eq!(sprite_profile(4.0 * sigma, sigma), 0.0);
    }

    /// A prograde short-period comet and 1P/Halley (retrograde) shortly after perihelion: the
    /// dust tail trails the motion (mean grain offset · v < 0) and points away from the Sun
    /// (· r̂_h > 0); the splats are at most one per grain, every covariance is positive definite,
    /// and the weights add up to the emission history H = Σ P(t_e)/P_now·Δt.
    #[test]
    fn dust_tail_lags_motion() {
        let comets = [
            // 1P/Halley's SBDB elements (q, e, i, Ω, ω, T_p), 9 days after perihelion.
            (
                conic_from_perihelion(
                    0.574_863_831_374_341_3,
                    0.967_935_995_695_321_1,
                    162.190_530_043_912_9,
                    59.098_947_206_124_37,
                    112.241_431_463_776_4,
                    2_446_469.973_616_146_5,
                ),
                9.0,
            ),
            // A 67P-like prograde orbit (q 1.24 AU, e 0.64, i 7°), 20 days after perihelion.
            (
                conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5),
                20.0,
            ),
        ];
        for (conic, days) in comets {
            let t = conic.tp.add_seconds(days * DAY);
            let (r, v) = heliocentric_state_icrs(&conic, t);
            let grains = dust_rows(&conic, 10.0, t, 0..DUST_PARAMS.n_beta);
            assert_eq!(grains.len(), GRAINS_PER_TAIL);
            let mean =
                grains.iter().map(|g| g.pos_rel_nucleus_m).sum::<DVec3>() / grains.len() as f64;
            assert!(
                mean.dot(v) < 0.0,
                "mean offset {mean} m leads the motion {v}"
            );
            assert!(mean.dot(r) > 0.0, "mean offset {mean} m points sunward");
            let now = dust_production(r.length() / AU, 10.0);
            let splats = tail_grains(&grains, now);
            assert!(!splats.is_empty() && splats.len() <= GRAINS_PER_TAIL);
            assert!(
                splats
                    .iter()
                    .all(|g| g.weight >= 0.0 && g.cov_m2.determinant() > 0.0)
            );
            let sqrt_beta: f64 = (0..DUST_PARAMS.n_beta as usize)
                .map(|j| grains[j * DUST_PARAMS.n_age as usize].beta.sqrt())
                .sum();
            let dt = DUST_PARAMS.max_age_days * DAY / f64::from(DUST_PARAMS.n_age);
            let h: f64 = grains
                .iter()
                .map(|g| g.weight / (now * sqrt_beta) * dt)
                .sum();
            let got: f64 = splats.iter().map(|g| g.weight).sum();
            assert!((got / h - 1.0).abs() < 1e-12, "Σw {got} s vs H {h} s");
            println!(
                "{days} d after perihelion: {} splats, H {:.1} d",
                splats.len(),
                h / DAY
            );
        }
    }

    /// C1L: with a constant production (a circular orbit) the weights add up to the 60 days of
    /// ages; merging keeps the tail's light and its first and second moments about the nucleus
    /// (Σw, Σw·x, Σw·(cov + x xᵀ)) and only ever reduces the splat count.
    #[test]
    fn dust_weights_and_merging_keep_the_light() {
        let conic = conic_from_perihelion(0.8, 0.0, 0.0, 0.0, 0.0, 2_451_545.0);
        let t = conic.tp.add_seconds(30.0 * DAY);
        let grains = dust_rows(&conic, 10.0, t, 0..DUST_PARAMS.n_beta);
        let now = dust_production(0.8, 10.0);
        let apart = tail_grains_merged_within(&grains, now, 0.0);
        assert_eq!(apart.len(), GRAINS_PER_TAIL);
        let h: f64 = apart.iter().map(|g| g.weight).sum();
        assert!((h / (60.0 * DAY) - 1.0).abs() < 1e-9, "H {} d", h / DAY);
        let moments = |splats: &[TailGrain]| {
            let w: f64 = splats.iter().map(|g| g.weight).sum();
            let m1 = splats.iter().map(|g| g.offset_m * g.weight).sum::<DVec3>();
            let m2 = splats
                .iter()
                .map(|g| (g.cov_m2 + outer(g.offset_m)) * g.weight)
                .fold(DMat3::ZERO, |a, c| a + c);
            (w, m1, m2)
        };
        let (w0, a0, b0) = moments(&apart);
        for conic_t in [(conic, t), {
            let c = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
            (c, c.tp.add_seconds(20.0 * DAY))
        }] {
            let g = dust_rows(&conic_t.0, 10.0, conic_t.1, 0..DUST_PARAMS.n_beta);
            let now = dust_production(
                heliocentric_state_icrs(&conic_t.0, conic_t.1).0.length() / AU,
                10.0,
            );
            let (w0, a0, b0) = moments(&tail_grains_merged_within(&g, now, 0.0));
            let merged = tail_grains(&g, now);
            let (w1, a1, b1) = moments(&merged);
            assert!(merged.len() < GRAINS_PER_TAIL, "nothing merged");
            assert!((w1 / w0 - 1.0).abs() < 1e-12);
            assert!((a1 - a0).length() <= 1e-9 * a0.length());
            let scale = b0.abs().to_cols_array().into_iter().fold(0.0, f64::max);
            let diff = (b1 - b0)
                .abs()
                .to_cols_array()
                .into_iter()
                .fold(0.0, f64::max);
            assert!(
                diff <= 1e-9 * scale,
                "second moments differ by {diff:e} of {scale:e}"
            );
            println!("{} → {} splats", g.len(), merged.len());
        }
        assert!(w0 > 0.0 && a0.length() > 0.0 && b0.determinant() > 0.0);
        // Members farther apart than the threshold stay apart.
        let far = |x: f64| Splat {
            grain: TailGrain {
                offset_m: DVec3::new(x, 0.0, 0.0),
                cov_m2: DMat3::IDENTITY,
                weight: 1.0,
            },
            spread_m: 1.0,
        };
        assert!(merge(&[far(0.0), far(1.0), far(2.0), far(3.0)], 2.0).is_some());
        assert!(merge(&[far(0.0), far(1.0), far(2.0), far(30.0)], 2.0).is_none());
    }

    /// The splat ellipse (CPU twin of `splat_axes` in comet.wgsl): σ₁² cos²φ + σ₂² sin²φ etc.
    /// rebuild the covariance projected on (right, up) plus the floor; a flat (rank-1)
    /// covariance seen edge-on still gets the floor on its minor axis.
    #[test]
    fn splat_axes_rebuild_the_projected_covariance() {
        let right = DVec3::new(0.6, 0.8, 0.0);
        let up = DVec3::new(-0.48, 0.36, 0.8);
        assert!(right.dot(up).abs() < 1e-12);
        let a = DVec3::new(3.0e9, -1.0e9, 2.0e8);
        let b = DVec3::new(2.0e8, 5.0e8, -7.0e8);
        for (cov, floor) in [
            (outer(a) + outer(b), 1.0e6),
            (
                outer(a) * 0.25 + DMat3::from_diagonal(DVec3::splat(4.0e16)),
                3.0e7,
            ),
            (outer(right * 2.0e9), 1.0e7),
            (DMat3::ZERO, 2.0e6),
        ] {
            let (s1, s2, phi) = splat_axes(cov_factor(cov), right, up, floor);
            assert!(s1 >= s2 && s2 >= floor * (1.0 - 1e-9), "{s1} {s2} {floor}");
            let (c, s) = (phi.cos(), phi.sin());
            let p = s1 * s1 * c * c + s2 * s2 * s * s;
            let q = (s1 * s1 - s2 * s2) * c * s;
            let r = s1 * s1 * s * s + s2 * s2 * c * c;
            let want = [
                right.dot(cov * right) + floor * floor,
                right.dot(cov * up),
                up.dot(cov * up) + floor * floor,
            ];
            let tol = 1e-8 * (s1 * s1);
            assert!(
                (p - want[0]).abs() <= tol
                    && (q - want[1]).abs() <= tol
                    && (r - want[2]).abs() <= tol
            );
        }
        let (s1, s2, _) = splat_axes(cov_factor(outer(up * 5.0e8)), right, up, 1.0e6);
        assert!((s1 / 5.0e8 - 1.0).abs() < 1e-5 && (s2 / 1.0e6 - 1.0).abs() < 1e-6);
    }

    /// C1L: `DUST_HEAD_TIME_S` is what its documented inputs give: NEOWISE's visual law at Aravind
    /// et al.'s first night (r_h 0.63 AU, Δ 0.69 AU, Earth–Sun 1.0160 AU), their A(0)fρ 34.99 m
    /// taken back to the observed phase with the Halley–Marcus composite, the 1.11 km/s dust
    /// speed, and HG(0.6) at the scattering angle.
    #[test]
    fn neowise_head_time_is_derived() {
        let (tau, phase_deg, phi) = neowise_calibration();
        println!(
            "τ {tau:.4e} s ({:.3} d), phase {phase_deg:.3}°, Φ_HM {phi:.4}",
            tau / DAY
        );
        assert!((phase_deg - 100.56).abs() < 0.01, "{phase_deg}");
        assert!((phi - 0.7103).abs() < 1e-4, "{phi}");
        assert!(
            (tau / DUST_HEAD_TIME_S - 1.0).abs() < 1e-3,
            "{tau} vs {DUST_HEAD_TIME_S}"
        );
    }

    /// C1L (owner: NEOWISE's dust tail was short and not visibly curved): at row 20's geometry
    /// (2020-07-15 03:00 UTC from the Earth, fov 20°, 1 080 lines, up = ICRS +z) the splats,
    /// rasterised on the CPU at 1/4 of the lines, give a dust tail brighter than 10⁻⁴ cd/m²
    /// (22.6 mag/arcsec², about the natural night sky) at 3° from the nucleus, whose centre line's
    /// direction from the nucleus turns by more than 4° between 1° and 4° (a tangent turn of more
    /// than 8°: curved, not a straight streak), on the side of the ion tail away from the comet's
    /// motion. Prints the peak luminance and centre line by radius.
    #[test]
    fn neowise_dust_tail_is_long_and_curved() {
        use uni_physics::standish::{Planet, heliocentric_state_icrs as planet_state};
        let conic = conic_from_perihelion(
            0.294_651_249_380_919_6,
            0.999_178_026_253_129_2,
            128.937_502_759_480_9,
            61.010_428_185_369_88,
            37.278_658_448_125_7,
            2_459_034.178_898_044_4,
        );
        let t = Epoch::from_unix_utc(1_594_782_000.0);
        let (nucleus, vel) = heliocentric_state_icrs(&conic, t);
        let (earth, _) = planet_state(Planet::EmBary, t);
        let rel = nucleus - earth;
        let (r_au, delta_au) = (nucleus.length() / AU, rel.length() / AU);
        let law = uni_sim::world::comet_visual_law(NEOWISE_SPKID).expect("NEOWISE's law");
        let coma = crate::comet::coma::comet_total_intensity_cd(law.m1(), law.k1, r_au, delta_au);
        let grains = dust_rows(&conic, law.k1, t, 0..DUST_PARAMS.n_beta);
        let splats = tail_grains(&grains, dust_production(r_au, law.k1));
        let rate = dust_emission_cd_per_s(coma);
        let fwd = rel.normalize();
        let right = fwd.cross(DVec3::Z).normalize();
        let up = right.cross(fwd);
        // Row 20's pixel, and the test's 4 × coarser grid.
        let f_full = 540.0 / 10.0_f64.to_radians().tan();
        let (w, h, k) = (480_usize, 270_usize, 4.0);
        let f = f_full / k;
        let mut img = vec![0.0_f64; w * h];
        let (mut light, mut drawn) = (0.0, 0.0);
        let (mut fill, mut count) = ([0.0_f64; 3], [0_u32; 3]);
        for g in &splats {
            let p = rel + g.offset_m;
            let d = p.length();
            let mut i = rate * dust_grain_light(g.offset_m, g.weight, rel, -nucleus);
            light += i;
            let pixel_m = d / f_full;
            let (mut s1, mut s2, phi) =
                splat_axes(cov_factor(g.cov_m2), right, up, SPRITE_MIN_PX * pixel_m);
            // The pass's two paths: wide splats on the coarse one, with its larger cap.
            let cap = if s2 >= crate::comet::COARSER_MIN_SIGMA_PX * pixel_m {
                crate::comet::COARSER_MAX_PX
            } else if s2 >= crate::comet::COARSE_MIN_SIGMA_PX * pixel_m {
                crate::comet::COARSE_MAX_PX
            } else {
                crate::comet::SPRITE_MAX_PX
            };
            let s_max = cap * pixel_m;
            if s1 > s_max {
                i *= s_max / s1;
                s1 = s_max;
            }
            if s2 > s_max {
                i *= s_max / s2;
                s2 = s_max;
            }
            drawn += i;
            {
                // The fill per path: texels of the path's target (quads clipped roughly).
                let level = if s2 >= crate::comet::COARSER_MIN_SIGMA_PX * pixel_m {
                    2
                } else if s2 >= crate::comet::COARSE_MIN_SIGMA_PX * pixel_m {
                    1
                } else {
                    0
                };
                let ds = [1.0, 8.0, 64.0][level];
                let (a, b) = (4.0 * s1 / pixel_m / ds, 4.0 * s2 / pixel_m / ds);
                let (tw, th) = (1920.0 / ds, 1080.0 / ds);
                fill[level] += a.min(tw * 1.5) * b.min(th * 1.5);
                count[level] += 1;
            }
            let z = p.dot(fwd);
            let (cx, cy) = (
                0.5 * w as f64 + f * p.dot(right) / z,
                0.5 * h as f64 - f * p.dot(up) / z,
            );
            let m_per_px = d / f;
            let reach = SPRITE_SIGMAS * s1 / m_per_px + 1.0;
            let (e1, e2) = (
                DVec3::new(phi.cos(), phi.sin(), 0.0),
                DVec3::new(-phi.sin(), phi.cos(), 0.0),
            );
            let x0 = (cx - reach).floor().max(0.0) as usize;
            let y0 = (cy - reach).floor().max(0.0) as usize;
            let x1 = ((cx + reach).ceil().max(0.0) as usize).min(w);
            let y1 = ((cy + reach).ceil().max(0.0) as usize).min(h);
            for y in y0..y1 {
                for x in x0..x1 {
                    // Screen offset → metres on the sprite plane (x right, y up).
                    let v = DVec3::new(x as f64 + 0.5 - cx, cy - (y as f64 + 0.5), 0.0) * m_per_px;
                    let q = (v.dot(e1) / s1).hypot(v.dot(e2) / s2);
                    img[y * w + x] += i * sprite_profile(q, 1.0) / (s1 * s2);
                }
            }
        }
        // On circles about the nucleus, by radius in degrees: the brightest pixel, and the
        // tail's centre line (the luminance-weighted mean direction).
        let px_per_deg = f * 1.0_f64.to_radians().tan();
        let angle = |v: DVec3| v.dot(up).atan2(v.dot(right)).to_degrees();
        let circle = |deg: f64| {
            let r = deg * px_per_deg;
            let (mut peak, mut sx, mut sy) = (0.0_f64, 0.0, 0.0);
            for a in 0..1440 {
                let az = (f64::from(a) * 0.25).to_radians();
                let (x, y) = (0.5 * w as f64 + r * az.cos(), 0.5 * h as f64 - r * az.sin());
                if x >= 0.0 && y >= 0.0 && (x as usize) < w && (y as usize) < h {
                    let l = img[y as usize * w + x as usize];
                    peak = peak.max(l);
                    (sx, sy) = (sx + l * az.cos(), sy + l * az.sin());
                }
            }
            (peak, sy.atan2(sx).to_degrees())
        };
        let rows: Vec<(f64, f64, f64)> = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
            .map(|d| {
                let (peak, centre) = circle(d);
                (d, peak, centre)
            })
            .to_vec();
        let ion = angle(ion_tail_dir(-nucleus, vel));
        let motion = angle(vel);
        println!(
            "row 20: {} splats, tail {:.3e} cd = {:.2} × the coma ({:.2} % cut by SPRITE_MAX_PX); \
             ion {ion:.1}°, motion {motion:.1}°; by radius (deg, peak cd/m², centre line °):",
            splats.len(),
            light,
            light / coma,
            100.0 * (1.0 - drawn / light),
        );
        for r in &rows {
            println!("  {:.1}°  {:.3e}  {:.1}°", r.0, r.1, r.2);
        }
        println!(
            "splats and fill by path (full, 1/8, 1/64; texels, unclipped): {count:?} {fill:.0?}"
        );
        let at = |d: f64| {
            rows.iter()
                .find(|r| r.0 == d)
                .map_or((0.0, 0.0), |r| (r.1, r.2))
        };
        assert!(
            at(3.0).0 > 1.0e-4,
            "the tail at 3°: {:.2e} cd/m²",
            at(3.0).0
        );
        // A straight streak keeps one direction from the nucleus; an arc's direction from its
        // start turns by half its tangent's turn.
        let turn = at(1.0).1 - at(4.0).1;
        assert!(
            turn.abs() > 4.0,
            "the centre line turns by {turn:.1}° between 1° and 4°"
        );
        // Trailing: the centre line lies on the far side of the ion axis from the motion.
        let side = |a: f64| (a - ion).to_radians().sin();
        assert!(
            side(at(2.0).1) * side(motion) < 0.0,
            "centre line {:.1}° vs ion {ion:.1}°, motion {motion:.1}°",
            at(2.0).1
        );
    }

    /// Review I13: with 20 active comets on screen only the 8 brightest by `total_v` get tails;
    /// a comet without a magnitude law or beyond 5 AU never does; refreshed comets wait
    /// `TAIL_REFRESH_WALL_S` and then come back oldest first.
    #[test]
    fn tails_capped_at_eight() {
        let mut views: Vec<CometView> = (0..20_u32)
            .map(|id| {
                let mut v = static_view(
                    id,
                    DVec3::new(0.0, 0.0, -0.5 * AU),
                    DVec3::new(-AU, 0.0, 0.0),
                    Some(8.0),
                    Some(10.0),
                );
                v.total_v = Some(5.0 + 0.37 * f64::from((id * 7) % 20));
                v
            })
            .collect();
        let picked = tail_schedule(&views, &HashMap::new(), 0.0);
        assert_eq!(picked.len(), MAX_TAILED_COMETS);
        let mut mags: Vec<f64> = views.iter().filter_map(|v| v.total_v).collect();
        mags.sort_by(f64::total_cmp);
        let mut got: Vec<f64> = picked.iter().filter_map(|&i| views[i].total_v).collect();
        got.sort_by(f64::total_cmp);
        assert_eq!(got, mags[..MAX_TAILED_COMETS].to_vec());
        assert_eq!(tailed_comets(&views).len(), MAX_TAILED_COMETS);

        // The brightest comet loses its law; another moves beyond 5 AU: neither is picked.
        let brightest = picked[0];
        views[brightest].m1 = None;
        views[brightest].total_v = None;
        let second = tailed_comets(&views)[0];
        views[second].r_au = 5.1;
        let picked = tail_schedule(&views, &HashMap::new(), 0.0);
        assert_eq!(picked.len(), MAX_TAILED_COMETS);
        assert!(!picked.contains(&brightest) && !picked.contains(&second));

        // All refreshed at wall 0: none due at 0.1 s; all due again at 0.3 s, oldest first.
        let mut last: HashMap<ObjectId, f64> = picked.iter().map(|&i| (views[i].id, 0.0)).collect();
        assert!(tail_schedule(&views, &last, 0.1).is_empty());
        last.insert(views[picked[5]].id, -1.0);
        let again = tail_schedule(&views, &last, 0.3);
        assert_eq!(again.len(), MAX_TAILED_COMETS);
        assert_eq!(again[0], picked[5]);
    }

    /// Review I13: under warp 10⁹ for 2 s of wall time at 60 fps (every frame moves the epoch
    /// 193 days), each tailed comet is refreshed at most 9 times (every 0.25 s of wall time) and
    /// at least 6 (the round robin reaches all eight), comets beyond the eight never, and no frame
    /// does more than the 1 ms share of Kepler solves at the plan's 0.5 µs each.
    #[test]
    fn tail_refresh_rate_limited_under_warp() {
        let conic = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
        let t0 = conic.tp;
        let views: Vec<CometView> = (0..10_u32)
            .map(|id| {
                let mut v = comet_on(id, conic, t0, DVec3::new(0.0, AU, 0.0), 8.0, 10.0);
                v.total_v = Some(6.0 + f64::from(id));
                v
            })
            .collect();
        let tailed: Vec<ObjectId> = tailed_comets(&views).iter().map(|&i| views[i].id).collect();
        assert_eq!(tailed.len(), 8);
        let mut manager = TailManager::default();
        let mut started: HashMap<ObjectId, usize> = HashMap::new();
        let mut completed: HashMap<ObjectId, usize> = HashMap::new();
        let (mut worst, mut total) = (0, 0);
        let clock = std::time::Instant::now();
        for frame in 0..120_u32 {
            let wall = f64::from(frame) / 60.0;
            let epoch = t0.add_seconds(1.0e9 * wall);
            let work = manager.update(&views, epoch, wall, NOMINAL_SOLVES_PER_FRAME);
            worst = worst.max(work.solves);
            total += work.solves;
            for id in work.started {
                *started.entry(id).or_default() += 1;
            }
            for id in work.completed {
                *completed.entry(id).or_default() += 1;
            }
        }
        println!(
            "120 frames: worst {worst} solves/frame, {total} in all, {:.2} µs per solve (this \
             build); started {started:?}, completed {completed:?}",
            clock.elapsed().as_secs_f64() * 1e6 / total as f64
        );
        assert!(
            worst as f64 * KEPLER_SOLVE_US <= 1_000.0 * TAIL_CPU_SHARE_MS,
            "{worst} solves in one frame"
        );
        for id in &tailed {
            let n = started.get(id).copied().unwrap_or(0);
            assert!(n <= 9, "comet {id} refreshed {n} times in 2 s");
            let done = completed.get(id).copied().unwrap_or(0);
            assert!(done >= 6, "comet {id} completed only {done} refreshes");
        }
        assert!(started.keys().all(|id| tailed.contains(id)));
        assert_eq!(manager.tails.len(), 8);
        for (id, tail) in &manager.tails {
            assert!(tailed.contains(id));
            assert!(!tail.grains.is_empty() && tail.grains.len() <= GRAINS_PER_TAIL);
            assert!(tail.reach_m > 0.0 && tail.reach_m.is_finite());
        }
    }

    /// At real time a tail is not recomputed until an hour of sim time has passed, however much
    /// wall time goes by; a date jump refreshes it; a comet gone from the screen is dropped after
    /// `TAIL_EVICT_WALL_S`.
    #[test]
    fn tail_refresh_waits_for_an_hour_of_sim_time() {
        let conic = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
        let t0 = conic.tp;
        let views = vec![comet_on(1, conic, t0, DVec3::new(0.0, AU, 0.0), 8.0, 10.0)];
        let mut manager = TailManager::default();
        let work = manager.update(&views, t0, 0.0, usize::MAX);
        assert_eq!(
            (work.started.clone(), work.completed.clone()),
            (vec![1], vec![1])
        );
        // 50 minutes of sim time over 3 000 s of wall time: no refresh.
        for k in 1..=10 {
            let wall = 300.0 * f64::from(k);
            let work = manager.update(&views, t0.add_seconds(wall), wall, usize::MAX);
            if wall < TAIL_SIM_REFRESH_S {
                assert!(work.started.is_empty(), "refreshed at {wall} s");
            }
        }
        let work = manager.update(&views, t0.add_seconds(-5.0 * DAY), 3_100.0, usize::MAX);
        assert_eq!(work.completed, vec![1]);
        // Off screen for longer than the eviction time: dropped.
        manager.update(&[], t0, 3_100.0 + TAIL_EVICT_WALL_S + 1.0, usize::MAX);
        assert!(manager.tails.is_empty() && manager.last_refresh.is_empty());
    }
}
