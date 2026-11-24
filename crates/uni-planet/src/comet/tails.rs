//! The dust and ion tails (Task C1).
//!
//! - **Dust** (Task C1L): `uni_physics::dust::finson_probstein` (Finson & Probstein 1968, ApJ 154,
//!   327) on `CometView::conic` (J2000 ecliptic, as it expects; lead ruling A.1: never rotate it
//!   to ICRS first), β ∈ [0.001, 0.9] in 16 log steps × ages up to 60 d in 64 steps = 1 024 grains
//!   ([`DUST_PARAMS`]). Dust production at emission ∝ r_h^exponent ([`dust_production_exponent`]):
//!   a comet's measured A(0)fρ slope where [`COMET_DUST_LAWS`] has one (NEOWISE: r_h^−4.0, Biver
//!   et al. 2022; C1L fix round 2, lead ruling C1L-R1), else −max(K1/2.5 − 2, 0), the total
//!   visual magnitude law without the sunlight it already contains (C1L fix round 1, ruling
//!   C1L-I1).
//!   - *Light.* Each grain stands for the dust emitted in its cell of ln β × age. The dust's light
//!     production now ([`dust_rate_cd_per_s`]) is v·A(0)fρ(r_h)·E☉/(2π) from a comet's dust law,
//!     else the coma's light divided by [`DUST_HEAD_TIME_S`], a time calibrated on
//!     C/2020 F3 (NEOWISE) from its measured dust production A(0)fρ (Aravind et al. 2025), its
//!     measured dust speed (Manzini et al. 2021) and the steady-outflow relation of A'Hearn et al.
//!     1984 (derivation on the constant); both agree at that anchor. A grain carries that rate ×
//!     its emission weight
//!     [`TailGrain::weight`] (production at emission relative to now × its size class's share ×
//!     the cell's age span) × the sunlight at its own heliocentric distance × a
//!     Henyey–Greenstein phase function with g = 0.6 ([`DUST_PHASE_G`]). The tail is therefore
//!     many times brighter than the coma just after perihelion (11.5 × for row 20's NEOWISE, 12 d
//!     after it, with its dust law; 3.55 × with the K1 slope: `neowise_dust_tail_is_long_and_curved`
//!     prints it) and fainter on the way in; it depends on the viewing phase as the phase
//!     function does.
//!     The v0 ratio "dust = 0.5 × coma" is gone (C1L: it made NEOWISE's tail short).
//!   - *Colour.* Sunlight (5 772 K), with no dust reddening (approximate: the C1L brief describes
//!     NEOWISE's dust tail as yellowish, the render reads whitish; review C1L minor 8, an M2
//!     item).
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
//!   across it, CO⁺ blue (the comet-tail system's 36 bands summed through the CMFs, Rousselot et
//!   al. 2024 g-factors at Opitom et al. 2019 wavelengths, [`ion_colour`]; C1L fix round 2, lead
//!   ruling C1L-R2), 0.2 × the coma's light (v0; [`ION_TAIL_LIGHT`]). Drawn as a ribbon that turns about its axis to face the camera,
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

use glam::{DMat3, DVec3, UVec2, Vec3};
use uni_core::{AU, CameraView, Epoch};
use uni_physics::dust::{DustGrain, DustParams, finson_probstein};
use uni_physics::kepler::Conic;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{COMET_ACTIVE_AU, CometView};
use uni_sim::world::PerihelionLeg;

use super::{MAX_TAILED_COMETS, TAIL_REFRESH_WALL_S, line_colour};

/// Solar-wind speed [m/s] for the ion tail's aberration (ruling 8; Brandt 1968).
pub const SOLAR_WIND_MPS: f64 = 400.0e3;
/// Ion-tail length at r_h = 1 AU [m]; it scales as 1/r_h (plan C1).
pub const ION_TAIL_LENGTH_AT_1AU_M: f64 = 0.3 * AU;
/// Longest ion tail drawn [m] (plan C1).
pub const ION_TAIL_MAX_LENGTH_M: f64 = AU;
/// Ion-tail width [m], the FWHM of its Gaussian cross-section (plan C1: 10⁵ km).
pub const ION_TAIL_WIDTH_M: f64 = 1.0e8;
/// The ion tail's light as a fraction of the coma's (plan C1, v0; approximate, card note
/// `uni_ui::theme::NOTE_TAILS`). C1L fix round 2 derived it from Aravind et al. 2025's CO⁺ fluxes
/// with every band of the system counted ([`aravind_co_plus_luminance`]) and this ribbon's
/// profile calibrated at their outermost aperture: 0.34 (`ion_light_from_aravind`), which would
/// make the line brighter, not the 10⁻⁶–10⁻⁵ cd/m² lead ruling C1L-R2 expected. The v0 value
/// stays until the lead rules (the along-tail profile beyond 10⁵ km is not measured there).
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
///   (`uni_sim::world::COMET_VISUAL_LAWS`: V 3.18 there). The law rests on COBS estimates at
///   r_h 0.431–0.494 AU (its valid range), so at 0.63 AU it is extrapolated along its K1 = 9.63,
///   Aravind et al.'s slope of their aperture photometry over 0.63–1.61 AU (review C1L minor 3;
///   a COBS median for 2020-07-21/23 would remove the extrapolation: a change to uni-sim's data,
///   not made here). For NEOWISE itself the law's M1 cancels between τ and the tail; every other
///   comet's τ carries it;
/// - HG(0.6) at the scattering angle 180° − α = 0.5257.
///
/// τ = I_coma·HG/(v·A(α)fρ·E☉/2π) = 5.42 × 10⁵ s = 6.3 d; `neowise_head_time_is_derived`
/// recomputes it. Approximations (card note `uni_ui::theme::NOTE_TAILS`: "Dust tail calibrated on
/// C/2020 F3 (NEOWISE) only; other comets approximate", lead ruling C1L-c):
/// - **One comet's value for all.** NEOWISE is not a dusty comet: Aravind et al. 2025 (§3.1.1)
///   classify it as gas-rich, with dust-to-gas ratios (log₁₀ A(0)fρ/Q(OH) = −25.42 ± 0.40)
///   "similar to typical gas-rich comets". The typical comets of A'Hearn et al. 1995 (Icarus 118,
///   223) scatter by about ±0.4 dex in that ratio about −25.8 (as quoted secondhand by the C1L
///   review; not checked against the original): another comet's dust tail may be drawn too
///   bright or too faint by about × 2.5 at 1σ, more for unusual comets. A per-comet A(0)fρ table
///   is an M2 item.
/// - One dust speed for all grain sizes and distances.
/// - The coma has no phase function, the dust does: the tail-to-coma ratio runs 1.43, 3.60 and
///   51 at phases 10°, 90° and 160° (`dust_record_carries_the_emission_rate`), where near
///   forward scattering the coma's own dust would brighten too (review C1L minor 7).
///
/// The v0 ratio (tail = 0.5 × coma) it replaces drew row 20's NEOWISE tail 7 × too faint
/// (3.55 × the coma now).
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

/// A comet's measured dust production (C1L fix round 2, lead ruling R1, 2026-11-24; the dust
/// counterpart of `uni_sim::world::COMET_VISUAL_LAWS`): the phase-corrected A(0)fρ as a power of
/// r_h from a primary refereed source, and its absolute level at an anchor.
///
/// **Production from Afρ.** A(0)fρ is the dust's scattering cross-section inside the aperture
/// ρ, A(0)·N(ρ)·σ̄/(πρ²) × ρ, with the r_h⁻² sunlight and the Δ already divided out (A'Hearn et
/// al. 1984, AJ 89, 579). A steady coma flowing out at v holds, inside ρ, the dust emitted over
/// the last πρ/(2v) (the relation [`DUST_HEAD_TIME_S`] rests on), so the cross-section emitted
/// per second is Ṡ = (2v/π)·N(ρ)σ̄/ρ = 2v·A(0)fρ/A(0) ∝ v·A(0)fρ. With one ejection speed for
/// every r_h ([`DUST_EJECTION_SPEED_MPS`], Manzini et al. 2021) the production goes as A(0)fρ:
/// P(r_h) ∝ r_h^`afrho_slope` ([`dust_production_exponent`]). The light it carries is lit at
/// each grain's own distance ([`dust_grain_light`]), so no r_h⁻² is added (lead ruling C1L-I1).
///
/// **Light now.** The dust's light production per unit of the phase function is
/// v·A(0)fρ(r_h)·E☉(r_h)/(2π)·Φ_HM(α₀)/HG(θ₀) ([`dust_rate_cd_per_s`]): A'Hearn's steady
/// outflow (the [`DUST_HEAD_TIME_S`] derivation) with the Halley–Marcus composite matched to the
/// renderer's HG(0.6) at the anchor's phase α₀ (θ₀ = 180° − α₀). At the anchor it equals
/// I_coma/τ exactly, so the two calibrations agree there (`dust_law_matches_tau_at_the_anchor`).
///
/// **Range.** The law is used at every r_h. Outside `r_valid_au` on `leg` it is extrapolated
/// ([`CometDustLaw::covers`], the rule of `CometVisualLaw::covers`). A comet without a row keeps
/// r_h^−max(K1/2.5 − 2, 0) and I_coma/τ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CometDustLaw {
    /// SBDB primary SPK-ID: the row applies to the comet with this SPK-ID.
    pub spkid: i32,
    /// SBDB name, for readers (the SPK-ID decides).
    pub name: &'static str,
    /// The r_h exponent of the phase-corrected A(0)fρ.
    pub afrho_slope: f64,
    /// Where the slope comes from.
    pub slope_source: &'static str,
    /// The heliocentric distances [AU] the slope's observations span, on `leg`.
    pub r_valid_au: (f64, f64),
    /// The leg the slope's observations were made on.
    pub leg: PerihelionLeg,
    /// The absolute level: A(0)fρ [m] at `anchor_r_au` [AU] (phase-corrected to 0°).
    pub anchor_afrho0_m: f64,
    pub anchor_r_au: f64,
}

impl CometDustLaw {
    /// A(0)fρ [m] at heliocentric distance `r_au`: the anchor × (r_h/r_anchor)^slope.
    pub fn afrho0_m(&self, r_au: f64) -> f64 {
        self.anchor_afrho0_m * (r_au / self.anchor_r_au).powf(self.afrho_slope)
    }

    /// Whether the slope rests on observations at `r_au` [AU] on this leg (`outbound`: after
    /// perihelion); elsewhere it is extrapolated.
    pub fn covers(&self, r_au: f64, outbound: bool) -> bool {
        let leg = match self.leg {
            PerihelionLeg::Both => true,
            PerihelionLeg::Inbound => !outbound,
            PerihelionLeg::Outbound => outbound,
        };
        leg && (self.r_valid_au.0..=self.r_valid_au.1).contains(&r_au)
    }
}

/// The comets with a measured dust-production slope ([`CometDustLaw`]).
///
/// **C/2020 F3 (NEOWISE).** Biver et al. 2022 ("Observations of comet C/2020 F3 (NEOWISE) with
/// IRAM telescopes", A&A 668, A171; read 2026-11-24): "the dust production in comet C/2020 F3
/// also varied steeply with heliocentric distance (Afρ(0) ∝ r_h^−4.0, Fig. 19)". Their Fig. 19
/// covers the post-perihelion weeks 13 July – 14 August 2020; its Afρ values come from red-layer
/// images, phase-corrected with Schleicher 2007 and Marcus 2007. That span is r_h 0.410–1.115 AU
/// on the SBDB orbit (`neowise_dust_law_range_is_the_papers`). Their absolute values (red band;
/// 10 000 cm on 21.9 July in their Fig. 2) are not used: the level stays Aravind et al. 2025's
/// green-continuum A(0)fρ = 34.99 m on 2020-07-22 at r_h 0.63 AU ([`NEOWISE_AFRHO0_M`]), the
/// anchor [`DUST_HEAD_TIME_S`] is calibrated at, so only the slope is Biver's.
pub const COMET_DUST_LAWS: [CometDustLaw; 1] = [CometDustLaw {
    spkid: NEOWISE_SPKID,
    name: "C/2020 F3 (NEOWISE)",
    afrho_slope: -4.0,
    slope_source: "Biver et al. 2022",
    r_valid_au: (0.410, 1.115),
    leg: PerihelionLeg::Outbound,
    anchor_afrho0_m: NEOWISE_AFRHO0_M,
    anchor_r_au: NEOWISE_CALIBRATION_AU[0],
}];

/// The [`COMET_DUST_LAWS`] row of the comet with SBDB SPK-ID `spkid`, if it has one.
pub fn comet_dust_law(spkid: i32) -> Option<&'static CometDustLaw> {
    COMET_DUST_LAWS.iter().find(|l| l.spkid == spkid)
}

/// The r_h exponent of a comet's dust production (the weights [`dust_rows`] gives the
/// emissions): its [`CometDustLaw`]'s A(0)fρ slope where it has one, else −max(K1/2.5 − 2, 0)
/// from its total visual slope `k1` ([`dust_production`]).
pub fn dust_production_exponent(spkid: i32, k1: f64) -> f64 {
    comet_dust_law(spkid).map_or(-(k1 / 2.5 - 2.0).max(0.0), |l| l.afrho_slope)
}

/// The dust's light production [cd/s per unit of HG] of comet `view` whose coma carries
/// `coma_cd` [cd] (`TailGpu::sun.w`): from its [`CometDustLaw`] where it has one,
/// v·A(0)fρ(r_h)·E☉(r_h)/(2π)·Φ_HM(α₀)/HG(θ₀) (the row's doc), else I_coma/τ
/// ([`dust_emission_cd_per_s`]). 0 for a non-positive or non-finite result.
pub fn dust_rate_cd_per_s(view: &CometView, coma_cd: f64) -> f64 {
    let Some(law) = comet_dust_law(view.spkid) else {
        return dust_emission_cd_per_s(coma_cd);
    };
    // The only anchor is NEOWISE's calibration night (an M2 per-comet A(0)fρ table adds others).
    let (alpha, phi) = neowise_calibration_phase();
    let phase_match = phi / dust_phase(-alpha.to_radians().cos());
    let r_m = view.r_au * AU;
    let sun_lux = super::sun_intensity_cd() / (r_m * r_m);
    let rate = DUST_EJECTION_SPEED_MPS * law.afrho0_m(view.r_au) * sun_lux * phase_match
        / std::f64::consts::TAU;
    if rate > 0.0 && rate.is_finite() {
        rate
    } else {
        0.0
    }
}

/// The NEOWISE calibration night's phase angle α [°] (the Sun–comet–Earth angle from
/// [`NEOWISE_CALIBRATION_AU`]'s three sides) and the Halley–Marcus Φ_HM(α) there.
fn neowise_calibration_phase() -> (f64, f64) {
    let [r_au, delta_au, earth_au] = NEOWISE_CALIBRATION_AU;
    let cos_alpha =
        (r_au * r_au + delta_au * delta_au - earth_au * earth_au) / (2.0 * r_au * delta_au);
    let alpha = cos_alpha.clamp(-1.0, 1.0).acos().to_degrees();
    let [p100, p101] = HALLEY_MARCUS_100_101;
    (alpha, p100 + (p101 - p100) * (alpha - 100.0))
}

/// [`DUST_HEAD_TIME_S`] from its inputs (its doc): returns (τ [s], the phase angle α [°] of the
/// calibration night, Φ_HM(α)). NaN when uni-sim has no visual law for NEOWISE.
pub fn neowise_calibration() -> (f64, f64, f64) {
    let [r_au, delta_au, _] = NEOWISE_CALIBRATION_AU;
    let (alpha, phi) = neowise_calibration_phase();
    let cos_alpha = alpha.to_radians().cos();
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
/// band, which Afghan et al. 2024 (PSJ 5, 264) bound at the β = 0.82 syndyne. Their second band,
/// beyond the β = 1.20 syndyne, is left out (the plan's β < 1: above 1 radiation pressure beats
/// gravity and the grains leave on hyperbolic orbits; Burns, Lamy & Soter 1979, Icarus 40, 1).
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
/// data table, DOI 10.25039/CIE.DS.xvudnb9b; read 2026-11-23). C1 drew the ion tail in this one
/// band's colour; since C1L fix round 2 it is the reference the CMF fit
/// ([`cie_1931_cmf_fit`]) is checked against.
pub const CO_PLUS_CMF: [f64; 3] = [0.244_879_7, 0.008_908_72, 1.188_497_1];

/// Fluorescence efficiencies g [erg s⁻¹ ion⁻¹] of the ¹²CO⁺ comet-tail system A²Πᵢ–X²Σ⁺ at
/// r_h = 1 au and heliocentric velocity 0, row v′ = 0–5, column v″ = 0–5 (Rousselot et al. 2024,
/// "¹²CO⁺ and ¹³CO⁺ fluorescence models for measuring the ¹²C/¹³C isotopic ratio in comets",
/// A&A 683, arXiv:2311.05700, Table 1; read 2026-11-24). Aravind et al. 2025 convert their
/// CO⁺ (3-0) and (2-0) fluxes with its 1.70 and 1.36 × 10⁻¹⁴. All bands scale as r_h⁻² together
/// (fluorescence), so their ratios hold at any r_h; the Swings effect (Doppler shifts into the
/// solar lines, a factor of up to about 2 per band, Rousselot et al. §4) is left out.
pub const CO_PLUS_G: [[f64; 6]; 6] = [
    [8.40e-16, 1.87e-15, 1.81e-15, 9.99e-16, 3.45e-16, 7.63e-17],
    [7.68e-15, 8.41e-15, 2.19e-15, 7.10e-18, 7.63e-16, 7.87e-16],
    [1.36e-14, 5.38e-15, 9.28e-17, 2.25e-15, 1.03e-15, 2.94e-18],
    [1.70e-14, 9.24e-16, 3.07e-15, 1.90e-15, 5.49e-17, 8.79e-16],
    [1.04e-14, 1.97e-16, 2.96e-15, 2.23e-17, 9.53e-16, 2.80e-16],
    [5.12e-15, 1.21e-15, 1.04e-15, 4.14e-16, 5.21e-16, 4.16e-17],
];

/// Observed CO⁺ A²Π–X²Σ band wavelengths [nm] as (v′, v″, λ) (Opitom et al. 2019, "High
/// resolution optical spectroscopy of the N₂-rich comet C/2016 R2 (PanSTARRS)", A&A 624, A64,
/// Table 3; read 2026-11-24).
pub const CO_PLUS_BANDS_NM: [(u32, u32, f64); 15] = [
    (6, 0, 341.0),
    (7, 1, 351.0),
    (5, 0, 359.0),
    (4, 0, 378.0),
    (3, 0, 400.0),
    (2, 0, 425.0),
    (4, 2, 452.0),
    (1, 0, 454.0),
    (2, 1, 468.0),
    (3, 2, 484.0),
    (0, 0, 490.0),
    (1, 1, 504.0),
    (0, 1, 547.0),
    (1, 2, 567.0),
    (0, 2, 619.0),
];

/// The band wavenumbers' vibrational fit [cm⁻¹], ν(v′, v″) = c₀ + c₁v′ − c₂v′(v′ + 1) − c₃v″ +
/// c₄v″(v″ + 1) (the anharmonic-oscillator term differences, Herzberg 1950, *Spectra of Diatomic
/// Molecules*, ch. III), least squares over [`CO_PLUS_BANDS_NM`]: it places the 23 bands of
/// [`CO_PLUS_G`] Opitom et al. do not list within about 1.5 nm, their largest residual.
pub fn co_plus_vibrational_fit() -> [f64; 5] {
    let row = |v1: f64, v2: f64| [1.0, v1, -v1 * (v1 + 1.0), -v2, v2 * (v2 + 1.0)];
    let mut ata = [[0.0_f64; 5]; 5];
    let mut aty = [0.0_f64; 5];
    for &(v1, v2, nm) in &CO_PLUS_BANDS_NM {
        let a = row(f64::from(v1), f64::from(v2));
        let y = 1.0e7 / nm;
        for i in 0..5 {
            aty[i] += a[i] * y;
            for j in 0..5 {
                ata[i][j] += a[i] * a[j];
            }
        }
    }
    // Gauss–Jordan elimination with partial pivoting on the 5 × 5 normal equations.
    for i in 0..5 {
        let p = (i..5)
            .max_by(|&a, &b| ata[a][i].abs().total_cmp(&ata[b][i].abs()))
            .unwrap_or(i);
        ata.swap(i, p);
        aty.swap(i, p);
        for r in 0..5 {
            if r != i && ata[i][i] != 0.0 {
                let f = ata[r][i] / ata[i][i];
                for c in 0..5 {
                    ata[r][c] -= f * ata[i][c];
                }
                aty[r] -= f * aty[i];
            }
        }
    }
    std::array::from_fn(|i| aty[i] / ata[i][i])
}

/// The wavelength [nm] of CO⁺ band (v′, v″): Opitom et al.'s where they list it
/// ([`CO_PLUS_BANDS_NM`]), else the vibrational fit's ([`co_plus_vibrational_fit`]).
pub fn co_plus_band_nm(v1: u32, v2: u32) -> f64 {
    if let Some(&(_, _, nm)) = CO_PLUS_BANDS_NM.iter().find(|b| b.0 == v1 && b.1 == v2) {
        return nm;
    }
    let c = co_plus_vibrational_fit();
    let (a, b) = (f64::from(v1), f64::from(v2));
    1.0e7 / (c[0] + c[1] * a - c[2] * a * (a + 1.0) - c[3] * b + c[4] * b * (b + 1.0))
}

/// CIE 1931 2° colour-matching functions x̄, ȳ, z̄ at `lambda_nm`: the multi-lobe piecewise
/// Gaussian fit of Wyman, Sloan & Shirley 2013 ("Simple Analytic Approximations to the CIE XYZ
/// Color Matching Functions", JCGT 2(2)), the fit `uni_core::color` uses (private there, so
/// repeated here). Against the CIE 2018 table: within 5 % at 427 nm, ȳ within 1 % at 516 nm
/// (`cmf_fit_matches_the_cie_table`); x̄ near its 505 nm minimum and ȳ below 410 nm (4 × 10⁻⁴)
/// are rougher, where the CO⁺ bands carry little x̄ or ȳ (about a per cent of the totals).
pub fn cie_1931_cmf_fit(lambda_nm: f64) -> DVec3 {
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

/// The CO⁺ comet-tail system's tristimulus values per ion at 1 au: Σ over the 36 bands of
/// [`CO_PLUS_G`] of g·(x̄, ȳ, z̄)(λ) [erg s⁻¹ ion⁻¹] (a line spectrum's XYZ is its lines' powers
/// weighted by the CMFs, CIE 15:2004). Its Y is the photopic share: × 683 lm/W
/// ([`LUMINOUS_EFFICACY_LM_PER_W`]) per 10⁷ erg it is lumens.
pub fn co_plus_xyz() -> DVec3 {
    let mut xyz = DVec3::ZERO;
    for (v1, row) in (0_u32..).zip(CO_PLUS_G.iter()) {
        for (v2, &g) in (0_u32..).zip(row.iter()) {
            xyz += g * cie_1931_cmf_fit(co_plus_band_nm(v1, v2));
        }
    }
    xyz
}

/// The maximum luminous efficacy of radiation K_m = 683 lm/W (the candela's definition, CGPM
/// 1979, kept in the 2019 SI): luminance = K_m ∫ L_e,λ ȳ(λ) dλ.
pub const LUMINOUS_EFFICACY_LM_PER_W: f64 = 683.0;

/// Aravind et al. 2025 (A&A 701, arXiv:2507.21707; read 2026-11-24) Table 4, the CO⁺ ion fluxes
/// of C/2020 F3 (NEOWISE) on 2020-07-24 (r_h 0.66, Δ 0.69 AU; HFOSC on the 2-m HCT): (projected
/// distance from the photocentre [km], CO⁺ (3-0), CO⁺ (2-0) [erg cm⁻² s⁻¹]), each in a 12″
/// aperture along the 1.92″-wide slit ([`ARAVIND_APERTURE_ARCSEC`]), extracted tailward with
/// the slit 22° from the anti-sunward direction (their §2.2).
pub const ARAVIND_CO_PLUS_FLUX: [(f64, f64, f64); 5] = [
    (35_809.0, 3.91e-13, 1.49e-13),
    (53_783.0, 2.48e-13, 1.12e-13),
    (71_688.0, 1.74e-13, 7.71e-14),
    (89_294.0, 1.57e-13, 7.48e-14),
    (103_548.0, 1.48e-13, 6.99e-14),
];
/// Aravind et al.'s apertures [″]: 12″ along the slit × the slit's 1.92″ width (our reading of
/// their §2.2: a long-slit extraction box; a 12″ circle would make the luminance 4.9 × lower).
pub const ARAVIND_APERTURE_ARCSEC: [f64; 2] = [12.0, 1.92];
/// The angle between their slit and the anti-sunward direction [°] (Aravind et al. §2.2).
pub const ARAVIND_SLIT_TO_TAIL_DEG: f64 = 22.0;
/// Their ion night: r_h, Δ and the Earth–Sun distance [AU] on 2020-07-24 (the last from the
/// Earth's orbit on that date, `ion_light_is_derived` recomputes it).
pub const ARAVIND_ION_NIGHT_AU: [f64; 3] = [0.66, 0.69, 1.0158];

/// The CO⁺ luminance [cd/m²] in Aravind et al.'s aperture `k` ([`ARAVIND_CO_PLUS_FLUX`]): their
/// two bands' surface brightness (F₃₀ + F₂₀)/Ω [10⁻³ W m⁻² sr⁻¹ per erg cm⁻² s⁻¹ sr⁻¹], scaled to
/// every band of the system by fluorescence equilibrium, Σg / (g₃₀ + g₂₀), and to luminance by
/// 683 lm/W × the system's ȳ-weighted share ([`co_plus_xyz`]):
/// L = 683 × (F₃₀ + F₂₀)/Ω × Y_CO⁺ / (g₃₀ + g₂₀). The two measured bands carry 0.8 % of that
/// luminance: the system's light that the eye sees is in its green–yellow bands (v″ = 1–3, 504–
/// 590 nm), where ȳ is 100 × larger than at 400–427 nm.
pub fn aravind_co_plus_luminance(k: usize) -> f64 {
    let Some(&(_, f30, f20)) = ARAVIND_CO_PLUS_FLUX.get(k) else {
        return 0.0;
    };
    let arcsec = std::f64::consts::PI / (180.0 * 3600.0);
    let omega = ARAVIND_APERTURE_ARCSEC[0] * ARAVIND_APERTURE_ARCSEC[1] * arcsec * arcsec;
    // erg cm⁻² s⁻¹ → W m⁻²: 10⁻⁷ J/erg × 10⁴ cm²/m².
    let radiance = (f30 + f20) * 1.0e-3 / omega;
    let g_measured = CO_PLUS_G[3][0] + CO_PLUS_G[2][0];
    LUMINOUS_EFFICACY_LM_PER_W * radiance * co_plus_xyz().y / g_measured
}

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

/// The ion tail's colour, linear Rec. 709 with luminance 1: the CO⁺ comet-tail system's
/// band-summed tristimulus values ([`co_plus_xyz`], C1L fix round 2, lead ruling C1L-R2) through
/// the CIE path ([`line_colour`]). A blue: the energy is in the 400–470 nm bands, the luminance in
/// the 504–590 nm ones. C1's single 427 nm chromaticity, mixed 50 % with white, read violet.
pub fn ion_colour() -> Vec3 {
    line_colour(co_plus_xyz().to_array())
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

/// [`splat_axes`]'s body for one float type: written once, so the f64 CPU twin and the f32
/// copy the range tests run ([`splat_axes_f32`], the shader's arithmetic) do the same operations
/// in the same order as `splat_axes` in `comet.wgsl`.
macro_rules! splat_axes_in {
    ($(#[$attr:meta])* $vis:vis fn $name:ident($f:ty, $v3:ty, $m3:ty)) => {
        $(#[$attr])*
        $vis fn $name(factor: $m3, right: $v3, up: $v3, floor_m: $f) -> ($f, $f, $f) {
            let v = [factor.x_axis, factor.y_axis, factor.z_axis]
                .map(|c| (right.dot(c), up.dot(c)));
            // C1L fix round 1 (review I2): every length is divided by the largest projected
            // component or the floor, k, before it is squared, so no square exceeds 9 and the
            // f32 shader cannot overflow (squaring metres overflowed f32 above σ₁ ≈ 6.1 × 10⁹ m,
            // and the splat vanished); σ₁ and σ₂ are scaled back by k.
            let k = v.iter().fold(floor_m.abs(), |a, c| a.max(c.0.abs()).max(c.1.abs()));
            if !(k > 0.0) {
                return (0.0, 0.0, 0.0);
            }
            let u = v.map(|c| (c.0 / k, c.1 / k));
            let f = floor_m / k;
            let f2 = f * f;
            let p = u[0].0 * u[0].0 + u[1].0 * u[1].0 + u[2].0 * u[2].0;
            let q = u[0].0 * u[0].1 + u[1].0 * u[1].1 + u[2].0 * u[2].1;
            let r = u[0].1 * u[0].1 + u[1].1 * u[1].1 + u[2].1 * u[2].1;
            let c01 = u[0].0 * u[1].1 - u[0].1 * u[1].0;
            let c02 = u[0].0 * u[2].1 - u[0].1 * u[2].0;
            let c12 = u[1].0 * u[2].1 - u[1].1 * u[2].0;
            let det = c01 * c01 + c02 * c02 + c12 * c12 + f2 * (p + r) + f2 * f2;
            let m = 0.5 * (p + r) + f2;
            let l1 = m + (m * m - det).max(0.0).sqrt();
            let l2 = det / l1;
            (l1.sqrt() * k, l2.sqrt() * k, 0.5 * (2.0 * q).atan2(p - r))
        }
    };
}

splat_axes_in!(
    /// A splat's screen-plane ellipse (C1L): the 3D covariance L Lᵀ (`factor` = L, [`cov_factor`])
    /// projected onto the plane spanned by the unit vectors `right` and `up` (the sprite's plane,
    /// facing the camera), plus `floor_m`² on both axes (the smallest σ drawn, [`SPRITE_MIN_PX`]
    /// pixels). With uᵢ = (right·Lᵢ, up·Lᵢ)/k the columns projected and divided by k, the
    /// largest |component| or the floor (so nothing squared exceeds 9: no f32 overflow at any
    /// size, fix round 1), the 2D covariance C/k² has trace Σ|uᵢ|² + 2f² and determinant
    /// Σ_{i<j}(uᵢ × uⱼ)² + f²Σ|uᵢ|² + f⁴, f = floor/k (Cauchy–Binet: a sum of squares, so an
    /// elongated splat's minor axis does not cancel in f32); σ₁² = k²(m + √(m² − det)),
    /// m = trace/2, σ₂² = k⁴det/σ₁². Returns (σ₁, σ₂, φ) [m, m, rad]: σ₁ ≥ σ₂ along the axes
    /// cos φ·right + sin φ·up and −sin φ·right + cos φ·up, φ = ½ atan2(2C₁₂, C₁₁ − C₂₂); (0, 0, 0)
    /// for a zero covariance and floor. CPU twin of the shader's `splat_axes` (same order of
    /// operations); probed on the GPU for σ from 10⁶ to 10¹² m.
    pub fn splat_axes(f64, DVec3, DMat3)
);

splat_axes_in!(
    /// [`splat_axes`] in f32, as the shader computes it (the range tests: it must stay finite
    /// and agree with the f64 twin for splats of 10⁸–10¹² m, review I2).
    #[cfg(test)]
    pub(crate) fn splat_axes_f32(f32, glam::Vec3, glam::Mat3)
);

/// The C1L shader's `splat_axes` before fix round 1, in f32 (metres squared unscaled): kept for
/// the tests that show where it overflowed (review I2: σ₁ above ≈ 6.1 × 10⁹ m).
#[cfg(test)]
pub(crate) fn splat_axes_unscaled_f32(
    factor: glam::Mat3,
    right: glam::Vec3,
    up: glam::Vec3,
    floor_m: f32,
) -> (f32, f32, f32) {
    let f2 = floor_m * floor_m;
    let u = [factor.x_axis, factor.y_axis, factor.z_axis].map(|c| (right.dot(c), up.dot(c)));
    let p = u[0].0 * u[0].0 + u[1].0 * u[1].0 + u[2].0 * u[2].0;
    let q = u[0].0 * u[0].1 + u[1].0 * u[1].1 + u[2].0 * u[2].1;
    let r = u[0].1 * u[0].1 + u[1].1 * u[1].1 + u[2].1 * u[2].1;
    let cross = |a: (f32, f32), b: (f32, f32)| a.0 * b.1 - a.1 * b.0;
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

/// The camera as `vs_dust` sees it (C1L fix round 1): its right and up axes [root axes]
/// (`camera_right()`, `camera_up()` in `comet.wgsl`) and one pixel's size at unit distance
/// [rad], 2·tan(fov/2)/height (`pixel_size_at(d)/d`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct SpriteCamera {
    pub(crate) right: DVec3,
    pub(crate) up: DVec3,
    pub(crate) pixel_rad: f64,
}

impl SpriteCamera {
    /// `camera` at internal size `viewport`.
    pub(crate) fn new(camera: &CameraView, viewport: UVec2) -> Self {
        let orient = camera.orient.normalize();
        SpriteCamera {
            right: orient * DVec3::X,
            up: orient * DVec3::Y,
            pixel_rad: 2.0 * (0.5 * camera.fov_y_rad).tan() / f64::from(viewport.y.max(1)),
        }
    }
}

/// One dust splat as `vs_dust` draws it (C1L fix round 1; [`splat_draw`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SplatDraw {
    /// The splat's centre relative to the camera [m].
    pub(crate) rel: DVec3,
    /// The ellipse's unit axes on the sprite plane (which faces the camera across `rel`)...
    pub(crate) e1: DVec3,
    pub(crate) e2: DVec3,
    /// ...and their σ [m] after the `SPRITE_MAX_PX` cap.
    pub(crate) s1: f64,
    pub(crate) s2: f64,
    /// The luminous intensity drawn [cd], after the cap.
    pub(crate) drawn_cd: f64,
    /// The splat's light before the cap [cd]: the dust's light production `rate` × its
    /// [`dust_grain_light`].
    pub(crate) light_cd: f64,
}

impl SplatDraw {
    /// The splat's brightest point [cd/m²], its centre: what `vs_dust`'s visibility cull compares
    /// with `GLOW_CULL_PRE` after `scene_pre()` (the tests' twin of that cull).
    #[cfg(test)]
    pub(crate) fn peak_cd_m2(&self) -> f64 {
        self.drawn_cd * sprite_profile(0.0, 1.0) / (self.s1 * self.s2)
    }
}

/// The CPU twin of `vs_dust`'s per-splat maths (C1L fix round 1): grain `g` of a tail whose dust
/// light production is `rate` [cd/s per unit of HG] ([`dust_emission_cd_per_s`]), nucleus at
/// `nucleus_rel` from the camera, the Sun at `sun_rel` from the nucleus [m], seen by `cam`. The
/// sprite plane faces the camera across the splat's line of sight (`facing_right`), the ellipse
/// is [`splat_axes`] with a floor of `SPRITE_MIN_PX` pixels, and each axis wider than
/// `SPRITE_MAX_PX` pixels is cut to it with the light scaled by the same ratio (the surface
/// brightness is kept; the light beyond is lost). `None` with the camera within 1 m of the
/// splat (the pass draws nothing there). The pass's visibility cull ([`SplatDraw::peak_cd_m2`]
/// below `GLOW_CULL_PRE` at the frame's exposure) is left to the caller.
pub(crate) fn splat_draw(
    g: &TailGrain,
    rate: f64,
    nucleus_rel: DVec3,
    sun_rel: DVec3,
    cam: &SpriteCamera,
) -> Option<SplatDraw> {
    let rel = nucleus_rel + g.offset_m;
    let dist = rel.length();
    if !(dist > 1.0) {
        return None;
    }
    let dir = rel / dist;
    let light = rate * dust_grain_light(g.offset_m, g.weight, nucleus_rel, sun_rel);
    // `facing_right` in comet.wgsl.
    let r = dir.cross(cam.up);
    let right = if r.length_squared() < 1e-8 {
        cam.right
    } else {
        r.normalize()
    };
    let up = right.cross(dir);
    let px = dist * cam.pixel_rad;
    let (mut s1, mut s2, phi) = splat_axes(cov_factor(g.cov_m2), right, up, SPRITE_MIN_PX * px);
    let s_max = super::SPRITE_MAX_PX * px;
    let mut drawn = light;
    if s1 > s_max {
        drawn *= s_max / s1;
        s1 = s_max;
    }
    if s2 > s_max {
        drawn *= s_max / s2;
        s2 = s_max;
    }
    let (c, s) = (phi.cos(), phi.sin());
    Some(SplatDraw {
        rel,
        e1: c * right + s * up,
        e2: c * up - s * right,
        s1,
        s2,
        drawn_cd: drawn,
        light_cd: light,
    })
}

/// One dust tail the comet pass draws, seen from the frame's camera (C1L fix round 1, lead
/// ruling C1L-I3): its light and the share of it the pass draws.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DustDrawn {
    /// The comet.
    pub id: ObjectId,
    /// The tail's dust light [cd]: Σ over its splats of the dust's light production × each
    /// splat's [`dust_grain_light`] (the light before `SPRITE_MAX_PX` cuts it).
    pub light_cd: f64,
    /// The share of that light the pass draws, 0–1: a splat wider than `SPRITE_MAX_PX` pixels on
    /// an axis keeps its surface brightness and loses the light beyond, so a tail seen from
    /// close by draws little of its light (row 21: 1P/Halley from 2 × 10⁶ km). The pass's
    /// visibility cull (light below a thousandth of black) is not counted.
    pub drawn_fraction: f64,
    /// The tail's splats.
    pub splats: usize,
}

/// The light of `grains` (a tail's splats) [cd] and the part of it the pass draws [cd], from
/// [`splat_draw`] (arguments as there).
pub(crate) fn dust_light_drawn(
    grains: &[TailGrain],
    rate: f64,
    nucleus_rel: DVec3,
    sun_rel: DVec3,
    cam: &SpriteCamera,
) -> (f64, f64) {
    grains
        .iter()
        .filter_map(|g| splat_draw(g, rate, nucleus_rel, sun_rel, cam))
        .fold((0.0, 0.0), |(l, d), s| (l + s.light_cd, d + s.drawn_cd))
}

/// The dust tails' luminance on a frame, pixel by pixel (C1L fix round 1, lead ruling C1L-b;
/// [`dust_tail_luminance`]).
#[derive(Clone, Debug, Default)]
pub struct DustRaster {
    /// The frame's size [px].
    pub size: UVec2,
    /// Luminance [cd/m²], not pre-exposed, row-major from the top-left pixel: pixel (x, y) is
    /// `luminance[y · width + x]`, sampled at its centre (x + ½, y + ½) as `pixel_dir` and
    /// `measure::Cam::ray` do.
    pub luminance: Vec<f64>,
    /// The tails drawn, their light and the share drawn ([`DustDrawn`]).
    pub tails: Vec<DustDrawn>,
}

impl DustRaster {
    /// The luminance [cd/m²] of pixel (x, y); 0 outside the frame.
    pub fn at(&self, x: u32, y: u32) -> f64 {
        if x < self.size.x && y < self.size.y {
            self.luminance[y as usize * self.size.x as usize + x as usize]
        } else {
            0.0
        }
    }
}

/// The CPU twin of the comet pass's dust tails (C1L fix round 1, lead ruling C1L-b: the paper's
/// "Exposure on paper" adds it to its world light): the luminance [cd/m²] each pixel of a
/// frame of `size` internal pixels gets from the dust splats of `comets` (`Snapshot::comets`)
/// at `epoch`, seen from `camera`.
///
/// - **Which tails:** what a capture's first frame draws: the comets `plan_frame` picks (on
///   screen, active, at most `MAX_TAILED_COMETS`), each tail computed in full at `epoch`.
/// - **Each splat:** [`splat_draw`], so the `SPRITE_MAX_PX` cap and the `SPRITE_MIN_PX` floor
///   act as on the GPU, and its profile [`sprite_profile`] where each pixel's ray meets the
///   splat's plane (the GPU's perspective-correct interpolation over the quad), sampled at the
///   pixel's centre.
/// - **Left out:** the visibility cull (light below a thousandth of black at the frame's
///   exposure), occlusion by bodies and nuclei (the pass is depth-tested), the coma and the ion
///   tail (the paper's own coma twin adds the coma), and f32 rounding.
///
/// Cost: the splats' pixel area (each at most (4 × 64)² pixels); row 20 at 1 920 × 1 080 is
/// about 10⁶ pixel samples.
pub fn dust_tail_luminance(
    comets: &[CometView],
    epoch: Epoch,
    camera: &CameraView,
    size: UVec2,
) -> DustRaster {
    let size = size.max(UVec2::ONE);
    let scene = super::CometScene {
        comets: comets.to_vec(),
        epoch,
        eye_adapted: false,
    };
    let mut manager = TailManager::default();
    let plan = super::plan_frame(
        &scene,
        camera,
        size,
        0.0,
        &mut manager,
        usize::MAX,
        usize::MAX,
    );
    let cam = SpriteCamera::new(camera, size);
    let (w, h) = (size.x as usize, size.y as usize);
    let mut luminance = vec![0.0_f64; w * h];
    let orient = camera.orient.normalize();
    let (cr, cu, cf) = (orient * DVec3::X, orient * DVec3::Y, orient * DVec3::NEG_Z);
    let tan_y = (0.5 * camera.fov_y_rad).tan();
    let tan_x = tan_y * w as f64 / h as f64;
    for draw in &plan.tails {
        let (Some(view), Some(tail)) = (
            comets.iter().find(|c| c.id == draw.id),
            manager.tails.get(&draw.id),
        ) else {
            continue;
        };
        let rate = dust_rate_cd_per_s(view, super::coma::coma_intensity_cd(view));
        for g in &tail.grains {
            let Some(sp) = splat_draw(g, rate, view.nucleus_rel_m, view.sun_rel_m, &cam) else {
                continue;
            };
            if !(sp.drawn_cd > 0.0 && sp.s1 > 0.0 && sp.s2 > 0.0) {
                continue;
            }
            rasterise_splat(&sp, (cr, cu, cf), (tan_x, tan_y), size, &mut luminance);
        }
    }
    DustRaster {
        size,
        luminance,
        tails: plan.dust_drawn,
    }
}

/// Adds splat `sp`'s luminance to `out` (a `size` frame, row-major from the top left) for the
/// camera with axes `axes` = (right, up, forward) [root axes] and half-angle tangents `tan_xy`
/// ([`dust_tail_luminance`]).
fn rasterise_splat(
    sp: &SplatDraw,
    axes: (DVec3, DVec3, DVec3),
    tan_xy: (f64, f64),
    size: UVec2,
    out: &mut [f64],
) {
    let (cr, cu, cf) = axes;
    let (tan_x, tan_y) = tan_xy;
    let (w, h) = (f64::from(size.x), f64::from(size.y));
    let dist = sp.rel.length();
    let n = sp.rel / dist;
    // The quad's corners (±SPRITE_SIGMAS σ on both axes) on screen: their bounding box, or the
    // whole frame when one is behind the camera.
    let corners = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)].map(|(a, b)| {
        sp.rel + sp.e1 * (a * SPRITE_SIGMAS * sp.s1) + sp.e2 * (b * SPRITE_SIGMAS * sp.s2)
    });
    let (mut x0, mut x1, mut y0, mut y1) = (0.0, w, 0.0, h);
    if corners.iter().all(|p| p.dot(cf) > 0.0) {
        let screen = corners.map(|p| {
            let z = p.dot(cf);
            (
                0.5 * (p.dot(cr) / (z * tan_x) + 1.0) * w,
                0.5 * (1.0 - p.dot(cu) / (z * tan_y)) * h,
            )
        });
        x0 = screen.iter().map(|s| s.0).fold(f64::INFINITY, f64::min);
        x1 = screen.iter().map(|s| s.0).fold(f64::NEG_INFINITY, f64::max);
        y0 = screen.iter().map(|s| s.1).fold(f64::INFINITY, f64::min);
        y1 = screen.iter().map(|s| s.1).fold(f64::NEG_INFINITY, f64::max);
    }
    let px0 = (x0 - 0.5).floor().max(0.0) as usize;
    let px1 = ((x1 - 0.5).ceil() + 1.0).clamp(0.0, w) as usize;
    let py0 = (y0 - 0.5).floor().max(0.0) as usize;
    let py1 = ((y1 - 0.5).ceil() + 1.0).clamp(0.0, h) as usize;
    // A pixel's ray is d = forward + a·right + b·up; it meets the plane through the centre
    // facing the camera at t·d, t = dist/(d·n), where its coordinates are (t·d − rel)·eᵢ.
    let dots = |v: DVec3| (cf.dot(v), cr.dot(v), cu.dot(v));
    let (n0, n1, n2) = dots(n);
    let (a0, a1, a2) = dots(sp.e1);
    let (b0, b1, b2) = dots(sp.e2);
    let (r1, r2) = (sp.rel.dot(sp.e1), sp.rel.dot(sp.e2));
    let scale = sp.drawn_cd / (sp.s1 * sp.s2);
    let row = size.x as usize;
    for y in py0..py1 {
        let b = (1.0 - 2.0 * (y as f64 + 0.5) / h) * tan_y;
        for x in px0..px1 {
            let a = (2.0 * (x as f64 + 0.5) / w - 1.0) * tan_x;
            let dn = n0 + a * n1 + b * n2;
            if !(dn > 0.0) {
                continue;
            }
            let t = dist / dn;
            let q1 = (t * (a0 + a * a1 + b * a2) - r1) / sp.s1;
            let q2 = (t * (b0 + a * b1 + b * b2) - r2) / sp.s2;
            let q = q1.hypot(q2);
            if q < SPRITE_SIGMAS {
                out[y * row + x] += scale * sprite_profile(q, 1.0);
            }
        }
    }
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

/// The grains of β rows `rows` (indices 0..16 into [`DUST_PARAMS`]'s β grid) of `conic` at
/// `t_obs`, with dust production ∝ r_h^`exponent` ([`dust_production_exponent`]). Row-major as
/// `finson_probstein` (β-major, ages within a row).
pub(crate) fn dust_rows(
    conic: &Conic,
    exponent: f64,
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
    // Production ∝ r_h^exponent ([`dust_production_exponent`]; plan C1 as amended by lead
    // rulings C1L-I1 and C1L-R1).
    finson_probstein(conic, t_obs, &p, &|r_h| dust_production(r_h, exponent))
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
fn sigma_bound_m(cov: DMat3) -> f64 {
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

/// The dust production law's value at heliocentric distance `r_au`: r_h^`exponent`, the factor
/// `dust_rows` weights the emissions with. The exponent is a [`CometDustLaw`]'s A(0)fρ slope
/// where the comet has one (lead ruling C1L-R1), else −max(K1/2.5 − 2, 0) from its total visual
/// slope K1 ([`dust_production_exponent`]; C1L fix round 1, review I1, lead ruling C1L-I1,
/// 2026-11-24), explained here.
///
/// - **Why −2.** The magnitude law m = M1 + 5 log₁₀ Δ + K1 log₁₀ r_h (SBDB's total-magnitude law;
///   `uni_sim::world::COMET_VISUAL_LAWS` where a comet has a cited one) makes the comet's light
///   ∝ r_h^(−K1/2.5) at a fixed Δ. That light already contains the sunlight on it, E☉ ∝ r_h⁻²:
///   a coma of constant scattering cross-section has K1 = 5. So the cross-section goes as
///   r_h^−(K1/2.5 − 2), and at a constant outflow speed so does the dust production (a steady
///   coma holds production × crossing time; A'Hearn et al. 1984, AJ 89, 579, the relation
///   [`DUST_HEAD_TIME_S`] rests on). Each grain is then lit again by the Sun at its own distance
///   at the time it is seen ([`dust_grain_light`]: (r_h/r)²). C1L used r_h^(−K1/2.5), which
///   counted the r⁻² twice: at row 20 it weighted perihelion's dust 5.3 × today's instead of
///   2.2 × (r_h 0.455 against q 0.295 AU), and drew the tail 5.49 × the coma instead of 3.55 ×
///   (`neowise_dust_tail_is_long_and_curved`).
/// - **Approximate.** K1 is the slope of the comet's TOTAL visual light, which for many comets
///   is dominated by gas (C₂ Swan bands in the V band), not dust; using it for the dust
///   production is the plan's approximation (card note `uni_ui::theme::NOTE_TAILS`). A K1 below
///   5 would make the production grow with r_h; it is held constant there (exponent floored at
///   0). That touches 103 of the 508 SBDB comets in `data/baked/comets.ron` (100 at K1 = 4.5, 2 at
///   4.8, 1 at 4.0; C1L review r1), whose unfloored exponents would be +0.2 to +0.4
///   (`dust_production_floors_k1_below_5`).
pub(crate) fn dust_production(r_au: f64, exponent: f64) -> f64 {
    r_au.powf(exponent)
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
    /// The production's r_h exponent ([`dust_production_exponent`]).
    exponent: f64,
    sun_rel: DVec3,
    grains: Vec<DustGrain>,
    rows_done: u32,
}

/// The dust tails' CPU side (no GPU objects, so it is tested on its own): per-comet tails keyed
/// by id, the schedule's refresh times, the refresh in progress, and each drawn tail's last
/// [`DustDrawn`].
#[derive(Debug, Default)]
pub(crate) struct TailManager {
    pub(crate) tails: HashMap<ObjectId, TailState>,
    pub(crate) last_refresh: HashMap<ObjectId, f64>,
    pub(crate) drawn: HashMap<ObjectId, DrawnCache>,
    job: Option<TailJob>,
    next_version: u64,
}

/// A tail's last [`DustDrawn`], the dust version and wall time [s] it was computed at
/// (`plan_frame` recomputes it at most every [`DRAWN_REFRESH_WALL_S`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct DrawnCache {
    pub(crate) version: u64,
    pub(crate) wall_s: f64,
    pub(crate) drawn: DustDrawn,
}

/// A drawn tail's [`DustDrawn`] is recomputed once it is this old [s of wall time] or its tail
/// was refreshed (C1L fix round 1; the tail's own wall-time rule, `TAIL_REFRESH_WALL_S`).
pub(crate) const DRAWN_REFRESH_WALL_S: f64 = TAIL_REFRESH_WALL_S;

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
                    exponent: dust_production_exponent(view.spkid, k1),
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
                .extend(dust_rows(&job.conic, job.exponent, job.epoch, range));
            job.rows_done += rows;
            work.solves += (rows as usize + 1) * n_age;
            if let Some(done) = self.job.take_if(|j| j.rows_done >= DUST_PARAMS.n_beta) {
                let production_now = dust_production(done.sun_rel.length() / AU, done.exponent);
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
        self.drawn.retain(|id, _| tails.contains_key(id));
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

    /// The production exponent of a comet with K1 = 10 and no dust law: −max(10/2.5 − 2, 0).
    const K1_10_EXPONENT: f64 = -2.0;

    /// `view`'s dust tail at its conic epoch, as `TailManager` computes it (the exponent from
    /// its dust law or K1, merged splats).
    fn tail_of(view: &CometView) -> Vec<TailGrain> {
        let exponent = dust_production_exponent(view.spkid, view.k1.unwrap_or(0.0));
        tail_grains(
            &dust_rows(
                &view.conic,
                exponent,
                view.conic_epoch,
                0..DUST_PARAMS.n_beta,
            ),
            dust_production(view.r_au, exponent),
        )
    }

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
            let grains = dust_rows(&conic, K1_10_EXPONENT, t, 0..DUST_PARAMS.n_beta);
            assert_eq!(grains.len(), GRAINS_PER_TAIL);
            let mean =
                grains.iter().map(|g| g.pos_rel_nucleus_m).sum::<DVec3>() / grains.len() as f64;
            assert!(
                mean.dot(v) < 0.0,
                "mean offset {mean} m leads the motion {v}"
            );
            assert!(mean.dot(r) > 0.0, "mean offset {mean} m points sunward");
            let now = dust_production(r.length() / AU, K1_10_EXPONENT);
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
        let grains = dust_rows(&conic, K1_10_EXPONENT, t, 0..DUST_PARAMS.n_beta);
        let now = dust_production(0.8, K1_10_EXPONENT);
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
            let g = dust_rows(&conic_t.0, K1_10_EXPONENT, conic_t.1, 0..DUST_PARAMS.n_beta);
            let now = dust_production(
                heliocentric_state_icrs(&conic_t.0, conic_t.1).0.length() / AU,
                K1_10_EXPONENT,
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
    /// covariance seen edge-on still gets the floor on its minor axis. Fix round 1 (review I2):
    /// also for splats of 10⁸–10¹² m, with the camera inside a tail (an 802 m floor under a
    /// 10¹⁰ m splat), where the f32 copy (`splat_axes_f32`, the shader's arithmetic) stays finite
    /// and within 10⁻³ of the f64 twin while the C1L f32 maths overflowed.
    #[test]
    fn splat_axes_rebuild_the_projected_covariance() {
        let right = DVec3::new(0.6, 0.8, 0.0);
        let up = DVec3::new(-0.48, 0.36, 0.8);
        assert!(right.dot(up).abs() < 1e-12);
        let a = DVec3::new(3.0e9, -1.0e9, 2.0e8);
        let b = DVec3::new(2.0e8, 5.0e8, -7.0e8);
        let spread = |s: f64| DMat3::from_diagonal(DVec3::splat(s * s));
        let mut f32_report = Vec::new();
        for (cov, floor) in [
            (outer(a) + outer(b), 1.0e6),
            (outer(a) * 0.25 + spread(2.0e8), 3.0e7),
            (outer(right * 2.0e9), 1.0e7),
            (DMat3::ZERO, 2.0e6),
            // Fix round 1: 10⁸–10¹² m.
            (outer(a * 300.0) + outer(b * 30.0) + spread(3.3e9), 8.0e9),
            (outer(a * 30.0) * 0.25 + spread(5.2e7), 802.0),
            (outer(up * 1.0e12) + spread(1.0e8), 1.0e3),
            (spread(1.0e12), 1.0e9),
            (outer(b * 100.0) + spread(1.0e8), 802.0),
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
            // √(m² − det) splits a round splat's axes only to √ε ≈ 10⁻⁸ of σ² in f64.
            let tol = 1e-7 * (s1 * s1);
            assert!(
                (p - want[0]).abs() <= tol
                    && (q - want[1]).abs() <= tol
                    && (r - want[2]).abs() <= tol,
                "σ₁ {s1:e} σ₂ {s2:e}: rebuilt {:?} vs {want:?} (tol {tol:e})",
                [p, q, r]
            );
            if s1 == 0.0 {
                continue;
            }
            // The shader's f32 arithmetic, before and after fix round 1.
            let l = cov_factor(cov);
            let l32 =
                glam::Mat3::from_cols(l.x_axis.as_vec3(), l.y_axis.as_vec3(), l.z_axis.as_vec3());
            let (r32, u32_, f32_) = (right.as_vec3(), up.as_vec3(), floor as f32);
            let new = splat_axes_f32(l32, r32, u32_, f32_);
            let old = splat_axes_unscaled_f32(l32, r32, u32_, f32_);
            let err = (f64::from(new.0) / s1 - 1.0)
                .abs()
                .max((f64::from(new.1) / s2 - 1.0).abs());
            assert!(err < 1e-3, "σ₁ {s1:e}: f32 {new:?} vs {s1:e}, {s2:e}");
            f32_report.push(format!(
                "σ₁ {s1:.1e} σ₂ {s2:.1e}: f32 rel {err:.0e}, C1L f32 {}",
                if old.0.is_finite() && old.1.is_finite() && old.1 > 0.0 {
                    "ok"
                } else {
                    "overflowed"
                }
            ));
            if s1 > 1.0e10 {
                assert!(!(old.0.is_finite() && old.1.is_finite()), "{old:?}");
            }
        }
        println!("splat_axes in f32: {}", f32_report.join("; "));
        let (s1, s2, _) = splat_axes(cov_factor(outer(up * 5.0e8)), right, up, 1.0e6);
        assert!((s1 / 5.0e8 - 1.0).abs() < 1e-5 && (s2 / 1.0e6 - 1.0).abs() < 1e-6);
        assert_eq!(splat_axes(DMat3::ZERO, right, up, 0.0), (0.0, 0.0, 0.0));
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

    /// A camera looking from `cam_helio` [m] at `target_rel` (target − camera) with `up_hint`
    /// up, at fov `fov_deg` (the bookmarks' `look_at` and `up`).
    fn look_at(target_rel: DVec3, up_hint: DVec3, fov_deg: f64) -> CameraView {
        let fwd = target_rel.normalize();
        let right = fwd.cross(up_hint).normalize();
        let up = right.cross(fwd);
        CameraView {
            pos: uni_core::UniversePos::from_meters(DVec3::ZERO),
            orient: glam::DQuat::from_mat3(&DMat3::from_cols(right, up, -fwd)),
            fov_y_rad: fov_deg.to_radians(),
            velocity_mps: DVec3::ZERO,
        }
    }

    /// C/2020 F3 (NEOWISE)'s SBDB orbit (`comets.ron`: q, e, i, Ω, ω, T_p).
    fn neowise_conic() -> Conic {
        conic_from_perihelion(
            0.294_651_249_380_919_6,
            0.999_178_026_253_129_2,
            128.937_502_759_480_9,
            61.010_428_185_369_88,
            37.278_658_448_125_7,
            2_459_034.178_898_044_4,
        )
    }

    /// C/2020 F3 (NEOWISE) at row 20 (`comet-neowise-2020`: 2020-07-15 03:00 UTC from the
    /// Earth–Moon barycentre, fov 20°, up = ICRS +z, 1 920 × 1 080): its view, the camera, its
    /// heliocentric velocity, and its coma's light [cd].
    fn neowise_row_20() -> (CometView, CameraView, DVec3, f64) {
        use uni_physics::standish::{Planet, heliocentric_state_icrs as planet_state};
        let conic = neowise_conic();
        let t = Epoch::from_unix_utc(1_594_782_000.0);
        let (_, vel) = heliocentric_state_icrs(&conic, t);
        let (earth, _) = planet_state(Planet::EmBary, t);
        let law = uni_sim::world::comet_visual_law(NEOWISE_SPKID).expect("NEOWISE's law");
        let mut view = comet_on(1, conic, t, earth, law.m1(), law.k1);
        // The real SPK-ID, so its dust law ([`COMET_DUST_LAWS`]) applies.
        view.spkid = NEOWISE_SPKID;
        let camera = look_at(view.nucleus_rel_m, DVec3::Z, 20.0);
        let coma = crate::comet::coma::coma_intensity_cd(&view);
        (view, camera, vel, coma)
    }

    /// The CMF fit (Wyman, Sloan & Shirley 2013) against the CIE 2018 table values this module
    /// holds: within 5 % at 427 nm, ȳ within 1 % and z̄ within 6 % at 516 nm (x̄, 0.035 there near
    /// its minimum, within 30 %).
    #[test]
    fn cmf_fit_matches_the_cie_table() {
        let at427 = cie_1931_cmf_fit(427.0);
        let at516 = cie_1931_cmf_fit(516.0);
        let rel = |a: f64, b: f64| (a / b - 1.0).abs();
        let c516 = crate::comet::coma::C2_SWAN_CMF;
        println!("CMF fit 427 nm {at427} vs {CO_PLUS_CMF:?}; 516 nm {at516} vs {c516:?}");
        for k in 0..3 {
            assert!(rel(at427[k], CO_PLUS_CMF[k]) < 0.05, "427 nm [{k}]");
        }
        assert!(rel(at516.y, c516[1]) < 0.01 && rel(at516.z, c516[2]) < 0.06);
        assert!(rel(at516.x, c516[0]) < 0.30);
    }

    /// Lead ruling C1L-R1: NEOWISE's dust law's valid range is Biver et al. 2022's Fig. 19 span,
    /// 2020-07-13 to 2020-08-14 (from the start of the first day to the end of the last), on the
    /// SBDB orbit, after perihelion.
    #[test]
    fn neowise_dust_law_range_is_the_papers() {
        let conic = neowise_conic();
        let r_at = |iso: &str| {
            let t = Epoch::parse_iso_utc(iso).expect("date");
            heliocentric_state_icrs(&conic, t).0.length() / AU
        };
        let (from, to) = (r_at("2020-07-13T00:00:00Z"), r_at("2020-08-15T00:00:00Z"));
        println!("Biver et al. 2022 Fig. 19 span: r_h {from:.4}–{to:.4} AU");
        let law = comet_dust_law(NEOWISE_SPKID).expect("NEOWISE's dust law");
        assert!((law.r_valid_au.0 - from).abs() < 1e-3 && (law.r_valid_au.1 - to).abs() < 1e-3);
        assert!(law.covers(0.455, true) && !law.covers(0.455, false) && !law.covers(0.3, true));
        assert_eq!(law.afrho_slope, -4.0);
        assert!((law.afrho0_m(law.anchor_r_au) - NEOWISE_AFRHO0_M).abs() < 1e-12);
        assert!((law.afrho0_m(0.5 * law.anchor_r_au) / NEOWISE_AFRHO0_M - 16.0).abs() < 1e-12);
    }

    /// Lead ruling C1L-R1: at the anchor (NEOWISE on 2020-07-22, r_h 0.63 AU, Δ 0.69 AU) the dust
    /// law's light production equals the coma's light / τ, the calibration the law-less comets
    /// keep; away from it the law's r_h⁻⁴ × E☉ ∝ r_h⁻⁶ replaces the coma's r_h^−3.85 (row 20,
    /// r_h 0.455 AU: × 2.0).
    #[test]
    fn dust_law_matches_tau_at_the_anchor() {
        let visual = uni_sim::world::comet_visual_law(NEOWISE_SPKID).expect("NEOWISE's law");
        let [r_au, delta_au, _] = NEOWISE_CALIBRATION_AU;
        let at = |r: f64, d: f64| {
            let mut v = static_view(
                9,
                DVec3::new(0.0, 0.0, -d * AU),
                DVec3::new(-r * AU, 0.0, 0.0),
                Some(visual.m1()),
                Some(visual.k1),
            );
            v.spkid = NEOWISE_SPKID;
            let coma = crate::comet::coma::comet_total_intensity_cd(visual.m1(), visual.k1, r, d);
            (dust_rate_cd_per_s(&v, coma), coma / DUST_HEAD_TIME_S)
        };
        let (law, tau) = at(r_au, delta_au);
        assert!((law / tau - 1.0).abs() < 1e-3, "{law:e} vs {tau:e} cd/s");
        let (law, tau) = at(0.455, 0.784);
        let want = (0.455_f64 / r_au).powf(-6.0 + visual.k1 / 2.5);
        println!(
            "row 20: the dust law's rate is {:.3} × the coma's / τ",
            law / tau
        );
        assert!((law / tau / want - 1.0).abs() < 1e-3);
        // A comet without a dust law keeps the coma's / τ.
        let v = static_view(9, DVec3::Z * AU, DVec3::X * AU, Some(8.0), Some(10.0));
        assert_eq!(dust_rate_cd_per_s(&v, 1.0e15), 1.0e15 / DUST_HEAD_TIME_S);
    }

    /// Review r1 minor 4: the production exponent is −max(K1/2.5 − 2, 0) without a dust law, so
    /// K1 ≤ 5 gives a constant production (the 103 SBDB comets at K1 4.0–4.8), and a dust law's
    /// slope replaces it.
    #[test]
    fn dust_production_floors_k1_below_5() {
        for k1 in [0.0, 4.0, 4.5, 4.8, 5.0] {
            let e = dust_production_exponent(1, k1);
            assert_eq!(e, 0.0, "K1 {k1}");
            assert_eq!(dust_production(0.3, e), 1.0);
        }
        assert!((dust_production_exponent(1, 9.63) + 1.852).abs() < 1e-12);
        assert_eq!(dust_production_exponent(1, 10.0), K1_10_EXPONENT);
        assert_eq!(dust_production_exponent(NEOWISE_SPKID, 9.63), -4.0);
        assert!((dust_production(0.5, -4.0) - 16.0).abs() < 1e-12);
    }

    /// Lead ruling C1L-R2: Aravind et al.'s CO⁺ fluxes as luminance, and the ion light they imply.
    /// - The vibrational fit places Opitom et al.'s 15 bands within 1.5 nm.
    /// - The two measured bands (3-0, 2-0) carry < 1 % of the system's luminance (Rousselot et
    ///   al.'s g and the CIE ȳ), so the outermost aperture (1.04 × 10⁵ km) is 1.33 × 10⁻⁴ cd/m².
    /// - The colour is a blue (B > G > R; R:G 0.38), not C1's violet (R > G).
    /// - The ribbon calibrated there, with its present profile, carries 0.34 × the coma's light
    ///   (C1's v0: 0.2), which puts row 20's line at 5.6, 4.2 and 3.0 × 10⁻⁴ cd/m² at 1°, 5° and
    ///   10°: the ruling's expected 10⁻⁶–10⁻⁵ is not reached (the fix round 2 report). Prints the
    ///   numbers; `ION_TAIL_LIGHT` stays C1's 0.2 until the lead rules.
    #[test]
    fn ion_light_from_aravind() {
        use uni_physics::standish::{Planet, heliocentric_state_icrs as planet_state};
        let c = co_plus_vibrational_fit();
        let worst = CO_PLUS_BANDS_NM
            .iter()
            .map(|&(a, b, nm)| {
                let (a, b) = (f64::from(a), f64::from(b));
                (1.0e7 / (c[0] + c[1] * a - c[2] * a * (a + 1.0) - c[3] * b + c[4] * b * (b + 1.0))
                    - nm)
                    .abs()
            })
            .fold(0.0, f64::max);
        let xyz = co_plus_xyz();
        let g2 = CO_PLUS_G[3][0] + CO_PLUS_G[2][0];
        let two_band = (CO_PLUS_G[3][0] * cie_1931_cmf_fit(400.0).y
            + CO_PLUS_G[2][0] * cie_1931_cmf_fit(425.0).y)
            / xyz.y;
        let col = ion_colour();
        assert!(worst < 1.5, "{worst} nm");
        assert!(two_band < 0.01, "{two_band}");
        assert!(
            col.z > col.y && col.y > col.x && col.x / col.y < 0.5,
            "{col}"
        );
        println!(
            "CO+ fit {c:?}, worst residual {worst:.2} nm; Y/(g30+g20) {:.4}; the two measured \
             bands carry {:.2} % of the luminance; colour {col}, xy ({:.3}, {:.3})",
            xyz.y / g2,
            100.0 * two_band,
            xyz.x / xyz.element_sum(),
            xyz.y / xyz.element_sum()
        );
        for k in 0..5 {
            println!(
                "  ρ {:.0} km: {:.3e} cd/m²",
                ARAVIND_CO_PLUS_FLUX[k].0,
                aravind_co_plus_luminance(k)
            );
        }
        assert!((aravind_co_plus_luminance(4) / 1.328e-4 - 1.0).abs() < 0.01);
        assert_eq!(aravind_co_plus_luminance(5), 0.0);
        let t = Epoch::parse_iso_utc("2020-07-24T21:00:00Z").expect("date");
        let (earth, _) = planet_state(Planet::EmBary, t);
        let (nuc, _) = heliocentric_state_icrs(&neowise_conic(), t);
        println!(
            "2020-07-24: r_h {:.4}, Δ {:.4}, Earth–Sun {:.4} AU",
            nuc.length() / AU,
            (nuc - earth).length() / AU,
            earth.length() / AU
        );
        assert!((earth.length() / AU - ARAVIND_ION_NIGHT_AU[2]).abs() < 5e-4);
        let [r, d, e] = ARAVIND_ION_NIGHT_AU;
        let cos_a = (r * r + d * d - e * e) / (2.0 * r * d);
        let sin_psi = (1.0 - cos_a * cos_a).sqrt();
        let len = ion_tail_length_m(r);
        let (rho, _, _) = ARAVIND_CO_PLUS_FLUX[4];
        let a = ARAVIND_SLIT_TO_TAIL_DEG.to_radians();
        let (s_m, x_m) = (rho * 1e3 * a.cos() / sin_psi, rho * 1e3 * a.sin());
        let per_cd = ion_tail_brightness(s_m, x_m, len, 1.0, sin_psi);
        let i_ion = aravind_co_plus_luminance(4) / per_cd;
        let law = uni_sim::world::comet_visual_law(NEOWISE_SPKID).expect("law");
        let coma = crate::comet::coma::comet_total_intensity_cd(law.m1(), law.k1, r, d);
        let ratio = i_ion / coma;
        assert!((0.30..0.38).contains(&ratio), "{ratio}");
        println!(
            "ion night: sin ψ {sin_psi:.4}; I_ion {i_ion:.3e} cd = {ratio:.4} × the coma ({coma:.3e} cd)"
        );
        // Row 20.
        let (view, camera, vel, coma20) = neowise_row_20();
        let axis = ion_tail_dir(view.sun_rel_m, vel);
        let len20 = ion_tail_length_m(view.r_au);
        let pixel_rad = 2.0 * (0.5 * camera.fov_y_rad).tan() / 1080.0;
        for deg in [1.0_f64, 5.0, 10.0] {
            let angle = |s: f64| angle_deg(view.nucleus_rel_m, view.nucleus_rel_m + axis * s);
            let (mut lo, mut hi) = (0.0, len20);
            for _ in 0..100 {
                let mid = 0.5 * (lo + hi);
                if angle(mid) < deg { lo = mid } else { hi = mid }
            }
            let p = view.nucleus_rel_m + axis * lo;
            let sp = axis.cross(p.normalize()).length();
            let sigma = ion_drawn_sigma_m(p.length() * pixel_rad);
            let at =
                |light: f64| ion_tail_drawn_brightness(lo, 0.0, len20, light * coma20, sp, sigma);
            println!(
                "  row 20 at {deg}°: s {:.3e} m, sin ψ {sp:.3}; ion line now (0.2) {:.3e}, derived ({ratio:.3}) {:.3e} cd/m²",
                lo,
                at(ION_TAIL_LIGHT),
                at(ratio)
            );
        }
    }

    /// C1L (owner: NEOWISE's dust tail was short and not visibly curved): at row 20's geometry
    /// ([`neowise_row_20`]) the pub CPU twin of the dust pass ([`dust_tail_luminance`], fix
    /// round 1: the rasteriser this test had inside, now what the paper uses) gives a dust tail
    /// brighter than 10⁻⁴ cd/m² (22.6 mag/arcsec², about the natural night sky) at 3° from the
    /// nucleus, whose centre line's direction from the nucleus turns by more than 4° between 1°
    /// and 4° (a tangent turn of more than 8°: curved, not a straight streak), on the side of the
    /// ion tail away from the comet's motion. Prints the tail's light, the share the
    /// `SPRITE_MAX_PX` cap draws, the light the C1L shader's f32 `splat_axes` dropped (review
    /// I2), and the peak luminance and centre line by radius.
    #[test]
    fn neowise_dust_tail_is_long_and_curved() {
        let (view, camera, vel, coma) = neowise_row_20();
        let size = UVec2::new(1920, 1080);
        let clock = std::time::Instant::now();
        let raster =
            dust_tail_luminance(std::slice::from_ref(&view), view.conic_epoch, &camera, size);
        let raster_ms = clock.elapsed().as_secs_f64() * 1e3;
        assert_eq!(raster.tails.len(), 1);
        let tail = raster.tails[0];
        // The light the C1L shader dropped: splats whose f32 `splat_axes` overflowed (review I2).
        let cam = SpriteCamera::new(&camera, size);
        let grains = tail_of(&view);
        let rate = dust_rate_cd_per_s(&view, coma);
        // The share of the emission weight emitted where the dust law is extrapolated (lead
        // ruling C1L-R1: outside Biver et al.'s post-perihelion 0.410–1.115 AU).
        let law = comet_dust_law(NEOWISE_SPKID).expect("NEOWISE's dust law");
        let raw = dust_rows(
            &view.conic,
            law.afrho_slope,
            view.conic_epoch,
            0..DUST_PARAMS.n_beta,
        );
        let (mut w_all, mut w_extra) = (0.0, 0.0);
        for g in &raw {
            let t_e = view.conic_epoch.add_seconds(-g.age_s);
            let r_e = heliocentric_state_icrs(&view.conic, t_e).0.length() / AU;
            let outbound = t_e.seconds_since(view.conic.tp) > 0.0;
            w_all += g.weight;
            if !law.covers(r_e, outbound) {
                w_extra += g.weight;
            }
        }
        println!(
            "row 20: {:.1} % of the emission weight comes from where the dust law is \
             extrapolated (perihelion to 13 July, and before perihelion)",
            100.0 * w_extra / w_all
        );
        // The drawn share alone (`dust_light_drawn`, what `plan_frame` adds per tail), averaged.
        let clock = std::time::Instant::now();
        let mut share = 0.0;
        for _ in 0..200 {
            let (l, d) = dust_light_drawn(&grains, rate, view.nucleus_rel_m, view.sun_rel_m, &cam);
            share += d / l / 200.0;
        }
        let share_ms = clock.elapsed().as_secs_f64() * 1e3 / 200.0;
        assert!(
            (share - tail.drawn_fraction).abs() < 1e-9,
            "{share} vs {tail:?}"
        );
        println!("row 20: the drawn share alone {share_ms:.4} ms per tail (this build)");
        let (mut overflow_light, mut overflow_n) = (0.0, 0);
        // The quads' fill [px] that pass the GPU's visibility cull at the capture's auto
        // exposure (EV −5.29, C1L report; pre = 1/(1.2·2^EV), CLAUDE.md §2), with and without
        // the splats the C1L f32 maths dropped.
        let pre = 1.0 / (1.2 * 2.0_f64.powf(-5.29));
        let (mut fill_all, mut fill_dropped) = (0.0, 0.0);
        for g in &grains {
            let Some(sp) = splat_draw(g, rate, view.nucleus_rel_m, view.sun_rel_m, &cam) else {
                continue;
            };
            let dir = sp.rel.normalize();
            let r = dir.cross(cam.up).normalize();
            let u = r.cross(dir);
            let l = cov_factor(g.cov_m2);
            let l32 =
                glam::Mat3::from_cols(l.x_axis.as_vec3(), l.y_axis.as_vec3(), l.z_axis.as_vec3());
            let floor = (SPRITE_MIN_PX * sp.rel.length() * cam.pixel_rad) as f32;
            let old = splat_axes_unscaled_f32(l32, r.as_vec3(), u.as_vec3(), floor);
            let px = sp.rel.length() * cam.pixel_rad;
            let fill = if sp.peak_cd_m2() * pre >= crate::comet::GLOW_CULL_PRE {
                (2.0 * SPRITE_SIGMAS).powi(2) * sp.s1 * sp.s2 / (px * px)
            } else {
                0.0
            };
            fill_all += fill;
            if !(old.0.is_finite() && old.1.is_finite() && old.1 > 0.0) {
                overflow_light += sp.drawn_cd;
                overflow_n += 1;
                fill_dropped += fill;
            }
        }
        println!(
            "row 20 quad fill past the cull at EV −5.29: {:.2} Mpx with the fix round 1 maths, \
             {:.2} Mpx with the C1L f32 maths (which dropped {:.2} Mpx)",
            fill_all * 1e-6,
            (fill_all - fill_dropped) * 1e-6,
            fill_dropped * 1e-6
        );
        // On circles about the nucleus (the frame's centre), by radius in degrees: the brightest
        // pixel, and the tail's centre line (the luminance-weighted mean direction).
        let f = 540.0 / 10.0_f64.to_radians().tan();
        let px_per_deg = f * 1.0_f64.to_radians().tan();
        let (cx, cy) = (960.0, 540.0);
        let right = camera.orient * DVec3::X;
        let up = camera.orient * DVec3::Y;
        let angle = |v: DVec3| v.dot(up).atan2(v.dot(right)).to_degrees();
        let circle = |deg: f64| {
            let r = deg * px_per_deg;
            let (mut peak, mut sx, mut sy) = (0.0_f64, 0.0, 0.0);
            for a in 0..1440 {
                let az = (f64::from(a) * 0.25).to_radians();
                let (x, y) = (cx + r * az.cos(), cy - r * az.sin());
                if x >= 0.0 && y >= 0.0 {
                    let l = raster.at(x as u32, y as u32);
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
        let ion = angle(ion_tail_dir(view.sun_rel_m, vel));
        let motion = angle(vel);
        let lit = raster.luminance.iter().filter(|&&l| l >= 1e-3).count();
        println!(
            "row 20: {} splats, tail {:.3e} cd = {:.2} × the coma, drawn {:.2} % \
             ({:.2} % cut by SPRITE_MAX_PX); C1L f32 maths dropped {overflow_n} splats, \
             {:.2} % of the drawn light; {lit} px ≥ 10⁻³ cd/m²; ion {ion:.1}°, motion \
             {motion:.1}°; raster {raster_ms:.0} ms (test build); by radius (deg, peak cd/m², \
             centre line °):",
            tail.splats,
            tail.light_cd,
            tail.light_cd / coma,
            100.0 * tail.drawn_fraction,
            100.0 * (1.0 - tail.drawn_fraction),
            100.0 * overflow_light / (tail.light_cd * tail.drawn_fraction),
        );
        for r in &rows {
            println!("  {:.1}°  {:.3e}  {:.1}°", r.0, r.1, r.2);
        }
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
        // Row 20 keeps most of its light (the far tail's widest splats are cut).
        assert!(
            tail.drawn_fraction > crate::comet::DUST_DRAWN_NOTE_BELOW,
            "{tail:?}"
        );
    }

    /// Fix round 1 (review I2 lets the far tail's wide splats draw again): row 20's dust quad
    /// fill [Mpx] past `vs_dust`'s visibility cull at the capture's auto exposure, for the cull
    /// at 1, 10 and 100 × `GLOW_CULL_PRE`, and the brightest pixel the culled splats would have
    /// added, against AgX's black point at that exposure (2^−12.47393 pre-exposed, uni-post
    /// `agx`). Prints only (the budget is measured on the GPU); asserts that the cull at
    /// `GLOW_CULL_PRE` hides less than 1/100 of black anywhere.
    #[test]
    fn row_20_dust_fill_by_cull() {
        let (view, camera, _, coma) = neowise_row_20();
        let size = UVec2::new(1920, 1080);
        let cam = SpriteCamera::new(&camera, size);
        let grains = tail_of(&view);
        let rate = dust_rate_cd_per_s(&view, coma);
        let splats: Vec<SplatDraw> = grains
            .iter()
            .filter_map(|g| splat_draw(g, rate, view.nucleus_rel_m, view.sun_rel_m, &cam))
            .collect();
        let orient = camera.orient;
        let axes = (orient * DVec3::X, orient * DVec3::Y, orient * DVec3::NEG_Z);
        let tan_y = 10.0_f64.to_radians().tan();
        let tan_xy = (tan_y * 1920.0 / 1080.0, tan_y);
        for ev in [-5.29, -5.8] {
            // pre = 1/(1.2·2^EV) (CLAUDE.md §2); AgX's black point in cd/m² at that exposure.
            let pre = 1.0 / (1.2 * 2.0_f64.powf(ev));
            let black = 2.0_f64.powf(-12.47393) / pre;
            let mut report = Vec::new();
            for mult in [1.0, 10.0, 100.0] {
                let cut = mult * crate::comet::GLOW_CULL_PRE;
                let (mut fill, mut drawn_light, mut culled) = (0.0, 0.0, Vec::new());
                for sp in &splats {
                    let px = sp.rel.length() * cam.pixel_rad;
                    if sp.peak_cd_m2() * pre >= cut {
                        fill += (2.0 * SPRITE_SIGMAS).powi(2) * sp.s1 * sp.s2 / (px * px);
                        drawn_light += sp.drawn_cd;
                    } else {
                        culled.push(*sp);
                    }
                }
                let mut img = vec![0.0; 1920 * 1080];
                for sp in &culled {
                    rasterise_splat(sp, axes, tan_xy, size, &mut img);
                }
                let worst = img.iter().copied().fold(0.0, f64::max);
                if mult == 1.0 {
                    assert!(worst < 0.01 * black, "{worst:e} vs black {black:e}");
                }
                report.push(format!(
                    "× {mult}: {:.2} Mpx, {} splats culled, light kept {:.2} %, culled light at \
                     most {:.1e} of black",
                    fill * 1e-6,
                    culled.len(),
                    100.0 * drawn_light / splats.iter().map(|s| s.drawn_cd).sum::<f64>(),
                    worst / black
                ));
            }
            println!("row 20 at EV {ev}, cull {}", report.join("; "));
        }
    }

    /// Fix round 1: the pub twin's rasteriser keeps a splat's drawn light (Σ L·A_px over its
    /// pixels, A_px = (d·pixel)² the pixel's footprint on the sprite plane, matches `drawn_cd`
    /// within 1 % for splats of 1.5–60 px), its brightest pixel is at most the splat's centre
    /// ([`SplatDraw::peak_cd_m2`], which `vs_dust`'s visibility cull reads), and a splat wider
    /// than `SPRITE_MAX_PX` is cut to it on that axis with its light scaled by the same ratio.
    #[test]
    fn dust_raster_keeps_the_drawn_light() {
        let size = UVec2::new(480, 270);
        let camera = look_at(DVec3::new(0.0, 0.0, -1.0), DVec3::Y, 20.0);
        let cam = SpriteCamera::new(&camera, size);
        let orient = camera.orient;
        let axes = (orient * DVec3::X, orient * DVec3::Y, orient * DVec3::NEG_Z);
        let tan_y = 10.0_f64.to_radians().tan();
        let tan_xy = (tan_y * 480.0 / 270.0, tan_y);
        let dist = 1.0e11;
        let px = dist * cam.pixel_rad;
        let nucleus_rel = DVec3::new(0.0, 0.0, -dist);
        let sun_rel = DVec3::new(-AU, 0.2 * AU, 0.0);
        for (sigma_px, axis) in [
            (1.5, DVec3::X),
            (8.0, DVec3::new(0.6, 0.8, 0.0)),
            (60.0, DVec3::Y),
            (200.0, DVec3::X),
        ] {
            let sigma = sigma_px * px;
            let g = TailGrain {
                offset_m: DVec3::ZERO,
                cov_m2: outer(axis * sigma)
                    + DMat3::from_diagonal(DVec3::splat((0.4 * sigma).powi(2))),
                weight: 1.0e5,
            };
            let sp = splat_draw(&g, 1.0e9, nucleus_rel, sun_rel, &cam).expect("in front");
            let mut img = vec![0.0; 480 * 270];
            rasterise_splat(&sp, axes, tan_xy, size, &mut img);
            let sum: f64 = img.iter().sum::<f64>() * px * px;
            let max = img.iter().copied().fold(0.0, f64::max);
            println!(
                "σ {sigma_px} px: drawn {:.4e} of {:.4e} cd, raster {:.4e} cd, peak {:.3e} vs \
                 max {:.3e} cd/m²",
                sp.drawn_cd,
                sp.light_cd,
                sum,
                sp.peak_cd_m2(),
                max
            );
            assert!(max <= sp.peak_cd_m2() * (1.0 + 1e-9));
            assert!(
                (sum / sp.drawn_cd - 1.0).abs() < 0.01,
                "{sum} vs {}",
                sp.drawn_cd
            );
            // The cap: each axis at most SPRITE_MAX_PX, the light scaled by the same ratios.
            let (right, up) = (cam.right, cam.up);
            let (u1, u2, _) = splat_axes(cov_factor(g.cov_m2), right, up, SPRITE_MIN_PX * px);
            let cap = crate::comet::SPRITE_MAX_PX * px;
            let want = (cap / u1).min(1.0) * (cap / u2).min(1.0);
            assert!((sp.drawn_cd / sp.light_cd / want - 1.0).abs() < 1e-9);
            assert!(sp.s1 <= cap * (1.0 + 1e-12) && sp.s2 <= cap * (1.0 + 1e-12));
            if sigma_px > 100.0 {
                assert!(want < 0.5 && (sp.s1 / cap - 1.0).abs() < 1e-12);
            } else if sigma_px < 10.0 {
                assert_eq!(sp.drawn_cd, sp.light_cd);
            }
        }
    }

    /// Review minor 5 and lead ruling C1L-I3: at row 21 (`comet-halley-1986`: 1P/Halley on
    /// 1986-03-13 from 2.0 × 10⁶ km along its orbit's normal, fov 20°, 1 920 × 1 080) the
    /// `SPRITE_MAX_PX` cap draws only a few per cent of the dust tail's light, so the renderer's
    /// grey note is raised; the share is computed within the CPU budget (printed in this test
    /// build, which is unoptimised).
    #[test]
    fn halley_row_21_draws_a_small_share_of_its_dust() {
        let conic = conic_from_perihelion(
            0.574_863_831_374_341_3,
            0.967_935_995_695_321_1,
            162.190_530_043_912_9,
            59.098_947_206_124_37,
            112.241_431_463_776_4,
            2_446_469.973_616_146_5,
        );
        let t = Epoch::parse_iso_utc("1986-03-13T00:00:00Z").expect("date");
        let (nucleus, vel) = heliocentric_state_icrs(&conic, t);
        // bookmarks/m1.ron row 21: the camera relative to the nucleus [m] and its up.
        let cam_rel = DVec3::new(
            -522_087_931.581_177_4,
            -468_233_681.906_447_8,
            1_873_013_991.091_796_4,
        );
        let up = DVec3::new(
            0.779_875_156_652_241_4,
            -0.622_889_856_613_611_8,
            0.061_668_197_350_952_75,
        );
        println!(
            "row 21: camera {:.3e} m from the nucleus, {:.1}° from the orbit's normal",
            cam_rel.length(),
            angle_deg(cam_rel, nucleus.cross(vel))
        );
        // SBDB's M1 5.5, K1 8.0 (no cited visual law for Halley).
        let view = comet_on(2, conic, t, nucleus + cam_rel, 5.5, 8.0);
        let size = UVec2::new(1920, 1080);
        let camera = look_at(view.nucleus_rel_m, up, 20.0);
        let scene = crate::comet::CometScene {
            comets: vec![view],
            epoch: t,
            eye_adapted: false,
        };
        let mut manager = TailManager::default();
        let plan =
            crate::comet::plan_frame(&scene, &camera, size, 0.0, &mut manager, usize::MAX, 0);
        assert!(
            plan.dust_drawn.is_empty(),
            "computed with drawn_per_frame 0"
        );
        let clock = std::time::Instant::now();
        let plan = crate::comet::plan_frame(&scene, &camera, size, 0.1, &mut manager, 0, 1);
        let ms = clock.elapsed().as_secs_f64() * 1e3;
        assert_eq!(plan.dust_drawn.len(), 1);
        let d = plan.dust_drawn[0];
        let note = crate::comet::dust_partly_drawn_note(d.drawn_fraction);
        // The share alone, per tail (`dust_light_drawn`, what `plan_frame` adds), averaged over
        // repeats: the ruling's budget is 0.05 ms per tail in release.
        let (view, tail) = (&scene.comets[0], &manager.tails[&d.id]);
        let cam = SpriteCamera::new(&camera, size);
        let rate = dust_rate_cd_per_s(view, crate::comet::coma::coma_intensity_cd(view));
        let reps = 200;
        let clock = std::time::Instant::now();
        let mut sink = 0.0;
        for _ in 0..reps {
            let (l, dr) =
                dust_light_drawn(&tail.grains, rate, view.nucleus_rel_m, view.sun_rel_m, &cam);
            sink += dr / l;
        }
        let share_ms = clock.elapsed().as_secs_f64() * 1e3 / f64::from(reps);
        assert!((sink / f64::from(reps) - d.drawn_fraction).abs() < 1e-12);
        println!(
            "row 21: {} splats, tail {:.3e} cd, drawn {:.2} % ({ms:.2} ms for plan_frame with the \
             share; the share alone {share_ms:.4} ms per tail; this build); note: {note}",
            d.splats,
            d.light_cd,
            100.0 * d.drawn_fraction
        );
        assert!(d.drawn_fraction > 0.0 && d.drawn_fraction < 0.1, "{d:?}");
        assert!(d.drawn_fraction < crate::comet::DUST_DRAWN_NOTE_BELOW);
        // Not due again for DRAWN_REFRESH_WALL_S: the cached value is reported.
        let again = crate::comet::plan_frame(&scene, &camera, size, 0.2, &mut manager, 0, 1);
        assert_eq!(again.dust_drawn, plan.dust_drawn);
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
