//! Milky Way galaxy model v0: thin and thick disks, halo, bar, four spiral arms and a dust layer,
//! the Milky Way band they make on the sky, and a WGSL twin in `shaders/galaxy.wgsl` (M1 Task A5).
//!
//! **Frame.** Galactocentric: origin at Sgr A*, +x from the Sun's projection toward the centre,
//! +y toward Galactic longitude 90° (the direction of rotation at the Sun), +z toward the north
//! Galactic pole; distances [pc]. Its axes are the Galactic axes of `uni_core::icrs_to_galactic`:
//! the Sun's 25 pc height tilts the true plane by atan(25 / 8277) = 0.17° against them, which v0
//! ignores.
//!
//! **Stars** [stars/pc³]: Jurić et al. 2008 thin and thick disks and halo, plus the bar of
//! Bland-Hawthorn & Gerhard 2016, scaled so the total at the Sun equals the star count of the
//! local luminosity function ([`lf`]: 0.106 stars/pc³, ≈ 0.1 as the 10 pc census gives). The thin
//! disk is 15 % denser at the centre of a spiral arm (four arms from Reid et al. 2019).
//!
//! **Light** [L☉,V/pc³]: stars × the mean V luminosity per star of the same luminosity function
//! (0.0519 L☉,V/pc³ at the Sun; Flynn et al. 2006 measure 0.056). Colour: a blackbody colour
//! temperature per component.
//!
//! **Dust** [A_V mag/pc]: an exponential layer that follows the arms (3× denser at an arm centre),
//! 1 mag/kpc in its midplane below the Sun, 80 pc thick there and flaring outward. Its midplane is
//! the gas layer's, z_d = 19.5 pc, 5.5 pc below the Sun, not the stellar disk's z = 0.
//!
//! **Band** [cd/m², linear Rec.709]: the light of stars fainter than the catalogue limit, summed
//! along the ray through the dust (see [`GalaxyModel::band_radiance`]).
//!
//! v0 approximations, all on purpose and labelled on screen as "galaxy model v0": every component
//! uses the local mean V luminosity per star (local M/L_V = 0.043 / 0.0519 = 0.83), so the old
//! thick disk, halo and bar, which lack the local young bright stars, are too bright per star;
//! extinction is grey (no reddening) and ignored in the faint-star cut; the dust is a smooth
//! layer, with none of the discrete local clouds that shape the real band (see `DUST_Z_MID_PC`);
//! the arms are smooth log spirals with no Local (Orion) arm; the stellar disks have no warp or
//! flare, and there is no nuclear star cluster.

use std::f64::consts::{LN_10, PI, TAU};

use glam::{DVec2, DVec3};
use uni_core::{M_V_SUN, blackbody_linear_srgb, icrs_to_galactic, star_illuminance_lux};

use crate::lf;

// ---- Geometry -------------------------------------------------------------------------------

/// R₀, the Sun's distance from Sgr A* [pc] (GRAVITY Collaboration 2022, A&A 657, L12:
/// 8277 ± 9 pc).
const R0_PC: f64 = 8277.0;
/// z☉, the Sun's height above the Galactic plane [pc], the midplane of the stellar disk (Jurić et
/// al. 2008, ApJ 673, 864, Table 10, Z₀). The dust midplane lies lower, at `DUST_Z_MID_PC`.
const Z_SUN_PC: f64 = 25.0;

// ---- Disks and halo: Jurić et al. 2008, ApJ 673, 864, Table 10 -------------------------------
// ρ(R, z) = ρ_D(R₀) [e^{−(R−R₀)/L₁ − |z|/H₁} + f e^{−(R−R₀)/L₂ − |z|/H₂}
//                    + f_h (R₀ / √(R² + (z/q)²))^{n_h}]   (their Eqs. 21–24)
// Disk values from the bias-corrected column; the halo has no bias-corrected values, so those
// are the measured column.

/// Thin disk scale length L₁ [pc].
const THIN_L_PC: f64 = 2600.0;
/// Thin disk scale height H₁ [pc].
const THIN_H_PC: f64 = 300.0;
/// Thick disk scale length L₂ [pc].
const THICK_L_PC: f64 = 3600.0;
/// Thick disk scale height H₂ [pc].
const THICK_H_PC: f64 = 900.0;
/// Thick-to-thin density ratio f at the Sun.
const THICK_F: f64 = 0.12;
/// Halo-to-thin density ratio f_h at the Sun.
const HALO_F: f64 = 0.0051;
/// Halo axis ratio q = c/a.
const HALO_Q: f64 = 0.64;
/// Halo power-law index n_h.
const HALO_N: f64 = 2.77;
/// Halo core radius [pc]: a v0 choice, not from Jurić et al. (whose fit covers the outer halo);
/// it keeps r^{−n_h} finite at the centre, where the bar is ~20× denser than this halo.
const HALO_CORE_PC: f64 = 500.0;

// ---- Bar: Bland-Hawthorn & Gerhard 2016, ARA&A 54, 529 ----------------------------------------
// A triaxial exponential ρ = ρ₀ e^{−r_s}, r_s = √((x'/x₀)² + (y'/y₀)² + (z/z₀)²), with mass
// 8π ρ₀ x₀ y₀ z₀, cut off beyond the long bar's half-length.

/// Angle between the bar's major axis and the Sun–centre line [deg], near end at positive
/// longitude (§4.2.1: φ = 27° ± 2°).
const BAR_ANGLE_DEG: f64 = 27.0;
/// Density scale lengths x₀ : y₀ : z₀ near the centre [pc] (§4.2.1: 0.70 : 0.44 : 0.18 kpc, from
/// Wegg & Gerhard 2013, MNRAS 435, 1874).
const BAR_SCALE_PC: [f64; 3] = [700.0, 440.0, 180.0];
/// Stellar mass [M☉] (§4.2.4: 1.4–1.7 × 10¹⁰ M☉ in the bulge region).
const BAR_MASS_MSUN: f64 = 1.5e10;
/// Half-length of the long bar [pc] (§4.3: R_lb = 5.0 ± 0.2 kpc); the density tapers off beyond.
const BAR_HALF_LENGTH_PC: f64 = 5000.0;
/// Gaussian width of that taper [pc]: a v0 choice (the exponential is already 10⁻³ of its
/// central value there).
const BAR_TAPER_PC: f64 = 500.0;
/// Local stellar mass density ρ*,⊙ [M☉/pc³] (McKee, Parravano & Hollenbach 2015, ApJ 814, 13,
/// Eq. 38). The bar gets the local number of stars per M☉, `lf` stars / ρ*,⊙.
const STELLAR_MASS_DENSITY_SUN: f64 = 0.043;

// ---- Spiral arms: Reid et al. 2019, ApJ 885, 131, Table 2 --------------------------------------
// ln(R / R_ref) = −(β − β_ref) tan ψ, with β the galactocentric azimuth, 0 toward the Sun and
// increasing in the direction of Galactic rotation.

/// R₀ of Reid et al. 2019 [pc]. The arms are evaluated in their frame shifted along x by
/// R₀ − 8150 pc, so the heliocentric positions their maser parallaxes measure are kept.
const REID_R0_PC: f64 = 8150.0;
/// Number of spiral arms.
const ARM_COUNT: usize = 4;
/// Per arm: (R_ref [pc], β_ref [deg], ψ [deg]). Reid et al. fit two pitch angles per arm around a
/// kink; v0 keeps one log spiral per arm through the kink (R_kink, β_kink), with the pitch on the
/// side of the kink nearest the Sun (ψ< for β < β_kink). Norma–Outer is one arm in Reid et al.;
/// it uses the Outer arm's kink and its inner pitch ψ> = 9.4°, which brings it one turn inward
/// to 4.33 kpc at β = 18°, 0.13 kpc from Norma's kink (4.46 kpc, 18°). (Norma's own ψ< = −1.0°
/// would wind into a ring.)
const ARMS: [(f64, f64, f64); ARM_COUNT] = [
    (12_240.0, 18.0, 9.4), // Norma–Outer
    (4_910.0, 23.0, 14.1), // Scutum–Centaurus
    (6_040.0, 24.0, 17.1), // Sagittarius–Carina
    (8_870.0, 40.0, 10.3), // Perseus
];
/// Gaussian σ of an arm's cross-section [pc] (brief value; Reid et al. 2019 give widths
/// w(R) = 336 + 36 (R − 8.15 kpc) pc, 0.31–0.35 kpc at the Local and Perseus kinks).
const ARM_SIGMA_PC: f64 = 400.0;
/// Extra thin-disk stars and light at an arm's centre: 15 % (the research brief's 10–20 % for old
/// stars; Drimmel & Spergel 2001, ApJ 556, 181, measure arm/interarm NIR surface brightness
/// ratios of 1.20 in J and 1.32 in K).
const ARM_LIGHT_CONTRAST: f64 = 0.15;
/// Extra dust at an arm's centre: 3× the inter-arm density (v0 value of the brief; dust and young
/// stars are concentrated in the arms, research brief §1.2).
const ARM_DUST_CONTRAST: f64 = 2.0;
/// The arms fade in between these radii [pc] in Reid's frame: they start near the bar's end
/// (R_lb = 5 kpc, above) and Reid's innermost arm, the 3-kpc arm (R_kink 3.52 kpc). A v0 choice
/// that stops the log spirals winding into the bar region.
const ARM_TAPER_PC: [f64; 2] = [3000.0, 4000.0];

// ---- Dust ---------------------------------------------------------------------------------------
// κ(R, z) = κ₀ e^{−(R−R₀)/h_r − |z − z_d|/h_z(R)} (1 + 2 A), A = the arm strength. The midplane
// z_d and h_z(R₀) = 80 pc are the lead's ruling on A5 review I1 (task A5r), which also ratifies
// A5's radial profile, h_r and the flare (A5 decision D4).

/// A_V per unit length in the dust midplane directly below the Sun [mag/pc]: 1 mag/kpc (brief
/// value; κ₀ is set so the rate at (x☉, y☉, z_d) is this, arms included). Checked against Drimmel
/// & Spergel 2001, ApJ 556, 181: their axisymmetric dust disk alone gives κ_V ρ_d(R☉) = 0.0180 ×
/// 1098 e^{−1/0.28} = 0.56 mag/kpc (Table 1, Eq. 13), with the Sun in a gap of their Local arm;
/// mean values in the plane over kpc paths are quoted from 0.7 to 1.8 mag/kpc. With h_z = 80 pc
/// the layer's full thickness at the Sun's radius holds 2 κ₀ h_z = 0.16 mag, against the 0.12 mag
/// Melchior, Combes & Gould 2007 (arXiv:astro-ph/0610884, footnote 3) give for the total disc
/// thickness; the column from the Sun to the north Galactic pole is 0.075 mag. v0 keeps the
/// in-plane rate and accepts a full column 1.3× theirs.
const DUST_A_V_SUN_MAG_PER_PC: f64 = 1.0e-3;
/// Z☉, the Sun's height above the plane of high-mass star-forming regions [pc] (Reid et al. 2019,
/// ApJ 885, 131: 5.5 ± 5.8 pc). Those regions trace the gas and dust layer; `Z_SUN_PC` (Jurić et
/// al. 2008) is the Sun's height above the stellar disk.
const Z_SUN_ABOVE_GAS_PC: f64 = 5.5;
/// Height of the dust midplane z_d in this frame [pc]: z☉ − Z☉ = 25 − 5.5 = 19.5 pc (lead ruling
/// A5r), parallel to the stellar disk (v0: no tilt or warp). With the midplane at z = 0, 25 pc
/// below the Sun, a ray at b = −5° stayed 50 pc closer to it than its mirror at b = +5° beyond
/// ~570 pc: toward l = 0° it crossed A_V = 3.39 mag against 2.16 (to 10 kpc), and the band at
/// (0°, −5°) was 0.44× that at (0°, +5°). With z_d (and h_z = 80 pc) the columns are 1.75 and
/// 1.51 mag and the band ratio is 0.97. The real sky is brighter to the south; that extra
/// southern brightness also comes from discrete local clouds north of the plane (Ophiuchus, the
/// Pipe Nebula, the Aquila Rift, at ~120–250 pc), which a smooth v0 layer cannot model. The
/// on-screen "galaxy model v0" label covers it; a 3-D dust map is M4.
const DUST_Z_MID_PC: f64 = Z_SUN_PC - Z_SUN_ABOVE_GAS_PC;
/// Dust scale height at the Sun's radius [pc]: 80 pc (lead ruling A5r; the A5 brief's 120 pc put
/// A_V = 4.6 mag in Baade's window). Misiriotis et al. 2006, A&A 459, 113, Table 2, fit 100 pc for
/// the cold dust; Drimmel & Spergel 2001 a sech² layer of 188 pc there. With z_d, 80 pc gives
/// A_V = 2.4 mag in Baade's window, (l, b) = (1°, −3.9°), where Stanek 1996 (ApJ 460, L37)
/// measures 1.26–2.79 mag across the field.
const DUST_H_Z_PC: f64 = 80.0;
/// Dust scale length [pc] (Misiriotis et al. 2006, Table 2: cold dust h_c = 5.0 kpc, about twice
/// the stars'; A5 decision D4, ratified in the lead's A5r ruling). Measured in A5 with a 120 pc
/// layer at z = 0: Drimmel & Spergel 2001's 2.26 kpc, with their flat core inside 4 kpc, put
/// A_V = 48 mag in the plane between the Sun and Sgr A* (5 kpc: 23 mag) and 7 mag in front of the
/// bulge at (l, b) = (0°, −5°) (5 kpc: 3.4 mag), which hid the bulge.
const DUST_H_R_PC: f64 = 5000.0;
/// The dust layer flares (Drimmel & Spergel 2001, Table 1): h(R) = h₀ + h₁ (R − r_f) beyond
/// r_f = 4.40 kpc, with h₀ = 134.4 pc and h₁ = 14.8 pc/kpc, 188 pc at their R☉ = 8 kpc. v0 keeps
/// that shape and scales it to `DUST_H_Z_PC` at R₀ (A5 decision D4, ratified in the lead's A5r
/// ruling): h_z(R) = 80 pc · h(R) / h(R₀), 56 pc inside r_f, so the layer in front of the bulge is
/// thinner than at the Sun.
const DUST_FLARE_H0_PC: f64 = 134.4;
const DUST_FLARE_H1: f64 = 14.8e-3;
const DUST_FLARE_R_PC: f64 = 4400.0;

// ---- Light --------------------------------------------------------------------------------------

/// Blackbody colour temperatures [K] of the thin disk, the thick disk and halo, and the bar.
/// v0 approximations (brief values): roughly the integrated colours of a mixed young-and-old disk
/// and of old K-giant-dominated populations; not fitted to photometry.
const THIN_TEMP_K: f64 = 5000.0;
const OLD_TEMP_K: f64 = 4500.0;
const BAR_TEMP_K: f64 = 4200.0;

// ---- Band integration ---------------------------------------------------------------------------

/// The band is integrated from `BAND_S_MIN_PC` to `BAND_S_MAX_PC` along the ray in
/// `BAND_STEPS` log-spaced steps (brief). Light from the first parsec is left out: only stars
/// with M_V > 15 are fainter than V = 10 that close, ~10⁻⁵ of the light.
const BAND_S_MIN_PC: f64 = 1.0;
const BAND_S_MAX_PC: f64 = 30_000.0;
const BAND_STEPS: u32 = 128;
/// Below this optical depth per step the slab factor uses its series (both twins).
const BAND_SERIES_MAX_DTAU: f64 = 1e-3;
/// Optical depth per magnitude of extinction: e^{−τ} = 10^{−0.4 A} (Pogson's magnitude scale).
const TAU_PER_MAG: f64 = 0.4 * LN_10;

/// The Milky Way model v0. Construct it with [`GalaxyModel::milky_way`]; the CPU twin of
/// `galaxy.wgsl`. Its parameter groups are `GalaxyGpu`'s fields, in f64 and in the same order, so
/// each Rust function reads the same numbers as its WGSL twin.
pub struct GalaxyModel {
    sun: DVec3,
    disk: [f64; 4],
    norm: [f64; 4],
    halo: [f64; 4],
    bar: [f64; 4],
    bar_scale: [f64; 4],
    arm_shape: [f64; 4],
    arm_taper: [f64; 4],
    arms: [[f64; 4]; ARM_COUNT],
    dust: [f64; 4],
    dust_flare: [f64; 4],
    light: [f64; 4],
    color_thin: DVec3,
    color_old: DVec3,
    color_bar: DVec3,
    lf: [[f64; 3]; lf::LF_BINS],
}

/// The model's parameters for the GPU: vec4-sized members only, laid out like `GalaxyGpu` in
/// `galaxy.wgsl` (CLAUDE.md §3 uniform layout rule). Lengths in pc.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GalaxyGpu {
    /// L₁, H₁, L₂, H₂: thin and thick disk scale lengths and heights.
    pub disk: [f32; 4],
    /// R₀, f (thick/thin at the Sun), thin-disk density in the plane at R₀ [stars/pc³], mean V
    /// luminosity per star [L☉,V].
    pub norm: [f32; 4],
    /// f_h, n_h, core radius, q.
    pub halo: [f32; 4],
    /// cos α, sin α of the bar angle, central density [stars/pc³], half-length.
    pub bar: [f32; 4],
    /// x₀, y₀, z₀ scale lengths, end-taper width.
    pub bar_scale: [f32; 4],
    /// Gaussian σ, light contrast, dust contrast, x shift into Reid's frame.
    pub arm_shape: [f32; 4],
    /// Inner fade-in start and end radii, 0, 0.
    pub arm_taper: [f32; 4],
    /// Per arm: β_ref [rad], ln(R_ref / 1 pc), 1 / tan ψ, sin ψ.
    pub arms: [[f32; 4]; ARM_COUNT],
    /// κ₀ [A_V mag/pc in the dust midplane at R₀ between the arms], h_r, h_z at R₀, z_d (height
    /// of the dust midplane).
    pub dust: [f32; 4],
    /// Flare h(R) = h₀ + h₁ max(R − r_f, 0): h₀, h₁ [pc/pc], r_f, h(R₀); h_z(R) = h_z(R₀) h(R)/h(R₀).
    pub dust_flare: [f32; 4],
    /// Radiance per V luminosity column [cd/m² per (L☉,V/pc²)], 0, 0, 0.
    pub light: [f32; 4],
    /// Linear Rec.709 colour (Y = 1) of the thin disk; w = 0.
    pub color_thin: [f32; 4],
    /// Colour of the thick disk and the halo.
    pub color_old: [f32; 4],
    /// Colour of the bar.
    pub color_bar: [f32; 4],
    /// Per luminosity-function bin: x_bright, x_faint, light weight, 0 (`lf::faint_bins`).
    pub lf: [[f32; 4]; lf::LF_BINS],
}

/// `ShaderLib` path of the WGSL twin (B5 imports it and binds `GalaxyGpu` at its own group).
pub const GALAXY_WGSL_PATH: &str = "procgen/galaxy.wgsl";
/// Source of the WGSL twin.
pub const GALAXY_WGSL_SRC: &str = include_str!("../shaders/galaxy.wgsl");

impl GalaxyModel {
    pub fn milky_way() -> Self {
        let n_sun = lf::local_number_density();
        let [x0, y0, z0] = BAR_SCALE_PC;
        let bar_rho0 = BAR_MASS_MSUN / (8.0 * PI * x0 * y0 * z0);
        let (sin_a, cos_a) = BAR_ANGLE_DEG.to_radians().sin_cos();
        let color = |t: f64| blackbody_linear_srgb(t).as_dvec3();
        let mut g = Self {
            sun: DVec3::new(-R0_PC, 0.0, Z_SUN_PC),
            disk: [THIN_L_PC, THIN_H_PC, THICK_L_PC, THICK_H_PC],
            // The thin-disk density is set below, once the shape at the Sun is known.
            norm: [
                R0_PC,
                THICK_F,
                1.0,
                lf::local_v_luminosity_density() / n_sun,
            ],
            halo: [HALO_F, HALO_N, HALO_CORE_PC, HALO_Q],
            bar: [
                cos_a,
                sin_a,
                bar_rho0 * n_sun / STELLAR_MASS_DENSITY_SUN,
                BAR_HALF_LENGTH_PC,
            ],
            bar_scale: [x0, y0, z0, BAR_TAPER_PC],
            arm_shape: [
                ARM_SIGMA_PC,
                ARM_LIGHT_CONTRAST,
                ARM_DUST_CONTRAST,
                R0_PC - REID_R0_PC,
            ],
            arm_taper: [ARM_TAPER_PC[0], ARM_TAPER_PC[1], 0.0, 0.0],
            arms: ARMS.map(|(r_ref, beta_ref_deg, psi_deg)| {
                let psi = f64::to_radians(psi_deg);
                [
                    beta_ref_deg.to_radians(),
                    r_ref.ln(),
                    1.0 / psi.tan(),
                    psi.sin(),
                ]
            }),
            // κ₀ is set below, once the arm strength at the Sun is known.
            dust: [1.0, DUST_H_R_PC, DUST_H_Z_PC, DUST_Z_MID_PC],
            dust_flare: [
                DUST_FLARE_H0_PC,
                DUST_FLARE_H1,
                DUST_FLARE_R_PC,
                DUST_FLARE_H0_PC + DUST_FLARE_H1 * (R0_PC - DUST_FLARE_R_PC),
            ],
            // A source of V luminosity L at 10 pc gives E = star_illuminance_lux(M_V☉) L; a column
            // of j [L☉,V/pc³] over ds [pc] seen per steradian gives j ds · E(10 pc) · (10 pc)².
            light: [star_illuminance_lux(M_V_SUN) * 100.0, 0.0, 0.0, 0.0],
            color_thin: color(THIN_TEMP_K),
            color_old: color(OLD_TEMP_K),
            color_bar: color(BAR_TEMP_K),
            lf: lf::faint_bins(),
        };
        // Total star density at the Sun = the luminosity function's count.
        let sun_shape = g.components(g.sun, g.arm_strength(g.sun));
        g.norm[2] = (n_sun - sun_shape[3]) / (sun_shape[0] + sun_shape[1] + sun_shape[2]);
        // A_V rate in the dust midplane directly below the Sun = DUST_A_V_SUN_MAG_PER_PC.
        let plane = DVec3::new(g.sun.x, g.sun.y, DUST_Z_MID_PC);
        g.dust[0] = DUST_A_V_SUN_MAG_PER_PC / g.extinction(plane, g.arm_strength(plane));
        g
    }

    /// The Sun's galactocentric position [pc].
    pub fn sun_galactocentric_pc(&self) -> DVec3 {
        self.sun
    }

    /// Galactocentric position [pc] of an ICRS position [pc, origin SSB].
    pub fn icrs_pc_to_galactocentric(&self, p_icrs_pc: DVec3) -> DVec3 {
        icrs_to_galactic(p_icrs_pc) + self.sun
    }

    /// Star density [stars/pc³].
    pub fn star_density(&self, p_gc_pc: DVec3) -> f64 {
        self.components(p_gc_pc, self.arm_strength(p_gc_pc))
            .iter()
            .sum()
    }

    /// V-band luminosity density [L☉,V/pc³].
    pub fn v_luminosity_density(&self, p_gc_pc: DVec3) -> f64 {
        self.star_density(p_gc_pc) * self.norm[3]
    }

    /// Dust extinction A_V per unit length [mag/pc].
    pub fn extinction_v_per_pc(&self, p_gc_pc: DVec3) -> f64 {
        self.extinction(p_gc_pc, self.arm_strength(p_gc_pc))
    }

    /// Band radiance [cd/m², linear Rec.709] seen from `cam_gc_pc` along `dir_gc`, from stars
    /// fainter than `v_lim`; the CPU twin of `gal_band_radiance`.
    ///
    /// ∫ j(s) f_faint(s) e^{−τ(s)} ds from 1 pc to 30 kpc in 128 log-spaced steps: j is the light
    /// at the step's geometric midpoint, f_faint = [`lf::faint_light_fraction`]`(s, v_lim)` keeps
    /// only stars fainter than the catalogue limit (brighter ones are drawn one by one), and
    /// within a step the dust attenuates exactly as a uniform slab: ∫₀^Δs e^{−κt} dt =
    /// (1 − e^{−κΔs}) / κ.
    pub fn band_radiance(&self, cam_gc_pc: DVec3, dir_gc: DVec3, v_lim: f64) -> DVec3 {
        let dir = dir_gc.normalize_or_zero();
        let x_lim = 10_f64.powf(-0.4 * v_lim);
        let ratio = BAND_S_MAX_PC / BAND_S_MIN_PC;
        let mut s0 = BAND_S_MIN_PC;
        let mut tau = 0.0_f64;
        let mut sum = DVec3::ZERO;
        for i in 1..=BAND_STEPS {
            let s1 = BAND_S_MIN_PC * ratio.powf(f64::from(i) / f64::from(BAND_STEPS));
            let s = (s0 * s1).sqrt();
            let ds = s1 - s0;
            let p = cam_gc_pc + dir * s;
            let arms = self.arm_strength(p);
            let [thin, thick, halo, bar] = self.components(p, arms);
            let rgb =
                thin * self.color_thin + (thick + halo) * self.color_old + bar * self.color_bar;
            let dtau = self.extinction(p, arms) * TAU_PER_MAG * ds;
            // Below dτ = 10⁻³ the series 1 − dτ/2 + dτ²/6 (error < 5·10⁻¹¹), where the GPU's f32
            // 1 − e^{−dτ} would lose digits.
            let path = if dtau < BAND_SERIES_MAX_DTAU {
                ds * (1.0 - dtau * (0.5 - dtau / 6.0))
            } else {
                ds * -(-dtau).exp_m1() / dtau
            };
            let faint = lf::faint_fraction_at(&self.lf, x_lim * (s / 10.0) * (s / 10.0));
            sum += rgb * (faint * (-tau).exp() * path);
            tau += dtau;
            s0 = s1;
        }
        sum * (self.norm[3] * self.light[0])
    }

    /// The parameters packed for the GPU.
    pub fn gpu(&self) -> GalaxyGpu {
        let f = |v: [f64; 4]| v.map(|x| x as f32);
        let rgb = |c: DVec3| f(c.extend(0.0).to_array());
        GalaxyGpu {
            disk: f(self.disk),
            norm: f(self.norm),
            halo: f(self.halo),
            bar: f(self.bar),
            bar_scale: f(self.bar_scale),
            arm_shape: f(self.arm_shape),
            arm_taper: f(self.arm_taper),
            arms: self.arms.map(f),
            dust: f(self.dust),
            dust_flare: f(self.dust_flare),
            light: f(self.light),
            color_thin: rgb(self.color_thin),
            color_old: rgb(self.color_old),
            color_bar: rgb(self.color_bar),
            lf: self.lf.map(|[a, b, w]| f([a, b, w, 0.0])),
        }
    }

    /// How close `p` is to a spiral arm's centre line: 1 on it, falling as a Gaussian in the
    /// distance to the nearest arm (max over the arms), faded in between `ARM_TAPER_PC`.
    ///
    /// The arm crosses the circle of radius R at the azimuth β_R = β_ref − ln(R/R_ref)/tan ψ (mod
    /// 2π); a point Δβ away along that circle is R |Δβ| sin ψ from the arm (the arm meets the
    /// circle at the pitch angle ψ).
    fn arm_strength(&self, p_gc_pc: DVec3) -> f64 {
        let [sigma, _, _, x_shift] = self.arm_shape;
        let q = DVec2::new(p_gc_pc.x + x_shift, p_gc_pc.y);
        let r = q.length().max(1.0);
        let beta = PI - q.y.atan2(q.x);
        let mut strength = 0.0_f64;
        for [beta_ref, ln_r_ref, cot_psi, sin_psi] in self.arms {
            let d_beta = wrap_pi(beta - beta_ref + (r.ln() - ln_r_ref) * cot_psi);
            let d = r * d_beta.abs() * sin_psi;
            strength = strength.max((-0.5 * (d / sigma) * (d / sigma)).exp());
        }
        let [r_in0, r_in1, _, _] = self.arm_taper;
        strength * smoothstep(r_in0, r_in1, r)
    }

    /// Star density of each component [stars/pc³]: thin disk (with arms), thick disk, halo, bar.
    fn components(&self, p: DVec3, arms: f64) -> [f64; 4] {
        let [l1, h1, l2, h2] = self.disk;
        let [r0, f_thick, n_thin0, _] = self.norm;
        let [f_halo, n_halo, r_core, q] = self.halo;
        let [_, light_contrast, _, _] = self.arm_shape;
        let r = p.truncate().length();
        let z = p.z.abs();
        let thin = (-(r - r0) / l1 - z / h1).exp() * (1.0 + light_contrast * arms);
        let thick = f_thick * (-(r - r0) / l2 - z / h2).exp();
        let r2 = r * r + (p.z / q) * (p.z / q);
        let halo = f_halo * (r0 * r0 / r2.max(r_core * r_core)).powf(0.5 * n_halo);
        [
            n_thin0 * thin,
            n_thin0 * thick,
            n_thin0 * halo,
            self.bar_density(p),
        ]
    }

    /// Bar star density [stars/pc³].
    fn bar_density(&self, p: DVec3) -> f64 {
        let [cos_a, sin_a, n_bar0, half_length] = self.bar;
        let [x0, y0, z0, taper] = self.bar_scale;
        // Major axis (cos α, −sin α): its near end, toward the Sun, is at positive longitude.
        let xb = p.x * cos_a - p.y * sin_a;
        let yb = p.x * sin_a + p.y * cos_a;
        let rs = ((xb / x0) * (xb / x0) + (yb / y0) * (yb / y0) + (p.z / z0) * (p.z / z0)).sqrt();
        let beyond = ((xb * xb + yb * yb).sqrt() - half_length).max(0.0) / taper;
        n_bar0 * (-rs - 0.5 * beyond * beyond).exp()
    }

    /// Dust extinction [A_V mag/pc] given the arm strength at `p`.
    fn extinction(&self, p: DVec3, arms: f64) -> f64 {
        let [kappa0, h_r, h_z_sun, z_mid] = self.dust;
        let [h0, h1, r_flare, h_sun] = self.dust_flare;
        let [_, _, dust_contrast, _] = self.arm_shape;
        let r = p.truncate().length();
        let h_z = h_z_sun * (h0 + h1 * (r - r_flare).max(0.0)) / h_sun;
        let z = (p.z - z_mid).abs();
        kappa0 * (-(r - self.norm[0]) / h_r - z / h_z).exp() * (1.0 + dust_contrast * arms)
    }
}

/// `a` wrapped into [−π, π) (the same formula as `gal_wrap_pi` in WGSL).
fn wrap_pi(a: f64) -> f64 {
    a - TAU * ((a + PI) / TAU).floor()
}

/// GLSL/WGSL `smoothstep`: 0 below `e0`, 1 above `e1`, Hermite 3t² − 2t³ between.
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    /// Unit vector toward Galactic longitude and latitude [deg] (Galactic axes, which are also the
    /// galactocentric axes).
    fn lb_dir(l_deg: f64, b_deg: f64) -> DVec3 {
        let (l, b) = (l_deg.to_radians(), b_deg.to_radians());
        DVec3::new(b.cos() * l.cos(), b.cos() * l.sin(), b.sin())
    }

    /// A_V [mag] from the Sun along `dir` out to `s_max_pc`: midpoint rule, 1 pc steps.
    fn a_v_from_sun(g: &GalaxyModel, dir: DVec3, s_max_pc: f64) -> f64 {
        let sun = g.sun_galactocentric_pc();
        let steps = s_max_pc.round() as usize;
        (0..steps)
            .map(|i| g.extinction_v_per_pc(sun + dir * (i as f64 + 0.5)))
            .sum()
    }

    /// Rec.709 luminance of a linear RGB colour (ITU-R BT.709-6).
    fn luminance(rgb: DVec3) -> f64 {
        rgb.dot(DVec3::new(0.2126, 0.7152, 0.0722))
    }

    #[test]
    fn sun_position() {
        let g = GalaxyModel::milky_way();
        assert!((g.sun_galactocentric_pc() - DVec3::new(-8277.0, 0.0, 25.0)).length() < 1e-9);
    }

    /// The Galactic Centre's ICRS direction (ESA 1997) seen from the SSB lands on +x, R₀ away.
    #[test]
    fn icrs_galactic_centre_maps_to_origin() {
        let g = GalaxyModel::milky_way();
        let gc_dir =
            uni_core::radec_to_icrs(266.4050_f64.to_radians(), (-28.9362_f64).to_radians());
        let p = g.icrs_pc_to_galactocentric(gc_dir * 8277.0);
        assert!(p.truncate().length() < 1.0, "{p}");
        assert!((p.z - 25.0).abs() < 1.0, "{p}");
    }

    #[test]
    fn local_density_near_0p1() {
        let g = GalaxyModel::milky_way();
        let n = g.star_density(g.sun_galactocentric_pc());
        assert!((0.09..=0.11).contains(&n), "{n} stars/pc³");
    }

    #[test]
    fn local_luminosity_density_matches_source() {
        let g = GalaxyModel::milky_way();
        let j = g.v_luminosity_density(g.sun_galactocentric_pc());
        let cited = lf::LOCAL_V_LUMINOSITY_DENSITY_FLYNN_2006;
        assert!((j / cited - 1.0).abs() < 0.10, "{j} vs {cited} L☉/pc³");
    }

    #[test]
    fn extinction_sanity() {
        let g = GalaxyModel::milky_way();
        let ngp = a_v_from_sun(&g, DVec3::Z, 30_000.0);
        assert!((0.03..=0.15).contains(&ngp), "A_V to the NGP {ngp}");
        let l90 = a_v_from_sun(&g, lb_dir(90.0, 0.0), 1000.0);
        assert!(
            (0.5..=2.0).contains(&l90),
            "A_V along l = 90°, 1 kpc: {l90}"
        );
    }

    #[test]
    fn band_brighter_toward_centre() {
        let g = GalaxyModel::milky_way();
        let sun = g.sun_galactocentric_pc();
        let band = |dir: DVec3| luminance(g.band_radiance(sun, dir, 10.0));
        let centre = band(lb_dir(0.0, -5.0));
        let anticentre = band(lb_dir(180.0, 0.0));
        let ngp = band(DVec3::Z);
        assert!(
            centre > 2.0 * anticentre,
            "centre {centre} anticentre {anticentre}"
        );
        assert!(anticentre > 2.0 * ngp, "anticentre {anticentre} NGP {ngp}");
    }

    /// Lead ruling A5r: the band 5° south of the Galactic Centre is not dimmer than 5° north. The
    /// real sky is brighter to the south (Baade's window, the Sagittarius star cloud); a dust
    /// midplane at z = 0, 25 pc below the Sun, made the south 0.44× the north.
    #[test]
    fn band_south_not_dimmer() {
        let g = GalaxyModel::milky_way();
        let sun = g.sun_galactocentric_pc();
        let band = |dir: DVec3| luminance(g.band_radiance(sun, dir, 10.0));
        let south = band(lb_dir(0.0, -5.0));
        let north = band(lb_dir(0.0, 5.0));
        assert!(south >= 0.9 * north, "south {south} north {north}");
    }

    /// Baade's window, (l, b) = (1°, −3.9°): A_V varies from 1.26 to 2.79 mag across the field
    /// (Stanek 1996, ApJ 460, L37). Summed to 10 kpc, past the bulge.
    #[test]
    fn baades_window_extinction() {
        let g = GalaxyModel::milky_way();
        let a_v = a_v_from_sun(&g, lb_dir(1.0, -3.9), 10_000.0);
        assert!((1.0..=3.0).contains(&a_v), "A_V in Baade's window {a_v}");
    }

    /// The dust layer is symmetric about its midplane z_d = 19.5 pc (not z = 0), the rate in that
    /// midplane below the Sun is the 1 mag/kpc the brief pins, and z_d reaches the GPU in
    /// `GalaxyGpu::dust.w`, which `gal_extinction_at` reads.
    #[test]
    fn dust_midplane_at_gas_layer() {
        let g = GalaxyModel::milky_way();
        let sun = g.sun_galactocentric_pc();
        let z_d = 19.5;
        assert_eq!(g.gpu().dust[3], z_d as f32);
        let rate = |z: f64| g.extinction_v_per_pc(DVec3::new(sun.x, sun.y, z));
        assert!((rate(z_d) / 1e-3 - 1.0).abs() < 1e-12, "{}", rate(z_d));
        for dz in [5.5, 80.0, 300.0] {
            let (above, below) = (rate(z_d + dz), rate(z_d - dz));
            assert!(
                (above / below - 1.0).abs() < 1e-12,
                "±{dz} pc: {above} {below}"
            );
        }
    }

    /// Every position and direction gives a finite, non-negative band, including a camera at
    /// Sgr A*, far above the disk and inside the bar.
    #[test]
    fn band_finite_everywhere() {
        let g = GalaxyModel::milky_way();
        let cams = [
            g.sun_galactocentric_pc(),
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, 50_000.0),
            DVec3::new(1500.0, -700.0, 30.0),
        ];
        for cam in cams {
            for l in (0..360).step_by(30) {
                for b in [-90.0, -45.0, -5.0, 0.0, 5.0, 45.0, 90.0] {
                    let rgb = g.band_radiance(cam, lb_dir(l as f64, b), 10.0);
                    assert!(
                        rgb.is_finite() && rgb.min_element() >= 0.0,
                        "{cam} {l} {b}: {rgb}"
                    );
                }
            }
        }
    }

    /// Reid et al. 2019 Table 2: the Sagittarius–Carina arm passes through R = 6.04 kpc at
    /// β = 24° (R₀ = 8.15 kpc, β = 0 toward the Sun, increasing with Galactic rotation), i.e. at
    /// heliocentric (R₀ − R cos β, R sin β) = l 43°, d 3.6 kpc. Perseus: 8.87 kpc at β = 40°.
    #[test]
    fn arms_pass_through_reid_reference_points() {
        let g = GalaxyModel::milky_way();
        let sun = g.sun_galactocentric_pc();
        for (r_kpc, beta_deg) in [(6.04, 24.0), (8.87, 40.0), (4.91, 23.0), (12.24, 18.0)] {
            let (r, beta) = (r_kpc * 1000.0, f64::to_radians(beta_deg));
            let helio = DVec3::new(8150.0 - r * beta.cos(), r * beta.sin(), -sun.z);
            let a = g.arm_strength(sun + helio);
            assert!(a > 0.99, "arm at R {r_kpc} kpc β {beta_deg}°: {a}");
        }
        assert!(g.arm_strength(sun) < 0.05, "the Sun lies between the arms");
    }

    /// The bar's stars integrate to the stellar mass of BHG16 §4.2.4 at the local stars per M☉.
    #[test]
    fn bar_mass_matches_source() {
        let g = GalaxyModel::milky_way();
        let (h, n) = (50.0, 120_i32);
        let mut stars = 0.0;
        for i in -n..n {
            for j in -n..n {
                for k in -n / 4..n / 4 {
                    let p = DVec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5) * h;
                    stars += g.bar_density(p) * h * h * h;
                }
            }
        }
        let mass = stars * STELLAR_MASS_DENSITY_SUN / lf::local_number_density();
        assert!((mass / BAR_MASS_MSUN - 1.0).abs() < 0.02, "{mass:e} M☉");
    }

    #[test]
    fn galaxy_gpu_layout_matches_wgsl() {
        let module = naga::front::wgsl::parse_str(GALAXY_WGSL_SRC)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(GALAXY_WGSL_SRC)));
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("GalaxyGpu"))
            .expect("struct GalaxyGpu in galaxy.wgsl");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("GalaxyGpu is not a struct");
        };
        assert_eq!(layouter[handle].size as usize, size_of::<GalaxyGpu>());
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust: Vec<(String, usize)> = [
            ("disk", offset_of!(GalaxyGpu, disk)),
            ("norm", offset_of!(GalaxyGpu, norm)),
            ("halo", offset_of!(GalaxyGpu, halo)),
            ("bar", offset_of!(GalaxyGpu, bar)),
            ("bar_scale", offset_of!(GalaxyGpu, bar_scale)),
            ("arm_shape", offset_of!(GalaxyGpu, arm_shape)),
            ("arm_taper", offset_of!(GalaxyGpu, arm_taper)),
            ("arms", offset_of!(GalaxyGpu, arms)),
            ("dust", offset_of!(GalaxyGpu, dust)),
            ("dust_flare", offset_of!(GalaxyGpu, dust_flare)),
            ("light", offset_of!(GalaxyGpu, light)),
            ("color_thin", offset_of!(GalaxyGpu, color_thin)),
            ("color_old", offset_of!(GalaxyGpu, color_old)),
            ("color_bar", offset_of!(GalaxyGpu, color_bar)),
            ("lf", offset_of!(GalaxyGpu, lf)),
        ]
        .into_iter()
        .map(|(n, o)| (n.to_owned(), o))
        .collect();
        assert_eq!(wgsl, rust);
    }

    /// CLAUDE.md §3: every WGSL file is validated by a naga test in the crate that owns it.
    #[test]
    fn galaxy_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(GALAXY_WGSL_SRC)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(GALAXY_WGSL_SRC)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(GALAXY_WGSL_SRC)));
        for name in [
            "gal_density",
            "gal_luminosity",
            "gal_extinction",
            "gal_band_radiance",
        ] {
            assert!(
                module
                    .functions
                    .iter()
                    .any(|(_, f)| f.name.as_deref() == Some(name)),
                "galaxy.wgsl has no fn {name}"
            );
        }
    }
}
