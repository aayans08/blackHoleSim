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
//! **Detail** (Task B4L; on screen as "Milky Way detail: procedural (real dust map in M4)"): the
//! smooth model above is the *mean*; three seeded, deterministic fields break it up, each keeping
//! its mean approximately (to the ~10 % a non-Gaussian fBm allows,
//! `fbm_variance_and_lognormal_mean`) so the calibrated extinction and light stay what the
//! smooth model says:
//! - the dust is clumpy: κ × D(p), D log-normal with σ_ln = [`DUST_LN_SIGMA`] (the 3-D density of
//!   supersonic interstellar turbulence is log-normal, Padoan, Nordlund & Jones 1997) on an fBm of
//!   value noise from the turbulent outer scale [`DUST_NOISE_SCALE_PC`] down to sub-pc;
//! - the thin disk's light is clumped into star clouds on the scale of star complexes,
//!   [`STAR_NOISE_SCALE_PC`] (Efremov 1995), with σ_ln = [`STAR_LN_SIGMA`];
//! - twelve named local dust clouds ([`NAMED_CLOUDS`]: the Cygnus and Aquila rifts, Ophiuchus, the
//!   Pipe, the Coalsack, Taurus, …) sit where they are on the sky at their measured distances
//!   (Zucker et al. 2020), as Gaussian ellipsoids broken up by their own fBm. They make the Great
//!   Rift run from Cygnus to Sagittarius north of the plane, branching and breaking up.
//!
//! The fields are fixed in the galaxy (they move with parallax), but their shapes are made up:
//! only the named clouds' positions, distances and sizes are real, and those approximately. Each
//! field fades out the octaves finer than the footprint of the sample (the environment cube's
//! texel), and lowers its log-normal mean term with the variance it keeps, so a coarse sample is
//! (approximately) the mean of fine ones. [`GalaxyModel::band_extinction_mag`] integrates the
//! dust the band renders (smooth × clumps + named clouds); the range checks of A5 run on it as
//! well as on the smooth mean.
//!
//! v0 approximations, all on purpose and labelled on screen as "galaxy model v0": every component
//! uses the local mean V luminosity per star (local M/L_V = 0.043 / 0.0519 = 0.83), so the old
//! thick disk, halo and bar, which lack the local young bright stars, are too bright per star;
//! extinction is grey (no reddening) and ignored in the faint-star cut; the smooth dust layer
//! is the mean of the real one, and its clumps and clouds are procedural except for the named
//! clouds' places (see `DUST_Z_MID_PC` and **Detail**); the arms are smooth log spirals with no
//! Local (Orion) arm; the stellar disks have no warp or flare, and there is no nuclear star
//! cluster.

use std::f64::consts::{LN_10, PI, TAU};

use glam::{DVec2, DVec3, UVec3};
use uni_core::{M_V_SUN, blackbody_linear_srgb, icrs_to_galactic, pcg3d, star_illuminance_lux};

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

// ---- Detail (Task B4L) ------------------------------------------------------------------------
// Three mean-preserving log-normal fields and twelve named clouds (module doc, **Detail**). The
// noise is value noise on the integer lattice, hashed with pcg3d (Jarzynski & Olano 2020; the
// engine's one integer hash, `uni_core::pcg3d`, copied bit for bit as `gal_pcg3d` in WGSL), in
// coordinates relative to the Sun so the f32 twin keeps its digits near the camera.

/// Octave lacunarity: each octave's wavelength is the previous one ÷ 2.03, a non-integer ratio so
/// the octaves' lattices never line up (common fBm practice, as in the owner-approved preview's
/// `fbm`).
const NOISE_LACUNARITY: f64 = 2.03;
/// Amplitude ratio of successive octaves (fBm gain 0.7, Hurst exponent −log₂ 0.7 ≈ 0.51). A
/// procedural choice: rough enough that the projected dust keeps structure at every scale, in the
/// spirit of the shallow k^−2.65 power spectrum of dust emission (Miville-Deschênes et al. 2010,
/// A&A 518, L104), not fitted to it. 0.8 was tried first on CPU previews of the band and gave
/// uniform small blotches everywhere; 0.7 keeps larger coherent lanes.
const NOISE_GAIN: f64 = 0.7;
/// Each octave fades out as its wavelength λ shrinks from 1/0.3 to 1/0.6 of the sample's footprint
/// (the preview's `fbm` band limit): finer structure would only alias between texels.
const NOISE_LOD_FADE: [f64; 2] = [0.3, 0.6];
/// Variance of one octave of value noise at a uniformly random point. The 8 lattice values are
/// independent and uniform on [−1, 1) (variance 1/3); with smoothstep weights s(u) = 3u² − 2u³
/// per axis, E[(1 − s)² + s²] = 26/35 for u uniform, so Var = (26/35)³ / 3 = 0.1367
/// (`value_noise_variance_matches_theory` samples it).
const VALUE_NOISE_VAR: f64 = (26.0 / 35.0) * (26.0 / 35.0) * (26.0 / 35.0) / 3.0;
/// The octaves rotate by this axis and angle between levels (a generic, non-axis-aligned
/// rotation, so the value-noise lattice leaves no axis-aligned streaks): axis (2, 3, 6)/7, 0.9 rad.
const NOISE_ROT_AXIS: [f64; 3] = [2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0];
const NOISE_ROT_ANGLE: f64 = 0.9;

/// σ of ln D, the clumpy dust's log-normal factor (mean 1). Supersonic isothermal turbulence gives
/// a log-normal density with σ_ln² = ln(1 + b²𝓜²) (Padoan, Nordlund & Jones 1997, MNRAS 288, 145;
/// b ≈ 0.3–1, Federrath, Klessen & Schmidt 2008, ApJ 688, L79), 1 to 2 for the Mach numbers 𝓜 ≈
/// 3–10 of the cold ISM; 1.2 is a choice in that range, held to measured columns by
/// `extinction_ranges_on_the_rendered_dust`: at 1.6 (B4L's first value) the clumps opened
/// Baade's window to A_V ≈ 1.07, below the 1.26–2.79 Stanek 1996 measures there, and the
/// brightest sample of a 0.5° grid over the bulge (|l| < 30°, |b| < 10°) reached 4.7e-3 cd/m²;
/// at 1.2 the window is 1.32 and that sample 3.5e-3.
const DUST_LN_SIGMA: f64 = 1.2;
/// Wavelength of the dust field's largest octave [pc]: the turbulent outer scale of the ISM,
/// ~100 pc (Haverkorn et al. 2008, ApJ 680, 362, from Faraday rotation in interarm regions).
const DUST_NOISE_SCALE_PC: f64 = 100.0;
/// Octaves of the dust field: 100 pc down to 100/2.03⁷ = 0.7 pc (only near the camera; farther
/// away the footprint fades the fine ones).
const DUST_NOISE_OCTAVES: u32 = 8;
/// σ of ln S, the thin disk's star-cloud factor (mean 1): a procedural choice, mild enough that the
/// band's large-scale light stays the smooth model's.
const STAR_LN_SIGMA: f64 = 0.5;
/// Wavelength of the star field's largest octave [pc]: star complexes, the largest groupings of
/// young stars, are ~600 pc across (Efremov 1995, AJ 110, 2757).
const STAR_NOISE_SCALE_PC: f64 = 600.0;
/// Octaves of the star field: 600, 296 and 146 pc.
const STAR_NOISE_OCTAVES: u32 = 3;
/// σ of ln of the named clouds' internal structure (mean 1): a procedural choice that breaks each
/// Gaussian envelope into cores, filaments and holes. Column-density PDFs of nearby clouds are
/// log-normal with σ_ln ≈ 0.3–0.6 above A_V ≈ 1 (Kainulainen et al. 2009, A&A 508, L35); the
/// larger value here also stands for the envelope's own patchiness.
const CLOUD_LN_SIGMA: f64 = 1.2;
/// Wavelength of the cloud field's largest octave [pc] and its octaves (8 pc to 0.1 pc): about a
/// cloud's own size, so each envelope breaks into several pieces.
const CLOUD_NOISE_SCALE_PC: f64 = 8.0;
const CLOUD_NOISE_OCTAVES: u32 = 7;
/// No noise within this distance of the camera [pc]: the Sun sits in the Local Bubble, a cavity
/// ~100 pc across almost free of dust (Lallement et al. 2003, A&A 411, 447), so the smooth mean
/// there is already more than the real dust.
const NOISE_MIN_PC: f64 = 30.0;
/// The dust and star fields are evaluated only on steps whose smooth optical depth reaches this
/// (elsewhere there is too little dust to clump visibly, and the thin disk is far from the ray):
/// even a 3σ clump (D ≈ e^4.8) then changes the step's transmission by under 10 %, and the
/// field's mean is the smooth value anyway.
const NOISE_MIN_DTAU: f64 = 1e-3;
/// The march stops once the optical depth behind the camera passes this: e^−15 = 3 × 10⁻⁷, so
/// everything farther adds under 3 × 10⁻⁵ of the band (16× below the cube's f16 rounding), even
/// toward the bulge behind the plane's lane, where the foreground is faintest
/// (`band_march_stops_only_when_nothing_is_left`; at τ = 12 that ray was off by 5 × 10⁻⁵). Most of the cube's in-plane texels stop well
/// before 30 kpc.
const BAND_TAU_STOP: f64 = 15.0;
/// Seeds of the three fields (third pcg3d lane offsets; arbitrary odd constants).
const DUST_SEED: u32 = 0x2545_F491;
const STAR_SEED: u32 = 0x9E37_79B9;
const CLOUD_SEED: u32 = 0x85EB_CA6B;
/// Octave k of a field hashes with seed + k × this (an arbitrary constant, so the octaves of one
/// field never hash the same lattice values).
const OCTAVE_SEED_STEP: u32 = 0x68E3_1DA4;
/// A named cloud is skipped for a ray that passes more than √50 = 7.1 σ from its centre (its
/// column is then below e^−25 of the peak).
const CLOUD_Q_MAX: f64 = 50.0;
/// A named cloud's column below this optical depth is left out of the march.
const CLOUD_MIN_TAU: f64 = 1e-4;
/// A named cloud whose peak lies more than this many of its σ along the ray behind the camera is
/// skipped: less than 1e-9 of its column is ahead (a ray toward Ophiuchus runs back through
/// Taurus, its antipode).
const CLOUD_BEHIND_SIGMAS: f64 = 6.0;

/// One named local dust cloud, a Gaussian ellipsoid: its centre at Galactic (l, b) [deg] and
/// distance [pc] from the Sun, its σ along its long and short axes on the sky [deg] (the long axis
/// `pa_deg` from the +l direction toward +b), its σ along the line of sight [pc], and A_V [mag]
/// through its centre from the Sun before the internal structure.
struct NamedCloud {
    l_deg: f64,
    b_deg: f64,
    d_pc: f64,
    sigma_long_deg: f64,
    sigma_short_deg: f64,
    pa_deg: f64,
    sigma_r_pc: f64,
    a_v_mag: f64,
}

/// Number of named clouds (`GalaxyGpu::clouds` holds four vec4s per cloud).
pub const NAMED_CLOUD_COUNT: usize = 12;

/// The named clouds. Distances are the dust-based Gaia DR2 ones of Zucker et al. 2020 (A&A 633,
/// A51, Table A.1; the median of each cloud's sightlines, whose (l, b) also place and size it);
/// the Aquila Rift's is Straižys, Černis & Bartašiūtė 2003 (A&A 405, 585: 225 ± 55 pc). Sizes and
/// position angles are read off those sightlines and the well-known shapes on the sky
/// (approximate). A_V through the centre is a v0 choice of 1.5–3 mag, the order of the cloud
/// envelopes those distances were measured through (0.15 < E(B−V) < 5 mag, Zucker et al. 2020
/// §2.1); dense cores far above that are left to the internal structure.
const NAMED_CLOUDS: [NamedCloud; NAMED_CLOUD_COUNT] = [
    // Cygnus Rift (Northern Coalsack ≈ 560 pc at l 91–93°, b +3.5…+4.3°; North America ≈ 790 pc
    // at l 82–86°, b −2.2…0°): the northern end of the Great Rift. A_V 2.5 (B4L's first 3.0 put
    // its mean column toward l 90°, b 0° at 1.16 mag and the whole mean A_V within 1 kpc at
    // 2.09, past the 0.5–2.0 the plane's 0.7–1.8 mag/kpc allow; 2.5 gives 0.97 and 1.89).
    cloud(80.0, 1.5, 650.0, 8.0, 2.5, 0.0, 80.0, 2.5),
    // Serpens–Aquila (W40, Serpens Main and Low: 440–560 pc at l 28–33°, b +2.6…+5.2°).
    cloud(30.0, 3.5, 480.0, 3.5, 1.5, 0.0, 50.0, 3.0),
    // Aquila Rift (Straižys et al. 2003; Zucker et al. 2020's sightlines at l 16–22°, b +9…+17°):
    // from l 40°, b 0° up toward Ophiuchus.
    cloud(27.0, 7.0, 225.0, 10.0, 3.0, 149.0, 30.0, 2.0),
    // Ophiuchus (ρ Oph and its streamers: 118–167 pc, median 139 pc, at l 349–359°, b +12…+18°).
    cloud(354.0, 15.5, 139.0, 4.5, 2.5, 0.0, 15.0, 2.5),
    // Pipe Nebula (B59 at l 356.9°, b +7.3°: 180 pc), its stem running toward l 1°, b +4°.
    cloud(359.3, 5.5, 180.0, 3.5, 0.8, 155.0, 15.0, 2.0),
    // Lupus clouds (108–239 pc, median ~160 pc, at l 339–347°, b +6…+17°).
    cloud(341.0, 12.0, 160.0, 4.5, 3.5, 90.0, 20.0, 1.5),
    // Corona Australis (147–165 pc at l 359.5–0.8°, b −17.8…−21.0°).
    cloud(0.0, -19.5, 155.0, 2.5, 1.0, 160.0, 15.0, 2.0),
    // The Coalsack (182–193 pc at l 300–303°, b −2.6…+3.1°).
    cloud(301.5, -0.5, 188.0, 2.6, 2.2, 0.0, 15.0, 1.8),
    // Chamaeleon (161–210 pc at l 297.5–303.3°, b −14.2…−16.7°).
    cloud(300.0, -15.5, 190.0, 3.0, 1.5, 0.0, 20.0, 1.5),
    // Taurus (129–170 pc, median 145 pc, at l 166–176°, b −12…−19°).
    cloud(172.0, -15.0, 145.0, 6.0, 3.5, 0.0, 20.0, 2.0),
    // Perseus (256–331 pc, median 295 pc, at l 157–161°, b −16…−18°).
    cloud(159.0, -17.5, 295.0, 2.5, 1.0, 20.0, 30.0, 2.0),
    // Aquila South (123–143 pc at l 37–39°, b −15…−19°).
    cloud(38.0, -17.0, 132.0, 2.0, 1.5, 0.0, 15.0, 1.5),
];

/// [`NamedCloud`] in the order of its fields (a `const fn` so the table reads as rows).
#[allow(clippy::too_many_arguments)]
const fn cloud(
    l_deg: f64,
    b_deg: f64,
    d_pc: f64,
    sigma_long_deg: f64,
    sigma_short_deg: f64,
    pa_deg: f64,
    sigma_r_pc: f64,
    a_v_mag: f64,
) -> NamedCloud {
    NamedCloud {
        l_deg,
        b_deg,
        d_pc,
        sigma_long_deg,
        sigma_short_deg,
        pa_deg,
        sigma_r_pc,
        a_v_mag,
    }
}

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
    detail_sigma: [f64; 4],
    detail_scale: [f64; 4],
    detail_octaves: [u32; 4],
    detail_seed: [u32; 4],
    noise_rot: [[f64; 4]; 3],
    clouds: [[f64; 4]; 4 * NAMED_CLOUD_COUNT],
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
    /// Detail (B4L): σ_ln of the dust, star and cloud fields; fBm gain.
    pub detail_sigma: [f32; 4],
    /// 1 / largest wavelength [1/pc] of the dust, star and cloud fields; no noise nearer than w [pc].
    pub detail_scale: [f32; 4],
    /// Octaves of the dust, star and cloud fields; number of named clouds.
    pub detail_octaves: [u32; 4],
    /// Seeds of the dust, star and cloud fields; 0.
    pub detail_seed: [u32; 4],
    /// Rows of the rotation between octaves; w of row 0: the lacunarity; w of row 1: z☉ [pc] (the
    /// Sun is at (−R₀, 0, z☉)); w of row 2: 0.
    pub noise_rot: [[f32; 4]; 3],
    /// Per named cloud, four vec4s: centre − Sun [pc] and peak A_V rate [mag/pc]; then the rows of
    /// its whitening matrix W (W (x − c) has unit σ on every axis) [1/pc], w = 0.
    pub clouds: [[f32; 4]; 4 * NAMED_CLOUD_COUNT],
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
            detail_sigma: [DUST_LN_SIGMA, STAR_LN_SIGMA, CLOUD_LN_SIGMA, NOISE_GAIN],
            detail_scale: [
                1.0 / DUST_NOISE_SCALE_PC,
                1.0 / STAR_NOISE_SCALE_PC,
                1.0 / CLOUD_NOISE_SCALE_PC,
                NOISE_MIN_PC,
            ],
            detail_octaves: [
                DUST_NOISE_OCTAVES,
                STAR_NOISE_OCTAVES,
                CLOUD_NOISE_OCTAVES,
                NAMED_CLOUD_COUNT as u32,
            ],
            detail_seed: [DUST_SEED, STAR_SEED, CLOUD_SEED, 0],
            noise_rot: noise_rotation(),
            clouds: [[0.0; 4]; 4 * NAMED_CLOUD_COUNT],
        };
        // Total star density at the Sun = the luminosity function's count.
        let sun_shape = g.components(g.sun, g.arm_strength(g.sun));
        g.norm[2] = (n_sun - sun_shape[3]) / (sun_shape[0] + sun_shape[1] + sun_shape[2]);
        // A_V rate in the dust midplane directly below the Sun = DUST_A_V_SUN_MAG_PER_PC.
        let plane = DVec3::new(g.sun.x, g.sun.y, DUST_Z_MID_PC);
        g.dust[0] = DUST_A_V_SUN_MAG_PER_PC / g.extinction(plane, g.arm_strength(plane));
        for (k, c) in NAMED_CLOUDS.iter().enumerate() {
            g.clouds[4 * k..4 * k + 4].copy_from_slice(&cloud_gpu(c));
        }
        g
    }

    /// The smooth model v0 alone: [`Self::milky_way`] with the detail switched off (every
    /// log-normal σ and every named cloud's extinction 0), so its band is the mean the detail
    /// varies about.
    pub fn milky_way_smooth() -> Self {
        let mut g = Self::milky_way();
        g.detail_sigma[0] = 0.0;
        g.detail_sigma[1] = 0.0;
        g.detail_sigma[2] = 0.0;
        for k in 0..NAMED_CLOUD_COUNT {
            g.clouds[4 * k][3] = 0.0;
        }
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
    /// fainter than `v_lim`, for a sample whose footprint on the sky is `footprint_rad` [rad]
    /// (the detail's octaves finer than it fade out; 0 keeps them all); the CPU twin of
    /// `gal_band_radiance`.
    ///
    /// ∫ j(s) f_faint(s) e^{−τ(s)} ds from 1 pc to 30 kpc in 128 log-spaced steps: j is the light
    /// at the step's geometric midpoint, f_faint = [`lf::faint_light_fraction`]`(s, v_lim)` keeps
    /// only stars fainter than the catalogue limit (brighter ones are drawn one by one), and
    /// within a step the dust attenuates exactly as a uniform slab: ∫₀^Δs e^{−κt} dt =
    /// (1 − e^{−κΔs}) / κ.
    ///
    /// Detail (module doc): on each step from `NOISE_MIN_PC` on whose smooth optical depth reaches
    /// `NOISE_MIN_DTAU`, the dust is × D and the thin disk's light × S (log-normal, mean 1); each
    /// named cloud adds its exact Gaussian column between the step's ends, τ_c [Φ(s₁) − Φ(s₀)]
    /// with Φ the normal CDF along the ray, τ_c being its whole column on this ray × its own
    /// log-normal structure at the ray's closest approach.
    pub fn band_radiance(
        &self,
        cam_gc_pc: DVec3,
        dir_gc: DVec3,
        v_lim: f64,
        footprint_rad: f64,
    ) -> DVec3 {
        self.band_march(cam_gc_pc, dir_gc, v_lim, footprint_rad, BAND_TAU_STOP)
    }

    /// [`Self::band_radiance`], stopping once the optical depth passes `tau_stop`.
    fn band_march(
        &self,
        cam_gc_pc: DVec3,
        dir_gc: DVec3,
        v_lim: f64,
        footprint_rad: f64,
        tau_stop: f64,
    ) -> DVec3 {
        let dir = dir_gc.normalize_or_zero();
        let x_lim = 10_f64.powf(-0.4 * v_lim);
        let ratio = BAND_S_MAX_PC / BAND_S_MIN_PC;
        let footprint = footprint_rad.max(0.0);
        // Noise and clouds work relative to the Sun (f32 precision near the camera on the GPU).
        let cam_rel = cam_gc_pc - self.sun;
        let clouds = self.ray_clouds(cam_rel, dir, footprint);
        let mut cloud_tau0 = clouds.column_to(BAND_S_MIN_PC);
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
            let (mut dtau, star_clouds) =
                self.step_dust(p, arms, cam_rel + dir * s, s, ds, footprint);
            let cloud_tau1 = clouds.column_to(s1);
            dtau += cloud_tau1 - cloud_tau0;
            cloud_tau0 = cloud_tau1;
            let rgb = thin * star_clouds * self.color_thin
                + (thick + halo) * self.color_old
                + bar * self.color_bar;
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
            if tau > tau_stop {
                break;
            }
        }
        sum * (self.norm[3] * self.light[0])
    }

    /// One march step's clumpy dust and star-cloud factor: the step at midpoint `s` [pc] of
    /// length `ds`, at galactocentric `p` with arm strength `arms`, `q` = p − Sun. Returns
    /// (optical depth of the smooth layer × D over the step, S for the thin disk's light); D and
    /// S are drawn only past `NOISE_MIN_PC` on steps whose smooth optical depth reaches
    /// `NOISE_MIN_DTAU`, else 1. Shared by [`Self::band_radiance`] and
    /// [`Self::band_extinction_mag`], so the range checks see the dust the band renders.
    fn step_dust(
        &self,
        p: DVec3,
        arms: f64,
        q: DVec3,
        s: f64,
        ds: f64,
        footprint: f64,
    ) -> (f64, f64) {
        let [sigma_dust, sigma_star, _, _] = self.detail_sigma;
        let [k_dust, k_star, _, _] = self.detail_octaves;
        let [seed_dust, seed_star, _, _] = self.detail_seed;
        let [f_dust, f_star, _, noise_min_pc] = self.detail_scale;
        let mut dtau = self.extinction(p, arms) * TAU_PER_MAG * ds;
        let mut star_clouds = 1.0;
        if s > noise_min_pc && dtau >= NOISE_MIN_DTAU {
            let fp = footprint * s;
            let (n, kept) = self.fbm(q, f_dust, k_dust, seed_dust, fp);
            dtau *= (sigma_dust * n - 0.5 * sigma_dust * sigma_dust * kept).exp();
            let (n, kept) = self.fbm(q, f_star, k_star, seed_star, fp);
            star_clouds = (sigma_star * n - 0.5 * sigma_star * sigma_star * kept).exp();
        }
        (dtau, star_clouds)
    }

    /// A_V [mag] the band's march crosses from `cam_gc_pc` along `dir_gc` out to `s_max_pc`, at a
    /// footprint of `footprint_rad`: the march's own steps (1 pc to 30 kpc, 128 log-spaced), its
    /// clumpy dust ([`Self::step_dust`], smooth × D with the march's gating) and the named clouds'
    /// exact columns. The step that holds `s_max_pc` adds the share (s_max − s₀)/Δs of its dust
    /// and the clouds' column up to `s_max_pc`. The A5 range checks run on this, so they see the
    /// dust the band renders, not only the smooth mean ([`Self::extinction_v_per_pc`]).
    pub fn band_extinction_mag(
        &self,
        cam_gc_pc: DVec3,
        dir_gc: DVec3,
        s_max_pc: f64,
        footprint_rad: f64,
    ) -> f64 {
        let dir = dir_gc.normalize_or_zero();
        let ratio = BAND_S_MAX_PC / BAND_S_MIN_PC;
        let footprint = footprint_rad.max(0.0);
        let cam_rel = cam_gc_pc - self.sun;
        let clouds = self.ray_clouds(cam_rel, dir, footprint);
        let s_max = s_max_pc.clamp(BAND_S_MIN_PC, BAND_S_MAX_PC);
        let mut s0 = BAND_S_MIN_PC;
        let mut tau = 0.0;
        for i in 1..=BAND_STEPS {
            let s1 = BAND_S_MIN_PC * ratio.powf(f64::from(i) / f64::from(BAND_STEPS));
            let s = (s0 * s1).sqrt();
            let ds = s1 - s0;
            let p = cam_gc_pc + dir * s;
            let (dtau, _) =
                self.step_dust(p, self.arm_strength(p), cam_rel + dir * s, s, ds, footprint);
            tau += dtau * ((s_max - s0) / ds).clamp(0.0, 1.0);
            if s1 >= s_max {
                break;
            }
            s0 = s1;
        }
        let cloud_tau = clouds.column_to(s_max) - clouds.column_to(BAND_S_MIN_PC);
        (tau + cloud_tau) / TAU_PER_MAG
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
            detail_sigma: f(self.detail_sigma),
            detail_scale: f(self.detail_scale),
            detail_octaves: self.detail_octaves,
            detail_seed: self.detail_seed,
            noise_rot: self.noise_rot.map(f),
            clouds: self.clouds.map(f),
        }
    }

    /// fBm of value noise at `q` [pc, relative to the Sun]: octave k has wavelength
    /// λ_k = λ₀ / 2.03^k (`inv_scale` = 1/λ₀), amplitude gain^k, and its lattice rotated by the
    /// octave rotation k times; `octaves` of them, each × (1 − smoothstep(0.3, 0.6, fp/λ_k)) for a
    /// footprint of `fp_pc` [pc]. Returns (n, kept): n scaled so that with every octave kept its
    /// variance is 1, and kept = the share of that variance the footprint keeps (Σ w²a² / Σ a²).
    /// Twin of `gal_fbm`.
    // `!(x > 0)` on purpose: a NaN weight stops the loop, as in the WGSL twin.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn fbm(&self, q: DVec3, inv_scale: f64, octaves: u32, seed: u32, fp_pc: f64) -> (f64, f64) {
        let gain = self.detail_sigma[3];
        let lacunarity = self.noise_rot[0][3];
        let rot = self.noise_rot.map(|r| DVec3::new(r[0], r[1], r[2]));
        // Σ a_k² over all the octaves: a geometric series.
        let g2 = gain * gain;
        let total = (1.0 - g2.powi(octaves as i32)) / (1.0 - g2);
        let mut p = q * inv_scale;
        let mut inv_lambda = inv_scale;
        let mut amp = 1.0;
        let (mut sum, mut kept) = (0.0, 0.0);
        for k in 0..octaves {
            let w = 1.0 - smoothstep(NOISE_LOD_FADE[0], NOISE_LOD_FADE[1], fp_pc * inv_lambda);
            if !(w > 0.0) {
                break;
            }
            sum += w * amp * value_noise(p, seed.wrapping_add(k.wrapping_mul(OCTAVE_SEED_STEP)));
            kept += w * w * amp * amp;
            p = DVec3::new(rot[0].dot(p), rot[1].dot(p), rot[2].dot(p)) * lacunarity;
            inv_lambda *= lacunarity;
            amp *= gain;
        }
        (sum / (total * VALUE_NOISE_VAR).sqrt(), kept / total)
    }

    /// The named clouds along the ray from `cam_rel` [pc, relative to the Sun] in unit direction
    /// `dir`, for a footprint of `footprint` [rad]. Twin of the cloud set-up in
    /// `gal_band_radiance`.
    // `!(x > 0)` and `!(q < max)` on purpose: a NaN skips the cloud, as in the WGSL twin.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn ray_clouds(&self, cam_rel: DVec3, dir: DVec3, footprint: f64) -> RayClouds {
        let mut out = RayClouds::default();
        let [_, _, sigma, _] = self.detail_sigma;
        let [_, _, inv_scale, _] = self.detail_scale;
        let [_, _, octaves, count] = self.detail_octaves;
        let [_, _, seed, _] = self.detail_seed;
        for k in 0..(count as usize).min(NAMED_CLOUD_COUNT) {
            let c = &self.clouds[4 * k..4 * k + 4];
            let kappa0 = c[0][3];
            if !(kappa0 > 0.0) {
                continue;
            }
            let row = |i: usize| DVec3::new(c[i][0], c[i][1], c[i][2]);
            let v = DVec3::new(c[0][0], c[0][1], c[0][2]) - cam_rel;
            let dw = DVec3::new(row(1).dot(dir), row(2).dot(dir), row(3).dot(dir));
            let vw = DVec3::new(row(1).dot(v), row(2).dot(v), row(3).dot(v));
            let a = dw.length_squared();
            let s_mid = dw.dot(vw) / a;
            let q = (vw - dw * s_mid).length_squared();
            if !(q < CLOUD_Q_MAX) || s_mid * a.sqrt() < -CLOUD_BEHIND_SIGMAS {
                continue;
            }
            let mut tau = kappa0 * (-0.5 * q).exp() * (TAU / a).sqrt() * TAU_PER_MAG;
            let s_near = s_mid.max(0.0);
            let (n, kept) = self.fbm(
                cam_rel + dir * s_near,
                inv_scale,
                octaves,
                seed.wrapping_add(k as u32),
                footprint * s_near.max(1.0),
            );
            tau *= (sigma * n - 0.5 * sigma * sigma * kept).exp();
            if tau >= CLOUD_MIN_TAU {
                out.tau[k] = tau;
                out.s_mid[k] = s_mid;
                out.sqrt_a[k] = a.sqrt();
            }
        }
        out
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

/// Rows of the octave rotation (Rodrigues' formula about `NOISE_ROT_AXIS` by
/// `NOISE_ROT_ANGLE`); row 0's w carries the lacunarity, row 1's z☉ (`GalaxyGpu::noise_rot`).
fn noise_rotation() -> [[f64; 4]; 3] {
    let [x, y, z] = NOISE_ROT_AXIS;
    let (s, c) = NOISE_ROT_ANGLE.sin_cos();
    let t = 1.0 - c;
    [
        [
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            NOISE_LACUNARITY,
        ],
        [
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            Z_SUN_PC,
        ],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
    ]
}

/// Unit vector toward Galactic (l, b) [deg]; also the galactocentric axes.
fn galactic_dir(l_deg: f64, b_deg: f64) -> DVec3 {
    let (l, b) = (l_deg.to_radians(), b_deg.to_radians());
    DVec3::new(b.cos() * l.cos(), b.cos() * l.sin(), b.sin())
}

/// A named cloud packed for `GalaxyGpu::clouds`: (centre − Sun, κ₀), then the rows of W.
///
/// The cloud's axes are e_r (toward it from the Sun) and, on the sky, e_l and e_b (toward +l and
/// +b) turned by its position angle; W = Σ eᵢ/σᵢ as rows, so the extinction rate is
/// κ₀ exp(−|W (x − c)|²/2). κ₀ = A_V / (√(2π) σ_r) gives A_V on the line from the Sun through c.
fn cloud_gpu(c: &NamedCloud) -> [[f64; 4]; 4] {
    let (l, b) = (c.l_deg.to_radians(), c.b_deg.to_radians());
    let e_r = galactic_dir(c.l_deg, c.b_deg);
    let e_l = DVec3::new(-l.sin(), l.cos(), 0.0);
    let e_b = DVec3::new(-b.sin() * l.cos(), -b.sin() * l.sin(), b.cos());
    let (sp, cp) = c.pa_deg.to_radians().sin_cos();
    let e_long = e_l * cp + e_b * sp;
    let e_short = e_b * cp - e_l * sp;
    let sigma_long = c.d_pc * c.sigma_long_deg.to_radians();
    let sigma_short = c.d_pc * c.sigma_short_deg.to_radians();
    let centre = e_r * c.d_pc;
    let kappa0 = c.a_v_mag / ((TAU).sqrt() * c.sigma_r_pc);
    let row = |e: DVec3, sigma: f64| [e.x / sigma, e.y / sigma, e.z / sigma, 0.0];
    [
        [centre.x, centre.y, centre.z, kappa0],
        row(e_r, c.sigma_r_pc),
        row(e_long, sigma_long),
        row(e_short, sigma_short),
    ]
}

/// One octave of value noise at `p` [lattice units]: the 8 lattice values around it, uniform on
/// [−1, 1) from `pcg3d(cell, cell, cell ^ seed)` (24 bits of the first lane), blended with
/// smoothstep weights. Continuous across cells, so the f32 twin agrees where the two round the
/// cell index differently. Twin of `gal_value_noise`.
fn value_noise(p: DVec3, seed: u32) -> f64 {
    let cell = p.floor();
    let f = p - cell;
    let u = f * f * (DVec3::splat(3.0) - 2.0 * f);
    let base = [
        cell.x as i32 as u32,
        cell.y as i32 as u32,
        cell.z as i32 as u32,
    ];
    let corner = |dx: u32, dy: u32, dz: u32| {
        let h = pcg3d(UVec3::new(
            base[0].wrapping_add(dx),
            base[1].wrapping_add(dy),
            base[2].wrapping_add(dz) ^ seed,
        ));
        f64::from(h.x >> 8) * (2.0 / 16_777_216.0) - 1.0
    };
    let lerp = |a: f64, b: f64, t: f64| a + t * (b - a);
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), u.x);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), u.x);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), u.x);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

/// erf(x) by Abramowitz & Stegun 1964, *Handbook of Mathematical Functions*, eq. 7.1.26 (after
/// Hastings 1955), |error| ≤ 1.5e-7: the formula the GPU twin `gal_erf` can afford.
#[allow(clippy::excessive_precision)] // A&S's published digits.
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-a * a).exp();
    if x >= 0.0 { y } else { -y }
}

/// The named clouds one ray meets: per cloud its whole column τ_c [optical depth], the distance
/// s_m of its density peak along the ray, and √a, the inverse of its σ along the ray (τ_c = 0 for
/// a cloud the ray misses).
#[derive(Default)]
struct RayClouds {
    tau: [f64; NAMED_CLOUD_COUNT],
    s_mid: [f64; NAMED_CLOUD_COUNT],
    sqrt_a: [f64; NAMED_CLOUD_COUNT],
}

impl RayClouds {
    /// The clouds' optical depth from −∞ to `s` along the ray: Σ τ_c Φ((s − s_m)√a), Φ the normal
    /// CDF ½(1 + erf(x/√2)). Differences of it give a step's exact Gaussian column.
    fn column_to(&self, s: f64) -> f64 {
        (0..NAMED_CLOUD_COUNT)
            .filter(|&k| self.tau[k] > 0.0)
            .map(|k| {
                let x = (s - self.s_mid[k]) * self.sqrt_a[k];
                self.tau[k] * 0.5 * (1.0 + erf(x * std::f64::consts::FRAC_1_SQRT_2))
            })
            .sum()
    }
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
        let band = |dir: DVec3| luminance(g.band_radiance(sun, dir, 10.0, 0.0));
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
        let band = |dir: DVec3| luminance(g.band_radiance(sun, dir, 10.0, 0.0));
        let south = band(lb_dir(0.0, -5.0));
        let north = band(lb_dir(0.0, 5.0));
        assert!(south >= 0.9 * north, "south {south} north {north}");
    }

    /// Mean A_V [mag] out to `s_max_pc` over 15 rays around (l, b) (±0.4° in l, ±0.2° in b): what
    /// the band renders at a 512² cube's texel footprint ([`GalaxyModel::band_extinction_mag`]),
    /// averaged because the clumps make neighbouring rays differ.
    fn a_v_rendered(g: &GalaxyModel, l: f64, b: f64, s_max_pc: f64) -> f64 {
        let sun = g.sun_galactocentric_pc();
        let mut sum = 0.0;
        for dl in [-0.4, -0.2, 0.0, 0.2, 0.4] {
            for db in [-0.2, 0.0, 0.2] {
                sum += g.band_extinction_mag(sun, lb_dir(l + dl, b + db), s_max_pc, 2.0 / 512.0);
            }
        }
        sum / 15.0
    }

    /// Review r0 I2: A5's range checks on the dust the band renders (smooth × clumps + named
    /// clouds), not only the smooth mean: the north Galactic pole to 30 kpc and l = 90°, b = 0° to
    /// 1 kpc in A5's ranges, and Baade's window to 10 kpc inside the 1.26–2.79 mag Stanek 1996
    /// (ApJ 460, L37) measures across it (A5's smooth check allows 1.0–3.0).
    #[test]
    fn extinction_ranges_on_the_rendered_dust() {
        let g = GalaxyModel::milky_way();
        let ngp = a_v_rendered(&g, 0.0, 90.0, 30_000.0);
        assert!((0.03..=0.15).contains(&ngp), "A_V to the NGP {ngp}");
        let l90 = a_v_rendered(&g, 90.0, 0.0, 1000.0);
        assert!(
            (0.5..=2.0).contains(&l90),
            "A_V along l = 90°, 1 kpc: {l90}"
        );
        let baade = a_v_rendered(&g, 1.0, -3.9, 10_000.0);
        assert!(
            (1.26..=2.79).contains(&baade),
            "A_V in Baade's window {baade}"
        );
    }

    /// The same ranges on the detail's mean: the clumps' σ at 0 (D = 1), the named clouds' Gaussian
    /// columns kept. Toward l = 90° the Cygnus Rift adds 0.97 mag within 1 kpc. And the helper's
    /// log-spaced quadrature agrees with A5's 1-pc sum on the smooth layer.
    #[test]
    fn extinction_ranges_on_the_mean_detail() {
        let mut g = GalaxyModel::milky_way();
        g.detail_sigma[0] = 0.0;
        g.detail_sigma[1] = 0.0;
        g.detail_sigma[2] = 0.0;
        let sun = g.sun_galactocentric_pc();
        let a_v = |l: f64, b: f64, s: f64| g.band_extinction_mag(sun, lb_dir(l, b), s, 0.0);
        assert!((0.03..=0.15).contains(&a_v(0.0, 90.0, 30_000.0)));
        let l90 = a_v(90.0, 0.0, 1000.0);
        assert!(
            (0.5..=2.0).contains(&l90),
            "A_V along l = 90°, 1 kpc: {l90}"
        );
        let smooth = GalaxyModel::milky_way_smooth();
        let l90_smooth = smooth.band_extinction_mag(sun, lb_dir(90.0, 0.0), 1000.0, 0.0);
        assert!(
            (l90 - l90_smooth - 0.97).abs() < 0.01,
            "Cygnus Rift {}",
            l90 - l90_smooth
        );
        let baade = a_v(1.0, -3.9, 10_000.0);
        assert!(
            (1.26..=2.79).contains(&baade),
            "A_V in Baade's window {baade}"
        );
        for (l, b, s_max) in [
            (0.0, 90.0, 30_000.0),
            (90.0, 0.0, 1000.0),
            (1.0, -3.9, 10_000.0),
        ] {
            let march = smooth.band_extinction_mag(sun, lb_dir(l, b), s_max, 0.0);
            let sum = a_v_from_sun(&smooth, lb_dir(l, b), s_max);
            assert!(
                (march / sum - 1.0).abs() < 0.02,
                "({l}, {b}): {march} vs {sum}"
            );
        }
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
                    let rgb = g.band_radiance(cam, lb_dir(l as f64, b), 10.0, 0.0);
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

    /// galaxy.wgsl with the module-scope `galaxy` its importers declare (its contract), so it
    /// parses and validates on its own.
    fn galaxy_wgsl_with_decl() -> String {
        format!("{GALAXY_WGSL_SRC}\n@group(0) @binding(0) var<uniform> galaxy: GalaxyGpu;\n")
    }

    #[test]
    fn galaxy_gpu_layout_matches_wgsl() {
        let module = naga::front::wgsl::parse_str(&galaxy_wgsl_with_decl())
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&galaxy_wgsl_with_decl())));
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
            ("detail_sigma", offset_of!(GalaxyGpu, detail_sigma)),
            ("detail_scale", offset_of!(GalaxyGpu, detail_scale)),
            ("detail_octaves", offset_of!(GalaxyGpu, detail_octaves)),
            ("detail_seed", offset_of!(GalaxyGpu, detail_seed)),
            ("noise_rot", offset_of!(GalaxyGpu, noise_rot)),
            ("clouds", offset_of!(GalaxyGpu, clouds)),
        ]
        .into_iter()
        .map(|(n, o)| (n.to_owned(), o))
        .collect();
        assert_eq!(wgsl, rust);
    }

    /// CLAUDE.md §3: every WGSL file is validated by a naga test in the crate that owns it.
    #[test]
    fn galaxy_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(&galaxy_wgsl_with_decl())
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&galaxy_wgsl_with_decl())));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&galaxy_wgsl_with_decl())));
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

    // ---- Detail (B4L) ----------------------------------------------------------------------

    /// `n` deterministic points spread over a 4 kpc cube around the Sun (splitmix64).
    fn sample_points(n: usize) -> Vec<DVec3> {
        let mut counter = 0x5EED_u64;
        let mut unit = || {
            counter += 1;
            (uni_core::splitmix64(counter) >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| DVec3::new(unit(), unit(), unit()) * 4000.0 - 2000.0)
            .collect()
    }

    /// One value-noise octave has mean 0 and the variance (26/35)³/3 the docs derive.
    #[test]
    fn value_noise_variance_matches_theory() {
        let pts = sample_points(200_000);
        let v: Vec<f64> = pts.iter().map(|p| value_noise(*p * 0.137, 7)).collect();
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / v.len() as f64;
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!(
            (var / VALUE_NOISE_VAR - 1.0).abs() < 0.03,
            "var {var} vs {VALUE_NOISE_VAR}"
        );
        assert!(v.iter().all(|x| (-1.0..1.0).contains(x)));
    }

    /// With every octave kept the fBm has unit variance and `kept` = 1; a coarse footprint drops
    /// octaves, and the variance left matches `kept`. The log-normal factors built from it keep a
    /// mean of 1 either way (the dust's σ_ln 1.6 is the hardest case).
    #[test]
    fn fbm_variance_and_lognormal_mean() {
        let g = GalaxyModel::milky_way();
        let pts = sample_points(40_000);
        let sigma = DUST_LN_SIGMA;
        for fp in [0.0, 30.0] {
            let nk: Vec<(f64, f64)> = pts
                .iter()
                .map(|p| {
                    g.fbm(
                        *p,
                        1.0 / DUST_NOISE_SCALE_PC,
                        DUST_NOISE_OCTAVES,
                        DUST_SEED,
                        fp,
                    )
                })
                .collect();
            let kept = nk[0].1;
            assert!(nk.iter().all(|(_, k)| *k == kept));
            let var = nk.iter().map(|(n, _)| n * n).sum::<f64>() / nk.len() as f64;
            assert!(
                (var / kept - 1.0).abs() < 0.1,
                "fp {fp}: var {var} kept {kept}"
            );
            let mean = nk
                .iter()
                .map(|(n, k)| (sigma * n - 0.5 * sigma * sigma * k).exp())
                .sum::<f64>()
                / nk.len() as f64;
            assert!((mean - 1.0).abs() < 0.12, "fp {fp}: mean D {mean}");
            if fp == 0.0 {
                assert!((kept - 1.0).abs() < 1e-12, "{kept}");
            } else {
                assert!(kept < 0.9, "fp {fp}: kept {kept}");
            }
        }
    }

    /// A ray from the Sun through a named cloud's centre crosses exactly its A_V (internal
    /// structure off), and that column dims the band behind it: the Coalsack's 1.8 mag leaves
    /// under half of the smooth band at its centre.
    #[test]
    fn named_cloud_columns_and_the_coalsack() {
        let mut g = GalaxyModel::milky_way();
        g.detail_sigma[2] = 0.0;
        for (k, c) in NAMED_CLOUDS.iter().enumerate() {
            let rc = g.ray_clouds(DVec3::ZERO, galactic_dir(c.l_deg, c.b_deg), 0.0);
            let want = c.a_v_mag * TAU_PER_MAG;
            assert!(
                (rc.tau[k] / want - 1.0).abs() < 1e-9,
                "cloud {k}: {} vs {want}",
                rc.tau[k]
            );
            assert!(
                (rc.s_mid[k] / c.d_pc - 1.0).abs() < 1e-9,
                "cloud {k}: {}",
                rc.s_mid[k]
            );
            // The whole column lies ahead of the camera.
            assert!(
                rc.column_to(1.0) < 1e-6 * want,
                "cloud {k}: {}",
                rc.column_to(1.0)
            );
            assert!((rc.column_to(1e5) - rc.tau.iter().sum::<f64>()).abs() < 1e-9);
        }
        let smooth = GalaxyModel::milky_way_smooth();
        let sun = g.sun_galactocentric_pc();
        let coalsack = lb_dir(301.5, -0.5);
        let dim = luminance(g.band_radiance(sun, coalsack, 10.0, 0.0));
        let mean = luminance(smooth.band_radiance(sun, coalsack, 10.0, 0.0));
        assert!(dim < 0.5 * mean, "Coalsack {dim} vs smooth {mean}");
    }

    /// The detail varies the band about the smooth model's: over the band (|b| ≤ 15°, all l) the
    /// clumps alone make it brighter on average, not darker (clumpy dust lets more light through
    /// than smooth dust of the same mean: e^−τ is convex), the named clouds then add dust of their
    /// own on top of the smooth layer, and the whole detail keeps the mean band within −20 % and
    /// ×2 of the smooth one (measured: 0.97 at σ_ln 1.2); neighbouring samples 0.5° apart differ,
    /// where the smooth band is nearly flat.
    #[test]
    fn detail_varies_the_band_about_its_mean() {
        let g = GalaxyModel::milky_way();
        let smooth = GalaxyModel::milky_way_smooth();
        let mut clumps = GalaxyModel::milky_way();
        for k in 0..NAMED_CLOUD_COUNT {
            clumps.clouds[4 * k][3] = 0.0;
        }
        let mut sum_c = 0.0;
        let sun = g.sun_galactocentric_pc();
        let fp = 2.0 / 512.0;
        let (mut sum_d, mut sum_s, mut step_d, mut step_s) = (0.0, 0.0, 0.0, 0.0);
        let mut n = 0.0;
        for l in (0..360).step_by(15) {
            for b in [-15.0, -8.0, -3.0, 0.0, 3.0, 8.0, 15.0] {
                let band = |m: &GalaxyModel, dl: f64| {
                    luminance(m.band_radiance(sun, lb_dir(f64::from(l) + dl, b), 10.0, fp))
                };
                let (d0, d1) = (band(&g, 0.0), band(&g, 0.5));
                let (s0, s1) = (band(&smooth, 0.0), band(&smooth, 0.5));
                sum_c += band(&clumps, 0.0);
                sum_d += d0;
                sum_s += s0;
                step_d += (d1 / d0).ln().abs();
                step_s += (s1 / s0).ln().abs();
                n += 1.0;
            }
        }
        let ratio = sum_d / sum_s;
        assert!(
            (0.8..2.0).contains(&ratio),
            "detailed / smooth mean band {ratio}"
        );
        assert!(
            sum_c > sum_s,
            "clumps alone darken the band: {}",
            sum_c / sum_s
        );
        assert!(sum_d < sum_c, "the named clouds add no dust");
        assert!(step_d / n > 0.05, "detail barely varies: {}", step_d / n);
        assert!(step_d > 5.0 * step_s, "detail {step_d} vs smooth {step_s}");
    }

    /// Stopping the march at τ = 15 changes the band by under 3 × 10⁻⁵ anywhere, the plane's lane
    /// toward the bulge (where it stops soonest) included, from the Sun and from the bar's edge.
    #[test]
    fn band_march_stops_only_when_nothing_is_left() {
        let g = GalaxyModel::milky_way();
        let fp = 2.0 / 512.0;
        let cams = [g.sun_galactocentric_pc(), DVec3::new(-1500.0, 600.0, 120.0)];
        let mut stopped = 0;
        for cam in cams {
            for l in (0..360).step_by(20) {
                for b in [-4.0, -1.0, 0.0, 0.5, 2.0, 6.0] {
                    let d = lb_dir(f64::from(l), b);
                    let full = g.band_march(cam, d, 10.0, fp, f64::INFINITY);
                    let cut = g.band_radiance(cam, d, 10.0, fp);
                    let rel = (cut - full).abs().max_element() / full.max_element();
                    assert!(rel < 3e-5, "{cam} l {l} b {b}: {rel}");
                    stopped += i32::from(cut != full);
                }
            }
        }
        assert!(stopped > 20, "the stop never fires: {stopped}");
    }

    /// `gal_pcg3d` in galaxy.wgsl is common/hash.wgsl's `pcg3d` (the engine's one integer hash,
    /// probed bit-exact against `uni_core::pcg3d` under `universe --selftest`) line for line.
    #[test]
    fn gal_pcg3d_is_the_engine_hash() {
        let engine = include_str!("../../uni-gfx/shaders/common/hash.wgsl");
        let body = |src: &str, name: &str| -> Vec<String> {
            let start = src
                .find(&format!("fn {name}("))
                .unwrap_or_else(|| panic!("fn {name}"));
            src[start..]
                .lines()
                .skip(1)
                .take_while(|l| *l != "}")
                .map(|l| l.trim().to_owned())
                .filter(|l| !l.is_empty())
                .collect()
        };
        let ours = body(GALAXY_WGSL_SRC, "gal_pcg3d");
        assert!(ours.len() > 8, "{ours:?}");
        assert_eq!(ours, body(engine, "pcg3d"));
    }

    /// The detail constants of galaxy.wgsl equal their Rust twins.
    #[test]
    fn detail_constants_match_wgsl() {
        let module = naga::front::wgsl::parse_str(&galaxy_wgsl_with_decl())
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&galaxy_wgsl_with_decl())));
        let lit = |name: &str| -> naga::Literal {
            let (_, c) = module
                .constants
                .iter()
                .find(|(_, c)| c.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("const {name}"));
            match module.global_expressions[c.init] {
                naga::Expression::Literal(l) => l,
                ref e => panic!("{name}: {e:?}"),
            }
        };
        let f = |name: &str, want: f64| match lit(name) {
            naga::Literal::F32(v) => assert_eq!(v, want as f32, "{name}"),
            l => panic!("{name}: {l:?}"),
        };
        let u = |name: &str, want: u32| match lit(name) {
            naga::Literal::U32(v) => assert_eq!(v, want, "{name}"),
            l => panic!("{name}: {l:?}"),
        };
        f("GAL_NOISE_FADE_START", NOISE_LOD_FADE[0]);
        f("GAL_NOISE_FADE_END", NOISE_LOD_FADE[1]);
        f("GAL_VALUE_NOISE_VAR", VALUE_NOISE_VAR);
        f("GAL_CLOUD_Q_MAX", CLOUD_Q_MAX);
        f("GAL_CLOUD_MIN_TAU", CLOUD_MIN_TAU);
        f("GAL_NOISE_MIN_DTAU", NOISE_MIN_DTAU);
        f("GAL_BAND_TAU_STOP", BAND_TAU_STOP);
        f("GAL_CLOUD_BEHIND_SIGMAS", CLOUD_BEHIND_SIGMAS);
        u("GAL_OCTAVE_SEED_STEP", OCTAVE_SEED_STEP);
        u("GAL_MAX_CLOUDS", NAMED_CLOUD_COUNT as u32);
    }
}
