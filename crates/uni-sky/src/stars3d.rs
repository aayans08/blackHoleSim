//! AT-HYG stars in 3-D, pass "stars" (Task B4).
//!
//! Every star of `World::star_data()` is one instance of a screen-aligned quad (four vertices,
//! two triangles from an index buffer), blended additively into `hdr` at the far plane (depth 0)
//! with no depth attachment: bodies occlude a star through `lens_mask` alone (lead ruling B9c,
//! [`STARS3D_DEPTH`]). Per star the vertex stage of `stars3d.wgsl`:
//! 1. skips the instance slots in [`StarInput::suppressed`] (near stars uni-sim draws as surfaces
//!    or points): all four vertices go to one point outside the clip volume, a degenerate quad;
//! 2. places the star: p = (sector centre − camera) + offset + v·t, in f32 (below);
//! 3. takes d = |p|, m = M_V + 5 log₁₀ d − 5 ([`star_apparent_mag`]) and
//!    E = 10^(−0.4 (m + 13.99)) lx (`uni_core::star_illuminance_lux`, as [`star_lux`]);
//! 4. scales E by the twinkle factor (only under an atmosphere, [`StarInput::scintillation`]):
//!    log-normal, with Young's (1967) naked-eye scintillation index at the zenith, on a TDB
//!    clock ([`scintillation_index`], [`twinkle_clock`]; approximate, see there);
//! 5. splats it with an energy-normalised eye PSF at E/Ω(θ) × `star_pre()`, Ω(θ) = Ω_c·cos³θ as
//!    M0 (`crate::stars`): M0's erf-integrated Gaussian core (`crate::psf`) holding
//!    1 − [`STARS3D_HALO_FRACTION`] of the light, plus, for every star, the eye's glare halo,
//!    a Moffat profile (β = 1.36, α = 0.1°) fitted to the CIE general disability glare equation
//!    (Task B4L; see [`STARS3D_HALO_FRACTION`]). Every star gets the same profile, so a bright
//!    star is not given a bigger image: its halo simply rises above the display's black over
//!    more pixels (the radius grows as E^(1/2β) = E^0.37), and the brightest stars glow while
//!    faint ones stay points. (The halo is renormalised inside the footprint radius, which grows
//!    with E, so the PSF is energy-conserving but not exactly linear: a faint star's halo is
//!    packed into its 2-px footprint, below the display's black.) It culls the splat when no
//!    pixel would rise above `psf::VISIBILITY_CUTOFF` ([`star_is_visible`]), and sizes its quad
//!    with [`footprint_radius_px`] (M0's rule with this halo and a 2-px minimum,
//!    [`STARS3D_MIN_RADIUS_PX`], review m16). The core's WGSL is M0's own: `stars3d.wgsl`
//!    imports M0's `stars.wgsl` and lays its group-1 bindings out around M0's two (review r0 I3;
//!    see [`LENS_MASK_BINDING`]);
//! 6. for a drawn star only, reads the colour from B−V through a 128-entry Planck table
//!    (`uni_core::bv_to_teff`, then `uni_core::blackbody_linear_srgb`, luminance 1), boosted by
//!    [`vivid_color`] while [`StarInput::vivid_colours`] is on.
//!
//! The fragment stage writes radiance × PSF weight × (1 − r)(1 − g) of `lens_mask` (the frame
//! contract's sky-light rule, `star_sky_factor`), clamped to `HDR_MAX`. That factor is the only
//! occlusion: every opaque writer stores its coverage in g (bodies and star surfaces their
//! coverage, rings their opacity, atmospheres 1 − T), so a star behind a body's interior gets 0
//! and one behind a body's anti-aliased edge pixel its (1 − coverage) share. Before task B9c a
//! far-plane depth test (`GreaterEqual` against 0) also ran, and since the bodies pass writes
//! depth on every edge fragment it dropped the sky there entirely.
//!
//! **Precision (design §5.2, Review Focus 5).** [`StarField3::new`] sorts the stars by 16-pc
//! sector ([`crate::sectors`]) and stores each star's f32 offset from its sector's centre. The
//! sector centres go to the GPU relative to an anchor near the camera (f64 difference, then f32),
//! and each frame's uniform carries camera − anchor (f64 difference, then f32). The centres are
//! re-uploaded only when the camera has moved more than [`REANCHOR_PC`] from the anchor: the
//! baked catalogue occupies 166 011 sectors, 2.66 MB, which is too much to upload every frame the
//! camera moves. `sectors::tests::sector_offsets_precision_at_1kpc` bounds the direction error.
//!
//! Proper motion is linear in time with the catalogue's Cartesian velocity (km/s × 1.0227 × 10⁻⁶
//! = pc/yr); a star without a radial velocity moves tangentially only, as the catalogue gives it
//! (A7). Aberration is M3's.
//!
//! The `twin` module (tests) and the free functions here are the CPU twins of `stars3d.wgsl`;
//! `crate::probes` checks them on the GPU under `universe --selftest`. M0's `StarField` stays as
//! the BSC5 fallback.

use std::f32::consts::{LOG2_10, LOG10_2, TAU};
use std::sync::OnceLock;

use anyhow::Context as _;
use bytemuck::Zeroable as _;
use glam::{DVec3, IVec3, UVec3, Vec3};
use uni_core::{Epoch, blackbody_linear_srgb, bv_to_teff, pcg3d};
use uni_data::{FLAG_NO_VELOCITY, Star3, f16_to_f32, f32_to_f16};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};
use wgpu::util::DeviceExt as _;

use crate::STARS3D_WGSL;
use crate::psf::{CORE_SIGMA_PX, MAX_RADIUS_PX, VISIBILITY_CUTOFF};
use crate::sectors::{SectorTable, centres_rel, sector_centre_pc, sector_of};
use crate::stars::pixel_solid_angle;

/// Smallest footprint radius of a 3-D star's quad [px] (review m16). M0's
/// `psf::MIN_RADIUS_PX` (3 px) is unchanged.
pub const STARS3D_MIN_RADIUS_PX: f32 = 2.0;

/// Share of every 3-D star's light in the eye's glare halo (Task B4L), with
/// [`STARS3D_HALO_ALPHA_PX`] and [`STARS3D_HALO_BETA`].
///
/// The eye's stray-light point-spread function beyond 0.1° is the CIE general disability glare
/// equation (Vos & van den Berg 1999, CIE 135/1; CIE 146:2002; used as the eye's glare PSF by
/// Bará & Bao-Varela 2023, Int. J. Sustainable Lighting, arXiv:2212.09103): L_veil/E_glare =
/// 10/θ³ + (5/θ² + 0.1 p/θ)(1 + (A/62.5)⁴) + 0.0025 p [sr⁻¹], θ in degrees, valid 0.1°–100°;
/// here for a 25-year-old (A = 25) of average pigmentation (p = 0.5). The halo is a Moffat
/// profile (Moffat 1969) h·(β − 1)/(πα²)·(1 + θ²/α²)^(−β), its width fixed at α = 0.1°, where
/// the equation starts and the eye's optical core (≈ 1′) has long ended, and β and h fitted to
/// the equation (minimax in log) over 0.3°–3°, where a bright star's glow shows: β = 1.36,
/// h = 0.239, within ±12 % of the equation there (0.73× at 0.2°; `halo_follows_the_cie_glare_
/// equation`). It holds 0.165 of the light between 0.1° and 3° (the equation: 0.219) and 0.053
/// inside 0.1° (the equation says nothing there); the equation's light beyond 3° (0.055, a veil
/// tens of degrees wide) is left to uni-post's bloom. A β = 1.5 fit to the 10/θ³ term alone
/// gave half the equation's glow at 2°.
///
/// The halo is fixed in display pixels, as the eye looking at the screen would see it, at the
/// reference display of the default view: a 60° field over 1080 lines, 0.0556° per pixel, so
/// α = 0.1° = 1.8 px. Every star has it (the same profile whatever its light); a faint star's
/// halo stays below the display's black. Because it is fixed in pixels, the "0.1°" holds only at
/// that reference field: at a 2° zoom the halo is 0.0033° on the sky, as an eye at a telescope
/// sees glare in its own field, not the sky's.
pub const STARS3D_HALO_FRACTION: f32 = 0.239;
/// Moffat width α of the 3-D halo [px]: 0.1° at 0.0556° per pixel ([`STARS3D_HALO_FRACTION`]).
pub const STARS3D_HALO_ALPHA_PX: f32 = 1.8;
/// Moffat exponent β of the 3-D halo, fitted with [`STARS3D_HALO_FRACTION`].
pub const STARS3D_HALO_BETA: f32 = 1.36;
/// The halo's part of a footprint ends where the halo falls to the display's black: AgX maps
/// everything at or below 2^−12.47393 = 1.76e-4 (its log2 range's floor, Sobotka; uni-post's
/// `AGX_MIN_EV`) to code 0. The halo is lowered to 0 at the radius and renormalised inside, so
/// the star's energy is kept and no edge shows; cutting it at `psf::VISIBILITY_CUTOFF` instead
/// (2.55 stops lower, which the core keeps) would draw 12-px quads around V = 5 stars whose
/// halos nobody can see, and cost the stars pass ~3× the fragments.
pub const STARS3D_HALO_CUTOFF: f32 = 1.76e-4;

/// A clamped halo's last pixel ring (r = R − 1) is tapered to at most this × [`STARS3D_HALO_CUTOFF`]
/// (final review render I2; [`halo_shape`]): half the display's black, so the glow ends at
/// black with a margin for the fixed-point search's last step.
pub const STARS3D_TAPER_TARGET: f32 = 0.5;
/// Largest window exponent n of a clamped halo: enough for the largest splat energy
/// ([`MAX_SPLAT_ENERGY`], 1e30), which needs n ≈ 18.5.
pub const STARS3D_TAPER_N_MAX: f32 = 24.0;
/// Fixed-point steps of [`halo_shape`]'s exponent search (n rises toward the fixed point).
pub const STARS3D_TAPER_STEPS: usize = 3;
/// W_n, the energy of the 3-D halo windowed by (1 − r²/R²)^n inside R = `psf::MAX_RADIUS_PX`
/// (64 px), for n = 2, 3, …, 24: ∫₀^R 2πr·Moffat(r)·(1 − r²/R²)^n dr, by Simpson's rule in
/// s = ln(1 + r²/α²), where the integrand is (β − 1)e^((1−β)s)(1 − α²(e^s − 1)/R²)^n
/// (`taper_energy_table_by_quadrature` recomputes it). W₂ is [`stars3d_halo_energy`]`(64)`.
pub const STARS3D_TAPER_ENERGY: [f32; 23] = [
    8.566_199_1e-1,
    8.378_380_8e-1,
    8.225_283_8e-1,
    8.094_601_4e-1,
    7.979_791_9e-1,
    7.876_916_9e-1,
    7.783_400_7e-1,
    7.697_455_7e-1,
    7.617_784_8e-1,
    7.543_413_2e-1,
    7.473_587_8e-1,
    7.407_713_1e-1,
    7.345_309_3e-1,
    7.285_983_1e-1,
    7.229_408_1e-1,
    7.175_309_7e-1,
    7.123_454_6e-1,
    7.073_642_6e-1,
    7.025_700_6e-1,
    6.979_477_6e-1,
    6.934_841_3e-1,
    6.891_674_9e-1,
    6.849_874_4e-1,
];

/// 1 km/s in pc per Julian year: 10³ m × 365.25 d × 86 400 s (IAU 1976 Julian year) ÷ 1 pc
/// (IAU 2012 B2) = 1.0227 × 10⁻⁶ (brief).
pub const PC_PER_YR_PER_KMS: f32 = (1.0e3 * 365.25 * uni_core::DAY / uni_core::PC) as f32;
/// V-band zero point of `uni_core::star_illuminance_lux`: E = 10^(−0.4 (m + 13.99)) lx (Allen
/// 1973, *Astrophysical Quantities*, 3rd ed., p. 197).
pub(crate) const LUX_ZERO_POINT_MAG: f32 = 13.99;
/// Largest display energy a splat carries, so a source at the camera (distance → 0) stays finite
/// on the GPU; far above anything that is not clamped to `HDR_MAX` anyway.
pub(crate) const MAX_SPLAT_ENERGY: f32 = 1.0e30;
/// Entries of the B−V → colour table.
pub(crate) const BV_LUT_LEN: usize = 128;
/// B−V range of the table: the range `uni_core::bv_to_teff` clamps to (Ballesteros 2012).
pub(crate) const BV_LUT_MIN: f32 = -0.4;
pub(crate) const BV_LUT_MAX: f32 = 2.0;
/// B−V packed for a star whose colour is not finite: the Sun's, 0.65 (Cox 2000, *Allen's
/// Astrophysical Quantities*, 4th ed.), 5778 K by `bv_to_teff`.
pub(crate) const UNKNOWN_BV: f32 = 0.65;
/// Absolute magnitude packed for a star with a non-finite position or magnitude: its
/// illuminance underflows to 0 in f32, so the vertex stage culls it and its slot stays valid.
pub(crate) const INVISIBLE_ABS_MAG: f32 = 1.0e30;
/// Saturation factor of the vivid colours about the luminance axis. The owner-approved look
/// preview used 1.6 (`.superpowers/look-preview/v2/parts/78-frame.js`, `uSat`), but its tone
/// curve then added 1.28× saturation after AgX (`50-sky.js`, `tone`); the engine's AgX
/// (uni-post) has no such boost and pulls bright colours toward white, so the engine's star
/// colours need the product, about 2.2 (Task B4L, judged on captures). Non-physical and
/// labelled "Vivid star colours · non-physical"; the luminance is kept exactly.
pub(crate) const VIVID_SATURATION: f32 = 2.2;
/// Young's (1967) scintillation coefficient [cm^(2/3) s^(1/2)]: a star's relative rms intensity
/// through an aperture D [cm] at air mass X from altitude h, averaged over t [s], is
/// σ_I = 0.09·D^(−2/3)·X^1.75·e^(−h/8000 m)·(2t)^(−1/2) (Young 1967, AJ 72, 747, in the form
/// Stefansson et al. 2017, arXiv:1710.01790, quote; in SI units, with σ_I² ∝ (cos γ)^(−3), it is
/// Osborn et al. 2015, MNRAS 452, 1707, eq. 2, who find it ~1.5× below measured medians at large
/// telescopes). It is fitted to telescope apertures above the Fresnel scale (a few cm to tens of
/// cm, Osborn et al. 2015 §2); a 7 mm pupil is below it, so the naked-eye value is an
/// extrapolation (approximate; the HUD shows "Twinkling: approximate model").
pub(crate) const YOUNG_SCINTILLATION_COEFF: f64 = 0.09;
/// The naked eye's aperture D [cm]: a dark-adapted pupil of about 7 mm (lead's r0 ruling; the
/// pupil spans roughly 2 to 8 mm from bright to dark). A modelling choice.
pub(crate) const EYE_PUPIL_CM: f64 = 0.7;
/// The eye's integration time t [s] in Young's (2t)^(−1/2), and the interval between independent
/// twinkle draws (what the eye sees decorrelates over its own averaging time): Bloch's law of
/// complete temporal summation, about 0.1 s in dim light (Barlow 1958, J. Physiol. 141, 337; the
/// commonly quoted order of magnitude, not checked against the paper here). A modelling choice.
pub(crate) const TWINKLE_EYE_S: f64 = 0.1;
/// Twinkle draws per day: 86 400 s / [`TWINKLE_EYE_S`], a whole number, so the clock counts whole
/// days first and its f64 fraction stays exact at any epoch.
pub(crate) const TWINKLE_STEPS_PER_DAY: i64 = 864_000;
/// Third `pcg3d` lane of the twinkle hash, so it never repeats another hash of (slot, step).
pub(crate) const TWINKLE_SEED: u32 = 0x7F4A_7C15;
/// The star pass re-uploads the sector centres when the camera is further than this [pc] from
/// the anchor they were uploaded against. The f32 path's direction error is then dominated not
/// by camera − anchor (≤ 3e-8 pc per axis, half an ulp below 1 pc) but by the sector centre −
/// anchor and the star's offset (each rounded to f32 at 8–16 pc: 5e-7 to 1e-6 pc) and their f32
/// difference: about 2e-6 pc in all, 2e-5 rad (0.02 px at 1080 lines, 60°) for a star at the
/// 0.1-pc radius inside which uni-sim suppresses stars. `sector_offsets_precision_at_1kpc`
/// measures 8.1e-7 rad over its cases (review r0 M2).
pub(crate) const REANCHOR_PC: f64 = 1.0;
/// Relative luminance weights of linear Rec. 709: ITU-R BT.709-6 (2015), item 3.2.
const REC709_Y: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// The index buffer of one splat quad: triangles (0, 1, 2) and (2, 1, 3) over the corners of
/// `splat_corner` in the shaders (0 (−,−), 1 (+,−), 2 (−,+), 3 (+,+)). Four distinct vertices
/// per instance, so a GPU that reuses transformed vertices runs the vertex stage four times per
/// star instead of six (unmeasured on the M3).
pub(crate) const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

/// The B−V → colour table the star pass reads: the twin of `struct BvColourLut` in
/// `stars3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BvLut {
    /// rgb: Planck colour (linear Rec. 709, luminance 1) at B−V = BV_LUT_MIN + i·step; w: 0.
    pub entries: [[f32; 4]; BV_LUT_LEN],
}

/// What the star pass needs per frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug)]
pub struct StarInput {
    /// Years since the catalogue epoch (`uni_data::STAR3_EPOCH_JD_TDB`, J2000.0), for proper
    /// motion.
    pub years_since_epoch: f64,
    /// Camera − SSB [pc], ICRS axes, f64.
    pub camera_pc: DVec3,
    /// Indices into the slice given to [`StarField3::new`] that the pass must skip (≤ 8; uni-sim's
    /// `Snapshot::suppressed_stars`). Indices past the slice are ignored, and only the first 8
    /// are used.
    pub suppressed: Vec<u32>,
    /// The "Vivid star colours · non-physical" option (owner feedback, 2026-11-22; on by default):
    /// a saturation boost beyond Planck chromaticity that keeps each star's Rec. 709 luminance
    /// ([`vivid_color`]). The HUD shows `uni_ui::VIVID_STAR_COLOURS_LABEL` while it is on.
    pub vivid_colours: bool,
    /// Scintillation (twinkle) strength in [0, 1]: `uni_atmo::scintillation_strength` for the
    /// camera, 0 in space. The star pass takes it as Young's (1967) altitude factor at the
    /// zenith ([`scintillation_index`]) and drives the twinkle deterministically from `pcg3d`
    /// of the star's instance slot and the frame's TDB seconds ([`twinkle_clock`]), so captures
    /// at a fixed epoch repeat. Non-zero exactly inside an atmosphere shell, so the point pass
    /// also reads it as "the camera is inside an atmosphere" ([`crate::PointSprites`]). Clamped
    /// to [0, 1]; NaN is 0.
    pub scintillation: f32,
}

impl Default for StarInput {
    /// At the catalogue epoch, at the SSB, nothing suppressed, vivid colours on (the owner's
    /// default), no twinkle.
    fn default() -> Self {
        Self {
            years_since_epoch: 0.0,
            camera_pc: DVec3::ZERO,
            suppressed: Vec::new(),
            vivid_colours: true,
            scintillation: 0.0,
        }
    }
}

/// One star as the GPU sees it: the twin of `struct StarGpu` in `stars3d.wgsl` (32 bytes, two
/// vec4s; review m15: `uni_data::Star3` is a 28-byte disk record and not a valid WGSL storage
/// struct).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarGpu {
    /// xyz: offset from the star's sector centre [pc] at the catalogue epoch; w: absolute
    /// magnitude M_V.
    pub offset_absmag: [f32; 4],
    /// x: f16 (vx, vy) [km/s]; y: f16 (vz, B−V) (each pair low half first, as
    /// `uni_data::Star3` packs them); z: the `uni_data` flag bits; w: the sector index.
    pub vel_bv_flags: [u32; 4],
}

/// The star pass's per-frame uniform: the twin of `struct StarsParams` in `stars3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StarsParams {
    /// xyz: camera − anchor [pc] (f64 difference, then f32); w: years since the catalogue epoch.
    pub camera_years: [f32; 4],
    /// x: twinkle log-amplitude σ (0 = steady); y: 1 with vivid colours, else 0; z: the
    /// twinkle phase ([`TwinkleClock::phase`]); w: 0.
    pub look: [f32; 4],
    /// x: the twinkle draw ([`TwinkleClock::step`]); yzw: 0.
    pub twinkle_step: [u32; 4],
    /// Instance slots of the suppressed stars, [`NO_STAR`] where unused.
    pub suppressed: [[u32; 4]; 2],
}

/// Marks an unused entry of [`StarsParams::suppressed`].
pub(crate) const NO_STAR: u32 = u32::MAX;

/// The stars in instance order, ready to upload.
pub(crate) struct PackedStars {
    pub gpu: Vec<StarGpu>,
    pub table: SectorTable,
    /// Instance slot of each input star.
    pub slots: Vec<u32>,
}

/// Sorts `stars` by sector and packs them (the CPU half of [`StarField3::new`]). A star with a
/// non-finite position is put in sector (0, 0, 0) and made invisible, so every input keeps a
/// slot. The caller guarantees that `stars.len()` fits a `u32`.
pub(crate) fn pack_stars(stars: &[Star3]) -> PackedStars {
    let positions: Vec<Option<DVec3>> = stars
        .iter()
        .map(|s| {
            let p = DVec3::from_array(s.pos_pc.map(f64::from));
            p.is_finite().then_some(p)
        })
        .collect();
    let keys: Vec<IVec3> = positions
        .iter()
        .map(|p| sector_of(p.unwrap_or(DVec3::ZERO)))
        .collect();
    let table = SectorTable::build(&keys);
    let gpu = table
        .order
        .iter()
        .zip(&table.sector_index)
        .map(|(&i, &sector)| {
            let i = i as usize;
            star_gpu(&stars[i], positions[i], sector, sector_centre_pc(keys[i]))
        })
        .collect();
    let slots = table.slots();
    PackedStars { gpu, table, slots }
}

/// Packs one star of sector `sector` (centre `centre_pc`). The velocity is zeroed for a star
/// flagged "no velocity" (as uni-sim's `star_pos_pc`); a non-finite B−V becomes [`UNKNOWN_BV`];
/// a non-finite position (`pos_pc` = `None`) or magnitude makes the star invisible.
fn star_gpu(s: &Star3, pos_pc: Option<DVec3>, sector: u32, centre_pc: DVec3) -> StarGpu {
    let (offset, abs_mag) = match pos_pc {
        Some(p) if s.abs_mag_v.is_finite() => ((p - centre_pc).as_vec3(), s.abs_mag_v),
        _ => (Vec3::ZERO, INVISIBLE_ABS_MAG),
    };
    let v = s.velocity_kms();
    let v = if s.flags() & FLAG_NO_VELOCITY != 0 || !v.is_finite() {
        Vec3::ZERO
    } else {
        v
    };
    let bv = if s.bv().is_finite() {
        s.bv()
    } else {
        UNKNOWN_BV
    };
    let pack = |lo: f32, hi: f32| u32::from(f32_to_f16(lo)) | u32::from(f32_to_f16(hi)) << 16;
    StarGpu {
        offset_absmag: offset.extend(abs_mag).to_array(),
        vel_bv_flags: [pack(v.x, v.y), pack(v.z, bv), s.flags(), sector],
    }
}

/// The two binary16 halves of `word`, low half first. Twin of `unpack_f16_pair` in
/// `stars3d.wgsl`, which decodes the bits itself (naga gates WGSL's `unpack2x16float` behind a
/// capability `uni_gfx::validate_wgsl` does not enable).
pub(crate) fn unpack_f16_pair(word: u32) -> [f32; 2] {
    [f16_to_f32(word as u16), f16_to_f32((word >> 16) as u16)]
}

/// Position of a star relative to the camera [pc] at `years` after the catalogue epoch:
/// (`sector_rel` + offset) + v·years, in f32 as the GPU does. `sector_rel` is the star's sector
/// centre − camera [pc]. Twin of `star_rel_pos_pc` in `stars3d.wgsl`.
pub(crate) fn star_rel_pos_pc(star: &StarGpu, sector_rel: Vec3, years: f32) -> Vec3 {
    let [vx, vy] = unpack_f16_pair(star.vel_bv_flags[0]);
    let [vz, _] = unpack_f16_pair(star.vel_bv_flags[1]);
    let v = Vec3::new(vx, vy, vz) * PC_PER_YR_PER_KMS;
    (sector_rel + Vec3::from_slice(&star.offset_absmag[..3])) + v * years
}

/// The star pass's uniform for `input` at twinkle time `clock`, with the sector centres uploaded
/// relative to `anchor_pc` and `slots` mapping star indices to instance slots.
pub(crate) fn stars_params(
    input: &StarInput,
    clock: &TwinkleClock,
    anchor_pc: DVec3,
    slots: &[u32],
) -> StarsParams {
    let mut suppressed = [NO_STAR; 8];
    let kept = input
        .suppressed
        .iter()
        .filter_map(|&i| slots.get(i as usize).copied());
    for (entry, slot) in suppressed.iter_mut().zip(kept) {
        *entry = slot;
    }
    let sigma = twinkle_log_sigma(scintillation_index(input.scintillation, clock.time_rate));
    let camera = (input.camera_pc - anchor_pc).as_vec3();
    StarsParams {
        camera_years: camera.extend(input.years_since_epoch as f32).to_array(),
        look: [
            sigma,
            if input.vivid_colours { 1.0 } else { 0.0 },
            clock.phase,
            0.0,
        ],
        twinkle_step: [clock.step, 0, 0, 0],
        suppressed: [
            [suppressed[0], suppressed[1], suppressed[2], suppressed[3]],
            [suppressed[4], suppressed[5], suppressed[6], suppressed[7]],
        ],
    }
}

/// Group-1 binding of `lens_mask` in both splat passes. The declaration is M0's, in
/// `stars.wgsl`, which `stars3d.wgsl` imports so that both 3-D passes run M0's PSF WGSL itself
/// rather than a copy (review r0 I3). The import also declares M0's `stars` instance buffer at
/// binding 0, which neither 3-D pass uses or binds; the 3-D bindings follow from 2 up, so no two
/// globals of a composed module share a slot (`splat_layouts_match_entry_point_use`).
pub(crate) const LENS_MASK_BINDING: u32 = 1;
/// Group-1 binding of the star pass's per-frame [`StarsParams`].
pub(crate) const STARS_PARAMS_BINDING: u32 = 2;
/// Group-1 binding of the star pass's B−V colour table ([`BvLut`]).
pub(crate) const BV_LUT_BINDING: u32 = 3;
/// Group-1 binding of the star records ([`StarGpu`], instance order), `catalogue` in WGSL.
pub(crate) const CATALOGUE_BINDING: u32 = 4;
/// Group-1 binding of the sector centres − anchor [pc].
pub(crate) const SECTORS_BINDING: u32 = 5;

/// Bind-group-layout entry of the lens mask (read with `textureLoad`, so unfilterable), which
/// the vertex stage also reads for the attachment size. Shared by the star and point passes, at
/// [`LENS_MASK_BINDING`].
pub(crate) fn lens_mask_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: LENS_MASK_BINDING,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Bind-group-layout entry of a vertex-stage buffer of `ty` whose bound size is at least
/// `min_size` bytes.
pub(crate) fn vertex_buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_size: usize,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(min_size as u64),
        },
        count: None,
    }
}

/// The star pass's group-1 layout: the lens mask, the per-frame uniform, the B−V table, the star
/// records and the sector centres.
pub(crate) fn star_bgl_entries() -> [wgpu::BindGroupLayoutEntry; 5] {
    let storage = wgpu::BufferBindingType::Storage { read_only: true };
    let uniform = wgpu::BufferBindingType::Uniform;
    [
        lens_mask_entry(),
        vertex_buffer_entry(STARS_PARAMS_BINDING, uniform, size_of::<StarsParams>()),
        vertex_buffer_entry(BV_LUT_BINDING, uniform, size_of::<BvLut>()),
        vertex_buffer_entry(CATALOGUE_BINDING, storage, size_of::<StarGpu>()),
        vertex_buffer_entry(SECTORS_BINDING, storage, size_of::<[f32; 4]>()),
    ]
}

/// How a sky pass meets the depth buffer (lead ruling B9c).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SkyDepth {
    /// No depth attachment and no depth state: far-plane sky light (the band, catalogue stars)
    /// is occluded by the `lens_mask` factors alone. Every opaque writer stores its coverage in
    /// g, so a body's interior pixel keeps none of the sky and an anti-aliased edge pixel its
    /// (1 − coverage) share. A far-plane test (`GreaterEqual` against the cleared 0) drops the
    /// whole sky on every edge fragment the bodies pass writes depth for: B9c measured it on
    /// Venus's night limb (22 pixels with no band behind them).
    MaskOnly,
    /// A read-only test against the scene depth with this compare, never written: the reflected
    /// bodies' point sprites, drawn at their true distance, which can be behind a body.
    Test(wgpu::CompareFunction),
}

impl SkyDepth {
    /// The pipeline's depth state: none for [`SkyDepth::MaskOnly`], else read-only with the
    /// compare.
    pub(crate) fn state(self) -> Option<wgpu::DepthStencilState> {
        match self {
            SkyDepth::MaskOnly => None,
            SkyDepth::Test(compare) => Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(compare),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
        }
    }

    /// The pass's depth attachment: `view` read-only (`depth_ops: None`) exactly when
    /// [`SkyDepth::state`] has a depth state (wgpu requires the two to agree).
    pub(crate) fn attachment(
        self,
        view: &wgpu::TextureView,
    ) -> Option<wgpu::RenderPassDepthStencilAttachment<'_>> {
        self.state()
            .map(|_| wgpu::RenderPassDepthStencilAttachment {
                view,
                depth_ops: None,
                stencil_ops: None,
            })
    }
}

/// The 3-D star pass's depth: none ([`SkyDepth::MaskOnly`], lead ruling B9c).
pub(crate) const STARS3D_DEPTH: SkyDepth = SkyDepth::MaskOnly;

/// The splat pipeline of the point pass (`vs_points`/`fs_points`, depth `depth_compare` at the
/// true distance, read-only): [`splat_pipeline_with`] with [`SkyDepth::Test`].
pub(crate) fn splat_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry_points: (&str, &str),
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    splat_pipeline_with(
        device,
        module,
        layout,
        entry_points,
        SkyDepth::Test(depth_compare),
    )
}

/// The splat pipeline of the star (`vs_stars`/`fs_stars`, [`STARS3D_DEPTH`]: no depth) or point
/// (`vs_points`/`fs_points`, depth `Greater` at the true distance) pass: an indexed triangle
/// list, no face culling (the quad's winding depends on the y flip), additive blending into
/// `hdr`'s colour, depth as `depth` says (never written).
pub(crate) fn splat_pipeline_with(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry_points: (&str, &str),
    depth: SkyDepth,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry_points.0),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(entry_points.0),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry_points.1),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ADDITIVE),
                // Light adds to colour only; hdr's alpha is left alone.
                write_mask: wgpu::ColorWrites::COLOR,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: depth.state(),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The index buffer holding [`QUAD_INDICES`].
pub(crate) fn quad_index_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat quad indices"),
        contents: bytemuck::cast_slice(&QUAD_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    })
}

/// Begins the point pass's render pass `label`, which adds into `hdr` and tests (never writes)
/// depth: [`begin_splat_pass_with`] with [`SkyDepth::Test`] (the compare is the pipeline's).
pub(crate) fn begin_splat_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    targets: &FrameTargets,
    timer: &mut PassTimer,
    label: &'static str,
) -> wgpu::RenderPass<'e> {
    begin_splat_pass_with(
        enc,
        targets,
        timer,
        label,
        SkyDepth::Test(wgpu::CompareFunction::Greater),
    )
}

/// Begins a render pass named `label` that adds into `hdr`, with the depth attachment `depth`
/// asks for ([`SkyDepth::attachment`]; never written).
pub(crate) fn begin_splat_pass_with<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    targets: &FrameTargets,
    timer: &mut PassTimer,
    label: &'static str,
    depth: SkyDepth,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &targets.hdr_view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: depth.attachment(&targets.depth_view),
        timestamp_writes: timer.render_writes(label),
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// The 3-D star pass (`RenderFeature` named "stars").
///
/// Group 0 is the renderer's. Group 1 ([`star_bgl_entries`]): binding 1 the lens mask (M0's
/// declaration; rebuilt in `resize`, when the targets are recreated), 2 the per-frame
/// [`StarsParams`], 3 the B−V colour table, 4 the stars (instance order), 5 the sector centres −
/// anchor. Binding 0 is M0's `stars`, unused here (see [`LENS_MASK_BINDING`]).
pub struct StarField3 {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    star_buffer: wgpu::Buffer,
    sector_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    lut_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// Group 1 for the current targets. `None` until the first `prepare` or `resize`.
    bind_group: Option<wgpu::BindGroup>,
    input: SharedInput<StarInput>,
    /// The occupied sectors, in the order of the sector buffer.
    sectors: Vec<IVec3>,
    /// Instance slot of each star of the slice given to `new`.
    slots: Vec<u32>,
    /// The position [pc] the sector buffer's centres are relative to.
    anchor_pc: DVec3,
    count: u32,
}

impl StarField3 {
    /// Uploads `stars` (sorted by sector) and builds the pass. `stars` is `World::star_data()`:
    /// AT-HYG followed by the archive-only exoplanet hosts, the slice that
    /// `Snapshot::suppressed_stars` and [`StarInput::suppressed`] index (not `WorldData::stars`,
    /// which `World::new` consumes and extends). `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`. An empty slice is allowed (the pass then draws
    /// nothing).
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        stars: &[Star3],
        input: SharedInput<StarInput>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let count =
            u32::try_from(stars.len()).context("more stars than one instanced draw can hold")?;
        let mut packed = pack_stars(stars);
        let sectors = std::mem::take(&mut packed.table.sectors);
        let camera = input.get().camera_pc;
        let anchor_pc = if camera.is_finite() {
            camera
        } else {
            DVec3::ZERO
        };
        let mut centres = centres_rel(&sectors, anchor_pc);
        // A storage binding cannot be empty: an empty sky keeps one dark entry and draws none.
        if packed.gpu.is_empty() {
            packed.gpu.push(StarGpu::zeroed());
        }
        if centres.is_empty() {
            centres.push([0.0; 4]);
        }
        let star_bytes: &[u8] = bytemuck::cast_slice(&packed.gpu);
        let sector_bytes: &[u8] = bytemuck::cast_slice(&centres);
        let limit = device.limits().max_storage_buffer_binding_size;
        anyhow::ensure!(
            star_bytes.len() as u64 <= limit && sector_bytes.len() as u64 <= limit,
            "{count} stars need {} B and {} sectors {} B of storage; the device allows {limit} B",
            star_bytes.len(),
            centres.len(),
            sector_bytes.len()
        );
        let params = stars_params(
            &input.get(),
            &TwinkleClock::default(),
            anchor_pc,
            &packed.slots,
        );

        crate::register_shaders(lib);
        let module = lib.module(device, STARS3D_WGSL)?;

        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let star_buffer = init("stars3d stars", star_bytes, wgpu::BufferUsages::STORAGE);
        let sector_buffer = init(
            "stars3d sectors",
            sector_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params_buffer = init(
            "stars3d params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let lut_buffer = init(
            "stars3d B-V colours",
            bytemuck::bytes_of(bv_lut()),
            wgpu::BufferUsages::UNIFORM,
        );
        let index_buffer = quad_index_buffer(device);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stars3d bgl"),
            entries: &star_bgl_entries(),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stars3d layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        // Stars sit on the far plane (depth 0) with no depth test: lens_mask's (1 − r)(1 − g)
        // occludes them (lead ruling B9c, STARS3D_DEPTH).
        let pipeline = splat_pipeline_with(
            device,
            &module,
            &layout,
            ("vs_stars", "fs_stars"),
            STARS3D_DEPTH,
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the 3-D star pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            bgl,
            star_buffer,
            sector_buffer,
            params_buffer,
            lut_buffer,
            index_buffer,
            bind_group: None,
            input,
            sectors,
            slots: packed.slots,
            anchor_pc,
            count,
        })
    }

    /// Number of stars uploaded. Every one is drawn; suppressed, off-screen, behind-camera and
    /// invisible stars are culled per frame in the vertex shader.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// (Re)creates group 1 against `targets` (the lens mask view changes with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        fn buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stars3d bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: LENS_MASK_BINDING,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
                buffer(STARS_PARAMS_BINDING, &self.params_buffer),
                buffer(BV_LUT_BINDING, &self.lut_buffer),
                buffer(CATALOGUE_BINDING, &self.star_buffer),
                buffer(SECTORS_BINDING, &self.sector_buffer),
            ],
        }));
    }
}

impl RenderFeature for StarField3 {
    fn name(&self) -> &'static str {
        "stars"
    }

    /// Reads this frame's [`StarInput`], re-uploads the sector centres when the camera has left
    /// the anchor's 1-pc neighbourhood, and writes the per-frame uniform with the twinkle clock
    /// of `ctx`'s epoch. A non-finite camera position keeps the previous frame's uniform.
    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let input = self.input.get();
        if !input.camera_pc.is_finite() {
            return;
        }
        // Both positions are finite here (the anchor starts finite and only takes finite ones).
        if (input.camera_pc - self.anchor_pc).length() > REANCHOR_PC {
            self.anchor_pc = input.camera_pc;
            let centres = centres_rel(&self.sectors, self.anchor_pc);
            if !centres.is_empty() {
                gpu.queue
                    .write_buffer(&self.sector_buffer, 0, bytemuck::cast_slice(&centres));
            }
        }
        let clock = twinkle_clock(ctx.epoch, ctx.sim_dt_s, ctx.dt_real_s);
        let params = stars_params(&input, &clock, self.anchor_pc, &self.slots);
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
    }

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.count == 0 {
            return;
        }
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        let mut pass = begin_splat_pass_with(enc, targets, timer, "stars", STARS3D_DEPTH);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..self.count);
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

/// Apparent V magnitude of a star of absolute magnitude `abs_mag` at `d_pc` parsecs:
/// M + 5 log₁₀ d − 5 (distance modulus), with log₁₀ d = log₁₀ 2 · log₂ d as in WGSL (which has
/// no log10). Twin of `star_apparent_mag` in `stars3d.wgsl`. d = 0 gives −∞.
pub fn star_apparent_mag(abs_mag: f32, d_pc: f32) -> f32 {
    abs_mag + 5.0 * LOG10_2 * d_pc.log2() - 5.0
}

/// Illuminance [lux] of a star of apparent V magnitude `v_mag`, in f32 as the GPU computes it:
/// 10^(−0.4 (m + 13.99)) = 2^(−0.4 log₂10 (m + 13.99)) (`uni_core::star_illuminance_lux`).
/// Twin of `star_lux` in `stars3d.wgsl`.
pub(crate) fn star_lux(v_mag: f32) -> f32 {
    (-0.4 * LOG2_10 * (v_mag + LUX_ZERO_POINT_MAG)).exp2()
}

/// The 3-D halo's Moffat profile with unit energy over the plane [1/px²]:
/// (β − 1)/(πα²)·(1 + r²/α²)^(−β) with [`STARS3D_HALO_ALPHA_PX`] and [`STARS3D_HALO_BETA`]
/// (Moffat 1969, A&A 3, 455). Twin of `stars3d_moffat` in `stars3d.wgsl`.
pub fn stars3d_moffat(r_px: f32) -> f32 {
    let a2 = STARS3D_HALO_ALPHA_PX * STARS3D_HALO_ALPHA_PX;
    (STARS3D_HALO_BETA - 1.0) / (std::f32::consts::PI * a2)
        * (1.0 + r_px * r_px / a2).powf(-STARS3D_HALO_BETA)
}

/// Halo weight of a pixel whose centre is `r_px` from the star, for a footprint of `radius_px`
/// with weight scale `scale` and window exponent `taper` ([`halo_shape`]): [`stars3d_moffat`] ×
/// the window (1 − r²/R²)^n, which takes it smoothly to 0 at the radius, × the scale. For an
/// unclamped footprint n = 2 and the scale is 1 / [`stars3d_halo_energy`]: unit energy inside R
/// (M0's `psf::halo_weight` lowers its profile by the value at R instead, which left a visible
/// ring where the radius is clamped, as Sirius showed at a 2° field); sampled at pixel centres
/// the weights then sum to 1 within 1.2 % for any radius ≥ 2 px and sub-pixel position
/// (`stars3d_halo_conserves_energy`). n = 2 is a square, as the WGSL's `select`. Twin of
/// `stars3d_halo_weight` in `stars3d.wgsl`.
pub fn stars3d_halo_weight(r_px: f32, radius_px: f32, scale: f32, taper: f32) -> f32 {
    if !(r_px < radius_px) {
        return 0.0;
    }
    let window = 1.0 - (r_px * r_px) / (radius_px * radius_px);
    let w = if taper <= 2.0 {
        window * window
    } else {
        window.powf(taper)
    };
    stars3d_moffat(r_px) * w * scale
}

/// The 3-D halo's energy inside radius `rho_px`: 1 − (1 + ρ²/α²)^(1 − β) (Moffat 1969). Twin of
/// `stars3d_moffat_energy` in `stars3d.wgsl`.
pub fn stars3d_moffat_energy(rho_px: f32) -> f32 {
    let a2 = STARS3D_HALO_ALPHA_PX * STARS3D_HALO_ALPHA_PX;
    1.0 - (1.0 + rho_px * rho_px / a2).powf(1.0 - STARS3D_HALO_BETA)
}

/// W_n at R = `psf::MAX_RADIUS_PX` for n in [2, [`STARS3D_TAPER_N_MAX`]] (clamped), linear in
/// ln W between [`STARS3D_TAPER_ENERGY`]'s integer n (within 6e-4 of the quadrature at half
/// integers). Twin of `stars3d_taper_energy` in `stars3d.wgsl`.
pub fn stars3d_taper_energy(n: f32) -> f32 {
    let x = n.clamp(2.0, STARS3D_TAPER_N_MAX) - 2.0;
    let i = (x as usize).min(21);
    let f = x - i as f32;
    let (a, b) = (
        STARS3D_TAPER_ENERGY[i].ln(),
        STARS3D_TAPER_ENERGY[i + 1].ln(),
    );
    (a + (b - a) * f).exp()
}

/// The halo's footprint ([`halo_shape`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloShape {
    /// Footprint radius R [px].
    pub radius_px: f32,
    /// Weight scale of [`stars3d_halo_weight`].
    pub scale: f32,
    /// Window exponent n (2 unless the radius is clamped).
    pub taper: f32,
}

/// The halo's footprint for a splat of display energy `energy` and halo share `h` (final review
/// render I2: a point's glare never ends above display black, and its light is conserved up to
/// the clamp).
///
/// - Unclamped ([`footprint_radius_unclamped_px`] ≤ 64 px, where the halo has fallen to the
///   display's black): R = that radius, n = 2, scale = 1/W₂(R), unit energy (B4L, unchanged).
/// - Clamped (a brighter source): R = 64 px and the halo keeps K = M(R) + (1 − M(R_free)) of its
///   light, M = [`stars3d_moffat_energy`], R_free the unclamped radius: its own light inside the
///   clamp plus the tail beyond R_free that was below black anyway (so K → 1 where the clamp
///   starts, and the scale is continuous there). The dropped remainder is the halo's light
///   between R and R_free, M(R_free) − M(R): at most 1 − M(64) = 7.6 % of the halo (1.8 % of
///   the source) for the brightest sources. The CIE glare beyond 3° is uni-post's bloom's in the
///   design anyway ([`STARS3D_HALO_FRACTION`]).
///   The window exponent n grows from 2 until the halo's last pixel ring, h·E·K/W_n·Moffat(R −
///   1)·(1 − (R − 1)²/R²)^n, is at most [`STARS3D_TAPER_TARGET`] × [`STARS3D_HALO_CUTOFF`]: the
///   glow then ends at black whatever the source's light (`clamped_halo_ends_at_black`). Found
///   by [`STARS3D_TAPER_STEPS`] fixed-point steps from n = 2 (W_n falls slowly with n, so n
///   rises to the fixed point from below).
///
/// Twin of `stars3d_halo_shape` in `stars3d.wgsl`.
pub fn halo_shape(energy: f32, h: f32) -> HaloShape {
    let r_free = footprint_radius_unclamped_px(energy, h);
    if !(r_free > MAX_RADIUS_PX) {
        return HaloShape {
            radius_px: r_free,
            scale: 1.0 / stars3d_halo_energy(r_free),
            taper: 2.0,
        };
    }
    let radius = MAX_RADIUS_PX;
    let kept = 1.0 - (stars3d_moffat_energy(r_free) - stars3d_moffat_energy(radius)).max(0.0);
    let edge = h * energy * stars3d_moffat(radius - 1.0);
    let ring = 1.0 - (radius - 1.0) * (radius - 1.0) / (radius * radius);
    let mut n = 2.0_f32;
    for _ in 0..STARS3D_TAPER_STEPS {
        let scale = kept / stars3d_taper_energy(n);
        let need = (STARS3D_TAPER_TARGET * STARS3D_HALO_CUTOFF / (edge * scale)).ln() / ring.ln();
        n = need.clamp(2.0, STARS3D_TAPER_N_MAX);
    }
    HaloShape {
        radius_px: radius,
        scale: kept / stars3d_taper_energy(n),
        taper: n,
    }
}

/// PSF weight of pixel `pix` (integer index; pixel k spans [k, k + 1]) for a splat centred at
/// `centre` [px] with halo share `h` and halo footprint `shape`: (1 − h)·the Gaussian core
/// integrated over the pixel + h·the windowed halo at the pixel's centre. Twin of `psf_weight`
/// in `stars3d.wgsl`; the paper exposure (uni-app) draws the world-light Sun with it.
pub fn psf_weight(pix: glam::Vec2, centre: glam::Vec2, h: f32, shape: &HaloShape) -> f32 {
    let edge = pix - centre;
    let core = crate::psf::pixel_coverage(edge.x, CORE_SIGMA_PX)
        * crate::psf::pixel_coverage(edge.y, CORE_SIGMA_PX);
    let mut weight = (1.0 - h) * core;
    if h > 0.0 {
        let r = (pix + glam::Vec2::splat(0.5) - centre).length();
        weight += h * stars3d_halo_weight(r, shape.radius_px, shape.scale, shape.taper);
    }
    weight
}

/// The windowed halo's energy inside a footprint of `radius_px`: with A = α², m = A/R²,
/// k = 1 + m and u = 1 + r²/A, ∫₀^R 2πr I(r)(1 − r²/R²)² dr = (β − 1) [F(u)]₁^(1+R²/A),
/// F(u) = k² u^(1−β)/(1 − β) − 2km u^(2−β)/(2 − β) + m² u^(3−β)/(3 − β), which tends to 1 as
/// R → ∞ (`halo_energy_closed_form`). Twin of `stars3d_halo_energy` in `stars3d.wgsl`.
pub fn stars3d_halo_energy(radius_px: f32) -> f32 {
    let b = STARS3D_HALO_BETA;
    let a2 = STARS3D_HALO_ALPHA_PX * STARS3D_HALO_ALPHA_PX;
    let m = a2 / (radius_px * radius_px);
    let k = 1.0 + m;
    let f = |u: f32| {
        k * k * u.powf(1.0 - b) / (1.0 - b) - 2.0 * k * m * u.powf(2.0 - b) / (2.0 - b)
            + m * m * u.powf(3.0 - b) / (3.0 - b)
    };
    (b - 1.0) * (f(1.0 + radius_px * radius_px / a2) - f(1.0))
}

/// Footprint (quad) radius [px] of a 3-D star or point sprite with display-normalised energy
/// `energy` and halo share `halo_fraction`, before the clamp: the distance where the core has
/// fallen below `psf::VISIBILITY_CUTOFF` (M0's rule) and the 3-D halo below
/// [`STARS3D_HALO_CUTOFF`], at least [`STARS3D_MIN_RADIUS_PX`]. The core falls to its cutoff at
/// σ·√(2 ln(peak/cutoff)); the halo, peak h·E·I(0), at α·√((peak/cutoff)^(1/β) − 1). Stars
/// whose core peak is below 113 × the cutoff keep the 2-px minimum; the shaded pixels then reach
/// at least 2 px from the centre, which holds ≥ 99.6 % of the σ = 0.65 px core
/// (erf(2/(0.65√2))²), and every pixel beyond is below the cutoff. Twin of
/// `stars3d_footprint_radius_unclamped` in `stars3d.wgsl`. NaN gives the minimum.
pub fn footprint_radius_unclamped_px(energy: f32, halo_fraction: f32) -> f32 {
    let mut r = STARS3D_MIN_RADIUS_PX;
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    if core_peak > VISIBILITY_CUTOFF {
        r = r.max(CORE_SIGMA_PX * (2.0 * (core_peak / VISIBILITY_CUTOFF).ln()).sqrt());
    }
    let halo_peak = halo_fraction * energy * stars3d_moffat(0.0);
    if halo_peak > STARS3D_HALO_CUTOFF {
        let q = (halo_peak / STARS3D_HALO_CUTOFF).powf(1.0 / STARS3D_HALO_BETA);
        r = r.max(STARS3D_HALO_ALPHA_PX * (q - 1.0).sqrt());
    }
    r
}

/// Footprint (quad) radius [px]: [`footprint_radius_unclamped_px`] clamped to
/// `psf::MAX_RADIUS_PX` (64 px). A clamped halo is tapered to black inside it and keeps its
/// light up to the clamp ([`halo_shape`]). Twin of `stars3d_footprint_radius_px` in
/// `stars3d.wgsl`. NaN gives the minimum.
pub fn footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
    footprint_radius_unclamped_px(energy, halo_fraction).min(MAX_RADIUS_PX)
}

/// Whether a splat of display-normalised energy `energy` and halo share `halo_fraction` rises
/// above `psf::VISIBILITY_CUTOFF` (2.55 stops below AgX black) anywhere: its core peak
/// (1 − h)·E/(2πσ²) or its halo peak h·E·Moffat(0) is above the cutoff. The vertex stage culls
/// the rest (the budget ladder's "only stars above `VISIBILITY_CUTOFF` reach the rasteriser").
/// Twin of `star_is_visible` in `stars3d.wgsl`. NaN is invisible.
pub fn star_is_visible(energy: f32, halo_fraction: f32) -> bool {
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    let halo_peak = halo_fraction * energy * stars3d_moffat(0.0);
    core_peak > VISIBILITY_CUTOFF || halo_peak > VISIBILITY_CUTOFF
}

/// Number of `stars` that rise above `psf::VISIBILITY_CUTOFF` on screen at the eye-adapted star
/// exposure `eye_star_ev100`, field of view `fov_y_rad` and `height_px` lines, seen from the SSB
/// at the catalogue epoch (the lead's star-EV sweep, ruling 9).
///
/// Each star is taken on the optical axis (Ω = Ω_c), where its pixels are dimmest, so this is
/// the count the whole frame would draw if every star were at its centre: the vertex stage's
/// rule ([`star_is_visible`] of E·`exposure_scale(EV)`/Ω_c, no twinkle) in f32, with its culls
/// of a star at zero distance or without light, and `pack_stars`'s of a non-finite position or
/// magnitude. The distance is taken from the catalogue position, not through the sector
/// offsets, so a star within f32 rounding of the cutoff may count differently (review r0 M3).
pub fn visible_star_count(
    stars: &[Star3],
    eye_star_ev100: f32,
    fov_y_rad: f32,
    height_px: u32,
) -> u32 {
    let omega = pixel_solid_angle(fov_y_rad, height_px as f32);
    let pre = uni_gfx::exposure_scale(eye_star_ev100);
    let n = stars
        .iter()
        .filter(|s| {
            let pos = Vec3::from(s.pos_pc);
            if !pos.is_finite() || !s.abs_mag_v.is_finite() {
                return false;
            }
            let d = pos.length();
            let lux = star_lux(star_apparent_mag(s.abs_mag_v, d));
            d > 0.0
                && lux > 0.0
                && star_is_visible(
                    (lux * pre / omega).min(MAX_SPLAT_ENERGY),
                    STARS3D_HALO_FRACTION,
                )
        })
        .count();
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The "Vivid star colours · non-physical" colour boost: `c` (linear Rec. 709) with its
/// saturation raised beyond Planck chromaticity, keeping its Rec. 709 luminance
/// (0.2126 R + 0.7152 G + 0.0722 B, ITU-R BT.709-6 item 3.2) so every photometric check stays
/// physical. The app applies it to star-coloured point sprites; the star shader applies its WGSL
/// twin in `stars3d.wgsl` while [`StarInput::vivid_colours`] is on.
///
/// The colour's distance from the grey of its own luminance Y is scaled by
/// [`VIVID_SATURATION`] (1.6, the owner-approved look preview's value). A channel that would go
/// negative is clamped to 0 and the result rescaled to Y again, so very red or blue stars move to
/// the gamut edge with their luminance intact. Black, grey and non-finite colours are unchanged.
pub fn vivid_color(c: Vec3) -> Vec3 {
    let y = c.dot(REC709_Y);
    if !(y > 0.0) {
        return c;
    }
    let grey = Vec3::splat(y);
    // The clamp only removes negative parts, so the boosted luminance is ≥ y > 0.
    let boosted = (grey + VIVID_SATURATION * (c - grey)).max(Vec3::ZERO);
    boosted * (y / boosted.dot(REC709_Y))
}

/// `c` scaled to luminance 1; a black input gives white.
fn unit_luminance(c: Vec3) -> Vec3 {
    let y = c.dot(REC709_Y);
    if y > 0.0 { c / y } else { Vec3::ONE }
}

/// The shared B−V → colour table: [`BV_LUT_LEN`] Planck colours at B−V evenly spaced over
/// [`BV_LUT_MIN`, `BV_LUT_MAX`], each `blackbody_linear_srgb(bv_to_teff(B−V))` (Ballesteros 2012;
/// CIE 1931 CMFs) at luminance 1. Built on first use.
pub(crate) fn bv_lut() -> &'static BvLut {
    static LUT: OnceLock<BvLut> = OnceLock::new();
    LUT.get_or_init(|| BvLut {
        entries: std::array::from_fn(|i| {
            let bv = BV_LUT_MIN + (BV_LUT_MAX - BV_LUT_MIN) * i as f32 / (BV_LUT_LEN - 1) as f32;
            unit_luminance(blackbody_linear_srgb(bv_to_teff(f64::from(bv))))
                .extend(0.0)
                .to_array()
        }),
    })
}

/// Planck colour (linear Rec. 709, luminance 1) of a star of colour index `bv`, interpolated
/// linearly in B−V from the table the star pass reads and clamped to its ends (B−V −0.4 to 2.0).
/// A near star the app draws as a [`crate::PointSprite`] should take this colour (through
/// [`vivid_color`] while vivid colours are on), so it keeps the star pass's colour exactly when
/// it is handed over. Twin of `bv_colour` in `stars3d.wgsl`. NaN gives NaN.
pub fn bv_colour(bv: f32) -> Vec3 {
    let lut = bv_lut();
    let x =
        ((bv - BV_LUT_MIN) / (BV_LUT_MAX - BV_LUT_MIN)).clamp(0.0, 1.0) * (BV_LUT_LEN - 1) as f32;
    let i = (x as usize).min(BV_LUT_LEN - 2);
    let f = x - i as f32;
    let a = Vec3::from_slice(&lut.entries[i][..3]);
    let b = Vec3::from_slice(&lut.entries[i + 1][..3]);
    a + f * (b - a)
}

/// One twinkle draw for slot `slot` at time step `step`: from `pcg3d(slot, step, TWINKLE_SEED)`
/// (Jarzynski & Olano 2020), the sum of three uniforms (Irwin–Hall, mean 1.5, variance 1/4)
/// scaled to mean 0 and variance 1.
fn twinkle_noise(slot: u32, step: u32) -> f32 {
    let h = pcg3d(UVec3::new(slot, step, TWINKLE_SEED));
    let u = (h >> 8_u32).as_vec3() * (1.0 / 16_777_216.0);
    (u.x + u.y + u.z - 1.5) * 2.0
}

/// Where the twinkle is in time on one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TwinkleClock {
    /// The current draw: TDB seconds since J2000 / [`TWINKLE_EYE_S`], floored, modulo 2^32.
    pub step: u32,
    /// How far from draw `step` to the next, in [0, 1].
    pub phase: f32,
    /// Simulated seconds per real second on this frame, |sim dt / real dt|, at least 1.
    pub time_rate: f64,
}

impl Default for TwinkleClock {
    /// J2000.0 in real time (before the first frame).
    fn default() -> Self {
        Self {
            step: 0,
            phase: 0.0,
            time_rate: 1.0,
        }
    }
}

/// The twinkle clock at `epoch` for a frame that advanced the epoch by `sim_dt_s` in `dt_real_s`
/// real seconds (`FrameCtx`'s fields).
///
/// The draws follow the epoch's TDB seconds (lead's r0 ruling: sim seconds, never the frame
/// index), so the twinkle's speed does not follow the frame rate and a capture at a fixed epoch
/// repeats. It is the clock of `sim_time_s()`, computed here in f64 because the f32
/// `sim_time_s()` steps by up to 0.25 s, more than a draw. Whole days are counted first
/// ([`TWINKLE_STEPS_PER_DAY`]) so the fraction stays exact at any epoch; the step wraps modulo
/// 2^32 on both CPU and GPU, so the draws stay continuous across the wrap. Under time warp the
/// draws speed up with the simulation and [`scintillation_index`] lowers their amplitude by the
/// warp factor's square root. A non-finite epoch gives step 0 and phase 0; an unknown or
/// non-finite rate, a pause and slow motion give rate 1.
pub(crate) fn twinkle_clock(epoch: Epoch, sim_dt_s: f64, dt_real_s: f64) -> TwinkleClock {
    let rate = (sim_dt_s / dt_real_s).abs();
    let time_rate = if dt_real_s > 0.0 && rate.is_finite() {
        rate.max(1.0)
    } else {
        1.0
    };
    let draws = epoch.sec / TWINKLE_EYE_S;
    if !draws.is_finite() {
        return TwinkleClock {
            time_rate,
            ..TwinkleClock::default()
        };
    }
    let whole = draws.floor();
    // Modulo 2^32: the wrapping sum, truncated to its low 32 bits.
    let step = epoch
        .day
        .wrapping_mul(TWINKLE_STEPS_PER_DAY)
        .wrapping_add(whole as i64) as u32;
    TwinkleClock {
        step,
        phase: ((draws - whole) as f32).min(1.0),
        time_rate,
    }
}

/// Scintillation index σ_I (relative rms intensity) of a star at the zenith seen by the naked
/// eye, for B3's `strength` (`StarInput::scintillation`) and the clock's `time_rate`:
/// Young's (1967) σ_I = 0.09·D^(−2/3)·X^1.75·S·(2t)^(−1/2) ([`YOUNG_SCINTILLATION_COEFF`]) with
/// D = [`EYE_PUPIL_CM`], X = 1 and t = [`TWINKLE_EYE_S`] × `time_rate`; 0.255 at strength 1 in
/// real time.
///
/// - S is B3's `scintillation_strength`, in the place of Young's e^(−h/8000 m): it falls with
///   the camera's altitude as the air column above it and is 1 at Earth's sea level.
/// - X = 1 for every star (approximate): `StarInput` carries no local up, so the growth with
///   each star's air mass (X^1.75 in Young's form, X^1.5 in Osborn et al. 2015's) is not
///   modelled, and stars near the horizon twinkle as little as at the zenith.
/// - Under time warp W the eye's 0.1 s holds W·0.1 s of air, so t = W·0.1 s and the amplitude
///   falls as 1/√W (Young's averaging law); slow motion and a pause keep the real-time value,
///   since that law does not hold below the scintillation's own millisecond time scales.
///
/// Strength is clamped to [0, 1] (NaN is 0); a non-finite rate counts as 1.
pub(crate) fn scintillation_index(strength: f32, time_rate: f64) -> f64 {
    let s = if strength > 0.0 {
        f64::from(strength.min(1.0))
    } else {
        0.0
    };
    let rate = if time_rate.is_finite() {
        time_rate.max(1.0)
    } else {
        1.0
    };
    // At the zenith (see above).
    let air_mass: f64 = 1.0;
    YOUNG_SCINTILLATION_COEFF * EYE_PUPIL_CM.powf(-2.0 / 3.0) * air_mass.powf(1.75) * s
        / (2.0 * TWINKLE_EYE_S * rate).sqrt()
}

/// Log-amplitude σ of the log-normal twinkle I/⟨I⟩ = exp(σn − σ²/2) whose scintillation index
/// e^(σ²) − 1 is `sigma_i`²: σ = √ln(1 + σ_I²). Small-aperture scintillation is close to
/// log-normal (Dravins et al. 1997, PASP 109, 173). 0.251 at σ_I = 0.255.
pub(crate) fn twinkle_log_sigma(sigma_i: f64) -> f32 {
    (sigma_i * sigma_i).ln_1p().sqrt() as f32
}

/// The twinkle factor I/⟨I⟩ of slot `slot` at twinkle draw `step` and `phase` (a
/// [`TwinkleClock`]) for log-amplitude `sigma`: exp(σn − σ²/2), log-normal. n moves smoothly
/// from draw `step` to draw `step + 1` (smoothstep of `phase`, renormalised to unit variance),
/// driven only by the slot and the clock, so a capture at a fixed epoch repeats. The mean is 1
/// to 1e-4, so the photometry stays the catalogue's on average: at a draw n is Irwin–Hall, whose
/// exact mean factor is (sinh σ/σ)³·e^(−σ²/2), 0.99993 at σ = 0.25 (review r0 M10). `sigma` ≤ 0
/// or NaN gives exactly 1 (no twinkle in space). Twin of `twinkle_factor` in `stars3d.wgsl`.
pub(crate) fn twinkle_factor(slot: u32, step: u32, phase: f32, sigma: f32) -> f32 {
    if !(sigma > 0.0) {
        return 1.0;
    }
    let f = phase.clamp(0.0, 1.0);
    let t = f * f * (3.0 - 2.0 * f);
    let n0 = twinkle_noise(slot, step);
    let n1 = twinkle_noise(slot, step.wrapping_add(1));
    // A blend of two independent unit-variance draws has variance (1 − t)² + t².
    let n = (n0 + t * (n1 - n0)) / ((1.0 - t) * (1.0 - t) + t * t).sqrt();
    (sigma * n - 0.5 * sigma * sigma).exp()
}

/// CPU twins of the vertex stages (test only): what `splat_vertex` in `stars3d.wgsl` computes
/// for one star or point.
#[cfg(test)]
pub(crate) mod twin {
    use super::*;
    use crate::stars::{MIN_CLIP_W, pixel_solid_angle_off_axis};
    use glam::{Mat4, Vec2, Vec4};

    /// What the vertex stages read from group 0.
    #[derive(Clone, Copy, Debug)]
    pub struct ViewTwin {
        pub view_rot: Mat4,
        pub proj: Mat4,
        pub size: Vec2,
        pub fov_y: f32,
        pub star_pre: f32,
    }

    impl ViewTwin {
        /// A camera whose optical axis (camera −Z) points along `axis` [root axes], with
        /// `fov_y_deg` and `size` px, at star exposure `star_ev100`.
        pub fn looking(axis: Vec3, fov_y_deg: f32, size: Vec2, star_ev100: f32) -> Self {
            let cam_to_root = glam::Quat::from_rotation_arc(Vec3::NEG_Z, axis.normalize());
            let fov_y = fov_y_deg.to_radians();
            Self {
                view_rot: Mat4::from_quat(cam_to_root.inverse()),
                proj: uni_gfx::reversed_z_infinite_proj(fov_y, size.x / size.y, 0.1),
                size,
                fov_y,
                star_pre: uni_gfx::exposure_scale(star_ev100),
            }
        }
    }

    /// One splat as the vertex stage leaves it.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Splat {
        pub ndc: Vec2,
        pub centre_px: Vec2,
        pub radiance: Vec3,
        pub energy: f32,
        pub halo_fraction: f32,
        pub radius_px: f32,
        /// The halo's weight scale and window exponent ([`super::halo_shape`]).
        pub halo_scale: f32,
        pub taper: f32,
        pub depth: f32,
    }

    /// Clip position of a culled vertex: all four coincide outside the clip volume (z > w).
    pub const CULLED: Vec4 = Vec4::new(2.0, 2.0, 2.0, 1.0);

    /// Corner (±1, ±1) of index-buffer vertex `vi`. Twin of `splat_corner`.
    fn splat_corner(vi: u32) -> Vec2 {
        Vec2::new((vi & 1) as f32, (vi >> 1) as f32) * 2.0 - 1.0
    }

    /// The splat of a source in unit direction `dir` (root axes) with steady illuminance `lux`
    /// scaled by `twinkle`, and colour `colour`, at NDC depth `depth`, at the view's star
    /// exposure; `None` where the vertex stage culls it. Twin of `splat_vertex`.
    pub fn splat(
        view: &ViewTwin,
        dir: Vec3,
        lux: f32,
        twinkle: f32,
        colour: Vec3,
        depth: f32,
    ) -> Option<Splat> {
        splat_at(view, dir, lux, twinkle, colour, depth, view.star_pre)
    }

    /// [`splat`] written × `pre` instead of the star exposure. Twin of `splat_vertex_at`.
    pub fn splat_at(
        view: &ViewTwin,
        dir: Vec3,
        lux: f32,
        twinkle: f32,
        colour: Vec3,
        depth: f32,
        pre: f32,
    ) -> Option<Splat> {
        let view_dir = (view.view_rot * dir.extend(0.0)).truncate();
        let clip = view.proj * view_dir.extend(0.0);
        if !(clip.w > MIN_CLIP_W) || !(lux > 0.0) {
            return None;
        }
        let size = view.size;
        let ndc = Vec2::new(clip.x, clip.y) / clip.w;
        let centre = Vec2::new(0.5 * (ndc.x + 1.0) * size.x, 0.5 * (1.0 - ndc.y) * size.y);
        let cos_theta = -view_dir.z / view_dir.length();
        let omega = pixel_solid_angle_off_axis(view.fov_y, size.y, cos_theta);
        let energy = (lux * twinkle * pre / omega).min(MAX_SPLAT_ENERGY);
        // The eye's halo is part of every star's PSF (B4L), whatever it twinkles to.
        let h = STARS3D_HALO_FRACTION;
        if !star_is_visible(energy, h) {
            return None;
        }
        let shape = halo_shape(energy, h);
        let radius = shape.radius_px;
        let reach = Vec2::splat(radius + 1.0);
        if !(centre.cmpgt(-reach).all() && centre.cmplt(size + reach).all()) {
            return None;
        }
        Some(Splat {
            ndc,
            centre_px: centre,
            radiance: colour * energy,
            energy,
            halo_fraction: h,
            radius_px: radius,
            halo_scale: shape.scale,
            taper: shape.taper,
            depth,
        })
    }

    /// Clip position of index-buffer vertex `vi` (0..4) of `s`'s quad, or [`CULLED`].
    pub fn vertex(view: &ViewTwin, s: Option<Splat>, vi: u32) -> Vec4 {
        match s {
            None => CULLED,
            Some(s) => (s.ndc + splat_corner(vi) * (s.radius_px + 0.5) * 2.0 / view.size)
                .extend(s.depth)
                .extend(1.0),
        }
    }

    /// The splat of the star in slot `slot`, with `sector_rel` the uploaded sector centres
    /// (relative to the anchor). Twin of `vs_stars`.
    pub fn star_splat(
        view: &ViewTwin,
        packed: &PackedStars,
        sector_rel: &[[f32; 4]],
        params: &StarsParams,
        slot: u32,
    ) -> Option<Splat> {
        if params.suppressed.iter().flatten().any(|&s| s == slot) {
            return None;
        }
        let star = &packed.gpu[slot as usize];
        let sector = Vec3::from_slice(&sector_rel[star.vel_bv_flags[3] as usize][..3])
            - Vec3::from_slice(&params.camera_years[..3]);
        let p = star_rel_pos_pc(star, sector, params.camera_years[3]);
        let d = p.length();
        if !(d > 0.0) {
            return None;
        }
        let lux = star_lux(star_apparent_mag(star.offset_absmag[3], d));
        let twinkle = twinkle_factor(slot, params.twinkle_step[0], params.look[2], params.look[0]);
        // The colour only for a drawn splat, as `vs_stars`.
        let mut s = splat(view, p / d, lux, twinkle, Vec3::ONE, 0.0)?;
        let mut colour = bv_colour(unpack_f16_pair(star.vel_bv_flags[1])[1]);
        if params.look[1] > 0.5 {
            colour = vivid_color(colour);
        }
        s.radiance *= colour;
        Some(s)
    }

    /// The share of a sky source's light a star keeps under `lens_mask` value `mask`:
    /// (1 − r)(1 − g). Twin of `star_sky_factor` in M0's `stars.wgsl`, which both catalogue
    /// passes call (`stars3d.wgsl` imports it).
    pub fn star_sky_factor(mask: Vec4) -> f32 {
        (1.0 - mask.x.clamp(0.0, 1.0)) * (1.0 - mask.y.clamp(0.0, 1.0))
    }
}

/// naga helpers shared by the uni-sky B4 tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use uni_gfx::{ShaderLib, validate_wgsl};
    use wgpu::naga;

    pub(crate) fn module(path: &str) -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(path).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{path}: {e}\n{src}"))
    }

    /// Size and member offsets of the WGSL struct `name` in `m`, and whether every member is a
    /// vec4 or an array of vec4s (the uniform layout rule).
    pub(crate) fn wgsl_struct(m: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>, bool) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let is_vec4 = |h: naga::Handle<naga::Type>| {
            matches!(
                m.types[h].inner,
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    ..
                }
            )
        };
        let vec4_only = members.iter().all(|mb| match m.types[mb.ty].inner {
            naga::TypeInner::Array { base, .. } => is_vec4(base),
            naga::TypeInner::Struct { .. } => true, // checked on its own
            _ => is_vec4(mb.ty),
        });
        let offsets = members
            .iter()
            .map(|mb| (mb.name.clone().unwrap_or_default(), mb.offset as usize))
            .collect();
        (layouter[handle].size as usize, offsets, vec4_only)
    }

    /// Whether `block`, or a block nested in it, calls `f`.
    pub(crate) fn calls(block: &naga::Block, f: naga::Handle<naga::Function>) -> bool {
        block.iter().any(|s| match s {
            naga::Statement::Call { function, .. } => *function == f,
            naga::Statement::Block(b) => calls(b, f),
            naga::Statement::If { accept, reject, .. } => calls(accept, f) || calls(reject, f),
            naga::Statement::Loop {
                body, continuing, ..
            } => calls(body, f) || calls(continuing, f),
            naga::Statement::Switch { cases, .. } => cases.iter().any(|c| calls(&c.body, f)),
            _ => false,
        })
    }

    pub(crate) fn func(m: &naga::Module, name: &str) -> naga::Handle<naga::Function> {
        m.functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(name))
            .map(|(h, _)| h)
            .unwrap_or_else(|| panic!("no fn {name}"))
    }

    pub(crate) fn entry<'a>(m: &'a naga::Module, name: &str) -> &'a naga::Function {
        &m.entry_points
            .iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("no entry point {name}"))
            .function
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::twin::*;
    use super::*;
    use crate::psf::{
        CORE_SIGMA_PX, HALO_ALPHA_PX, HALO_BETA, HALO_ENERGY_FRACTION, HALO_MIN_LUX, MAX_RADIUS_PX,
        MIN_RADIUS_PX, VISIBILITY_CUTOFF, pixel_coverage,
    };
    use crate::stars::{DEFAULT_STAR_EV100, MIN_CLIP_W, pixel_solid_angle};
    use crate::{POINTS_WGSL, STARS3D_WGSL};
    use core::mem::{offset_of, size_of};
    use glam::{Vec2, Vec4};
    use std::f32::consts::TAU;
    use wgpu::naga;

    /// ITU-R BT.709-6 item 3.2 luminance weights.
    fn y709(c: Vec3) -> f32 {
        c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    fn baked_stars() -> Vec<Star3> {
        let path = uni_data::data_dir().join("baked/athyg_m10.bin");
        uni_data::load_athyg(&path).expect("committed athyg_m10.bin")
    }

    fn pos_f64(s: &Star3) -> DVec3 {
        DVec3::from_array(s.pos_pc.map(f64::from))
    }

    #[test]
    fn apparent_mag_from_ssb_matches_catalogue() {
        let stars = baked_stars();
        let step = stars.len() / 100;
        let mut checked = 0;
        for s in stars.iter().step_by(step).take(100) {
            let d = pos_f64(s).length();
            let want = f64::from(s.abs_mag_v) + 5.0 * d.log10() - 5.0;
            let got = star_apparent_mag(s.abs_mag_v, d as f32);
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "M {} d {d}: {got} vs {want}",
                s.abs_mag_v
            );
            checked += 1;
        }
        assert_eq!(checked, 100);
    }

    #[test]
    fn star_lux_matches_uni_core() {
        for v in [-26.74, -12.0, -1.46, 0.0, 6.5, 10.0, 15.0] {
            let want = uni_core::star_illuminance_lux(v);
            let got = f64::from(star_lux(v as f32));
            // f32 rounding of V itself (half an ulp at 26.74 is 1e-6 mag) dominates.
            assert!((got / want - 1.0).abs() < 5e-6, "V {v}: {got} vs {want}");
        }
    }

    /// Barnard's star moves 10.39″/yr on the sky (the brief's value; Gaia measures its proper
    /// motion as 10.39″/yr), so 1039″ ± 2 % in 100 years, through the packed f16 velocity and the
    /// f32 GPU path, seen from the SSB.
    #[test]
    fn barnard_proper_motion_100yr() {
        let stars = baked_stars();
        let names_path = uni_data::data_dir().join("baked/athyg_names.ron");
        let names = uni_data::load_star_names(&names_path).expect("committed athyg_names.ron");
        let index = names
            .entries
            .iter()
            .find(|n| n.proper.as_deref() == Some("Barnard's Star"))
            .expect("Barnard's Star in athyg_names.ron")
            .index;
        let packed = pack_stars(&stars);
        let slot = packed.slots[index as usize] as usize;
        let star = &packed.gpu[slot];
        let sector = packed.table.sector_index[slot] as usize;
        let rel = packed.table.centres_rel(DVec3::ZERO);
        let sector_rel = Vec3::from_slice(&rel[sector][..3]);
        let p0 = star_rel_pos_pc(star, sector_rel, 0.0).as_dvec3();
        let p1 = star_rel_pos_pc(star, sector_rel, 100.0).as_dvec3();
        assert!((p0 - pos_f64(&stars[index as usize])).length() < 1e-5);
        let moved = p0.angle_between(p1).to_degrees() * 3600.0;
        let want = 10.39 * 100.0;
        assert!((moved / want - 1.0).abs() < 0.02, "{moved}″ vs {want}″");
    }

    /// A row of three stars ahead of the camera; the middle star's index is suppressed. Its four
    /// vertices coincide (zero area, outside the clip volume); its neighbours have real quads.
    #[test]
    fn suppressed_star_is_degenerate() {
        let stars: Vec<Star3> = (0..3)
            .map(|i| {
                let x = 0.2 * (i as f32 - 1.0);
                Star3::new([x, 0.0, -10.0], 0.0, [0.0; 3], 0.6, i, 0)
            })
            .collect();
        let packed = pack_stars(&stars);
        let input = StarInput {
            suppressed: vec![1],
            ..StarInput::default()
        };
        let params = stars_params(&input, &TwinkleClock::default(), DVec3::ZERO, &packed.slots);
        let rel = packed.table.centres_rel(DVec3::ZERO);
        let view = ViewTwin::looking(Vec3::NEG_Z, 60.0, Vec2::new(1920.0, 1080.0), -4.0);
        // Twice the area of triangle (a, b, c) in NDC.
        let area = |a: Vec4, b: Vec4, c: Vec4| {
            let (a, b, c) = (
                a.truncate().truncate(),
                b.truncate().truncate(),
                c.truncate().truncate(),
            );
            (b - a).perp_dot(c - a).abs()
        };
        for (index, suppressed) in [(0, false), (1, true), (2, false)] {
            let slot = packed.slots[index];
            let s = star_splat(&view, &packed, &rel, &params, slot);
            let v: Vec<Vec4> = (0..4).map(|vi| vertex(&view, s, vi)).collect();
            // The index buffer's two triangles: (0, 1, 2) and (2, 1, 3).
            let quad = area(v[0], v[1], v[2]) + area(v[2], v[1], v[3]);
            if suppressed {
                assert!(s.is_none());
                assert_eq!(quad, 0.0, "star {index}");
                assert!(v.iter().all(|p| p.z > p.w), "outside the clip volume");
            } else {
                assert!(quad > 0.0, "star {index}: {v:?}");
            }
        }
    }

    #[test]
    fn footprint_minimum_two_px() {
        // The faintest visible splat: its core peak, E/(2πσ²), just above the cutoff.
        let threshold = VISIBILITY_CUTOFF * TAU * CORE_SIGMA_PX * CORE_SIGMA_PX;
        let faintest = threshold * 1.0001;
        assert!(star_is_visible(faintest, 0.0));
        assert!(!star_is_visible(threshold * 0.9999, 0.0));
        assert!(!star_is_visible(f32::NAN, 0.0));
        assert_eq!(footprint_radius_px(faintest, 0.0), STARS3D_MIN_RADIUS_PX);
        // Through the vertex twin: on the axis at the default exposure the faintest drawn star
        // (its core holding 1 − h of the light) has a 2 px quad radius, and one 0.01 % fainter
        // is culled.
        let view = ViewTwin::looking(Vec3::X, 60.0, Vec2::new(1920.0, 1080.0), DEFAULT_STAR_EV100);
        let lux = threshold * pixel_solid_angle(view.fov_y, view.size.y)
            / (view.star_pre * (1.0 - STARS3D_HALO_FRACTION));
        let s = splat(&view, Vec3::X, lux * 1.0001, 1.0, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.radius_px, 2.0);
        assert!(splat(&view, Vec3::X, lux * 0.9999, 1.0, Vec3::ONE, 0.0).is_none());
        // Sirius (V = −1.46): its eye halo reaches the display's black at R = 42.5 px (B4L; M0's
        // 2 % glare stopped at 18.31 px).
        let s = splat(&view, Vec3::X, star_lux(-1.46), 1.0, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.halo_fraction, STARS3D_HALO_FRACTION);
        let halo_at_r = s.energy * s.halo_fraction * stars3d_moffat(s.radius_px);
        assert!(
            (halo_at_r / STARS3D_HALO_CUTOFF - 1.0).abs() < 1e-3,
            "{halo_at_r}"
        );
        assert!((s.radius_px - 42.51).abs() < 0.05, "{}", s.radius_px);
    }

    /// Without a halo the 3-D footprint is M0's above M0's 3-px minimum and falls to 2 px below
    /// it; with the eye's halo it is where the core or the halo falls to the cutoff.
    #[test]
    fn footprint_matches_m0_rule() {
        let mut energy = 1e-5;
        while energy < 1e10 {
            let m0 = crate::psf::footprint_radius_px(energy, 0.0);
            let r = footprint_radius_px(energy, 0.0);
            if m0 > MIN_RADIUS_PX {
                assert_eq!(r, m0, "energy {energy}");
            } else {
                assert!((STARS3D_MIN_RADIUS_PX..=MIN_RADIUS_PX).contains(&r));
            }
            let h = STARS3D_HALO_FRACTION;
            let r = footprint_radius_px(energy, h);
            let core_peak = (1.0 - h) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
            let core_at = |x: f32| core_peak * (-0.5 * (x / CORE_SIGMA_PX).powi(2)).exp();
            let halo_at = |x: f32| h * energy * stars3d_moffat(x);
            if r > STARS3D_MIN_RADIUS_PX && r < MAX_RADIUS_PX {
                // At R one of the two meets its cutoff and neither is above it.
                let (c, hl) = (
                    core_at(r) / VISIBILITY_CUTOFF,
                    halo_at(r) / STARS3D_HALO_CUTOFF,
                );
                assert!(
                    (c.max(hl) - 1.0).abs() < 2e-3,
                    "{energy}: core {c} halo {hl}"
                );
            }
            energy *= 1.5;
        }
        assert_eq!(footprint_radius_px(f32::NAN, 0.0), STARS3D_MIN_RADIUS_PX);
        assert_eq!(
            footprint_radius_px(1e30, STARS3D_HALO_FRACTION),
            MAX_RADIUS_PX
        );
    }

    /// The 3-D halo is the CIE general disability glare equation (Vos & van den Berg 1999, CIE
    /// 146:2002; A = 25 yr, p = 0.5) at the reference display (0.0556° per pixel): per
    /// steradian, h·I(r)/Ω_px is within 12 % of 10/θ³ + (5/θ² + 0.1p/θ)(1 + (A/62.5)⁴) + 0.0025p
    /// from 0.3° to 3° (the fit's range), and holds 0.165 of the light between 0.1° and 3°
    /// against the equation's 0.219.
    #[test]
    fn halo_follows_the_cie_glare_equation() {
        let deg_per_px = 60.0_f64 / 1080.0;
        let px_sr = (deg_per_px.to_radians()).powi(2);
        let cie = |t: f64| {
            let (age, p) = (25.0_f64, 0.5_f64);
            10.0 / t.powi(3)
                + (5.0 / t.powi(2) + 0.1 * p / t) * (1.0 + (age / 62.5).powi(4))
                + 0.0025 * p
        };
        for i in 0..=20 {
            let theta = 0.3 * 10f64.powf(f64::from(i) / 20.0);
            let r = (theta / deg_per_px) as f32;
            let ours = f64::from(STARS3D_HALO_FRACTION * stars3d_moffat(r)) / px_sr;
            assert!(
                (ours / cie(theta) - 1.0).abs() < 0.12,
                "{theta}°: {ours} vs {}",
                cie(theta)
            );
        }
        // Light between 0.1° and 3°: the unwindowed Moffat's (1 + R²/α²)^(1−β) differences.
        let alpha = f64::from(STARS3D_HALO_ALPHA_PX) * deg_per_px;
        let beyond = |t: f64| (1.0 + (t / alpha).powi(2)).powf(1.0 - f64::from(STARS3D_HALO_BETA));
        let ours = f64::from(STARS3D_HALO_FRACTION) * (beyond(0.1) - beyond(3.0));
        assert!((ours - 0.165).abs() < 0.005, "{ours}");
    }

    /// The windowed halo's closed-form energy matches a numerical integral of I(r)(1 − r²/R²)².
    #[test]
    fn halo_energy_closed_form() {
        for radius in [2.0_f32, 3.0, 10.0, 32.75, 64.0] {
            let n = 200_000;
            let dr = f64::from(radius) / f64::from(n);
            let numeric: f64 = (0..n)
                .map(|i| {
                    let r = (f64::from(i) + 0.5) * dr;
                    let w = 1.0 - (r * r) / f64::from(radius * radius);
                    2.0 * std::f64::consts::PI
                        * r
                        * f64::from(stars3d_moffat(r as f32))
                        * w
                        * w
                        * dr
                })
                .sum();
            let closed = f64::from(stars3d_halo_energy(radius));
            assert!(
                (closed / numeric - 1.0).abs() < 2e-5,
                "R {radius}: {closed} vs {numeric}"
            );
        }
        assert!((stars3d_halo_energy(1e6) - 1.0).abs() < 1e-3);
    }

    /// The halo sampled at pixel centres sums to 1 within 1.2 % for every footprint radius from
    /// 2 px and any sub-pixel position, and the whole PSF (core + halo) to 1 within 0.5 %.
    #[test]
    fn stars3d_halo_conserves_energy() {
        for radius in [2.0_f32, 2.5, 3.0, 5.0, 10.0, 32.75, 64.0] {
            let inv_energy = 1.0 / stars3d_halo_energy(radius);
            for (sx, sy) in [(0.5, 0.5), (0.0, 0.0), (0.13, 0.91), (0.99, 0.37)] {
                let n = radius.ceil() as i32 + 1;
                let (mut halo, mut core) = (0.0_f32, 0.0_f32);
                for dy in -n..=n {
                    for dx in -n..=n {
                        let (ex, ey) = (dx as f32 - sx, dy as f32 - sy);
                        let r = Vec2::new(ex + 0.5, ey + 0.5).length();
                        if r < radius + 0.5 {
                            halo += stars3d_halo_weight(r, radius, inv_energy, 2.0);
                            core += pixel_coverage(ex, CORE_SIGMA_PX)
                                * pixel_coverage(ey, CORE_SIGMA_PX);
                        }
                    }
                }
                assert!(
                    (halo - 1.0).abs() < 0.012,
                    "R {radius} ({sx}, {sy}): halo {halo}"
                );
                let h = STARS3D_HALO_FRACTION;
                let total = (1.0 - h) * core + h * halo;
                assert!(
                    (total - 1.0).abs() < 0.005,
                    "R {radius} ({sx}, {sy}): {total}"
                );
            }
        }
    }

    /// Final review render I2: the taper table is W_n = ∫₀^64 2πr·Moffat(r)(1 − r²/64²)^n dr,
    /// recomputed here by Simpson's rule in s = ln(1 + r²/α²) (f64, 20 000 intervals); its n = 2
    /// entry is the closed form; log-linear interpolation is within 1e-3 of the quadrature at
    /// half-integer n.
    #[test]
    fn taper_energy_table_by_quadrature() {
        let (a, b, r) = (
            f64::from(STARS3D_HALO_ALPHA_PX),
            f64::from(STARS3D_HALO_BETA),
            f64::from(MAX_RADIUS_PX),
        );
        let w = |n: f64| {
            let s_max = (1.0 + r * r / (a * a)).ln();
            let steps = 20_000;
            let h = s_max / f64::from(steps);
            let f = |s: f64| {
                let base = (1.0 - a * a * (s.exp() - 1.0) / (r * r)).max(0.0);
                (b - 1.0) * ((1.0 - b) * s).exp() * base.powf(n)
            };
            let mut sum = f(0.0) + f(s_max);
            for i in 1..steps {
                sum += f(f64::from(i) * h) * if i % 2 == 1 { 4.0 } else { 2.0 };
            }
            sum * h / 3.0
        };
        for (k, &table) in STARS3D_TAPER_ENERGY.iter().enumerate() {
            let q = w(2.0 + k as f64);
            assert!(
                (f64::from(table) / q - 1.0).abs() < 1e-6,
                "n = {}: {table} vs {q}",
                k + 2
            );
        }
        assert!((STARS3D_TAPER_ENERGY[0] / stars3d_halo_energy(MAX_RADIUS_PX) - 1.0).abs() < 1e-5);
        for k in 0..22 {
            let n = 2.5 + k as f32;
            let q = w(f64::from(n));
            let got = f64::from(stars3d_taper_energy(n));
            assert!((got / q - 1.0).abs() < 1e-3, "n = {n}: {got} vs {q}");
        }
        // Clamped to the table's range.
        assert_eq!(stars3d_taper_energy(1.0), STARS3D_TAPER_ENERGY[0]);
        assert!((stars3d_taper_energy(99.0) / STARS3D_TAPER_ENERGY[22] - 1.0).abs() < 1e-6);
    }

    /// Final review render I2: a clamped halo ends at the display's black whatever the source's
    /// light (its last pixel ring, r = R − 1, at most half the cutoff), keeps its light up to the
    /// clamp (the sum of its pixel weights is K = M(64) + 1 − M(R_free) within 2 %), and its
    /// scale is continuous where the clamp starts; an unclamped halo is B4L's (n = 2, unit
    /// energy). The belt-view Sun at star exposure (energy 1.6e11) was a 129-px flat-white disk
    /// with a one-pixel edge before (final review): now its glow ends at black.
    #[test]
    fn clamped_halo_ends_at_black() {
        let h = STARS3D_HALO_FRACTION;
        // Unclamped: as before.
        for energy in [1.0_f32, 30.0, 100.0] {
            let s = halo_shape(energy, h);
            assert!(s.radius_px < MAX_RADIUS_PX, "{energy}");
            assert_eq!(s.taper, 2.0);
            assert_eq!(s.scale, 1.0 / stars3d_halo_energy(s.radius_px));
            assert_eq!(s.radius_px, footprint_radius_px(energy, h));
        }
        let mut last_n = 2.0;
        for e in 3..=30 {
            let energy = 10f32.powi(e);
            let s = halo_shape(energy, h);
            if footprint_radius_unclamped_px(energy, h) <= MAX_RADIUS_PX {
                continue;
            }
            assert_eq!(s.radius_px, MAX_RADIUS_PX);
            assert!(
                s.taper >= last_n && s.taper <= STARS3D_TAPER_N_MAX,
                "{energy}: {s:?}"
            );
            last_n = s.taper;
            // The last ring and everything beyond it is at or below the display's black.
            for r in [
                MAX_RADIUS_PX - 1.0,
                MAX_RADIUS_PX - 0.5,
                MAX_RADIUS_PX - 0.01,
            ] {
                let v = h * energy * stars3d_halo_weight(r, s.radius_px, s.scale, s.taper);
                assert!(v <= STARS3D_HALO_CUTOFF, "{energy} at {r} px: {v} ({s:?})");
            }
            // The light kept up to the clamp.
            let kept = 1.0
                - (stars3d_moffat_energy(footprint_radius_unclamped_px(energy, h))
                    - stars3d_moffat_energy(MAX_RADIUS_PX))
                .max(0.0);
            let n = MAX_RADIUS_PX as i32 + 1;
            let mut sum = 0.0_f64;
            for dy in -n..=n {
                for dx in -n..=n {
                    let r = Vec2::new(dx as f32 + 0.5 - 0.3, dy as f32 + 0.5 - 0.7).length();
                    sum += f64::from(stars3d_halo_weight(r, s.radius_px, s.scale, s.taper));
                }
            }
            assert!(
                (sum / f64::from(kept) - 1.0).abs() < 0.02,
                "{energy}: {sum} vs {kept}"
            );
            assert!(kept > 1.0 - (1.0 - stars3d_moffat_energy(MAX_RADIUS_PX)) - 1e-6);
        }
        assert!(
            last_n > 10.0,
            "the cap's energy needs a steep taper: {last_n}"
        );
        // Continuous where the clamp starts: the energy whose unclamped radius is 64 px.
        let (mut lo, mut hi) = (1.0_f32, 1.0e6);
        for _ in 0..60 {
            let mid = (lo * hi).sqrt();
            if footprint_radius_unclamped_px(mid, h) > MAX_RADIUS_PX {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let (a, b) = (halo_shape(lo, h), halo_shape(hi, h));
        assert!((a.scale / b.scale - 1.0).abs() < 1e-3, "{a:?} {b:?}");
        assert!((a.radius_px - b.radius_px).abs() < 1e-3 && b.taper < 2.001);
    }
    #[test]
    fn visible_star_count_follows_limiting_magnitude() {
        let stars = baked_stars();
        let (fov, h) = (60f32.to_radians(), 1080);
        let apparent: Vec<f64> = stars
            .iter()
            .map(|s| f64::from(s.abs_mag_v) + 5.0 * pos_f64(s).length().log10() - 5.0)
            .collect();
        let below = |v: f64| apparent.iter().filter(|&&m| m < v).count() as u32;
        // At the default star EV every V ≤ 10 star is drawn (the limit is V ≈ 13.6 with 24 % of
        // the light in the eye's halo, B4L). The bake also holds 520 nearby stars fainter than
        // that (apparent V up to 21), which are not.
        let n = visible_star_count(&stars, DEFAULT_STAR_EV100, fov, h);
        assert!(n >= below(10.0 + 1e-3), "{n}");
        assert_eq!(stars.len() as u32 - n, 520);
        let mut last = u32::MAX;
        for ev in [-4.0f32, 0.0, 2.0, 4.0, 6.0] {
            let n = visible_star_count(&stars, ev, fov, h);
            assert!(n <= last, "EV {ev}: {n} > {last}");
            last = n;
            // Limiting V: the energy where the core peak, (1 − h) of the light, meets the cutoff.
            let omega = pixel_solid_angle(fov, h as f32);
            let lux_lim = VISIBILITY_CUTOFF * TAU * CORE_SIGMA_PX * CORE_SIGMA_PX * omega
                / (uni_gfx::exposure_scale(ev) * (1.0 - STARS3D_HALO_FRACTION));
            let v_lim = -2.5 * f64::from(lux_lim).log10() - 13.99;
            assert!(
                below(v_lim - 0.01) <= n && n <= below(v_lim + 0.01),
                "EV {ev}: {n}, V_lim {v_lim}"
            );
        }
        assert!(
            last < stars.len() as u32 / 2,
            "EV 6 hides most stars: {last}"
        );
    }

    #[test]
    fn vivid_color_boosts_saturation_and_keeps_luminance() {
        for t in [
            2000.0, 2500.0, 3500.0, 4500.0, 5772.0, 9940.0, 20_000.0, 40_000.0,
        ] {
            let c = uni_core::blackbody_linear_srgb(t);
            let c = c / y709(c);
            let v = vivid_color(c);
            assert!((y709(v) - 1.0).abs() < 1e-5, "{t} K: {c} → {v}");
            assert!(v.min_element() >= 0.0, "{t} K: {v}");
            // Further from grey than the Planck colour, unless it already is grey.
            let chroma = |x: Vec3| (x - Vec3::splat(y709(x))).length();
            assert!(chroma(v) > chroma(c) * 1.2, "{t} K: {c} → {v}");
        }
        assert!((vivid_color(Vec3::ONE) - Vec3::ONE).abs().max_element() < 1e-6);
        assert!(
            (vivid_color(Vec3::splat(0.3)) - Vec3::splat(0.3))
                .abs()
                .max_element()
                < 1e-6
        );
        assert_eq!(vivid_color(Vec3::ZERO), Vec3::ZERO);
        // Inside the gamut it is a plain saturation scale about the luminance.
        let c = Vec3::new(1.2, 1.0, 0.8);
        let y = Vec3::splat(y709(c));
        let want = y + VIVID_SATURATION * (c - y);
        assert!((vivid_color(c) - want).abs().max_element() < 1e-6);
    }

    #[test]
    fn bv_colour_matches_planck() {
        let unit = |c: Vec3| c / y709(c);
        let mut worst = 0.0f32;
        for i in 0..=480 {
            let bv = -0.4 + 2.4 * i as f32 / 480.0;
            let want = unit(uni_core::blackbody_linear_srgb(uni_core::bv_to_teff(
                f64::from(bv),
            )));
            let got = bv_colour(bv);
            worst = worst.max((got - want).abs().max_element());
            assert!((y709(got) - 1.0).abs() < 1e-3, "B−V {bv}: Y {}", y709(got));
        }
        // Measured 2.6e-5 (linear interpolation over 0.019 in B−V).
        assert!(worst < 1e-4, "worst channel error {worst}");
        // Outside the table it clamps to its ends: blue-white and orange-red.
        assert_eq!(bv_colour(-3.0), bv_colour(BV_LUT_MIN));
        assert_eq!(bv_colour(9.0), bv_colour(BV_LUT_MAX));
        assert!(bv_colour(-0.3).z > bv_colour(-0.3).x);
        assert!(bv_colour(1.6).x > bv_colour(1.6).z);
    }

    /// Lead's r0 ruling (review r0 I1): the twinkle's strength is Young's (1967) scintillation
    /// index for the naked eye, σ_I = 0.09·D^(−2/3)·X^1.75·e^(−h/8 km)·(2t)^(−1/2) with D = 0.7 cm,
    /// X = 1 (the zenith: `StarInput` carries no local up), t = 0.1 s (the eye's integration),
    /// and B3's `scintillation_strength` in the place of e^(−h/8 km).
    #[test]
    fn twinkle_amplitude_follows_young_1967() {
        let zenith = 0.09 * 0.7f64.powf(-2.0 / 3.0) * 0.2f64.powf(-0.5);
        assert!((zenith - 0.2553).abs() < 1e-4, "{zenith}");
        assert!((scintillation_index(1.0, 1.0) - zenith).abs() < 1e-12);
        // Linear in B3's strength (Young's e^(−h/8 km)); clamped to [0, 1]; NaN is 0.
        assert!((scintillation_index(0.25, 1.0) - 0.25 * zenith).abs() < 1e-12);
        assert_eq!(scintillation_index(7.0, 1.0), scintillation_index(1.0, 1.0));
        assert_eq!(scintillation_index(0.0, 1.0), 0.0);
        assert_eq!(scintillation_index(-1.0, 1.0), 0.0);
        assert_eq!(scintillation_index(f32::NAN, 1.0), 0.0);
        // Time warp W: the eye's 0.1 s holds W·0.1 s of air, so Young's (2t)^(−1/2) gives σ/√W.
        assert!((scintillation_index(1.0, 100.0) - zenith / 10.0).abs() < 1e-12);
        // Slow motion and pause keep the real-time amplitude; a non-finite rate counts as 1.
        assert_eq!(scintillation_index(1.0, 0.1), scintillation_index(1.0, 1.0));
        assert_eq!(
            scintillation_index(1.0, f64::NAN),
            scintillation_index(1.0, 1.0)
        );
        // The log-normal amplitude whose scintillation index e^(σ²) − 1 is σ_I².
        let s = twinkle_log_sigma(zenith);
        assert!(((f64::from(s) * f64::from(s)).exp_m1() - zenith * zenith).abs() < 1e-6);
        assert_eq!(twinkle_log_sigma(0.0), 0.0);
        // The uniform carries it.
        let input = StarInput {
            scintillation: 1.0,
            ..StarInput::default()
        };
        let p = stars_params(&input, &TwinkleClock::default(), DVec3::ZERO, &[]);
        assert_eq!(p.look[0], s);
        let steady = stars_params(
            &StarInput::default(),
            &TwinkleClock::default(),
            DVec3::ZERO,
            &[],
        );
        assert_eq!(steady.look[0], 0.0);
    }

    /// Lead's r0 ruling: the twinkle runs on TDB seconds (the epoch the frame shows), never the
    /// frame index, so its speed does not follow the frame rate and a capture at a fixed epoch
    /// repeats.
    #[test]
    fn twinkle_clock_runs_on_tdb_seconds() {
        let at = |s: f64| twinkle_clock(Epoch::from_tdb_seconds(s), 1.0 / 60.0, 1.0 / 60.0);
        let t0 = 851_234_567.25; // 2026-12-22, TDB seconds since J2000
        let c = at(t0);
        // Any frame rate: the same instant, the same clock.
        let c30 = twinkle_clock(Epoch::from_tdb_seconds(t0), 1.0 / 30.0, 1.0 / 30.0);
        assert_eq!((c.step, c.phase), (c30.step, c30.phase));
        assert_eq!(c.time_rate, 1.0);
        // Half a draw later: the same draw, half-way to the next; a draw later: the next one.
        let half = at(t0 + 0.5 * TWINKLE_EYE_S);
        assert_eq!(half.step, c.step);
        assert!((half.phase - c.phase - 0.5).abs() < 1e-6, "{half:?} {c:?}");
        assert_eq!(at(t0 + TWINKLE_EYE_S).step, c.step.wrapping_add(1));
        // Across midnight the draws keep counting by one.
        let day = Epoch {
            day: 9_853,
            sec: 0.0,
        };
        let before = twinkle_clock(day.add_seconds(-0.05), 0.0, 0.0);
        let after = twinkle_clock(day, 0.0, 0.0);
        assert_eq!(after.step, before.step.wrapping_add(1));
        assert!((before.phase - 0.5).abs() < 1e-6 && after.phase == 0.0);
        assert_eq!(
            (TWINKLE_STEPS_PER_DAY as f64) * TWINKLE_EYE_S,
            uni_core::DAY
        );
        // The factor is a function of the instant alone: sampled at 60 and at 30 fps over 2 s,
        // the shared instants agree, and 2 s hold 20 draws at either rate.
        let sigma = twinkle_log_sigma(scintillation_index(1.0, 1.0));
        let factor = |c: TwinkleClock| twinkle_factor(5, c.step, c.phase, sigma);
        for k in 0..60 {
            let s60 = at(t0 + f64::from(2 * k) / 60.0);
            let s30 = twinkle_clock(Epoch::from_tdb_seconds(t0 + f64::from(k) / 30.0), 1.0, 1.0);
            assert_eq!(factor(s60), factor(s30), "frame {k}");
        }
        assert_eq!(at(t0 + 2.0).step.wrapping_sub(c.step), 20);
        // Time rate: |sim dt / real dt|, at least 1; paused, unknown or non-finite is 1.
        let rate = |sim: f64, real: f64| twinkle_clock(Epoch::J2000, sim, real).time_rate;
        assert_eq!(rate(1.0 / 6.0, 1.0 / 60.0), 10.0);
        assert_eq!(rate(-1.0 / 6.0, 1.0 / 60.0), 10.0);
        assert_eq!(rate(0.0, 1.0 / 60.0), 1.0);
        assert_eq!(rate(1.0 / 600.0, 1.0 / 60.0), 1.0);
        assert_eq!(rate(1.0, 0.0), 1.0);
        assert_eq!(rate(f64::NAN, 1.0 / 60.0), 1.0);
        // A non-finite epoch gives the clock's start.
        let bad = twinkle_clock(
            Epoch {
                day: 0,
                sec: f64::NAN,
            },
            0.0,
            0.0,
        );
        assert_eq!((bad.step, bad.phase), (0, 0.0));
    }

    /// The shader reads no frame index for the twinkle (review r0 I1).
    #[test]
    fn twinkle_ignores_the_frame_index() {
        let src = include_str!("../shaders/stars3d.wgsl");
        assert!(!src.contains("camera.time.x"));
        let m = module(STARS3D_WGSL);
        assert!(calls(
            &entry(&m, "vs_stars").body,
            func(&m, "twinkle_factor")
        ));
    }

    #[test]
    fn twinkle_is_deterministic_with_unit_mean() {
        let sigma = twinkle_log_sigma(scintillation_index(1.0, 1.0));
        assert_eq!(twinkle_factor(7, 123, 0.3, 0.0), 1.0);
        assert_eq!(twinkle_factor(7, 123, 0.3, f32::NAN), 1.0);
        assert_eq!(
            twinkle_factor(7, 123, 0.3, sigma),
            twinkle_factor(7, 123, 0.3, sigma)
        );
        assert_ne!(
            twinkle_factor(7, 123, 0.3, sigma),
            twinkle_factor(8, 123, 0.3, sigma)
        );
        // Continuous across draws, the u32 wrap included.
        for step in [0, 99, u32::MAX] {
            let end = twinkle_factor(42, step, 1.0, sigma);
            let start = twinkle_factor(42, step.wrapping_add(1), 0.0, sigma);
            assert!((end / start - 1.0).abs() < 1e-6, "{step}: {end} vs {start}");
        }
        // Over 20 000 draws, six frames per draw (60 fps): mean 1 and the scintillation index
        // σ_I² of Young's formula. At a draw n is Irwin–Hall, so the exact mean there is
        // (sinh σ/σ)³·e^(−σ²/2) = 0.99993, not 1; the blends between draws are closer to normal.
        let (n, per) = (20_000u32, 6u32);
        let samples: Vec<f64> = (0..n)
            .flat_map(|step| {
                (0..per)
                    .map(move |k| f64::from(twinkle_factor(42, step, k as f32 / per as f32, sigma)))
            })
            .collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let index = var / (mean * mean);
        let want = scintillation_index(1.0, 1.0).powi(2);
        assert!((mean - 1.0).abs() < 0.01, "mean {mean}");
        assert!((index / want - 1.0).abs() < 0.1, "index {index} vs {want}");
        // Smooth in time: from one frame to the next it changes far less than independent
        // draws every frame would.
        let steps = |v: &[f64]| {
            v.windows(2)
                .map(|w| (w[1].ln() - w[0].ln()).abs())
                .sum::<f64>()
                / v.len() as f64
        };
        let jumpy: Vec<f64> = (0..n * per)
            .map(|i| f64::from(sigma * twinkle_noise(42, i)).exp())
            .collect();
        assert!(steps(&samples) < 0.5 * steps(&jumpy));
        assert!(samples.iter().all(|&x| x > 0.0));
    }

    /// Review r0 M7, B4L: the halo share never depends on the star's light or its twinkle (every
    /// star has the eye's halo), so no star's halo switches on or off, M0's 1e-6 lx threshold
    /// included.
    #[test]
    fn halo_share_ignores_the_twinkle() {
        let view = ViewTwin::looking(Vec3::X, 60.0, Vec2::new(1920.0, 1080.0), -4.0);
        let above = HALO_MIN_LUX * 1.01;
        let s = splat(&view, Vec3::X, above, 0.5, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.halo_fraction, STARS3D_HALO_FRACTION);
        let below = HALO_MIN_LUX * 0.99;
        let s = splat(&view, Vec3::X, below, 2.0, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.halo_fraction, STARS3D_HALO_FRACTION);
        let faint = HALO_MIN_LUX * 1e-4;
        let s = splat(&view, Vec3::X, faint, 1.0, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.halo_fraction, STARS3D_HALO_FRACTION);
        // The twinkle scales the light itself.
        let steady = splat(&view, Vec3::X, above, 1.0, Vec3::ONE, 0.0).unwrap();
        let dim = splat(&view, Vec3::X, above, 0.5, Vec3::ONE, 0.0).unwrap();
        assert!((dim.energy / steady.energy - 0.5).abs() < 1e-6);
    }

    #[test]
    fn star_gpu_packing_round_trips_and_sanitises() {
        let s = Star3::new(
            [1.0, -20.0, 300.0],
            4.5,
            [10.0, -20.0, 30.0],
            1.2,
            9,
            uni_data::FLAG_NAMED,
        );
        let packed = pack_stars(&[s]);
        let g = packed.gpu[0];
        let key = crate::sectors::sector_of(pos_f64(&s));
        assert_eq!(packed.table.sectors, vec![key]);
        let centre = crate::sectors::sector_centre_pc(key);
        let off = Vec3::from_slice(&g.offset_absmag[..3]).as_dvec3();
        assert!((centre + off - pos_f64(&s)).length() < 1e-5);
        assert_eq!(g.offset_absmag[3], 4.5);
        assert_eq!(g.vel_bv_flags[0], s.vel_bv[0]);
        assert_eq!(g.vel_bv_flags[1], s.vel_bv[1]);
        assert_eq!(g.vel_bv_flags[2], uni_data::FLAG_NAMED);
        assert_eq!(g.vel_bv_flags[3], 0);
        // "No velocity" stars stay put; a NaN colour becomes the Sun's B−V; an unusable
        // position or magnitude makes the star invisible.
        let still = Star3::new(
            [5.0, 0.0, 0.0],
            1.0,
            [100.0; 3],
            f32::NAN,
            1,
            uni_data::FLAG_NO_VELOCITY,
        );
        let bad = Star3::new([f32::NAN, 0.0, 0.0], 1.0, [0.0; 3], 0.5, 2, 0);
        let packed = pack_stars(&[still, bad]);
        let g = packed.gpu[packed.slots[0] as usize];
        assert_eq!(g.vel_bv_flags[0], 0);
        assert_eq!(uni_data::f16_to_f32(g.vel_bv_flags[1] as u16), 0.0);
        let unknown = uni_data::f16_to_f32(uni_data::f32_to_f16(UNKNOWN_BV));
        assert_eq!(
            uni_data::f16_to_f32((g.vel_bv_flags[1] >> 16) as u16),
            unknown
        );
        let g = packed.gpu[packed.slots[1] as usize];
        assert!(Vec4::from(g.offset_absmag).is_finite());
        assert_eq!(star_lux(star_apparent_mag(g.offset_absmag[3], 1.0)), 0.0);
    }

    #[test]
    fn stars_params_maps_suppressed_indices_to_slots() {
        let input = StarInput {
            years_since_epoch: 26.9,
            camera_pc: DVec3::new(1.5, 2.0, -3.0),
            suppressed: vec![2, 0, 99, 1, 3, 4, 5, 6, 7, 8],
            vivid_colours: false,
            scintillation: 0.5,
        };
        let slots = [4, 3, 2, 1, 0, 5, 6, 7, 8];
        let clock = TwinkleClock {
            step: 0xDEAD_BEEF,
            phase: 0.25,
            time_rate: 4.0,
        };
        let p = stars_params(&input, &clock, DVec3::new(1.0, 2.0, -2.0), &slots);
        assert_eq!(p.camera_years, [0.5, 0.0, -1.0, 26.9]);
        let sigma = twinkle_log_sigma(scintillation_index(0.5, 4.0));
        assert_eq!(p.look, [sigma, 0.0, 0.25, 0.0]);
        assert_eq!(p.twinkle_step, [0xDEAD_BEEF, 0, 0, 0]);
        // Index 99 is out of range and skipped; at most 8 slots are kept.
        assert_eq!(p.suppressed, [[2, 4, 3, 1], [0, 5, 6, 7]]);
        let p = stars_params(
            &StarInput::default(),
            &TwinkleClock::default(),
            DVec3::ZERO,
            &slots,
        );
        assert_eq!(p.suppressed, [[NO_STAR; 4]; 2]);
        assert_eq!(p.look[1], 1.0);
        // Scintillation is clamped to [0, 1].
        let loud = StarInput {
            scintillation: 7.0,
            ..StarInput::default()
        };
        assert_eq!(
            stars_params(&loud, &TwinkleClock::default(), DVec3::ZERO, &slots).look[0],
            twinkle_log_sigma(scintillation_index(1.0, 1.0))
        );
    }

    #[test]
    fn stars3d_points_shaders_validate() {
        for (path, vs, fs) in [
            (STARS3D_WGSL, "vs_stars", "fs_stars"),
            (POINTS_WGSL, "vs_points", "fs_points"),
        ] {
            let m = module(path);
            let stage = |name: &str| {
                m.entry_points
                    .iter()
                    .find(|e| e.name == name)
                    .map(|e| e.stage)
            };
            assert_eq!(stage(vs), Some(naga::ShaderStage::Vertex), "{path}");
            assert_eq!(stage(fs), Some(naga::ShaderStage::Fragment), "{path}");
        }
    }

    #[test]
    fn stars3d_points_layouts_match_wgsl() {
        let m = module(STARS3D_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "StarGpu");
        assert_eq!(size, 32);
        assert_eq!(size, size_of::<StarGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "offset_absmag".to_owned(),
                    offset_of!(StarGpu, offset_absmag)
                ),
                ("vel_bv_flags".to_owned(), offset_of!(StarGpu, vel_bv_flags)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "StarsParams");
        assert_eq!(size, size_of::<StarsParams>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "camera_years".to_owned(),
                    offset_of!(StarsParams, camera_years)
                ),
                ("look".to_owned(), offset_of!(StarsParams, look)),
                (
                    "twinkle_step".to_owned(),
                    offset_of!(StarsParams, twinkle_step)
                ),
                ("suppressed".to_owned(), offset_of!(StarsParams, suppressed)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "BvColourLut");
        assert_eq!(size, size_of::<BvLut>());
        assert!(vec4_only);
        assert_eq!(offsets, vec![("entries".to_owned(), 0)]);

        let m = module(POINTS_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "PointGpu");
        assert_eq!(size, size_of::<crate::points::PointGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "dir_dist".to_owned(),
                    offset_of!(crate::points::PointGpu, dir_dist)
                ),
                (
                    "color_lux".to_owned(),
                    offset_of!(crate::points::PointGpu, color_lux)
                ),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "PointsParams");
        assert_eq!(size, size_of::<crate::points::PointsParams>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![(
                "sky".to_owned(),
                offset_of!(crate::points::PointsParams, sky)
            )]
        );

        // Group 1 bindings: M0's `stars` at 0 (imported with M0's PSF, used by neither 3-D
        // pass), the lens mask at 1 (M0's declaration, shared), the star pass's 2–5 and the point
        // pass's own buffers at 6 and 7.
        let binding = |m: &naga::Module, name: &str| {
            m.global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .and_then(|(_, v)| v.binding.clone())
                .unwrap_or_else(|| panic!("no binding {name}"))
        };
        let rb = |binding| naga::ResourceBinding { group: 1, binding };
        for (name, b) in [
            ("stars", 0),
            ("lens_mask", LENS_MASK_BINDING),
            ("stars_params", STARS_PARAMS_BINDING),
            ("bv_lut", BV_LUT_BINDING),
            ("catalogue", CATALOGUE_BINDING),
            ("sectors", SECTORS_BINDING),
            ("points", crate::points::POINTS_BINDING),
            ("points_params", crate::points::POINTS_PARAMS_BINDING),
        ] {
            assert_eq!(binding(&m, name), rb(b), "{name}");
        }
    }

    /// Each splat pass's group-1 layout is exactly what its entry points use, checked the way
    /// wgpu checks a pipeline layout against a shader (naga's per-entry-point global use, which
    /// includes the functions an entry point calls): every group-1 global `vs_*` or `fs_*` uses
    /// has an entry of the same kind, at least the shader's size, visible to exactly the stages
    /// that use it, and every entry is used. No other group is used but the renderer's group 0,
    /// and no two globals of the composed module share a slot, so M0's `stars` at binding 0,
    /// which comes with M0's PSF (review r0 I3), cannot clash with a 3-D binding.
    #[test]
    fn splat_layouts_match_entry_point_use() {
        use std::collections::{BTreeMap, HashSet};
        use wgpu::ShaderStages;
        for (path, vs, fs, entries) in [
            (
                STARS3D_WGSL,
                "vs_stars",
                "fs_stars",
                star_bgl_entries().to_vec(),
            ),
            (
                POINTS_WGSL,
                "vs_points",
                "fs_points",
                crate::points::points_bgl_entries().to_vec(),
            ),
        ] {
            let m = module(path);
            let caps =
                naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16;
            let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
                .validate(&m)
                .unwrap();
            let mut slots = HashSet::new();
            for (_, v) in m.global_variables.iter() {
                if let Some(b) = &v.binding {
                    assert!(
                        slots.insert((b.group, b.binding)),
                        "{path}: two globals at {b:?}"
                    );
                }
            }
            // Group-1 binding → (stages that use it, its global).
            let mut used = BTreeMap::new();
            for (i, ep) in m.entry_points.iter().enumerate() {
                let stage = match ep.name.as_str() {
                    n if n == vs => ShaderStages::VERTEX,
                    n if n == fs => ShaderStages::FRAGMENT,
                    _ => continue,
                };
                let ep_info = info.get_entry_point(i);
                for (h, v) in m.global_variables.iter() {
                    let Some(b) = &v.binding else { continue };
                    if ep_info[h].is_empty() {
                        continue;
                    }
                    match b.group {
                        0 => {}
                        1 => {
                            used.entry(b.binding).or_insert((ShaderStages::NONE, h)).0 |= stage;
                        }
                        g => panic!("{path}: {} uses group {g}", ep.name),
                    }
                }
            }
            let mut want: Vec<u32> = entries.iter().map(|e| e.binding).collect();
            want.sort_unstable();
            assert_eq!(used.keys().copied().collect::<Vec<_>>(), want, "{path}");
            for e in &entries {
                let (stages, h) = used[&e.binding];
                let v = &m.global_variables[h];
                let name = v.name.as_deref().unwrap_or("?");
                assert_eq!(e.visibility, stages, "{path}: {name}");
                let ok = match (&e.ty, v.space) {
                    (
                        wgpu::BindingType::Buffer {
                            ty,
                            min_binding_size,
                            ..
                        },
                        space,
                    ) => {
                        let class = match ty {
                            wgpu::BufferBindingType::Uniform => naga::AddressSpace::Uniform,
                            wgpu::BufferBindingType::Storage { read_only } => {
                                assert!(read_only, "{path}: {name}");
                                naga::AddressSpace::Storage {
                                    access: naga::StorageAccess::LOAD,
                                }
                            }
                        };
                        let size = m.types[v.ty].inner.size(m.to_ctx()) as u64;
                        class == space && min_binding_size.is_some_and(|s| s.get() >= size)
                    }
                    (
                        wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { .. },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        naga::AddressSpace::Handle,
                    ) => matches!(
                        m.types[v.ty].inner,
                        naga::TypeInner::Image {
                            dim: naga::ImageDimension::D2,
                            arrayed: false,
                            class: naga::ImageClass::Sampled {
                                kind: naga::ScalarKind::Float,
                                multi: false,
                            },
                        }
                    ),
                    _ => false,
                };
                assert!(ok, "{path}: {name} does not match its layout entry");
            }
            // The lens mask the 3-D passes read is M0's declaration.
            assert_eq!(
                m.global_variables[used[&LENS_MASK_BINDING].1]
                    .name
                    .as_deref(),
                Some("lens_mask")
            );
        }
    }

    /// Review r0 I3: the 3-D passes run M0's PSF core WGSL itself, not a copy of it.
    /// `stars3d.wgsl` imports M0's `stars.wgsl`; neither 3-D shader defines a PSF constant or
    /// function of M0's again; and the splat code calls M0's core (`pixel_coverage`, `psf_erf`)
    /// and solid angle. The footprint radius (`stars3d_footprint_radius_px`, review m16's 2-px
    /// minimum; `footprint_matches_m0_rule`) and, since B4L, the eye's halo (`stars3d_moffat`,
    /// `stars3d_halo_weight`) are the 3-D pass's own; the splat calls neither M0's 3-px footprint
    /// nor M0's glare.
    #[test]
    fn splat_runs_m0_psf_wgsl() {
        let own = [
            ("stars3d.wgsl", include_str!("../shaders/stars3d.wgsl")),
            ("points.wgsl", include_str!("../shaders/points.wgsl")),
        ];
        let import = format!("#import \"{}\"", crate::STARS_SHADER);
        assert!(own[0].1.lines().any(|l| l.trim() == import));
        let m0 = module(crate::STARS_SHADER);
        for f in [
            "psf_erf",
            "pixel_coverage",
            "moffat",
            "halo_weight",
            "pixel_solid_angle",
            "pixel_solid_angle_off_axis",
            "footprint_radius_px",
        ] {
            func(&m0, f);
            for (file, src) in own {
                assert!(!src.contains(&format!("fn {f}(")), "{file} defines {f}");
            }
        }
        for c in [
            "CORE_SIGMA_PX",
            "HALO_ENERGY_FRACTION",
            "HALO_MIN_LUX",
            "HALO_ALPHA_PX",
            "HALO_BETA",
            "MAX_RADIUS_PX",
            "VISIBILITY_CUTOFF",
            "MIN_CLIP_W",
            "CULLED",
        ] {
            assert!(
                m0.constants
                    .iter()
                    .any(|(_, k)| k.name.as_deref() == Some(c)),
                "M0 has no {c}"
            );
            for (file, src) in own {
                assert!(!src.contains(&format!("const {c}:")), "{file} defines {c}");
            }
        }
        let m = module(STARS3D_WGSL);
        let fn_calls = |caller: &str, callee: &str| {
            calls(&m.functions[func(&m, caller)].body, func(&m, callee))
        };
        for (caller, callee) in [
            ("psf_weight", "pixel_coverage"),
            ("psf_weight", "stars3d_halo_weight"),
            ("pixel_coverage", "psf_erf"),
            ("splat_vertex", "splat_vertex_at"),
            ("splat_vertex_at", "pixel_solid_angle_off_axis"),
            ("splat_vertex_at", "star_is_visible"),
            ("splat_vertex_at", "stars3d_halo_shape"),
            ("stars3d_halo_shape", "stars3d_footprint_radius_unclamped"),
            ("stars3d_halo_shape", "stars3d_halo_energy"),
            ("stars3d_halo_shape", "stars3d_taper_energy"),
            ("stars3d_halo_shape", "stars3d_moffat_energy"),
            (
                "stars3d_footprint_radius_px",
                "stars3d_footprint_radius_unclamped",
            ),
            ("stars3d_footprint_radius_unclamped", "stars3d_moffat"),
            ("star_is_visible", "stars3d_moffat"),
            ("stars3d_halo_weight", "stars3d_moffat"),
        ] {
            assert!(fn_calls(caller, callee), "{caller} does not call {callee}");
        }
        // The 3-D halo is the eye's (B4L), not M0's glare.
        assert!(!fn_calls("splat_vertex_at", "footprint_radius_px"));
        assert!(!fn_calls("stars3d_halo_shape", "footprint_radius_px"));
        assert!(!fn_calls("psf_weight", "halo_weight"));
        assert!(!fn_calls("star_is_visible", "moffat"));
    }

    /// The value of `const name: f32 = <literal>;` (or u32) in `m`.
    fn wgsl_const(m: &naga::Module, name: &str) -> naga::Literal {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no const {name}"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(l) => l,
            ref e => panic!("const {name} is not a literal: {e:?}"),
        }
    }

    #[test]
    fn shader_constants_match_cpu() {
        let m = module(STARS3D_WGSL);
        for (name, cpu) in [
            ("CORE_SIGMA_PX", CORE_SIGMA_PX),
            ("HALO_ENERGY_FRACTION", HALO_ENERGY_FRACTION),
            ("HALO_MIN_LUX", HALO_MIN_LUX),
            ("HALO_ALPHA_PX", HALO_ALPHA_PX),
            ("HALO_BETA", HALO_BETA),
            ("STARS3D_MIN_RADIUS_PX", STARS3D_MIN_RADIUS_PX),
            ("MAX_RADIUS_PX", MAX_RADIUS_PX),
            ("VISIBILITY_CUTOFF", VISIBILITY_CUTOFF),
            ("MIN_CLIP_W", MIN_CLIP_W),
            ("PC_PER_YR_PER_KMS", PC_PER_YR_PER_KMS),
            ("LOG10_2", std::f32::consts::LOG10_2),
            ("LOG2_10", std::f32::consts::LOG2_10),
            ("LUX_ZERO_POINT_MAG", LUX_ZERO_POINT_MAG),
            ("MAX_SPLAT_ENERGY", MAX_SPLAT_ENERGY),
            ("BV_LUT_MIN", BV_LUT_MIN),
            ("BV_LUT_MAX", BV_LUT_MAX),
            ("VIVID_SATURATION", VIVID_SATURATION),
            ("STARS3D_HALO_FRACTION", STARS3D_HALO_FRACTION),
            ("STARS3D_HALO_ALPHA_PX", STARS3D_HALO_ALPHA_PX),
            ("STARS3D_HALO_BETA", STARS3D_HALO_BETA),
            ("STARS3D_HALO_CUTOFF", STARS3D_HALO_CUTOFF),
            ("STARS3D_TAPER_TARGET", STARS3D_TAPER_TARGET),
            ("STARS3D_TAPER_N_MAX", STARS3D_TAPER_N_MAX),
        ] {
            assert_eq!(wgsl_const(&m, name), naga::Literal::F32(cpu), "{name}");
        }
        assert_eq!(
            wgsl_const(&m, "STARS3D_TAPER_STEPS"),
            naga::Literal::I32(STARS3D_TAPER_STEPS as i32)
        );
        // The taper table, entry by entry (decimal → f32 by naga and by rustc).
        let (_, table) = m
            .constants
            .iter()
            .find(|(_, k)| k.name.as_deref() == Some("STARS3D_TAPER_ENERGY"))
            .expect("const STARS3D_TAPER_ENERGY");
        let naga::Expression::Compose { components, .. } = &m.global_expressions[table.init] else {
            panic!("STARS3D_TAPER_ENERGY is not an array literal");
        };
        assert_eq!(components.len(), STARS3D_TAPER_ENERGY.len());
        for (k, (&c, cpu)) in components.iter().zip(STARS3D_TAPER_ENERGY).enumerate() {
            let naga::Expression::Literal(naga::Literal::F32(v)) = m.global_expressions[c] else {
                panic!("entry {k} is not an f32 literal");
            };
            assert!(
                (v - cpu).abs() <= cpu * f32::EPSILON,
                "entry {k}: {v} vs {cpu}"
            );
        }
        for (name, cpu) in [
            ("BV_LUT_LEN", BV_LUT_LEN as u32),
            ("TWINKLE_SEED", TWINKLE_SEED),
            ("NO_STAR", NO_STAR),
        ] {
            assert_eq!(wgsl_const(&m, name), naga::Literal::U32(cpu), "{name}");
        }
        // 1 km/s in pc/yr: the brief's 1.0227 × 10⁻⁶.
        assert!((PC_PER_YR_PER_KMS / 1.0227e-6 - 1.0).abs() < 1e-4);
    }

    /// Lead ruling B9c: the catalogue star passes (3-D and M0's BSC5 fallback) and the band have
    /// no depth state, so the pass has no depth attachment either ([`SkyDepth::attachment`]
    /// follows [`SkyDepth::state`]); the reflected bodies' points keep a read-only test at their
    /// true distance (they can be behind a body), never written.
    #[test]
    fn sky_passes_have_no_depth_test_and_points_keep_theirs() {
        assert!(SkyDepth::MaskOnly.state().is_none());
        for (what, depth) in [
            ("3-D stars", STARS3D_DEPTH),
            ("BSC5 stars", crate::stars::STARS_DEPTH),
            ("band", crate::band::BAND_DEPTH),
        ] {
            assert_eq!(depth, SkyDepth::MaskOnly, "{what}");
        }
        // The point pass's state, as `splat_pipeline` (its only pipeline helper) builds it.
        let points = SkyDepth::Test(wgpu::CompareFunction::Greater)
            .state()
            .expect("points test depth");
        assert_eq!(points.format, DEPTH_FORMAT);
        assert_eq!(points.depth_compare, Some(wgpu::CompareFunction::Greater));
        assert_eq!(points.depth_write_enabled, Some(false));
        assert_eq!(points.stencil, wgpu::StencilState::default());
    }

    /// Lead ruling B9c: with the depth test gone, `star_sky_factor` is a star's only occlusion,
    /// in both catalogue passes: a body's anti-aliased edge pixel (coverage 0.5 in g) keeps half
    /// of the star's light, its interior none. M0's BSC5 pass reads g too (it read only r).
    #[test]
    fn catalogue_stars_keep_the_uncovered_share_at_a_body_edge() {
        for path in [STARS3D_WGSL, crate::STARS_SHADER] {
            let m = module(path);
            let fs = if path == STARS3D_WGSL {
                "fs_stars"
            } else {
                "fs_main"
            };
            assert!(
                calls(&entry(&m, fs).body, func(&m, "star_sky_factor")),
                "{path}: {fs} must call star_sky_factor"
            );
        }
        for (mask, want, what) in [
            (
                Vec4::new(0.0, 0.5, 0.0, 0.0),
                0.5,
                "edge pixel, coverage 0.5",
            ),
            (Vec4::new(0.0, 1.0, 0.0, 0.0), 0.0, "body interior"),
            (
                Vec4::new(0.0, 0.25, 0.0, 0.0),
                0.75,
                "edge pixel, coverage 0.25",
            ),
            (
                Vec4::new(0.5, 0.5, 1.0, 1.0),
                0.25,
                "half captured, half covered; b, a ignored",
            ),
        ] {
            assert_eq!(star_sky_factor(mask), want, "{what}");
        }
    }

    /// The star pass writes sky light at the star exposure, masked by (1 − r)(1 − g).
    #[test]
    fn star_pass_uses_star_exposure_and_sky_mask() {
        let m = module(STARS3D_WGSL);
        assert!(calls(&entry(&m, "vs_stars").body, func(&m, "splat_vertex")));
        assert!(calls(
            &m.functions[func(&m, "splat_vertex")].body,
            func(&m, "star_pre")
        ));
        assert!(!calls(
            &m.functions[func(&m, "splat_vertex")].body,
            func(&m, "scene_pre")
        ));
        assert!(calls(
            &entry(&m, "fs_stars").body,
            func(&m, "star_sky_factor")
        ));
        for (mask, want) in [
            (Vec4::ZERO, 1.0),
            (Vec4::new(1.0, 0.0, 0.0, 0.0), 0.0),
            (Vec4::new(0.0, 1.0, 0.0, 0.0), 0.0),
            (Vec4::new(0.25, 0.5, 1.0, 1.0), 0.375),
        ] {
            assert_eq!(star_sky_factor(mask), want, "{mask}");
        }
    }
}
