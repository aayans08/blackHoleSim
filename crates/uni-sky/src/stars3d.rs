//! AT-HYG stars in 3-D, pass "stars" (Task B4; B0 stub).
//!
//! Stars sorted into 16-pc sectors ([`crate::sectors`]) with f32 offsets from each sector's f64
//! origin; per star on the GPU the position at the frame's epoch, the distance, the apparent
//! magnitude, the illuminance and M0's energy-normalised PSF (`crate::psf`, read-only) with a
//! 2-px minimum footprint ([`STARS3D_MIN_RADIUS_PX`]). Output E/Ω × `star_pre()` × (1 − r)(1 − g)
//! at the far plane. Indices in [`StarInput::suppressed`] (near stars drawn as surfaces or
//! points) emit degenerate quads. M0's `StarField` stays as the BSC5 fallback.

use glam::{DVec3, Vec3};
use uni_data::Star3;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

/// Smallest footprint radius of a 3-D star's quad [px] (review m16). M0's
/// `psf::MIN_RADIUS_PX` (3 px) is unchanged.
pub const STARS3D_MIN_RADIUS_PX: f32 = 2.0;

/// What the star pass needs per frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug)]
pub struct StarInput {
    /// Years since the catalogue epoch (`uni_data::STAR3_EPOCH_JD_TDB`, J2000.0), for proper
    /// motion.
    pub years_since_epoch: f64,
    /// Camera − SSB [pc], ICRS axes, f64.
    pub camera_pc: DVec3,
    /// Indices into the slice given to [`StarField3::new`] that the pass must skip (≤ 8; uni-sim's
    /// `Snapshot::suppressed_stars`).
    pub suppressed: Vec<u32>,
    /// The "Vivid star colours · non-physical" option (owner feedback, 2026-11-22; on by default):
    /// a saturation boost beyond Planck chromaticity that keeps each star's Rec. 709 luminance
    /// ([`vivid_color`]). The HUD shows `uni_ui::VIVID_STAR_COLOURS_LABEL` while it is on.
    pub vivid_colours: bool,
    /// Scintillation (twinkle) strength in [0, 1]: `uni_atmo::scintillation_strength` for the
    /// camera, 0 in space. B4 drives the twinkle deterministically from `pcg3d` of the star index
    /// and the frame time, so captures repeat.
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

/// The 3-D star pass (`RenderFeature` named "stars").
pub struct StarField3 {
    _private: (),
}

impl StarField3 {
    /// Uploads `stars` (sorted by sector) and builds the pass. `stars` is `World::star_data()`:
    /// AT-HYG followed by the archive-only exoplanet hosts, the slice that
    /// `Snapshot::suppressed_stars` and [`StarInput::suppressed`] index (not `WorldData::stars`,
    /// which `World::new` consumes and extends). `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        stars: &[Star3],
        input: SharedInput<StarInput>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, stars, input);
        todo!("B4")
    }

    /// Number of stars uploaded.
    pub fn count(&self) -> u32 {
        todo!("B4")
    }
}

impl RenderFeature for StarField3 {
    fn name(&self) -> &'static str {
        "stars"
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        _enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        _timer: &mut PassTimer,
    ) {
    }
}

/// Apparent V magnitude of a star of absolute magnitude `abs_mag` at `d_pc` parsecs:
/// M + 5 log₁₀ d − 5 (distance modulus). Twin of the WGSL in `stars3d.wgsl`.
pub fn star_apparent_mag(abs_mag: f32, d_pc: f32) -> f32 {
    let _ = (abs_mag, d_pc);
    todo!("B4")
}

/// Number of `stars` that rise above `psf::VISIBILITY_CUTOFF` on screen at the eye-adapted star
/// exposure `eye_star_ev100`, field of view `fov_y_rad` and `height_px` lines, seen from the SSB
/// at the catalogue epoch (the lead's star-EV sweep, ruling 9).
pub fn visible_star_count(
    stars: &[Star3],
    eye_star_ev100: f32,
    fov_y_rad: f32,
    height_px: u32,
) -> u32 {
    let _ = (stars, eye_star_ev100, fov_y_rad, height_px);
    todo!("B4")
}

/// The "Vivid star colours · non-physical" colour boost: `c` (linear Rec. 709) with its
/// saturation raised beyond Planck chromaticity, keeping its Rec. 709 luminance
/// (0.2126 R + 0.7152 G + 0.0722 B, ITU-R BT.709-6 item 3.2) so every photometric check stays
/// physical. The app applies it to star-coloured point sprites; the star shader applies its WGSL
/// twin in `stars3d.wgsl` while [`StarInput::vivid_colours`] is on.
///
/// B0 stub: the identity (no boost). B4 implements the boost and keeps the luminance.
pub fn vivid_color(c: Vec3) -> Vec3 {
    c
}
