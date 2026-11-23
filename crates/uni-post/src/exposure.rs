//! Auto-exposure and the dual star exposure: the resolve pass "exposure" and its settings (Task
//! B6; B0 stub).
//!
//! Pass 1 of the M1 frame contract: scene EV = adapted (Auto) or `manual.scene_ev100` (Manual);
//! star EV = `eye_star_ev100` while [`ExposureSettings::eye_adapted_in_effect`], else the scene
//! EV; written into the renderer's exposure buffer through a storage binding in the pass's own
//! bind group, importing only `common/exposure.wgsl`, never group 0 (review m14). The meter is
//! [`crate::histogram`].

use uni_gfx::{
    Exposure, ExposureUniform, FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};

use crate::histogram::ExposureMeter;

/// How the scene EV is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExposureMode {
    /// Metered (B6's rule) and adapted over time.
    Auto,
    /// [`ExposureSettings::manual`] (the HUD sliders, bookmarks, and the display-referred M0 debug
    /// grid).
    Manual,
}

/// The exposure controls the app sets every frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExposureSettings {
    pub mode: ExposureMode,
    /// The one manual source (review m13): the scene EV in Manual mode, and the app copies it
    /// into `FrameCtx::exposure` too.
    pub manual: Exposure,
    /// Added to the metered target [EV].
    pub compensation_ev: f32,
    /// The "Eye-adapted stars" option: stars and point sources at [`Self::eye_star_ev100`].
    pub eye_adapted: bool,
    /// Star EV100 while eye-adapted stars are in effect.
    pub eye_star_ev100: f32,
    /// Set by the app from `uni_atmo::camera_in_daylight`: a daylit sky hides the stars, so the
    /// eye-adapted exposure is off (review m12).
    pub daylight: bool,
    /// Adaptation time constants [s]: brightening, darkening.
    pub adapt_tau_s: [f32; 2],
    /// Jump straight to the target (τ = 0), for captures.
    pub settle: bool,
}

impl ExposureSettings {
    /// True when the eye-adapted star exposure is in effect: `eye_adapted && !daylight` (M1 frame
    /// contract). It drives the resolve pass and the HUD's amber "Eye-adapted stars" label
    /// (review m13), not `FrameCtx::exposure`.
    pub fn eye_adapted_in_effect(&self) -> bool {
        self.eye_adapted && !self.daylight
    }
}

/// Dark-adapted target with no metered world light [EV100]. Equals
/// `uni_sky::DEFAULT_STAR_EV100` (−4, the M0 gate's star sweep; checked by an app test, B9).
pub const NIGHT_EV100: f32 = -4.0;
/// Lowest exposure the resolve writes [EV100]; equals the start of `uni_app::scene::EV100_LIMITS`
/// (review m5: 2^EV never overflows; checked by an app test, B9).
pub const EV100_MIN: f32 = -30.0;
/// Highest exposure the resolve writes [EV100]; equals the end of `uni_app::scene::EV100_LIMITS`.
pub const EV100_MAX: f32 = 40.0;

/// One adaptation step: ev_prev + (ev_target − ev_prev)(1 − e^(−dt/τ)), with τ = `tau_s[0]`
/// when brightening the image (the EV falls) and `tau_s[1]` when darkening it (B6).
pub fn adapt(ev_prev: f32, ev_target: f32, dt_s: f32, tau_s: [f32; 2]) -> f32 {
    let _ = (ev_prev, ev_target, dt_s, tau_s);
    todo!("B6")
}

/// The exposure uniform the resolve pass writes for `s`, with the adapted scene EV (Auto) or
/// `None` (Manual, or no meter reading yet). CPU twin of `exposure.wgsl` (B6).
pub fn resolve_exposure(s: &ExposureSettings, adapted_scene_ev: Option<f32>) -> ExposureUniform {
    let _ = (s, adapted_scene_ev);
    todo!("B6")
}

/// The resolve pass (`RenderFeature` named "exposure"), made by [`auto_exposure`].
pub struct ExposureResolve {
    _private: (),
}

impl RenderFeature for ExposureResolve {
    fn name(&self) -> &'static str {
        "exposure"
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

/// The CPU's view of the exposures the GPU resolved, read back through a ring of mapped buffers
/// (like `PassTimer`); clones share it.
#[derive(Clone)]
pub struct ExposureReadback {
    _private: (),
}

impl ExposureReadback {
    /// The latest read-back [scene, star] EV100, at least one frame old; `None` before the first.
    pub fn last_ev100(&self) -> Option<[f32; 2]> {
        todo!("B6")
    }
}

/// Builds the auto-exposure passes: the resolve (push first) and the meter (push after the
/// atmosphere pass), and the read-back. `exposure_buffer` is `Renderer::exposure_buffer()`;
/// `camera_bgl` is `Renderer::camera_bind_group_layout()` (the meter binds group 0; the resolve
/// never does).
pub fn auto_exposure(
    gpu: &GpuContext,
    lib: &mut ShaderLib,
    camera_bgl: &wgpu::BindGroupLayout,
    exposure_buffer: &wgpu::Buffer,
    settings: SharedInput<ExposureSettings>,
) -> anyhow::Result<(ExposureResolve, ExposureMeter, ExposureReadback)> {
    let _ = (gpu, lib, camera_bgl, exposure_buffer, settings);
    todo!("B6")
}
