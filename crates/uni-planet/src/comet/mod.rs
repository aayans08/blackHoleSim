//! Comets, pass "comets" (Task C1; Task C0 wrote this stub, its constants and signatures).
//!
//! Draws every comet of `uni_sim::snapshot::Snapshot::comets` (the app copies them into
//! [`CometScene`]): a resolved nucleus as a real mesh ([`nucleus`]; opaque, depth write,
//! `lens_mask.g` = coverage; between `POINT_LOD_PX` and `BLEND_LOD_PX` a world writer in the LOD
//! band, exposure `pre_mix` with `body::exposure_weight(proj_radius_px, eye_adapted)` and
//! `lens_mask.a` = its star share, as a body); the coma of every active comet ([`coma`]); and the
//! dust and ion tails of at most [`MAX_TAILED_COMETS`] of them ([`tails`]), recomputed at most
//! every [`TAIL_REFRESH_WALL_S`] of wall time. World light: luminance × `scene_pre()` (frame
//! contract). A nucleus below `POINT_LOD_PX` is uni-sim's point, never drawn here.
//!
//! Sub-passes need their own timer labels; C6 adds them to `uni-verify`'s feature budgets.

pub mod coma;
pub mod nucleus;
pub mod tails;

use uni_core::Epoch;
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};
use uni_sim::snapshot::CometView;

/// Tails render for at most this many comets on screen, the most prominent (brightest predicted
/// total V from the camera) (ruling round 2, review I13).
pub const MAX_TAILED_COMETS: usize = 8;
/// Under time warp a comet's tail geometry is recomputed at most this often [s of wall time]
/// (ruling round 2, review I13).
pub const TAIL_REFRESH_WALL_S: f64 = 0.25;

/// What the comet pass draws this frame (a `SharedInput` the app sets before `render`).
#[derive(Clone, Debug)]
pub struct CometScene {
    /// `Snapshot::comets`: brightest first.
    pub comets: Vec<CometView>,
    /// The frame's epoch (tails are emitted up to it).
    pub epoch: Epoch,
    /// "Eye-adapted stars" is in effect (`ExposureSettings::eye_adapted_in_effect()`), as
    /// `BodyScene::eye_adapted`: a nucleus in the LOD band blends toward star exposure.
    pub eye_adapted: bool,
}

/// `uni_core::Epoch` has no `Default`, so this is by hand: no comets, at J2000.
impl Default for CometScene {
    fn default() -> Self {
        CometScene {
            comets: Vec::new(),
            epoch: Epoch::J2000,
            eye_adapted: false,
        }
    }
}

/// The comet pass (`RenderFeature` named "comets"). The app builds it only once C1 has landed
/// (lead rule: no stubbed renderer is constructed).
pub struct CometRenderer {
    _private: (),
}

impl CometRenderer {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<CometScene>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, input);
        todo!("C1")
    }

    /// The last frame's tail CPU time [ms] (`--bench` reports it as `comets_cpu_ms`).
    pub fn cpu_ms(&self) -> f32 {
        todo!("C1")
    }
}

impl RenderFeature for CometRenderer {
    fn name(&self) -> &'static str {
        "comets"
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

/// C1's probes (`coma_surface_brightness`: 64 radii, ±1e-3), called from
/// [`crate::gpu_probes`]. Task C0 stub: none (the lead's selftest runs it before C1 lands).
pub(crate) fn comet_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = (gpu, lib);
    Vec::new()
}
