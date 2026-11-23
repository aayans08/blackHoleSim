//! Unresolved bodies, the Sun and near stars drawn as points, pass "points" (Task B4; B0 stub).
//!
//! The same PSF as the star pass, from an instance buffer rebuilt per frame from the app's
//! `Vec<PointSprite>`, written × `star_pre()` × (1 − r) (review m6), with vertex depth from the
//! true distance and compare `Greater`, so bodies in front hide them.

use glam::{DVec3, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

/// One point source (from uni-sim's `PointView`).
#[derive(Clone, Copy, Debug)]
pub struct PointSprite {
    /// Point − camera [m], ICRS axes, f64.
    pub rel_pos_m: DVec3,
    /// Illuminance at the camera [lx].
    pub illuminance_lux: f64,
    /// Linear Rec. 709 colour with luminance 1. Reflected light is the solar colour, tinted by
    /// `uni_planet::BodyRenderer::mean_albedo_rgb` where the body has a map (carry-over 6);
    /// star colours pass through [`crate::vivid_color`] while vivid colours are on.
    pub color: Vec3,
}

/// The point-sprite pass (`RenderFeature` named "points").
pub struct PointSprites {
    _private: (),
}

impl PointSprites {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<Vec<PointSprite>>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, input);
        todo!("B4")
    }
}

impl RenderFeature for PointSprites {
    fn name(&self) -> &'static str {
        "points"
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
