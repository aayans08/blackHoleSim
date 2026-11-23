//! Bloom, pass "bloom" (Task B6; B0 stub).
//!
//! Jimenez 2014 (SIGGRAPH, "Next Generation Post Processing in Call of Duty: Advanced Warfare"):
//! a 6-level 13-tap downsample with a Karis average on the first level, a 3 × 3 tent upsample,
//! hdr = mix(hdr, bloom, strength), no threshold. The first downsample clamps its input to
//! [`BLOOM_INPUT_MAX`].

use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

/// Default bloom strength, the mix weight of the bloom chain (plan B6; an aesthetic choice of a
/// camera's or eye's faint scatter, not a measured value).
pub const DEFAULT_BLOOM_STRENGTH: f32 = 0.04;
/// The first downsample clamps its pre-exposed input to this: 2 stops over AgX white
/// (4 × 16.29 ≈ 65; uni-post `agx`), so the Sun's saturated glare does not wash out the faint
/// highlighted asteroid belt (plan B6's arithmetic).
pub const BLOOM_INPUT_MAX: f32 = 64.0;

/// The bloom pass (`RenderFeature` named "bloom").
pub struct Bloom {
    _private: (),
}

impl Bloom {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `strength` is the
    /// mix weight, set by the app (HUD), [`DEFAULT_BLOOM_STRENGTH`] by default.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        strength: SharedInput<f32>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, strength);
        todo!("B6")
    }
}

impl RenderFeature for Bloom {
    fn name(&self) -> &'static str {
        "bloom"
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
