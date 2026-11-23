//! The atmosphere render feature, pass "atmosphere" (Task B3; B0 stub).
//!
//! Pass 8 of the M1 frame contract: after the opaque passes, before the meter. Per pixel it
//! intersects each atmosphere's shell, clips by the scene depth and the planet's ground, and
//! marches the inscattered light (world light × `scene_pre()`), writing `lens_mask.g =
//! 1 − luminance(T)` with `Max`. Two variants: dual-source blending (two pipelines,
//! `atmosphere_dual.wgsl` + `atmosphere_mask.wgsl`) or grey transmittance (one pipeline,
//! `atmosphere_grey.wgsl`).

use glam::{DQuat, DVec3, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

use crate::lut::AtmosphereSet;
use crate::params::AtmosphereParams;

/// The atmospheres to draw this frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug, Default)]
pub struct AtmoScene {
    pub atmos: Vec<AtmoDraw>,
}

/// One atmosphere to draw.
#[derive(Clone, Debug)]
pub struct AtmoDraw {
    /// NAIF id of the body; selects its preset and LUTs.
    pub naif: i32,
    /// Body centre − camera [m], root (ICRS) axes.
    pub rel_pos_m: DVec3,
    /// Body-fixed → ICRS rotation.
    pub orient: DQuat,
    /// Sun − body [m], root axes.
    pub sun_rel_m: DVec3,
    /// Solar illuminance at the body [lx].
    pub sun_illuminance_lux: f64,
}

/// The atmosphere pass (`RenderFeature` named "atmosphere").
pub struct AtmosphereFeature {
    _private: (),
}

impl AtmosphereFeature {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `set` holds the
    /// LUTs; `dual_source` is `GpuContext::has_dual_source_blending()` and picks the variant.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        set: &AtmosphereSet,
        input: SharedInput<AtmoScene>,
        dual_source: bool,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, set, input, dual_source);
        todo!("B3")
    }
}

impl RenderFeature for AtmosphereFeature {
    fn name(&self) -> &'static str {
        "atmosphere"
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

/// The light a ray picks up through the atmosphere: from `ro_m` (camera − body centre [m]) along
/// the unit direction `rd` for at most `t_max_m` metres (the scene depth), with the Sun in the
/// unit direction `sun_dir` (all in the same axes). Returns (inscattered radiance per lux of sun
/// illuminance [cd/m² per lx], transmittance along the segment), per channel. CPU twin of the
/// view march in `atmosphere_common.wgsl` (B3).
pub fn segment_radiance(
    params: &AtmosphereParams,
    ro_m: DVec3,
    rd: DVec3,
    t_max_m: f64,
    sun_dir: DVec3,
) -> (Vec3, Vec3) {
    let _ = (params, ro_m, rd, t_max_m, sun_dir);
    todo!("B3")
}
