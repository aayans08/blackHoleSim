//! Saturn's rings, pass "rings" (Task B2; B0 stub).
//!
//! An annulus mesh in the ring plane (256 segments, exact plane depth), depth-tested without
//! depth write, premultiplied, `lens_mask.g` = opacity with `Max`. Lit face: reflectance ×
//! two-lobe Henyey–Greenstein; unlit face: diffuse transmission. Sunlight ×
//! [`planet_shadow_on_ring`] (penumbra by [`crate::sun_visible_fraction`]). The radii come from
//! the profile (A8's span; uni-sim's `RingView` uses the same). The app pushes this pass only
//! when `WorldData::rings` loaded (ruling A.3); without it the load warning is shown.

use glam::DVec3;
use uni_data::RingProfile;
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

use crate::body::BodyScene;

/// The ring pass (`RenderFeature` named "rings"). It draws the rings of `BodyScene::bodies`
/// whose view has a `ring`.
pub struct RingRenderer {
    _private: (),
}

impl RingRenderer {
    /// Uploads `profile` (radial reflectance and opacity) and builds the pass. `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        profile: &RingProfile,
        input: SharedInput<BodyScene>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, profile, input);
        todo!("B2")
    }
}

impl RenderFeature for RingRenderer {
    fn name(&self) -> &'static str {
        "rings"
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

/// Ring phase function for scattering angle cosine `cos_alpha`, lit or unlit face, normalised
/// over the sphere (∫ dΩ/4π = 1). Twin of the WGSL in `rings.wgsl`.
pub fn ring_phase(cos_alpha: f32, lit_face: bool) -> f32 {
    let _ = (cos_alpha, lit_face);
    todo!("B2")
}

/// Fraction of the Sun visible from ring point `p_ring` (planet-centred, body axes [m]) past the
/// planet's ellipsoid of `radii` [m], for the unit direction `sun_dir`: 0 in the umbra, 1 in full
/// sunlight. Twin of the WGSL in `rings.wgsl`.
pub fn planet_shadow_on_ring(p_ring: DVec3, sun_dir: DVec3, radii: DVec3) -> f32 {
    let _ = (p_ring, sun_dir, radii);
    todo!("B2")
}

/// B2's ring probes (`planet_shadow_on_ring`, `ring_phase`), called from [`crate::gpu_probes`].
/// B0 stub: none.
pub(crate) fn ring_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = (gpu, lib);
    Vec::new()
}
