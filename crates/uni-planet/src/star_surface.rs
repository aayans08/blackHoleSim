//! Resolved star surfaces (the Sun, near catalogue stars), pass "sun" (Task B2; B0 stub).
//!
//! Limb darkening (Neckel & Labs 1994, Solar Physics 153, 91) normalised so the disk-integrated
//! illuminance equals the view's `luminous_intensity_cd / d²`, animated granulation (Nordlund,
//! Stein & Asplund 2009, Living Rev. Solar Phys. 6, 2) hashed with `pcg3d` on integer-lattice
//! cube-face coordinates, colour `blackbody_linear_srgb(teff_k)`. A camera inside a photosphere
//! draws nothing and [`StarSurfaceRenderer::warnings`] reports "Inside <name>" (review C2). Uses
//! [`crate::cap_mesh`] and [`crate::lod_exposure_mix`]; nothing is sized per `ObjectId` at
//! construction (`World::objects()` grows at runtime).

use glam::Vec3;
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

use crate::body::BodyScene;

/// The star-surface pass (`RenderFeature` named "sun"). It draws `BodyScene::stars`.
pub struct StarSurfaceRenderer {
    _private: (),
}

impl StarSurfaceRenderer {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<BodyScene>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, input);
        todo!("B2")
    }

    /// Warnings for the HUD from the last prepared frame ("Inside Sun", review C2).
    pub fn warnings(&self) -> Vec<String> {
        todo!("B2")
    }
}

impl RenderFeature for StarSurfaceRenderer {
    fn name(&self) -> &'static str {
        "sun"
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

/// Limb darkening I(μ)/I(1) at (650, 550, 450) nm (Neckel & Labs 1994 fifth-order
/// polynomials). Twin of the WGSL in `star_surface.wgsl`.
pub fn limb_darkening(mu: f32) -> Vec3 {
    let _ = mu;
    todo!("B2")
}

/// 2∫₀¹ I(μ)/I(1) μ dμ per channel, the disk-averaged limb darkening used to normalise the
/// surface luminance to the star's flux.
pub fn limb_darkening_mean() -> Vec3 {
    todo!("B2")
}

/// Multiplicative granulation intensity (mean 1) at the unit-sphere point `p_unit` of a star of
/// `radius_m` with cells of `cell_m`, at time `t_s` [s]. Twin of the WGSL in
/// `star_surface.wgsl`.
pub fn granulation(p_unit: Vec3, cell_m: f32, radius_m: f32, t_s: f32) -> f32 {
    let _ = (p_unit, cell_m, radius_m, t_s);
    todo!("B2")
}

/// B2's probes (`limb_darkening`, `granulation`), called from [`crate::gpu_probes`]. B0 stub: none.
pub(crate) fn star_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = (gpu, lib);
    Vec::new()
}
