//! The Milky Way band v0 and the environment cubemap, passes "band-update" and "band" (Task B5;
//! B0 stub).
//!
//! [`BandCubemap`] renders the galaxy model's band radiance (uni-procgen's WGSL twin
//! `procgen/galaxy.wgsl`) into a 6 × 256² Rgba16Float cubemap with 5 mips, in units of
//! `uni_gfx::ENV_UNIT_CD_M2`, clamped to 60000; [`BandDraw`] draws it behind everything (ruling
//! 4: after the opaque passes and the meter, at depth 0) × `ENV_UNIT_CD_M2 × star_pre()` ×
//! (1 − r)(1 − g)(1 − b). The lens pass samples the same cubemap ([`BandCubemap::env_map`]) for
//! its escaped rays. Directions are root (ICRS) axes, faces in the WebGPU convention
//! (`uni_gfx::env::cube_dir`).

use std::sync::Arc;

use glam::{DVec3, Vec2, Vec3};
use uni_gfx::probe::ProbeResult;
use uni_gfx::{
    EnvMap, FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};
use uni_procgen::galaxy::GalaxyModel;

/// Edge of each cubemap face [texels] (B5 brief).
pub const ENV_SIZE: u32 = 256;
/// Mip levels of the cubemap (256 → 16; B5 brief).
pub const ENV_MIPS: u32 = 5;
/// Texel format of the cubemap (B5 brief; `uni_gfx::EnvMap`).
pub const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// What the band passes need per frame (set by the app before `Renderer::render`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BandInput {
    /// Camera − SSB [pc], ICRS axes.
    pub camera_icrs_pc: DVec3,
    /// `false` skips both passes (the HUD's band toggle).
    pub enabled: bool,
    /// Faintest V magnitude drawn as individual stars; fainter starlight goes into the band
    /// (`uni_procgen::lf::faint_light_fraction`).
    pub v_lim: f32,
}

/// The environment cubemap and its update pass (`RenderFeature` named "band-update").
pub struct BandCubemap {
    env: Arc<EnvMap>,
}

impl BandCubemap {
    /// Creates the cubemap and the update pass for `model`.
    ///
    /// B0 stub: creates the cubemap (6 × [`ENV_SIZE`]², [`ENV_MIPS`] mips, [`ENV_FORMAT`]; wgpu
    /// zero-initialises it, so it reads as a black sky) with its cube view and a trilinear
    /// sampler, and draws nothing. The M0 lens pass takes it from [`BandCubemap::env_map`] (ruling
    /// A.4). B5 fills it.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        model: &GalaxyModel,
        input: SharedInput<BandInput>,
    ) -> anyhow::Result<Self> {
        let _ = (lib, model, input);
        let device = &gpu.device;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("band env cubemap"),
            size: wgpu::Extent3d {
                width: ENV_SIZE,
                height: ENV_SIZE,
                depth_or_array_layers: 6,
            },
            mip_level_count: ENV_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("band env cube view"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("band env sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the environment cubemap: {err}");
        }
        Ok(Self {
            env: Arc::new(EnvMap {
                texture,
                view,
                sampler,
            }),
        })
    }

    /// The cubemap, shared with [`BandDraw`] and the lens pass (`uni_rel::LensTracer::new`).
    pub fn env_map(&self) -> Arc<EnvMap> {
        Arc::clone(&self.env)
    }
}

impl RenderFeature for BandCubemap {
    fn name(&self) -> &'static str {
        "band-update"
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

/// The band draw pass (`RenderFeature` named "band").
pub struct BandDraw {
    _private: (),
}

impl BandDraw {
    /// Builds the pass over `env` ([`BandCubemap::env_map`]). `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        env: Arc<EnvMap>,
        input: SharedInput<BandInput>,
    ) -> anyhow::Result<Self> {
        let _ = (gpu, lib, camera_bgl, env, input);
        todo!("B5")
    }
}

impl RenderFeature for BandDraw {
    fn name(&self) -> &'static str {
        "band"
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

/// The band radiance the update pass writes at texel `uv` of cube face `face` for a camera at
/// `cam_icrs_pc` [pc, ICRS, origin SSB]: `model.band_radiance` in the direction
/// `uni_gfx::env::cube_dir(face, uv)` (root axes, converted to galactocentric), in cd/m², linear
/// Rec. 709 (not divided by `ENV_UNIT_CD_M2`). CPU twin of `band_update.wgsl` (B5).
pub fn band_texel(
    model: &GalaxyModel,
    cam_icrs_pc: DVec3,
    face: u32,
    uv: Vec2,
    v_lim: f32,
) -> Vec3 {
    let _ = (model, cam_icrs_pc, face, uv, v_lim);
    todo!("B5")
}

/// B5's probes (`gal_band_radiance` GPU vs CPU, `cube_dir` write-then-sample), called from
/// [`crate::gpu_probes`]. B0 stub: none.
pub(crate) fn band_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    let _ = (gpu, lib);
    Vec::new()
}
