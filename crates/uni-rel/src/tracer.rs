//! Lens tracer render feature: half-resolution per-pixel geodesic trace on the GPU (M0 Task 9,
//! M1 Task B7).
//!
//! For the first lens in [`FrameCtx::lenses`] (only one in M1), [`LensTracer`] works out the
//! screen rectangle where light is bent by at least half a pixel, traces one Binet geodesic
//! per sample there (at the resolution scale) and upsamples the result bilinearly into `hdr`
//! and `lens_mask`. It writes nothing outside the rectangle (uni-gfx `targets` pass contract).
//! The shaders are `shaders/lens_trace.wgsl` (passes), `binet.wgsl` (integrator) and
//! `debug_sky.wgsl`.
//!
//! **Environment** ([`LensEnv`], read from its [`SharedInput`] in `prepare`):
//! - [`LensEnv::Band`]: escaped rays sample the environment cubemap ([`EnvMap`], the Milky Way
//!   band) at their exit direction, × `ENV_UNIT_CD_M2 × star_pre()` (sky light at star
//!   exposure), at a mip level from the finite-difference spread of neighbouring samples' exit
//!   directions (physics research §1.4; [`env_lod`] is the CPU twin of `lens_env_lod`). The band
//!   pass draws the same cubemap around the rectangle and skips the pixels with `lens_mask.b = 1`.
//! - [`LensEnv::DebugGrid`]: the M0 debug grid, exactly as at m0.1, with
//!   [`crate::DebugSkyBackground`] drawing it around the rectangle (ruling 14).
//!
//! **Mask** (M1 frame contract): traced pixels get `lens_mask = (captured fraction, 0, 1, 0)`;
//! captured light is black. From inside the horizon the whole frame is drawn black: a labelled
//! placeholder, not physics ([`LensCoverage::InsideHorizon`]). **Foreground bodies are not lensed** (design §5.3, review m7): the
//! resolve reads the depth texture and leaves every pixel with depth ≠ 0 untouched, colour and
//! mask, also from inside a horizon.
//!
//! The brute-force step budget only reaches the hole from cameras within [`TRACE_REACH`]
//! (5000 M). Farther out, nothing is traced and the frame shows the unlensed sky with no
//! shadow; [`lens_coverage`] reports this as [`LensCoverage::BeyondReach`], and the app must
//! put [`LensCoverage::warning`] on the HUD (Global Constraints, Honesty).
//!
//! The lens is treated as a static Schwarzschild hole seen by a static camera; `LensView::spin`
//! is ignored in M0 (Kerr comes in M3, design §5.5).
//!
//! Order (M1 pass 10): after the opaque passes, the atmosphere and the meter; before the band
//! (× (1 − r)(1 − g)(1 − b)), the stars (× (1 − r)(1 − g)) and the tonemap. When nothing is
//! traced and the camera is outside every horizon, the pass encodes nothing at all.

use std::f64::consts::FRAC_PI_2;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Context as _;
use glam::{DVec3, Mat4, UVec2, Vec3};
use uni_gfx::{
    EnvMap, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT,
    LensView, PassTimer, RenderFeature, ShaderLib, SharedInput,
};

use crate::{LensEnv, binet};

/// Most RK4 steps per GPU ray (the brief's budget).
const GPU_MAX_STEPS: u32 = 2000;
/// Smallest GPU escape radius [M]; a ray escapes beyond max(2·r_cam, this), moving outwards.
const GPU_MIN_R_FAR: f64 = 1000.0;

/// Farthest camera distance from the hole [M] at which the lens is traced.
///
/// The step rule caps dλ at 5 M (`binet::step_size`), so a GPU ray covers at most about
/// 2000 × 5 = 1e4 M. A ray aimed at the hole needs ≈ (r_cam − 250)/5 steps to reach r = 250 M,
/// where the cap stops applying, and about 330 more to be captured. These numbers are measured
/// with the CPU twin; with them:
/// - captures fail from about 8600 M on (a ray inside the shadow ends short of the hole);
/// - from about 7500 M, rays that graze the photon sphere run out before they leave it.
///
/// At 5000 M, rays aimed at the hole are captured within about 1300 steps. Every other ray has
/// passed periapsis before the budget ends, and its exit direction is within 2e-5 rad (about
/// 0.02 px at 1080p and 60°) of a long-budget trace. `gpu_budget_is_exact_within_trace_reach`
/// and `gpu_budget_fails_beyond_trace_reach` pin both sides.
pub const TRACE_REACH: f64 = 5000.0;

/// HUD line for [`LensCoverage::BeyondReach`] (Global Constraints, Honesty).
pub const BEYOND_REACH_WARNING: &str = "Lens not traced (camera > 5000 M): unlensed, no shadow";

/// Interactive default resolution scale; captures set 1.0.
pub const DEFAULT_RESOLUTION_SCALE: f32 = 0.5;
const MIN_RESOLUTION_SCALE: f32 = 0.25;
const MAX_RESOLUTION_SCALE: f32 = 1.0;

/// A shared handle to the lens tracer's resolution scale (the fraction of the internal
/// resolution at which rays are traced), clamped to 0.25..=1.0.
///
/// The `Renderer` owns its features, so the app keeps a clone of this handle to steer the
/// tracer (and the debug background, whose unlensed lines match the traced width). `set` takes
/// effect from the next frame's `prepare`.
#[derive(Clone, Debug)]
pub struct LensScale(Arc<AtomicU32>);

impl LensScale {
    /// A handle at `s`, clamped to 0.25..=1.0 (NaN gives the interactive default 0.5).
    pub fn new(s: f32) -> Self {
        let s = clamp_resolution_scale(s, DEFAULT_RESOLUTION_SCALE);
        Self(Arc::new(AtomicU32::new(s.to_bits())))
    }

    /// Sets the scale, clamped to 0.25..=1.0; NaN is ignored.
    pub fn set(&self, s: f32) {
        let s = clamp_resolution_scale(s, self.get());
        self.0.store(s.to_bits(), Ordering::Relaxed);
    }

    /// The current scale, in 0.25..=1.0.
    pub fn get(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}
/// Ray samples: exit direction (xyz) and capture flag (w). f32, because f16's ~5e-4 rad
/// direction error is half a pixel at 1080p with a 60° field of view.
const SAMPLES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
/// Workgroup edge of both entry points (`@workgroup_size(8, 8, 1)`).
const WORKGROUP: u32 = 8;

/// GPU twin of `struct LensParams` in `shaders/lens_trace.wgsl` (vec4 members only;
/// `lens_params_layout_matches_wgsl` checks the layout with naga).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct LensParams {
    /// xyz: camera position relative to the hole [M], root axes; w: r_cam [M].
    cam: [f32; 4],
    /// Traced rectangle in full-resolution pixels: min x, min y, width, height (the whole
    /// frame when the camera is inside the horizon).
    rect: [u32; 4],
    /// Samples across the rectangle (x, y; 0 = nothing traced), max steps,
    /// camera inside the horizon (1 = the whole frame is black).
    grid: [u32; 4],
    /// Full-resolution pixels per sample (x, y), r_far [M], 1 / resolution scale.
    scale: [f32; 4],
}

impl LensParams {
    /// Whether the pass writes anything: rays to trace, or a frame to black out.
    fn writes_pixels(&self) -> bool {
        let traced = self.grid[0] > 0 && self.grid[1] > 0;
        let inside = self.grid[3] != 0;
        (traced || inside) && self.rect[2] > 0 && self.rect[3] > 0
    }
}

/// `LensEnvParams.mode.x` for [`LensEnv::DebugGrid`] (twin of `LENS_ENV_DEBUG_GRID`).
const LENS_ENV_DEBUG_GRID: u32 = 0;
/// `LensEnvParams.mode.x` for [`LensEnv::Band`] (twin of `LENS_ENV_BAND`).
const LENS_ENV_BAND: u32 = 1;

/// GPU twin of `struct LensEnvParams` in `shaders/lens_trace.wgsl` (one vec4;
/// `lens_env_params_layout_matches_wgsl` checks the layout with naga). [`LensParams`] keeps
/// m0.1's layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, bytemuck::Pod, bytemuck::Zeroable)]
struct LensEnvParams {
    /// x: the environment escaped rays show ([`LENS_ENV_DEBUG_GRID`] or [`LENS_ENV_BAND`]);
    /// yzw: 0.
    mode: [u32; 4],
}

impl LensEnvParams {
    fn new(env: LensEnv) -> Self {
        let code = match env {
            LensEnv::DebugGrid => LENS_ENV_DEBUG_GRID,
            LensEnv::Band => LENS_ENV_BAND,
        };
        Self {
            mode: [code, 0, 0, 0],
        }
    }
}

/// Mip level of a cube with `face_texels` texels per face edge and `levels` mips, for a sample
/// whose footprint on the sky is `footprint` [rad] in direction `dir` (any length). CPU twin of
/// `lens_env_lod` in `shaders/lens_trace.wgsl` (`universe --selftest` probe `lens_env`).
///
/// A cube face is the gnomonic projection of the sky onto the plane at distance 1, which
/// stretches angles radially by 1/m² (m = the unit direction's major-axis component), so one
/// texel spans 2m²/N rad there (2/N at a face centre, a third of that toward a corner). The level
/// is log2(footprint / texel angle), clamped to [0, levels − 1]: the level a GPU selects from the
/// larger screen-space derivative of the face coordinates (Vulkan specification, "Scale Factor
/// Operation, LOD Operation and Image Level(s) Selection"). The footprint is the spread of
/// neighbouring samples' exit directions, the finite-difference Jacobian of physics research
/// §1.4 (the cheap substitute for the ray bundles of James et al. 2015, DNGR).
pub(crate) fn env_lod(dir: Vec3, footprint: f32, face_texels: f32, levels: f32) -> f32 {
    let m = dir.normalize().abs().max_element();
    let texel = 2.0 * m * m / face_texels;
    let lod = (footprint.max(1e-12) / texel).log2();
    lod.min(levels - 1.0).max(0.0)
}

/// A rectangle of full-resolution pixels, `[min, min + size)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PixelRect {
    min: UVec2,
    size: UVec2,
}

/// What the lens pass draws this frame. [`lens_coverage`] and [`LensTracer`] use the same plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LensCoverage {
    /// No lens, or its light bending is under half a pixel everywhere on screen. The pass
    /// draws the unlensed sky, which is then correct to half a pixel.
    Unlensed,
    /// Geodesics are traced on the GPU over the lens rectangle.
    Traced,
    /// The camera is on or inside the horizon, and the frame is drawn black
    /// ([`inside_horizon`]). That is a placeholder, not a simulated view: an observer inside
    /// r = 2M still receives light from outside (it falls in behind them), but no static
    /// observer exists there and the tracer models only static ones (`static_observer_ray`).
    /// The app labels it "Inside the event horizon: view not simulated" (uni-ui's
    /// `INSIDE_HORIZON_LABEL`; final review render I3); M3 adds the infalling view.
    InsideHorizon,
    /// The lens bends light by at least half a pixel on screen, but the camera is farther than
    /// [`TRACE_REACH`] from it, so the step budget cannot reach the hole. Nothing is traced:
    /// the frame shows the unlensed sky with no shadow and `lens_mask` stays 0. The app must
    /// label this on screen ([`LensCoverage::warning`]).
    BeyondReach,
}

impl LensCoverage {
    /// The HUD warning for this coverage, if it needs one. The app adds it to the HUD's
    /// warnings. `InsideHorizon` has its own amber HUD label (uni-ui's `INSIDE_HORIZON_LABEL`),
    /// so it returns `None` here.
    pub fn warning(self) -> Option<&'static str> {
        match self {
            Self::BeyondReach => Some(BEYOND_REACH_WARNING),
            Self::Unlensed | Self::Traced | Self::InsideHorizon => None,
        }
    }
}

/// What the lens pass draws for `ctx`. This is the same plan [`LensTracer`] makes in
/// `prepare`, sized by `ctx.viewport`, so the app can label the frame without access to the
/// feature. `LensTracer` plans with `FrameTargets::size`, which the app makes equal to
/// `ctx.viewport`.
pub fn lens_coverage(ctx: &FrameCtx) -> LensCoverage {
    frame_params(
        ctx.lenses.first(),
        ctx.view_rot,
        ctx.proj,
        ctx.viewport,
        MAX_RESOLUTION_SCALE,
    )
    .0
}

/// True when the camera is on or inside the lens's horizon (r < 2(1 + ε) M, the capture
/// radius of [`binet::trace`]), including at its centre. The lens pass then draws the whole
/// frame black as a placeholder (no static observer exists for r ≤ 2M, so nothing there is
/// simulated; [`LensCoverage::InsideHorizon`]); the app shows the amber label "Inside the event
/// horizon: view not simulated (no static observer inside r = 2M; an infalling view comes in
/// M3)" (uni-ui's `INSIDE_HORIZON_LABEL`; Review Focus 2; final review render I3).
pub fn inside_horizon(lens: &LensView) -> bool {
    camera_in_m(lens).is_some_and(|c| c.length() < binet::CAPTURE_RADIUS)
}

/// Camera position relative to the hole in units of M = GM/c², or `None` if the lens has no
/// positive, finite mass.
fn camera_in_m(lens: &LensView) -> Option<DVec3> {
    let m = uni_core::gravitational_radius(lens.mass_kg);
    (m > 0.0 && m.is_finite()).then(|| -lens.rel_pos_m / m)
}

/// Half-angle [rad] of the cone around the hole's direction inside which light is deflected by
/// at least half a pixel, for a static camera at `r_cam` [M] and pixels `pixel_angle` [rad]
/// across. The weak-field deflection α = 2r_s/b = 4M/b (Einstein 1915) falls to half a pixel
/// at b_max = 8M/pixel_angle, and a static observer sees impact parameter b at
/// sin δ = b·√(1 − 2/r)/r (CLAUDE.md §6). π/2 once b_max reaches every forward direction.
fn cone_half_angle(r_cam: f64, pixel_angle: f64) -> f64 {
    let b_max = 8.0 / pixel_angle;
    let s = b_max * (1.0 - 2.0 / r_cam).sqrt() / r_cam;
    if s >= 1.0 { FRAC_PI_2 } else { s.asin() }
}

/// Slopes c/w (low, high) of the two lines through the origin tangent to the circle of radius
/// `radius` centred at (c, w), where w > radius (the circle lies in front).
fn tangent_slopes(c: f64, w: f64, radius: f64) -> (f64, f64) {
    let centre = c.atan2(w);
    let half = (radius / c.hypot(w)).asin();
    ((centre - half).tan(), (centre + half).tan())
}

/// Full-resolution pixels covering the cone of half-angle `theta` around the unit camera-space
/// direction `axis` (camera looks along −Z, +Y up), padded by one pixel and clamped to the
/// screen. `proj_x`, `proj_y` are the projection's x and y scales (f/aspect and f).
///
/// The cone is the tangent cone of the sphere of radius sin θ centred at `axis`. Its projected
/// x range is the range of x/w (w = −z) over that sphere, i.e. over the sphere's disk in the
/// x–w plane: the slopes of the two tangent lines from the eye. Likewise for y. A cone that
/// reaches the camera plane projects to an unbounded region (full screen); one entirely
/// behind it projects to nothing.
fn cone_rect(axis: DVec3, theta: f64, proj_x: f64, proj_y: f64, size: UVec2) -> PixelRect {
    let full = PixelRect {
        min: UVec2::ZERO,
        size,
    };
    let empty = PixelRect {
        min: UVec2::ZERO,
        size: UVec2::ZERO,
    };
    if theta >= FRAC_PI_2 {
        return full;
    }
    let radius = theta.sin();
    let ahead = -axis.z; // cosine of the angle between the axis and the view direction
    if ahead <= -radius {
        return empty;
    }
    if ahead <= radius {
        return full;
    }
    let (x_lo, x_hi) = tangent_slopes(axis.x, ahead, radius);
    let (y_lo, y_hi) = tangent_slopes(axis.y, ahead, radius);
    // pixel_dir's convention: x = (ndc_x + 1)/2·w; row = (1 − ndc_y)/2·h, row 0 at the top.
    let (w, h) = (f64::from(size.x), f64::from(size.y));
    let left = ((x_lo * proj_x + 1.0) * 0.5 * w).floor() - 1.0;
    let right = ((x_hi * proj_x + 1.0) * 0.5 * w).ceil() + 1.0;
    let top = ((1.0 - y_hi * proj_y) * 0.5 * h).floor() - 1.0;
    let bottom = ((1.0 - y_lo * proj_y) * 0.5 * h).ceil() + 1.0;
    let (left, right) = (left.clamp(0.0, w), right.clamp(0.0, w));
    let (top, bottom) = (top.clamp(0.0, h), bottom.clamp(0.0, h));
    if right <= left || bottom <= top {
        return empty;
    }
    PixelRect {
        min: UVec2::new(left as u32, top as u32),
        size: UVec2::new((right - left) as u32, (bottom - top) as u32),
    }
}

/// Samples needed to cover `rect_size` pixels at resolution `scale`: ⌈size·scale⌉, at least 1.
fn sample_grid(rect_size: UVec2, scale: f32) -> UVec2 {
    (rect_size.as_vec2() * scale)
        .ceil()
        .as_uvec2()
        .max(UVec2::ONE)
}

/// `s` clamped to 0.25..=1.0; a NaN keeps `current`.
fn clamp_resolution_scale(s: f32, current: f32) -> f32 {
    if s.is_nan() {
        current
    } else {
        s.clamp(MIN_RESOLUTION_SCALE, MAX_RESOLUTION_SCALE)
    }
}

/// GPU escape radius [M] for a camera at `r_cam` [M]: max(2·r_cam, 1000).
fn gpu_r_far(r_cam: f64) -> f64 {
    (2.0 * r_cam).max(GPU_MIN_R_FAR)
}

/// This frame's plan for `lens` (the first lens, if any) on a `size` frame: what the pass
/// draws, and its GPU parameters.
///
/// - [`LensCoverage::Unlensed`]: there is no lens (or its mass is not positive and finite),
///   its cone is under half a pixel, or its rectangle is off screen. Nothing is traced, and the
///   resolve pass draws only the unlensed sky.
/// - [`LensCoverage::InsideHorizon`]: sets the black-frame flag, with the whole frame as the
///   rectangle (a placeholder frame, not a simulated view).
/// - [`LensCoverage::BeyondReach`]: the rectangle is on screen but r_cam > [`TRACE_REACH`].
///   Nothing is traced, as for `Unlensed`.
/// - [`LensCoverage::Traced`]: the rectangle is traced.
///
/// `cam` is written only when tracing (r_cam ≤ [`TRACE_REACH`]), so every float sent to the
/// GPU is finite however far the camera is (Review Focus 2).
fn frame_params(
    lens: Option<&LensView>,
    view_rot: Mat4,
    proj: Mat4,
    size: UVec2,
    scale: f32,
) -> (LensCoverage, LensParams) {
    let size = size.max(UVec2::ONE);
    let mut p = LensParams {
        cam: [0.0; 4],
        rect: [0; 4],
        grid: [0; 4],
        scale: [1.0, 1.0, 0.0, 1.0 / scale],
    };
    let unlensed = (LensCoverage::Unlensed, p);
    let Some(cam) = lens.and_then(camera_in_m) else {
        return unlensed;
    };
    let r_cam = cam.length();
    if !r_cam.is_finite() {
        return unlensed;
    }
    if r_cam < binet::CAPTURE_RADIUS {
        p.grid[3] = 1;
        p.rect = [0, 0, size.x, size.y];
        return (LensCoverage::InsideHorizon, p);
    }
    let (proj_x, proj_y) = (f64::from(proj.x_axis.x), f64::from(proj.y_axis.y));
    if !(proj_x > 0.0 && proj_y > 0.0 && proj_x.is_finite() && proj_y.is_finite()) {
        return unlensed;
    }
    // Angular size of one full-resolution pixel at the screen centre (2 NDC units over h rows).
    let pixel_angle = 2.0 / (proj_y * f64::from(size.y));
    let theta = cone_half_angle(r_cam, pixel_angle);
    if theta < 0.5 * pixel_angle {
        return unlensed;
    }
    let axis = view_rot
        .as_dmat4()
        .transform_vector3(-cam / r_cam)
        .normalize();
    let rect = cone_rect(axis, theta, proj_x, proj_y, size);
    if rect.size.x == 0 || rect.size.y == 0 {
        return unlensed;
    }
    // Lensing is visible on screen. Beyond the reach the step budget ends before the rays get
    // near the hole, so a trace would draw the unlensed sky anyway, at full cost.
    if r_cam > TRACE_REACH {
        return (LensCoverage::BeyondReach, p);
    }
    let grid = sample_grid(rect.size, scale);
    let per_sample = rect.size.as_vec2() / grid.as_vec2();
    p.cam = [cam.x as f32, cam.y as f32, cam.z as f32, r_cam as f32];
    p.rect = [rect.min.x, rect.min.y, rect.size.x, rect.size.y];
    p.grid = [grid.x, grid.y, GPU_MAX_STEPS, 0];
    p.scale = [
        per_sample.x,
        per_sample.y,
        gpu_r_far(r_cam) as f32,
        1.0 / scale,
    ];
    (LensCoverage::Traced, p)
}

/// A compute-visible group-1 entry.
fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    }
}

/// A uniform buffer of `T`.
fn uniform_entry<T>(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size_of::<T>() as u64),
        },
    )
}

/// Group 1, binding 0: the `LensParams` uniform.
fn params_entry() -> wgpu::BindGroupLayoutEntry {
    uniform_entry::<LensParams>(0)
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
    )
}

/// Group 1 of `cs_trace`: params and the samples it writes.
fn trace_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [params_entry(), storage_entry(1, SAMPLES_FORMAT)]
}

/// Group 1 of `cs_resolve`: params, the samples it reads, `hdr` and `lens_mask`, the
/// environment mode, the depth texture, and the environment cubemap with its sampler.
fn resolve_layout_entries() -> [wgpu::BindGroupLayoutEntry; 8] {
    [
        params_entry(),
        entry(
            2,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        storage_entry(3, HDR_FORMAT),
        storage_entry(4, LENS_MASK_FORMAT),
        uniform_entry::<LensEnvParams>(5),
        entry(
            6,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
        ),
        // uni_gfx::EnvMap: an Rgba16Float cube (filterable) and its trilinear sampler.
        entry(
            7,
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            },
        ),
        entry(
            8,
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        ),
    ]
}

fn compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    entry_point: &str,
    layouts: [&wgpu::BindGroupLayout; 2],
) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts.map(Some),
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// The brute-force Schwarzschild lens pass (`RenderFeature` named "lens"). Group 0 is the
/// renderer's ([`FrameBindings::camera`]); group 1 holds the parameters and textures.
pub struct LensTracer {
    trace_pipeline: wgpu::ComputePipeline,
    resolve_pipeline: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    params_buffer: wgpu::Buffer,
    params: LensParams,
    /// The `LensEnvParams` uniform, written in `prepare` from `mode`.
    env_params_buffer: wgpu::Buffer,
    resolution_scale: LensScale,
    /// The environment cubemap escaped rays sample with [`LensEnv::Band`].
    env: Arc<EnvMap>,
    /// The lens environment the app selects each frame, read in `prepare`.
    mode: SharedInput<LensEnv>,
    /// Size-dependent resources; `prepare` rebuilds them after a resize or a scale change.
    frame: Option<FrameResources>,
}

/// Resources that depend on the frame targets and the resolution scale.
struct FrameResources {
    target_size: UVec2,
    samples_size: UVec2,
    /// Held alongside the bind groups that view it.
    _samples: wgpu::Texture,
    trace_bind_group: wgpu::BindGroup,
    resolve_bind_group: wgpu::BindGroup,
}

impl LensTracer {
    /// Registers the uni-rel shaders in `lib`, compiles the lens pass and creates its
    /// pipelines and uniforms. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `scale`
    /// is the resolution-scale handle the app steers; `env` is the environment cubemap
    /// (`uni_sky::BandCubemap::env_map`, sampled by escaped rays with [`LensEnv::Band`]) and
    /// `mode` the lens environment the app sets each frame. Shader or pipeline validation errors
    /// are returned, not raised on the device.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        scale: LensScale,
        env: Arc<EnvMap>,
        mode: SharedInput<LensEnv>,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib
            .module(device, crate::LENS_TRACE_WGSL)
            .context("compiling the lens shader")?;

        // Pipeline creation reports validation errors to the device, not as a Result.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens trace bgl"),
            entries: &trace_layout_entries(),
        });
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("lens resolve bgl"),
            entries: &resolve_layout_entries(),
        });
        let trace_pipeline = compute_pipeline(
            device,
            "lens trace",
            &module,
            "cs_trace",
            [camera_bgl, &trace_bgl],
        );
        let resolve_pipeline = compute_pipeline(
            device,
            "lens resolve",
            &module,
            "cs_resolve",
            [camera_bgl, &resolve_bgl],
        );
        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params_buffer = uniform("lens params", size_of::<LensParams>());
        let env_params_buffer = uniform("lens env params", size_of::<LensEnvParams>());
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the lens pipelines: {err}");
        }

        Ok(Self {
            trace_pipeline,
            resolve_pipeline,
            trace_bgl,
            resolve_bgl,
            params_buffer,
            params: bytemuck::Zeroable::zeroed(),
            env_params_buffer,
            resolution_scale: scale,
            env,
            mode,
            frame: None,
        })
    }

    /// Fraction of full resolution at which rays are traced, clamped to 0.25..=1.0 (NaN is
    /// ignored). Default 0.5 for interactive use; captures use 1.0. Applies from the next frame.
    /// The same as `set` on the [`LensScale`] handle the tracer was made with.
    pub fn set_resolution_scale(&mut self, s: f32) {
        self.resolution_scale.set(s);
    }

    fn frame_resources(
        &self,
        device: &wgpu::Device,
        targets: &FrameTargets,
        samples_size: UVec2,
    ) -> FrameResources {
        let samples = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("lens samples"),
            size: wgpu::Extent3d {
                width: samples_size.x,
                height: samples_size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SAMPLES_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = samples.create_view(&wgpu::TextureViewDescriptor::default());
        let params = || wgpu::BindGroupEntry {
            binding: 0,
            resource: self.params_buffer.as_entire_binding(),
        };
        let trace_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens trace bg"),
            layout: &self.trace_bgl,
            entries: &[
                params(),
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
            ],
        });
        let resolve_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lens resolve bg"),
            layout: &self.resolve_bgl,
            entries: &[
                params(),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&targets.hdr_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.env_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&targets.depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.env.view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&self.env.sampler),
                },
            ],
        });
        FrameResources {
            target_size: targets.size,
            samples_size,
            _samples: samples,
            trace_bind_group,
            resolve_bind_group,
        }
    }
}

impl RenderFeature for LensTracer {
    fn name(&self) -> &'static str {
        "lens"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        let scale = self.resolution_scale.get();
        (_, self.params) = frame_params(
            ctx.lenses.first(),
            ctx.view_rot,
            ctx.proj,
            targets.size,
            scale,
        );
        if !self.params.writes_pixels() {
            return; // encode records nothing this frame
        }
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        let env_params = LensEnvParams::new(self.mode.get());
        gpu.queue.write_buffer(
            &self.env_params_buffer,
            0,
            bytemuck::bytes_of(&env_params),
        );
        // Sized for the largest rectangle (the whole frame) at the current scale.
        let samples_size = sample_grid(targets.size, scale);
        let stale = self
            .frame
            .as_ref()
            .is_none_or(|f| f.target_size != targets.size || f.samples_size != samples_size);
        if stale {
            self.frame = Some(self.frame_resources(&gpu.device, targets, samples_size));
        }
    }

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        bindings: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if !self.params.writes_pixels() {
            return; // nothing traced and outside every horizon: the background stays
        }
        let Some(frame) = &self.frame else {
            return; // prepare has not run yet
        };
        // One pass, two dispatches: each dispatch is its own usage scope, so the samples
        // written by cs_trace are visible to cs_resolve.
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("lens"),
            timestamp_writes: timer.compute_writes("lens"),
        });
        let grid = UVec2::new(self.params.grid[0], self.params.grid[1]);
        if grid.x > 0 && grid.y > 0 {
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, bindings.camera, &[]);
            pass.set_bind_group(1, &frame.trace_bind_group, &[]);
            pass.dispatch_workgroups(grid.x.div_ceil(WORKGROUP), grid.y.div_ceil(WORKGROUP), 1);
        }
        // Only the rectangle: pixels outside it keep the background.
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, bindings.camera, &[]);
        pass.set_bind_group(1, &frame.resolve_bind_group, &[]);
        pass.dispatch_workgroups(
            self.params.rect[2].div_ceil(WORKGROUP),
            self.params.rect[3].div_ceil(WORKGROUP),
            1,
        );
    }

    fn resize(&mut self, _gpu: &GpuContext, _targets: &FrameTargets) {
        // The bind groups view the old targets; prepare rebuilds them.
        self.frame = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BINET_WGSL, DEBUG_BACKGROUND_WGSL, DEBUG_SKY_WGSL, LENS_TRACE_WGSL};
    use core::mem::{offset_of, size_of};
    use glam::{DQuat, DVec2, IVec2, Vec2, Vec4};
    use uni_gfx::env::cube_dir;
    use uni_gfx::{reversed_z_infinite_proj, validate_wgsl};
    use wgpu::naga;

    fn lib() -> ShaderLib {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib
    }

    fn lens_trace_module() -> naga::Module {
        let src = lib().compose(LENS_TRACE_WGSL).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn lens_shaders_validate() {
        let lib = lib();
        for path in [
            BINET_WGSL,
            DEBUG_SKY_WGSL,
            DEBUG_BACKGROUND_WGSL,
            LENS_TRACE_WGSL,
        ] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}");
            }
        }
        // Names the lens pass and the selftest (Task 11) rely on.
        let m = lens_trace_module();
        let has_fn = |name: &str| {
            m.functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some(name))
        };
        for name in [
            "binet_trace",
            "static_observer_ray",
            "debug_sky",
            "lens_env_lod",
            "lens_env_units",
        ] {
            assert!(has_fn(name), "missing fn {name}");
        }
        let entries: Vec<&str> = m.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(entries, ["cs_trace", "cs_resolve"]);
    }

    /// Size and member offsets of the WGSL struct `name` in `module`.
    fn wgsl_struct_layout(module: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name} in lens_trace.wgsl"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let offsets = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (layouter[handle].size as usize, offsets)
    }

    /// `LensParams` keeps m0.1's layout (64 B: cam, rect, grid, scale at 0, 16, 32, 48), and
    /// the WGSL twin matches it. B7 adds `LensEnvParams` rather than changing it.
    #[test]
    fn lens_params_layout_unchanged() {
        let module = lens_trace_module();
        let (size, wgsl) = wgsl_struct_layout(&module, "LensParams");
        let m01: Vec<(String, usize)> = [("cam", 0), ("rect", 16), ("grid", 32), ("scale", 48)]
            .into_iter()
            .map(|(n, o)| (n.to_owned(), o))
            .collect();
        assert_eq!(size_of::<LensParams>(), 64);
        assert_eq!(size, 64);
        assert_eq!(wgsl, m01);
        let rust: Vec<(String, usize)> = [
            ("cam", offset_of!(LensParams, cam)),
            ("rect", offset_of!(LensParams, rect)),
            ("grid", offset_of!(LensParams, grid)),
            ("scale", offset_of!(LensParams, scale)),
        ]
        .into_iter()
        .map(|(n, o)| (n.to_owned(), o))
        .collect();
        assert_eq!(rust, m01);
    }

    #[test]
    fn lens_env_params_layout_matches_wgsl() {
        let module = lens_trace_module();
        let (size, wgsl) = wgsl_struct_layout(&module, "LensEnvParams");
        assert_eq!(size_of::<LensEnvParams>(), 16);
        assert_eq!(size, size_of::<LensEnvParams>());
        assert_eq!(
            wgsl,
            [("mode".to_owned(), offset_of!(LensEnvParams, mode))]
        );
    }

    /// The literal value of WGSL constant `name`.
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

    /// The WGSL twins of the environment unit and the mode codes, and the mode written for each
    /// [`LensEnv`].
    #[test]
    fn lens_env_constants_match_wgsl() {
        let m = lens_trace_module();
        assert_eq!(
            wgsl_const(&m, "LENS_ENV_UNIT_CD_M2"),
            naga::Literal::F32(uni_gfx::ENV_UNIT_CD_M2)
        );
        assert_eq!(
            wgsl_const(&m, "LENS_ENV_DEBUG_GRID"),
            naga::Literal::U32(LENS_ENV_DEBUG_GRID)
        );
        assert_eq!(
            wgsl_const(&m, "LENS_ENV_BAND"),
            naga::Literal::U32(LENS_ENV_BAND)
        );
        assert_eq!(
            LensEnvParams::new(LensEnv::DebugGrid).mode,
            [LENS_ENV_DEBUG_GRID, 0, 0, 0]
        );
        assert_eq!(
            LensEnvParams::new(LensEnv::Band).mode,
            [LENS_ENV_BAND, 0, 0, 0]
        );
    }

    /// Texels per face edge of the band's cubemap (`uni_sky::ENV_SIZE`, B5) and its mips
    /// (`uni_sky::ENV_MIPS`); uni-rel does not depend on uni-sky, so they are repeated here.
    const BAND_FACE: f32 = 256.0;
    const BAND_MIPS: f32 = 5.0;

    /// The mip level is log2(footprint / texel angle), clamped to the map's levels; one texel
    /// spans 2/N rad at a face centre and 2m²/N rad in general (m = the direction's major-axis
    /// component), which is the geometry of `cube_dir`.
    #[test]
    fn env_lod_from_the_exit_direction_footprint() {
        let (n, levels) = (BAND_FACE, BAND_MIPS);
        let texel = 2.0 / n;
        for k in 0..5 {
            let lod = env_lod(Vec3::Z, texel * 2f32.powi(k), n, levels);
            assert!((lod - k as f32).abs() < 1e-5, "{k}: {lod}");
        }
        // Clamped at level 0 (the map is magnified) and at the last level.
        assert_eq!(env_lod(Vec3::NEG_X, 0.1 * texel, n, levels), 0.0);
        assert_eq!(env_lod(Vec3::Y, 1e3 * texel, n, levels), levels - 1.0);
        assert_eq!(env_lod(Vec3::NEG_Y, 0.0, n, levels), 0.0);
        assert_eq!(env_lod(Vec3::Z, 1.0, n, 1.0), 0.0);
        // Toward a corner (m² = 1/3) a texel is 3× smaller across, so log2(3) levels coarser;
        // the direction's length does not matter.
        let lod = env_lod(Vec3::ONE, texel, n, levels);
        assert!((lod - 3f32.log2()).abs() < 1e-5, "{lod}");
        assert_eq!(env_lod(Vec3::ONE * 7.0, texel, n, levels), lod);

        // The texel angle 2m²/N against cube_dir's own geometry: the angle between neighbouring
        // texel centres along a face axis, from the centre out to the edge (in f64, from the
        // chord: acos of an f32 dot product near 1 loses ~1e-3 of a 0.007 rad angle).
        for x in [128u32, 160, 200, 240, 255] {
            let uv = |x: u32| Vec2::new((x as f32 + 0.5) / n, 0.5);
            let (d0, d1) = (
                cube_dir(4, uv(x - 1)).as_dvec3(),
                cube_dir(4, uv(x)).as_dvec3(),
            );
            let angle = 2.0 * ((d1 - d0).length() / 2.0).asin();
            let m = (d0 + d1).normalize().z;
            let predicted = 2.0 * m * m / f64::from(n);
            assert!(
                (angle / predicted - 1.0).abs() < 1e-4,
                "texel {x}: {angle} vs {predicted}"
            );
        }
    }

    /// No seam where the traced rectangle meets the band: at its edge light is bent by under
    /// half a pixel, so a sample's footprint is the unlensed one. At 1080 rows and 60°, even at
    /// the screen centre (the largest pixel) and toward a cube corner (the smallest texel),
    /// half-resolution samples select level 0, the level the band pass draws.
    #[test]
    fn env_lod_is_level_0_at_the_rect_edge() {
        // Angular size of one pixel at the screen centre, as frame_params works it out.
        let pixel = (2.0 / (f64::from(proj().y_axis.y) * f64::from(H))) as f32;
        let per_sample = 1.0 / DEFAULT_RESOLUTION_SCALE;
        for dir in [Vec3::Z, Vec3::new(1.0, 1.0, 0.0), Vec3::ONE] {
            assert_eq!(
                env_lod(dir, pixel * per_sample, BAND_FACE, BAND_MIPS),
                0.0,
                "{dir}"
            );
        }
        // Worst case, for the record: log2(fp·N·3/2) ≈ −0.28 at the corner.
        let worst = (pixel * per_sample * BAND_FACE * 1.5).log2();
        assert!(worst < -0.2, "{worst}");
    }

    // ---- cs_resolve: CPU model ----

    /// A value no pass writes, so an untouched texel shows.
    const SENTINEL: Vec4 = Vec4::splat(7.0);

    /// CPU model of `cs_resolve` (shaders/lens_trace.wgsl), statement for statement; the GPU
    /// pass itself is checked by B9's captures, as cargo test cannot reach the GPU. One
    /// invocation per pixel of the dispatch (the rectangle). `samples` is cs_trace's grid
    /// (row-major, `p.grid[0]` wide; xyz the exit direction, w = 1 if captured); `depth`, `hdr`
    /// and `mask` are frame-sized (row-major, `size.x` wide). `shade` is `shade_sample`'s colour
    /// for an escaped sample (the environment, in either mode).
    fn resolve_model(
        p: &LensParams,
        size: UVec2,
        samples: &[Vec4],
        depth: &[f32],
        hdr: &mut [Vec4],
        mask: &mut [Vec4],
        shade: impl Fn(Vec4) -> Vec3,
    ) {
        let at = |v: UVec2| (v.y * size.x + v.x) as usize;
        let shade_sample = |s: Vec4| {
            if s.w > 0.5 {
                Vec4::W
            } else {
                shade(s).extend(0.0)
            }
        };
        let rect_min = UVec2::new(p.rect[0], p.rect[1]);
        for gy in 0..p.rect[3] {
            for gx in 0..p.rect[2] {
                let gid = UVec2::new(gx, gy);
                let px = rect_min + gid;
                if px.x >= size.x || px.y >= size.y {
                    continue;
                }
                if depth[at(px)] != 0.0 {
                    continue; // a body is in front: not lensed
                }
                if p.grid[3] != 0 {
                    hdr[at(px)] = Vec4::W;
                    mask[at(px)] = Vec4::new(1.0, 0.0, 1.0, 0.0);
                    continue;
                }
                let q = (gid.as_vec2() + 0.5) / Vec2::new(p.scale[0], p.scale[1]) - 0.5;
                let q0 = q.floor();
                let t = (q - q0).clamp(Vec2::ZERO, Vec2::ONE);
                let last = UVec2::new(p.grid[0], p.grid[1]).as_ivec2() - 1;
                let i0 = q0.as_ivec2().clamp(IVec2::ZERO, last);
                let i1 = (q0.as_ivec2() + 1).clamp(IVec2::ZERO, last);
                let load = |i: IVec2| samples[(i.y * p.grid[0] as i32 + i.x) as usize];
                let s00 = load(i0);
                let s10 = load(IVec2::new(i1.x, i0.y));
                let s01 = load(IVec2::new(i0.x, i1.y));
                let s11 = load(i1);
                let top = shade_sample(s00).lerp(shade_sample(s10), t.x);
                let bottom = shade_sample(s01).lerp(shade_sample(s11), t.x);
                let c = top.lerp(bottom, t.y);
                hdr[at(px)] = c.truncate().extend(1.0);
                mask[at(px)] = Vec4::new(c.w, 0.0, 1.0, 0.0);
            }
        }
    }

    /// Hand-made plan: rectangle `rect`, `grid` samples, `per_sample` pixels per sample.
    fn plan(rect: [u32; 4], grid: [u32; 2], per_sample: [f32; 2]) -> LensParams {
        LensParams {
            cam: [0.0, 0.0, 30.0, 30.0],
            rect,
            grid: [grid[0], grid[1], GPU_MAX_STEPS, 0],
            scale: [per_sample[0], per_sample[1], 1000.0, 1.0 / per_sample[0]],
        }
    }

    /// The M1 mask contract on traced pixels: r = captured fraction, g = 0, b = 1, a = 0; the
    /// colour is black where captured and the environment where escaped; nothing outside the
    /// rectangle changes.
    #[test]
    fn resolve_writes_mask_channels() {
        let size = UVec2::new(12, 10);
        let n = (size.x * size.y) as usize;
        let sky = Vec3::new(0.25, 0.5, 0.75);
        let escaped = Vec4::new(0.0, 0.6, 0.8, 0.0);

        // Full resolution over x 3..9, y 2..7: one sample per pixel.
        let p = plan([3, 2, 6, 5], [6, 5], [1.0, 1.0]);
        let captured = |i: u32, j: u32| (i + 2 * j) % 3 == 0;
        let samples: Vec<Vec4> = (0..5)
            .flat_map(|j| (0..6).map(move |i| if captured(i, j) { Vec4::W } else { escaped }))
            .collect();
        let depth = vec![0.0; n];
        let (mut hdr, mut mask) = (vec![SENTINEL; n], vec![SENTINEL; n]);
        resolve_model(&p, size, &samples, &depth, &mut hdr, &mut mask, |_| sky);
        for y in 0..size.y {
            for x in 0..size.x {
                let k = (y * size.x + x) as usize;
                if !((3..9).contains(&x) && (2..7).contains(&y)) {
                    assert_eq!((hdr[k], mask[k]), (SENTINEL, SENTINEL), "({x}, {y})");
                    continue;
                }
                let c = captured(x - 3, y - 2);
                let r = if c { 1.0 } else { 0.0 };
                assert_eq!(mask[k], Vec4::new(r, 0.0, 1.0, 0.0), "({x}, {y})");
                let colour = if c { Vec4::W } else { sky.extend(1.0) };
                assert_eq!(hdr[k], colour, "({x}, {y})");
            }
        }

        // Half resolution: r is the bilinear capture fraction between a captured and an
        // escaped sample, (1, 0.75, 0.25, 0) across four pixels, and the colour fades likewise.
        let p = plan([0, 0, 4, 1], [2, 1], [2.0, 1.0]);
        let samples = [Vec4::W, escaped];
        let (mut hdr, mut mask) = (vec![SENTINEL; 4], vec![SENTINEL; 4]);
        resolve_model(
            &p,
            UVec2::new(4, 1),
            &samples,
            &[0.0; 4],
            &mut hdr,
            &mut mask,
            |_| sky,
        );
        for (k, r) in [1.0, 0.75, 0.25, 0.0].into_iter().enumerate() {
            assert_eq!(mask[k], Vec4::new(r, 0.0, 1.0, 0.0), "pixel {k}");
            let want = (sky * (1.0 - r)).extend(1.0);
            assert!((hdr[k] - want).abs().max_element() < 1e-6, "pixel {k}");
        }

        // From inside the horizon the rectangle is the whole frame: black, r = 1, b = 1.
        let lens = lens_at(DVec3::new(1.5, 0.0, 0.0));
        let (_, p) = frame_params(Some(&lens), Mat4::IDENTITY, proj(), size, 0.5);
        let (mut hdr, mut mask) = (vec![SENTINEL; n], vec![SENTINEL; n]);
        resolve_model(&p, size, &[], &depth, &mut hdr, &mut mask, |_| sky);
        assert!(hdr.iter().all(|&c| c == Vec4::W));
        assert!(mask.iter().all(|&m| m == Vec4::new(1.0, 0.0, 1.0, 0.0)));
    }

    /// Review m7, design §5.3: a pixel a body drew (depth ≠ 0; reversed-Z clears to 0) keeps
    /// its colour and its mask, so foreground bodies are not lensed and keep their sky
    /// occlusion `g`; its neighbours are lensed as usual. Also from inside the horizon.
    #[test]
    fn lens_skips_foreground_pixels() {
        let size = UVec2::new(8, 6);
        let n = (size.x * size.y) as usize;
        let at = |x: u32, y: u32| (y * size.x + x) as usize;
        let body = Vec4::new(5.0, 4.0, 3.0, 1.0);
        let body_mask = Vec4::new(0.0, 1.0, 0.0, 0.0);
        // Depth 0.3 (a body), 1.0 (at the near plane) and the smallest positive f32.
        let foreground = [(3, 2, 0.3), (5, 4, 1.0), (0, 5, f32::MIN_POSITIVE)];
        let mut depth = vec![0.0; n];
        let (mut hdr0, mut mask0) = (vec![SENTINEL; n], vec![SENTINEL; n]);
        for (x, y, d) in foreground {
            depth[at(x, y)] = d;
            hdr0[at(x, y)] = body;
            mask0[at(x, y)] = body_mask;
        }
        let sky = Vec3::new(0.1, 0.2, 0.3);

        // Traced over the whole frame at half resolution, one captured sample.
        let p = plan([0, 0, 8, 6], [4, 3], [2.0, 2.0]);
        let samples: Vec<Vec4> = (0..12)
            .map(|k| {
                if k == 5 {
                    Vec4::W
                } else {
                    Vec4::new(0.0, 0.0, 1.0, 0.0)
                }
            })
            .collect();
        let inside_lens = lens_at(DVec3::new(0.0, 1.0, 0.0));
        let (_, inside) = frame_params(Some(&inside_lens), Mat4::IDENTITY, proj(), size, 0.5);
        for p in [p, inside] {
            let (mut hdr, mut mask) = (hdr0.clone(), mask0.clone());
            resolve_model(&p, size, &samples, &depth, &mut hdr, &mut mask, |_| sky);
            for y in 0..size.y {
                for x in 0..size.x {
                    let k = at(x, y);
                    if depth[k] != 0.0 {
                        assert_eq!((hdr[k], mask[k]), (body, body_mask), "({x}, {y})");
                    } else {
                        assert_eq!((mask[k].y, mask[k].z, mask[k].w), (0.0, 1.0, 0.0));
                        assert_ne!(hdr[k], SENTINEL, "({x}, {y})");
                    }
                }
            }
        }
    }

    /// What a binding is, as a comparable string, from the WGSL side.
    fn wgsl_binding_kind(module: &naga::Module, var: &naga::GlobalVariable) -> String {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        match (&var.space, &module.types[var.ty].inner) {
            (naga::AddressSpace::Uniform, _) => format!("uniform {}", layouter[var.ty].size),
            (naga::AddressSpace::Handle, naga::TypeInner::Sampler { comparison: false }) => {
                "sampler".into()
            }
            (
                naga::AddressSpace::Handle,
                naga::TypeInner::Image {
                    dim: naga::ImageDimension::Cube,
                    arrayed: false,
                    class:
                        naga::ImageClass::Sampled {
                            kind: naga::ScalarKind::Float,
                            multi: false,
                        },
                },
            ) => "cube float".into(),
            (
                naga::AddressSpace::Handle,
                naga::TypeInner::Image {
                    dim: naga::ImageDimension::D2,
                    arrayed: false,
                    class,
                },
            ) => match class {
                naga::ImageClass::Storage { format, access } => {
                    assert_eq!(
                        *access,
                        naga::StorageAccess::STORE,
                        "storage textures are write-only"
                    );
                    format!("storage {format:?}")
                }
                naga::ImageClass::Sampled {
                    kind: naga::ScalarKind::Float,
                    multi: false,
                } => "texture float".into(),
                naga::ImageClass::Depth { multi: false } => "depth".into(),
                other => panic!("unexpected image class {other:?}"),
            },
            other => panic!("unexpected binding {other:?}"),
        }
    }

    /// The same, from a Rust bind group layout entry.
    fn rust_binding_kind(e: &wgpu::BindGroupLayoutEntry) -> String {
        assert_eq!(e.visibility, wgpu::ShaderStages::COMPUTE);
        match e.ty {
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: Some(size),
            } => format!("uniform {size}"),
            wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering) => "sampler".into(),
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::Cube,
                multisampled: false,
            } => "cube float".into(),
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            } => "depth".into(),
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format,
                view_dimension: wgpu::TextureViewDimension::D2,
            } => format!(
                "storage {:?}",
                match format {
                    wgpu::TextureFormat::Rgba32Float => naga::StorageFormat::Rgba32Float,
                    wgpu::TextureFormat::Rgba16Float => naga::StorageFormat::Rgba16Float,
                    wgpu::TextureFormat::R32Float => naga::StorageFormat::R32Float,
                    other => panic!("unexpected storage format {other:?}"),
                }
            ),
            wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            } => "texture float".into(),
            other => panic!("unexpected layout entry {other:?}"),
        }
    }

    /// Each entry point uses exactly the group-1 bindings of its layout, with matching types, so
    /// pipeline creation cannot fail on layout. At group 0 cs_trace uses the camera only, and
    /// cs_resolve the camera and the exposure (`star_pre()`, for the band environment).
    #[test]
    fn bind_group_layouts_match_shader_bindings() {
        let module = lens_trace_module();
        let caps = naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap();
        let layouts: [(&str, Vec<wgpu::BindGroupLayoutEntry>, &[(u32, &str)]); 2] = [
            (
                "cs_trace",
                trace_layout_entries().to_vec(),
                &[(0, "camera")],
            ),
            (
                "cs_resolve",
                resolve_layout_entries().to_vec(),
                &[(0, "camera"), (1, "exposure")],
            ),
        ];
        for (i, (name, entries, group0)) in layouts.iter().enumerate() {
            assert_eq!(module.entry_points[i].name, *name);
            let used = info.get_entry_point(i);
            let mut wgsl: Vec<(u32, String)> = Vec::new();
            let mut wgsl_group0: Vec<(u32, &str)> = Vec::new();
            for (h, var) in module.global_variables.iter() {
                if used[h].is_empty() {
                    continue;
                }
                let rb = var.binding.as_ref().expect("resource binding");
                if rb.group == 0 {
                    wgsl_group0.push((rb.binding, var.name.as_deref().unwrap_or_default()));
                    continue;
                }
                assert_eq!(rb.group, 1, "{name}: {:?}", var.name);
                wgsl.push((rb.binding, wgsl_binding_kind(&module, var)));
            }
            wgsl_group0.sort();
            assert_eq!(wgsl_group0, *group0, "{name}");
            wgsl.sort();
            let mut rust: Vec<(u32, String)> = entries
                .iter()
                .map(|e| (e.binding, rust_binding_kind(e)))
                .collect();
            rust.sort();
            assert_eq!(wgsl, rust, "{name}");
        }
    }

    // ---- CPU planning ----

    const W: u32 = 1920;
    const H: u32 = 1080;

    fn proj() -> Mat4 {
        reversed_z_infinite_proj(60f32.to_radians(), W as f32 / H as f32, 0.1)
    }

    /// A 10 M☉ lens with the camera at `cam_m` (in units of M, hole at the origin).
    fn lens_at(cam_m: DVec3) -> LensView {
        let mass_kg = 10.0 * uni_core::M_SUN;
        LensView {
            rel_pos_m: -cam_m * uni_core::gravitational_radius(mass_kg),
            mass_kg,
            spin: 0.0,
        }
    }

    /// Pixel coordinates (x right, y down from the top) of a camera-space direction in front.
    fn project(d: DVec3, sx: f64, sy: f64) -> DVec2 {
        let w = -d.z;
        assert!(w > 0.0);
        DVec2::new(
            (d.x / w * sx + 1.0) * 0.5 * f64::from(W),
            (1.0 - d.y / w * sy) * 0.5 * f64::from(H),
        )
    }

    #[test]
    fn cone_rect_bounds_the_projected_cone_tightly() {
        let p = proj();
        let (sx, sy) = (f64::from(p.x_axis.x), f64::from(p.y_axis.y));
        let size = UVec2::new(W, H);
        let axes = [
            DVec3::NEG_Z,
            DVec3::new(0.3, -0.2, -1.0).normalize(),
            DVec3::new(-0.5, 0.4, -0.7).normalize(),
            DVec3::new(0.9, 0.0, -0.6).normalize(), // partly off screen: clamped
        ];
        for axis in axes {
            for theta in [0.002, 0.05, 0.3] {
                let rect = cone_rect(axis, theta, sx, sy, size);
                // Orthonormal basis around the axis; sample the cone's boundary.
                let u = axis.any_orthonormal_vector();
                let v = axis.cross(u);
                let (mut lo, mut hi) =
                    (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
                for k in 0..4096 {
                    let phi = std::f64::consts::TAU * f64::from(k) / 4096.0;
                    let d = axis * theta.cos() + (u * phi.cos() + v * phi.sin()) * theta.sin();
                    let q = project(d, sx, sy);
                    lo = lo.min(q);
                    hi = hi.max(q);
                }
                let screen = DVec2::new(f64::from(W), f64::from(H));
                let (lo, hi) = (lo.clamp(DVec2::ZERO, screen), hi.clamp(DVec2::ZERO, screen));
                let ctx = format!("axis {axis} theta {theta}: rect {rect:?}, cone {lo}..{hi}");
                if hi.x <= lo.x || hi.y <= lo.y {
                    assert_eq!(rect.size, UVec2::ZERO, "off screen: {ctx}");
                    continue;
                }
                let min = rect.min.as_dvec2();
                let max = (rect.min + rect.size).as_dvec2();
                assert!(min.x <= lo.x && min.y <= lo.y, "{ctx}");
                assert!(max.x >= hi.x && max.y >= hi.y, "{ctx}");
                // Tight: one pixel of padding plus rounding.
                assert!(lo.x - min.x <= 2.0 && lo.y - min.y <= 2.0, "{ctx}");
                assert!(max.x - hi.x <= 2.0 && max.y - hi.y <= 2.0, "{ctx}");
            }
        }
    }

    #[test]
    fn cone_rect_full_or_empty_when_unbounded_or_behind() {
        let size = UVec2::new(W, H);
        let full = PixelRect {
            min: UVec2::ZERO,
            size,
        };
        // A cone of ≥ 90° covers every direction in front of the camera.
        assert_eq!(
            cone_rect(DVec3::NEG_Z, std::f64::consts::FRAC_PI_2, 1.0, 1.7, size),
            full
        );
        // Straddling the camera plane: its projection is unbounded.
        assert_eq!(cone_rect(DVec3::X, 0.1, 1.0, 1.7, size), full);
        // Entirely behind the camera.
        assert_eq!(cone_rect(DVec3::Z, 0.1, 1.0, 1.7, size).size, UVec2::ZERO);
        // In front but entirely off screen to the right.
        let off = DVec3::new(3.0, 0.0, -1.0).normalize();
        assert_eq!(cone_rect(off, 0.01, 1.0, 1.7, size).size, UVec2::ZERO);
    }

    #[test]
    fn close_camera_traces_full_screen_at_half_resolution() {
        // Camera 30 M from the hole along +z, looking along −z: the hole is dead ahead.
        let lens = lens_at(DVec3::new(0.0, 0.0, 30.0));
        let (coverage, p) =
            frame_params(Some(&lens), Mat4::IDENTITY, proj(), UVec2::new(W, H), 0.5);
        assert_eq!(coverage, LensCoverage::Traced);
        assert_eq!(p.rect, [0, 0, W, H]);
        assert_eq!(p.grid, [W / 2, H / 2, 2000, 0]);
        assert_eq!(p.scale, [2.0, 2.0, 1000.0, 2.0]);
        assert!(
            (p.cam[2] - 30.0).abs() < 1e-4
                && p.cam[0].abs() < 1e-6
                && (p.cam[3] - 30.0).abs() < 1e-4
        );
        // Full resolution: one sample per pixel.
        let (_, p) = frame_params(Some(&lens), Mat4::IDENTITY, proj(), UVec2::new(W, H), 1.0);
        assert_eq!((p.grid[0], p.grid[1]), (W, H));
        assert_eq!((p.scale[0], p.scale[1], p.scale[3]), (1.0, 1.0, 1.0));
    }

    /// Nothing traced, and every float the GPU receives is finite.
    fn assert_untraced(p: &LensParams, ctx: &str) {
        assert_eq!(p.grid, [0; 4], "{ctx}: {p:?}");
        assert_eq!(p.rect, [0; 4], "{ctx}: {p:?}");
        assert!(
            p.cam.iter().chain(&p.scale).all(|x| x.is_finite()),
            "{ctx}: {p:?}"
        );
    }

    /// Review I1: from 1e6 M, light within ~7 px of the hole is bent by half a pixel or more,
    /// but the step budget cannot carry a ray to the hole. Nothing is traced, and the coverage
    /// asks the app for a HUD label.
    #[test]
    fn far_camera_beyond_reach_is_labelled_not_traced() {
        let size = UVec2::new(W, H);
        let lens = lens_at(DVec3::new(0.0, 0.0, 1e6));
        let (coverage, p) = frame_params(Some(&lens), Mat4::IDENTITY, proj(), size, 0.5);
        assert_eq!(coverage, LensCoverage::BeyondReach);
        assert_eq!(coverage.warning(), Some(BEYOND_REACH_WARNING));
        assert_untraced(&p, "1e6 M");

        // Either side of the reach.
        let inside = lens_at(DVec3::new(0.0, 0.0, 0.999 * TRACE_REACH));
        let (coverage, p) = frame_params(Some(&inside), Mat4::IDENTITY, proj(), size, 0.5);
        assert_eq!(coverage, LensCoverage::Traced);
        assert_eq!(coverage.warning(), None);
        assert_eq!(p.grid, [W / 2, H / 2, GPU_MAX_STEPS, 0]);
        assert_eq!(p.scale[2], (2.0 * 0.999 * TRACE_REACH) as f32);
        let outside = lens_at(DVec3::new(0.0, 0.0, 1.001 * TRACE_REACH));
        let (coverage, p) = frame_params(Some(&outside), Mat4::IDENTITY, proj(), size, 0.5);
        assert_eq!(coverage, LensCoverage::BeyondReach);
        assert_untraced(&p, "just beyond the reach");

        // Hole behind the camera: no lensing on screen, so nothing to label.
        let back = Mat4::from_quat(DQuat::from_rotation_y(std::f64::consts::PI).as_quat());
        let (coverage, p) = frame_params(Some(&lens), back, proj(), size, 0.5);
        assert_eq!(coverage, LensCoverage::Unlensed);
        assert_untraced(&p, "behind");
        // So far away that the whole cone is under half a pixel: unlensed is exact to ½ px.
        // 1e40 M also overflows f32, which must not reach the GPU.
        for r in [1e9, 1e40] {
            let far = lens_at(DVec3::new(0.0, 0.0, r));
            let (coverage, p) = frame_params(Some(&far), Mat4::IDENTITY, proj(), size, 0.5);
            assert_eq!(coverage, LensCoverage::Unlensed, "{r}");
            assert_untraced(&p, "far");
        }
    }

    /// The cone is narrower than the screen only for coarse pixels inside the reach: at 180
    /// rows and 60°, θ_max = asin(8/(pixel·r)·√(1 − 2/r)) ≈ 18° from 4000 M.
    #[test]
    fn low_resolution_traces_a_rect_around_the_hole() {
        let (w, h) = (320, 180);
        let proj = reversed_z_infinite_proj(60f32.to_radians(), w as f32 / h as f32, 0.1);
        let lens = lens_at(DVec3::new(0.0, 0.0, 4000.0));
        let (coverage, p) = frame_params(Some(&lens), Mat4::IDENTITY, proj, UVec2::new(w, h), 1.0);
        assert_eq!(coverage, LensCoverage::Traced);
        let (min, size) = (
            UVec2::new(p.rect[0], p.rect[1]),
            UVec2::new(p.rect[2], p.rect[3]),
        );
        // tan 18.2° × f = 0.57 NDC: about 51 px each way from the centre, plus padding.
        assert!(
            size.x > 95 && size.x < 115 && size.y > 95 && size.y < 115,
            "{p:?}"
        );
        let centre = UVec2::new(w / 2, h / 2);
        assert!(
            min.cmple(centre).all() && (min + size).cmpgt(centre).all(),
            "{p:?}"
        );
        assert_eq!((p.grid[0], p.grid[1]), (size.x, size.y));
    }

    /// A ray from a static camera at r [M] on +x, at angle `delta` from the hole's direction,
    /// turning towards +y: (camera, initial tangent).
    fn ray_at(r: f64, delta: f64) -> (DVec3, DVec3) {
        let cam = DVec3::new(r, 0.0, 0.0);
        let local = DVec3::new(-delta.cos(), delta.sin(), 0.0);
        (cam, binet::static_observer_ray(cam, local).unwrap())
    }

    /// What the GPU pass shows for a ray, from the CPU twin with the pass's budget and escape
    /// radius: `None` if captured, or else the exit direction (the last tangent if the budget
    /// ran out, as `cs_trace` does).
    fn gpu_view(cam: DVec3, v0: DVec3) -> Option<DVec3> {
        let r_far = gpu_r_far(cam.length());
        let (end, _, v) = binet::trace_state(cam, v0, r_far, GPU_MAX_STEPS);
        (end != binet::RayEnd::Captured).then(|| v.normalize())
    }

    /// The reference: the same integrator with a budget it never exhausts, escaping at 10·r_cam.
    /// There the remaining deflection, ≈ 0.75·b³/D⁴ in Binet form, is below 1e-7 rad for every
    /// b ≤ r_cam.
    fn long_trace(cam: DVec3, v0: DVec3) -> binet::RayEnd {
        binet::trace(cam, v0, 10.0 * cam.length(), 1_000_000)
    }

    /// Review I1: at the reach, the GPU budget gives the long-budget answer for every ray: the
    /// same capture flag, and an exit direction within 2e-5 rad. This covers both sides of the
    /// shadow edge (b_c = 3√3 ≈ 5.196 M), out to the edge of the hemisphere (b ≈ r_cam),
    /// towards and away from the hole.
    #[test]
    fn gpu_budget_is_exact_within_trace_reach() {
        let r = TRACE_REACH;
        let lapse = (1.0 - 2.0 / r).sqrt();
        let bs = [
            0.0, 1.0, 3.0, 5.0, 5.19, 5.2, 5.25, 5.5, 6.0, 8.0, 10.0, 20.0, 50.0, 100.0, 300.0,
            1000.0, 3000.0, 4990.0,
        ];
        let mut captured = 0;
        for b in bs {
            // sin δ = b·√(1 − 2/r)/r for a static observer (CLAUDE.md §6).
            let towards = (b * lapse / r).asin();
            for delta in [towards, std::f64::consts::PI - towards] {
                let (cam, v0) = ray_at(r, delta);
                let gpu = gpu_view(cam, v0);
                match long_trace(cam, v0) {
                    binet::RayEnd::Captured => {
                        assert_eq!(gpu, None, "b = {b}, δ = {delta}: not captured");
                        captured += 1;
                    }
                    binet::RayEnd::Escaped { dir } => {
                        let got = gpu.unwrap_or_else(|| panic!("b = {b}, δ = {delta}: captured"));
                        let err = (got - dir).length();
                        assert!(err < 2e-5, "b = {b}, δ = {delta}: off by {err:e} rad");
                    }
                    binet::RayEnd::MaxSteps => panic!("reference ran out of steps"),
                }
            }
        }
        // b ≤ 5.19 towards the hole are captured; everything else escapes.
        assert_eq!(captured, 5);
    }

    /// Review I1: just beyond the reach the budget starts to fail, which is why the tracer stops
    /// there. At 1.5× the reach, a ray grazing the photon sphere comes out more than 1e-4 rad
    /// off. At 2× the reach, a ray aimed straight at the hole runs out ~90 M short of it, so the
    /// shadow would vanish.
    #[test]
    fn gpu_budget_fails_beyond_trace_reach() {
        let r = 1.5 * TRACE_REACH;
        let towards = (5.2 * (1.0 - 2.0 / r).sqrt() / r).asin();
        let (cam, v0) = ray_at(r, towards);
        let binet::RayEnd::Escaped { dir } = long_trace(cam, v0) else {
            panic!("b = 5.2 escapes");
        };
        let err = (gpu_view(cam, v0).expect("escapes") - dir).length();
        assert!(err > 1e-4, "{err:e}");

        let r = 2.0 * TRACE_REACH;
        let (cam, v0) = ray_at(r, 0.0);
        assert_eq!(long_trace(cam, v0), binet::RayEnd::Captured);
        let (end, x, _) = binet::trace_state(cam, v0, gpu_r_far(r), GPU_MAX_STEPS);
        assert_eq!(end, binet::RayEnd::MaxSteps);
        assert!(x.length() > 50.0, "stopped at r = {}", x.length());
    }

    /// The public predicate is the tracer's own plan, sized by the viewport.
    #[test]
    fn lens_coverage_matches_the_tracer_plan() {
        let ctx_at = |cam_m: Option<DVec3>| FrameCtx {
            frame_index: 0,
            epoch: uni_core::Epoch { day: 0, sec: 0.0 },
            dt_real_s: 0.0,
            camera: uni_core::CameraView {
                pos: uni_core::UniversePos::default(),
                orient: DQuat::IDENTITY,
                fov_y_rad: 60f64.to_radians(),
                velocity_mps: DVec3::ZERO,
            },
            view_rot: Mat4::IDENTITY,
            proj: proj(),
            viewport: UVec2::new(W, H),
            output_size: UVec2::new(W, H),
            exposure: uni_gfx::Exposure {
                scene_ev100: 0.0,
                star_ev100: 0.0,
            },
            lenses: cam_m.map(lens_at).into_iter().collect(),
            lights: Vec::new(),
            sim_dt_s: 0.0,
            jitter_px: glam::Vec2::ZERO,
        };
        let cases = [
            (None, LensCoverage::Unlensed),
            (Some(DVec3::new(0.0, 0.0, 30.0)), LensCoverage::Traced),
            (Some(DVec3::new(0.0, 0.0, 1.0)), LensCoverage::InsideHorizon),
            (Some(DVec3::new(0.0, 0.0, 1e6)), LensCoverage::BeyondReach),
            (Some(DVec3::new(0.0, 0.0, 1e9)), LensCoverage::Unlensed),
        ];
        for (cam_m, expected) in cases {
            let ctx = ctx_at(cam_m);
            assert_eq!(lens_coverage(&ctx), expected, "{cam_m:?}");
            let plan = frame_params(
                ctx.lenses.first(),
                ctx.view_rot,
                ctx.proj,
                ctx.viewport,
                0.5,
            );
            assert_eq!(plan.0, expected, "{cam_m:?}");
        }
        assert_eq!(LensCoverage::InsideHorizon.warning(), None);
        assert_eq!(LensCoverage::Unlensed.warning(), None);
    }

    #[test]
    fn beyond_reach_warning_names_the_reach() {
        assert!(BEYOND_REACH_WARNING.contains(&format!("> {TRACE_REACH} M")));
    }

    // Review Focus 2: inside or on the horizon, or at the centre, the frame is black and no
    // NaN reaches the GPU.
    #[test]
    fn camera_inside_horizon_blacks_out_the_frame() {
        for cam in [
            DVec3::new(1.5, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::ZERO,
        ] {
            let lens = lens_at(cam);
            assert!(inside_horizon(&lens), "{cam}");
            let (coverage, p) =
                frame_params(Some(&lens), Mat4::IDENTITY, proj(), UVec2::new(W, H), 0.5);
            assert_eq!(coverage, LensCoverage::InsideHorizon, "{cam}");
            assert_eq!(p.grid, [0, 0, 0, 1], "{cam}");
            // The black frame is written by the resolve pass over the whole frame.
            assert_eq!(p.rect, [0, 0, W, H], "{cam}");
            assert!(p.writes_pixels());
            assert!(p.cam.iter().chain(&p.scale).all(|x| x.is_finite()), "{p:?}");
        }
        assert!(!inside_horizon(&lens_at(DVec3::new(2.1, 0.0, 0.0))));
    }

    #[test]
    fn no_lens_or_bad_mass_traces_nothing() {
        let (coverage, p) = frame_params(None, Mat4::IDENTITY, proj(), UVec2::new(W, H), 0.5);
        assert_eq!(coverage, LensCoverage::Unlensed);
        assert_eq!(p.grid, [0, 0, 0, 0]);
        assert_eq!(p.scale[3], 2.0);
        let mut lens = lens_at(DVec3::new(0.0, 0.0, 30.0));
        lens.mass_kg = 0.0;
        assert!(!inside_horizon(&lens));
        let (coverage, p) =
            frame_params(Some(&lens), Mat4::IDENTITY, proj(), UVec2::new(W, H), 0.5);
        assert_eq!(coverage, LensCoverage::Unlensed);
        assert_eq!(p.grid, [0, 0, 0, 0]);
        assert!(p.cam.iter().chain(&p.scale).all(|x| x.is_finite()), "{p:?}");
    }

    #[test]
    fn sample_grid_rounds_up() {
        assert_eq!(sample_grid(UVec2::new(W, H), 0.5), UVec2::new(960, 540));
        assert_eq!(sample_grid(UVec2::new(3, 5), 0.5), UVec2::new(2, 3));
        assert_eq!(sample_grid(UVec2::new(1, 1), 0.25), UVec2::ONE);
        assert_eq!(sample_grid(UVec2::new(7, 9), 1.0), UVec2::new(7, 9));
    }

    #[test]
    fn resolution_scale_is_clamped() {
        assert_eq!(clamp_resolution_scale(2.0, 0.5), 1.0);
        assert_eq!(clamp_resolution_scale(0.1, 0.5), 0.25);
        assert_eq!(clamp_resolution_scale(0.75, 0.5), 0.75);
        assert_eq!(clamp_resolution_scale(f32::NAN, 0.5), 0.5);
    }

    #[test]
    fn lens_scale_handle_is_shared_and_clamped() {
        let a = LensScale::new(2.0);
        assert_eq!(a.get(), 1.0);
        let b = a.clone();
        b.set(0.1);
        assert_eq!(a.get(), 0.25);
        a.set(f32::NAN);
        assert_eq!(b.get(), 0.25);
        a.set(0.75);
        assert_eq!(b.get(), 0.75);
        assert_eq!(LensScale::new(f32::NAN).get(), DEFAULT_RESOLUTION_SCALE);
    }

    /// Review I3: the pass writes only its traced pixels. With no lens, a lens too far to bend
    /// light by half a pixel, or one beyond the trace reach, it writes nothing at all, so the
    /// background (black space in sky views) stays. When it traces, it writes the rectangle.
    #[test]
    fn lens_pass_writes_only_where_it_traces() {
        let size = UVec2::new(W, H);
        let (_, none) = frame_params(None, Mat4::IDENTITY, proj(), size, 0.5);
        assert!(!none.writes_pixels());
        for r in [1e6, 1e9] {
            let lens = lens_at(DVec3::new(0.0, 0.0, r));
            let (_, p) = frame_params(Some(&lens), Mat4::IDENTITY, proj(), size, 0.5);
            assert!(!p.writes_pixels(), "{r} M");
        }
        // A coarse frame from 4000 M traces a rectangle around the hole, not the whole frame.
        let (w, h) = (320, 180);
        let small = reversed_z_infinite_proj(60f32.to_radians(), w as f32 / h as f32, 0.1);
        let lens = lens_at(DVec3::new(0.0, 0.0, 4000.0));
        let (_, p) = frame_params(Some(&lens), Mat4::IDENTITY, small, UVec2::new(w, h), 1.0);
        assert!(p.writes_pixels());
        assert!(p.rect[2] < w && p.rect[3] < h, "{p:?}");
    }
}
