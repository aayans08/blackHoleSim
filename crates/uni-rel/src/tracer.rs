//! Lens tracer render feature: half-resolution per-pixel geodesic trace on the GPU (M0 Task 9).
//!
//! For the first lens in [`FrameCtx::lenses`] (only one in M0), [`LensTracer`] works out the
//! screen rectangle where light is bent by at least half a pixel, traces one Binet geodesic
//! per sample there (at the resolution scale) and upsamples the result bilinearly into `hdr`
//! and `lens_mask`. It writes nothing outside the rectangle (uni-gfx `targets` pass contract):
//! the background drawn before it stays there, and in M0 that background is the same debug grid
//! ([`crate::DebugSkyBackground`]). The shaders are `shaders/lens_trace.wgsl` (passes),
//! `binet.wgsl` (integrator) and `debug_sky.wgsl`.
//!
//! The brute-force step budget only reaches the hole from cameras within [`TRACE_REACH`]
//! (5000 M). Farther out, nothing is traced and the frame shows the unlensed sky with no
//! shadow; [`lens_coverage`] reports this as [`LensCoverage::BeyondReach`], and the app must
//! put [`LensCoverage::warning`] on the HUD (Global Constraints, Honesty).
//!
//! The lens is treated as a static Schwarzschild hole seen by a static camera; `LensView::spin`
//! is ignored in M0 (Kerr comes in M3, design §5.5).
//!
//! Order: after the background, before the stars (which multiply by `1 − lens_mask`) and the
//! tonemap. When nothing is traced and the camera is outside every horizon, the pass encodes
//! nothing at all.

use std::f64::consts::FRAC_PI_2;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::Context as _;
use glam::{DVec3, Mat4, UVec2};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT, LensView,
    PassTimer, RenderFeature, ShaderLib,
};

use crate::binet;

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
    /// The camera is on or inside the horizon, so the frame is black ([`inside_horizon`]).
    InsideHorizon,
    /// The lens bends light by at least half a pixel on screen, but the camera is farther than
    /// [`TRACE_REACH`] from it, so the step budget cannot reach the hole. Nothing is traced:
    /// the frame shows the unlensed sky with no shadow and `lens_mask` stays 0. The app must
    /// label this on screen ([`LensCoverage::warning`]).
    BeyondReach,
}

impl LensCoverage {
    /// The HUD warning for this coverage, if it needs one. The app adds it to the HUD's
    /// warnings. `InsideHorizon` has its own HUD flag, so it returns `None` here.
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
/// frame black; the app shows "inside event horizon" (Review Focus 2).
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
///   rectangle.
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

/// Group 1, binding 0: the `LensParams` uniform.
fn params_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size_of::<LensParams>() as u64),
        },
        count: None,
    }
}

fn storage_entry(binding: u32, format: wgpu::TextureFormat) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Group 1 of `cs_trace`: params and the samples it writes.
fn trace_layout_entries() -> [wgpu::BindGroupLayoutEntry; 2] {
    [params_entry(), storage_entry(1, SAMPLES_FORMAT)]
}

/// Group 1 of `cs_resolve`: params, the samples it reads, `hdr` and `lens_mask`.
fn resolve_layout_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    [
        params_entry(),
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        storage_entry(3, HDR_FORMAT),
        storage_entry(4, LENS_MASK_FORMAT),
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
    resolution_scale: LensScale,
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
    /// is the resolution-scale handle the app steers. Shader or pipeline validation errors are
    /// returned, not raised on the device.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        scale: LensScale,
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
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lens params"),
            size: size_of::<LensParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
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
            resolution_scale: scale,
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
    use glam::{DQuat, DVec2};
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
        for name in ["binet_trace", "static_observer_ray", "debug_sky"] {
            assert!(has_fn(name), "missing fn {name}");
        }
        let entries: Vec<&str> = m.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(entries, ["cs_trace", "cs_resolve"]);
    }

    #[test]
    fn lens_params_layout_matches_wgsl() {
        let module = lens_trace_module();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("LensParams"))
            .expect("struct LensParams in lens_trace.wgsl");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("LensParams is not a struct");
        };
        assert_eq!(size_of::<LensParams>(), 64);
        assert_eq!(layouter[handle].size as usize, size_of::<LensParams>());
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust: Vec<(String, usize)> = [
            ("cam", offset_of!(LensParams, cam)),
            ("rect", offset_of!(LensParams, rect)),
            ("grid", offset_of!(LensParams, grid)),
            ("scale", offset_of!(LensParams, scale)),
        ]
        .into_iter()
        .map(|(n, o)| (n.to_owned(), o))
        .collect();
        assert_eq!(wgsl, rust);
    }

    /// What a binding is, as a comparable string, from the WGSL side.
    fn wgsl_binding_kind(module: &naga::Module, var: &naga::GlobalVariable) -> String {
        match (&var.space, &module.types[var.ty].inner) {
            (naga::AddressSpace::Uniform, _) => "uniform".into(),
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
                min_binding_size,
                ..
            } => {
                assert_eq!(
                    min_binding_size.map(|s| s.get()),
                    Some(size_of::<LensParams>() as u64)
                );
                "uniform".into()
            }
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

    /// Each entry point uses exactly the group-1 bindings of its layout (plus the camera at
    /// group 0, binding 0), with matching types, so pipeline creation cannot fail on layout.
    #[test]
    fn bind_group_layouts_match_shader_bindings() {
        let module = lens_trace_module();
        let caps = naga::valid::Capabilities::default() | naga::valid::Capabilities::SHADER_FLOAT16;
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap();
        let layouts: [(&str, Vec<wgpu::BindGroupLayoutEntry>); 2] = [
            ("cs_trace", trace_layout_entries().to_vec()),
            ("cs_resolve", resolve_layout_entries().to_vec()),
        ];
        for (i, (name, entries)) in layouts.iter().enumerate() {
            assert_eq!(module.entry_points[i].name, *name);
            let used = info.get_entry_point(i);
            let mut wgsl: Vec<(u32, String)> = Vec::new();
            for (h, var) in module.global_variables.iter() {
                if used[h].is_empty() {
                    continue;
                }
                let rb = var.binding.as_ref().expect("resource binding");
                if rb.group == 0 {
                    assert_eq!((rb.binding, var.name.as_deref()), (0, Some("camera")));
                    continue;
                }
                assert_eq!(rb.group, 1, "{name}: {:?}", var.name);
                wgsl.push((rb.binding, wgsl_binding_kind(&module, var)));
            }
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
