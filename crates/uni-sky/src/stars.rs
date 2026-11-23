//! Star field render feature: catalogue stars as instanced PSF splats (M0 Task 8).
//!
//! [`StarField`] uploads one [`StarInstance`] per catalogue star and draws them all with one
//! instanced draw of a six-vertex quad, blended additively into `hdr`. The projection, PSF,
//! brightness and culling rules live in `shaders/stars.wgsl`. [`crate::psf`] and
//! [`project_star`], [`off_axis_cos`], [`pixel_solid_angle`], [`pixel_solid_angle_off_axis`],
//! [`star_display_energy`] and [`star_display_energy_off_axis`] here are their CPU twins.

use std::sync::OnceLock;

use anyhow::Context as _;
use bytemuck::Zeroable as _;
use glam::{Mat4, Vec2, Vec3};
use uni_core::{blackbody_linear_srgb, star_illuminance_lux};
use uni_data::CatalogStar;
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer, RenderFeature,
    ShaderLib,
};
use wgpu::util::DeviceExt as _;

use crate::stars3d::SkyDepth;

/// Path of the star shader in the [`ShaderLib`].
pub const STARS_SHADER: &str = "sky/stars.wgsl";

/// The BSC5 star pass's depth: none, as the 3-D star pass's (lead ruling B9c,
/// `stars3d::SkyDepth::MaskOnly`). `star_sky_factor`'s (1 − r)(1 − g) of `lens_mask` occludes
/// the stars, so a star behind a body's anti-aliased edge pixel keeps its (1 − coverage) share.
pub(crate) const STARS_DEPTH: SkyDepth = SkyDepth::MaskOnly;

/// Default star exposure [EV at ISO 100]: an artistic choice, the "Eye-adapted stars" dual
/// exposure of design §7 item 4. The HUD exposes it as the star EV slider.
///
/// On screen, a star's pixels are E·w/(Ω(θ)·1.2·2^star_ev100), where w is the PSF weight and
/// Ω(θ) = Ω_c·cos³θ the solid angle of a pixel θ off the optical axis. The shader writes exactly
/// that into the pre-exposed HDR target (E/Ω(θ) × exposure_scale(star_ev100), uni-gfx
/// `camera.wgsl`), and the tonemap applies AgX to it with no further exposure, so the scene EV
/// never touches a star. The HDR values are display values: Sirius's peak, 35.36 here, stays far
/// inside f16 at any scene EV (`star_hdr_values_ignore_the_scene_ev_and_fit_f16`).
/// Calibrated on the optical axis at 1920×1080 with fov_y = 60° against uni-post's AgX, which
/// clamps log2 of the exposed value to [−12.47393, 4.026069]: black is 2^−12.47393 =
/// 0.18·2^−10 ≈ 1.76e-4 and white 2^4.026069 = 0.18·2^6.5 ≈ 16.3. The peaks listed are core
/// peaks: the energy times the Gaussian core's centre weight, for a star centred on its pixel. For
/// V ≤ 1 the shader moves 2% of the energy into the wide halo, so the rendered centre is about
/// 1.8% lower. The 8-bit codes are uni-post's `agx` of a grey pixel at the peak, as the sRGB
/// target stores it. A footprint radius is where the PSF falls below
/// [`crate::psf::VISIBILITY_CUTOFF`], 2.55 stops under AgX black.
/// - V = 6.5, the naked-eye limit, peaks at 2.315e-2: 7.04 stops above AgX black, about
///   46/255. It keeps the minimum 3 px footprint.
/// - V = 0 peaks at 9.215 (about 249/255), a white core; its glare footprint reaches 13.75 px.
/// - Sirius (V = −1.46) peaks at 35.36, past AgX white (255/255); its glare footprint reaches
///   18.31 px.
///
/// The rule: V = 6.5 is clearly visible on black (≥ about 7 stops above AgX black); V ≤ 1 stars
/// saturate to a white core with a visible glare. The M0 gate's capture sweep chose −4.0: at the
/// earlier 1.0, V = 6.5 sat only 2.04 stops above black (about 3/255) and the sky looked black.
///
/// Off the axis, pixel values are 1/cos³θ higher (+1.6 EV at the side edges and +1.9 EV in the
/// corners at 60° and 16:9). Off-axis pixels subtend less sky, so this keeps each star as bright
/// relative to the sky around it as at the centre.
///
/// Each +1 EV halves star brightness. `default_star_exposure_calibration` pins the peaks and
/// radii above.
pub const DEFAULT_STAR_EV100: f32 = -4.0;

/// Stars whose clip-space w is at most this are behind the camera or on its plane, and are
/// culled. Directions are unit vectors, so an on-screen star has w ≥ cos(fov/2), far above this.
pub const MIN_CLIP_W: f32 = 1.0e-6;

/// Relative luminance weights of linear Rec. 709 / sRGB: ITU-R BT.709-6 (2015), item 3.2.
const REC709_Y: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// One star as the GPU sees it: the twin of `struct StarInstance` in `stars.wgsl`
/// (32 bytes, two vec4s; `star_instance_layout_matches_wgsl` checks the layout with naga).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarInstance {
    /// xyz: unit direction to the star in root (ICRS) axes. w: illuminance at the observer
    /// [lux], finite and ≥ 0.
    pub dir: [f32; 4],
    /// Linear sRGB colour with luminance Y = 1. w is unused (0).
    pub color: [f32; 4],
}

/// Packs a catalogue star for the GPU.
///
/// - Illuminance is `uni_core::star_illuminance_lux(v_mag)`.
/// - The colour is the Planck colour at `teff_k`, normalised to Y = 1, read from a 256-entry
///   table built once from `uni_core::blackbody_linear_srgb`.
/// - Bad input never reaches the GPU as NaN. A zero or non-finite direction gives an all-zero
///   `dir` (no light, culled). A NaN or overflowing magnitude gives 0 lux. A non-positive or
///   non-finite temperature gives white.
pub fn star_instance(s: &CatalogStar) -> StarInstance {
    let dir = s.dir_icrs.normalize_or_zero();
    // f64 → f32 overflows to ∞ for absurd magnitudes, and a NaN magnitude gives NaN.
    let lux = star_illuminance_lux(f64::from(s.v_mag)) as f32;
    let lux = if lux.is_finite() && dir != Vec3::ZERO {
        lux
    } else {
        0.0
    };
    StarInstance {
        dir: dir.extend(lux).to_array(),
        color: planck_lut().colour(s.teff_k).extend(0.0).to_array(),
    }
}

/// Registers `stars.wgsl` in `lib` under [`STARS_SHADER`].
pub fn register_shaders(lib: &mut ShaderLib) {
    lib.add(STARS_SHADER, include_str!("../shaders/stars.wgsl"));
}

/// Solid angle Ω_c of one pixel on the optical axis [sr]: (2·tan(fov_y/2) / H)², the small-angle
/// value at the image centre. `fov_y` is the full vertical field of view [rad] and `height_px`
/// the image height. Off the axis use [`pixel_solid_angle_off_axis`]. Twin of
/// `pixel_solid_angle` in `stars.wgsl`.
pub fn pixel_solid_angle(fov_y: f32, height_px: f32) -> f32 {
    let side = 2.0 * (0.5 * fov_y).tan() / height_px;
    side * side
}

/// Solid angle [sr] of a pixel whose centre lies θ off the optical axis, with
/// `cos_theta` = cos θ (from [`off_axis_cos`]): Ω(θ) = Ω_c·cos³θ.
///
/// On a rectilinear (pinhole) image plane at distance d, that pixel is d/cosθ from the eye and
/// tilted by θ to the line of sight, so it subtends dA·cosθ/(d/cosθ)² = (dA/d²)·cos³θ. This uses
/// dω = dA·cosθ/r² (Pharr, Jakob & Humphreys, *Physically Based Rendering*, 3rd ed. 2016,
/// §5.5.3). `off_axis_pixel_solid_angle_matches_exact_pixel` checks it against exact pixel
/// solid angles. Twin of `pixel_solid_angle_off_axis` in `stars.wgsl`.
pub fn pixel_solid_angle_off_axis(fov_y: f32, height_px: f32, cos_theta: f32) -> f32 {
    pixel_solid_angle(fov_y, height_px) * cos_theta * cos_theta * cos_theta
}

/// Cosine of the angle θ between `dir` (root axes) and the camera's optical axis, view-space −Z:
/// cosθ = −z_view/|dir_view|, with dir_view = view_rot·dir. Under the reversed-Z infinite
/// projection, −z_view is also clip w. Twin of `cos_theta` in `vs_main` of `stars.wgsl`.
pub fn off_axis_cos(view_rot: Mat4, dir: Vec3) -> f32 {
    let v = (view_rot * dir.extend(0.0)).truncate();
    -v.z / v.length()
}

/// A star's display-normalised energy on the optical axis: [`star_display_energy_off_axis`]
/// with θ = 0, i.e. E/(Ω_c·1.2·2^star_ev100).
pub fn star_display_energy(lux: f32, fov_y: f32, height_px: f32, star_ev100: f32) -> f32 {
    star_display_energy_off_axis(lux, fov_y, height_px, star_ev100, 1.0)
}

/// A star's display-normalised energy θ off the optical axis (`cos_theta` = cos θ): the sum of
/// its pre-exposed HDR pixel values, E/(Ω(θ)·1.2·2^star_ev100) with
/// Ω(θ) = [`pixel_solid_angle_off_axis`]. That is E/Ω(θ) times the star pre-exposure
/// exposure_scale(star_ev100) = 1/(1.2·2^star_ev100) (uni-gfx `exposure_scale`; Lagarde & de
/// Rousiers 2014 §4.1.1). The tonemap applies no further exposure, and the scene EV plays no
/// part.
///
/// Dividing by the pixel's own solid angle keeps a star consistent with the sky radiance in the
/// same pixels: off-axis pixels subtend less solid angle, so the star's energy is 1/cos³θ larger
/// there. Twin of `energy` in `vs_main` of `stars.wgsl`.
pub fn star_display_energy_off_axis(
    lux: f32,
    fov_y: f32,
    height_px: f32,
    star_ev100: f32,
    cos_theta: f32,
) -> f32 {
    // Same order of operations as the shader: lux · star_pre() / Ω(θ).
    lux * uni_gfx::exposure_scale(star_ev100)
        / pixel_solid_angle_off_axis(fov_y, height_px, cos_theta)
}

/// Framebuffer position [px] of the star in direction `dir` (root axes), or `None` when the star
/// is behind the camera, on its plane, or projects to a non-finite position. Twin of the
/// projection in `vs_main` of `stars.wgsl`.
///
/// `view_proj` is `proj * view_rot`. Stars are at infinity, so `dir` goes in with w = 0. Under
/// the reversed-Z infinite projection that lands on depth 0 (the far plane), with clip w = −z_view.
/// x runs right and y down, and pixel k spans [k, k + 1]. A star in the direction
/// `pixel_dir(k)` (uni-gfx `camera.wgsl`) lands on k + 0.5.
pub fn project_star(view_proj: Mat4, dir: Vec3, size_px: Vec2) -> Option<Vec2> {
    let clip = view_proj * dir.extend(0.0);
    if !(clip.w > MIN_CLIP_W) {
        return None;
    }
    let ndc = Vec2::new(clip.x, clip.y) / clip.w;
    let p = Vec2::new(
        0.5 * (ndc.x + 1.0) * size_px.x,
        0.5 * (1.0 - ndc.y) * size_px.y,
    );
    p.is_finite().then_some(p)
}

/// The star field render feature (`name() == "stars"`).
///
/// Group 0 is the renderer's (camera and exposure, [`FrameBindings::camera`]). Group 1 holds the
/// instance buffer and the lens mask. It is rebuilt in `resize`, because the targets are
/// recreated then. The pass runs after the lens pass and multiplies by `1 − lens_mask`
/// (uni-gfx `targets` pass contract).
pub struct StarField {
    pipeline: wgpu::RenderPipeline,
    star_bgl: wgpu::BindGroupLayout,
    instance_buffer: wgpu::Buffer,
    /// Group 1 for the current targets. `None` until the first `prepare` or `resize`.
    star_bind_group: Option<wgpu::BindGroup>,
    count: u32,
}

impl StarField {
    /// Uploads `stars` and builds the pipeline. `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`. An empty slice is allowed (an empty sky): the
    /// feature then draws nothing.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        stars: &[CatalogStar],
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let count =
            u32::try_from(stars.len()).context("more stars than one instanced draw can hold")?;
        let mut instances: Vec<StarInstance> = stars.iter().map(star_instance).collect();
        if instances.is_empty() {
            // A storage binding cannot be empty, so an empty sky keeps one dark instance and
            // draws none.
            instances.push(StarInstance::zeroed());
        }
        let bytes: &[u8] = bytemuck::cast_slice(&instances);
        let limit = device.limits().max_storage_buffer_binding_size;
        anyhow::ensure!(
            bytes.len() as u64 <= limit,
            "{count} stars need {} B of storage; the device allows {limit} B",
            bytes.len()
        );

        register_shaders(lib);
        let module = lib.module(device, STARS_SHADER)?;

        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("star instances"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let star_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stars bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<StarInstance>() as u64,
                        ),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    // The vertex stage reads the attachment size and the fragment stage the mask.
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // Read with textureLoad only, so it is bound as unfilterable.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stars layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&star_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stars"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend: Some(wgpu::BlendState::ADDITIVE),
                    // Light adds to colour only; hdr's alpha is left alone.
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            // Triangle list, no face culling (the quad's winding depends on the y flip).
            primitive: wgpu::PrimitiveState::default(),
            // Stars sit on the far plane (depth 0) with no depth test: lens_mask's
            // (1 − r)(1 − g) occludes them (lead ruling B9c, STARS_DEPTH).
            depth_stencil: STARS_DEPTH.state(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the star pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            star_bgl,
            instance_buffer,
            star_bind_group: None,
            count,
        })
    }

    /// Number of catalogue stars uploaded. Every one is drawn, and off-screen or behind-camera
    /// stars are culled per frame in the vertex shader.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// (Re)creates group 1 against `targets` (the lens mask view changes with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.star_bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stars bg"),
            layout: &self.star_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.instance_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
            ],
        }));
    }
}

impl RenderFeature for StarField {
    fn name(&self) -> &'static str {
        "stars"
    }

    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        if self.star_bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
    }

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.count == 0 {
            return;
        }
        if self.star_bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(star_bind_group) = &self.star_bind_group else {
            return;
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("stars"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: STARS_DEPTH.attachment(&targets.depth_view),
            timestamp_writes: timer.render_writes("stars"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, star_bind_group, &[]);
        pass.draw(0..6, 0..self.count);
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

/// Entries in the Planck colour table.
const LUT_LEN: usize = 256;
/// Temperature range of the table [K], log-spaced. It runs from 1000 K (cool brown dwarfs) to
/// 40 000 K (O stars). BSC5 temperatures from B−V clamped to [−0.4, 2.0] fall in
/// 3169–21 708 K (Ballesteros 2012), well inside it.
const LUT_T_MIN_K: f64 = 1000.0;
const LUT_T_MAX_K: f64 = 40_000.0;

/// Blackbody colours (linear sRGB, Y = 1), log-spaced in temperature.
///
/// `uni_core::blackbody_linear_srgb` sums Planck × CMF over 401 wavelengths, which is too slow
/// to call once per star for large catalogues (and, later, procedural stars). This table calls
/// it 256 times, once, and interpolates linearly in ln T. Steps are about 1.5% in T; the error is
/// below 1e-3 per channel (`planck_lut_matches_blackbody`).
struct PlanckLut([Vec3; LUT_LEN]);

impl PlanckLut {
    fn new() -> Self {
        let span = (LUT_T_MAX_K / LUT_T_MIN_K).ln();
        Self(std::array::from_fn(|i| {
            let t = LUT_T_MIN_K * (span * i as f64 / (LUT_LEN - 1) as f64).exp();
            unit_luminance(blackbody_linear_srgb(t))
        }))
    }

    /// Colour at `teff_k`, clamped to the table's range. A non-positive or non-finite
    /// temperature gives white (1, 1, 1).
    fn colour(&self, teff_k: f32) -> Vec3 {
        if !(teff_k > 0.0 && teff_k.is_finite()) {
            return Vec3::ONE;
        }
        let t = f64::from(teff_k).clamp(LUT_T_MIN_K, LUT_T_MAX_K);
        let x = (t / LUT_T_MIN_K).ln() / (LUT_T_MAX_K / LUT_T_MIN_K).ln() * (LUT_LEN - 1) as f64;
        let i = (x.floor() as usize).min(LUT_LEN - 2);
        let f = (x - i as f64) as f32;
        self.0[i].lerp(self.0[i + 1], f)
    }
}

/// The shared Planck table, built on first use.
fn planck_lut() -> &'static PlanckLut {
    static LUT: OnceLock<PlanckLut> = OnceLock::new();
    LUT.get_or_init(PlanckLut::new)
}

/// `c` scaled to luminance Y = 1. `blackbody_linear_srgb` has Y = 1 before it clamps negative
/// channels to 0, and the clamp raises Y slightly for the reddest and bluest stars. A black
/// input gives white.
fn unit_luminance(c: Vec3) -> Vec3 {
    let y = c.dot(REC709_Y);
    if y > 0.0 { c / y } else { Vec3::ONE }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::psf::{
        CORE_SIGMA_PX, HALO_ALPHA_PX, HALO_BETA, HALO_ENERGY_FRACTION, HALO_MIN_LUX, MAX_RADIUS_PX,
        MIN_RADIUS_PX, VISIBILITY_CUTOFF, footprint_radius_px, halo_fraction, halo_weight,
        psf_weights,
    };
    use core::mem::{offset_of, size_of};
    use glam::{DVec3, Quat, Vec4};
    use uni_core::{blackbody_linear_srgb, star_illuminance_lux};
    use uni_gfx::{reversed_z_infinite_proj, validate_wgsl};
    use wgpu::naga;

    #[test]
    fn sirius_instance() {
        // Sirius (HR 2491): RA 101.2872°, Dec −16.7161° (J2000), V = −1.46, Teff ≈ 9940 K.
        let (ra, dec) = (101.2872f32.to_radians(), (-16.7161f32).to_radians());
        let dir = Vec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin());
        let s = CatalogStar {
            dir_icrs: dir,
            v_mag: -1.46,
            teff_k: 9940.0,
            hr: 2491,
        };
        let inst = star_instance(&s);
        let want = star_illuminance_lux(-1.46) as f32;
        assert!(
            (inst.dir[3] - want).abs() <= want * 1e-6,
            "{} vs {want}",
            inst.dir[3]
        );
        assert!(inst.color[2] > inst.color[0], "{:?}", inst.color);
        assert!((Vec3::from_slice(&inst.dir[..3]) - dir).length() < 1e-6);
        let y = Vec3::from_slice(&inst.color[..3]).dot(REC709_Y);
        assert!((y - 1.0).abs() < 1e-5, "Y = {y}");
        assert_eq!(inst.color[3], 0.0);
    }

    #[test]
    fn star_instance_sanitises_bad_input() {
        let bad = CatalogStar {
            dir_icrs: Vec3::ZERO,
            v_mag: f32::NAN,
            teff_k: f32::NAN,
            hr: 0,
        };
        let inst = star_instance(&bad);
        assert_eq!(inst.dir, [0.0; 4]);
        assert_eq!(inst.color, [1.0, 1.0, 1.0, 0.0]);
        // A non-unit direction is normalised; a NaN magnitude gives no light.
        let s = CatalogStar {
            dir_icrs: Vec3::new(0.0, 0.0, 2.0),
            v_mag: f32::NAN,
            teff_k: 5772.0,
            hr: 1,
        };
        assert_eq!(star_instance(&s).dir, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn star_instance_size_32() {
        assert_eq!(size_of::<StarInstance>(), 32);
    }

    fn stars_module() -> naga::Module {
        let mut lib = ShaderLib::with_common();
        register_shaders(&mut lib);
        let src = lib.compose(STARS_SHADER).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}"))
    }

    #[test]
    fn stars_shader_validates() {
        let m = stars_module();
        let stage = |name: &str| {
            m.entry_points
                .iter()
                .find(|e| e.name == name)
                .map(|e| e.stage)
        };
        assert_eq!(stage("vs_main"), Some(naga::ShaderStage::Vertex));
        assert_eq!(stage("fs_main"), Some(naga::ShaderStage::Fragment));
    }

    #[test]
    fn star_instance_layout_matches_wgsl() {
        let m = stars_module();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("StarInstance"))
            .expect("struct StarInstance in stars.wgsl");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("StarInstance is not a struct");
        };
        assert_eq!(layouter[handle].size as usize, size_of::<StarInstance>());
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust = vec![
            ("dir".to_owned(), offset_of!(StarInstance, dir)),
            ("color".to_owned(), offset_of!(StarInstance, color)),
        ];
        assert_eq!(wgsl, rust);

        // group(1): binding 0 is the read-only instance array, binding 1 the lens mask.
        let binding = |name: &str| {
            m.global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .and_then(|(_, v)| v.binding.clone())
        };
        let rb = |binding| naga::ResourceBinding { group: 1, binding };
        assert_eq!(binding("stars"), Some(rb(0)));
        assert_eq!(binding("lens_mask"), Some(rb(1)));
        let (_, stars) = m
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("stars"))
            .unwrap();
        assert_eq!(
            stars.space,
            naga::AddressSpace::Storage {
                access: naga::StorageAccess::LOAD
            }
        );
    }

    /// The value of `const name: f32 = <literal>;` in `m`.
    fn wgsl_f32_const(m: &naga::Module, name: &str) -> f32 {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no const {name} in stars.wgsl"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => v,
            ref e => panic!("const {name} is not an f32 literal: {e:?}"),
        }
    }

    #[test]
    fn shader_constants_match_cpu() {
        let m = stars_module();
        for (name, cpu) in [
            ("CORE_SIGMA_PX", CORE_SIGMA_PX),
            ("HALO_ENERGY_FRACTION", HALO_ENERGY_FRACTION),
            ("HALO_MIN_LUX", HALO_MIN_LUX),
            ("HALO_ALPHA_PX", HALO_ALPHA_PX),
            ("HALO_BETA", HALO_BETA),
            ("MIN_RADIUS_PX", MIN_RADIUS_PX),
            ("MAX_RADIUS_PX", MAX_RADIUS_PX),
            ("VISIBILITY_CUTOFF", VISIBILITY_CUTOFF),
            ("MIN_CLIP_W", MIN_CLIP_W),
        ] {
            assert_eq!(wgsl_f32_const(&m, name), cpu, "{name}");
        }
    }

    #[test]
    fn planck_lut_matches_blackbody() {
        let lut = planck_lut();
        for t in [
            1000.0, 1500.0, 2345.0, 3169.0, 4500.0, 5772.0, 7000.0, 9940.0, 15000.0, 21708.0,
            33000.0, 40000.0,
        ] {
            let want = unit_luminance(blackbody_linear_srgb(f64::from(t)));
            let got = lut.colour(t);
            assert!(
                (got - want).abs().max_element() < 1e-3,
                "{t} K: {got} vs {want}"
            );
            assert!((got.dot(REC709_Y) - 1.0).abs() < 1e-5);
        }
        // Outside the table the colour clamps to its ends; an unusable temperature is white.
        assert_eq!(lut.colour(500.0), lut.colour(1000.0));
        assert_eq!(lut.colour(1e6), lut.colour(40_000.0));
        assert_eq!(lut.colour(f32::NAN), Vec3::ONE);
        assert_eq!(lut.colour(-5.0), Vec3::ONE);
    }

    /// CPU copy of `pixel_dir` in uni-gfx's common/camera.wgsl (pixel index → root direction).
    fn pixel_dir(inv_view_proj: Mat4, size: Vec2, pix: Vec2) -> Vec3 {
        let uv = (pix + 0.5) / size;
        let p = inv_view_proj * Vec4::new(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y, 1.0, 1.0);
        (p.truncate() / p.w).normalize()
    }

    #[test]
    fn stars_project_onto_the_pixel_dir_convention() {
        let size = Vec2::new(1920.0, 1080.0);
        // Root → camera rotation, yawed and pitched so no axis is trivial.
        let cam_to_root = Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3);
        let view = Mat4::from_quat(cam_to_root.inverse());
        let view_proj = reversed_z_infinite_proj(60f32.to_radians(), size.x / size.y, 0.1) * view;
        let inv = view_proj.inverse();
        for pix in [
            Vec2::new(0.0, 0.0),
            Vec2::new(959.0, 539.0),
            Vec2::new(1919.0, 1079.0),
            Vec2::new(123.0, 1000.0),
        ] {
            let dir = pixel_dir(inv, size, pix);
            let p = project_star(view_proj, dir, size).expect("in front of the camera");
            // The star through a pixel's centre direction lands on that pixel's centre.
            assert!((p - (pix + 0.5)).length() < 5e-3, "{pix} → {p}");
            // A direction (w = 0) lands exactly on the reversed-Z far plane, in front of the
            // camera.
            let clip = view_proj * dir.extend(0.0);
            assert_eq!(clip.z, 0.0);
            assert!(clip.w > 0.0);
        }
    }

    #[test]
    fn stars_behind_the_camera_are_culled() {
        // No view rotation: the camera looks along −Z with +Y up.
        let view_proj = reversed_z_infinite_proj(1.0, 800.0 / 600.0, 0.1);
        let size = Vec2::new(800.0, 600.0);
        assert_eq!(
            project_star(view_proj, Vec3::NEG_Z, size),
            Some(Vec2::new(400.0, 300.0))
        );
        // Row 0 is the top: a star above the axis has a smaller y.
        let up = project_star(view_proj, Vec3::new(0.0, 0.1, -1.0).normalize(), size).unwrap();
        assert!(up.y < 300.0 && (up.x - 400.0).abs() < 1e-3, "{up}");
        assert_eq!(project_star(view_proj, Vec3::Z, size), None);
        assert_eq!(project_star(view_proj, Vec3::X, size), None); // on the camera plane
        assert_eq!(
            project_star(view_proj, Vec3::new(f32::NAN, 0.0, -1.0), size),
            None
        );
    }

    #[test]
    fn default_star_exposure_calibration() {
        // Calibration point: 1920×1080, fov_y = 60°.
        let (fov_y, h) = (60f32.to_radians(), 1080.0);
        let omega = pixel_solid_angle(fov_y, h);
        assert!((omega - 1.143_118e-6).abs() < 1e-11, "{omega}");
        let lux = |v_mag: f64| star_illuminance_lux(v_mag) as f32;
        let energy = |v_mag: f64| star_display_energy(lux(v_mag), fov_y, h, DEFAULT_STAR_EV100);
        let radius = |v_mag: f64| footprint_radius_px(energy(v_mag), halo_fraction(lux(v_mag)));
        // Core centre-pixel weight of a star centred on its pixel: the doc's "core peak".
        let centre = psf_weights(Vec2::splat(0.5), CORE_SIGMA_PX, 3)[24];
        let peak = |v_mag: f64| energy(v_mag) * centre;
        // The centre pixel the shader renders: the core minus the halo's share, plus the halo.
        let rendered = |v_mag: f64| {
            let hf = halo_fraction(lux(v_mag));
            energy(v_mag) * ((1.0 - hf) * centre + hf * halo_weight(0.0, radius(v_mag)))
        };
        // uni-post's AgX (agx.rs) clamps log2 of the exposed value to [−12.47393, 4.026069], so
        // black is 2^−12.47393 = 0.18·2^−10, and 0.18 is mid-grey. Grey passes the inset
        // matrix unchanged (its rows sum to 1).
        let agx_black = (-12.47393f32).exp2();

        // V = 6.5, the naked-eye limit: clearly visible on black, about 7 stops above it (the
        // doc's 7.04 to 0.01 stops), and still on the minimum footprint.
        let faint = (peak(6.5) / agx_black).log2();
        assert!((6.5..=7.5).contains(&faint), "{faint}");
        assert!((faint - 7.04).abs() < 0.01, "{faint}");
        assert_eq!(radius(6.5), MIN_RADIUS_PX);
        assert_eq!(rendered(6.5), peak(6.5)); // no halo, so the core peak is what renders
        // At the earlier default, 1.0, it sat only 2.04 stops above black.
        let old = (star_display_energy(lux(6.5), fov_y, h, 1.0) * centre / agx_black).log2();
        assert!((old - 2.04).abs() < 0.01, "{old}");
        // V ≤ 1: a white core (above 1.0 after exposure, as rendered) and a glare wider than the
        // minimum footprint, within the cap. The halo's 2% share leaves the rendered centre
        // about 1.8% below the core peak.
        for v_mag in [1.0, 0.0, -1.46] {
            let (p, r, shown) = (peak(v_mag), radius(v_mag), rendered(v_mag));
            assert!(shown > 1.0, "V = {v_mag}: rendered centre {shown}");
            assert!(
                r > MIN_RADIUS_PX && r <= MAX_RADIUS_PX,
                "V = {v_mag}: R {r}"
            );
            assert!(
                (1.0 - shown / p - 0.018).abs() < 5e-4,
                "V = {v_mag}: {shown} vs {p}"
            );
        }

        // The doc comment's numbers: peaks to 0.1%, footprint radii to 0.01 px.
        for (v_mag, want_peak, want_radius) in [
            (6.5, 2.315e-2, MIN_RADIUS_PX),
            (0.0, 9.215, 13.75),
            (-1.46, 35.36, 18.31),
        ] {
            let (p, r) = (peak(v_mag), radius(v_mag));
            assert!((p / want_peak - 1.0).abs() < 1e-3, "V = {v_mag}: peak {p}");
            assert!((r - want_radius).abs() < 0.01, "V = {v_mag}: R {r}");
        }
    }

    /// Review I2: stars are pre-exposed with the star EV only. The scene EV never scales a star's
    /// HDR value (at M0 it pushed Sirius past f16 above scene EV ≈ 10.5 and so dimmed it), and
    /// even the slider's brightest star exposure keeps Sirius inside f16.
    #[test]
    fn star_hdr_values_ignore_the_scene_ev_and_fit_f16() {
        let (fov_y, h) = (60f32.to_radians(), 1080.0);
        let centre = psf_weights(Vec2::splat(0.5), CORE_SIGMA_PX, 3)[24];
        let sirius = star_illuminance_lux(-1.46) as f32;
        // The pre-exposed peak is a display value: 35.36 at the default star EV.
        let peak = |star_ev: f32| star_display_energy(sirius, fov_y, h, star_ev) * centre;
        assert!((peak(DEFAULT_STAR_EV100) / 35.36 - 1.0).abs() < 1e-3);
        // EV100 −10 is the HUD slider's brightest star exposure: 64× the default.
        assert!(peak(-10.0) < uni_gfx::HDR_MAX, "{}", peak(-10.0));
        assert!(peak(-10.0) > 1000.0);
        // The shader's vertex stage reads the star pre-exposure and never the scene's.
        let m = stars_module();
        let func = |name: &str| {
            m.functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .map(|(h, _)| h)
                .unwrap_or_else(|| panic!("no fn {name}"))
        };
        let vs = &m
            .entry_points
            .iter()
            .find(|e| e.name == "vs_main")
            .expect("vs_main")
            .function;
        assert!(calls(&vs.body, func("star_pre")));
        assert!(!calls(&vs.body, func("scene_pre")));
    }

    /// The camera → root rotation used by the off-axis tests: yawed and pitched, so no axis is
    /// trivial.
    fn tilted_camera() -> Quat {
        Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.3)
    }

    #[test]
    fn off_axis_solid_angle_factor() {
        let (fov_y, size) = (60f32.to_radians(), Vec2::new(1920.0, 1080.0));
        let cam_to_root = tilted_camera();
        let view = Mat4::from_quat(cam_to_root.inverse());
        // The optical axis, and a direction 30° off it diagonally (both screen axes non-zero).
        let axis = cam_to_root * Vec3::NEG_Z;
        let tilt = cam_to_root * Vec3::new(1.0, 1.0, 0.0).normalize();
        let off = Quat::from_axis_angle(tilt, 30f32.to_radians()) * axis;
        // The off-axis star is on screen, so the factor reaches a visible splat.
        let proj = reversed_z_infinite_proj(fov_y, size.x / size.y, 0.1);
        let p = project_star(proj * view, off, size).expect("in front of the camera");
        assert!(p.cmpge(Vec2::ZERO).all() && p.cmplt(size).all(), "{p}");

        let lux = star_illuminance_lux(2.0) as f32;
        let energy = |dir: Vec3| {
            let cos_theta = off_axis_cos(view, dir);
            star_display_energy_off_axis(lux, fov_y, size.y, DEFAULT_STAR_EV100, cos_theta)
        };
        let ratio = energy(off) / energy(axis);
        // 1/cos³(30°) = 1.5396.
        let want = 1.0 / 30f32.to_radians().cos().powi(3);
        assert!((want - 1.5396).abs() < 1e-4, "{want}");
        assert!((ratio / want - 1.0).abs() < 1e-4, "{ratio} vs {want}");
    }

    #[test]
    fn on_axis_unchanged() {
        let (fov_y, h, ev) = (60f32.to_radians(), 1080.0, DEFAULT_STAR_EV100);
        // The on-axis pixel solid angle before the off-axis fix: (2·tan(fov_y/2)/H)².
        let side = 2.0 * (0.5 * fov_y).tan() / h;
        let omega_c = side * side;
        // An unrotated camera's axis (−Z) has cosθ exactly 1; a rotated camera's own axis has
        // cosθ = 1 to f32 rounding.
        let cos_unrotated = off_axis_cos(Mat4::IDENTITY, Vec3::NEG_Z);
        assert_eq!(cos_unrotated, 1.0);
        let cam_to_root = tilted_camera();
        let view = Mat4::from_quat(cam_to_root.inverse());
        let cos_rotated = off_axis_cos(view, cam_to_root * Vec3::NEG_Z);
        for v_mag in [-1.46, 0.0, 6.5] {
            let lux = star_illuminance_lux(v_mag) as f32;
            // The energy formula before the off-axis fix: E/(Ω_c·1.2·2^star_ev100). The
            // pre-exposed form multiplies by 1/(1.2·2^ev) first, so it agrees to f32 rounding.
            let old = lux / (omega_c * 1.2 * ev.exp2());
            let on_axis = star_display_energy(lux, fov_y, h, ev);
            assert!((on_axis / old - 1.0).abs() < 1e-6, "{on_axis} vs {old}");
            let e = star_display_energy_off_axis(lux, fov_y, h, ev, cos_unrotated);
            assert_eq!(e, on_axis);
            let e = star_display_energy_off_axis(lux, fov_y, h, ev, cos_rotated);
            assert!((e / old - 1.0).abs() < 1e-6, "V = {v_mag}: {e} vs {old}");
        }
    }

    /// Solid angle [sr] of the spherical triangle with unit-vector corners a, b, c: Van Oosterom
    /// & Strackee 1983, "The Solid Angle of a Plane Triangle", IEEE Trans. Biomed. Eng. 30(2), 125.
    fn triangle_solid_angle(a: DVec3, b: DVec3, c: DVec3) -> f64 {
        let num = a.dot(b.cross(c)).abs();
        let den = 1.0 + a.dot(b) + b.dot(c) + c.dot(a);
        2.0 * num.atan2(den)
    }

    #[test]
    fn off_axis_pixel_solid_angle_matches_exact_pixel() {
        // Ω_c·cos³θ at a pixel's centre direction against the exact solid angle of the whole
        // pixel, at the centre, the edge midpoints and the corners of a 1920×1080, 60° frame.
        let (fov_y, size) = (60f32.to_radians(), Vec2::new(1920.0, 1080.0));
        let view = Mat4::from_quat(tilted_camera().inverse());
        let inv = (reversed_z_infinite_proj(fov_y, size.x / size.y, 0.1) * view).inverse();
        // View-space direction through framebuffer point (x, y) [px], on the image plane z = −1.
        let (w, h) = (f64::from(size.x), f64::from(size.y));
        let t = (0.5 * f64::from(fov_y)).tan();
        let tx = t * w / h;
        let corner = |x: f64, y: f64| {
            DVec3::new((2.0 * x / w - 1.0) * tx, (1.0 - 2.0 * y / h) * t, -1.0).normalize()
        };
        for pix in [
            Vec2::new(959.0, 539.0),
            Vec2::new(1919.0, 539.0),
            Vec2::new(959.0, 0.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(1919.0, 1079.0),
        ] {
            let (x, y) = (f64::from(pix.x), f64::from(pix.y));
            let [a, b, c, d] = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                .map(|(dx, dy)| corner(x + dx, y + dy));
            let exact = triangle_solid_angle(a, b, c) + triangle_solid_angle(a, c, d);
            // The pixel's centre direction by uni-gfx's `pixel_dir`, through the f32 twins.
            let cos_theta = off_axis_cos(view, pixel_dir(inv, size, pix));
            let approx = f64::from(pixel_solid_angle_off_axis(fov_y, size.y, cos_theta));
            assert!(
                (approx / exact - 1.0).abs() < 1e-5,
                "{pix}: {approx} vs {exact}"
            );
        }
    }

    /// Whether `block`, or a block nested in it, calls `f`.
    fn calls(block: &naga::Block, f: naga::Handle<naga::Function>) -> bool {
        block.iter().any(|s| match s {
            naga::Statement::Call { function, .. } => *function == f,
            naga::Statement::Block(b) => calls(b, f),
            naga::Statement::If { accept, reject, .. } => calls(accept, f) || calls(reject, f),
            naga::Statement::Loop {
                body, continuing, ..
            } => calls(body, f) || calls(continuing, f),
            naga::Statement::Switch { cases, .. } => cases.iter().any(|c| calls(&c.body, f)),
            _ => false,
        })
    }

    #[test]
    fn shader_uses_off_axis_solid_angle() {
        let m = stars_module();
        let func = |name: &str| {
            m.functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .map(|(h, _)| h)
                .unwrap_or_else(|| panic!("no fn {name} in stars.wgsl"))
        };
        let vs = &m
            .entry_points
            .iter()
            .find(|e| e.name == "vs_main")
            .expect("vs_main")
            .function;
        // vs_main divides by the off-axis solid angle, never the on-axis one directly.
        assert!(calls(&vs.body, func("pixel_solid_angle_off_axis")));
        assert!(!calls(&vs.body, func("pixel_solid_angle")));
    }
}
