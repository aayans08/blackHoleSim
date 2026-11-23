//! The Milky Way band v0 and the environment cubemap: passes "band-update" and "band" (Task B5).
//!
//! **Cubemap.** [`BandCubemap`] renders the galaxy model's band into a 6 × 512² Rgba16Float
//! cubemap with 6 mips ([`ENV_SIZE`], [`ENV_MIPS`], [`ENV_FORMAT`]). Each texel of mip 0 holds
//! `gal_band_radiance(cam_gc, dir_gc, v_lim, footprint)` (uni-procgen's WGSL twin
//! `procgen/galaxy.wgsl`; the footprint is the texel's angular size, [`texel_footprint_rad`], so
//! the model's procedural detail stops at the texel)
//! through the texel's centre, as luminance in units of `uni_gfx::ENV_UNIT_CD_M2` (1e-4 cd/m²,
//! review m10; absolute, not pre-exposed) clamped to [`ENV_MAX_UNITS`]. Mips 1–5 are 2 × 2 box
//! means. Directions are root (ICRS) axes and faces follow the WebGPU convention
//! (`uni_gfx::env::cube_dir`); the model's galactocentric frame is reached with
//! `uni_core::icrs_to_galactic`. The CPU twin of one texel is [`band_texel`]. The lens pass
//! samples the same cubemap ([`BandCubemap::env_map`]) for its escaped rays, so the lensed sky
//! matches the band drawn around it.
//!
//! **Refresh.** All six faces on the first frame the pass runs, so a capture is complete and
//! deterministic from its first frame. Afterwards, once the camera has moved more than
//! max(1 pc, 1 % of its distance to the Galactic centre) since the last build, or the star limit
//! `v_lim` changed, one face per frame for six frames, all from the position the rebuild started
//! at. In the Solar System, 8.3 kpc from the centre, that takes an 83 pc move: the cubemap never
//! refreshes there (B5 brief).
//!
//! **Draw.** [`BandDraw`] draws a full-screen triangle at depth 0 after the opaque passes and the
//! meter (ruling 4): mip 0 in each pixel's direction × `ENV_UNIT_CD_M2 × star_pre()` ×
//! (1 − r)(1 − g)(1 − b) of `lens_mask`, blended additively, depth test `GreaterEqual` without
//! writing (only where no body was drawn), clamped to `HDR_MAX`. [`BandInput::enabled`] `= false`
//! skips both passes.
//!
//! **Honesty.** The band is the galaxy model v0 (a smooth dust layer, no clouds or clusters), so
//! the HUD shows "Milky Way band: galaxy model v0 (real dust map in M4)" while it is on screen
//! (`uni_sim` honesty flag `BandModel`).

use std::sync::Arc;

use anyhow::Context as _;
use glam::{DVec3, Vec2, Vec3};
use uni_core::{ICRS_TO_GALACTIC, icrs_to_galactic};
use uni_data::f16_to_f32;
use uni_gfx::env::cube_dir;
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    DEPTH_FORMAT, ENV_UNIT_CD_M2, EnvMap, FrameBindings, FrameCtx, FrameTargets, GpuContext,
    HDR_FORMAT, HDR_MAX, PassTimer, RenderFeature, ShaderLib, SharedInput,
};
use uni_procgen::galaxy::{GalaxyGpu, GalaxyModel};
use wgpu::util::DeviceExt as _;

use crate::{BAND_DRAW_WGSL, BAND_UPDATE_WGSL};

/// Edge of each cubemap face [texels]: 90° / 512 = 0.18° per texel, about 2 px at a 90° field
/// over 1080 lines. B5's brief had 256 (0.35°, 4–6 px per texel), too coarse for the galaxy
/// model's dust detail (Task B4L); the face costs 4× as much to build, once in the Solar System.
pub const ENV_SIZE: u32 = 512;
/// Mip levels of the cubemap (512 → 16, as B5's 256 → 16).
pub const ENV_MIPS: u32 = 6;
/// Texel format of the cubemap (B5 brief; `uni_gfx::EnvMap`).
pub const ENV_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Largest value written into a texel [`ENV_UNIT_CD_M2` units]: `HDR_MAX`, 6 cd/m², below f16's
/// largest finite value (65504; B5 brief). Twin of `BAND_ENV_MAX_UNITS` in band_update.wgsl.
pub const ENV_MAX_UNITS: f32 = HDR_MAX;

/// Faces of a cube.
const FACES: u32 = 6;
/// Edge of the update and downsample workgroups (8 × 8 × 1 in band_update.wgsl).
const WORKGROUP: u32 = 8;
/// A rebuild needs a move of more than max(`REFRESH_MIN_PC`, `REFRESH_FRACTION` × the camera's
/// distance to the Galactic centre) since the last build (B5 brief).
const REFRESH_MIN_PC: f64 = 1.0;
const REFRESH_FRACTION: f64 = 0.01;
/// Timing labels (the `RenderFeature` names).
const UPDATE_NAME: &str = "band-update";
const DRAW_NAME: &str = "band";

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

/// The update pass's parameters (group 1, binding 1). Twin of `BandUpdateParams` in
/// band_update.wgsl (`band_layouts_match_wgsl`): vec4-sized members only.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct BandUpdateParams {
    /// xyz: camera, galactocentric [pc]; w: v_lim [V mag].
    cam_gc_v_lim: [f32; 4],
    /// Rows of `uni_core::ICRS_TO_GALACTIC` (the galactic x, y, z axes in ICRS); w = 0.
    icrs_to_gal: [[f32; 4]; 3],
    /// x: first face of the batch; y: number of faces; zw: 0.
    faces: [u32; 4],
}

impl BandUpdateParams {
    fn new(batch: &FaceBatch) -> Self {
        let cam = batch.cam_gc_pc.as_vec3();
        Self {
            cam_gc_v_lim: [cam.x, cam.y, cam.z, batch.v_lim],
            icrs_to_gal: ICRS_TO_GALACTIC.map(|[x, y, z]| [x as f32, y as f32, z as f32, 0.0]),
            faces: [batch.first, batch.count, 0, 0],
        }
    }
}

/// The faces one frame (re)builds, and from where.
#[derive(Clone, Copy, Debug, PartialEq)]
struct FaceBatch {
    /// First face (WebGPU layer order).
    first: u32,
    /// Number of faces, from `first` on.
    count: u32,
    /// Camera, galactocentric [pc].
    cam_gc_pc: DVec3,
    /// Faintest V magnitude drawn as single stars.
    v_lim: f32,
}

/// Where a build is taken from.
#[derive(Clone, Copy, Debug, PartialEq)]
struct BuildPoint {
    cam_gc_pc: DVec3,
    v_lim: f32,
}

/// When the cubemap is rebuilt (B5 brief; see the module docs).
#[derive(Default)]
struct RefreshPolicy {
    /// The last complete build; `None` before the first.
    built: Option<BuildPoint>,
    /// A rebuild in progress, one face per frame: where it is taken from and the next face.
    pending: Option<(BuildPoint, u32)>,
}

impl RefreshPolicy {
    /// The faces to build this frame for a camera at `cam_gc_pc` [pc, galactocentric] with star
    /// limit `v_lim`, or `None`. Call once per frame the pass runs.
    fn next(&mut self, cam_gc_pc: DVec3, v_lim: f32) -> Option<FaceBatch> {
        let batch = |at: BuildPoint, first: u32, count: u32| FaceBatch {
            first,
            count,
            cam_gc_pc: at.cam_gc_pc,
            v_lim: at.v_lim,
        };
        if let Some((at, face)) = self.pending {
            if face + 1 == FACES {
                self.built = Some(at);
                self.pending = None;
            } else {
                self.pending = Some((at, face + 1));
            }
            return Some(batch(at, face, 1));
        }
        let here = BuildPoint { cam_gc_pc, v_lim };
        match self.built {
            None => {
                self.built = Some(here);
                Some(batch(here, 0, FACES))
            }
            Some(built) if needs_rebuild(built, here) => {
                self.pending = Some((here, 1));
                Some(batch(here, 0, 1))
            }
            Some(_) => None,
        }
    }
}

/// True when the cubemap built at `built` is stale at `now`: the camera has moved more than
/// max(1 pc, 1 % of its distance to the Galactic centre), or the star limit changed.
fn needs_rebuild(built: BuildPoint, now: BuildPoint) -> bool {
    let threshold_pc = (REFRESH_FRACTION * now.cam_gc_pc.length()).max(REFRESH_MIN_PC);
    (now.cam_gc_pc - built.cam_gc_pc).length() > threshold_pc || now.v_lim != built.v_lim
}

/// Galactocentric position [pc] of `p_icrs_pc` [pc, ICRS, origin SSB], given `origin_gc_pc`,
/// the SSB's galactocentric position: `GalaxyModel::icrs_pc_to_galactocentric` (galaxy.rs).
fn galactocentric(origin_gc_pc: DVec3, p_icrs_pc: DVec3) -> DVec3 {
    icrs_to_galactic(p_icrs_pc) + origin_gc_pc
}

/// Luminance [cd/m²] → environment texel units: ÷ `ENV_UNIT_CD_M2`, clamped to
/// [0, [`ENV_MAX_UNITS`]] per channel. Twin of `band_env_units` in band_update.wgsl.
fn env_units(cd_m2: Vec3) -> Vec3 {
    (cd_m2 / ENV_UNIT_CD_M2).clamp(Vec3::ZERO, Vec3::splat(ENV_MAX_UNITS))
}

/// The uv of the centre of texel (`x`, `y`) of a face.
fn texel_uv(x: u32, y: u32) -> Vec2 {
    (Vec2::new(x as f32, y as f32) + 0.5) / ENV_SIZE as f32
}

/// Angular size [rad] of a mip-0 texel in unit direction `dir` (root axes): a face is the
/// gnomonic projection of the sky onto the plane at distance 1, whose texels span 2/N there and
/// shrink on the sky by m² (m the direction's major-axis component), so 2m²/N (the geometry of
/// `cube_dir`; uni-rel's `lens_env_lod` uses the same). The band's detail fades out below it.
/// Twin of `band_texel_footprint` in band_update.wgsl.
pub fn texel_footprint_rad(dir: Vec3) -> f32 {
    let a = dir.abs();
    let m = a.x.max(a.y).max(a.z);
    2.0 * m * m / ENV_SIZE as f32
}

/// The face and uv a cube sample in direction `dir` reads: the inverse of `cube_dir` (the
/// Vulkan specification's "Cube Map Face Selection" table, which WebGPU follows; the major axis
/// picks the face).
fn cube_face_uv(dir: Vec3) -> (u32, Vec2) {
    let a = dir.abs();
    let (face, sc, tc, rc) = if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 {
            (0, -dir.z, -dir.y, a.x)
        } else {
            (1, dir.z, -dir.y, a.x)
        }
    } else if a.y >= a.z {
        if dir.y > 0.0 {
            (2, dir.x, dir.z, a.y)
        } else {
            (3, dir.x, -dir.z, a.y)
        }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, a.z)
    } else {
        (5, -dir.x, -dir.y, a.z)
    };
    (face, (Vec2::new(sc, tc) / rc + 1.0) * 0.5)
}

/// The environment cubemap and its update pass (`RenderFeature` named "band-update").
///
/// Group 0 is unused. Group 1 is `galaxy`, `band_params` and mip 0 as storage for
/// `cs_band_update`; `band_params`, mip n − 1 and mip n for each `cs_band_downsample` level.
pub struct BandCubemap {
    env: Arc<EnvMap>,
    input: SharedInput<BandInput>,
    /// The SSB's galactocentric position [pc] (`GalaxyModel::icrs_pc_to_galactocentric(0)`).
    origin_gc_pc: DVec3,
    update_pipeline: wgpu::ComputePipeline,
    downsample_pipeline: wgpu::ComputePipeline,
    update_bind_group: wgpu::BindGroup,
    /// Mip n − 1 → mip n, for n = 1 … `ENV_MIPS` − 1.
    downsample_bind_groups: Vec<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
    refresh: RefreshPolicy,
    /// Planned (and its parameters uploaded) in `prepare`; dispatched by the next `encode`.
    batch: Option<FaceBatch>,
}

impl BandCubemap {
    /// Creates the cubemap (6 × [`ENV_SIZE`]², [`ENV_MIPS`] mips, [`ENV_FORMAT`]; zeroed, so it
    /// reads as a black sky until the first build), its cube view and trilinear sampler, and the
    /// update and downsample pipelines for `model`. Registers uni-sky's shaders in `lib`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        model: &GalaxyModel,
        input: SharedInput<BandInput>,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib.module(device, BAND_UPDATE_WGSL)?;
        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("band env cubemap"),
            size: wgpu::Extent3d {
                width: ENV_SIZE,
                height: ENV_SIZE,
                depth_or_array_layers: FACES,
            },
            mip_level_count: ENV_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ENV_FORMAT,
            // COPY_SRC: the `band_update` probe reads every mip back.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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
        // One 2D-array view (all six faces) per mip, for the storage writes and the box filter.
        let mip_views: Vec<wgpu::TextureView> = (0..ENV_MIPS)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("band env mip"),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    base_array_layer: 0,
                    array_layer_count: Some(FACES),
                    ..Default::default()
                })
            })
            .collect();
        let galaxy_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("band galaxy model"),
            contents: bytemuck::bytes_of(&model.gpu()),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("band-update params"),
            size: size_of::<BandUpdateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform = |binding: u32, size: usize| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size as u64),
            },
            count: None,
        };
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: ENV_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
            count: None,
        };
        let params_size = size_of::<BandUpdateParams>();
        let update_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("band-update bgl"),
            entries: &[
                uniform(0, size_of::<GalaxyGpu>()),
                uniform(1, params_size),
                storage(2),
            ],
        });
        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("band downsample bgl"),
            entries: &[
                uniform(1, params_size),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // Read with textureLoad only.
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(4),
            ],
        });
        let pipeline = |label: &str, bgl: &wgpu::BindGroupLayout, entry: &str| {
            // Group 0 (the camera) is not used: the pass is camera-independent but for
            // `band_params`.
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[None, Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let update_pipeline = pipeline("band-update", &update_bgl, "cs_band_update");
        let downsample_pipeline =
            pipeline("band downsample", &downsample_bgl, "cs_band_downsample");
        let update_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("band-update bg"),
            layout: &update_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: galaxy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&mip_views[0]),
                },
            ],
        });
        let downsample_bind_groups = mip_views
            .windows(2)
            .map(|pair| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("band downsample bg"),
                    layout: &downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&pair[0]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&pair[1]),
                        },
                    ],
                })
            })
            .collect();
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the band-update pass: {err}");
        }
        Ok(Self {
            env: Arc::new(EnvMap {
                texture,
                view,
                sampler,
            }),
            input,
            origin_gc_pc: model.icrs_pc_to_galactocentric(DVec3::ZERO),
            update_pipeline,
            downsample_pipeline,
            update_bind_group,
            downsample_bind_groups,
            params_buffer,
            refresh: RefreshPolicy::default(),
            batch: None,
        })
    }

    /// The cubemap, shared with [`BandDraw`] and the lens pass (`uni_rel::LensTracer::new`).
    pub fn env_map(&self) -> Arc<EnvMap> {
        Arc::clone(&self.env)
    }

    /// The faces to build this frame for `input` (see the module docs), with their parameters
    /// uploaded; `None` when the band is off, the input is not finite or nothing is stale.
    fn plan(&mut self, gpu: &GpuContext, input: &BandInput) -> Option<FaceBatch> {
        if !input.enabled {
            return None;
        }
        let cam_gc_pc = galactocentric(self.origin_gc_pc, input.camera_icrs_pc);
        if !cam_gc_pc.is_finite() || !input.v_lim.is_finite() {
            return None; // keep the last build rather than poison the policy
        }
        let batch = self.refresh.next(cam_gc_pc, input.v_lim)?;
        gpu.queue.write_buffer(
            &self.params_buffer,
            0,
            bytemuck::bytes_of(&BandUpdateParams::new(&batch)),
        );
        Some(batch)
    }

    /// Records the planned batch, if any: the band into mip 0 of its faces, then the box filter
    /// down the mips of the same faces, in one compute pass.
    fn encode_batch(&mut self, enc: &mut wgpu::CommandEncoder, timer: &mut PassTimer) {
        let Some(batch) = self.batch.take() else {
            return;
        };
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(UPDATE_NAME),
            timestamp_writes: timer.compute_writes(UPDATE_NAME),
        });
        pass.set_pipeline(&self.update_pipeline);
        pass.set_bind_group(1, &self.update_bind_group, &[]);
        let groups = ENV_SIZE.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, batch.count);
        pass.set_pipeline(&self.downsample_pipeline);
        for (level, bind_group) in (1..ENV_MIPS).zip(&self.downsample_bind_groups) {
            let groups = (ENV_SIZE >> level).div_ceil(WORKGROUP);
            pass.set_bind_group(1, bind_group, &[]);
            pass.dispatch_workgroups(groups, groups, batch.count);
        }
    }
}

impl RenderFeature for BandCubemap {
    fn name(&self) -> &'static str {
        UPDATE_NAME
    }

    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {
        if self.batch.is_none() {
            let input = self.input.get();
            self.batch = self.plan(gpu, &input);
        }
    }

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        _targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        self.encode_batch(enc, timer);
    }
}

/// The band draw pass (`RenderFeature` named "band").
///
/// Group 0 is the renderer's (camera and exposure). Group 1 holds the cubemap, its sampler and
/// the lens mask; it is rebuilt in `resize`, because the targets are recreated then.
pub struct BandDraw {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    env: Arc<EnvMap>,
    input: SharedInput<BandInput>,
    /// Group 1 for the current targets. `None` until the first `prepare` or `resize`.
    bind_group: Option<wgpu::BindGroup>,
    /// [`BandInput::enabled`], read in `prepare`.
    enabled: bool,
}

impl BandDraw {
    /// Builds the pass over `env` ([`BandCubemap::env_map`]). `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`. Registers uni-sky's shaders in `lib`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        env: Arc<EnvMap>,
        input: SharedInput<BandInput>,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let module = lib.module(device, BAND_DRAW_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let texture = |binding: u32, filterable: bool, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension,
                multisampled: false,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("band bgl"),
            entries: &[
                texture(0, true, wgpu::TextureViewDimension::Cube),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // The lens mask, read with textureLoad only.
                texture(2, false, wgpu::TextureViewDimension::D2),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("band layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(DRAW_NAME),
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
            primitive: wgpu::PrimitiveState::default(),
            // The triangle sits on the far plane (depth 0). Reversed-Z GreaterEqual against the
            // cleared 0.0 passes only where no body was drawn (ruling 4); depth is never written.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the band pass: {err}");
        }
        Ok(Self {
            pipeline,
            bgl,
            env,
            input,
            bind_group: None,
            enabled: false,
        })
    }

    /// (Re)creates group 1 against `targets` (the lens mask view changes with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("band bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.env.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.env.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
            ],
        }));
    }
}

impl RenderFeature for BandDraw {
    fn name(&self) -> &'static str {
        DRAW_NAME
    }

    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        self.enabled = self.input.with(|input| input.enabled);
        if self.bind_group.is_none() {
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
        if !self.enabled {
            return;
        }
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(DRAW_NAME),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.hdr_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            // Read-only depth (`depth_ops: None`): tested, never written.
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: timer.render_writes(DRAW_NAME),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

/// The band radiance the update pass writes at texel `uv` of cube face `face` for a camera at
/// `cam_icrs_pc` [pc, ICRS, origin SSB]: `model.band_radiance` in the direction
/// `uni_gfx::env::cube_dir(face, uv)` (root axes, rotated into galactic axes), at the texel's
/// footprint [`texel_footprint_rad`], in cd/m², linear
/// Rec. 709 (not divided by `ENV_UNIT_CD_M2`). CPU twin of `cs_band_update` in band_update.wgsl,
/// which stores it ÷ `ENV_UNIT_CD_M2`, clamped to [`ENV_MAX_UNITS`] (probe `band_update`).
pub fn band_texel(
    model: &GalaxyModel,
    cam_icrs_pc: DVec3,
    face: u32,
    uv: Vec2,
    v_lim: f32,
) -> Vec3 {
    let dir = cube_dir(face, uv);
    let dir_gc = icrs_to_galactic(dir.as_dvec3());
    let cam_gc = model.icrs_pc_to_galactocentric(cam_icrs_pc);
    model
        .band_radiance(
            cam_gc,
            dir_gc,
            f64::from(v_lim),
            f64::from(texel_footprint_rad(dir)),
        )
        .as_vec3()
}

// ---- GPU probes (`universe --selftest`) -------------------------------------------------------

/// Star limit of the `band_update` probe: the catalogue limit V = 10 (B4's drawn-star limit).
const PROBE_V_LIM: f32 = 10.0;
/// Largest relative |GPU − CPU| of `gal_band_radiance` and of a band texel (B5 brief: ±1 %).
const BAND_TOLERANCE: f64 = 1e-2;
/// Largest |sampled − `cube_dir`| in the write-then-sample probe (B5 brief: ±1e-6).
const CUBE_SAMPLE_TOLERANCE: f64 = 1e-6;
/// Largest relative error of a box-filtered mip texel against the mean of its four children as
/// read back: one f16 rounding of the mean (2⁻¹¹ = 4.9e-4) with margin.
const MIP_TOLERANCE: f64 = 1e-3;

/// One ray of the `band_radiance` probe. Twin of `BandProbeRay` in its WGSL
/// (`band_layouts_match_wgsl`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
struct BandProbeRay {
    /// xyz: camera, galactocentric [pc]; w: v_lim [V mag].
    cam_v_lim: [f32; 4],
    /// xyz: unit direction, galactic axes; w: footprint [rad] (the detail's band limit).
    dir: [f32; 4],
}

/// Shader path of the `band_radiance` probe.
const BAND_RADIANCE_PROBE: &str = "probe/band_radiance.wgsl";

/// Runs `gal_band_radiance` from procgen/galaxy.wgsl on each ray. The model's parameters travel
/// in the input buffer, ahead of the rays (`run_compute` binds one input).
const BAND_RADIANCE_PROBE_WGSL: &str = r#"// probe/band_radiance.wgsl: `universe --selftest` probe
// of gal_band_radiance (Task B5) against `GalaxyModel::band_radiance`.
#import "procgen/galaxy.wgsl"

// Twin of uni_sky::band::BandProbeRay.
struct BandProbeRay {
    // xyz: camera, galactocentric [pc]; w: v_lim [V mag].
    cam_v_lim: vec4<f32>,
    // xyz: unit direction, galactic axes; w: footprint [rad] (the detail's band limit).
    dir: vec4<f32>,
}

struct BandProbeIn {
    galaxy: GalaxyGpu,
    rays: array<BandProbeRay>,
}

@group(0) @binding(0) var<storage, read> probe_in: BandProbeIn;
// procgen/galaxy.wgsl reads the importer's `galaxy`; the probe fills it from its input.
var<private> galaxy: GalaxyGpu;
// rgb: band radiance [cd/m²]; w: 0.
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in.rays)) {
        return;
    }
    galaxy = probe_in.galaxy;
    let ray = probe_in.rays[gid.x];
    let rgb = gal_band_radiance(ray.cam_v_lim.xyz, ray.dir.xyz, ray.cam_v_lim.w, ray.dir.w);
    probe_out[gid.x] = vec4<f32>(rgb, 0.0);
}
"#;

/// Shader path of the `band_cube_sample` probe.
const CUBE_SAMPLE_PROBE: &str = "probe/band_cube_sample.wgsl";

/// Writes `cube_dir` of every texel centre into a cube with `cs_write`, then samples it with
/// `cs_sample` in given directions: the write-then-sample check of the face convention deferred
/// from A0a (its `cube_dir` probe compares arithmetic only).
const CUBE_SAMPLE_PROBE_WGSL: &str = r#"// probe/band_cube_sample.wgsl: `universe --selftest`
// write-then-sample probe of common/cube.wgsl's face convention (Task B5).
#import "common/cube.wgsl"

// cs_sample: xyz the direction to sample (root axes); w unused.
@group(0) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
// cs_sample: the sampled texel, xyz cube_dir of its centre, w its id (face, row, column).
@group(0) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;
@group(0) @binding(2) var probe_cube: texture_cube<f32>;
@group(0) @binding(3) var probe_nearest: sampler;
// cs_write: every face of the same cube as a 2D array.
@group(0) @binding(4) var probe_cube_out: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(probe_cube_out);
    if (gid.x >= size.x || gid.y >= size.y || gid.z >= 6u) {
        return;
    }
    // The texel-centre uv of band_update.wgsl.
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    // Below 2^24, so exact in f32.
    let id = f32((gid.z * size.y + gid.y) * size.x + gid.x);
    textureStore(probe_cube_out, gid.xy, gid.z, vec4<f32>(cube_dir(gid.z, uv), id));
}

@compute @workgroup_size(64, 1, 1)
fn cs_sample(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    probe_out[gid.x] = textureSampleLevel(probe_cube, probe_nearest, probe_in[gid.x].xyz, 0.0);
}
"#;

/// Registers the probe shaders in `lib`.
fn add_probe_shaders(lib: &mut ShaderLib) {
    lib.add(BAND_RADIANCE_PROBE, BAND_RADIANCE_PROBE_WGSL);
    lib.add(CUBE_SAMPLE_PROBE, CUBE_SAMPLE_PROBE_WGSL);
}

/// Unit vector toward galactic longitude and latitude [deg] (galactic axes).
fn lb_dir(l_deg: f64, b_deg: f64) -> DVec3 {
    let (l, b) = (l_deg.to_radians(), b_deg.to_radians());
    DVec3::new(b.cos() * l.cos(), b.cos() * l.sin(), b.sin())
}

/// The 64 rays of the `band_radiance` probe: from the Sun at V = 10, the band's plane at 8
/// longitudes × 5 latitudes; from the Sun at V = 6.5, both poles, Baade's window, the Coalsack,
/// Ophiuchus and 3 others; from 1 kpc above the disk at V = 12, 8 rays down onto it; from the
/// bar's edge at V = 12, 8 rays across the bulge. Footprints alternate between a face-centre
/// texel's (2/N) and 0 (every octave of the detail).
fn band_probe_rays(model: &GalaxyModel) -> Vec<BandProbeRay> {
    let texel = 2.0 / ENV_SIZE as f32;
    let mut k = 0;
    let mut ray = |cam: DVec3, dir: DVec3, v_lim: f32| {
        k += 1;
        BandProbeRay {
            cam_v_lim: [cam.x as f32, cam.y as f32, cam.z as f32, v_lim],
            dir: [
                dir.x as f32,
                dir.y as f32,
                dir.z as f32,
                if k % 2 == 0 { texel } else { 0.0 },
            ],
        }
    };
    let sun = model.sun_galactocentric_pc();
    let mut rays = Vec::with_capacity(64);
    for i in 0..8 {
        for b in [-10.0, -3.0, 0.0, 2.0, 6.0] {
            rays.push(ray(sun, lb_dir(45.0 * f64::from(i), b), 10.0));
        }
    }
    for (l, b) in [
        (0.0, 90.0),
        (0.0, -90.0),
        (1.0, -3.9),
        (-10.0, 5.0),
        (301.5, -0.5),
        (354.0, 15.5),
        (300.0, 20.0),
        (120.0, -15.0),
    ] {
        rays.push(ray(sun, lb_dir(l, b), 6.5));
    }
    let above = DVec3::new(-4000.0, 3000.0, 1000.0);
    let bar_edge = DVec3::new(-1500.0, 600.0, 120.0);
    for i in 0..8 {
        let l = 45.0 * f64::from(i) + 20.0;
        rays.push(ray(above, lb_dir(l, -30.0), 12.0));
        rays.push(ray(
            bar_edge,
            lb_dir(l, 10.0 * f64::from(i % 3) - 10.0),
            12.0,
        ));
    }
    rays
}

/// The 24 (face, column, row) texels of the `band_cube_sample` probe: per face a corner, two
/// texels near the other edges and one inside, so a flipped or swapped axis shows.
fn cube_sample_picks() -> Vec<(u32, u32, u32)> {
    (0..FACES)
        .flat_map(|face| {
            [(0, 0), (ENV_SIZE - 1, 9), (9, ENV_SIZE - 1), (131, 77)].map(|(x, y)| (face, x, y))
        })
        .collect()
}

/// The 64 (face, column, row) texels of the `band_update` probe: those toward the Galactic
/// centre and anticentre and the two Galactic poles, then 60 spread over all six faces.
fn band_update_picks() -> Vec<(u32, u32, u32)> {
    let texel = |dir: DVec3| {
        let (face, uv) = cube_face_uv(dir.as_vec3());
        let xy = (uv * ENV_SIZE as f32)
            .floor()
            .clamp(Vec2::ZERO, Vec2::splat((ENV_SIZE - 1) as f32));
        (face, xy.x as u32, xy.y as u32)
    };
    // Rows 0 and 2 of the rotation: the Galactic centre and the north Galactic pole in ICRS.
    let [centre, _, pole] = ICRS_TO_GALACTIC.map(DVec3::from_array);
    let mut picks = vec![texel(centre), texel(-centre), texel(pole), texel(-pole)];
    picks.extend((0..60).map(|k| {
        (
            k % FACES,
            (97 * k + 13) % ENV_SIZE,
            (61 * k + 29) % ENV_SIZE,
        )
    }));
    picks
}

/// The larger of two errors, NaN if either is NaN (so a NaN anywhere fails a check).
fn worst(a: f64, e: f64) -> f64 {
    if a.is_nan() || e.is_nan() {
        f64::NAN
    } else {
        a.max(e)
    }
}

/// B5's probes, called from [`crate::gpu_probes`]: `gal_band_radiance` GPU vs CPU
/// (`band_radiance`), the cube face convention written then sampled on the GPU
/// (`band_cube_sample`), and the real update pass against [`band_texel`] and its box-filtered
/// mips (`band_update`). Registers their shaders in `lib`.
pub(crate) fn band_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    crate::register_shaders(lib);
    add_probe_shaders(lib);
    let model = GalaxyModel::milky_way();
    let probe = |name: &str, result: anyhow::Result<(bool, String)>| {
        let (pass, detail) = result.unwrap_or_else(|e| (false, format!("{e:#}")));
        ProbeResult {
            name: name.to_owned(),
            pass,
            detail,
        }
    };
    vec![
        probe("band_radiance", check_band_radiance(gpu, lib, &model)),
        probe("band_cube_sample", check_cube_sample(gpu, lib)),
        probe("band_update", check_band_update(gpu, lib, &model)),
    ]
}

/// `gal_band_radiance` on the GPU against `GalaxyModel::band_radiance` for 64 rays: (pass,
/// detail).
fn check_band_radiance(
    gpu: &GpuContext,
    lib: &ShaderLib,
    model: &GalaxyModel,
) -> anyhow::Result<(bool, String)> {
    let rays = band_probe_rays(model);
    let n = u32::try_from(rays.len()).context("too many band probe rays")?;
    let mut input = bytemuck::bytes_of(&model.gpu()).to_vec();
    input.extend_from_slice(bytemuck::cast_slice(&rays));
    let src = lib.compose(BAND_RADIANCE_PROBE)?;
    let bytes = run_compute(
        gpu,
        &src,
        "cs_main",
        &input,
        u64::from(n) * 16,
        [n.div_ceil(64), 1, 1],
    )?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes);
    anyhow::ensure!(out.len() == rays.len(), "band_radiance: short read-back");
    let worst_rel = rays
        .iter()
        .zip(&out)
        .map(|(r, g)| {
            let cpu = model.band_radiance(
                Vec3::from_slice(&r.cam_v_lim[..3]).as_dvec3(),
                Vec3::from_slice(&r.dir[..3]).as_dvec3(),
                f64::from(r.cam_v_lim[3]),
                f64::from(r.dir[3]),
            );
            let gpu = Vec3::from_slice(&g[..3]).as_dvec3();
            (gpu - cpu).abs().max_element() / cpu.max_element()
        })
        .fold(0.0, worst);
    Ok((
        worst_rel <= BAND_TOLERANCE,
        format!(
            "{n} rays from 3 cameras, max relative |GPU − CPU| {worst_rel:.1e} \
             (tolerance {BAND_TOLERANCE:.0e})"
        ),
    ))
}

/// Maps every buffer in `buffers` (MAP_READ, already filled by a submitted copy), waits for the
/// GPU and returns their contents.
fn read_back(
    gpu: &GpuContext,
    buffers: &[&wgpu::Buffer],
    what: &str,
) -> anyhow::Result<Vec<Vec<u8>>> {
    let receivers: Vec<_> = buffers
        .iter()
        .map(|buffer| {
            let (tx, rx) = std::sync::mpsc::channel();
            buffer.map_async(wgpu::MapMode::Read, .., move |r| {
                // The receiver outlives the wait below; a send error cannot be acted on here.
                let _ = tx.send(r);
            });
            rx
        })
        .collect();
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .with_context(|| format!("{what}: waiting for the GPU"))?;
    let mut out = Vec::with_capacity(buffers.len());
    for (buffer, rx) in buffers.iter().zip(receivers) {
        // The wait above runs the callbacks; `try_recv` turns a missing one into an error.
        rx.try_recv()
            .with_context(|| format!("{what}: the read-back callback did not run"))?
            .with_context(|| format!("{what}: mapping the read-back"))?;
        let bytes = buffer
            .get_mapped_range(..)
            .with_context(|| format!("{what}: reading the read-back"))?
            .to_vec();
        buffer.unmap();
        out.push(bytes);
    }
    Ok(out)
}

/// The write-then-sample check of the cube face convention: a compute pass writes `cube_dir` of
/// each texel centre (and the texel's id) into an Rgba32Float cube; a second samples it (nearest)
/// at the CPU `cube_dir` of 24 texel centres. The right texel must come back, holding the CPU
/// direction within ±1e-6. (pass, detail).
fn check_cube_sample(gpu: &GpuContext, lib: &ShaderLib) -> anyhow::Result<(bool, String)> {
    let device = &gpu.device;
    let module = lib.module(device, CUBE_SAMPLE_PROBE)?;
    let picks = cube_sample_picks();
    let dirs: Vec<[f32; 4]> = picks
        .iter()
        .map(|&(face, x, y)| cube_dir(face, texel_uv(x, y)).extend(0.0).to_array())
        .collect();
    let n = u32::try_from(dirs.len()).context("too many cube sample picks")?;
    let out_bytes = size_of_val(dirs.as_slice()) as u64;

    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("band_cube_sample cube"),
        size: wgpu::Extent3d {
            width: ENV_SIZE,
            height: ENV_SIZE,
            depth_or_array_layers: FACES,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let layers = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("band_cube_sample layers"),
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let cube = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("band_cube_sample cube view"),
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    // Nearest, so a sample returns one texel exactly.
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("band_cube_sample nearest"),
        ..Default::default()
    });
    let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let buffer_entry = |binding: u32, read_only: bool| {
        entry(
            binding,
            wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
        )
    };
    let write_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("band_cube_sample write bgl"),
        entries: &[entry(
            4,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba32Float,
                view_dimension: wgpu::TextureViewDimension::D2Array,
            },
        )],
    });
    let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("band_cube_sample sample bgl"),
        entries: &[
            buffer_entry(0, true),
            buffer_entry(1, false),
            entry(
                2,
                wgpu::BindingType::Texture {
                    // Rgba32Float is unfilterable without FLOAT32_FILTERABLE; nearest needs no
                    // filtering.
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::Cube,
                    multisampled: false,
                },
            ),
            entry(
                3,
                wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
            ),
        ],
    });
    let pipeline = |bgl: &wgpu::BindGroupLayout, entry_point: &str| {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(entry_point),
            bind_group_layouts: &[Some(bgl)],
            immediate_size: 0,
        });
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let write_pipeline = pipeline(&write_bgl, "cs_write");
    let sample_pipeline = pipeline(&sample_bgl, "cs_sample");
    let input = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("band_cube_sample input"),
        contents: bytemuck::cast_slice(&dirs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let buffer = |label: &str, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: out_bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let output = buffer(
        "band_cube_sample output",
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let readback = buffer(
        "band_cube_sample read-back",
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let write_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("band_cube_sample write bg"),
        layout: &write_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 4,
            resource: wgpu::BindingResource::TextureView(&layers),
        }],
    });
    let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("band_cube_sample sample bg"),
        layout: &sample_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&cube),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("band_cube_sample"),
    });
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("band_cube_sample write"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&write_pipeline);
        pass.set_bind_group(0, &write_bg, &[]);
        let groups = ENV_SIZE.div_ceil(WORKGROUP);
        pass.dispatch_workgroups(groups, groups, FACES);
    }
    {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("band_cube_sample sample"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&sample_pipeline);
        pass.set_bind_group(0, &sample_bg, &[]);
        pass.dispatch_workgroups(n.div_ceil(64), 1, 1);
    }
    enc.copy_buffer_to_buffer(&output, 0, &readback, 0, out_bytes);
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("band_cube_sample: {err}");
    }
    let bytes = read_back(gpu, &[&readback], "band_cube_sample")?;
    let out: Vec<[f32; 4]> = bytemuck::pod_collect_to_vec(&bytes[0]);
    anyhow::ensure!(
        out.len() == picks.len(),
        "band_cube_sample: short read-back"
    );

    let mut right_texel = 0;
    let mut worst_err = 0.0_f64;
    for (&(face, x, y), (dir, got)) in picks.iter().zip(dirs.iter().zip(&out)) {
        let id = ((face * ENV_SIZE + y) * ENV_SIZE + x) as f32;
        if got[3] == id {
            right_texel += 1;
        }
        let err = (Vec3::from_slice(&got[..3]) - Vec3::from_slice(&dir[..3])).length();
        worst_err = worst(worst_err, f64::from(err));
    }
    Ok((
        right_texel == picks.len() && worst_err <= CUBE_SAMPLE_TOLERANCE,
        format!(
            "{n} face/uv pairs written then sampled: right texel {right_texel}/{n}, max \
             |sampled − cube_dir| {worst_err:.1e} (tolerance {CUBE_SAMPLE_TOLERANCE:.0e})"
        ),
    ))
}

/// Rgba16Float texels read back from a copy of one mip of all six faces (`row` bytes per row,
/// `size` rows per face) → rgb, indexed `(face × size + y) × size + x`.
fn decode_rgba16f(bytes: &[u8], size: u32, row: u32) -> anyhow::Result<Vec<Vec3>> {
    let (size, row) = (size as usize, row as usize);
    anyhow::ensure!(
        bytes.len() >= row * size * FACES as usize && row >= 8 * size,
        "band_update: short read-back ({} bytes for {size}² × 6 at {row} bytes per row)",
        bytes.len()
    );
    let channel = |at: usize| f16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
    let mut out = Vec::with_capacity(size * size * FACES as usize);
    for line in 0..size * FACES as usize {
        for x in 0..size {
            let t = line * row + 8 * x;
            out.push(Vec3::new(channel(t), channel(t + 2), channel(t + 4)));
        }
    }
    Ok(out)
}

/// The real update pass for a camera at the SSB (all six faces, as on the first frame), read
/// back: every mip-0 texel finite and in [0, `ENV_MAX_UNITS`], 64 texels against
/// [`band_texel`] (±1 %), and every texel of mips 1–4 the mean of its four children (±1e-3).
/// (pass, detail).
fn check_band_update(
    gpu: &GpuContext,
    lib: &mut ShaderLib,
    model: &GalaxyModel,
) -> anyhow::Result<(bool, String)> {
    let input = BandInput {
        camera_icrs_pc: DVec3::ZERO,
        enabled: true,
        v_lim: PROBE_V_LIM,
    };
    let mut cube = BandCubemap::new(gpu, lib, model, SharedInput::new(input))?;
    let device = &gpu.device;
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    cube.batch = cube.plan(gpu, &input);
    anyhow::ensure!(
        cube.batch.map(|b| (b.first, b.count)) == Some((0, FACES)),
        "band_update: the first build must cover all six faces, got {:?}",
        cube.batch
    );
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("band_update probe"),
    });
    cube.encode_batch(&mut enc, &mut PassTimer::disabled());
    let mips: Vec<(u32, u32, wgpu::Buffer)> = (0..ENV_MIPS)
        .map(|level| {
            let size = ENV_SIZE >> level;
            let row = (8 * size).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("band_update read-back"),
                size: u64::from(row * size * FACES),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &cube.env.texture,
                    mip_level: level,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(row),
                        rows_per_image: Some(size),
                    },
                },
                wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: FACES,
                },
            );
            (size, row, buffer)
        })
        .collect();
    gpu.queue.submit(std::iter::once(enc.finish()));
    if let Some(err) = pollster::block_on(scope.pop()) {
        anyhow::bail!("band_update: {err}");
    }
    let buffers: Vec<&wgpu::Buffer> = mips.iter().map(|(_, _, b)| b).collect();
    let bytes = read_back(gpu, &buffers, "band_update")?;
    let texels = mips
        .iter()
        .zip(&bytes)
        .map(|((size, row, _), b)| decode_rgba16f(b, *size, *row))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let at = |size: u32, face: u32, x: u32, y: u32| ((face * size + y) * size + x) as usize;

    let mip0 = &texels[0];
    let in_range = mip0
        .iter()
        .all(|t| t.is_finite() && t.min_element() >= 0.0 && t.max_element() <= ENV_MAX_UNITS);
    let brightest = mip0.iter().map(|t| t.max_element()).fold(0.0, f32::max);
    let worst_texel = band_update_picks()
        .into_iter()
        .map(|(face, x, y)| {
            let want = env_units(band_texel(
                model,
                DVec3::ZERO,
                face,
                texel_uv(x, y),
                PROBE_V_LIM,
            ))
            .as_dvec3();
            let got = mip0[at(ENV_SIZE, face, x, y)].as_dvec3();
            (got - want).abs().max_element() / want.max_element()
        })
        .fold(0.0, worst);
    let mut worst_mip = 0.0_f64;
    for level in 1..ENV_MIPS as usize {
        let (fine, coarse) = (&texels[level - 1], &texels[level]);
        let size = ENV_SIZE >> level;
        for face in 0..FACES {
            for y in 0..size {
                for x in 0..size {
                    let child = |dx: u32, dy: u32| fine[at(2 * size, face, 2 * x + dx, 2 * y + dy)];
                    let mean = 0.25 * (child(0, 0) + child(1, 0) + child(0, 1) + child(1, 1));
                    let got = coarse[at(size, face, x, y)];
                    // A floor of 1e-4 units (1e-8 cd/m²) keeps f16's subnormal steps out.
                    let err = (got - mean).abs().max_element() / mean.max_element().max(1e-4);
                    worst_mip = worst(worst_mip, f64::from(err));
                }
            }
        }
    }
    Ok((
        in_range && worst_texel <= BAND_TOLERANCE && worst_mip <= MIP_TOLERANCE,
        format!(
            "6 faces × {ENV_SIZE}², {ENV_MIPS} mips: mip 0 finite and in [0, {ENV_MAX_UNITS}] \
             {in_range}; 64 texels vs band_texel max relative {worst_texel:.1e} (tolerance \
             {BAND_TOLERANCE:.0e}); mips 1–4 vs box mean max relative {worst_mip:.1e} \
             (tolerance {MIP_TOLERANCE:.0e}); brightest texel {brightest:.3} units \
             ({:.2e} cd/m²)",
            brightest * ENV_UNIT_CD_M2
        ),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::offset_of;
    use uni_data::f32_to_f16;
    use uni_gfx::validate_wgsl;
    use wgpu::naga;

    /// Rec. 709 luminance (ITU-R BT.709-6 item 3.2).
    fn luminance(c: Vec3) -> f32 {
        c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    /// Largest per-channel |a − b| relative to the largest channel of `b`.
    fn rel_err(a: Vec3, b: Vec3) -> f32 {
        (a - b).abs().max_element() / b.max_element()
    }

    fn lib() -> ShaderLib {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        add_probe_shaders(&mut lib);
        lib
    }

    fn module(path: &str) -> naga::Module {
        let src = lib().compose(path).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{path}: {e}\n{src}"))
    }

    /// (size, [(member, offset)]) of WGSL struct `name` as naga lays it out.
    fn wgsl_layout(m: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let offsets = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (layouter[handle].size as usize, offsets)
    }

    fn named(fields: &[(&str, usize)]) -> Vec<(String, usize)> {
        fields.iter().map(|&(n, o)| (n.to_owned(), o)).collect()
    }

    /// (group, binding) of global `name`.
    fn binding(m: &naga::Module, name: &str) -> Option<(u32, u32)> {
        m.global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some(name))
            .and_then(|(_, v)| v.binding.as_ref().map(|b| (b.group, b.binding)))
    }

    /// The value of `const name: f32 = <literal>;` in `m`.
    fn wgsl_f32_const(m: &naga::Module, name: &str) -> f32 {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no const {name}"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => v,
            ref e => panic!("const {name} is not an f32 literal: {e:?}"),
        }
    }

    /// Review m10: texels hold luminance in units of `ENV_UNIT_CD_M2` = 1e-4 cd/m², so the
    /// faintest band light kept stays a normal f16 and bright light stays finite (clamped to
    /// 60000 units, below f16's 65504), on the CPU twin and in both shaders.
    #[test]
    fn env_unit_keeps_faint_sky_normal_in_f16() {
        let f16_round_trip = |x: f32| f16_to_f32(f32_to_f16(x));
        // 1e-6 cd/m² → 0.01 units: normal (≥ 2^-14) and kept to < 1e-3 relative.
        let faint = env_units(Vec3::splat(1.0e-6));
        assert!((faint.x - 0.01).abs() < 1e-9, "{faint}");
        assert!(faint.x >= 2_f32.powi(-14));
        let stored = f16_round_trip(faint.x);
        assert!((stored / faint.x - 1.0).abs() < 1e-3, "{stored}");
        // 1 cd/m² → 1e4 units, finite in f16.
        let one = env_units(Vec3::splat(1.0));
        assert!((one.x - 1.0e4).abs() < 1e-2, "{one}");
        assert!(f16_round_trip(one.x).is_finite());
        // 10 cd/m² clamps to 60000 (exact in f16), per channel.
        let bright = env_units(Vec3::new(10.0, 1.0, 1.0e-6));
        assert_eq!(bright.x, 60_000.0);
        assert_eq!(f16_round_trip(bright.x), 60_000.0);
        assert!((bright.y - 1.0e4).abs() < 1e-2 && (bright.z - 0.01).abs() < 1e-9);
        assert_eq!(ENV_MAX_UNITS, HDR_MAX);
        // Never negative.
        assert_eq!(env_units(Vec3::new(-1.0, 0.0, 0.0)), Vec3::ZERO);

        // The shaders use the same unit and clamp.
        let update = module(BAND_UPDATE_WGSL);
        assert_eq!(
            wgsl_f32_const(&update, "BAND_ENV_UNIT_CD_M2"),
            ENV_UNIT_CD_M2
        );
        assert_eq!(wgsl_f32_const(&update, "BAND_ENV_MAX_UNITS"), ENV_MAX_UNITS);
        let draw = module(BAND_DRAW_WGSL);
        assert_eq!(wgsl_f32_const(&draw, "BAND_ENV_UNIT_CD_M2"), ENV_UNIT_CD_M2);
    }

    /// B5 brief: all six faces on the first frame; afterwards one face per frame once the
    /// camera has moved more than max(1 pc, 1 % of its distance to the Galactic centre) since
    /// the last build; never in the Solar System.
    #[test]
    fn refresh_policy() {
        // 50 pc from Sgr A*: the threshold is max(1 pc, 0.5 pc) = 1 pc.
        let cam = DVec3::new(40.0, 30.0, 0.0);
        let mut policy = RefreshPolicy::default();
        let first = policy.next(cam, 10.0).expect("first frame builds");
        assert_eq!((first.first, first.count, first.cam_gc_pc), (0, FACES, cam));
        assert_eq!(first.v_lim, 10.0);
        for _ in 0..10 {
            assert_eq!(
                policy.next(cam, 10.0),
                None,
                "a static camera never refreshes"
            );
        }
        // Moved 0.9 pc: below the threshold.
        assert_eq!(policy.next(cam + DVec3::X * 0.9, 10.0), None);
        // Moved 2 pc: one face per frame for six frames, all from the position the rebuild
        // started at, even if the camera moves on meanwhile; then nothing.
        let moved = cam + DVec3::new(2.0, 0.0, 0.0);
        for face in 0..FACES {
            let now = moved + DVec3::Y * (0.1 * f64::from(face));
            let b = policy.next(now, 10.0).expect("one face per frame");
            assert_eq!(
                (b.first, b.count, b.cam_gc_pc),
                (face, 1, moved),
                "frame {face}"
            );
        }
        // The camera ended 0.5 pc from the rebuilt position: no new rebuild.
        assert_eq!(policy.next(moved + DVec3::Y * 0.5, 10.0), None);
        // A new star limit rebuilds the same way.
        let b = policy.next(moved, 9.0).expect("v_lim changed");
        assert_eq!((b.first, b.count, b.v_lim), (0, 1, 9.0));

        // In the Solar System the threshold is 1 % of 8277 pc = 82.8 pc: never reached.
        let model = GalaxyModel::milky_way();
        let sun = model.sun_galactocentric_pc();
        let mut policy = RefreshPolicy::default();
        assert_eq!(policy.next(sun, 10.0).map(|b| b.count), Some(FACES));
        let au_pc = 1.0 / 206_264.806; // IAU 2012 B2 au; 1 pc = 648000/π au
        for step in [1000.0 * au_pc, 1.0, 10.0, 80.0] {
            assert_eq!(policy.next(sun + DVec3::Z * step, 10.0), None, "{step} pc");
        }
        assert!(
            policy.next(sun + DVec3::Z * 85.0, 10.0).is_some(),
            "85 pc > 82.8 pc"
        );
    }

    /// The CPU twin samples the model in the right frame: the texel toward the Galactic centre
    /// (ICRS direction from ESA 1997, row 0 of `ICRS_TO_GALACTIC`) holds the model's radiance
    /// toward the centre, and a camera off the SSB is moved into the model's frame.
    #[test]
    fn band_texel_matches_model() {
        let model = GalaxyModel::milky_way();
        let sun = model.sun_galactocentric_pc();
        let gc_icrs = DVec3::from_array(ICRS_TO_GALACTIC[0]);
        let (face, uv) = cube_face_uv(gc_icrs.as_vec3());
        assert_eq!(face, 3, "the centre (RA 266°, Dec −29°) is on the −Y face");
        let dir = cube_dir(face, uv).as_dvec3();
        assert!(dir.angle_between(gc_icrs) < 1e-6, "{dir} vs {gc_icrs}");

        let got = band_texel(&model, DVec3::ZERO, face, uv, 10.0);
        let fp = f64::from(texel_footprint_rad(dir.as_vec3()));
        let want = model
            .band_radiance(sun, icrs_to_galactic(dir), 10.0, fp)
            .as_vec3();
        assert!(rel_err(got, want) < 1e-6, "{got} vs {want}");
        // Exactly toward the centre (galactic +x) the radiance differs only by the texel
        // direction's f32 rounding.
        let centre = model.band_radiance(sun, DVec3::X, 10.0, fp).as_vec3();
        assert!(rel_err(got, centre) < 1e-4, "{got} vs {centre}");

        // A camera off the SSB, with another star limit.
        let cam = DVec3::new(120.0, -40.0, 300.0);
        let got = band_texel(&model, cam, face, uv, 8.0);
        let want = model
            .band_radiance(
                model.icrs_pc_to_galactocentric(cam),
                icrs_to_galactic(dir),
                8.0,
                fp,
            )
            .as_vec3();
        assert!(rel_err(got, want) < 1e-6, "{got} vs {want}");

        // The rotation is pinned by a second axis: the texel toward the north Galactic pole
        // (RA 192.86°, Dec 27.13°, row 2 of `ICRS_TO_GALACTIC`) holds the model's radiance
        // toward galactic +z, far fainter than the centre's.
        let ngp_icrs = DVec3::from_array(ICRS_TO_GALACTIC[2]);
        let (face, uv) = cube_face_uv(ngp_icrs.as_vec3());
        let ngp = band_texel(&model, DVec3::ZERO, face, uv, 10.0);
        let fp_pole = f64::from(texel_footprint_rad(cube_dir(face, uv)));
        let pole = model.band_radiance(sun, DVec3::Z, 10.0, fp_pole).as_vec3();
        assert!(rel_err(ngp, pole) < 1e-4, "{ngp} vs {pole}");
        assert!(
            luminance(centre) > 2.0 * luminance(ngp),
            "{centre} vs {ngp}"
        );

        // The update pass's camera conversion (an offset from the SSB's galactocentric
        // position) is the model's.
        let origin = model.icrs_pc_to_galactocentric(DVec3::ZERO);
        for p in [DVec3::ZERO, cam, DVec3::new(-3.0e4, 2.0e3, 7.5e3)] {
            let d = galactocentric(origin, p) - model.icrs_pc_to_galactocentric(p);
            assert!(d.length() < 1e-9, "{p}: {d}");
        }
    }

    /// The update pass's parameters carry the batch, the camera in f32 and the ICRS → Galactic
    /// rotation rows of ESA 1997 (`uni_core::ICRS_TO_GALACTIC`).
    #[test]
    fn update_params_carry_the_batch() {
        let batch = FaceBatch {
            first: 4,
            count: 1,
            cam_gc_pc: DVec3::new(-8277.0, 0.25, 25.0),
            v_lim: 9.0,
        };
        let p = BandUpdateParams::new(&batch);
        assert_eq!(p.cam_gc_v_lim, [-8277.0, 0.25, 25.0, 9.0]);
        assert_eq!(p.faces, [4, 1, 0, 0]);
        let d = DVec3::new(0.3, -0.5, 0.81).normalize();
        let rotated = Vec3::from_array(
            p.icrs_to_gal
                .map(|r| Vec3::from_slice(&r[..3]).dot(d.as_vec3())),
        );
        assert!((rotated.as_dvec3() - icrs_to_galactic(d)).length() < 1e-6);
        assert!(p.icrs_to_gal.iter().all(|r| r[3] == 0.0));
    }

    /// `cube_face_uv` inverts `cube_dir` (the probe uses it to pick texels).
    #[test]
    fn cube_face_uv_inverts_cube_dir() {
        for face in 0..FACES {
            for (x, y) in [(0, 0), (255, 9), (9, 255), (131, 77)] {
                let uv = (Vec2::new(x as f32, y as f32) + 0.5) / ENV_SIZE as f32;
                let (f, back) = cube_face_uv(cube_dir(face, uv));
                assert_eq!(f, face);
                assert!((back - uv).length() < 1e-6, "{face} {uv} -> {back}");
            }
        }
    }

    /// CLAUDE.md §3: every uniform and storage struct matches its WGSL twin, and every binding
    /// sits where the Rust bind group layouts put it.
    #[test]
    fn band_layouts_match_wgsl() {
        let update = module(BAND_UPDATE_WGSL);
        let (size, offsets) = wgsl_layout(&update, "BandUpdateParams");
        assert_eq!(size, size_of::<BandUpdateParams>());
        assert_eq!(
            offsets,
            named(&[
                ("cam_gc_v_lim", offset_of!(BandUpdateParams, cam_gc_v_lim)),
                ("icrs_to_gal", offset_of!(BandUpdateParams, icrs_to_gal)),
                ("faces", offset_of!(BandUpdateParams, faces)),
            ])
        );
        for (name, slot) in [
            ("galaxy", (1, 0)),
            ("band_params", (1, 1)),
            ("band_out", (1, 2)),
            ("band_mip_src", (1, 3)),
            ("band_mip_dst", (1, 4)),
        ] {
            assert_eq!(binding(&update, name), Some(slot), "{name}");
        }

        let draw = module(BAND_DRAW_WGSL);
        for (name, slot) in [
            ("camera", (0, 0)),
            ("exposure", (0, 1)),
            ("band_env", (1, 0)),
            ("band_sampler", (1, 1)),
            ("band_lens_mask", (1, 2)),
        ] {
            assert_eq!(binding(&draw, name), Some(slot), "{name}");
        }

        // The gal_band_radiance probe's input: GalaxyGpu, then the rays.
        let probe = module(BAND_RADIANCE_PROBE);
        let (size, offsets) = wgsl_layout(&probe, "BandProbeRay");
        assert_eq!(size, size_of::<BandProbeRay>());
        assert_eq!(
            offsets,
            named(&[
                ("cam_v_lim", offset_of!(BandProbeRay, cam_v_lim)),
                ("dir", offset_of!(BandProbeRay, dir)),
            ])
        );
        let (_, offsets) = wgsl_layout(&probe, "BandProbeIn");
        assert_eq!(
            offsets,
            named(&[
                ("galaxy", 0),
                ("rays", size_of::<uni_procgen::galaxy::GalaxyGpu>())
            ])
        );
        assert_eq!(binding(&probe, "probe_in"), Some((0, 0)));
        assert_eq!(binding(&probe, "probe_out"), Some((0, 1)));
    }

    /// CLAUDE.md §3: every WGSL file validates with naga, with the entry points the pipelines
    /// name.
    #[test]
    fn band_shaders_validate() {
        use naga::ShaderStage::{Compute, Fragment, Vertex};
        // Compute workgroups must match the dispatch arithmetic in Rust (`WORKGROUP`, and 64 for
        // the probes' one-dimensional dispatches), or texels go unwritten.
        let tile = [WORKGROUP, WORKGROUP, 1];
        let line = [64, 1, 1];
        for (path, entries) in [
            (
                BAND_UPDATE_WGSL,
                &[
                    ("cs_band_update", Compute, tile),
                    ("cs_band_downsample", Compute, tile),
                ][..],
            ),
            (
                BAND_DRAW_WGSL,
                &[("vs_main", Vertex, [0; 3]), ("fs_main", Fragment, [0; 3])][..],
            ),
            (BAND_RADIANCE_PROBE, &[("cs_main", Compute, line)][..]),
            (
                CUBE_SAMPLE_PROBE,
                &[("cs_write", Compute, tile), ("cs_sample", Compute, line)][..],
            ),
        ] {
            let m = module(path);
            for &(name, stage, workgroup) in entries {
                let ep = m
                    .entry_points
                    .iter()
                    .find(|ep| ep.name == name && ep.stage == stage)
                    .unwrap_or_else(|| panic!("{path}: no {stage:?} entry point {name}"));
                assert_eq!(ep.workgroup_size, workgroup, "{path}: {name}");
            }
        }
    }

    /// The gal_band_radiance probe covers 64 rays from three cameras, the band's plane, both
    /// poles and three star limits, and none of them is dark (a relative check needs light);
    /// the other probes' texels are in range.
    #[test]
    fn band_probe_rays_cover_the_sky() {
        let model = GalaxyModel::milky_way();
        let rays = band_probe_rays(&model);
        assert_eq!(rays.len(), 64);
        let mut cams: Vec<[f32; 3]> = rays
            .iter()
            .map(|r| [r.cam_v_lim[0], r.cam_v_lim[1], r.cam_v_lim[2]])
            .collect();
        cams.sort_by(|a, b| a.partial_cmp(b).unwrap());
        cams.dedup();
        assert_eq!(cams.len(), 3);
        assert!(rays.iter().any(|r| r.dir[2] > 0.999) && rays.iter().any(|r| r.dir[2] < -0.999));
        let mut v_lims: Vec<f32> = rays.iter().map(|r| r.cam_v_lim[3]).collect();
        v_lims.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v_lims.dedup();
        assert_eq!(v_lims.len(), 3);
        // The texels the other two probes check: in range, and 24 / 64 of them.
        let sample = cube_sample_picks();
        let update = band_update_picks();
        assert_eq!((sample.len(), update.len()), (24, 64));
        for &(face, x, y) in sample.iter().chain(&update) {
            assert!(
                face < FACES && x < ENV_SIZE && y < ENV_SIZE,
                "{face} {x} {y}"
            );
        }
        assert!((0..FACES).all(|f| sample.iter().filter(|p| p.0 == f).count() == 4));
        assert_eq!(
            update[0].0, 3,
            "the Galactic centre's texel is on the −Y face"
        );
        for r in &rays {
            let dir = Vec3::from_slice(&r.dir[..3]);
            assert!((dir.length() - 1.0).abs() < 1e-6);
            let cam = Vec3::from_slice(&r.cam_v_lim[..3]).as_dvec3();
            let rgb = model.band_radiance(
                cam,
                dir.as_dvec3(),
                f64::from(r.cam_v_lim[3]),
                f64::from(r.dir[3]),
            );
            assert!(rgb.min_element() > 0.0 && rgb.is_finite(), "{r:?}: {rgb}");
        }
    }
}
