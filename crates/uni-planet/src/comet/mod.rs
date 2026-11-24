//! Comets, pass "comets" (Task C1; Task C0 wrote this stub, its constants and signatures).
//!
//! Draws every comet of `uni_sim::snapshot::Snapshot::comets` (the app copies them into
//! [`CometScene`]): a resolved nucleus as a real mesh ([`nucleus`]; opaque, depth write,
//! `lens_mask.g` = coverage; between `POINT_LOD_PX` and `BLEND_LOD_PX` a world writer in the LOD
//! band, exposure `pre_mix` with `body::exposure_weight(proj_radius_px, eye_adapted)` and
//! `lens_mask.a` = its star share, as a body); the coma of every active comet ([`coma`]); and the
//! dust and ion tails of at most [`MAX_TAILED_COMETS`] of them ([`tails`]), recomputed at most
//! every [`TAIL_REFRESH_WALL_S`] of wall time. World light: luminance × `scene_pre()` (frame
//! contract). A nucleus below `POINT_LOD_PX` is uni-sim's point, never drawn here.
//!
//! **Light (lead ruling "comet double light").** SBDB's total magnitude (M1, K1) is the nucleus
//! plus the coma. The nucleus reflects sunlight as a p_V 0.04 lunar-Lambert body of its (assumed)
//! radius ([`nucleus::nucleus_illuminance_lux`], which the app's nucleus point should use too);
//! the coma carries the total minus that, floored at 0 ([`coma::coma_intensity_cd`]); the ion
//! tail adds 0.2 × the coma's light (v0 ratio, card note `uni_ui::theme::NOTE_TAILS`); the dust
//! tail carries the dust emitted over the last 60 days, at the coma's light /
//! [`tails::DUST_HEAD_TIME_S`] per second of current production (C1L, calibrated on NEOWISE),
//! lit and phased grain by grain. A dust splat wider than `SPRITE_MAX_PX` (the far tail, or a
//! tail seen from close by) and an ion tail seen end-on lose some of that light (approximate;
//! stated where they are drawn). Glow (coma, ion and dust) fainter than [`GLOW_CULL_PRE`] pre-exposed is not drawn.
//!
//! **Pass.** One render pass, timer label "comets" (no sub-pass, so no extra label): the nuclei
//! (opaque, depth write, `lens_mask.g` and `.a`), then the comae, ion tails and dust sprites
//! (additive, depth test only, no `lens_mask` write: optically thin). Every comet's bounds are
//! culled on the CPU before any GPU work ([`plan_frame`]). CPU: the whole `prepare`
//! ([`CometRenderer::cpu_ms`]) is kept near 1 ms by spreading the tail work over frames;
//! [`CometRenderer::set_settle_tails`] lets a capture finish every due tail in its first frame.

pub mod coma;
pub mod nucleus;
pub mod tails;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::Context as _;
use glam::{DMat3, DVec3, UVec2, Vec3};
use uni_core::{
    AU, CameraView, Epoch, M_V_SUN, PC, TEFF_SUN_K, blackbody_linear_srgb, star_illuminance_lux,
    xyz_to_linear_srgb,
};
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT,
    PassTimer, RenderFeature, ShaderLib, SharedInput,
};
use uni_physics::photometry::rho_from_geometric_albedo;
use uni_sim::registry::ObjectId;
use uni_sim::snapshot::{CometView, POINT_LOD_PX};
use wgpu::util::DeviceExt as _;

use self::coma::{
    COMA_EXTENT_SCALES, coma_colour, coma_drawn_brightness, coma_intensity_cd, coma_scale_m,
    coma_surface_brightness,
};
use self::nucleus::{
    Icosphere, NUCLEUS_ALBEDO, NUCLEUS_BRDF, NUCLEUS_SUBDIVISIONS, icosphere, nucleus_orientation,
    nucleus_vertices, vertex_normals,
};
use self::tails::{
    GRAINS_PER_TAIL, ION_HALF_WIDTH_SIGMAS, ION_SEGMENTS, ION_TAIL_LIGHT, KEPLER_SOLVE_US,
    TAIL_CPU_SHARE_MS, TailManager, TailWork, dust_emission_cd_per_s, dust_grain_light, dust_phase,
    dust_reach_estimate_m, ion_colour, ion_drawn_sigma_m, ion_sigma_m, ion_tail_brightness,
    ion_tail_dir, ion_tail_drawn_brightness, ion_tail_length_m, is_active, splat_axes,
    sprite_profile, tailed_comets,
};
use crate::body::{exposure_weight, lens_mask_target, sphere_in_view};

/// Tails render for at most this many comets on screen, the most prominent (brightest predicted
/// total V from the camera) (ruling round 2, review I13).
pub const MAX_TAILED_COMETS: usize = 8;
/// Under time warp a comet's tail geometry is recomputed at most this often [s of wall time]
/// (ruling round 2, review I13).
pub const TAIL_REFRESH_WALL_S: f64 = 0.25;

/// What the comet pass draws this frame (a `SharedInput` the app sets before `render`).
#[derive(Clone, Debug)]
pub struct CometScene {
    /// `Snapshot::comets`: brightest first.
    pub comets: Vec<CometView>,
    /// The frame's epoch (tails are emitted up to it).
    pub epoch: Epoch,
    /// "Eye-adapted stars" is in effect (`ExposureSettings::eye_adapted_in_effect()`), as
    /// `BodyScene::eye_adapted`: a nucleus in the LOD band blends toward star exposure.
    pub eye_adapted: bool,
}

/// `uni_core::Epoch` has no `Default`, so this is by hand: no comets, at J2000.
impl Default for CometScene {
    fn default() -> Self {
        CometScene {
            comets: Vec::new(),
            epoch: Epoch::J2000,
            eye_adapted: false,
        }
    }
}

/// The Sun's luminous intensity [cd]: its V-band illuminance at 10 pc times (10 pc)², with
/// M_V☉ = 4.83 (Cox 2000, *Allen's Astrophysical Quantities*, 4th ed.) on
/// `star_illuminance_lux`'s zero point (Allen 1973): 1.26 × 10⁵ lx at 1 AU. The same value as
/// uni-sim's (crate-private) `sun_intensity_cd`, so the nucleus, the coma and the point agree.
pub(crate) fn sun_intensity_cd() -> f64 {
    let ten_pc = 10.0 * PC;
    star_illuminance_lux(M_V_SUN) * ten_pc * ten_pc
}

/// Rec. 709 luminance of a linear colour (ITU-R BT.709-6, item 3.2: 0.2126, 0.7152, 0.0722).
#[cfg(test)]
pub(crate) fn luminance(c: Vec3) -> f32 {
    0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
}

/// The colour of a spectral line whose CIE 1931 colour-matching values are `cmf`: converted to
/// linear Rec. 709 (`uni_core::xyz_to_linear_srgb`), scaled to luminance 1, then moved toward
/// white (1, 1, 1) at constant luminance just far enough that no channel is negative (a spectral
/// colour lies outside every RGB gamut).
pub(crate) fn line_colour(cmf: [f64; 3]) -> Vec3 {
    let rgb = xyz_to_linear_srgb(DVec3::from_array(cmf));
    let y = 0.2126 * rgb.x + 0.7152 * rgb.y + 0.0722 * rgb.z;
    if !(y > 0.0 && y.is_finite()) {
        return Vec3::ONE;
    }
    let c = rgb / y;
    let min = c.min_element();
    // White has luminance 1 too, so every mix of the two keeps it.
    let c = if min < 0.0 {
        DVec3::ONE + (c - DVec3::ONE) / (1.0 - min)
    } else {
        c
    };
    c.max(DVec3::ZERO).as_vec3()
}

/// Bindings of group 1 (`comet.wgsl`): nuclei, comae, tails, grains (storage, read-only).
pub(crate) const COMET_BINDINGS: u32 = 4;

/// A nucleus mesh stays within this many mean radii of its centre (culling bound; the two-lobe
/// shapes reach ≈ 1.6, `nucleus_mean_radius_matches_diameter` checks it).
pub(crate) const NUCLEUS_BOUND_RADII: f64 = 2.5;
/// A camera nearer the centre than this many mean radii is inside every nucleus shape (their
/// smallest radius is ≈ 0.4): nothing is drawn. Elsewhere inside, back-face culling draws nothing.
pub(crate) const NUCLEUS_INSIDE_RADII: f64 = 0.25;

/// Twin of `struct NucleusGpu` in `comet.wgsl` (vec4 members only): one resolved nucleus.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct NucleusGpu {
    /// xyz: nucleus − camera [m], root axes; w: the exposure weight w (1 = scene exposure).
    pub(crate) centre: [f32; 4],
    /// The body → root rotation's columns (w: 0).
    pub(crate) axis_x: [f32; 4],
    pub(crate) axis_y: [f32; 4],
    pub(crate) axis_z: [f32; 4],
    /// xyz: unit direction from the nucleus to the Sun; w: sunlight E☉ [lx] at the nucleus.
    pub(crate) sun: [f32; 4],
    /// rgb: the Sun's colour, linear Rec. 709 with luminance 1; w: normal reflectance ρ.
    pub(crate) light: [f32; 4],
}

/// Twin of `struct ComaGpu` in `comet.wgsl`: one coma quad.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ComaGpu {
    /// xyz: nucleus − camera [m], root axes; w: the quad's half size [m].
    pub(crate) centre: [f32; 4],
    /// x: luminous intensity I [cd]; y: scale length ℓ [m]; z: pixel radius [m]; w: 0.
    pub(crate) profile: [f32; 4],
    /// rgb: colour, linear Rec. 709 with luminance 1; w: 0.
    pub(crate) colour: [f32; 4],
}

/// Twin of `struct TailGpu` in `comet.wgsl`: one tailed comet (a slot of the tail buffers).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TailGpu {
    /// xyz: nucleus − camera [m], root axes; w: the ion tail's luminous intensity [cd].
    pub(crate) nucleus: [f32; 4],
    /// xyz: Sun − nucleus [m]; w: the dust's light production [cd/s per unit of HG]
    /// (`tails::dust_emission_cd_per_s`: the coma's light / `DUST_HEAD_TIME_S`).
    pub(crate) sun: [f32; 4],
    /// xyz: the ion tail's direction (unit); w: its length [m].
    pub(crate) ion_axis: [f32; 4],
    /// rgb: the ion tail's colour (luminance 1); w: 0.
    pub(crate) ion_colour: [f32; 4],
    /// rgb: the dust's colour (sunlight, luminance 1); w: 0.
    pub(crate) dust_colour: [f32; 4],
}

/// Twin of `struct GrainGpu` in `comet.wgsl`: one dust splat (C1L).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GrainGpu {
    /// xyz: offset from the nucleus [m], root axes; w: the emission weight [s]
    /// (`tails::TailGrain::weight`).
    pub(crate) offset: [f32; 4],
    /// The 3D covariance's lower-triangular factor L (Σ = L Lᵀ, `tails::cov_factor`) [m]: L₀₀,
    /// L₁₀, L₂₀, L₁₁...
    pub(crate) cov_l0: [f32; 4],
    /// ...and L₂₁, L₂₂; zw: 0.
    pub(crate) cov_l1: [f32; 4],
}

impl GrainGpu {
    /// The GPU record of a splat.
    pub(crate) fn of(g: &tails::TailGrain) -> Self {
        let l = tails::cov_factor(g.cov_m2);
        GrainGpu {
            offset: vec4(g.offset_m, g.weight),
            cov_l0: [
                l.x_axis.x as f32,
                l.x_axis.y as f32,
                l.x_axis.z as f32,
                l.y_axis.y as f32,
            ],
            cov_l1: [l.y_axis.z as f32, l.z_axis.z as f32, 0.0, 0.0],
        }
    }
}

/// Glow (coma, ion ribbon and dust splats) whose brightest point would be written below this
/// pre-exposed value is not drawn (C1L): 1/1 000 of AgX's black point 2^−12.47393 = 1.75 × 10⁻⁴
/// (uni-post `agx`, after Sobotka's AgX and Wrensch 2023), so even a thousand such splats on one
/// pixel stay below black. Twin of `GLOW_CULL_PRE` in `comet.wgsl`. It removes the fill of faint
/// comets' comae and tails, which no exposure the frame has could show.
pub const GLOW_CULL_PRE: f64 = 1.75e-7;
/// A dust splat's σ is at most this many pixels on each axis (twin of `SPRITE_MAX_PX` in
/// `comet.wgsl`; C1 had 32 px on round sprites); a wider one keeps its surface brightness and
/// loses the light beyond (approximate). That is 9 % of row 20's dust light (its oldest, faintest
/// splats; `neowise_dust_tail_is_long_and_curved` prints it) and most of a tail seen from close
/// by: at row 21 (1P/Halley from 2 × 10⁶ km) 98 %, a diffuse glow of ≈ 4 × 10⁻⁴ cd/m² at the
/// frame's edge (C1L report, §6: a coarse-resolution path that keeps it cost 0.77 ms at row 20).
pub const SPRITE_MAX_PX: f64 = 64.0;

/// `[v.x, v.y, v.z, w]` in f32 (camera-relative values are formed in f64 first).
fn vec4(v: DVec3, w: f64) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w as f32]
}

/// `[c.x, c.y, c.z, 0]`.
fn rgb0(c: Vec3) -> [f32; 4] {
    [c.x, c.y, c.z, 0.0]
}

/// A nucleus to draw this frame.
#[derive(Clone, Debug)]
pub(crate) struct NucleusDraw {
    pub(crate) id: ObjectId,
    /// Shape seed: the SPK-ID.
    pub(crate) seed: u64,
    pub(crate) radius_m: f64,
    pub(crate) gpu: NucleusGpu,
}

/// A tailed comet to draw this frame.
#[derive(Clone, Debug)]
pub(crate) struct TailDraw {
    pub(crate) id: ObjectId,
    pub(crate) gpu: TailGpu,
    /// The version of its dust tail (`TailState::version`), `None` while it has none yet.
    pub(crate) dust_version: Option<u64>,
}

/// Everything the pass draws this frame, built on the CPU (no GPU objects; tested on its own).
#[derive(Clone, Debug, Default)]
pub(crate) struct FramePlan {
    pub(crate) nuclei: Vec<NucleusDraw>,
    pub(crate) comae: Vec<ComaGpu>,
    pub(crate) tails: Vec<TailDraw>,
    pub(crate) work: TailWork,
}

/// Builds this frame's draws from `scene` for `camera` at internal size `viewport`: culls every
/// comet's bounds, picks the tailed comets, and runs the tail work at wall time `wall_s` within
/// `max_solves` Kepler solves.
///
/// - A nucleus is drawn when uni-sim resolved it (≥ `POINT_LOD_PX`), it can be on screen, and the
///   camera is not inside it; its exposure weight is the body's ([`exposure_weight`]).
/// - An active comet (r_h < 5 AU with SBDB's M1, K1) with coma light draws its coma when the
///   coma's quad can be on screen.
/// - Its tails are candidates when a sphere around the nucleus reaching the farthest of the
///   coma, the ion tail and the dust tail (the last refresh's reach, or
///   `tails::dust_reach_estimate_m` before the first) can be on screen; the
///   [`MAX_TAILED_COMETS`] brightest candidates draw them.
pub(crate) fn plan_frame(
    scene: &CometScene,
    camera: &CameraView,
    viewport: UVec2,
    wall_s: f64,
    manager: &mut TailManager,
    max_solves: usize,
) -> FramePlan {
    let mut plan = FramePlan::default();
    let height = f64::from(viewport.y.max(1));
    let aspect = f64::from(viewport.x.max(1)) / height;
    // One pixel's side at unit distance [rad], and the radius of a disk of one pixel's area.
    let pixel_rad = 2.0 * (0.5 * camera.fov_y_rad).tan() / height;
    let pixel_disk_rad = pixel_rad / std::f64::consts::PI.sqrt();
    let sun_cd = sun_intensity_cd();
    let sun_colour = blackbody_linear_srgb(TEFF_SUN_K);
    let rho = rho_from_geometric_albedo(NUCLEUS_BRDF, NUCLEUS_ALBEDO);
    let coma_rgb = coma_colour();
    let mut candidates: Vec<CometView> = Vec::new();
    for view in &scene.comets {
        let rel = view.nucleus_rel_m;
        let d = rel.length();
        let r_sun = view.sun_rel_m.length();
        if !(d.is_finite() && d > 0.0 && r_sun.is_finite() && r_sun > 0.0) {
            continue;
        }
        let radius = view.nucleus_radius_m;
        if view.proj_radius_px >= POINT_LOD_PX
            && d > NUCLEUS_INSIDE_RADII * radius
            && sphere_in_view(camera, aspect, rel, NUCLEUS_BOUND_RADII * radius)
        {
            let seed = u64::from(view.spkid.unsigned_abs());
            let rot = DMat3::from_quat(nucleus_orientation(seed));
            let w = exposure_weight(view.proj_radius_px, scene.eye_adapted);
            plan.nuclei.push(NucleusDraw {
                id: view.id,
                seed,
                radius_m: radius,
                gpu: NucleusGpu {
                    centre: vec4(rel, f64::from(w)),
                    axis_x: vec4(rot.x_axis, 0.0),
                    axis_y: vec4(rot.y_axis, 0.0),
                    axis_z: vec4(rot.z_axis, 0.0),
                    sun: vec4(view.sun_rel_m / r_sun, sun_cd / (r_sun * r_sun)),
                    light: [sun_colour.x, sun_colour.y, sun_colour.z, rho as f32],
                },
            });
        }
        if !is_active(view) {
            continue;
        }
        let intensity = coma_intensity_cd(view);
        if !(intensity > 0.0 && intensity.is_finite()) {
            continue;
        }
        let scale = coma_scale_m(view.r_au);
        let pixel_m = d * pixel_disk_rad;
        let half = COMA_EXTENT_SCALES * scale.max(pixel_m);
        if sphere_in_view(camera, aspect, rel, half) {
            plan.comae.push(ComaGpu {
                centre: vec4(rel, half),
                profile: [intensity as f32, scale as f32, pixel_m as f32, 0.0],
                colour: rgb0(coma_rgb),
            });
        }
        let dust_reach = manager
            .tails
            .get(&view.id)
            .map_or_else(|| dust_reach_estimate_m(view.r_au), |t| t.reach_m);
        // The ribbon's half width with its pixel floor at its farthest point (review C1 I2).
        let ion_length = ion_tail_length_m(view.r_au);
        let ion_reach =
            ion_length + ION_HALF_WIDTH_SIGMAS * ion_drawn_sigma_m(pixel_rad * (d + ion_length));
        let reach = dust_reach.max(ion_reach).max(half);
        if sphere_in_view(camera, aspect, rel, reach) {
            candidates.push(view.clone());
        }
    }
    plan.work = manager.update(&candidates, scene.epoch, wall_s, max_solves);
    let ion_rgb = ion_colour();
    for i in tailed_comets(&candidates) {
        let view = &candidates[i];
        let intensity = coma_intensity_cd(view);
        let axis = ion_tail_dir(view.sun_rel_m, view.helio_vel_mps);
        let ion_light = if axis == DVec3::ZERO {
            0.0
        } else {
            ION_TAIL_LIGHT * intensity
        };
        // C1L: the dust's light production; each splat's light is this × its emission weight ×
        // (r_h/r)² × HG, evaluated on the GPU.
        let dust_rate = if manager.tails.contains_key(&view.id) {
            dust_emission_cd_per_s(intensity)
        } else {
            0.0
        };
        plan.tails.push(TailDraw {
            id: view.id,
            gpu: TailGpu {
                nucleus: vec4(view.nucleus_rel_m, ion_light),
                sun: vec4(view.sun_rel_m, dust_rate),
                ion_axis: vec4(axis, ion_tail_length_m(view.r_au)),
                ion_colour: rgb0(ion_rgb),
                dust_colour: rgb0(sun_colour),
            },
            dust_version: manager.tails.get(&view.id).map(|t| t.version),
        });
    }
    plan
}

/// Comets whose nucleus mesh was not drawn for this many frames lose it (rebuilt on return).
const MESH_KEEP_FRAMES: u64 = 600;
/// Nuclei and comae the storage buffers hold before they first grow (about 300 comets are
/// active at the default date; uni-sim lists them all).
const INITIAL_NUCLEI: usize = 4;
const INITIAL_COMAE: usize = 512;

/// A storage buffer of `T` records that grows (by powers of two) when a frame needs more.
struct RecordBuffer {
    buffer: wgpu::Buffer,
    capacity: usize,
    stride: usize,
    label: &'static str,
}

impl RecordBuffer {
    fn new(device: &wgpu::Device, label: &'static str, stride: usize, capacity: usize) -> Self {
        RecordBuffer {
            buffer: storage_buffer(device, label, (stride * capacity.max(1)) as u64),
            capacity: capacity.max(1),
            stride,
            label,
        }
    }

    /// Writes `bytes` (whole records); returns true when the buffer had to grow (the bind group
    /// must then be rebuilt).
    fn write(&mut self, gpu: &GpuContext, bytes: &[u8]) -> bool {
        let records = bytes.len() / self.stride;
        let grew = records > self.capacity;
        if grew {
            self.capacity = records.next_power_of_two();
            self.buffer = storage_buffer(
                &gpu.device,
                self.label,
                (self.stride * self.capacity) as u64,
            );
        }
        if !bytes.is_empty() {
            gpu.queue.write_buffer(&self.buffer, 0, bytes);
        }
        grew
    }
}

fn storage_buffer(device: &wgpu::Device, label: &'static str, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// One nucleus mesh on the GPU (positions and smooth normals interleaved, body axes [m]).
struct NucleusMesh {
    vertices: wgpu::Buffer,
    seed: u64,
    radius_m: f64,
    last_frame: u64,
}

/// A tail slot: the comet in it and the dust version its grains were uploaded at.
type Slot = Option<(ObjectId, Option<u64>)>;

/// The comet pass (`RenderFeature` named "comets"): one render pass (timer label "comets") with
/// four pipelines, nuclei first (opaque, depth write), then the additive coma, ion and dust
/// parts (depth test only). The app builds it only once C1 has landed (lead rule: no stubbed
/// renderer is constructed).
pub struct CometRenderer {
    input: SharedInput<CometScene>,
    nucleus_pipeline: wgpu::RenderPipeline,
    coma_pipeline: wgpu::RenderPipeline,
    ion_pipeline: wgpu::RenderPipeline,
    dust_pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    nuclei: RecordBuffer,
    comae: RecordBuffer,
    tails_buffer: wgpu::Buffer,
    grains_buffer: wgpu::Buffer,
    ico: Icosphere,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    meshes: HashMap<ObjectId, NucleusMesh>,
    manager: TailManager,
    slots: [Slot; MAX_TAILED_COMETS],
    /// This frame's draws: nucleus ids in record order, the coma count, (slot, dust splats).
    nucleus_draws: Vec<ObjectId>,
    coma_count: u32,
    tail_draws: Vec<(u32, u32)>,
    /// The dust splats uploaded per slot.
    slot_splats: [u32; MAX_TAILED_COMETS],
    clock: Instant,
    frame: u64,
    /// Measured cost of one Kepler solve [µs] (moving average; starts at the plan's 0.5 µs).
    us_per_solve: f64,
    settle: bool,
    cpu_ms: f32,
}

impl CometRenderer {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<CometScene>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, crate::COMET_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("comets bgl"),
            entries: &[storage(0), storage(1), storage(2), storage(3)],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("comets layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        // Nucleus: opaque over hdr; lens_mask g = coverage and a = star share with Max.
        let opaque = [
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::COLOR,
            }),
            Some(lens_mask_target()),
        ];
        // Coma and tails: added to hdr's colour; lens_mask untouched (optically thin).
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let glow = [
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState {
                    color: add,
                    alpha: add,
                }),
                write_mask: wgpu::ColorWrites::COLOR,
            }),
            Some(wgpu::ColorTargetState {
                format: LENS_MASK_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }),
        ];
        let nucleus_vertex = [Some(wgpu::VertexBufferLayout {
            array_stride: size_of::<[f32; 6]>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
        })];
        let pipeline = |label: &str,
                        vs: &str,
                        fs: &str,
                        buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
                        targets: &[Option<wgpu::ColorTargetState>],
                        opaque: bool| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                // The nucleus mesh is closed and wound counter-clockwise seen from outside:
                // back faces are culled (a camera inside it sees nothing). The sprites, quads
                // and ribbons face the camera by construction.
                primitive: wgpu::PrimitiveState {
                    cull_mode: opaque.then_some(wgpu::Face::Back),
                    ..Default::default()
                },
                // Reversed-Z: the nucleus writes depth; the glow is hidden behind bodies (and
                // behind the nucleus) but writes none, so later passes see what is behind it.
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(opaque),
                    depth_compare: Some(wgpu::CompareFunction::Greater),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let nucleus_pipeline = pipeline(
            "comet nuclei",
            "vs_nucleus",
            "fs_nucleus",
            &nucleus_vertex,
            &opaque,
            true,
        );
        let coma_pipeline = pipeline("comet comae", "vs_coma", "fs_coma", &[], &glow, false);
        let ion_pipeline = pipeline("comet ion tails", "vs_ion", "fs_ion", &[], &glow, false);
        let dust_pipeline = pipeline("comet dust tails", "vs_dust", "fs_dust", &[], &glow, false);
        let nuclei = RecordBuffer::new(
            device,
            "comet nuclei",
            size_of::<NucleusGpu>(),
            INITIAL_NUCLEI,
        );
        let comae = RecordBuffer::new(device, "comet comae", size_of::<ComaGpu>(), INITIAL_COMAE);
        let tails_buffer = storage_buffer(
            device,
            "comet tails",
            (MAX_TAILED_COMETS * size_of::<TailGpu>()) as u64,
        );
        let grains_buffer = storage_buffer(
            device,
            "comet dust grains",
            (MAX_TAILED_COMETS * GRAINS_PER_TAIL * size_of::<GrainGpu>()) as u64,
        );
        let ico = icosphere(NUCLEUS_SUBDIVISIONS);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("comet nucleus indices"),
            contents: bytemuck::cast_slice(&ico.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let index_count = u32::try_from(ico.indices.len()).context("nucleus index count")?;
        let bind_group = comet_bind_group(
            device,
            &bgl,
            [&nuclei.buffer, &comae.buffer, &tails_buffer, &grains_buffer],
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the comet pass: {err}");
        }
        Ok(Self {
            input,
            nucleus_pipeline,
            coma_pipeline,
            ion_pipeline,
            dust_pipeline,
            bgl,
            bind_group,
            nuclei,
            comae,
            tails_buffer,
            grains_buffer,
            ico,
            index_buffer,
            index_count,
            meshes: HashMap::new(),
            manager: TailManager::default(),
            slots: [None; MAX_TAILED_COMETS],
            slot_splats: [0; MAX_TAILED_COMETS],
            nucleus_draws: Vec::new(),
            coma_count: 0,
            tail_draws: Vec::new(),
            clock: Instant::now(),
            frame: 0,
            us_per_solve: KEPLER_SOLVE_US,
            settle: false,
            cpu_ms: 0.0,
        })
    }

    /// The last frame's tail CPU time [ms] (`--bench` reports it as `comets_cpu_ms`): the whole
    /// `prepare` (culling every listed comet, the tail work, the uploads).
    pub fn cpu_ms(&self) -> f32 {
        self.cpu_ms
    }

    /// With `on`, every due tail refresh completes within the frame instead of within the 1 ms
    /// share, so a capture's first frame already shows complete tails (captures and their
    /// numeric checks; the window leaves it off). The wall-time and sim-time refresh rules
    /// still apply.
    pub fn set_settle_tails(&mut self, on: bool) {
        self.settle = on;
    }

    /// Uploads `plan`'s records, builds the nucleus meshes it needs, and assigns tail slots.
    fn upload(&mut self, gpu: &GpuContext, plan: &FramePlan) {
        // Nuclei: one mesh per comet, rebuilt when its seed or radius changes.
        self.nucleus_draws.clear();
        let mut nuclei: Vec<NucleusGpu> = Vec::with_capacity(plan.nuclei.len());
        for draw in &plan.nuclei {
            let stale = self
                .meshes
                .get(&draw.id)
                .is_none_or(|m| m.seed != draw.seed || m.radius_m != draw.radius_m);
            if stale {
                let positions = nucleus_vertices(&self.ico, draw.seed, draw.radius_m);
                if positions.is_empty() {
                    continue;
                }
                let normals = vertex_normals(&positions, &self.ico.indices);
                let interleaved: Vec<[f32; 6]> = positions
                    .iter()
                    .zip(&normals)
                    .map(|(p, n)| [p.x, p.y, p.z, n.x, n.y, n.z])
                    .collect();
                let vertices = gpu
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("comet nucleus vertices"),
                        contents: bytemuck::cast_slice(&interleaved),
                        usage: wgpu::BufferUsages::VERTEX,
                    });
                self.meshes.insert(
                    draw.id,
                    NucleusMesh {
                        vertices,
                        seed: draw.seed,
                        radius_m: draw.radius_m,
                        last_frame: self.frame,
                    },
                );
            }
            if let Some(mesh) = self.meshes.get_mut(&draw.id) {
                mesh.last_frame = self.frame;
                self.nucleus_draws.push(draw.id);
                nuclei.push(draw.gpu);
            }
        }
        let frame = self.frame;
        self.meshes
            .retain(|_, m| frame.saturating_sub(m.last_frame) < MESH_KEEP_FRAMES);
        let mut grew = self.nuclei.write(gpu, bytemuck::cast_slice(&nuclei));
        grew |= self.comae.write(gpu, bytemuck::cast_slice(&plan.comae));
        self.coma_count = plan.comae.len() as u32;
        if grew {
            self.bind_group = comet_bind_group(
                &gpu.device,
                &self.bgl,
                [
                    &self.nuclei.buffer,
                    &self.comae.buffer,
                    &self.tails_buffer,
                    &self.grains_buffer,
                ],
            );
        }

        // Tails: a comet keeps its slot while it draws a tail; its grains are uploaded when
        // its dust tail's version changes.
        for slot in &mut self.slots {
            if slot.is_some_and(|(id, _)| !plan.tails.iter().any(|t| t.id == id)) {
                *slot = None;
            }
        }
        let mut records = [TailGpu::default(); MAX_TAILED_COMETS];
        self.tail_draws.clear();
        for tail in &plan.tails {
            let held = self
                .slots
                .iter()
                .position(|s| s.is_some_and(|(id, _)| id == tail.id));
            let Some(slot) = held.or_else(|| self.slots.iter().position(Option::is_none)) else {
                continue;
            };
            let uploaded = self.slots[slot].and_then(|(_, v)| v);
            self.slots[slot] = Some((tail.id, uploaded));
            if let Some(version) = tail.dust_version.filter(|&v| Some(v) != uploaded)
                && let Some(state) = self.manager.tails.get(&tail.id)
            {
                let grains: Vec<GrainGpu> = state
                    .grains
                    .iter()
                    .take(GRAINS_PER_TAIL)
                    .map(GrainGpu::of)
                    .collect();
                let offset = (slot * GRAINS_PER_TAIL * size_of::<GrainGpu>()) as u64;
                if !grains.is_empty() {
                    gpu.queue.write_buffer(
                        &self.grains_buffer,
                        offset,
                        bytemuck::cast_slice(&grains),
                    );
                }
                self.slot_splats[slot] = grains.len() as u32;
                self.slots[slot] = Some((tail.id, Some(version)));
            }
            let has_dust = self.slots[slot].is_some_and(|(_, v)| v.is_some());
            records[slot] = tail.gpu;
            let splats = if has_dust { self.slot_splats[slot] } else { 0 };
            self.tail_draws.push((slot as u32, splats));
        }
        if !self.tail_draws.is_empty() {
            gpu.queue
                .write_buffer(&self.tails_buffer, 0, bytemuck::cast_slice(&records));
        }
    }
}

fn comet_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    buffers: [&wgpu::Buffer; COMET_BINDINGS as usize],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = buffers
        .iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("comets bg"),
        layout: bgl,
        entries: &entries,
    })
}

impl RenderFeature for CometRenderer {
    fn name(&self) -> &'static str {
        "comets"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, _targets: &FrameTargets) {
        let start = Instant::now();
        self.frame += 1;
        let wall_s = self.clock.elapsed().as_secs_f64();
        let max_solves = if self.settle {
            usize::MAX
        } else {
            (1_000.0 * TAIL_CPU_SHARE_MS / self.us_per_solve) as usize
        };
        let input = self.input.clone();
        let manager = &mut self.manager;
        let plan = input.with(|scene| {
            plan_frame(
                scene,
                &ctx.camera,
                ctx.viewport,
                wall_s,
                manager,
                max_solves,
            )
        });
        if plan.work.solves > 0 && !self.settle {
            // The whole plan's time per solve (culling included): a cost estimate on the safe side.
            let us = start.elapsed().as_secs_f64() * 1e6 / plan.work.solves as f64;
            self.us_per_solve = 0.8 * self.us_per_solve + 0.2 * us.clamp(0.01, 1_000.0);
        }
        self.upload(gpu, &plan);
        self.cpu_ms = (start.elapsed().as_secs_f64() * 1e3) as f32;
    }

    fn encode(
        &mut self,
        _gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        timer: &mut PassTimer,
    ) {
        if self.nucleus_draws.is_empty() && self.coma_count == 0 && self.tail_draws.is_empty() {
            return;
        }
        let load = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("comets"),
            color_attachments: &[load(&targets.hdr_view), load(&targets.lens_mask_view)],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timer.render_writes("comets"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        if !self.nucleus_draws.is_empty() {
            pass.set_pipeline(&self.nucleus_pipeline);
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            for (k, id) in self.nucleus_draws.iter().enumerate() {
                let Some(mesh) = self.meshes.get(id) else {
                    continue;
                };
                let k = k as u32;
                pass.set_vertex_buffer(0, mesh.vertices.slice(..));
                pass.draw_indexed(0..self.index_count, 0, k..k + 1);
            }
        }
        if self.coma_count > 0 {
            pass.set_pipeline(&self.coma_pipeline);
            pass.draw(0..6, 0..self.coma_count);
        }
        if !self.tail_draws.is_empty() {
            pass.set_pipeline(&self.ion_pipeline);
            for &(slot, _) in &self.tail_draws {
                pass.draw(0..6 * ION_SEGMENTS, slot..slot + 1);
            }
            pass.set_pipeline(&self.dust_pipeline);
            let per_tail = GRAINS_PER_TAIL as u32;
            for &(slot, splats) in self.tail_draws.iter().filter(|(_, n)| *n > 0) {
                pass.draw(0..6, slot * per_tail..slot * per_tail + splats);
            }
        }
    }
}

/// `ShaderLib` path of the comet probes.
pub(crate) const COMET_PROBE_WGSL_PATH: &str = "probe/planet_comet.wgsl";

/// Compute entry points for the comet probes; input and output at group 2 (`comet.wgsl` owns
/// groups 0 and 1).
pub(crate) const COMET_PROBE_WGSL: &str = r#"// probe/planet_comet.wgsl: `universe --selftest` probes of comet.wgsl (Task C1).
#import "planet/comet.wgsl"

@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

// Twins of ION_PROBE_INTENSITY_CD, ION_PROBE_PIXEL_M, SPRITE_PROBE_SIGMA_M and the DUST_PROBE_*
// constants in comet/mod.rs.
const ION_PROBE_INTENSITY: f32 = 1.0e14;
const ION_PROBE_PIXEL: f32 = 1.2e8;
const SPRITE_PROBE_SIGMA: f32 = 3.0e7;
const DUST_PROBE_SUN_X: f32 = -1.8e11;
const DUST_PROBE_SUN_Y: f32 = 2.0e10;
const DUST_PROBE_OFFSET: f32 = 1.0e10;
const DUST_PROBE_NUCLEUS: f32 = 1.0e11;
const DUST_PROBE_WEIGHT: f32 = 4.8828125e-4;

// in: (ρ, I, ℓ, pixel); out: (coma_surface_brightness, coma_drawn_brightness, 0, 0).
@compute @workgroup_size(64, 1, 1)
fn cs_coma(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(
        coma_surface_brightness(c.x, c.y, c.z),
        coma_drawn_brightness(c.x, c.y, c.z, c.w),
        0.0,
        0.0,
    );
}

// in: (s, x, L, sin ψ); out: (ion_tail_brightness at ION_PROBE_INTENSITY,
// ion_tail_drawn_brightness at the σ of a pixel of ION_PROBE_PIXEL × s/L, that σ
// (ion_drawn_sigma), 0).
@compute @workgroup_size(64, 1, 1)
fn cs_ion(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let sigma = ion_drawn_sigma(ION_PROBE_PIXEL * c.x / c.z);
    probe_out[gid.x] = vec4<f32>(
        ion_tail_brightness(c.x, c.y, c.z, ION_PROBE_INTENSITY, c.w),
        ion_tail_drawn_brightness(c.x, c.y, c.z, ION_PROBE_INTENSITY, c.w, sigma),
        sigma,
        0.0,
    );
}

// C1L. in: four vec4 per case, (L00, L10, L20, L11), (L21, L22, floor, 0), (right, 0), (up, 0)
// (L the covariance's Cholesky factor); out[case]: splat_axes's ellipse rebuilt as the
// projected covariance
// (σ₁²c² + σ₂²s², (σ₁² − σ₂²)cs, σ₁²s² + σ₂²c², σ₁σ₂), c, s = cos φ, sin φ.
@compute @workgroup_size(64, 1, 1)
fn cs_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in) / 4u) {
        return;
    }
    let i = 4u * gid.x;
    let a = probe_in[i];
    let b = probe_in[i + 1u];
    let e = splat_axes(a, b.xy, probe_in[i + 2u].xyz, probe_in[i + 3u].xyz, b.z);
    let c = cos(e.z);
    let s = sin(e.z);
    let l1 = e.x * e.x;
    let l2 = e.y * e.y;
    probe_out[gid.x] = vec4<f32>(l1 * c * c + l2 * s * s, (l1 - l2) * c * s, l1 * s * s + l2 * c * c, e.x * e.y);
}

// in: (cos θ, r/σ, μ₀, μ); out: (dust_phase, sprite_profile at σ = SPRITE_PROBE_SIGMA,
// nucleus_reflectance, dust_grain_light of the grain at (r/σ, μ₀, μ) × DUST_PROBE_OFFSET from a
// nucleus at (cos θ, 0.5, −1) × DUST_PROBE_NUCLEUS, the Sun at (DUST_PROBE_SUN_X, _Y, 0)).
@compute @workgroup_size(64, 1, 1)
fn cs_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    let offset = vec3<f32>(c.y, c.z, c.w) * DUST_PROBE_OFFSET;
    let nucleus = vec3<f32>(c.x, 0.5, -1.0) * DUST_PROBE_NUCLEUS;
    let sun = vec3<f32>(DUST_PROBE_SUN_X, DUST_PROBE_SUN_Y, 0.0);
    probe_out[gid.x] = vec4<f32>(
        dust_phase(c.x),
        sprite_profile(c.y * SPRITE_PROBE_SIGMA, SPRITE_PROBE_SIGMA),
        nucleus_reflectance(c.z, c.w),
        dust_grain_light(offset, DUST_PROBE_WEIGHT, nucleus, sun),
    );
}
"#;

/// The ion tail's intensity in the probe [cd] (twin of `ION_PROBE_INTENSITY`).
const ION_PROBE_INTENSITY_CD: f64 = 1.0e14;
/// The ion probe's pixel scale [m] (twin of `ION_PROBE_PIXEL`): case (s, x, L, sin ψ) draws the
/// ribbon for a pixel of `ION_PROBE_PIXEL_M` × s/L ([`ion_probe_pixel_m`]), so the cases with
/// s/L > 0.47 are widened by the pixel floor and the others are not.
const ION_PROBE_PIXEL_M: f64 = 1.2e8;
/// The grain-light probe's geometry (twins of `DUST_PROBE_*`): the Sun relative to the nucleus
/// [m], the grain offset's and the nucleus's scales [m], and the grain's weight.
const DUST_PROBE_SUN_M: DVec3 = DVec3::new(-1.8e11, 2.0e10, 0.0);
const DUST_PROBE_OFFSET_M: f64 = 1.0e10;
const DUST_PROBE_NUCLEUS_M: f64 = 1.0e11;
const DUST_PROBE_WEIGHT: f64 = 1.0 / 2_048.0;

/// The pixel size [m] the ion probe's case `c` = (s, x, L, sin ψ) widens the ribbon for.
fn ion_probe_pixel_m(c: &[f32; 4]) -> f64 {
    ION_PROBE_PIXEL_M * f64::from(c[0]) / f64::from(c[2])
}

/// The grain-light geometry of phase-probe case `c` = (cos θ, r/σ, μ₀, μ), as `cs_phase` builds
/// it: the grain's offset (r/σ, μ₀, μ) × 10¹⁰ m, the nucleus at (cos θ, 0.5, −1) × 10¹¹ m from
/// the camera, the Sun at [`DUST_PROBE_SUN_M`] from the nucleus. Returns (offset, nucleus_rel,
/// sun_rel).
fn dust_probe_geometry(c: &[f32; 4]) -> (DVec3, DVec3, DVec3) {
    let [a, b, d, e] = c.map(f64::from);
    (
        DVec3::new(b, d, e) * DUST_PROBE_OFFSET_M,
        DVec3::new(a, 0.5, -1.0) * DUST_PROBE_NUCLEUS_M,
        DUST_PROBE_SUN_M,
    )
}
/// The sprite σ in the probe [m] (twin of `SPRITE_PROBE_SIGMA`).
const SPRITE_PROBE_SIGMA_M: f64 = 3.0e7;
/// Largest relative |GPU − CPU| of the coma profiles (plan C1: ±1e-3). A handful of f32
/// operations and one `exp` (a few ulp under Metal's fast math) err by ~1e-6; a wrong K, scale
/// or softening errs by ≥ 1e-2.
const COMA_PROBE_TOLERANCE: f64 = 1e-3;
/// Largest |GPU − CPU| of the ion profile and the sprite, relative to max(|CPU|, 10⁻³ × the
/// profile's peak) (near the tip the profile is a difference of two exponentials, which f32
/// resolves only to ~1e-7 of the peak).
const TAIL_PROBE_TOLERANCE: f64 = 1e-3;
/// Largest relative |GPU − CPU| of the HG phase function, the grain light and the ribbon's
/// drawn σ, and absolute of the reflectance (a few f32 operations: ~1e-6; the HG's slope
/// amplifies an error in cos θ by at most 3g/(1 − g)² ≈ 11).
const PHASE_PROBE_TOLERANCE: f64 = 1e-4;

/// Coma probe inputs: 64 cases (ρ, I, ℓ, pixel) over ρ/ℓ from 10⁻³ to 20, three scales, three
/// intensities, pixels from 10⁻⁴ ℓ (resolved) to 5 ℓ (sub-pixel coma).
pub(crate) fn coma_probe_inputs() -> Vec<[f32; 4]> {
    (0..64)
        .map(|k| {
            let scale = [1.0e8_f64, 2.4e9, 9.0e6][k % 3];
            let intensity = [1.0e12_f64, 3.0e15, 2.0e17][(k / 3) % 3];
            let pixel = scale * [1e-3, 0.3, 5.0, 1e-4][k % 4];
            let rho = scale * 10f64.powf(-3.0 + 4.3 * k as f64 / 63.0);
            [rho as f32, intensity as f32, scale as f32, pixel as f32]
        })
        .collect()
}

/// Ion-tail probe inputs: 64 cases (s, x, L, sin ψ): s from the base to 0.9 L, x across ±3σ,
/// three lengths, sin ψ from 1 to 10⁻⁴ (the axial cap).
pub(crate) fn ion_probe_inputs() -> Vec<[f32; 4]> {
    let sigma = ion_sigma_m();
    (0..64)
        .map(|k| {
            let length = [0.06, 0.3, 1.0][k % 3] * AU;
            let s = 0.9 * length * k as f64 / 63.0;
            let x = sigma * (-3.0 + 6.0 * ((k * 7) % 64) as f64 / 63.0);
            let sin_psi = [1.0, 0.4, 0.05, 1e-4][k % 4];
            [s as f32, x as f32, length as f32, sin_psi as f32]
        })
        .collect()
}

/// Phase and sprite probe inputs: 64 cases (cos θ, r/σ, μ₀, μ): cos θ over [−1, 1], r up to
/// 2.9σ, and both cosines over ranges that include unlit and back-facing points.
pub(crate) fn phase_probe_inputs() -> Vec<[f32; 4]> {
    (0..64)
        .map(|k| {
            let cos_theta = -1.0 + 2.0 * k as f64 / 63.0;
            let r = 2.9 * ((k * 5) % 64) as f64 / 63.0;
            let mu0 = -0.2 + 1.2 * ((k * 11) % 64) as f64 / 63.0;
            let mu = -0.1 + 1.1 * ((k * 13) % 64) as f64 / 63.0;
            [cos_theta as f32, r as f32, mu0 as f32, mu as f32]
        })
        .collect()
}

/// Splat probe inputs (C1L): 64 cases of four vec4 each, (L₀₀, L₁₀, L₂₀, L₁₁), (L₂₁, L₂₂,
/// floor, 0), (right, 0), (up, 0), L the Cholesky factor of covariances from two tangent
/// vectors and an ejection spread of 10⁶–10⁹ m (round, elongated, needles, flat and edge-on),
/// floors 10⁵–10⁸ m, and sprite planes at 64 orientations.
pub(crate) fn splat_probe_inputs() -> Vec<[f32; 4]> {
    let mut out = Vec::with_capacity(256);
    for k in 0..64_u32 {
        let f = f64::from(k);
        let a = DVec3::new((0.7 * f).cos(), (1.3 * f).sin(), (0.4 * f).cos())
            * 10f64.powf(7.0 + 3.0 * ((k * 5) % 64) as f64 / 63.0);
        let b = DVec3::new((2.1 * f).sin(), (0.5 * f).cos(), (1.7 * f).sin())
            * 10f64.powf(6.0 + 3.0 * ((k * 11) % 64) as f64 / 63.0);
        let spread = if k % 4 == 0 {
            0.0
        } else {
            10f64.powf(6.0 + 3.0 * ((k * 13) % 64) as f64 / 63.0)
        };
        let floor = 10f64.powf(5.0 + 3.0 * ((k * 7) % 64) as f64 / 63.0);
        // Every 8th case a needle (one long axis over a 10⁵ m floor), every 8th a round splat.
        let (a, b, spread, floor) = match k % 8 {
            5 => (a.normalize() * 1.0e10, b * 1e-4, 0.0, 1.0e5),
            6 => (a * 1e-3, b * 1e-3, 1.0e9, floor),
            _ => (a, b, spread, floor),
        };
        let cov = tails_outer(a) * 0.25
            + tails_outer(b) * 0.25
            + DMat3::from_diagonal(DVec3::splat(spread * spread));
        let (yaw, pitch) = (0.1 + 0.37 * f, -1.2 + 2.4 * ((k * 3) % 64) as f64 / 63.0);
        let dir = DVec3::new(
            pitch.cos() * yaw.cos(),
            pitch.cos() * yaw.sin(),
            pitch.sin(),
        );
        // Edge-on cases: the line of sight in the plane of a and b.
        let dir = if k % 8 == 3 {
            (a.normalize() + 0.3 * b.normalize()).normalize()
        } else {
            dir
        };
        let right = dir.cross(DVec3::Z).normalize_or(DVec3::X);
        let up = right.cross(dir);
        let l = tails::cov_factor(cov);
        out.push([
            l.x_axis.x as f32,
            l.x_axis.y as f32,
            l.x_axis.z as f32,
            l.y_axis.y as f32,
        ]);
        out.push([l.y_axis.z as f32, l.z_axis.z as f32, floor as f32, 0.0]);
        out.push([right.x as f32, right.y as f32, right.z as f32, 0.0]);
        out.push([up.x as f32, up.y as f32, up.z as f32, 0.0]);
    }
    out
}

/// v vᵀ (the probe inputs' covariances).
fn tails_outer(v: DVec3) -> DMat3 {
    DMat3::from_cols(v * v.x, v * v.y, v * v.z)
}

/// A splat probe case's covariance factor L, floor and plane, as `cs_splat` reads them
/// (f32 → f64).
fn splat_probe_case(c: &[[f32; 4]]) -> (DMat3, f64, DVec3, DVec3) {
    let v = |i: usize| c[i].map(f64::from);
    let (a, b, r, u) = (v(0), v(1), v(2), v(3));
    let factor = DMat3::from_cols(
        DVec3::new(a[0], a[1], a[2]),
        DVec3::new(0.0, a[3], b[0]),
        DVec3::new(0.0, 0.0, b[1]),
    );
    (
        factor,
        b[2],
        DVec3::new(r[0], r[1], r[2]),
        DVec3::new(u[0], u[1], u[2]),
    )
}

/// Largest |GPU − CPU| of the splat ellipse's rebuilt covariance, relative to σ₁² (C1L): a
/// projection, a square root and an atan2 in f32 err by ~1e-6; a wrong axis or angle errs by
/// ≳ 1e-2.
const SPLAT_PROBE_TOLERANCE: f64 = 1e-3;

fn run_probe(
    gpu: &GpuContext,
    lib: &ShaderLib,
    entry: &str,
    input: &[[f32; 4]],
) -> anyhow::Result<Vec<[f32; 4]>> {
    let src = lib.compose(COMET_PROBE_WGSL_PATH)?;
    let n = u32::try_from(input.len()).context("too many probe cases")?;
    let bytes = run_compute(
        gpu,
        &src,
        entry,
        bytemuck::cast_slice(input),
        (input.len() * 16) as u64,
        [n.div_ceil(64), 1, 1],
    )?;
    Ok(bytemuck::pod_collect_to_vec(&bytes))
}

/// The worst of `errors`; NaN anywhere makes it NaN (which fails every check).
fn worst(errors: impl Iterator<Item = f64>) -> f64 {
    errors.fold(0.0_f64, |a, e| {
        if a.is_nan() || e.is_nan() {
            f64::NAN
        } else {
            a.max(e)
        }
    })
}

fn probe_result(name: &str, outcome: anyhow::Result<(bool, String)>) -> ProbeResult {
    let (pass, detail) = outcome.unwrap_or_else(|e| (false, format!("{e:#}")));
    ProbeResult {
        name: name.to_owned(),
        pass,
        detail,
    }
}

/// C1's probes, called from [`crate::gpu_probes`] after it registered the crate's shaders:
/// `coma_surface_brightness` and `coma_drawn_brightness` (64 radii, ±1e-3, plan C1),
/// `ion_tail_brightness`, `ion_tail_drawn_brightness` and `ion_drawn_sigma` (64 cases, both
/// sides of the pixel floor), `dust_phase`, `sprite_profile`, `nucleus_reflectance` and
/// `dust_grain_light` (64 cases), and `splat_axes` (64 cases, C1L).
pub(crate) fn comet_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    lib.add(COMET_PROBE_WGSL_PATH, COMET_PROBE_WGSL);
    let lib = &*lib;
    let rel = |g: f32, c: f64, floor: f64| (f64::from(g) - c).abs() / c.abs().max(floor);
    let coma = || -> anyhow::Result<(bool, String)> {
        let input = coma_probe_inputs();
        let out = run_probe(gpu, lib, "cs_coma", &input)?;
        let err = worst(input.iter().zip(&out).flat_map(|(c, g)| {
            let [rho, i, l, a] = c.map(f64::from);
            [
                rel(g[0], coma_surface_brightness(rho, i, l), f64::MIN_POSITIVE),
                rel(g[1], coma_drawn_brightness(rho, i, l, a), f64::MIN_POSITIVE),
            ]
        }));
        Ok((
            err <= COMA_PROBE_TOLERANCE,
            format!(
                "{} radii, exact and drawn profiles, max relative |GPU − CPU| {err:.1e} \
                 (tolerance {COMA_PROBE_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let ion = || -> anyhow::Result<(bool, String)> {
        let input = ion_probe_inputs();
        let out = run_probe(gpu, lib, "cs_ion", &input)?;
        let (mut exact, mut drawn, mut sigmas) = (Vec::new(), Vec::new(), Vec::new());
        for (c, g) in input.iter().zip(&out) {
            let [s, x, l, sin_psi] = c.map(f64::from);
            let i = ION_PROBE_INTENSITY_CD;
            let peak = ion_tail_brightness(0.0, 0.0, l, i, sin_psi);
            exact.push(rel(
                g[0],
                ion_tail_brightness(s, x, l, i, sin_psi),
                1e-3 * peak,
            ));
            let sigma = ion_drawn_sigma_m(ion_probe_pixel_m(c));
            let peak = ion_tail_drawn_brightness(0.0, 0.0, l, i, sin_psi, sigma);
            let cpu = ion_tail_drawn_brightness(s, x, l, i, sin_psi, sigma);
            drawn.push(rel(g[1], cpu, 1e-3 * peak));
            sigmas.push(rel(g[2], sigma, f64::MIN_POSITIVE));
        }
        let (exact, drawn, sigmas) = (
            worst(exact.into_iter()),
            worst(drawn.into_iter()),
            worst(sigmas.into_iter()),
        );
        Ok((
            exact <= TAIL_PROBE_TOLERANCE
                && drawn <= TAIL_PROBE_TOLERANCE
                && sigmas <= PHASE_PROBE_TOLERANCE,
            format!(
                "{} cases, max |GPU − CPU| of max(|CPU|, 1e-3 peak): exact {exact:.1e}, drawn \
                 (pixel floor) {drawn:.1e} (tolerance {TAIL_PROBE_TOLERANCE:.0e}); \
                 ion_drawn_sigma rel {sigmas:.1e} (tolerance {PHASE_PROBE_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let phase = || -> anyhow::Result<(bool, String)> {
        let input = phase_probe_inputs();
        let out = run_probe(gpu, lib, "cs_phase", &input)?;
        let sigma = SPRITE_PROBE_SIGMA_M;
        let peak = sprite_profile(0.0, sigma);
        let (mut hg, mut sprite, mut refl, mut grain) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (c, g) in input.iter().zip(&out) {
            let [cos_theta, r, mu0, mu] = c.map(f64::from);
            hg.push(rel(g[0], dust_phase(cos_theta), f64::MIN_POSITIVE));
            sprite.push(rel(g[1], sprite_profile(r * sigma, sigma), 1e-3 * peak));
            let cpu = uni_physics::photometry::reflectance(NUCLEUS_BRDF, mu0, mu);
            refl.push((f64::from(g[2]) - cpu).abs());
            let (offset, nucleus_rel, sun_rel) = dust_probe_geometry(c);
            let cpu = dust_grain_light(offset, DUST_PROBE_WEIGHT, nucleus_rel, sun_rel);
            grain.push(rel(g[3], cpu, f64::MIN_POSITIVE));
        }
        let (hg, sprite, refl, grain) = (
            worst(hg.into_iter()),
            worst(sprite.into_iter()),
            worst(refl.into_iter()),
            worst(grain.into_iter()),
        );
        Ok((
            hg <= PHASE_PROBE_TOLERANCE
                && sprite <= TAIL_PROBE_TOLERANCE
                && refl <= PHASE_PROBE_TOLERANCE
                && grain <= PHASE_PROBE_TOLERANCE,
            format!(
                "{} cases: dust_phase rel {hg:.1e}, sprite_profile {sprite:.1e}, \
                 nucleus_reflectance abs {refl:.1e}, dust_grain_light rel {grain:.1e} \
                 (tolerances {PHASE_PROBE_TOLERANCE:.0e}, {TAIL_PROBE_TOLERANCE:.0e}, \
                 {PHASE_PROBE_TOLERANCE:.0e}, {PHASE_PROBE_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let splat = || -> anyhow::Result<(bool, String)> {
        let input = splat_probe_inputs();
        let out = run_probe(gpu, lib, "cs_splat", &input)?;
        let err = worst(input.chunks(4).zip(&out).flat_map(|(c, g)| {
            let (factor, floor, right, up) = splat_probe_case(c);
            let (s1, s2, phi) = splat_axes(factor, right, up, floor);
            let (cs, sn) = (phi.cos(), phi.sin());
            let (l1, l2) = (s1 * s1, s2 * s2);
            let cpu = [
                l1 * cs * cs + l2 * sn * sn,
                (l1 - l2) * cs * sn,
                l1 * sn * sn + l2 * cs * cs,
            ];
            let area = s1 * s2;
            [
                (f64::from(g[0]) - cpu[0]).abs() / l1,
                (f64::from(g[1]) - cpu[1]).abs() / l1,
                (f64::from(g[2]) - cpu[2]).abs() / l1,
                (f64::from(g[3]) - area).abs() / area,
            ]
        }));
        Ok((
            err <= SPLAT_PROBE_TOLERANCE,
            format!(
                "{} cases, max |GPU − CPU| of the rebuilt ellipse / σ₁² and of σ₁σ₂ {err:.1e} \
                 (tolerance {SPLAT_PROBE_TOLERANCE:.0e})",
                input.len() / 4
            ),
        ))
    };
    vec![
        probe_result("coma_surface_brightness", coma()),
        probe_result("ion_tail_brightness", ion()),
        probe_result("comet_phase_and_sprite", phase()),
        probe_result("comet_dust_splat", splat()),
    ]
}

/// Test fixtures shared by the comet modules' tests.
#[cfg(test)]
pub(crate) mod test_support {
    use glam::DVec3;
    use uni_core::{AU, Epoch};
    use uni_physics::kepler::Conic;
    use uni_physics::photometry::comet_total_v;
    use uni_physics::smallbody::{conic_from_perihelion, heliocentric_state_icrs};
    use uni_sim::registry::ObjectId;
    use uni_sim::snapshot::{CometView, proj_radius_px};

    /// Internal frame height and field of view the fixtures project with.
    pub(crate) const HEIGHT_PX: f64 = 1080.0;
    pub(crate) const FOV_Y_RAD: f64 = std::f64::consts::FRAC_PI_3;

    fn view(
        id: ObjectId,
        rel: DVec3,
        sun_rel: DVec3,
        vel: DVec3,
        conic: Conic,
        epoch: Epoch,
        m1: Option<f64>,
        k1: Option<f64>,
    ) -> CometView {
        let (r_au, delta_au) = (sun_rel.length() / AU, rel.length() / AU);
        CometView {
            id,
            name: format!("comet {id}"),
            nucleus_rel_m: rel,
            nucleus_radius_m: 2_000.0,
            radius_assumed: true,
            spkid: 1_000_000 + id as i32,
            sun_rel_m: sun_rel,
            helio_vel_mps: vel,
            r_au,
            delta_au,
            total_v: m1
                .zip(k1)
                .map(|(m1, k1)| comet_total_v(m1, k1, r_au, delta_au)),
            m1,
            k1,
            conic,
            conic_epoch: epoch,
            proj_radius_px: proj_radius_px(2_000.0, rel.length(), FOV_Y_RAD, HEIGHT_PX),
        }
    }

    /// A comet with nucleus `rel` from the camera and the Sun `sun_rel` from it, at rest, on a
    /// circular J2000-ecliptic orbit of its heliocentric radius (only the tails use the orbit).
    pub(crate) fn static_view(
        id: ObjectId,
        rel: DVec3,
        sun_rel: DVec3,
        m1: Option<f64>,
        k1: Option<f64>,
    ) -> CometView {
        let conic = conic_from_perihelion(sun_rel.length() / AU, 0.0, 0.0, 0.0, 0.0, 2_451_545.0);
        view(id, rel, sun_rel, DVec3::ZERO, conic, Epoch::J2000, m1, k1)
    }

    /// The comet on `conic` at `epoch`, seen from heliocentric position `cam_helio` [m].
    pub(crate) fn comet_on(
        id: ObjectId,
        conic: Conic,
        epoch: Epoch,
        cam_helio: DVec3,
        m1: f64,
        k1: f64,
    ) -> CometView {
        let (r, v) = heliocentric_state_icrs(&conic, epoch);
        view(id, r - cam_helio, -r, v, conic, epoch, Some(m1), Some(k1))
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use glam::DQuat;
    use uni_core::{AU, DAY, UniversePos};
    use uni_gfx::{ShaderLib, validate_wgsl};
    use uni_physics::smallbody::{conic_from_perihelion, heliocentric_state_icrs};
    use wgpu::naga;

    use super::tails::{GRAINS_PER_TAIL, ION_MIN_SIGMA_PX, ION_SEGMENTS, dust_grain_light};
    use super::test_support::{comet_on, static_view};
    use super::*;
    use crate::body::exposure_weight;

    fn camera() -> CameraView {
        CameraView {
            pos: UniversePos::from_meters(DVec3::ZERO),
            orient: DQuat::IDENTITY,
            fov_y_rad: test_support::FOV_Y_RAD,
            velocity_mps: DVec3::ZERO,
        }
    }

    const VIEWPORT: UVec2 = UVec2::new(1920, 1080);

    fn scene(comets: Vec<CometView>, eye_adapted: bool) -> CometScene {
        CometScene {
            comets,
            epoch: Epoch::J2000,
            eye_adapted,
        }
    }

    /// Plan C1: only comets with r_h < 5 AU get a coma and tails. At 5.1 AU nothing but the
    /// (unresolved) nucleus point would show; at 4.9 AU the coma, the ion tail and, once its
    /// refresh has run, the dust tail are drawn.
    #[test]
    fn inactive_beyond_5au() {
        for (r_au, active) in [(5.1, false), (4.9, true)] {
            let sun_rel = DVec3::new(-r_au * AU, 0.0, 0.0);
            let rel = DVec3::new(0.0, 0.0, -0.5 * AU);
            let view = static_view(1, rel, sun_rel, Some(6.0), Some(10.0));
            let mut manager = TailManager::default();
            let plan = plan_frame(
                &scene(vec![view], false),
                &camera(),
                VIEWPORT,
                0.0,
                &mut manager,
                usize::MAX,
            );
            assert!(
                plan.nuclei.is_empty(),
                "a 2 km nucleus 0.5 AU away is a point"
            );
            if active {
                assert_eq!(plan.comae.len(), 1, "r_h {r_au}: no coma");
                assert_eq!(plan.tails.len(), 1, "r_h {r_au}: no tail");
                assert!(
                    plan.tails[0].dust_version.is_some(),
                    "r_h {r_au}: no dust tail"
                );
                assert!(plan.comae[0].profile[0] > 0.0);
                assert!(plan.tails[0].gpu.nucleus[3] > 0.0 && plan.tails[0].gpu.sun[3] > 0.0);
            } else {
                assert!(plan.comae.is_empty(), "r_h {r_au}: coma drawn");
                assert!(plan.tails.is_empty(), "r_h {r_au}: tail drawn");
                assert!(manager.tails.is_empty() && plan.work.solves == 0);
            }
        }
    }

    /// A nucleus is drawn as a mesh only once resolved (≥ POINT_LOD_PX), with the body's LOD
    /// exposure weight; never with the camera inside it; and comets behind the camera are culled.
    #[test]
    fn nucleus_resolved_and_comets_culled() {
        let sun_rel = DVec3::new(-AU, 0.0, 0.0);
        // 2 km nucleus at 1 000 km: 2 000/1e6/tan(30°)·540 = 1.87 px.
        for (dist, eye, drawn) in [
            (1.0e6, true, true),
            (3.0e6, true, false),
            (300.0, false, false),
        ] {
            let view = static_view(
                2,
                DVec3::new(0.0, 0.0, -dist),
                sun_rel,
                Some(6.0),
                Some(10.0),
            );
            let px = view.proj_radius_px;
            let mut m = TailManager::default();
            let plan = plan_frame(&scene(vec![view], eye), &camera(), VIEWPORT, 0.0, &mut m, 0);
            assert_eq!(
                plan.nuclei.len(),
                usize::from(drawn),
                "{dist} m ({px:.2} px)"
            );
            if drawn {
                assert_eq!(plan.nuclei[0].gpu.centre[3], exposure_weight(px, eye));
                assert!(plan.nuclei[0].gpu.centre[3] < 1.0, "LOD band at {px} px");
                assert_eq!(plan.nuclei[0].seed, 1_000_002);
            }
        }
        // Behind the camera, 1.5 AU away: the coma (≲ 10⁶ km) and the tails are culled.
        let behind = static_view(
            3,
            DVec3::new(0.0, 0.0, 1.5 * AU),
            sun_rel,
            Some(6.0),
            Some(10.0),
        );
        let mut m = TailManager::default();
        let plan = plan_frame(
            &scene(vec![behind], false),
            &camera(),
            VIEWPORT,
            0.0,
            &mut m,
            0,
        );
        assert!(plan.comae.is_empty() && plan.tails.is_empty());
    }

    /// C1L: the tail record's dust rate (`TailGpu::sun.w`) is the coma's light / τ, so the splats
    /// the pass draws (`dust_grain_light` × sun.w, from the record's f32 positions and weights)
    /// carry I_coma/τ × Σ w (r_h/r)² HG(θ): from afar, where every grain is seen at the
    /// nucleus's scattering angle, that is I_coma × H × HG(θ₀)/τ. The light follows the phase
    /// function (forward scattering brighter) instead of the v0 constant 0.5 × coma.
    #[test]
    fn dust_record_carries_the_emission_rate() {
        let conic = conic_from_perihelion(1.24, 0.64, 7.04, 50.1, 12.8, 2_459_885.5);
        let t = conic.tp.add_seconds(20.0 * DAY);
        let (r, _) = heliocentric_state_icrs(&conic, t);
        let to_sun = -r.normalize();
        let side = to_sun.any_orthonormal_vector();
        let xyz = |v: [f32; 4]| DVec3::new(v[0].into(), v[1].into(), v[2].into());
        let mut drawn_by_phase = Vec::new();
        for phase_deg in [10.0_f64, 90.0, 160.0] {
            let a = phase_deg.to_radians();
            let to_camera = a.cos() * to_sun + a.sin() * side;
            let view = comet_on(5, conic, t, r + 3.0 * AU * to_camera, 8.0, 10.0);
            let look = CameraView {
                orient: DQuat::from_rotation_arc(DVec3::NEG_Z, view.nucleus_rel_m.normalize()),
                ..camera()
            };
            let scene = CometScene {
                comets: vec![view.clone()],
                epoch: t,
                eye_adapted: false,
            };
            let mut manager = TailManager::default();
            let plan = plan_frame(&scene, &look, VIEWPORT, 0.0, &mut manager, usize::MAX);
            assert_eq!(plan.tails.len(), 1, "phase {phase_deg}°");
            assert!(plan.tails[0].dust_version.is_some());
            let gpu = plan.tails[0].gpu;
            let coma = coma_intensity_cd(&view);
            let rate = f64::from(gpu.sun[3]);
            assert!((rate / (coma / tails::DUST_HEAD_TIME_S) - 1.0).abs() < 1e-6);
            let grains = &manager.tails[&view.id].grains;
            let drawn: f64 = grains
                .iter()
                .map(|g| {
                    let rec = GrainGpu::of(g);
                    rate * dust_grain_light(
                        xyz(rec.offset),
                        f64::from(rec.offset[3]),
                        xyz(gpu.nucleus),
                        xyz(gpu.sun),
                    )
                })
                .sum();
            let h: f64 = grains.iter().map(|g| g.weight).sum();
            let lit: f64 = grains
                .iter()
                .map(|g| g.weight * r.length_squared() / (r + g.offset_m).length_squared())
                .sum::<f64>()
                / h;
            // The scattering angle at the nucleus: 180° − phase.
            let want = coma / tails::DUST_HEAD_TIME_S * h * lit * dust_phase(-a.cos());
            assert!(
                (drawn / want - 1.0).abs() < 0.02,
                "phase {phase_deg}°: {drawn:.4e} vs {want:.4e} cd"
            );
            drawn_by_phase.push(drawn / coma);
        }
        println!("dust / coma at phases 10°, 90°, 160°: {drawn_by_phase:.3?}");
        assert!(drawn_by_phase[0] < drawn_by_phase[1] && drawn_by_phase[1] < drawn_by_phase[2]);
    }

    /// naga's layout of one struct of `comet.wgsl`: its size and (name, offset) members.
    fn wgsl_struct(m: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>) {
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let fields = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (layouter[handle].size as usize, fields)
    }

    fn comet_module() -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(crate::COMET_WGSL).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"))
    }

    /// Global constraints: every GPU struct's size and member offsets match naga's layout of its
    /// WGSL twin.
    #[test]
    fn comet_layouts_match_wgsl() {
        let m = comet_module();
        let check = |name: &str, size: usize, rust: &[(&str, usize)]| {
            let (wsize, wfields) = wgsl_struct(&m, name);
            assert_eq!(wsize, size, "{name} size");
            let rust: Vec<(String, usize)> = rust.iter().map(|&(n, o)| (n.to_owned(), o)).collect();
            assert_eq!(wfields, rust, "{name} members");
        };
        check(
            "NucleusGpu",
            size_of::<NucleusGpu>(),
            &[
                ("centre", offset_of!(NucleusGpu, centre)),
                ("axis_x", offset_of!(NucleusGpu, axis_x)),
                ("axis_y", offset_of!(NucleusGpu, axis_y)),
                ("axis_z", offset_of!(NucleusGpu, axis_z)),
                ("sun", offset_of!(NucleusGpu, sun)),
                ("light", offset_of!(NucleusGpu, light)),
            ],
        );
        check(
            "ComaGpu",
            size_of::<ComaGpu>(),
            &[
                ("centre", offset_of!(ComaGpu, centre)),
                ("profile", offset_of!(ComaGpu, profile)),
                ("colour", offset_of!(ComaGpu, colour)),
            ],
        );
        check(
            "TailGpu",
            size_of::<TailGpu>(),
            &[
                ("nucleus", offset_of!(TailGpu, nucleus)),
                ("sun", offset_of!(TailGpu, sun)),
                ("ion_axis", offset_of!(TailGpu, ion_axis)),
                ("ion_colour", offset_of!(TailGpu, ion_colour)),
                ("dust_colour", offset_of!(TailGpu, dust_colour)),
            ],
        );
        check(
            "GrainGpu",
            size_of::<GrainGpu>(),
            &[
                ("offset", offset_of!(GrainGpu, offset)),
                ("cov_l0", offset_of!(GrainGpu, cov_l0)),
                ("cov_l1", offset_of!(GrainGpu, cov_l1)),
            ],
        );
        assert_eq!(
            [
                size_of::<NucleusGpu>(),
                size_of::<ComaGpu>(),
                size_of::<TailGpu>(),
                size_of::<GrainGpu>()
            ],
            [96, 48, 80, 48]
        );
    }

    /// `comet.wgsl` validates, has the four pipelines' entry points, binds group 1 as the Rust
    /// layout does, carries the WGSL twins with the CPU twins' arities, and its constants equal
    /// the Rust ones.
    #[test]
    fn comet_shader_validates() {
        let m = comet_module();
        let stages: Vec<(&str, naga::ShaderStage)> = m
            .entry_points
            .iter()
            .map(|e| (e.name.as_str(), e.stage))
            .collect();
        for part in ["nucleus", "coma", "ion", "dust"] {
            assert!(stages.contains(&(&format!("vs_{part}"), naga::ShaderStage::Vertex)));
            assert!(stages.contains(&(&format!("fs_{part}"), naga::ShaderStage::Fragment)));
        }
        let arity = |name: &str| {
            m.functions
                .iter()
                .find(|(_, f)| f.name.as_deref() == Some(name))
                .map(|(_, f)| f.arguments.len())
        };
        assert_eq!(arity("coma_surface_brightness"), Some(3));
        assert_eq!(arity("coma_drawn_brightness"), Some(4));
        assert_eq!(arity("ion_tail_brightness"), Some(5));
        assert_eq!(arity("ion_tail_drawn_brightness"), Some(6));
        assert_eq!(arity("ion_drawn_sigma"), Some(1));
        assert_eq!(arity("dust_grain_light"), Some(4));
        assert_eq!(arity("splat_axes"), Some(5));
        assert_eq!(arity("dust_phase"), Some(1));
        assert_eq!(arity("sprite_profile"), Some(2));
        assert_eq!(arity("nucleus_reflectance"), Some(2));
        let mut group1: Vec<u32> = m
            .global_variables
            .iter()
            .filter_map(|(_, v)| v.binding.as_ref())
            .filter(|b| b.group == 1)
            .map(|b| b.binding)
            .collect();
        group1.sort_unstable();
        assert_eq!(group1, (0..COMET_BINDINGS).collect::<Vec<_>>());
        let scalar = |name: &str| wgsl_scalar(&m, name);
        assert_eq!(scalar("GRAINS_PER_TAIL"), GRAINS_PER_TAIL as f64);
        assert_eq!(scalar("ION_SEGMENTS"), f64::from(ION_SEGMENTS));
        assert_eq!(scalar("ION_EFOLDS"), f32_of(tails::ION_TAIL_EFOLDS));
        assert_eq!(scalar("ION_WIDTH_M"), f32_of(tails::ION_TAIL_WIDTH_M));
        assert_eq!(
            scalar("ION_HALF_WIDTH_SIGMAS"),
            f32_of(tails::ION_HALF_WIDTH_SIGMAS)
        );
        assert_eq!(scalar("ION_MIN_SIGMA_PX"), f32_of(ION_MIN_SIGMA_PX));
        assert_eq!(scalar("DUST_G"), f32_of(tails::DUST_PHASE_G));
        assert_eq!(scalar("SPRITE_SIGMAS"), f32_of(tails::SPRITE_SIGMAS));
        assert_eq!(scalar("SPRITE_MIN_PX"), f32_of(tails::SPRITE_MIN_PX));
        assert_eq!(scalar("SPRITE_MAX_PX"), f32_of(SPRITE_MAX_PX));
        assert_eq!(scalar("GLOW_CULL_PRE"), f32_of(GLOW_CULL_PRE));
        assert_eq!(
            scalar("COMA_EXTENT_SCALES"),
            f32_of(coma::COMA_EXTENT_SCALES)
        );
        let nucleus_l = match nucleus::NUCLEUS_BRDF {
            uni_physics::photometry::Brdf::LunarLambert { l } => l,
            uni_physics::photometry::Brdf::Lambert => 0.0,
        };
        assert_eq!(scalar("NUCLEUS_L"), f32_of(nucleus_l));
    }

    /// The value of the scalar WGSL constant `name` in `m`.
    fn wgsl_scalar(m: &naga::Module, name: &str) -> f64 {
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("const {name}"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => f64::from(v),
            naga::Expression::Literal(naga::Literal::U32(v)) => f64::from(v),
            ref e => panic!("const {name} = {e:?}"),
        }
    }

    fn f32_of(x: f64) -> f64 {
        f64::from(x as f32)
    }

    /// The probe shader composes with `comet.wgsl` and validates, has the three compute entry
    /// points, and its constants equal the Rust twins the CPU side of the probes uses.
    #[test]
    fn comet_probe_shader_validates() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(COMET_PROBE_WGSL_PATH, COMET_PROBE_WGSL);
        let src = lib.compose(COMET_PROBE_WGSL_PATH).unwrap();
        let m = validate_wgsl(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        for entry in ["cs_coma", "cs_ion", "cs_phase", "cs_splat"] {
            assert!(
                m.entry_points
                    .iter()
                    .any(|e| e.name == entry && e.stage == naga::ShaderStage::Compute),
                "no entry point {entry}"
            );
        }
        let scalar = |name: &str| wgsl_scalar(&m, name);
        assert_eq!(
            scalar("ION_PROBE_INTENSITY"),
            f32_of(ION_PROBE_INTENSITY_CD)
        );
        assert_eq!(scalar("ION_PROBE_PIXEL"), f32_of(ION_PROBE_PIXEL_M));
        assert_eq!(scalar("SPRITE_PROBE_SIGMA"), f32_of(SPRITE_PROBE_SIGMA_M));
        assert_eq!(scalar("DUST_PROBE_SUN_X"), f32_of(DUST_PROBE_SUN_M.x));
        assert_eq!(scalar("DUST_PROBE_SUN_Y"), f32_of(DUST_PROBE_SUN_M.y));
        assert_eq!(scalar("DUST_PROBE_OFFSET"), f32_of(DUST_PROBE_OFFSET_M));
        assert_eq!(scalar("DUST_PROBE_NUCLEUS"), f32_of(DUST_PROBE_NUCLEUS_M));
        assert_eq!(scalar("DUST_PROBE_WEIGHT"), f32_of(DUST_PROBE_WEIGHT));
        assert_eq!(DUST_PROBE_SUN_M.z, 0.0);
    }

    /// The probe inputs: 64 cases each, inside each twin's domain, spanning the ranges the pass
    /// uses (the coma from 10⁻³ ℓ to 20 ℓ, sub-pixel and resolved).
    #[test]
    fn comet_probe_inputs_cover_the_cases() {
        let coma = coma_probe_inputs();
        assert_eq!(coma.len(), 64);
        let ratios: Vec<f32> = coma.iter().map(|c| c[0] / c[2]).collect();
        assert!(ratios.iter().any(|&r| r < 2e-3) && ratios.iter().any(|&r| r > 15.0));
        assert!(coma.iter().any(|c| c[3] > c[2]) && coma.iter().any(|c| c[3] < 1e-2 * c[2]));
        assert!(
            coma.iter()
                .all(|c| c.iter().all(|x| x.is_finite() && *x > 0.0))
        );
        let ion = ion_probe_inputs();
        assert_eq!(ion.len(), 64);
        assert!(ion.iter().all(|c| c[0] >= 0.0 && c[0] < c[2] && c[3] > 0.0));
        // Both sides of the ribbon's pixel floor (review C1 I2).
        let floored = |c: &[f32; 4]| ION_MIN_SIGMA_PX * ion_probe_pixel_m(c) > ion_sigma_m();
        assert!(ion.iter().filter(|c| floored(c)).count() >= 16);
        assert!(ion.iter().filter(|c| !floored(c)).count() >= 16);
        let phase = phase_probe_inputs();
        assert_eq!(phase.len(), 64);
        assert!(phase.iter().any(|c| c[0] < -0.9) && phase.iter().any(|c| c[0] > 0.9));
        assert!(phase.iter().any(|c| c[2] <= 0.0) && phase.iter().any(|c| c[3] <= 0.0));
        // The grain-light geometry (review C1 I1): never degenerate, forward and back scattering.
        let mut cos = Vec::new();
        for c in &phase {
            let (offset, nucleus_rel, sun_rel) = dust_probe_geometry(c);
            let (rel, helio) = (nucleus_rel + offset, offset - sun_rel);
            assert!(rel.length() > 1e10 && helio.length() > 1e11, "{c:?}");
            cos.push(helio.normalize().dot(-rel.normalize()));
            assert!(dust_grain_light(offset, DUST_PROBE_WEIGHT, nucleus_rel, sun_rel) > 0.0);
        }
        assert!(
            cos.iter().any(|&c| c > 0.5) && cos.iter().any(|&c| c < -0.5),
            "{cos:?}"
        );
        // The splat cases (C1L): orthonormal planes, finite positive floors, round and very
        // elongated ellipses, and edge-on ones whose minor axis is the floor.
        let splat = splat_probe_inputs();
        assert_eq!(splat.len(), 256);
        let mut ratios = Vec::new();
        for c in splat.chunks(4) {
            let (factor, floor, right, up) = splat_probe_case(c);
            assert!((right.length() - 1.0).abs() < 1e-6 && (up.length() - 1.0).abs() < 1e-6);
            assert!(right.dot(up).abs() < 1e-6 && floor > 0.0);
            let (s1, s2, _) = splat_axes(factor, right, up, floor);
            assert!(s1.is_finite() && s2 > 0.0);
            ratios.push((s1 / s2, s2 / floor));
        }
        assert!(
            ratios.iter().any(|r| r.0 > 100.0) && ratios.iter().any(|r| r.0 < 2.0),
            "{ratios:?}"
        );
        assert!(
            ratios.iter().filter(|r| r.1 < 1.01).count() >= 4,
            "{ratios:?}"
        );
    }
}
