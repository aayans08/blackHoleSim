//! AT-HYG stars in 3-D, pass "stars" (Task B4).
//!
//! Every star of `World::star_data()` is one instance of a screen-aligned quad (four vertices,
//! two triangles from an index buffer), blended additively into `hdr` at the far plane (depth 0,
//! compare `GreaterEqual`, as M0). Per star the vertex stage of `stars3d.wgsl`:
//! 1. skips the instance slots in [`StarInput::suppressed`] (near stars uni-sim draws as surfaces
//!    or points): all four vertices go to one point outside the clip volume, a degenerate quad;
//! 2. places the star: p = (sector centre − camera) + offset + v·t, in f32 (below);
//! 3. takes d = |p|, m = M_V + 5 log₁₀ d − 5 ([`star_apparent_mag`]) and
//!    E = 10^(−0.4 (m + 13.99)) lx (`uni_core::star_illuminance_lux`, as [`star_lux`]);
//! 4. multiplies E by the twinkle factor (only under an atmosphere, [`StarInput::scintillation`]);
//! 5. reads the colour from B−V through a 128-entry Planck table (`uni_core::bv_to_teff`, then
//!    `uni_core::blackbody_linear_srgb`, luminance 1), boosted by [`vivid_color`] while
//!    [`StarInput::vivid_colours`] is on;
//! 6. splats it with M0's energy-normalised PSF (`crate::psf`: erf-integrated Gaussian core,
//!    Moffat halo above 1e-6 lx) at E/Ω(θ) × `star_pre()`, Ω(θ) = Ω_c·cos³θ as M0
//!    (`crate::stars`), culls it when no pixel would rise above `psf::VISIBILITY_CUTOFF`
//!    ([`star_is_visible`]), and sizes its quad with [`footprint_radius_px`]: M0's rule with a
//!    2-px minimum ([`STARS3D_MIN_RADIUS_PX`], review m16).
//!
//! The fragment stage writes radiance × PSF weight × (1 − r)(1 − g) of `lens_mask` (the frame
//! contract's sky-light rule), clamped to `HDR_MAX`.
//!
//! **Precision (design §5.2, Review Focus 5).** [`StarField3::new`] sorts the stars by 16-pc
//! sector ([`crate::sectors`]) and stores each star's f32 offset from its sector's centre. The
//! sector centres go to the GPU relative to an anchor near the camera (f64 difference, then f32),
//! and each frame's uniform carries camera − anchor (f64 difference, then f32). The centres are
//! re-uploaded only when the camera has moved more than [`REANCHOR_PC`] from the anchor: the
//! baked catalogue occupies 166 011 sectors, 2.66 MB, which is too much to upload every frame the
//! camera moves. `sectors::tests::sector_offsets_precision_at_1kpc` bounds the direction error.
//!
//! Proper motion is linear in time with the catalogue's Cartesian velocity (km/s × 1.0227 × 10⁻⁶
//! = pc/yr); a star without a radial velocity moves tangentially only, as the catalogue gives it
//! (A7). Aberration is M3's.
//!
//! The `twin` module (tests) and the free functions here are the CPU twins of `stars3d.wgsl`;
//! `crate::probes` checks them on the GPU under `universe --selftest`. M0's `StarField` stays as
//! the BSC5 fallback.

use std::f32::consts::{LOG2_10, LOG10_2, TAU};
use std::sync::OnceLock;

use anyhow::Context as _;
use bytemuck::Zeroable as _;
use glam::{DVec3, IVec3, UVec3, Vec3};
use uni_core::{blackbody_linear_srgb, bv_to_teff, pcg3d};
use uni_data::{FLAG_NO_VELOCITY, Star3, f16_to_f32, f32_to_f16};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};
use wgpu::util::DeviceExt as _;

use crate::STARS3D_WGSL;
use crate::psf::{
    CORE_SIGMA_PX, HALO_ALPHA_PX, HALO_BETA, MAX_RADIUS_PX, VISIBILITY_CUTOFF, halo_fraction,
    moffat,
};
use crate::sectors::{SectorTable, centres_rel, sector_centre_pc, sector_of};
use crate::stars::pixel_solid_angle;

/// Smallest footprint radius of a 3-D star's quad [px] (review m16). M0's
/// `psf::MIN_RADIUS_PX` (3 px) is unchanged.
pub const STARS3D_MIN_RADIUS_PX: f32 = 2.0;

/// 1 km/s in pc per Julian year: 10³ m × 365.25 d × 86 400 s (IAU 1976 Julian year) ÷ 1 pc
/// (IAU 2012 B2) = 1.0227 × 10⁻⁶ (brief).
pub const PC_PER_YR_PER_KMS: f32 = (1.0e3 * 365.25 * uni_core::DAY / uni_core::PC) as f32;
/// V-band zero point of `uni_core::star_illuminance_lux`: E = 10^(−0.4 (m + 13.99)) lx (Allen
/// 1973, *Astrophysical Quantities*, 3rd ed., p. 197).
pub(crate) const LUX_ZERO_POINT_MAG: f32 = 13.99;
/// Largest display energy a splat carries, so a source at the camera (distance → 0) stays finite
/// on the GPU; far above anything that is not clamped to `HDR_MAX` anyway.
pub(crate) const MAX_SPLAT_ENERGY: f32 = 1.0e30;
/// Entries of the B−V → colour table.
pub(crate) const BV_LUT_LEN: usize = 128;
/// B−V range of the table: the range `uni_core::bv_to_teff` clamps to (Ballesteros 2012).
pub(crate) const BV_LUT_MIN: f32 = -0.4;
pub(crate) const BV_LUT_MAX: f32 = 2.0;
/// B−V packed for a star whose colour is not finite: the Sun's, 0.65 (Cox 2000, *Allen's
/// Astrophysical Quantities*, 4th ed.), 5778 K by `bv_to_teff`.
pub(crate) const UNKNOWN_BV: f32 = 0.65;
/// Absolute magnitude packed for a star with a non-finite position or magnitude: its
/// illuminance underflows to 0 in f32, so the vertex stage culls it and its slot stays valid.
pub(crate) const INVISIBLE_ABS_MAG: f32 = 1.0e30;
/// Saturation factor of the vivid colours about the luminance axis: 1.6, the owner-approved look
/// preview's value (`.superpowers/look-preview/v2/parts/78-frame.js`, `uSat`).
pub(crate) const VIVID_SATURATION: f32 = 1.6;
/// Twinkle log-amplitude σ at full scintillation strength: I/⟨I⟩ = exp(σn − σ²/2) with n of unit
/// variance, so the scintillation index σ_I² = e^(σ²) − 1 = 0.094 at strength 1 (measured over
/// 10⁵ frames: 0.092). A chosen value, not fitted to a cited measurement (approximate): enough to
/// see by eye. B3's `scintillation_strength` carries the dependence on the atmosphere; the
/// dependence on each star's elevation is not modelled (`StarInput` has no zenith).
pub(crate) const TWINKLE_SIGMA_MAX: f32 = 0.3;
/// Frames between independent twinkle values (smoothly interpolated in between): 15 Hz at 60 fps.
pub(crate) const TWINKLE_FRAMES: u32 = 4;
/// Third `pcg3d` lane of the twinkle hash, so it never repeats another hash of (slot, step).
pub(crate) const TWINKLE_SEED: u32 = 0x7F4A_7C15;
/// The star pass re-uploads the sector centres when the camera is further than this [pc] from
/// the anchor they were uploaded against. The f32 camera − anchor offset then errs by at most
/// 3e-8 pc per axis (half an ulp below 1 pc), 6e-7 rad for a star 0.05 pc away.
pub(crate) const REANCHOR_PC: f64 = 1.0;
/// Relative luminance weights of linear Rec. 709: ITU-R BT.709-6 (2015), item 3.2.
const REC709_Y: Vec3 = Vec3::new(0.2126, 0.7152, 0.0722);

/// The index buffer of one splat quad: triangles (0, 1, 2) and (2, 1, 3) over the corners of
/// `quad_corner` in the shaders (0 (−,−), 1 (+,−), 2 (−,+), 3 (+,+)). Four distinct vertices
/// per instance, so a GPU that reuses transformed vertices runs the vertex stage four times per
/// star instead of six (unmeasured on the M3).
pub(crate) const QUAD_INDICES: [u16; 6] = [0, 1, 2, 2, 1, 3];

/// The B−V → colour table the star pass reads: the twin of `struct BvColourLut` in
/// `stars3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BvLut {
    /// rgb: Planck colour (linear Rec. 709, luminance 1) at B−V = BV_LUT_MIN + i·step; w: 0.
    pub entries: [[f32; 4]; BV_LUT_LEN],
}

/// What the star pass needs per frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug)]
pub struct StarInput {
    /// Years since the catalogue epoch (`uni_data::STAR3_EPOCH_JD_TDB`, J2000.0), for proper
    /// motion.
    pub years_since_epoch: f64,
    /// Camera − SSB [pc], ICRS axes, f64.
    pub camera_pc: DVec3,
    /// Indices into the slice given to [`StarField3::new`] that the pass must skip (≤ 8; uni-sim's
    /// `Snapshot::suppressed_stars`). Indices past the slice are ignored, and only the first 8
    /// are used.
    pub suppressed: Vec<u32>,
    /// The "Vivid star colours · non-physical" option (owner feedback, 2026-11-22; on by default):
    /// a saturation boost beyond Planck chromaticity that keeps each star's Rec. 709 luminance
    /// ([`vivid_color`]). The HUD shows `uni_ui::VIVID_STAR_COLOURS_LABEL` while it is on.
    pub vivid_colours: bool,
    /// Scintillation (twinkle) strength in [0, 1]: `uni_atmo::scintillation_strength` for the
    /// camera, 0 in space. B4 drives the twinkle deterministically from `pcg3d` of the star's
    /// instance slot and the frame index, so captures repeat. Clamped to [0, 1]; NaN is 0.
    pub scintillation: f32,
}

impl Default for StarInput {
    /// At the catalogue epoch, at the SSB, nothing suppressed, vivid colours on (the owner's
    /// default), no twinkle.
    fn default() -> Self {
        Self {
            years_since_epoch: 0.0,
            camera_pc: DVec3::ZERO,
            suppressed: Vec::new(),
            vivid_colours: true,
            scintillation: 0.0,
        }
    }
}

/// One star as the GPU sees it: the twin of `struct StarGpu` in `stars3d.wgsl` (32 bytes, two
/// vec4s; review m15: `uni_data::Star3` is a 28-byte disk record and not a valid WGSL storage
/// struct).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct StarGpu {
    /// xyz: offset from the star's sector centre [pc] at the catalogue epoch; w: absolute
    /// magnitude M_V.
    pub offset_absmag: [f32; 4],
    /// x: f16 (vx, vy) [km/s]; y: f16 (vz, B−V) (each pair low half first, as
    /// `uni_data::Star3` packs them); z: the `uni_data` flag bits; w: the sector index.
    pub vel_bv_flags: [u32; 4],
}

/// The star pass's per-frame uniform: the twin of `struct StarsParams` in `stars3d.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StarsParams {
    /// xyz: camera − anchor [pc] (f64 difference, then f32); w: years since the catalogue epoch.
    pub camera_years: [f32; 4],
    /// x: twinkle log-amplitude σ (0 = steady); y: 1 with vivid colours, else 0; zw: 0.
    pub look: [f32; 4],
    /// Instance slots of the suppressed stars, [`NO_STAR`] where unused.
    pub suppressed: [[u32; 4]; 2],
}

/// Marks an unused entry of [`StarsParams::suppressed`].
pub(crate) const NO_STAR: u32 = u32::MAX;

/// The stars in instance order, ready to upload.
pub(crate) struct PackedStars {
    pub gpu: Vec<StarGpu>,
    pub table: SectorTable,
    /// Instance slot of each input star.
    pub slots: Vec<u32>,
}

/// Sorts `stars` by sector and packs them (the CPU half of [`StarField3::new`]). A star with a
/// non-finite position is put in sector (0, 0, 0) and made invisible, so every input keeps a
/// slot. The caller guarantees that `stars.len()` fits a `u32`.
pub(crate) fn pack_stars(stars: &[Star3]) -> PackedStars {
    let positions: Vec<Option<DVec3>> = stars
        .iter()
        .map(|s| {
            let p = DVec3::from_array(s.pos_pc.map(f64::from));
            p.is_finite().then_some(p)
        })
        .collect();
    let keys: Vec<IVec3> = positions
        .iter()
        .map(|p| sector_of(p.unwrap_or(DVec3::ZERO)))
        .collect();
    let table = SectorTable::build(&keys);
    let gpu = table
        .order
        .iter()
        .zip(&table.sector_index)
        .map(|(&i, &sector)| {
            let i = i as usize;
            star_gpu(&stars[i], positions[i], sector, sector_centre_pc(keys[i]))
        })
        .collect();
    let slots = table.slots();
    PackedStars { gpu, table, slots }
}

/// Packs one star of sector `sector` (centre `centre_pc`). The velocity is zeroed for a star
/// flagged "no velocity" (as uni-sim's `star_pos_pc`); a non-finite B−V becomes [`UNKNOWN_BV`];
/// a non-finite position (`pos_pc` = `None`) or magnitude makes the star invisible.
fn star_gpu(s: &Star3, pos_pc: Option<DVec3>, sector: u32, centre_pc: DVec3) -> StarGpu {
    let (offset, abs_mag) = match pos_pc {
        Some(p) if s.abs_mag_v.is_finite() => ((p - centre_pc).as_vec3(), s.abs_mag_v),
        _ => (Vec3::ZERO, INVISIBLE_ABS_MAG),
    };
    let v = s.velocity_kms();
    let v = if s.flags() & FLAG_NO_VELOCITY != 0 || !v.is_finite() {
        Vec3::ZERO
    } else {
        v
    };
    let bv = if s.bv().is_finite() {
        s.bv()
    } else {
        UNKNOWN_BV
    };
    let pack = |lo: f32, hi: f32| u32::from(f32_to_f16(lo)) | u32::from(f32_to_f16(hi)) << 16;
    StarGpu {
        offset_absmag: offset.extend(abs_mag).to_array(),
        vel_bv_flags: [pack(v.x, v.y), pack(v.z, bv), s.flags(), sector],
    }
}

/// The two binary16 halves of `word`, low half first. Twin of `unpack_f16_pair` in
/// `stars3d.wgsl`, which decodes the bits itself (naga gates WGSL's `unpack2x16float` behind a
/// capability `uni_gfx::validate_wgsl` does not enable).
pub(crate) fn unpack_f16_pair(word: u32) -> [f32; 2] {
    [f16_to_f32(word as u16), f16_to_f32((word >> 16) as u16)]
}

/// Position of a star relative to the camera [pc] at `years` after the catalogue epoch:
/// (`sector_rel` + offset) + v·years, in f32 as the GPU does. `sector_rel` is the star's sector
/// centre − camera [pc]. Twin of `star_rel_pos_pc` in `stars3d.wgsl`.
pub(crate) fn star_rel_pos_pc(star: &StarGpu, sector_rel: Vec3, years: f32) -> Vec3 {
    let [vx, vy] = unpack_f16_pair(star.vel_bv_flags[0]);
    let [vz, _] = unpack_f16_pair(star.vel_bv_flags[1]);
    let v = Vec3::new(vx, vy, vz) * PC_PER_YR_PER_KMS;
    (sector_rel + Vec3::from_slice(&star.offset_absmag[..3])) + v * years
}

/// The star pass's uniform for `input`, with the sector centres uploaded relative to
/// `anchor_pc` and `slots` mapping star indices to instance slots.
pub(crate) fn stars_params(input: &StarInput, anchor_pc: DVec3, slots: &[u32]) -> StarsParams {
    let mut suppressed = [NO_STAR; 8];
    let kept = input
        .suppressed
        .iter()
        .filter_map(|&i| slots.get(i as usize).copied());
    for (entry, slot) in suppressed.iter_mut().zip(kept) {
        *entry = slot;
    }
    // NaN and negative strengths are 0, and strengths above 1 are 1.
    let strength = if input.scintillation > 0.0 {
        input.scintillation.min(1.0)
    } else {
        0.0
    };
    let camera = (input.camera_pc - anchor_pc).as_vec3();
    StarsParams {
        camera_years: camera.extend(input.years_since_epoch as f32).to_array(),
        look: [
            TWINKLE_SIGMA_MAX * strength,
            if input.vivid_colours { 1.0 } else { 0.0 },
            0.0,
            0.0,
        ],
        suppressed: [
            [suppressed[0], suppressed[1], suppressed[2], suppressed[3]],
            [suppressed[4], suppressed[5], suppressed[6], suppressed[7]],
        ],
    }
}

/// Bind-group-layout entry of the lens mask (read with `textureLoad`, so unfilterable), which
/// the vertex stage also reads for the attachment size. Shared by the star and point passes.
pub(crate) fn lens_mask_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

/// Bind-group-layout entry of a vertex-stage buffer of `ty` whose bound size is at least
/// `min_size` bytes.
pub(crate) fn vertex_buffer_entry(
    binding: u32,
    ty: wgpu::BufferBindingType,
    min_size: usize,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(min_size as u64),
        },
        count: None,
    }
}

/// The splat pipeline of the star (`vs_stars`/`fs_stars`, depth `GreaterEqual` at the far plane)
/// or point (`vs_points`/`fs_points`, depth `Greater` at the true distance) pass: an indexed
/// triangle list, no face culling (the quad's winding depends on the y flip), additive blending
/// into `hdr`'s colour, depth tested and never written.
pub(crate) fn splat_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    layout: &wgpu::PipelineLayout,
    entry_points: (&str, &str),
    depth_compare: wgpu::CompareFunction,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(entry_points.0),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(entry_points.0),
            compilation_options: Default::default(),
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(entry_points.1),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ADDITIVE),
                // Light adds to colour only; hdr's alpha is left alone.
                write_mask: wgpu::ColorWrites::COLOR,
            })],
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(depth_compare),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// The index buffer holding [`QUAD_INDICES`].
pub(crate) fn quad_index_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("splat quad indices"),
        contents: bytemuck::cast_slice(&QUAD_INDICES),
        usage: wgpu::BufferUsages::INDEX,
    })
}

/// Begins a render pass named `label` that adds into `hdr` and tests (never writes) depth.
pub(crate) fn begin_splat_pass<'e>(
    enc: &'e mut wgpu::CommandEncoder,
    targets: &FrameTargets,
    timer: &mut PassTimer,
    label: &'static str,
) -> wgpu::RenderPass<'e> {
    enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
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
        timestamp_writes: timer.render_writes(label),
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// The 3-D star pass (`RenderFeature` named "stars").
///
/// Group 0 is the renderer's. Group 1: binding 0 the stars (instance order), 1 the sector
/// centres − anchor, 2 the per-frame [`StarsParams`], 3 the B−V colour table, 4 the lens mask
/// (rebuilt in `resize`, when the targets are recreated).
pub struct StarField3 {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    star_buffer: wgpu::Buffer,
    sector_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    lut_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    /// Group 1 for the current targets. `None` until the first `prepare` or `resize`.
    bind_group: Option<wgpu::BindGroup>,
    input: SharedInput<StarInput>,
    /// The occupied sectors, in the order of the sector buffer.
    sectors: Vec<IVec3>,
    /// Instance slot of each star of the slice given to `new`.
    slots: Vec<u32>,
    /// The position [pc] the sector buffer's centres are relative to.
    anchor_pc: DVec3,
    count: u32,
}

impl StarField3 {
    /// Uploads `stars` (sorted by sector) and builds the pass. `stars` is `World::star_data()`:
    /// AT-HYG followed by the archive-only exoplanet hosts, the slice that
    /// `Snapshot::suppressed_stars` and [`StarInput::suppressed`] index (not `WorldData::stars`,
    /// which `World::new` consumes and extends). `camera_bgl` is
    /// `Renderer::camera_bind_group_layout()`. An empty slice is allowed (the pass then draws
    /// nothing).
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        stars: &[Star3],
        input: SharedInput<StarInput>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        let count =
            u32::try_from(stars.len()).context("more stars than one instanced draw can hold")?;
        let mut packed = pack_stars(stars);
        let sectors = std::mem::take(&mut packed.table.sectors);
        let camera = input.get().camera_pc;
        let anchor_pc = if camera.is_finite() {
            camera
        } else {
            DVec3::ZERO
        };
        let mut centres = centres_rel(&sectors, anchor_pc);
        // A storage binding cannot be empty: an empty sky keeps one dark entry and draws none.
        if packed.gpu.is_empty() {
            packed.gpu.push(StarGpu::zeroed());
        }
        if centres.is_empty() {
            centres.push([0.0; 4]);
        }
        let star_bytes: &[u8] = bytemuck::cast_slice(&packed.gpu);
        let sector_bytes: &[u8] = bytemuck::cast_slice(&centres);
        let limit = device.limits().max_storage_buffer_binding_size;
        anyhow::ensure!(
            star_bytes.len() as u64 <= limit && sector_bytes.len() as u64 <= limit,
            "{count} stars need {} B and {} sectors {} B of storage; the device allows {limit} B",
            star_bytes.len(),
            centres.len(),
            sector_bytes.len()
        );
        let params = stars_params(&input.get(), anchor_pc, &packed.slots);

        crate::register_shaders(lib);
        let module = lib.module(device, STARS3D_WGSL)?;

        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let init = |label: &str, contents: &[u8], usage: wgpu::BufferUsages| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage,
            })
        };
        let star_buffer = init("stars3d stars", star_bytes, wgpu::BufferUsages::STORAGE);
        let sector_buffer = init(
            "stars3d sectors",
            sector_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let params_buffer = init(
            "stars3d params",
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let lut_buffer = init(
            "stars3d B-V colours",
            bytemuck::bytes_of(bv_lut()),
            wgpu::BufferUsages::UNIFORM,
        );
        let index_buffer = quad_index_buffer(device);
        let storage = wgpu::BufferBindingType::Storage { read_only: true };
        let uniform = wgpu::BufferBindingType::Uniform;
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("stars3d bgl"),
            entries: &[
                vertex_buffer_entry(0, storage, size_of::<StarGpu>()),
                vertex_buffer_entry(1, storage, size_of::<[f32; 4]>()),
                vertex_buffer_entry(2, uniform, size_of::<StarsParams>()),
                vertex_buffer_entry(3, uniform, size_of::<BvLut>()),
                lens_mask_entry(4),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("stars3d layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        // Stars sit on the far plane (depth 0). Reversed-Z GreaterEqual against the cleared 0.0
        // passes, and anything nearer occludes them.
        let pipeline = splat_pipeline(
            device,
            &module,
            &layout,
            ("vs_stars", "fs_stars"),
            wgpu::CompareFunction::GreaterEqual,
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the 3-D star pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            bgl,
            star_buffer,
            sector_buffer,
            params_buffer,
            lut_buffer,
            index_buffer,
            bind_group: None,
            input,
            sectors,
            slots: packed.slots,
            anchor_pc,
            count,
        })
    }

    /// Number of stars uploaded. Every one is drawn; suppressed, off-screen, behind-camera and
    /// invisible stars are culled per frame in the vertex shader.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// (Re)creates group 1 against `targets` (the lens mask view changes with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        fn buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stars3d bg"),
            layout: &self.bgl,
            entries: &[
                buffer(0, &self.star_buffer),
                buffer(1, &self.sector_buffer),
                buffer(2, &self.params_buffer),
                buffer(3, &self.lut_buffer),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
            ],
        }));
    }
}

impl RenderFeature for StarField3 {
    fn name(&self) -> &'static str {
        "stars"
    }

    /// Reads this frame's [`StarInput`], re-uploads the sector centres when the camera has left
    /// the anchor's 1-pc neighbourhood, and writes the per-frame uniform. A non-finite camera
    /// position keeps the previous frame's uniform.
    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let input = self.input.get();
        if !input.camera_pc.is_finite() {
            return;
        }
        // Both positions are finite here (the anchor starts finite and only takes finite ones).
        if (input.camera_pc - self.anchor_pc).length() > REANCHOR_PC {
            self.anchor_pc = input.camera_pc;
            let centres = centres_rel(&self.sectors, self.anchor_pc);
            if !centres.is_empty() {
                gpu.queue
                    .write_buffer(&self.sector_buffer, 0, bytemuck::cast_slice(&centres));
            }
        }
        let params = stars_params(&input, self.anchor_pc, &self.slots);
        gpu.queue
            .write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&params));
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
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(bind_group) = &self.bind_group else {
            return;
        };
        let mut pass = begin_splat_pass(enc, targets, timer, "stars");
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, bind_group, &[]);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..self.count);
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

/// Apparent V magnitude of a star of absolute magnitude `abs_mag` at `d_pc` parsecs:
/// M + 5 log₁₀ d − 5 (distance modulus), with log₁₀ d = log₁₀ 2 · log₂ d as in WGSL (which has
/// no log10). Twin of `star_apparent_mag` in `stars3d.wgsl`. d = 0 gives −∞.
pub fn star_apparent_mag(abs_mag: f32, d_pc: f32) -> f32 {
    abs_mag + 5.0 * LOG10_2 * d_pc.log2() - 5.0
}

/// Illuminance [lux] of a star of apparent V magnitude `v_mag`, in f32 as the GPU computes it:
/// 10^(−0.4 (m + 13.99)) = 2^(−0.4 log₂10 (m + 13.99)) (`uni_core::star_illuminance_lux`).
/// Twin of `star_lux` in `stars3d.wgsl`.
pub(crate) fn star_lux(v_mag: f32) -> f32 {
    (-0.4 * LOG2_10 * (v_mag + LUX_ZERO_POINT_MAG)).exp2()
}

/// Footprint (quad) radius [px] of a 3-D star or point sprite with display-normalised energy
/// `energy` and halo share `halo_fraction`: M0's `psf::footprint_radius_px` (the distance where
/// both the core and the halo have fallen below `psf::VISIBILITY_CUTOFF`) with the minimum
/// [`STARS3D_MIN_RADIUS_PX`] instead of M0's 3 px. Above 3 px the two agree exactly, so B9's
/// flux radii (Rigel 13.5 px, the Sun from α Cen 21.0 px) are unchanged. Stars whose core peak
/// is below 113 × the cutoff keep the 2-px minimum; the shaded pixels then reach at least 2 px
/// from the centre, which holds ≥ 99.6 % of the σ = 0.65 px core (erf(2/(0.65√2))²), and every
/// pixel beyond is below the cutoff. Twin of `footprint_radius_px` in `stars3d.wgsl`. NaN gives
/// the minimum.
pub fn footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
    let mut r = STARS3D_MIN_RADIUS_PX;
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    if core_peak > VISIBILITY_CUTOFF {
        r = r.max(CORE_SIGMA_PX * (2.0 * (core_peak / VISIBILITY_CUTOFF).ln()).sqrt());
    }
    let halo_peak = halo_fraction * energy * moffat(0.0);
    if halo_peak > VISIBILITY_CUTOFF {
        let q = (halo_peak / VISIBILITY_CUTOFF).powf(1.0 / HALO_BETA);
        r = r.max(HALO_ALPHA_PX * (q - 1.0).sqrt());
    }
    r.min(MAX_RADIUS_PX)
}

/// Whether a splat of display-normalised energy `energy` and halo share `halo_fraction` rises
/// above `psf::VISIBILITY_CUTOFF` (2.55 stops below AgX black) anywhere: its core peak
/// (1 − h)·E/(2πσ²) or its halo peak h·E·Moffat(0) is above the cutoff. The vertex stage culls
/// the rest (the budget ladder's "only stars above `VISIBILITY_CUTOFF` reach the rasteriser").
/// Twin of `star_is_visible` in `stars3d.wgsl`. NaN is invisible.
pub fn star_is_visible(energy: f32, halo_fraction: f32) -> bool {
    let core_peak = (1.0 - halo_fraction) * energy / (TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
    let halo_peak = halo_fraction * energy * moffat(0.0);
    core_peak > VISIBILITY_CUTOFF || halo_peak > VISIBILITY_CUTOFF
}

/// Number of `stars` that rise above `psf::VISIBILITY_CUTOFF` on screen at the eye-adapted star
/// exposure `eye_star_ev100`, field of view `fov_y_rad` and `height_px` lines, seen from the SSB
/// at the catalogue epoch (the lead's star-EV sweep, ruling 9).
///
/// Each star is taken on the optical axis (Ω = Ω_c), where its pixels are dimmest, so this is
/// the count the whole frame would draw if every star were at its centre: the GPU's rule
/// ([`star_is_visible`] of E·`exposure_scale(EV)`/Ω_c) in the GPU's f32 arithmetic.
pub fn visible_star_count(
    stars: &[Star3],
    eye_star_ev100: f32,
    fov_y_rad: f32,
    height_px: u32,
) -> u32 {
    let omega = pixel_solid_angle(fov_y_rad, height_px as f32);
    let pre = uni_gfx::exposure_scale(eye_star_ev100);
    let n = stars
        .iter()
        .filter(|s| {
            let d = Vec3::from(s.pos_pc).length();
            let lux = star_lux(star_apparent_mag(s.abs_mag_v, d));
            star_is_visible(
                (lux * pre / omega).min(MAX_SPLAT_ENERGY),
                halo_fraction(lux),
            )
        })
        .count();
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// The "Vivid star colours · non-physical" colour boost: `c` (linear Rec. 709) with its
/// saturation raised beyond Planck chromaticity, keeping its Rec. 709 luminance
/// (0.2126 R + 0.7152 G + 0.0722 B, ITU-R BT.709-6 item 3.2) so every photometric check stays
/// physical. The app applies it to star-coloured point sprites; the star shader applies its WGSL
/// twin in `stars3d.wgsl` while [`StarInput::vivid_colours`] is on.
///
/// The colour's distance from the grey of its own luminance Y is scaled by
/// [`VIVID_SATURATION`] (1.6, the owner-approved look preview's value). A channel that would go
/// negative is clamped to 0 and the result rescaled to Y again, so very red or blue stars move to
/// the gamut edge with their luminance intact. Black, grey and non-finite colours are unchanged.
pub fn vivid_color(c: Vec3) -> Vec3 {
    let y = c.dot(REC709_Y);
    if !(y > 0.0) {
        return c;
    }
    let grey = Vec3::splat(y);
    // The clamp only removes negative parts, so the boosted luminance is ≥ y > 0.
    let boosted = (grey + VIVID_SATURATION * (c - grey)).max(Vec3::ZERO);
    boosted * (y / boosted.dot(REC709_Y))
}

/// `c` scaled to luminance 1; a black input gives white.
fn unit_luminance(c: Vec3) -> Vec3 {
    let y = c.dot(REC709_Y);
    if y > 0.0 { c / y } else { Vec3::ONE }
}

/// The shared B−V → colour table: [`BV_LUT_LEN`] Planck colours at B−V evenly spaced over
/// [`BV_LUT_MIN`, `BV_LUT_MAX`], each `blackbody_linear_srgb(bv_to_teff(B−V))` (Ballesteros 2012;
/// CIE 1931 CMFs) at luminance 1. Built on first use.
pub(crate) fn bv_lut() -> &'static BvLut {
    static LUT: OnceLock<BvLut> = OnceLock::new();
    LUT.get_or_init(|| BvLut {
        entries: std::array::from_fn(|i| {
            let bv = BV_LUT_MIN + (BV_LUT_MAX - BV_LUT_MIN) * i as f32 / (BV_LUT_LEN - 1) as f32;
            unit_luminance(blackbody_linear_srgb(bv_to_teff(f64::from(bv))))
                .extend(0.0)
                .to_array()
        }),
    })
}

/// Planck colour (linear Rec. 709, luminance 1) of a star of colour index `bv`, interpolated
/// linearly in B−V from the table the star pass reads and clamped to its ends (B−V −0.4 to 2.0).
/// A near star the app draws as a [`crate::PointSprite`] should take this colour (through
/// [`vivid_color`] while vivid colours are on), so it keeps the star pass's colour exactly when
/// it is handed over. Twin of `bv_colour` in `stars3d.wgsl`. NaN gives NaN.
pub fn bv_colour(bv: f32) -> Vec3 {
    let lut = bv_lut();
    let x =
        ((bv - BV_LUT_MIN) / (BV_LUT_MAX - BV_LUT_MIN)).clamp(0.0, 1.0) * (BV_LUT_LEN - 1) as f32;
    let i = (x as usize).min(BV_LUT_LEN - 2);
    let f = x - i as f32;
    let a = Vec3::from_slice(&lut.entries[i][..3]);
    let b = Vec3::from_slice(&lut.entries[i + 1][..3]);
    a + f * (b - a)
}

/// One twinkle draw for slot `slot` at time step `step`: from `pcg3d(slot, step, TWINKLE_SEED)`
/// (Jarzynski & Olano 2020), the sum of three uniforms (Irwin–Hall, mean 1.5, variance 1/4)
/// scaled to mean 0 and variance 1.
fn twinkle_noise(slot: u32, step: u32) -> f32 {
    let h = pcg3d(UVec3::new(slot, step, TWINKLE_SEED));
    let u = (h >> 8_u32).as_vec3() * (1.0 / 16_777_216.0);
    (u.x + u.y + u.z - 1.5) * 2.0
}

/// The twinkle factor I/⟨I⟩ of slot `slot` on frame `frame` for log-amplitude `sigma`:
/// exp(σn − σ²/2), whose mean over frames is 1, so the photometry stays the catalogue's on
/// average. n moves smoothly between independent draws every [`TWINKLE_FRAMES`] frames
/// (smoothstep, renormalised to unit variance), driven only by the slot and the frame index, so
/// captures repeat. `sigma` ≤ 0 or NaN gives exactly 1 (no twinkle in space). Twin of
/// `twinkle_factor` in `stars3d.wgsl`.
pub(crate) fn twinkle_factor(slot: u32, frame: u32, sigma: f32) -> f32 {
    if !(sigma > 0.0) {
        return 1.0;
    }
    let step = frame / TWINKLE_FRAMES;
    let f = (frame % TWINKLE_FRAMES) as f32 / TWINKLE_FRAMES as f32;
    let t = f * f * (3.0 - 2.0 * f);
    let n0 = twinkle_noise(slot, step);
    let n1 = twinkle_noise(slot, step.wrapping_add(1));
    // A blend of two independent unit-variance draws has variance (1 − t)² + t².
    let n = (n0 + t * (n1 - n0)) / ((1.0 - t) * (1.0 - t) + t * t).sqrt();
    (sigma * n - 0.5 * sigma * sigma).exp()
}

/// CPU twins of the vertex stages (test only): what `splat_vertex` in `stars3d.wgsl` computes
/// for one star or point.
#[cfg(test)]
pub(crate) mod twin {
    use super::*;
    use crate::stars::{MIN_CLIP_W, pixel_solid_angle_off_axis};
    use glam::{Mat4, Vec2, Vec4};

    /// What the vertex stages read from group 0.
    #[derive(Clone, Copy, Debug)]
    pub struct ViewTwin {
        pub view_rot: Mat4,
        pub proj: Mat4,
        pub size: Vec2,
        pub fov_y: f32,
        pub star_pre: f32,
        pub frame: u32,
    }

    impl ViewTwin {
        /// A camera whose optical axis (camera −Z) points along `axis` [root axes], with
        /// `fov_y_deg` and `size` px, at star exposure `star_ev100`, on frame 0.
        pub fn looking(axis: Vec3, fov_y_deg: f32, size: Vec2, star_ev100: f32) -> Self {
            let cam_to_root = glam::Quat::from_rotation_arc(Vec3::NEG_Z, axis.normalize());
            let fov_y = fov_y_deg.to_radians();
            Self {
                view_rot: Mat4::from_quat(cam_to_root.inverse()),
                proj: uni_gfx::reversed_z_infinite_proj(fov_y, size.x / size.y, 0.1),
                size,
                fov_y,
                star_pre: uni_gfx::exposure_scale(star_ev100),
                frame: 0,
            }
        }
    }

    /// One splat as the vertex stage leaves it.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Splat {
        pub ndc: Vec2,
        pub centre_px: Vec2,
        pub radiance: Vec3,
        pub energy: f32,
        pub halo_fraction: f32,
        pub radius_px: f32,
        pub depth: f32,
    }

    /// Clip position of a culled vertex: all four coincide outside the clip volume (z > w).
    pub const CULLED: Vec4 = Vec4::new(2.0, 2.0, 2.0, 1.0);

    /// Corner (±1, ±1) of index-buffer vertex `vi`. Twin of `quad_corner`.
    fn quad_corner(vi: u32) -> Vec2 {
        Vec2::new((vi & 1) as f32, (vi >> 1) as f32) * 2.0 - 1.0
    }

    /// The splat of a source in unit direction `dir` (root axes) with illuminance `lux` and
    /// colour `colour`, at NDC depth `depth`; `None` where the vertex stage culls it. Twin of
    /// `splat_vertex`.
    pub fn splat(view: &ViewTwin, dir: Vec3, lux: f32, colour: Vec3, depth: f32) -> Option<Splat> {
        let view_dir = (view.view_rot * dir.extend(0.0)).truncate();
        let clip = view.proj * view_dir.extend(0.0);
        if !(clip.w > MIN_CLIP_W) || !(lux > 0.0) {
            return None;
        }
        let size = view.size;
        let ndc = Vec2::new(clip.x, clip.y) / clip.w;
        let centre = Vec2::new(0.5 * (ndc.x + 1.0) * size.x, 0.5 * (1.0 - ndc.y) * size.y);
        let cos_theta = -view_dir.z / view_dir.length();
        let omega = pixel_solid_angle_off_axis(view.fov_y, size.y, cos_theta);
        let energy = (lux * view.star_pre / omega).min(MAX_SPLAT_ENERGY);
        let h = halo_fraction(lux);
        if !star_is_visible(energy, h) {
            return None;
        }
        let radius = footprint_radius_px(energy, h);
        let reach = Vec2::splat(radius + 1.0);
        if !(centre.cmpgt(-reach).all() && centre.cmplt(size + reach).all()) {
            return None;
        }
        Some(Splat {
            ndc,
            centre_px: centre,
            radiance: colour * energy,
            energy,
            halo_fraction: h,
            radius_px: radius,
            depth,
        })
    }

    /// Clip position of index-buffer vertex `vi` (0..4) of `s`'s quad, or [`CULLED`].
    pub fn vertex(view: &ViewTwin, s: Option<Splat>, vi: u32) -> Vec4 {
        match s {
            None => CULLED,
            Some(s) => (s.ndc + quad_corner(vi) * (s.radius_px + 0.5) * 2.0 / view.size)
                .extend(s.depth)
                .extend(1.0),
        }
    }

    /// The splat of the star in slot `slot`, with `sector_rel` the uploaded sector centres
    /// (relative to the anchor). Twin of `vs_stars`.
    pub fn star_splat(
        view: &ViewTwin,
        packed: &PackedStars,
        sector_rel: &[[f32; 4]],
        params: &StarsParams,
        slot: u32,
    ) -> Option<Splat> {
        if params.suppressed.iter().flatten().any(|&s| s == slot) {
            return None;
        }
        let star = &packed.gpu[slot as usize];
        let sector = Vec3::from_slice(&sector_rel[star.vel_bv_flags[3] as usize][..3])
            - Vec3::from_slice(&params.camera_years[..3]);
        let p = star_rel_pos_pc(star, sector, params.camera_years[3]);
        let d = p.length();
        if !(d > 0.0) {
            return None;
        }
        let lux = star_lux(star_apparent_mag(star.offset_absmag[3], d))
            * twinkle_factor(slot, view.frame, params.look[0]);
        let mut colour = bv_colour(unpack_f16_pair(star.vel_bv_flags[1])[1]);
        if params.look[1] > 0.5 {
            colour = vivid_color(colour);
        }
        splat(view, p / d, lux, colour, 0.0)
    }

    /// The share of a sky source's light a star keeps under `lens_mask` value `mask`:
    /// (1 − r)(1 − g). Twin of `star_sky_factor`.
    pub fn star_sky_factor(mask: Vec4) -> f32 {
        (1.0 - mask.x.clamp(0.0, 1.0)) * (1.0 - mask.y.clamp(0.0, 1.0))
    }
}

/// naga helpers shared by the uni-sky B4 tests.
#[cfg(test)]
pub(crate) mod tests_support {
    use uni_gfx::{ShaderLib, validate_wgsl};
    use wgpu::naga;

    pub(crate) fn module(path: &str) -> naga::Module {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let src = lib.compose(path).unwrap();
        validate_wgsl(&src).unwrap_or_else(|e| panic!("{path}: {e}\n{src}"))
    }

    /// Size and member offsets of the WGSL struct `name` in `m`, and whether every member is a
    /// vec4 or an array of vec4s (the uniform layout rule).
    pub(crate) fn wgsl_struct(m: &naga::Module, name: &str) -> (usize, Vec<(String, usize)>, bool) {
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
        let is_vec4 = |h: naga::Handle<naga::Type>| {
            matches!(
                m.types[h].inner,
                naga::TypeInner::Vector {
                    size: naga::VectorSize::Quad,
                    ..
                }
            )
        };
        let vec4_only = members.iter().all(|mb| match m.types[mb.ty].inner {
            naga::TypeInner::Array { base, .. } => is_vec4(base),
            naga::TypeInner::Struct { .. } => true, // checked on its own
            _ => is_vec4(mb.ty),
        });
        let offsets = members
            .iter()
            .map(|mb| (mb.name.clone().unwrap_or_default(), mb.offset as usize))
            .collect();
        (layouter[handle].size as usize, offsets, vec4_only)
    }

    /// Whether `block`, or a block nested in it, calls `f`.
    pub(crate) fn calls(block: &naga::Block, f: naga::Handle<naga::Function>) -> bool {
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

    pub(crate) fn func(m: &naga::Module, name: &str) -> naga::Handle<naga::Function> {
        m.functions
            .iter()
            .find(|(_, f)| f.name.as_deref() == Some(name))
            .map(|(h, _)| h)
            .unwrap_or_else(|| panic!("no fn {name}"))
    }

    pub(crate) fn entry<'a>(m: &'a naga::Module, name: &str) -> &'a naga::Function {
        &m.entry_points
            .iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("no entry point {name}"))
            .function
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::*;
    use super::twin::*;
    use super::*;
    use crate::psf::{
        CORE_SIGMA_PX, HALO_ALPHA_PX, HALO_BETA, HALO_ENERGY_FRACTION, HALO_MIN_LUX, MAX_RADIUS_PX,
        MIN_RADIUS_PX, VISIBILITY_CUTOFF,
    };
    use crate::stars::{DEFAULT_STAR_EV100, MIN_CLIP_W, pixel_solid_angle};
    use crate::{POINTS_WGSL, STARS3D_WGSL};
    use core::mem::{offset_of, size_of};
    use glam::{Vec2, Vec4};
    use std::f32::consts::TAU;
    use wgpu::naga;

    /// ITU-R BT.709-6 item 3.2 luminance weights.
    fn y709(c: Vec3) -> f32 {
        c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
    }

    fn baked_stars() -> Vec<Star3> {
        let path = uni_data::data_dir().join("baked/athyg_m10.bin");
        uni_data::load_athyg(&path).expect("committed athyg_m10.bin")
    }

    fn pos_f64(s: &Star3) -> DVec3 {
        DVec3::from_array(s.pos_pc.map(f64::from))
    }

    #[test]
    fn apparent_mag_from_ssb_matches_catalogue() {
        let stars = baked_stars();
        let step = stars.len() / 100;
        let mut checked = 0;
        for s in stars.iter().step_by(step).take(100) {
            let d = pos_f64(s).length();
            let want = f64::from(s.abs_mag_v) + 5.0 * d.log10() - 5.0;
            let got = star_apparent_mag(s.abs_mag_v, d as f32);
            assert!(
                (f64::from(got) - want).abs() < 1e-4,
                "M {} d {d}: {got} vs {want}",
                s.abs_mag_v
            );
            checked += 1;
        }
        assert_eq!(checked, 100);
    }

    #[test]
    fn star_lux_matches_uni_core() {
        for v in [-26.74, -12.0, -1.46, 0.0, 6.5, 10.0, 15.0] {
            let want = uni_core::star_illuminance_lux(v);
            let got = f64::from(star_lux(v as f32));
            // f32 rounding of V itself (half an ulp at 26.74 is 1e-6 mag) dominates.
            assert!((got / want - 1.0).abs() < 5e-6, "V {v}: {got} vs {want}");
        }
    }

    /// Barnard's star moves 10.39″/yr on the sky (the brief's value; Gaia measures its proper
    /// motion as 10.39″/yr), so 1039″ ± 2 % in 100 years, through the packed f16 velocity and the
    /// f32 GPU path, seen from the SSB.
    #[test]
    fn barnard_proper_motion_100yr() {
        let stars = baked_stars();
        let names_path = uni_data::data_dir().join("baked/athyg_names.ron");
        let names = uni_data::load_star_names(&names_path).expect("committed athyg_names.ron");
        let index = names
            .entries
            .iter()
            .find(|n| n.proper.as_deref() == Some("Barnard's Star"))
            .expect("Barnard's Star in athyg_names.ron")
            .index;
        let packed = pack_stars(&stars);
        let slot = packed.slots[index as usize] as usize;
        let star = &packed.gpu[slot];
        let sector = packed.table.sector_index[slot] as usize;
        let rel = packed.table.centres_rel(DVec3::ZERO);
        let sector_rel = Vec3::from_slice(&rel[sector][..3]);
        let p0 = star_rel_pos_pc(star, sector_rel, 0.0).as_dvec3();
        let p1 = star_rel_pos_pc(star, sector_rel, 100.0).as_dvec3();
        assert!((p0 - pos_f64(&stars[index as usize])).length() < 1e-5);
        let moved = p0.angle_between(p1).to_degrees() * 3600.0;
        let want = 10.39 * 100.0;
        assert!((moved / want - 1.0).abs() < 0.02, "{moved}″ vs {want}″");
    }

    /// A row of three stars ahead of the camera; the middle star's index is suppressed. Its four
    /// vertices coincide (zero area, outside the clip volume); its neighbours have real quads.
    #[test]
    fn suppressed_star_is_degenerate() {
        let stars: Vec<Star3> = (0..3)
            .map(|i| {
                let x = 0.2 * (i as f32 - 1.0);
                Star3::new([x, 0.0, -10.0], 0.0, [0.0; 3], 0.6, i, 0)
            })
            .collect();
        let packed = pack_stars(&stars);
        let input = StarInput {
            suppressed: vec![1],
            ..StarInput::default()
        };
        let params = stars_params(&input, DVec3::ZERO, &packed.slots);
        let rel = packed.table.centres_rel(DVec3::ZERO);
        let view = ViewTwin::looking(Vec3::NEG_Z, 60.0, Vec2::new(1920.0, 1080.0), -4.0);
        // Twice the area of triangle (a, b, c) in NDC.
        let area = |a: Vec4, b: Vec4, c: Vec4| {
            let (a, b, c) = (
                a.truncate().truncate(),
                b.truncate().truncate(),
                c.truncate().truncate(),
            );
            (b - a).perp_dot(c - a).abs()
        };
        for (index, suppressed) in [(0, false), (1, true), (2, false)] {
            let slot = packed.slots[index];
            let s = star_splat(&view, &packed, &rel, &params, slot);
            let v: Vec<Vec4> = (0..4).map(|vi| vertex(&view, s, vi)).collect();
            // The index buffer's two triangles: (0, 1, 2) and (2, 1, 3).
            let quad = area(v[0], v[1], v[2]) + area(v[2], v[1], v[3]);
            if suppressed {
                assert!(s.is_none());
                assert_eq!(quad, 0.0, "star {index}");
                assert!(v.iter().all(|p| p.z > p.w), "outside the clip volume");
            } else {
                assert!(quad > 0.0, "star {index}: {v:?}");
            }
        }
    }

    #[test]
    fn footprint_minimum_two_px() {
        // The faintest visible splat: its core peak, E/(2πσ²), just above the cutoff.
        let threshold = VISIBILITY_CUTOFF * TAU * CORE_SIGMA_PX * CORE_SIGMA_PX;
        let faintest = threshold * 1.0001;
        assert!(star_is_visible(faintest, 0.0));
        assert!(!star_is_visible(threshold * 0.9999, 0.0));
        assert!(!star_is_visible(f32::NAN, 0.0));
        assert_eq!(footprint_radius_px(faintest, 0.0), STARS3D_MIN_RADIUS_PX);
        // Through the vertex twin: on the axis at the default exposure the faintest drawn star
        // has a 2 px quad radius, and one 0.01 % fainter is culled.
        let view = ViewTwin::looking(Vec3::X, 60.0, Vec2::new(1920.0, 1080.0), DEFAULT_STAR_EV100);
        let lux = threshold * pixel_solid_angle(view.fov_y, view.size.y) / view.star_pre;
        let s = splat(&view, Vec3::X, lux * 1.0001, Vec3::ONE, 0.0).expect("visible");
        assert_eq!(s.radius_px, 2.0);
        assert!(splat(&view, Vec3::X, lux * 0.9999, Vec3::ONE, 0.0).is_none());
        // Sirius (V = −1.46): M0's 18.31 px glare footprint.
        let s = splat(&view, Vec3::X, star_lux(-1.46), Vec3::ONE, 0.0).expect("visible");
        assert!(s.radius_px > 2.0);
        assert!((s.radius_px - 18.31).abs() < 0.01, "{}", s.radius_px);
    }

    /// Above M0's 3-px minimum the 3-D footprint is M0's (B9's flux radii come from it); below,
    /// it falls to 2 px.
    #[test]
    fn footprint_matches_m0_above_three_px() {
        for h in [0.0, HALO_ENERGY_FRACTION] {
            let mut energy = 1e-5;
            while energy < 1e10 {
                let m0 = crate::psf::footprint_radius_px(energy, h);
                let r = footprint_radius_px(energy, h);
                if m0 > MIN_RADIUS_PX {
                    assert_eq!(r, m0, "energy {energy}, halo {h}");
                } else {
                    assert!((STARS3D_MIN_RADIUS_PX..=MIN_RADIUS_PX).contains(&r));
                }
                energy *= 1.5;
            }
        }
        assert_eq!(footprint_radius_px(f32::NAN, 0.0), STARS3D_MIN_RADIUS_PX);
        assert_eq!(
            footprint_radius_px(1e30, HALO_ENERGY_FRACTION),
            MAX_RADIUS_PX
        );
    }

    #[test]
    fn visible_star_count_follows_limiting_magnitude() {
        let stars = baked_stars();
        let (fov, h) = (60f32.to_radians(), 1080);
        let apparent: Vec<f64> = stars
            .iter()
            .map(|s| f64::from(s.abs_mag_v) + 5.0 * pos_f64(s).length().log10() - 5.0)
            .collect();
        let below = |v: f64| apparent.iter().filter(|&&m| m < v).count() as u32;
        // At the default star EV every V ≤ 10 star is drawn (the limit is V ≈ 13.9). The bake
        // also holds 433 nearby stars fainter than that (apparent V up to 21), which are not.
        let n = visible_star_count(&stars, DEFAULT_STAR_EV100, fov, h);
        assert!(n >= below(10.0 + 1e-3), "{n}");
        assert_eq!(stars.len() as u32 - n, 433);
        let mut last = u32::MAX;
        for ev in [-4.0f32, 0.0, 2.0, 4.0, 6.0] {
            let n = visible_star_count(&stars, ev, fov, h);
            assert!(n <= last, "EV {ev}: {n} > {last}");
            last = n;
            // Limiting V: the energy where the core peak meets the cutoff.
            let omega = pixel_solid_angle(fov, h as f32);
            let lux_lim = VISIBILITY_CUTOFF * TAU * CORE_SIGMA_PX * CORE_SIGMA_PX * omega
                / uni_gfx::exposure_scale(ev);
            let v_lim = -2.5 * f64::from(lux_lim).log10() - 13.99;
            assert!(
                below(v_lim - 0.01) <= n && n <= below(v_lim + 0.01),
                "EV {ev}: {n}, V_lim {v_lim}"
            );
        }
        assert!(
            last < stars.len() as u32 / 2,
            "EV 6 hides most stars: {last}"
        );
    }

    #[test]
    fn vivid_color_boosts_saturation_and_keeps_luminance() {
        for t in [
            2000.0, 2500.0, 3500.0, 4500.0, 5772.0, 9940.0, 20_000.0, 40_000.0,
        ] {
            let c = uni_core::blackbody_linear_srgb(t);
            let c = c / y709(c);
            let v = vivid_color(c);
            assert!((y709(v) - 1.0).abs() < 1e-5, "{t} K: {c} → {v}");
            assert!(v.min_element() >= 0.0, "{t} K: {v}");
            // Further from grey than the Planck colour, unless it already is grey.
            let chroma = |x: Vec3| (x - Vec3::splat(y709(x))).length();
            assert!(chroma(v) > chroma(c) * 1.2, "{t} K: {c} → {v}");
        }
        assert!((vivid_color(Vec3::ONE) - Vec3::ONE).abs().max_element() < 1e-6);
        assert!(
            (vivid_color(Vec3::splat(0.3)) - Vec3::splat(0.3))
                .abs()
                .max_element()
                < 1e-6
        );
        assert_eq!(vivid_color(Vec3::ZERO), Vec3::ZERO);
        // Inside the gamut it is a plain saturation scale about the luminance.
        let c = Vec3::new(1.2, 1.0, 0.8);
        let y = Vec3::splat(y709(c));
        let want = y + VIVID_SATURATION * (c - y);
        assert!((vivid_color(c) - want).abs().max_element() < 1e-6);
    }

    #[test]
    fn bv_colour_matches_planck() {
        let unit = |c: Vec3| c / y709(c);
        let mut worst = 0.0f32;
        for i in 0..=480 {
            let bv = -0.4 + 2.4 * i as f32 / 480.0;
            let want = unit(uni_core::blackbody_linear_srgb(uni_core::bv_to_teff(
                f64::from(bv),
            )));
            let got = bv_colour(bv);
            worst = worst.max((got - want).abs().max_element());
            assert!((y709(got) - 1.0).abs() < 1e-3, "B−V {bv}: Y {}", y709(got));
        }
        // Measured 2.6e-5 (linear interpolation over 0.019 in B−V).
        assert!(worst < 1e-4, "worst channel error {worst}");
        // Outside the table it clamps to its ends: blue-white and orange-red.
        assert_eq!(bv_colour(-3.0), bv_colour(BV_LUT_MIN));
        assert_eq!(bv_colour(9.0), bv_colour(BV_LUT_MAX));
        assert!(bv_colour(-0.3).z > bv_colour(-0.3).x);
        assert!(bv_colour(1.6).x > bv_colour(1.6).z);
    }

    #[test]
    fn twinkle_is_deterministic_with_unit_mean() {
        assert_eq!(twinkle_factor(7, 123, 0.0), 1.0);
        assert_eq!(twinkle_factor(7, 123, f32::NAN), 1.0);
        let sigma = TWINKLE_SIGMA_MAX;
        assert_eq!(twinkle_factor(7, 123, sigma), twinkle_factor(7, 123, sigma));
        assert_ne!(twinkle_factor(7, 123, sigma), twinkle_factor(8, 123, sigma));
        let n = 20_000u32;
        let mean = (0..n)
            .map(|f| f64::from(twinkle_factor(42, f, sigma)))
            .sum::<f64>()
            / f64::from(n);
        assert!((mean - 1.0).abs() < 0.02, "{mean}");
        // Smooth in time: from one frame to the next it changes far less than independent
        // draws every frame would.
        let mean_step = |f: &dyn Fn(u32) -> f32| {
            (0..n - 1)
                .map(|i| f64::from((f(i + 1) - f(i)).abs()))
                .sum::<f64>()
                / f64::from(n)
        };
        let smooth = mean_step(&|i| twinkle_factor(42, i, sigma).ln());
        let jumpy = mean_step(&|i| sigma * twinkle_noise(42, i));
        assert!(smooth < 0.5 * jumpy, "{smooth} vs {jumpy}");
        for f in 0..1000 {
            assert!(twinkle_factor(3, f, sigma) > 0.0);
        }
    }

    #[test]
    fn star_gpu_packing_round_trips_and_sanitises() {
        let s = Star3::new(
            [1.0, -20.0, 300.0],
            4.5,
            [10.0, -20.0, 30.0],
            1.2,
            9,
            uni_data::FLAG_NAMED,
        );
        let packed = pack_stars(&[s]);
        let g = packed.gpu[0];
        let key = crate::sectors::sector_of(pos_f64(&s));
        assert_eq!(packed.table.sectors, vec![key]);
        let centre = crate::sectors::sector_centre_pc(key);
        let off = Vec3::from_slice(&g.offset_absmag[..3]).as_dvec3();
        assert!((centre + off - pos_f64(&s)).length() < 1e-5);
        assert_eq!(g.offset_absmag[3], 4.5);
        assert_eq!(g.vel_bv_flags[0], s.vel_bv[0]);
        assert_eq!(g.vel_bv_flags[1], s.vel_bv[1]);
        assert_eq!(g.vel_bv_flags[2], uni_data::FLAG_NAMED);
        assert_eq!(g.vel_bv_flags[3], 0);
        // "No velocity" stars stay put; a NaN colour becomes the Sun's B−V; an unusable
        // position or magnitude makes the star invisible.
        let still = Star3::new(
            [5.0, 0.0, 0.0],
            1.0,
            [100.0; 3],
            f32::NAN,
            1,
            uni_data::FLAG_NO_VELOCITY,
        );
        let bad = Star3::new([f32::NAN, 0.0, 0.0], 1.0, [0.0; 3], 0.5, 2, 0);
        let packed = pack_stars(&[still, bad]);
        let g = packed.gpu[packed.slots[0] as usize];
        assert_eq!(g.vel_bv_flags[0], 0);
        assert_eq!(uni_data::f16_to_f32(g.vel_bv_flags[1] as u16), 0.0);
        let unknown = uni_data::f16_to_f32(uni_data::f32_to_f16(UNKNOWN_BV));
        assert_eq!(
            uni_data::f16_to_f32((g.vel_bv_flags[1] >> 16) as u16),
            unknown
        );
        let g = packed.gpu[packed.slots[1] as usize];
        assert!(Vec4::from(g.offset_absmag).is_finite());
        assert_eq!(star_lux(star_apparent_mag(g.offset_absmag[3], 1.0)), 0.0);
    }

    #[test]
    fn stars_params_maps_suppressed_indices_to_slots() {
        let input = StarInput {
            years_since_epoch: 26.9,
            camera_pc: DVec3::new(1.5, 2.0, -3.0),
            suppressed: vec![2, 0, 99, 1, 3, 4, 5, 6, 7, 8],
            vivid_colours: false,
            scintillation: 0.5,
        };
        let slots = [4, 3, 2, 1, 0, 5, 6, 7, 8];
        let p = stars_params(&input, DVec3::new(1.0, 2.0, -2.0), &slots);
        assert_eq!(p.camera_years, [0.5, 0.0, -1.0, 26.9]);
        assert_eq!(p.look, [0.5 * TWINKLE_SIGMA_MAX, 0.0, 0.0, 0.0]);
        // Index 99 is out of range and skipped; at most 8 slots are kept.
        assert_eq!(p.suppressed, [[2, 4, 3, 1], [0, 5, 6, 7]]);
        let p = stars_params(&StarInput::default(), DVec3::ZERO, &slots);
        assert_eq!(p.suppressed, [[NO_STAR; 4]; 2]);
        assert_eq!(p.look[1], 1.0);
        // Scintillation is clamped to [0, 1].
        let loud = StarInput {
            scintillation: 7.0,
            ..StarInput::default()
        };
        assert_eq!(
            stars_params(&loud, DVec3::ZERO, &slots).look[0],
            TWINKLE_SIGMA_MAX
        );
    }

    #[test]
    fn stars3d_points_shaders_validate() {
        for (path, vs, fs) in [
            (STARS3D_WGSL, "vs_stars", "fs_stars"),
            (POINTS_WGSL, "vs_points", "fs_points"),
        ] {
            let m = module(path);
            let stage = |name: &str| {
                m.entry_points
                    .iter()
                    .find(|e| e.name == name)
                    .map(|e| e.stage)
            };
            assert_eq!(stage(vs), Some(naga::ShaderStage::Vertex), "{path}");
            assert_eq!(stage(fs), Some(naga::ShaderStage::Fragment), "{path}");
        }
    }

    #[test]
    fn stars3d_points_layouts_match_wgsl() {
        let m = module(STARS3D_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "StarGpu");
        assert_eq!(size, 32);
        assert_eq!(size, size_of::<StarGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "offset_absmag".to_owned(),
                    offset_of!(StarGpu, offset_absmag)
                ),
                ("vel_bv_flags".to_owned(), offset_of!(StarGpu, vel_bv_flags)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "StarsParams");
        assert_eq!(size, size_of::<StarsParams>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "camera_years".to_owned(),
                    offset_of!(StarsParams, camera_years)
                ),
                ("look".to_owned(), offset_of!(StarsParams, look)),
                ("suppressed".to_owned(), offset_of!(StarsParams, suppressed)),
            ]
        );
        let (size, offsets, vec4_only) = wgsl_struct(&m, "BvColourLut");
        assert_eq!(size, size_of::<BvLut>());
        assert!(vec4_only);
        assert_eq!(offsets, vec![("entries".to_owned(), 0)]);

        let m = module(POINTS_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "PointGpu");
        assert_eq!(size, size_of::<crate::points::PointGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                (
                    "dir_dist".to_owned(),
                    offset_of!(crate::points::PointGpu, dir_dist)
                ),
                (
                    "color_lux".to_owned(),
                    offset_of!(crate::points::PointGpu, color_lux)
                ),
            ]
        );

        // Group 1 bindings: the star pass's 0–4 and the point pass's own buffer at 5, sharing
        // the lens mask at 4.
        let binding = |m: &naga::Module, name: &str| {
            m.global_variables
                .iter()
                .find(|(_, v)| v.name.as_deref() == Some(name))
                .and_then(|(_, v)| v.binding.clone())
                .unwrap_or_else(|| panic!("no binding {name}"))
        };
        let rb = |binding| naga::ResourceBinding { group: 1, binding };
        for (name, b) in [
            ("stars", 0),
            ("sectors", 1),
            ("stars_params", 2),
            ("bv_lut", 3),
            ("lens_mask", 4),
            ("points", 5),
        ] {
            assert_eq!(binding(&m, name), rb(b), "{name}");
        }
    }

    /// The value of `const name: f32 = <literal>;` (or u32) in `m`.
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

    #[test]
    fn shader_constants_match_cpu() {
        let m = module(STARS3D_WGSL);
        for (name, cpu) in [
            ("CORE_SIGMA_PX", CORE_SIGMA_PX),
            ("HALO_ENERGY_FRACTION", HALO_ENERGY_FRACTION),
            ("HALO_MIN_LUX", HALO_MIN_LUX),
            ("HALO_ALPHA_PX", HALO_ALPHA_PX),
            ("HALO_BETA", HALO_BETA),
            ("STARS3D_MIN_RADIUS_PX", STARS3D_MIN_RADIUS_PX),
            ("MAX_RADIUS_PX", MAX_RADIUS_PX),
            ("VISIBILITY_CUTOFF", VISIBILITY_CUTOFF),
            ("MIN_CLIP_W", MIN_CLIP_W),
            ("PC_PER_YR_PER_KMS", PC_PER_YR_PER_KMS),
            ("LOG10_2", std::f32::consts::LOG10_2),
            ("LOG2_10", std::f32::consts::LOG2_10),
            ("LUX_ZERO_POINT_MAG", LUX_ZERO_POINT_MAG),
            ("MAX_SPLAT_ENERGY", MAX_SPLAT_ENERGY),
            ("BV_LUT_MIN", BV_LUT_MIN),
            ("BV_LUT_MAX", BV_LUT_MAX),
            ("VIVID_SATURATION", VIVID_SATURATION),
        ] {
            assert_eq!(wgsl_const(&m, name), naga::Literal::F32(cpu), "{name}");
        }
        for (name, cpu) in [
            ("BV_LUT_LEN", BV_LUT_LEN as u32),
            ("TWINKLE_FRAMES", TWINKLE_FRAMES),
            ("TWINKLE_SEED", TWINKLE_SEED),
            ("NO_STAR", NO_STAR),
        ] {
            assert_eq!(wgsl_const(&m, name), naga::Literal::U32(cpu), "{name}");
        }
        // 1 km/s in pc/yr: the brief's 1.0227 × 10⁻⁶.
        assert!((PC_PER_YR_PER_KMS / 1.0227e-6 - 1.0).abs() < 1e-4);
    }

    /// The star pass writes sky light at the star exposure, masked by (1 − r)(1 − g).
    #[test]
    fn star_pass_uses_star_exposure_and_sky_mask() {
        let m = module(STARS3D_WGSL);
        assert!(calls(&entry(&m, "vs_stars").body, func(&m, "splat_vertex")));
        assert!(calls(
            &m.functions[func(&m, "splat_vertex")].body,
            func(&m, "star_pre")
        ));
        assert!(!calls(
            &m.functions[func(&m, "splat_vertex")].body,
            func(&m, "scene_pre")
        ));
        assert!(calls(
            &entry(&m, "fs_stars").body,
            func(&m, "star_sky_factor")
        ));
        for (mask, want) in [
            (Vec4::ZERO, 1.0),
            (Vec4::new(1.0, 0.0, 0.0, 0.0), 0.0),
            (Vec4::new(0.0, 1.0, 0.0, 0.0), 0.0),
            (Vec4::new(0.25, 0.5, 1.0, 1.0), 0.375),
        ] {
            assert_eq!(star_sky_factor(mask), want, "{mask}");
        }
    }
}
