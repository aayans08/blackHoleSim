//! Resolved star surfaces (the Sun, near catalogue stars), pass "sun" (Task B2).
//!
//! Limb darkening (Neckel & Labs 1994, Solar Physics 153, 91) normalised so the disk-integrated
//! illuminance equals the view's `luminous_intensity_cd / d²`, animated granulation (Nordlund,
//! Stein & Asplund 2009, Living Rev. Solar Phys. 6, 2) hashed with `pcg3d` on integer-lattice
//! cube-face coordinates, colour `blackbody_linear_srgb(teff_k)`. A camera inside a photosphere
//! draws nothing and [`StarSurfaceRenderer::warnings`] reports "Inside <name>" (review C2). Uses
//! [`crate::cap_mesh`] and [`crate::lod_exposure_mix`]; nothing is sized per `ObjectId` at
//! construction (`World::objects()` grows at runtime): the GPU buffers grow with the number of
//! stars drawn.
//!
//! **Per pixel** (`star_surface.wgsl`): the cap mesh, drawn [`AA_MARGIN_PX`] beyond the
//! silhouette, supplies coverage and vertex depth; the shader re-intersects the exact sphere in
//! the star's unit-sphere space (the constant |o|² − 1 from the CPU in f64), so μ = √(1 − f²) for
//! a ray passing f radii from the centre, and antialiases the limb from the ray's angular
//! distance to the silhouette. Surface luminance per channel c:
//! L_c(μ) = I/(πR²) · colour_c · LD_c(μ)/⟨LD_c⟩ · G_c, written × the exposure (world light, scene
//! exposure; blended from star exposure below 4 px with [`crate::lod_exposure_mix`] while
//! eye-adapted stars are in effect) and clamped to `HDR_MAX`.
//!
//! **Approximations**, labelled on the card as `Granulation: procedural (measured cell size and
//! lifetime)`: the granulation pattern is procedural; the Sun's limb darkening is used for every
//! star; other stars' granules are scaled from the Sun's (see [`granule_cell_m`]); no sunspots,
//! faculae, chromosphere or corona.

use std::f64::consts::FRAC_PI_2;

use anyhow::Context as _;
use glam::{DMat3, DVec3, IVec2, UVec3, Vec2, Vec3};
use uni_core::{R_SUN_M, TEFF_SUN_K, blackbody_linear_srgb, pcg3d};
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};
use uni_sim::snapshot::StarSurfaceView;
use wgpu::util::DeviceExt as _;

use crate::body::BodyScene;
use crate::rings::lens_mask_green_max;
use crate::{CAP_RINGS, CAP_SEGMENTS, cap_indices, cap_mesh, lod_exposure_mix};

/// Vertices of one star's cap mesh, 1 + [`CAP_RINGS`] × [`CAP_SEGMENTS`].
const CAP_VERTICES: usize = 1 + (CAP_RINGS * CAP_SEGMENTS) as usize;

/// The cap mesh is drawn for a sphere this many pixels wider than the star (in angle, at the
/// frame centre), so the pixels just outside the silhouette are rasterised and the limb gets a
/// one-pixel coverage ramp instead of a stair-stepped edge.
pub const AA_MARGIN_PX: f64 = 1.5;

/// The star-surface pass (`RenderFeature` named "sun"). It draws `BodyScene::stars`.
pub struct StarSurfaceRenderer {
    input: SharedInput<BodyScene>,
    pipeline: wgpu::RenderPipeline,
    star_bgl: wgpu::BindGroupLayout,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    vertex_buffer: wgpu::Buffer,
    star_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Stars the two buffers hold.
    capacity: usize,
    /// Stars drawn this frame.
    count: u32,
    warnings: Vec<String>,
}

impl StarSurfaceRenderer {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<BodyScene>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, crate::STAR_SURFACE_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let indices = cap_indices(CAP_RINGS, CAP_SEGMENTS);
        let index_count = u32::try_from(indices.len()).context("cap mesh too large")?;
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("star cap indices"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let star_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(size_of::<StarGpu>() as u64),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sun layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&star_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sun"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: size_of::<[f32; 3]>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    // Premultiplied by the limb's coverage; hdr's alpha is left alone.
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    Some(lens_mask_green_max()),
                ],
            }),
            // No culling: the cap faces the camera by construction.
            primitive: wgpu::PrimitiveState::default(),
            // The star occludes what is behind it (frame contract: depth write, Greater).
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let capacity = 1;
        let (vertex_buffer, star_buffer) = star_buffers(device, capacity);
        let bind_group = star_bind_group(device, &star_bgl, &star_buffer);
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the star-surface pass: {err}");
        }
        Ok(Self {
            input,
            pipeline,
            star_bgl,
            index_buffer,
            index_count,
            vertex_buffer,
            star_buffer,
            bind_group,
            capacity,
            count: 0,
            warnings: Vec::new(),
        })
    }

    /// Warnings for the HUD from the last prepared frame ("Inside Sun", review C2).
    pub fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// The cap vertices and the star records for `capacity` stars.
fn star_buffers(device: &wgpu::Device, capacity: usize) -> (wgpu::Buffer, wgpu::Buffer) {
    let buffer = |label: &str, size: usize, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: (capacity * size) as u64,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    (
        buffer(
            "star cap vertices",
            CAP_VERTICES * size_of::<[f32; 3]>(),
            wgpu::BufferUsages::VERTEX,
        ),
        buffer(
            "star records",
            size_of::<StarGpu>(),
            wgpu::BufferUsages::STORAGE,
        ),
    )
}

fn star_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    stars: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("sun bg"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: stars.as_entire_binding(),
        }],
    })
}

impl RenderFeature for StarSurfaceRenderer {
    fn name(&self) -> &'static str {
        "sun"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, _targets: &FrameTargets) {
        let frame = self
            .input
            .with(|scene| plan_stars(scene, ctx.camera.fov_y_rad, f64::from(ctx.viewport.y)));
        self.warnings = frame.warnings;
        self.count = frame.draws.len() as u32;
        if frame.draws.is_empty() {
            return;
        }
        if frame.draws.len() > self.capacity {
            self.capacity = frame.draws.len().next_power_of_two();
            (self.vertex_buffer, self.star_buffer) = star_buffers(&gpu.device, self.capacity);
            self.bind_group = star_bind_group(&gpu.device, &self.star_bgl, &self.star_buffer);
        }
        let vertices: Vec<[f32; 3]> = frame
            .draws
            .iter()
            .flat_map(|d| d.vertices.iter().copied())
            .collect();
        let records: Vec<StarGpu> = frame.draws.iter().map(|d| d.gpu).collect();
        gpu.queue
            .write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        gpu.queue
            .write_buffer(&self.star_buffer, 0, bytemuck::cast_slice(&records));
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
        if self.count == 0 {
            return;
        }
        let load = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("sun"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.hdr_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.lens_mask_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: load,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: timer.render_writes("sun"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        for k in 0..self.count {
            // Star k's cap starts at vertex k·CAP_VERTICES; its record is instance k.
            pass.draw_indexed(
                0..self.index_count,
                (k as usize * CAP_VERTICES) as i32,
                k..k + 1,
            );
        }
    }
}

// ---- Limb darkening ------------------------------------------------------------------------

/// Neckel & Labs 1994, "Solar limb darkening 1986–1990 (λλ303 to 1099 nm)", Solar Physics 153,
/// 91–114, **Table I**: coefficients A₀…A₅ of P₅(μ) = Σ Aₙ μⁿ (their eq. 5, Σ Aₙ = 1), the mean of
/// the 1986 and 1987 east–west averages, corrected for scattered light. The rows nearest 650, 550
/// and 450 nm, for the r, g and b channels: **640.970 nm, 541.760 nm and 445.125 nm** (the last
/// is marked in the table as one of the six wavelengths observed only since 1988). Each row sums
/// to 1.00000. Twin of `LD_A0`…`LD_A5` in `star_surface.wgsl`.
const LD_COEFFS: [[f32; 6]; 3] = [
    [0.33644, 1.30590, -1.79238, 2.45040, -1.89979, 0.59943],
    [0.26073, 1.27428, -1.30352, 1.47085, -0.96618, 0.26384],
    [0.15248, 1.38517, -1.49615, 1.99886, -1.48155, 0.44119],
];

/// The 541.760 nm row of Table I evaluated by hand at μ = 0.5: A₀ + A₁/2 + A₂/4 + A₃/8 + A₄/16 +
/// A₅/32 = 0.26073 + 0.63714 − 0.32588 + 0.18385625 − 0.06038625 + 0.008245 = 0.703705
/// (`limb_darkening_550_half` checks the twin against it; B9's `sun-closeup` check uses the
/// twin).
pub const LD_550_AT_HALF_PAPER: f32 = 0.703_705;

/// Wavelengths [m] of the three Table I rows, r, g, b.
const LD_WAVELENGTHS_M: [f64; 3] = [640.970e-9, 541.760e-9, 445.125e-9];

/// Limb darkening I(μ)/I(1) at (650, 550, 450) nm (Neckel & Labs 1994 fifth-order
/// polynomials, Table I rows 640.970, 541.760 and 445.125 nm). μ is clamped to [0, 1]. Horner's
/// rule in f32, in the order of the WGSL twin in `star_surface.wgsl`.
pub fn limb_darkening(mu: f32) -> Vec3 {
    let mu = mu.clamp(0.0, 1.0);
    let a = |n: usize| Vec3::new(LD_COEFFS[0][n], LD_COEFFS[1][n], LD_COEFFS[2][n]);
    a(0) + mu * (a(1) + mu * (a(2) + mu * (a(3) + mu * (a(4) + mu * a(5)))))
}

/// 2∫₀¹ I(μ)/I(1) μ dμ per channel, the disk-averaged limb darkening used to normalise the
/// surface luminance to the star's flux: 2 Σ Aₙ/(n + 2), Neckel & Labs's F/I (their eq. 6; Table
/// I lists 0.8290, 0.8002 and 0.7596 for the three rows).
pub fn limb_darkening_mean() -> Vec3 {
    ld_mean_f64().as_vec3()
}

fn ld_mean_f64() -> DVec3 {
    let mean = |row: &[f32; 6]| {
        2.0 * row
            .iter()
            .enumerate()
            .map(|(n, a)| f64::from(*a) / (n as f64 + 2.0))
            .sum::<f64>()
    };
    DVec3::new(
        mean(&LD_COEFFS[0]),
        mean(&LD_COEFFS[1]),
        mean(&LD_COEFFS[2]),
    )
}

// ---- Granulation ---------------------------------------------------------------------------

/// The Sun's granule cell size [m]: the lattice spacing of the pattern, one granule per cell.
/// Granules are "of order 1 Mm" across (Nordlund, Stein & Asplund 2009, "Convective driving")
/// and long-lived ones approach 1.5 Mm equivalent diameter (Hirzberger et al. 1999, ApJ 515,
/// 441); the M1 plan's 1.3 Mm is used (1.299 Mm after rounding to 841 cells per face edge).
/// Measured mean spacing of the pattern: 1.33 Mm (`granulation_statistics`).
pub const GRANULE_CELL_SUN_M: f64 = 1.3e6;

/// Period [s] of each layer's pattern: a layer's weight rises from 0 to 1 and back over one
/// period, so a granule is visible at more than half its peak weight for half a period, 512 s ≈
/// 8.5 min, the plan's lifetime of ≈ 8 min (tracked granules live 4.5–9.2 min on average by
/// family, about 6 min overall: Hirzberger et al. 1999). 2^10 s divides the 2^22 s wrap of
/// `sim_time_s()`, so the animation is periodic across it. Measured temporal autocorrelation
/// (1/e) time: 8.07 min (`granulation_statistics`).
const GRAN_PERIOD_S: f32 = 1024.0;
/// Patterns per 2^22 s (the epoch is taken modulo this).
const GRAN_EPOCHS: i32 = 4096;
/// Largest lattice size per face edge (keeps the packed cell id inside a u32).
const GRAN_MAX_CELLS: f32 = 8192.0;
/// Each granule's centre is jittered over the middle 80 % of its cell (a fairly regular
/// pattern, as granulation is).
const GRAN_JITTER_MIN: f32 = 0.1;
const GRAN_JITTER_SPAN: f32 = 0.8;
/// Dark intergranular lanes: the intensity ramps up as F2 − F1 (lattice units) goes from 0 to
/// this, so a lane is about 0.3 cells ≈ 400 km wide (intergranular lanes are a few hundred km).
const GRAN_LANE: f32 = 0.3;
/// Per-granule brightness, uniform in [0.55, 1.0) of the brightest.
const GRAN_BRIGHT_MIN: f32 = 0.55;
const GRAN_BRIGHT_SPAN: f32 = 0.45;
/// Mean and standard deviation of one layer's intensity over the sphere (measured with 10^6
/// points; `granulation_layer_normalisation` re-measures them), divided out so each layer has
/// mean 0 and variance 1.
const GRAN_MEAN: f32 = 0.5109;
const GRAN_SIGMA: f32 = 0.3112;

/// RMS intensity contrast of the granulation at 550 nm, the `g` channel. The plan gives
/// "≈ 13 %" and asks for it to be checked (ruling 17). The sources: Hinode/BFI green continuum
/// (≈ 555 nm) at disk centre, 8.3 % as observed and **19.4 ± 1.4 %** after removing the
/// telescope's point spread function (Wedemeyer-Böhm & Rouppe van der Voort 2009, A&A 503,
/// 225), which 3-D radiation-MHD simulations reproduce; MURaM at 630 nm, **14.4 %** at full
/// resolution (Danilovic et al. 2008, A&A 484, L17, as quoted in arXiv:1709.06933), i.e. 16.3 %
/// at 550 nm after Planck's sensitivity scaling below. Nordlund, Stein & Asplund 2009
/// ("Observational constraints", granulation) state that simulated RMS intensities agree with
/// each other to 1–2 % of the continuum and are consistent with Hinode. 13 % matches neither;
/// B2 uses **0.18**, between the two estimates of the true contrast at 550 nm (decision in the
/// B2 report). Measured on the twin: 0.1825 over 4096 points (`granulation_statistics`).
pub const GRANULATION_RMS_550: f32 = 0.18;

/// The contrast fades out as a cell shrinks from 2 px to 1.25 px (the footprint filter: no
/// pattern below the Nyquist size, so no aliasing). Twin of `GRAN_FADE_*` in the WGSL.
const GRAN_FADE_LO_PX: f32 = 1.25;
const GRAN_FADE_HI_PX: f32 = 2.0;

/// 4/π and π/4 as f32, the equal-angle cube map's scale (Ronchi et al. 1996, "The 'cubed
/// sphere'", J. Comput. Phys. 124, 93: gnomonic coordinate tan(α), α uniform in [−π/4, π/4]).
const FRAC_4_PI: f32 = 1.273_239_5;
const FRAC_PI_4: f32 = std::f32::consts::FRAC_PI_4;
/// No granule yet (the nearest-two search's initial id).
const GRAN_NONE: u32 = u32::MAX;

/// Second radiation constant c₂ = hc/k [m·K] (CODATA 2018).
const C2_M_K: f64 = 1.438_776_877e-2;

/// Granule cell size [m] for a star of `teff_k` and `radius_m`: the Sun's 1.3 Mm ×
/// (T_eff/5772 K)·(R/R☉). **Approximation** (B2 brief): granules scale with the photosphere's
/// pressure scale height H_p ∝ T/g ∝ T·R²/M; taking M ∝ R, roughly true on the main sequence,
/// gives T·R. It ignores the mass, so giants' granules (far fewer and larger in reality) are
/// only indicative. A non-positive or non-finite input falls back to the Sun's value.
pub fn granule_cell_m(teff_k: f64, radius_m: f64) -> f64 {
    let t = if teff_k > 0.0 && teff_k.is_finite() {
        teff_k
    } else {
        TEFF_SUN_K
    };
    let r = if radius_m > 0.0 && radius_m.is_finite() {
        radius_m
    } else {
        R_SUN_M
    };
    GRANULE_CELL_SUN_M * (t / TEFF_SUN_K) * (r / R_SUN_M)
}

/// Fraction of the granulation contrast kept when one cell spans `cell_px` pixels: 0 at and
/// below 1.25 px, 1 at and above 2 px, smoothstep between. Twin of `granulation_fade`.
pub fn granulation_fade(cell_px: f32) -> f32 {
    smoothstep(GRAN_FADE_LO_PX, GRAN_FADE_HI_PX, cell_px)
}

/// Granulation contrast of the r and b channels relative to g for a star of `teff_k`: the
/// relative intensity change per relative temperature change, δI/I = x/(1 − e^{−x})·δT/T with
/// x = c₂/(λT) (Planck's law differentiated in T), at the three Table I wavelengths. For the Sun
/// 0.854 (r) and 1.210 (b): granules are bluer than the lanes. (Hinode's deconvolved contrasts,
/// 16.6 / 19.4 / 26.7 % at 668 / 555 / 450 nm, give 0.86 and 1.38.)
fn channel_contrast(teff_k: f64) -> (f32, f32) {
    let t = if teff_k > 0.0 && teff_k.is_finite() {
        teff_k
    } else {
        TEFF_SUN_K
    };
    let s = |lambda: f64| {
        let x = C2_M_K / (lambda * t);
        x / (1.0 - (-x).exp())
    };
    let g = s(LD_WAVELENGTHS_M[1]);
    (
        (s(LD_WAVELENGTHS_M[0]) / g) as f32,
        (s(LD_WAVELENGTHS_M[2]) / g) as f32,
    )
}

/// WGSL's `smoothstep`: t²(3 − 2t), t = clamp((x − e0)/(e1 − e0), 0, 1).
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cube face of the unit vector `p` and its gnomonic coordinates in [−1, 1]: faces 0/1 = ±x
/// with (y, z), 2/3 = ±y with (z, x), 4/5 = ±z with (x, y). Twin of `gran_cube`.
fn cube_coord(p: Vec3) -> (u32, Vec2) {
    let a = p.abs();
    if a.x >= a.y && a.x >= a.z {
        return (if p.x >= 0.0 { 0 } else { 1 }, Vec2::new(p.y, p.z) / a.x);
    }
    if a.y >= a.z {
        return (if p.y >= 0.0 { 2 } else { 3 }, Vec2::new(p.z, p.x) / a.y);
    }
    (if p.z >= 0.0 { 4 } else { 5 }, Vec2::new(p.x, p.y) / a.z)
}

/// Unit vector of face `face` at gnomonic coordinates `st`, the inverse of [`cube_coord`]. Twin
/// of `gran_dir`.
fn gran_dir(face: u32, st: Vec2) -> Vec3 {
    let sg = if face % 2 == 0 { 1.0 } else { -1.0 };
    let v = match face / 2 {
        0 => Vec3::new(sg, st.x, st.y),
        1 => Vec3::new(st.y, sg, st.x),
        _ => Vec3::new(st.x, st.y, sg),
    };
    v.normalize()
}

/// The 24 high bits of each lane as a float in [0, 1), exact in f32. Twin of `gran_unit`.
fn gran_unit(x: UVec3) -> Vec3 {
    (x >> 8_u32).as_vec3() * (1.0 / 16_777_216.0)
}

/// The lattice cell (face, i, j) that cell (`ci`, `cj`) of `face` names: itself inside the face,
/// else the cell of the neighbouring face that holds the would-be cell's centre, so the pattern
/// continues across cube edges. Twin of `gran_wrap`.
fn gran_wrap(face: u32, ci: i32, cj: i32, n: i32) -> UVec3 {
    if (0..n).contains(&ci) && (0..n).contains(&cj) {
        return UVec3::new(face, ci as u32, cj as u32);
    }
    let nf = n as f32;
    let a = (Vec2::new(ci as f32, cj as f32) + 0.5) / nf * 2.0 - 1.0;
    let (face2, st) = cube_coord(gran_dir(face, (a * FRAC_PI_4).map(f32::tan)));
    let q = (st.map(f32::atan) * FRAC_4_PI + 1.0) * 0.5 * nf;
    let ij = q.floor().as_ivec2().clamp(IVec2::ZERO, IVec2::splat(n - 1));
    UVec3::new(face2, ij.x as u32, ij.y as u32)
}

/// `pcg3d` of lattice cell `c` = (face, i, j) for pattern `salt` (epoch × 2 + layer). Twin of
/// `gran_hash`.
fn gran_hash(c: UVec3, salt: u32) -> UVec3 {
    pcg3d(UVec3::new(
        c.y,
        c.z,
        c.x.wrapping_add(6_u32.wrapping_mul(salt)),
    ))
}

/// The granule of cell `c` for `salt`: its unit vector and brightness. Twin of `gran_feature`.
fn gran_feature(c: UVec3, salt: u32, nf: f32) -> (Vec3, f32) {
    let h = gran_unit(gran_hash(c, salt));
    let pos = Vec2::new(c.y as f32, c.z as f32) + GRAN_JITTER_MIN + GRAN_JITTER_SPAN * h.truncate();
    let dir = gran_dir(c.x, ((pos / nf * 2.0 - 1.0) * FRAC_PI_4).map(f32::tan));
    (dir, h.z)
}

/// The two nearest granules of one layer: distances F1 ≤ F2 (lattice units), the nearest's id
/// and brightness.
#[derive(Clone, Copy, Debug)]
struct Nearest {
    f1: f32,
    f2: f32,
    id1: u32,
    b1: f32,
}

impl Nearest {
    const NONE: Nearest = Nearest {
        f1: f32::MAX,
        f2: f32::MAX,
        id1: GRAN_NONE,
        b1: 0.0,
    };

    /// Twin of `gran_consider`: a granule already nearest (a cell reached twice across a cube
    /// corner) is skipped.
    fn consider(self, id: u32, d: f32, b: f32) -> Nearest {
        if id == self.id1 {
            return self;
        }
        if d < self.f1 {
            return Nearest {
                f1: d,
                f2: self.f1,
                id1: id,
                b1: b,
            };
        }
        if d < self.f2 {
            return Nearest { f2: d, ..self };
        }
        self
    }

    /// One layer's intensity: dark lanes where the two nearest granules are equidistant,
    /// granule brightness varying per granule. Twin of `gran_intensity`.
    fn intensity(self) -> f32 {
        smoothstep(0.0, GRAN_LANE, self.f2 - self.f1)
            * (GRAN_BRIGHT_MIN + GRAN_BRIGHT_SPAN * self.b1)
    }
}

/// Lattice cells per cube-face edge: round(πR/(2·cell)), at least 1, at most 8192. Twin of
/// `gran_cells` (rounded as floor(x + 0.5), as WGSL's `round` rounds halves to even).
fn gran_cells(cell_m: f32, radius_m: f32) -> i32 {
    let n = (std::f32::consts::PI * radius_m / (2.0 * cell_m) + 0.5).floor();
    (if n >= 1.0 { n } else { 1.0 }).clamp(1.0, GRAN_MAX_CELLS) as i32
}

/// Pattern salt of `layer` at `phase` (time / period, plus ½ for layer 1): the epoch modulo
/// [`GRAN_EPOCHS`], × 2, + layer. Twin of `gran_salt`.
fn gran_salt(phase: f32, layer: u32) -> u32 {
    let e = phase.floor() as i32;
    (e.rem_euclid(GRAN_EPOCHS) as u32) * 2 + layer
}

/// Cross-fade weight of a layer at `phase`: 0 when its pattern is regenerated, 1 half a period
/// later. Twin of `gran_weight`.
fn gran_weight(phase: f32) -> f32 {
    1.0 - (2.0 * (phase - phase.floor()) - 1.0).abs()
}

/// The two layers' nearest granules at unit vector `p` on a lattice of `n` cells per face edge,
/// one loop over the 3 × 3 cells around p's (the twin of `granulation`'s loop), and each layer's
/// nearest granule centre (for the tests).
fn gran_near2(p: Vec3, n: i32, salts: [u32; 2]) -> [(Nearest, Vec3); 2] {
    let nf = n as f32;
    let (face, st) = cube_coord(p);
    let q = (st.map(f32::atan) * FRAC_4_PI + 1.0) * 0.5 * nf;
    let base = q.floor().as_ivec2().clamp(IVec2::ZERO, IVec2::splat(n - 1));
    let to_cells = 2.0 * nf / std::f32::consts::PI;
    let mut near = [(Nearest::NONE, Vec3::ZERO); 2];
    for dj in -1..=1 {
        for di in -1..=1 {
            let c = gran_wrap(face, base.x + di, base.y + dj, n);
            let id = (c.x * n as u32 + c.y) * n as u32 + c.z;
            for (layer, salt) in salts.iter().enumerate() {
                let (dir, b) = gran_feature(c, *salt, nf);
                let before = near[layer].0.id1;
                near[layer].0 = near[layer].0.consider(id, (p - dir).length() * to_cells, b);
                if near[layer].0.id1 != before {
                    near[layer].1 = dir;
                }
            }
        }
    }
    near
}

/// One layer's nearest granules at `p` (normalised) for `salt`, and the nearest's centre (tests).
#[cfg(test)]
fn gran_nearest(p: Vec3, n: i32, salt: u32) -> (Nearest, Vec3) {
    gran_near2(p.normalize(), n, [salt, salt])[0]
}

/// One layer's intensity at `p` for `salt` (the quantity [`GRAN_MEAN`] and [`GRAN_SIGMA`]
/// describe; tests).
#[cfg(test)]
fn gran_layer(p: Vec3, n: i32, salt: u32) -> f32 {
    gran_nearest(p, n, salt).0.intensity()
}

/// Multiplicative granulation intensity (mean 1) at the unit-sphere point `p_unit` of a star of
/// `radius_m` with cells of `cell_m`, at time `t_s` [s]. Twin of the WGSL in
/// `star_surface.wgsl`.
///
/// Cellular (Worley 1996, "A cellular texture basis function", SIGGRAPH) F2 − F1 noise: one
/// granule per cell of an equal-angle cube-sphere lattice of round(πR/(2·cell_m)) cells per face
/// edge, its centre and brightness from `pcg3d` of the integer cell (face, i, j) and the pattern's
/// epoch; dark lanes where two granules meet. Two layers regenerate their pattern every
/// [`GRAN_PERIOD_S`], half a period apart, and are cross-faded with triangle weights (Neyret
/// 2003, "Advected textures", SCA, the regeneration cross-fade), renormalised by
/// √(w₀² + w₁²) so the contrast does not dip mid-fade. Each layer is standardised by
/// [`GRAN_MEAN`] and [`GRAN_SIGMA`], so the result is 1 + [`GRANULATION_RMS_550`]·(unit-variance
/// noise): mean 1, RMS contrast 0.18. `t_s` is `sim_time_s()` (TDB seconds modulo 2^22); the
/// pattern is periodic in 2^22 s. A zero `p_unit` gives 1.
pub fn granulation(p_unit: Vec3, cell_m: f32, radius_m: f32, t_s: f32) -> f32 {
    if !(p_unit.dot(p_unit) > 0.0) {
        return 1.0;
    }
    let p = p_unit.normalize();
    let n = gran_cells(cell_m, radius_m);
    let phase0 = t_s / GRAN_PERIOD_S;
    let phase1 = phase0 + 0.5;
    let salts = [gran_salt(phase0, 0), gran_salt(phase1, 1)];
    let [(near0, _), (near1, _)] = gran_near2(p, n, salts);
    let v0 = (near0.intensity() - GRAN_MEAN) / GRAN_SIGMA;
    let v1 = (near1.intensity() - GRAN_MEAN) / GRAN_SIGMA;
    let w0 = gran_weight(phase0);
    let w1 = gran_weight(phase1);
    1.0 + GRANULATION_RMS_550 * (w0 * v0 + w1 * v1) / (w0 * w0 + w1 * w1).sqrt()
}

/// The `pcg3d` hash of `p_unit`'s own lattice cell for layer 0 at `t_s`: what the granulation
/// probe compares bit for bit. Twin of `granulation_cell_hash`.
fn granulation_cell_hash(p_unit: Vec3, cell_m: f32, radius_m: f32, t_s: f32) -> UVec3 {
    let p = p_unit.normalize();
    let n = gran_cells(cell_m, radius_m);
    let (face, st) = cube_coord(p);
    let q = (st.map(f32::atan) * FRAC_4_PI + 1.0) * 0.5 * n as f32;
    let base = q.floor().as_ivec2().clamp(IVec2::ZERO, IVec2::splat(n - 1));
    gran_hash(
        UVec3::new(face, base.x as u32, base.y as u32),
        gran_salt(t_s / GRAN_PERIOD_S, 0),
    )
}

// ---- Per-frame plan ------------------------------------------------------------------------

/// One star on the GPU: twin of `struct StarGpu` in `star_surface.wgsl` (six vec4s, 96 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct StarGpu {
    /// The camera in the star's unit-sphere space, root axes, (camera − centre)/R; w: |xyz|² − 1
    /// from f64 as (d − R)(d + R)/R², exact even metres above the photosphere.
    pub cam_unit: [f32; 4],
    /// Rows of the root → star-fixed rotation (granules ride with the rotating surface).
    pub to_body_x: [f32; 4],
    pub to_body_y: [f32; 4],
    pub to_body_z: [f32; 4],
    /// rgb: I/(πR²)·colour/⟨LD⟩ [cd/m²], the disk-centre luminance per channel; w: the exposure
    /// blend weight (1 = scene exposure).
    pub l0: [f32; 4],
    /// x: radius [m]; y: granule cell size [m]; z, w: the r and b granulation contrast relative
    /// to g ([`channel_contrast`]).
    pub params: [f32; 4],
}

/// One star to draw: its record and its cap mesh, camera-relative [m].
pub(crate) struct StarDraw {
    pub gpu: StarGpu,
    pub vertices: Vec<[f32; 3]>,
}

/// What the pass draws this frame, and its HUD warnings.
pub(crate) struct StarFrame {
    pub draws: Vec<StarDraw>,
    pub warnings: Vec<String>,
}

/// The GPU record of `view`, with exposure blend weight `exposure_mix`.
///
/// Photometry: the surface luminance at disk centre per channel is
/// L₀_c = I/(πR²) · colour_c / ⟨LD_c⟩, with I the view's luminous intensity [cd], colour =
/// `blackbody_linear_srgb(teff_k)` (luminance 1) and ⟨LD⟩ = [`limb_darkening_mean`]. Seen from
/// distance d the disk then delivers ∫ L dΩ = (2πR²/d²)·L₀·∫₀¹ LD(μ) μ dμ = I/d² · colour per
/// channel, luminance I/d²: the catalogue flux, so the resolved disk and the point it replaces
/// agree (Review Focus 4; `star_surface_flux_matches_catalogue_magnitude`).
pub(crate) fn star_gpu(view: &StarSurfaceView, exposure_mix: f32) -> StarGpu {
    let r = view.radius_m;
    let rel = view.rel_pos_m;
    let d = rel.length();
    let cam = -rel / r;
    let c = (d - r) * (d + r) / (r * r);
    let to_body = DMat3::from_quat(view.orient.conjugate());
    let row = |k: usize| to_body.row(k).extend(0.0).as_vec4().to_array();
    let teff = if view.teff_k > 0.0 && view.teff_k.is_finite() {
        view.teff_k
    } else {
        TEFF_SUN_K
    };
    let colour = blackbody_linear_srgb(teff).as_dvec3();
    let l0 = view.luminous_intensity_cd / (std::f64::consts::PI * r * r) * colour / ld_mean_f64();
    let (k_r, k_b) = channel_contrast(teff);
    StarGpu {
        cam_unit: cam.extend(c).as_vec4().to_array(),
        to_body_x: row(0),
        to_body_y: row(1),
        to_body_z: row(2),
        l0: [l0.x as f32, l0.y as f32, l0.z as f32, exposure_mix],
        params: [r as f32, granule_cell_m(teff, r) as f32, k_r, k_b],
    }
}

/// The stars of `scene` to draw at vertical field of view `fov_y_rad` over `height_px` internal
/// lines.
///
/// - A star whose centre is no farther than its radius (the camera on or inside the photosphere,
///   or at its centre) draws nothing and adds the warning `Inside <name>` (review C2), the name
///   from `scene.names` ("star" when it has none). A star with a non-finite position or radius
///   is skipped.
/// - Each other star gets a [`cap_mesh`] of [`CAP_RINGS`] × [`CAP_SEGMENTS`] for a sphere whose
///   silhouette is [`AA_MARGIN_PX`] wider (at most half-way to 90°), camera-relative in f64 and
///   then f32, and its [`star_gpu`] record with the exposure blend
///   `lod_exposure_mix(proj_radius_px)` while eye-adapted stars are in effect (1 otherwise).
pub(crate) fn plan_stars(scene: &BodyScene, fov_y_rad: f64, height_px: f64) -> StarFrame {
    let px_angle = 2.0 * (0.5 * fov_y_rad).tan() / height_px.max(1.0);
    let mut frame = StarFrame {
        draws: Vec::new(),
        warnings: Vec::new(),
    };
    for view in &scene.stars {
        let (rel, r) = (view.rel_pos_m, view.radius_m);
        if !(rel.is_finite() && r > 0.0 && r.is_finite()) {
            continue;
        }
        let d = rel.length();
        if !(d > r) {
            let name = scene
                .names
                .iter()
                .find(|(id, _)| *id == view.id)
                .map_or("star", |(_, name)| name.as_str());
            frame.warnings.push(format!("Inside {name}"));
            continue;
        }
        let theta_s = (r / d).asin();
        let margin = (AA_MARGIN_PX * px_angle).min(0.5 * (FRAC_PI_2 - theta_s));
        let mut r_mesh = d * (theta_s + margin).sin();
        let mut unit = cap_mesh(-rel / r_mesh, CAP_RINGS, CAP_SEGMENTS);
        if unit.is_empty() {
            r_mesh = r;
            unit = cap_mesh(-rel / r, CAP_RINGS, CAP_SEGMENTS);
        }
        if unit.len() != CAP_VERTICES {
            continue;
        }
        let mix = if scene.eye_adapted {
            lod_exposure_mix(view.proj_radius_px)
        } else {
            1.0
        };
        frame.draws.push(StarDraw {
            gpu: star_gpu(view, mix),
            vertices: unit
                .iter()
                .map(|v| (rel + *v * r_mesh).as_vec3().to_array())
                .collect(),
        });
    }
    frame
}

// ---- GPU probes ----------------------------------------------------------------------------

/// `ShaderLib` path of the star-surface probes.
const STAR_PROBE_WGSL_PATH: &str = "probe/planet_star.wgsl";

/// Compute entry points for the star-surface probes. Input and output at group 2
/// (star_surface.wgsl owns groups 0 and 1); the output is u32 so float results travel as bits.
const STAR_PROBE_WGSL: &str = r#"// probe/planet_star.wgsl: `universe --selftest` probes of star_surface.wgsl (Task B2).
#import "planet/star_surface.wgsl"

@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<u32>>;

// in: (μ, 0, 0, 0); out.xyz: limb_darkening(μ) as bits.
@compute @workgroup_size(64, 1, 1)
fn cs_limb(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    probe_out[gid.x] = vec4<u32>(bitcast<vec3<u32>>(limb_darkening(probe_in[gid.x].x)), 0u);
}

// in[2i]: (p, t_s), in[2i + 1]: (cell_m, radius_m, 0, 0); out[i].x: granulation as bits,
// out[i].yzw: the lattice hash of p's own cell (layer 0).
@compute @workgroup_size(64, 1, 1)
fn cs_gran(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (2u * gid.x + 1u >= arrayLength(&probe_in)) {
        return;
    }
    let a = probe_in[2u * gid.x];
    let b = probe_in[2u * gid.x + 1u];
    let g = granulation(a.xyz, b.x, b.y, a.w);
    probe_out[gid.x] = vec4<u32>(bitcast<u32>(g), granulation_cell_hash(a.xyz, b.x, b.y, a.w));
}
"#;

/// Largest |GPU − CPU| of a limb-darkening value (Horner's rule; fused multiply–adds differ by
/// ~1e-7; a wrong coefficient errs by ≥ 1e-3).
const LIMB_TOLERANCE: f32 = 1e-5;
/// Largest |GPU − CPU| of a granulation value (B2 brief: ±1e-4). The value is continuous in p,
/// so an f32 difference in `atan`/`tan` that flips which granule is nearest changes it by a
/// rounding error only.
const GRAN_TOLERANCE: f32 = 1e-4;

/// 64 μ from 0 to 1 inclusive.
fn limb_probe_inputs() -> Vec<[f32; 4]> {
    (0..64).map(|k| [k as f32 / 63.0, 0.0, 0.0, 0.0]).collect()
}

/// 256 probe points (p, cell_m, radius_m, t_s): every face, edge and corner cells as well as
/// random ones, each point 5–95 % of the way across its cell (so GPU and CPU agree on the cell
/// whose hash is compared bit for bit), on the Sun and (one in four) on a 4000 K star of 3 R☉;
/// times at epoch boundaries, at the 2^22 s wrap and random. Built in f64, cast once to f32.
fn granulation_probe_inputs() -> Vec<(Vec3, f32, f32, f32)> {
    let mut state = 0x5EED_u64;
    let mut rand = || {
        state = uni_core::splitmix64(state);
        (state >> 11) as f64 / (1_u64 << 53) as f64
    };
    let times = [0.0, 511.75, 512.0, 1023.75, 1024.0, 4_194_303.75];
    (0..256)
        .map(|k| {
            let (cell, radius) = if k % 4 == 3 {
                (granule_cell_m(4000.0, 3.0 * R_SUN_M), 3.0 * R_SUN_M)
            } else {
                (GRANULE_CELL_SUN_M, R_SUN_M)
            };
            let n = gran_cells(cell as f32, radius as f32);
            let last = f64::from(n - 1);
            let mut cell_ij = || (rand() * f64::from(n)).floor().min(last);
            let (i, j) = match (k / 6) % 5 {
                0 => (0.0, 0.0),
                1 => (last, cell_ij()),
                2 => (cell_ij(), 0.0),
                3 => (last, last),
                _ => (cell_ij(), cell_ij()),
            };
            let fx = 0.05 + 0.9 * rand();
            let fy = 0.05 + 0.9 * rand();
            let a = 2.0 * (i + fx) / f64::from(n) - 1.0;
            let b = 2.0 * (j + fy) / f64::from(n) - 1.0;
            let st = glam::DVec2::new(
                (a * std::f64::consts::FRAC_PI_4).tan(),
                (b * std::f64::consts::FRAC_PI_4).tan(),
            );
            let face = (k % 6) as u32;
            let sg = if face % 2 == 0 { 1.0 } else { -1.0 };
            let v = match face / 2 {
                0 => DVec3::new(sg, st.x, st.y),
                1 => DVec3::new(st.y, sg, st.x),
                _ => DVec3::new(st.x, st.y, sg),
            };
            let t = if k < times.len() * 8 {
                times[k % times.len()]
            } else {
                (rand() * 4_194_304.0).floor() * 0.25
            };
            (
                v.normalize().as_vec3(),
                cell as f32,
                radius as f32,
                t as f32,
            )
        })
        .collect()
}

/// B2's probes (`limb_darkening`, `granulation`), called from [`crate::gpu_probes`] after it
/// registered the crate's shaders.
pub(crate) fn star_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    lib.add(STAR_PROBE_WGSL_PATH, STAR_PROBE_WGSL);
    let lib = &*lib;
    let run = |entry: &str, input: &[[f32; 4]], cases: usize| -> anyhow::Result<Vec<[u32; 4]>> {
        let src = lib.compose(STAR_PROBE_WGSL_PATH)?;
        let n = u32::try_from(cases).context("too many probe cases")?;
        let bytes = run_compute(
            gpu,
            &src,
            entry,
            bytemuck::cast_slice(input),
            (cases * 16) as u64,
            [n.div_ceil(64), 1, 1],
        )?;
        Ok(bytemuck::pod_collect_to_vec(&bytes))
    };
    let limb = || -> anyhow::Result<(bool, String)> {
        let input = limb_probe_inputs();
        let out = run("cs_limb", &input, input.len())?;
        let mut worst = 0.0_f32;
        for (c, g) in input.iter().zip(&out) {
            let gpu_ld = Vec3::new(
                f32::from_bits(g[0]),
                f32::from_bits(g[1]),
                f32::from_bits(g[2]),
            );
            let err = (gpu_ld - limb_darkening(c[0])).abs().max_element();
            worst = if worst.is_nan() || err.is_nan() {
                f32::NAN
            } else {
                worst.max(err)
            };
        }
        Ok((
            worst <= LIMB_TOLERANCE,
            format!(
                "{} μ, max |GPU − CPU| {worst:.1e} (tolerance {LIMB_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let gran = || -> anyhow::Result<(bool, String)> {
        let cases = granulation_probe_inputs();
        let input: Vec<[f32; 4]> = cases
            .iter()
            .flat_map(|(p, cell, radius, t)| [[p.x, p.y, p.z, *t], [*cell, *radius, 0.0, 0.0]])
            .collect();
        let out = run("cs_gran", &input, cases.len())?;
        let (mut worst, mut hash_mismatches) = (0.0_f32, 0);
        for ((p, cell, radius, t), g) in cases.iter().zip(&out) {
            let err = (f32::from_bits(g[0]) - granulation(*p, *cell, *radius, *t)).abs();
            worst = if worst.is_nan() || err.is_nan() {
                f32::NAN
            } else {
                worst.max(err)
            };
            if UVec3::new(g[1], g[2], g[3]) != granulation_cell_hash(*p, *cell, *radius, *t) {
                hash_mismatches += 1;
            }
        }
        Ok((
            worst <= GRAN_TOLERANCE && hash_mismatches == 0,
            format!(
                "{} points, {hash_mismatches} lattice-hash mismatches, max |GPU − CPU| {worst:.1e} (tolerance {GRAN_TOLERANCE:.0e})",
                cases.len()
            ),
        ))
    };
    let result = |name: &str, outcome: anyhow::Result<(bool, String)>| {
        let (pass, detail) = outcome.unwrap_or_else(|e| (false, format!("{e:#}")));
        ProbeResult {
            name: name.to_owned(),
            pass,
            detail,
        }
    };
    vec![
        result("star_limb_darkening", limb()),
        result("star_granulation", gran()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rings::RingGpu;
    use glam::{DQuat, DVec3};
    use std::mem::{offset_of, size_of};
    use uni_core::{AU, M_V_SUN, PC, R_SUN_M, TEFF_SUN_K, splitmix64, star_illuminance_lux};
    use uni_gfx::{ShaderLib, validate_wgsl};
    use uni_sim::snapshot::StarSurfaceView;
    use wgpu::naga;

    /// Neckel & Labs 1994, Table I, column F/I (disk-averaged over disk-centre intensity, their
    /// eq. 6) at 640.970, 541.760 and 445.125 nm.
    const PAPER_F_OVER_I: [f32; 3] = [0.8290, 0.8002, 0.7596];
    /// Rec. 709 luminance weights (ITU-R BT.709-6 item 3.2).
    const Y709: [f64; 3] = [0.2126, 0.7152, 0.0722];

    /// Luminous intensity [cd] of a star of absolute V magnitude `abs_v`: its illuminance at
    /// 10 pc times (10 pc)², the same zero point as `uni_sim`'s (Allen 1973).
    fn intensity_cd(abs_v: f64) -> f64 {
        star_illuminance_lux(abs_v) * (10.0 * PC) * (10.0 * PC)
    }

    fn star_view(rel: DVec3, radius_m: f64, teff_k: f64, abs_v: f64) -> StarSurfaceView {
        StarSurfaceView {
            id: 0,
            catalog_index: None,
            rel_pos_m: rel,
            orient: DQuat::IDENTITY,
            radius_m,
            teff_k,
            luminous_intensity_cd: intensity_cd(abs_v),
            proj_radius_px: 100.0,
        }
    }

    /// Unit vector from a deterministic 64-bit stream (uniform on the sphere).
    fn sphere_point(state: &mut u64) -> Vec3 {
        let mut next = || {
            *state = splitmix64(*state);
            (*state >> 11) as f64 / (1u64 << 53) as f64
        };
        let z = 2.0 * next() - 1.0;
        let phi = std::f64::consts::TAU * next();
        let s = (1.0 - z * z).sqrt();
        DVec3::new(s * phi.cos(), s * phi.sin(), z).as_vec3()
    }

    #[test]
    fn limb_darkening_normalised() {
        let centre = limb_darkening(1.0);
        assert!((centre - Vec3::ONE).abs().max_element() <= 1e-5, "{centre}");
        // Decreasing toward the limb: LD rises strictly with μ on every channel.
        let mut prev = limb_darkening(0.0);
        for k in 1..=1000 {
            let ld = limb_darkening(k as f32 / 1000.0);
            assert!(
                ld.cmpgt(prev).all(),
                "μ = {}: {ld} after {prev}",
                k as f32 / 1000.0
            );
            prev = ld;
        }
        // Out-of-range μ is clamped.
        assert_eq!(limb_darkening(1.5), limb_darkening(1.0));
        assert_eq!(limb_darkening(-0.5), limb_darkening(0.0));
    }

    #[test]
    fn limb_darkening_550_half() {
        let ld = limb_darkening(0.5).y;
        assert!((0.68..=0.80).contains(&ld), "{ld}");
        assert!((ld - LD_550_AT_HALF_PAPER).abs() <= 1e-3, "{ld}");
    }

    #[test]
    fn limb_redder_than_centre() {
        let ld = limb_darkening(0.3);
        assert!(ld.z < ld.x, "{ld}");
    }

    #[test]
    fn limb_darkening_mean_matches_paper_f_over_i() {
        let mean = limb_darkening_mean();
        for (c, want) in PAPER_F_OVER_I.iter().enumerate() {
            assert!(
                (mean[c] - want).abs() <= 1e-4,
                "channel {c}: {} vs {want}",
                mean[c]
            );
        }
        // And against a numerical integral of the twin.
        let n = 100_000;
        let mut sum = Vec3::ZERO;
        for k in 0..n {
            let mu = (k as f32 + 0.5) / n as f32;
            sum += limb_darkening(mu) * mu;
        }
        let numeric = 2.0 * sum / n as f32;
        assert!(
            (numeric - mean).abs().max_element() <= 1e-4,
            "{numeric} vs {mean}"
        );
    }

    /// Illuminance [lx] the rendered disk delivers to the camera: luminance of the pass's
    /// per-pixel radiance (granulation faded out, as at these distances), integrated over the
    /// disk's solid angle with the cosine factor, by the midpoint rule in θ, the angle from the
    /// direction to the centre. The ray at θ passes |o|·sin θ radii from the centre, so the
    /// surface it hits has μ = √(1 − |o|² sin² θ).
    fn disk_illuminance_lux(view: &StarSurfaceView) -> f64 {
        let gpu = star_gpu(view, 1.0);
        let l0 = Vec3::from_slice(&gpu.l0[..3]).as_dvec3();
        let o = view.rel_pos_m.length() / view.radius_m;
        let theta_s = (1.0 / o).asin();
        let n = 20_000;
        let dt = theta_s / f64::from(n);
        let mut e = 0.0;
        for k in 0..n {
            let theta = (f64::from(k) + 0.5) * dt;
            let f = o * theta.sin();
            let mu = (1.0 - f * f).max(0.0).sqrt();
            let l = l0 * limb_darkening(mu as f32).as_dvec3();
            let y = Y709[0] * l.x + Y709[1] * l.y + Y709[2] * l.z;
            e += std::f64::consts::TAU * y * theta.cos() * theta.sin() * dt;
        }
        e
    }

    /// Review Focus 4: the resolved disk delivers the catalogue flux, so the point → disk
    /// handover keeps the star's magnitude.
    #[test]
    fn star_surface_flux_matches_catalogue_magnitude() {
        // The Sun at 1 AU: V = −26.74.
        let sun = star_view(
            DVec3::new(0.3, -0.4, 0.866) * AU,
            R_SUN_M,
            TEFF_SUN_K,
            M_V_SUN,
        );
        let e = disk_illuminance_lux(&sun);
        let want = star_illuminance_lux(-26.74);
        assert!((e / want - 1.0).abs() <= 0.01, "Sun: {e} lx vs {want} lx");

        // α Cen A-like: M_V 4.38, 5790 K, 1.2234 R☉ (Kervella et al. 2017), at 1.34 pc.
        let d = 1.34 * PC;
        let star = star_view(DVec3::new(0.0, d, 0.0), 1.2234 * R_SUN_M, 5790.0, 4.38);
        let m = 4.38 + 5.0 * (1.34_f64 / 10.0).log10();
        let e = disk_illuminance_lux(&star);
        let want = star_illuminance_lux(m);
        assert!(
            (e / want - 1.0).abs() <= 0.01,
            "α Cen A: {e} lx vs {want} lx"
        );
    }

    /// Review Focus 2, review C2: a camera inside a photosphere draws nothing and warns.
    #[test]
    fn camera_inside_star_draws_nothing() {
        for rel in [DVec3::new(0.5 * R_SUN_M, 0.0, 0.0), DVec3::ZERO] {
            let scene = BodyScene {
                stars: vec![star_view(rel, R_SUN_M, TEFF_SUN_K, M_V_SUN)],
                names: vec![(0, "Sun".to_owned())],
                ..Default::default()
            };
            let frame = plan_stars(&scene, 1.0, 1080.0);
            assert!(frame.draws.is_empty(), "camera at {rel}");
            assert_eq!(frame.warnings, vec!["Inside Sun".to_owned()]);
        }
        // Outside, the same star draws one finite cap and no warning.
        let scene = BodyScene {
            stars: vec![star_view(
                DVec3::new(0.0, 0.0, -3.0 * R_SUN_M),
                R_SUN_M,
                TEFF_SUN_K,
                M_V_SUN,
            )],
            names: vec![(0, "Sun".to_owned())],
            ..Default::default()
        };
        let frame = plan_stars(&scene, 1.0, 1080.0);
        assert!(frame.warnings.is_empty());
        assert_eq!(frame.draws.len(), 1);
        let draw = &frame.draws[0];
        assert!(!draw.vertices.is_empty());
        assert!(draw.vertices.iter().flatten().all(|v| v.is_finite()));
        let g = bytemuck::bytes_of(&draw.gpu);
        assert!(
            bytemuck::cast_slice::<u8, f32>(g)
                .iter()
                .all(|v| v.is_finite())
        );
    }

    #[test]
    fn granulation_statistics() {
        let cell = granule_cell_m(TEFF_SUN_K, R_SUN_M) as f32;
        let radius = R_SUN_M as f32;

        // Mean 1 ± 0.01 and RMS contrast in [0.08, 0.20] over 4096 points and times.
        let mut state = 7_u64;
        let (mut sum, mut sum2) = (0.0_f64, 0.0_f64);
        for k in 0..4096 {
            let p = sphere_point(&mut state);
            let t = (k as f32) * 1021.3 % 4_194_304.0;
            let g = f64::from(granulation(p, cell, radius, t));
            assert!(g.is_finite() && g >= 0.0, "{g}");
            sum += g;
            sum2 += g * g;
        }
        let mean = sum / 4096.0;
        let rms = (sum2 / 4096.0 - mean * mean).sqrt() / mean;
        eprintln!("granulation: mean {mean:.4}, relative RMS {rms:.4}");
        assert!((mean - 1.0).abs() <= 0.01, "mean {mean}");
        assert!((0.08..=0.20).contains(&rms), "RMS {rms}");

        // Mean cell spacing: √(area per granule). A 64 × 64 grid over a patch of one cube face
        // finds every granule owning a sample; those whose centre lies inside the patch are
        // counted against the patch's area on the sphere.
        let n = gran_cells(cell, radius);
        let half = 4.0 * std::f32::consts::FRAC_PI_2 / n as f32; // ±4 lattice cells in angle
        let (mut owners, mut inside) = (std::collections::HashSet::new(), 0);
        for i in 0..64 {
            for j in 0..64 {
                let a = -half + 2.0 * half * (i as f32 + 0.5) / 64.0;
                let b = -half + 2.0 * half * (j as f32 + 0.5) / 64.0;
                let p = Vec3::new(1.0, a.tan(), b.tan()).normalize();
                let (near, centre) = gran_nearest(p, n, 0);
                if owners.insert(near.id1) {
                    let a_c = (centre.y / centre.x).atan();
                    let b_c = (centre.z / centre.x).atan();
                    if a_c.abs() < half && b_c.abs() < half {
                        inside += 1;
                    }
                }
            }
        }
        // Area of the patch (|a|, |b| < half on the equal-angle face): ∫∫ dΩ.
        let mut area = 0.0_f64;
        let m = 400;
        for i in 0..m {
            for j in 0..m {
                let a =
                    f64::from(-half) + 2.0 * f64::from(half) * (f64::from(i) + 0.5) / f64::from(m);
                let b =
                    f64::from(-half) + 2.0 * f64::from(half) * (f64::from(j) + 0.5) / f64::from(m);
                let (s, t) = (a.tan(), b.tan());
                // dΩ = ds dt / (1 + s² + t²)^{3/2}, ds = sec² a da.
                let jac = (1.0 + s * s) * (1.0 + t * t) / (1.0 + s * s + t * t).powf(1.5);
                area += jac * (2.0 * f64::from(half) / f64::from(m)).powi(2);
            }
        }
        let spacing_mm = (area / f64::from(inside)).sqrt() * R_SUN_M / 1e6;
        eprintln!("granulation: {inside} granules in the patch, mean spacing {spacing_mm:.3} Mm");
        assert!((0.9..=1.6).contains(&spacing_mm), "spacing {spacing_mm} Mm");

        // Autocorrelation time: the lag at which the temporal autocorrelation of G − 1, averaged
        // over 128 points and 32 start times spread over a layer period, first falls below 1/e.
        let mut state = 11_u64;
        let points: Vec<Vec3> = (0..128).map(|_| sphere_point(&mut state)).collect();
        let step = 30.0_f32;
        let lags = 48;
        let mut acf = vec![0.0_f64; lags];
        for s in 0..32 {
            let t0 = 100_000.0 + s as f32 * 32.0;
            for p in &points {
                let g0 = f64::from(granulation(*p, cell, radius, t0)) - 1.0;
                for (l, a) in acf.iter_mut().enumerate() {
                    let g = f64::from(granulation(*p, cell, radius, t0 + step * l as f32)) - 1.0;
                    *a += g0 * g;
                }
            }
        }
        let a0 = acf[0];
        let tau = (1..lags)
            .find(|&l| acf[l] / a0 < (-1.0_f64).exp())
            .map(|l| {
                let (y0, y1) = (acf[l - 1] / a0, acf[l] / a0);
                let x = (y0 - (-1.0_f64).exp()) / (y0 - y1);
                (l as f64 - 1.0 + x) * f64::from(step)
            })
            .expect("autocorrelation never falls below 1/e");
        eprintln!("granulation: autocorrelation time {:.2} min", tau / 60.0);
        assert!((4.0..=12.0).contains(&(tau / 60.0)), "{} min", tau / 60.0);
    }

    /// The per-layer intensity's mean and standard deviation, which `granulation` divides out,
    /// match a fresh 40 000-point estimate.
    #[test]
    fn granulation_layer_normalisation() {
        let n = gran_cells(1.3e6, R_SUN_M as f32);
        let mut state = 3_u64;
        let (mut sum, mut sum2) = (0.0_f64, 0.0_f64);
        let count = 40_000;
        for k in 0..count {
            let p = sphere_point(&mut state);
            let v = f64::from(gran_layer(p, n, (k % 97) as u32));
            sum += v;
            sum2 += v * v;
        }
        let mean = sum / f64::from(count);
        let sd = (sum2 / f64::from(count) - mean * mean).sqrt();
        eprintln!("granulation layer: mean {mean:.5}, sd {sd:.5}");
        assert!((mean - f64::from(GRAN_MEAN)).abs() <= 0.005, "mean {mean}");
        assert!((sd / f64::from(GRAN_SIGMA) - 1.0).abs() <= 0.02, "sd {sd}");
    }

    #[test]
    fn granulation_is_continuous_and_periodic() {
        let cell = 1.3e6_f32;
        let radius = R_SUN_M as f32;
        let mut state = 5_u64;
        for _ in 0..64 {
            let p = sphere_point(&mut state);
            // Across an epoch boundary of either layer and across the 2^22 s wrap of sim time.
            for t in [1024.0_f32, 512.0, 4_194_304.0] {
                let before = granulation(p, cell, radius, t - 0.25);
                let after = granulation(p, cell, radius, t);
                assert!((before - after).abs() < 0.02, "t {t}: {before} → {after}");
            }
            let wrapped = granulation(p, cell, radius, 0.0);
            let end = granulation(p, cell, radius, 4_194_304.0);
            assert!((wrapped - end).abs() < 1e-5, "{wrapped} vs {end}");
        }
        // Continuous in space, also across cube-face edges (p on the x = y plane): points
        // 2e-6 apart (~0.001 cell) on either side of the edge differ by far less than a seam's
        // jump (a lane cut off at the edge would jump by ~GRANULATION_RMS_550).
        let step = 2e-6_f32;
        for k in 0..400 {
            let z = -0.5 + k as f32 / 400.0;
            let a = Vec3::new(1.0, 1.0 - step, z).normalize();
            let b = Vec3::new(1.0 - step, 1.0, z).normalize();
            let (ga, gb) = (
                granulation(a, cell, radius, 3000.0),
                granulation(b, cell, radius, 3000.0),
            );
            assert!((ga - gb).abs() < 0.02, "z {z}: {ga} vs {gb}");
        }
    }

    #[test]
    fn granule_cells_scale_with_the_star() {
        assert!((granule_cell_m(TEFF_SUN_K, R_SUN_M) - 1.3e6).abs() < 1.0);
        let giant = granule_cell_m(4000.0, 10.0 * R_SUN_M);
        assert!((giant - 1.3e6 * 4000.0 / 5772.0 * 10.0).abs() < 1.0);
        assert_eq!(granulation_fade(1.0), 0.0);
        assert_eq!(granulation_fade(1.25), 0.0);
        assert_eq!(granulation_fade(2.0), 1.0);
        assert_eq!(granulation_fade(1e6), 1.0);
        assert!(granulation_fade(1.6) > 0.0 && granulation_fade(1.6) < 1.0);
    }

    fn struct_layout(path: &str, name: &str) -> (usize, Vec<(String, usize)>) {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(path).unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let (handle, ty) = m
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name} in {path}"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let offsets = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (layouter[handle].size as usize, offsets)
    }

    #[test]
    fn star_ring_uniforms_layout_match_wgsl() {
        let (size, offsets) = struct_layout(crate::STAR_SURFACE_WGSL, "StarGpu");
        assert_eq!(size, size_of::<StarGpu>());
        assert_eq!(
            offsets,
            [
                ("cam_unit".to_owned(), offset_of!(StarGpu, cam_unit)),
                ("to_body_x".to_owned(), offset_of!(StarGpu, to_body_x)),
                ("to_body_y".to_owned(), offset_of!(StarGpu, to_body_y)),
                ("to_body_z".to_owned(), offset_of!(StarGpu, to_body_z)),
                ("l0".to_owned(), offset_of!(StarGpu, l0)),
                ("params".to_owned(), offset_of!(StarGpu, params)),
            ]
        );
        let (size, offsets) = struct_layout(crate::RINGS_WGSL, "RingGpu");
        assert_eq!(size, size_of::<RingGpu>());
        assert_eq!(
            offsets,
            [
                ("centre".to_owned(), offset_of!(RingGpu, centre)),
                ("axis_x".to_owned(), offset_of!(RingGpu, axis_x)),
                ("axis_y".to_owned(), offset_of!(RingGpu, axis_y)),
                ("normal".to_owned(), offset_of!(RingGpu, normal)),
                ("sun".to_owned(), offset_of!(RingGpu, sun)),
                ("sun_scaled".to_owned(), offset_of!(RingGpu, sun_scaled)),
                ("radii".to_owned(), offset_of!(RingGpu, radii)),
                ("sun_color".to_owned(), offset_of!(RingGpu, sun_color)),
            ]
        );
    }

    /// A scalar constant of the composed `path`, read from naga's module.
    fn wgsl_scalar(path: &str, name: &str) -> f64 {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(path).unwrap()).unwrap();
        let (_, c) = m
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("const {name} in {path}"));
        match m.global_expressions[c.init] {
            naga::Expression::Literal(naga::Literal::F32(v)) => f64::from(v),
            naga::Expression::Literal(naga::Literal::U32(v)) => f64::from(v),
            naga::Expression::Literal(naga::Literal::I32(v)) => f64::from(v),
            ref e => panic!("const {name} = {e:?}"),
        }
    }

    /// The WGSL twin's constants equal the Rust ones (the GPU probe catches a drift only under
    /// `--selftest`; this catches it in `cargo test`).
    #[test]
    fn star_wgsl_constants_match_rust() {
        let path = crate::STAR_SURFACE_WGSL;
        for (name, rust) in [
            ("GRAN_PERIOD_S", GRAN_PERIOD_S),
            ("GRAN_MAX_CELLS", GRAN_MAX_CELLS),
            ("GRAN_JITTER_MIN", GRAN_JITTER_MIN),
            ("GRAN_JITTER_SPAN", GRAN_JITTER_SPAN),
            ("GRAN_LANE", GRAN_LANE),
            ("GRAN_BRIGHT_MIN", GRAN_BRIGHT_MIN),
            ("GRAN_BRIGHT_SPAN", GRAN_BRIGHT_SPAN),
            ("GRAN_MEAN", GRAN_MEAN),
            ("GRAN_SIGMA", GRAN_SIGMA),
            ("GRAN_RMS_550", GRANULATION_RMS_550),
            ("GRAN_FADE_LO_PX", GRAN_FADE_LO_PX),
            ("GRAN_FADE_HI_PX", GRAN_FADE_HI_PX),
            ("FRAC_4_PI", FRAC_4_PI),
            ("FRAC_PI_4", FRAC_PI_4),
        ] {
            assert_eq!(wgsl_scalar(path, name), f64::from(rust), "{name}");
        }
        assert_eq!(wgsl_scalar(path, "GRAN_EPOCHS"), f64::from(GRAN_EPOCHS));
        assert_eq!(wgsl_scalar(path, "GRAN_NONE"), f64::from(GRAN_NONE));
    }

    /// Both pass shaders and both probe shaders compose (cross-file imports included) and
    /// validate with naga.
    #[test]
    fn star_ring_shaders_validate() {
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        lib.add(STAR_PROBE_WGSL_PATH, STAR_PROBE_WGSL);
        lib.add(
            crate::rings::RING_PROBE_WGSL_PATH,
            crate::rings::RING_PROBE_WGSL,
        );
        for path in [
            crate::STAR_SURFACE_WGSL,
            crate::RINGS_WGSL,
            STAR_PROBE_WGSL_PATH,
            crate::rings::RING_PROBE_WGSL_PATH,
        ] {
            let src = lib.compose(path).unwrap();
            if let Err(e) = validate_wgsl(&src) {
                panic!("{path}: {e}\n{src}");
            }
        }
        // The pass shader has the entry points the pipeline names.
        let m = validate_wgsl(&lib.compose(crate::STAR_SURFACE_WGSL).unwrap()).unwrap();
        let names: Vec<_> = m.entry_points.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"vs_main") && names.contains(&"fs_main"),
            "{names:?}"
        );
    }

    /// The probe inputs keep every point at least 5 % of a lattice cell from its cell's edges,
    /// so the GPU and CPU agree on the cell whose hash the probe compares bit for bit.
    #[test]
    fn star_probe_inputs_stay_inside_cells() {
        let inputs = granulation_probe_inputs();
        assert_eq!(inputs.len(), 256);
        for (p, cell, radius, _t) in &inputs {
            let n = gran_cells(*cell, *radius);
            let (_, st) = cube_coord(*p);
            let q = (st.map(f32::atan) * FRAC_4_PI + 1.0) * 0.5 * n as f32;
            let frac = q - q.floor();
            assert!(
                frac.min_element() >= 0.04 && frac.max_element() <= 0.96,
                "{q}"
            );
        }
    }
}
