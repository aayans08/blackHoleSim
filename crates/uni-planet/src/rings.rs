//! Saturn's rings, pass "rings" (Task B2).
//!
//! An annulus mesh in the ring plane (256 segments, exact plane depth), depth-tested without
//! depth write, premultiplied, `lens_mask.g` = the opacity along the view ray and `lens_mask.a`
//! = the star share 1 − w ([`crate::body::star_share`], lead ruling B6 I1), both with `Max`. The
//! exposure weight w is the planet's ([`crate::body::exposure_weight`]), drawn at the log-space
//! `pre_mix(star_pre, scene_pre, w)` (uni-gfx, task B11). The radii come from the profile (A8's
//! span; uni-sim's `RingView` uses the same, lead ruling after A4). The app pushes this pass only
//! when `WorldData::rings` loaded (ruling A.3); without it the load warning is shown. Card note
//! (review m11): `Rings: Solar System Scope texture; phase function approximate`.
//!
//! **Photometry.** Radiance L = (E/π)·ρ(r)·P(θ)·A(τ, μ₀, μ) [cd/m²], i.e. I/F = ρ·P·A, with
//! - E the sunlight at the planet [lx] × [`planet_shadow_on_ring`] (the ray to the Sun against
//!   the planet's ellipsoid, penumbra by [`crate::sun_visible_fraction`]);
//! - ρ(r) the profile's colour ("reflectance"; A8: the Solar System Scope texture's colour,
//!   decoded to linear) and τ(r) = −ln(1 − α) from its opacity α, the normal optical depth;
//! - P the phase function [`ring_phase`]: lit face 0.8·HG(−0.3) + 0.2·HG(+0.6), unlit face
//!   HG(+0.6) (B2 brief);
//! - A the geometry of single scattering in a particle layer, [`ring_layer_factor`].
//!
//! The brief writes the lit face as ρ·α·P and the unlit face as ρ·α(1 − α)·HG(+0.6), with no
//! dependence on where the Sun and the camera are. B2 keeps ρ and P exactly and replaces the
//! opacity factors by the classical single-scattering layer (Salo & French 2010, Icarus 210,
//! 785, eq. 6; decision recorded in the B2 report, awaiting the lead's ruling, review I2): it
//! equals the brief's ρ·P for an opaque ring seen from the Sun's direction, makes
//! both faces go dark when the Sun is in the ring plane (equinox; the brief's form would keep
//! them fully lit), keeps the opaque B ring dark on its unlit face while the C ring, the Cassini
//! division and the A ring glow there, and brightens thin rings seen at a slant. The test
//! `ring_photometry_with_the_real_profile` compares the two with the committed profile at the
//! plan's Saturn bookmarks: the B ring's lit face at `saturn-ring-plane` is I/F 0.48 here (0.32 in
//! the brief's form); the unlit face at `saturn-night-side` is 0.0035–0.031 here (0.027–0.063
//! in the brief's form; region means), both far below the 0.35 the plan's exposure table
//! assumed. [`ring_i_over_f`] is the CPU twin for re-deriving those exposure rows.
//!
//! Approximations (labelled by the card note): the texture's colour is not a measured
//! reflectance; one phase function for every ring; no multiple scattering, Saturnshine, moon
//! shadows or ring self-shadowing; the planet's penumbra uses the Sun's angular radius measured
//! in the ellipsoid's scaled space (≤ 10 % error in its 50 km width for Saturn's flattening).

use anyhow::Context as _;
use glam::{DVec3, Vec3, Vec4};
use uni_core::{R_SUN_M, TEFF_SUN_K, blackbody_linear_srgb};
use uni_data::RingProfile;
use uni_gfx::probe::{ProbeResult, run_compute};
use uni_gfx::{
    DEPTH_FORMAT, FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};

use crate::body::{BodyScene, exposure_weight, lens_mask_target};

/// Segments of the annulus (B2 brief; the performance ladder's step is 128). Twin of
/// `RING_SEGMENTS` in `rings.wgsl`.
pub const RING_SEGMENTS: u32 = 256;

/// Lit face: weight of the backscattering lobe and the two asymmetry parameters; unlit face:
/// the forward lobe. Values as the M1 plan gives them (B2 brief). Checked qualitatively: the
/// main rings' particles backscatter (Doyle, Dones & Cuzzi 1989, Icarus 80, 104, fitted the
/// B ring with backscattering particles, as Poulet et al. 2002, Icarus 158, 224, §2 summarise),
/// and dusty, small particles forward-scatter (Hedman & Stark 2015, ApJ 811, 67: three-lobe HG
/// fits to the G and D rings). The cited Cuzzi et al. 2009 chapter (*Saturn from
/// Cassini–Huygens*, ch. 15) was not accessible to B2, so the lobe weights and g values are the
/// plan's, not re-derived from it; the card note labels the phase function as approximate.
const LIT_W: f32 = 0.8;
const LIT_G1: f32 = -0.3;
const LIT_G2: f32 = 0.6;
const UNLIT_G: f32 = 0.6;

/// Cosines below this are treated as this, so 1/μ stays finite. Twin of `RING_MU_MIN`.
const MU_MIN: f32 = 1e-6;

/// Normal opacities at or above this are treated as this (τ ≤ 9.21), so τ stays finite. Twin of
/// `RING_ALPHA_MAX`.
const ALPHA_MAX: f32 = 0.9999;

/// Profile opacities at or below this are not drawn (the pass discards them).
const ALPHA_DRAWN_MIN: f32 = 1e-5;

/// The Sun's angular radius [rad] at Saturn's mean distance, used by the public
/// [`planet_shadow_on_ring`] (which has no Sun distance): R☉ / a♄ with R☉ = 6.957e8 m (IAU 2015
/// Resolution B3) and a♄ = 9.53667594 AU (Standish, JPL "Keplerian elements for approximate
/// positions of the major planets", Table 1, J2000). Saturn's distance varies by ±6 %, so does
/// this; the pass uses the frame's own value.
pub const SUN_ANGULAR_RADIUS_AT_SATURN_RAD: f32 = 4.876e-4;

/// The ring pass (`RenderFeature` named "rings"). It draws the rings of `BodyScene::bodies`
/// whose view has a `ring`, all with the one profile.
pub struct RingRenderer {
    input: SharedInput<BodyScene>,
    inner_m: f64,
    outer_m: f64,
    pipeline: wgpu::RenderPipeline,
    ring_bgl: wgpu::BindGroupLayout,
    profile_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    ring_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// Rings `ring_buffer` holds.
    capacity: usize,
    /// Rings drawn this frame.
    count: u32,
}

impl RingRenderer {
    /// Uploads `profile` (radial reflectance and opacity, premultiplied, with mips) and builds
    /// the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        profile: &RingProfile,
        input: SharedInput<BodyScene>,
    ) -> anyhow::Result<Self> {
        let (inner_m, outer_m) = (profile.inner_m, profile.outer_m);
        anyhow::ensure!(
            inner_m.is_finite() && outer_m.is_finite() && 0.0 <= inner_m && inner_m < outer_m,
            "ring profile radii {inner_m} m – {outer_m} m"
        );
        anyhow::ensure!(
            profile.samples.len() >= 2,
            "ring profile has fewer than 2 samples"
        );
        let device = &gpu.device;
        let width = u32::try_from(profile.samples.len()).context("ring profile too long")?;
        let max = device.limits().max_texture_dimension_2d;
        anyhow::ensure!(
            width <= max,
            "ring profile of {width} samples; the device allows {max}"
        );

        crate::register_shaders(lib);
        let module = lib.module(device, crate::RINGS_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

        let mips = profile_mips(profile);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ring profile"),
            size: wgpu::Extent3d {
                width,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: mips.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (level, texels) in mips.iter().enumerate() {
            let half: Vec<u16> = texels
                .iter()
                .flatten()
                .map(|&x| uni_data::f32_to_f16(x))
                .collect();
            let w = texels.len() as u32;
            gpu.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&half),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(w * 8),
                    rows_per_image: Some(1),
                },
                wgpu::Extent3d {
                    width: w,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
        }
        let profile_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ring profile sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let ring_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rings bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(size_of::<RingGpu>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rings layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&ring_bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("rings"),
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
                targets: &[
                    // Premultiplied "over": the ring adds its light and transmits 1 − α of
                    // what is behind it. hdr's alpha is left alone.
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    // lens_mask.g = max(g, opacity), lens_mask.a = max(a, 1 − w).
                    Some(lens_mask_target()),
                ],
            }),
            // Both faces are drawn.
            primitive: wgpu::PrimitiveState::default(),
            // Behind a body's globe the rings are hidden; in front they cover it. No depth
            // write: later passes still see the globe or the sky behind the translucent rings.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let capacity = 1;
        let ring_buffer = ring_buffer(device, capacity);
        let bind_group = ring_bind_group(device, &ring_bgl, &ring_buffer, &profile_view, &sampler);
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the ring pass: {err}");
        }
        Ok(Self {
            input,
            inner_m,
            outer_m,
            pipeline,
            ring_bgl,
            profile_view,
            sampler,
            ring_buffer,
            bind_group,
            capacity,
            count: 0,
        })
    }
}

fn ring_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ring records"),
        size: (capacity * size_of::<RingGpu>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn ring_bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
    view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rings bg"),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

impl RenderFeature for RingRenderer {
    fn name(&self) -> &'static str {
        "rings"
    }

    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {
        let rings = self
            .input
            .with(|scene| plan_rings(scene, self.inner_m, self.outer_m));
        self.count = rings.len() as u32;
        if rings.is_empty() {
            return;
        }
        if rings.len() > self.capacity {
            self.capacity = rings.len().next_power_of_two();
            self.ring_buffer = ring_buffer(&gpu.device, self.capacity);
            self.bind_group = ring_bind_group(
                &gpu.device,
                &self.ring_bgl,
                &self.ring_buffer,
                &self.profile_view,
                &self.sampler,
            );
        }
        gpu.queue
            .write_buffer(&self.ring_buffer, 0, bytemuck::cast_slice(&rings));
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
            label: Some("rings"),
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
            // Read-only depth: tested, never written.
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
                depth_ops: None,
                stencil_ops: None,
            }),
            timestamp_writes: timer.render_writes("rings"),
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, frame.camera, &[]);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..6 * RING_SEGMENTS, 0..self.count);
    }
}

/// Henyey & Greenstein 1941 (ApJ 93, 70): (1 − g²)/(1 + g² − 2g cos θ)^{3/2}, mean 1 over the
/// sphere. Twin of `henyey_greenstein` in `rings.wgsl`.
fn henyey_greenstein(g: f32, cos_t: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_t;
    (1.0 - g2) / (denom * denom.sqrt())
}

/// Ring phase function for scattering angle cosine `cos_alpha`, lit or unlit face, normalised
/// over the sphere (∫ dΩ/4π = 1). Twin of the WGSL in `rings.wgsl`.
///
/// `cos_alpha` is the cosine of the scattering angle θ between the sunlight's direction of
/// travel and the direction to the camera (θ = 180° − the phase angle; −1 is exact
/// backscatter). Lit face: 0.8·HG(g = −0.3) + 0.2·HG(g = +0.6); unlit face: HG(g = +0.6)
/// (B2 brief; checked as noted at `LIT_W`).
pub fn ring_phase(cos_alpha: f32, lit_face: bool) -> f32 {
    let c = cos_alpha.clamp(-1.0, 1.0);
    if lit_face {
        return LIT_W * henyey_greenstein(LIT_G1, c) + (1.0 - LIT_W) * henyey_greenstein(LIT_G2, c);
    }
    henyey_greenstein(UNLIT_G, c)
}

/// Geometry of single scattering in a plane-parallel layer of particles of normal optical depth
/// `tau`, lit at `mu0` and seen at `mu` (cosines of the angles from the ring normal), times 2 so
/// that an opaque layer seen from the Sun's own direction gives 1. I/F = ρ·P·this. Twin of
/// `ring_layer_factor` in `rings.wgsl`.
///
/// First-order (single) scattering in a plane-parallel slab (Chandrasekhar 1960, *Radiative
/// Transfer*, diffuse reflection and transmission to first order), the "classical
/// single-scattering model" of Saturn's rings that fits HST photometry of the A, C and inner B
/// rings (Cuzzi, French & Dones 2002, Icarus 158, 199; see also Cuzzi et al. 1984, *Planetary
/// Rings*, pp. 73–199):
/// - reflected: I/F = (A·P/4)·μ₀/(μ₀ + μ)·(1 − e^{−τ(1/μ₀ + 1/μ)}), with A the particles' Bond
///   albedo (ϖ₀). This is eq. 6 of Salo & French 2010 (Icarus 210, 785; arXiv:1007.0349, §3.1,
///   with μ′ = μ₀ = |sin B′| for the Sun and μ = |sin B| for the observer), read there in fix
///   round 2;
/// - transmitted: I/F = (A·P/4)·μ₀/(μ₀ − μ)·(e^{−τ/μ₀} − e^{−τ/μ}) (Chandrasekhar 1960).
///
/// With ρ ≡ A/8 (here ρ is the texture's colour, not a measured albedo) the factor is:
/// - lit face: 2μ₀/(μ₀ + μ)·(1 − e^{−τ(1/μ₀ + 1/μ)});
/// - unlit face: 2μ₀/(μ₀ − μ)·(e^{−τ/μ₀} − e^{−τ/μ}), and its limit 2(τ/μ)·e^{−τ/μ} at μ = μ₀.
///
/// The brief's lit-face factor α is this model's with the geometry factor 2μ₀/(μ₀ + μ) divided
/// out and the slant path 2/μ_eff = 1/μ₀ + 1/μ replaced by the normal one: Salo & French 2010,
/// footnote 2, write (μ + μ₀)/(2μ₀)·(I/F) = (A·P/8)·(1 − e^{−2τ/μ_eff}), i.e. ρ·P·α_eff.
///
/// 0 when τ ≤ 0 or μ₀ ≤ 0 (no particles, or the Sun in the ring plane).
pub fn ring_layer_factor(tau: f32, mu0: f32, mu: f32, lit_face: bool) -> f32 {
    if !(tau > 0.0) || !(mu0 > 0.0) {
        return 0.0;
    }
    let mu0 = mu0.max(MU_MIN);
    let mu = mu.max(MU_MIN);
    if lit_face {
        return 2.0 * mu0 / (mu0 + mu) * (1.0 - (-tau * (1.0 / mu0 + 1.0 / mu)).exp());
    }
    let e0 = (-tau / mu0).exp();
    let e1 = (-tau / mu).exp();
    if (mu0 - mu).abs() <= 1e-4 * mu {
        return 2.0 * (tau / mu) * e1;
    }
    2.0 * mu0 / (mu0 - mu) * (e0 - e1)
}

/// The rings' I/F (radiance over E/π, before the planet's shadow): ρ·P(θ)·A(τ, μ₀, μ), with τ =
/// −ln(1 − α) from the profile's normal opacity `alpha` (clamped to 0.9999; 0 at or below 1e-5,
/// which the pass discards), `cos_scatter` the scattering-angle cosine (−1 = backscatter, i.e.
/// −cos of the phase angle), `mu0` and `mu` the Sun's and the camera's |cosines| from the ring
/// normal, `lit_face` whether the camera is on the Sun's side. The CPU twin of the composition in
/// `rings.wgsl`'s `fs_main` (per channel there; `rho` here is one channel or a luminance), for
/// re-deriving the Saturn bookmarks' exposure (review I2: with the committed profile the unlit
/// face at `saturn-night-side` is 0.0035–0.031 of E/π, region means, not the plan's 0.35; see
/// `ring_photometry_with_the_real_profile`).
pub fn ring_i_over_f(
    rho: f32,
    alpha: f32,
    mu0: f32,
    mu: f32,
    cos_scatter: f32,
    lit_face: bool,
) -> f32 {
    if !(alpha > ALPHA_DRAWN_MIN) {
        return 0.0;
    }
    let tau = -(1.0 - alpha.min(ALPHA_MAX)).ln();
    rho * ring_phase(cos_scatter, lit_face) * ring_layer_factor(tau, mu0, mu.max(MU_MIN), lit_face)
}

/// The f32 core of [`planet_shadow_on_ring`], twin of `planet_shadow_on_ring` in `rings.wgsl`:
/// `p` the ring point in the planet's unit-ellipsoid space (body coordinates / radii), `d` the
/// unit direction to the Sun in that space, `sun_ang` the Sun's angular radius [rad]. The planet
/// is the unit sphere there: the ray from `p` along `d` meets the ellipsoid exactly when it meets
/// that sphere, so the umbra's edge is exact; the penumbra's width is measured in the scaled
/// space (≤ 10 % off for Saturn's flattening).
pub(crate) fn shadow_core(p: Vec3, d: Vec3, sun_ang: f32) -> f32 {
    let r = p.length();
    if !(r > 1.0) {
        return 0.0;
    }
    let to_c = -p / r;
    let sep = to_c.cross(d).length().atan2(to_c.dot(d));
    crate::sun_visible_fraction(sun_ang, (1.0 / r).asin(), sep)
}

/// Fraction of the Sun visible from ring point `p_ring` (planet-centred, body axes [m]) past the
/// planet's ellipsoid of `radii` [m], for the unit direction `sun_dir`: 0 in the umbra, 1 in full
/// sunlight. Twin of the WGSL in `rings.wgsl` (through [`shadow_core`], after scaling by the
/// radii in f64). The Sun's angular radius is [`SUN_ANGULAR_RADIUS_AT_SATURN_RAD`]; the pass
/// uses the frame's own.
pub fn planet_shadow_on_ring(p_ring: DVec3, sun_dir: DVec3, radii: DVec3) -> f32 {
    if !(radii.min_element() > 0.0) {
        return 1.0;
    }
    let p = (p_ring / radii).as_vec3();
    let d = (sun_dir / radii).normalize_or_zero().as_vec3();
    shadow_core(p, d, SUN_ANGULAR_RADIUS_AT_SATURN_RAD)
}

/// One ringed body on the GPU: twin of `struct RingGpu` in `rings.wgsl` (eight vec4s, 128 B).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RingGpu {
    /// Planet centre − camera [m], root axes; w: the exposure blend weight (1 = scene), the
    /// planet's [`exposure_weight`].
    pub centre: [f32; 4],
    /// The body x axis (root axes); w: inner radius [m].
    pub axis_x: [f32; 4],
    /// The body y axis (root axes); w: outer radius [m].
    pub axis_y: [f32; 4],
    /// The ring normal, the body z axis (root axes); w: the equatorial radius a [m].
    pub normal: [f32; 4],
    /// Unit direction to the Sun from the planet (root axes); w: sunlight at the planet [lx].
    pub sun: [f32; 4],
    /// The Sun's direction in the unit-ellipsoid space (unit); w: its angular radius [rad].
    pub sun_scaled: [f32; 4],
    /// a / radii.x, a / radii.y; zw: 0.
    pub radii: [f32; 4],
    /// The Sun's colour, linear Rec. 709 with luminance 1; w: 0.
    pub sun_color: [f32; 4],
}

fn vec4(v: DVec3, w: f64) -> [f32; 4] {
    v.extend(w).as_vec4().to_array()
}

/// The ring records for this frame: one per body of `scene` with a ring view, radii
/// `inner_m`…`outer_m` from the profile. Bodies with a non-finite position, Sun direction or
/// radius are skipped. Nothing is kept per object.
pub(crate) fn plan_rings(scene: &BodyScene, inner_m: f64, outer_m: f64) -> Vec<RingGpu> {
    let sun_color = blackbody_linear_srgb(TEFF_SUN_K).as_dvec3();
    scene
        .bodies
        .iter()
        .filter(|b| b.ring.is_some())
        .filter_map(|b| {
            let a = b.radii_m.x;
            let d_sun = b.sun_rel_m.length();
            let ok = b.rel_pos_m.is_finite()
                && b.radii_m.min_element() > 0.0
                && b.radii_m.is_finite()
                && d_sun > 0.0
                && d_sun.is_finite();
            if !ok {
                return None;
            }
            let sun = b.sun_rel_m / d_sun;
            let body_sun = b.orient.conjugate() * sun;
            // The planet's exposure weight, snapped (lead ruling B6 I1).
            let mix = exposure_weight(b.proj_radius_px, scene.eye_adapted);
            Some(RingGpu {
                centre: vec4(b.rel_pos_m, f64::from(mix)),
                axis_x: vec4(b.orient * DVec3::X, inner_m),
                axis_y: vec4(b.orient * DVec3::Y, outer_m),
                normal: vec4(b.orient * DVec3::Z, a),
                sun: vec4(sun, b.sun_illuminance_lux),
                sun_scaled: vec4((body_sun / b.radii_m).normalize(), R_SUN_M / d_sun),
                radii: vec4(DVec3::new(a / b.radii_m.x, a / b.radii_m.y, 0.0), 0.0),
                sun_color: vec4(sun_color, 0.0),
            })
        })
        .collect()
}

/// The profile as a premultiplied mip chain: level 0 is (ρ·α, α) per sample (α clamped to
/// [0, 1]); each next level halves the width (at least 1), each texel the mean of its two source
/// texels (and the third, unpaired one at the end of an odd row), so a distant view averages the
/// radiance-weighted colour and the coverage, not the colour alone.
fn profile_mips(profile: &RingProfile) -> Vec<Vec<[f32; 4]>> {
    let level0: Vec<[f32; 4]> = profile
        .samples
        .iter()
        .map(|&[r, g, b, a]| {
            let a = a.clamp(0.0, 1.0);
            [r * a, g * a, b * a, a]
        })
        .collect();
    let mut mips = vec![level0];
    while let Some(prev) = mips.last().filter(|l| l.len() > 1) {
        let w = (prev.len() / 2).max(1);
        let next = (0..w)
            .map(|i| {
                let end = if i + 1 == w { prev.len() } else { 2 * i + 2 };
                let src = &prev[2 * i..end];
                let sum = src
                    .iter()
                    .fold(Vec4::ZERO, |acc, t| acc + Vec4::from_array(*t));
                (sum / src.len() as f32).to_array()
            })
            .collect();
        mips.push(next);
    }
    mips
}

/// `ShaderLib` path of the ring probes.
pub(crate) const RING_PROBE_WGSL_PATH: &str = "probe/planet_rings.wgsl";

/// Compute entry points for the ring probes. Input and output at group 2 (rings.wgsl owns
/// groups 0 and 1).
pub(crate) const RING_PROBE_WGSL: &str = r#"// probe/planet_rings.wgsl: `universe --selftest` probes of rings.wgsl (Task B2).
#import "planet/rings.wgsl"

@group(2) @binding(0) var<storage, read> probe_in: array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> probe_out: array<vec4<f32>>;

// in: (cos θ, lit 0/1, 0, 0); out.x: ring_phase.
@compute @workgroup_size(64, 1, 1)
fn cs_phase(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(ring_phase(c.x, c.y > 0.5), 0.0, 0.0, 0.0);
}

// in: (τ, μ₀, μ, lit 0/1); out.x: ring_layer_factor.
@compute @workgroup_size(64, 1, 1)
fn cs_layer(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= arrayLength(&probe_in)) {
        return;
    }
    let c = probe_in[gid.x];
    probe_out[gid.x] = vec4<f32>(ring_layer_factor(c.x, c.y, c.z, c.w > 0.5), 0.0, 0.0, 0.0);
}

// in[2i]: (p, sun angular radius), in[2i + 1]: (d, 0); out[i].x: planet_shadow_on_ring.
@compute @workgroup_size(64, 1, 1)
fn cs_shadow(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (2u * gid.x + 1u >= arrayLength(&probe_in)) {
        return;
    }
    let a = probe_in[2u * gid.x];
    let b = probe_in[2u * gid.x + 1u];
    probe_out[gid.x] = vec4<f32>(planet_shadow_on_ring(a.xyz, b.xyz, a.w), 0.0, 0.0, 0.0);
}
"#;

/// Largest |GPU − CPU| relative to max(1, |CPU|) for the phase function and the layer factor
/// (a handful of f32 operations; a wrong lobe or sign errs by ≥ 0.01).
const RING_TWIN_TOLERANCE: f32 = 1e-5;
/// Largest |GPU − CPU| of the planet's shadow: the penumbra's fraction moves 1 per ~1e-3 rad of
/// separation, so exact f32 differences of ~3e-8 rad in `atan2`/`asin` would give ≤ 3e-5. On the
/// M3 the GPU measured 2.8e-4 (B9 selftest, `universe --selftest` at 6cd81e9): Metal compiles
/// with fast math, whose `asin`/`atan2`/`acos` are a few ulp off the CPU's correctly rounded
/// ones, and the penumbra's steep slope amplifies that. Lead ruling "selftest tolerances" (Wave B
/// gate): 1e-3, about 3.5× the measured error (a wrong umbra or penumbra errs by ≥ 0.01).
const RING_SHADOW_TOLERANCE: f32 = 1e-3;

/// Phase-function probe inputs: 64 scattering-angle cosines over [−1, 1], both faces.
fn phase_probe_inputs() -> Vec<[f32; 4]> {
    (0..64)
        .map(|k| [-1.0 + 2.0 * (k / 2) as f32 / 31.0, (k % 2) as f32, 0.0, 0.0])
        .collect()
}

/// Layer-factor probe inputs: 64 (τ, μ₀, μ, face) including μ₀ = μ, μ₀ = 0, τ = 0 and opaque.
fn layer_probe_inputs() -> Vec<[f32; 4]> {
    let taus = [0.0, 0.01, 0.1, 0.5, 1.5, 9.2, 0.3, 2.0];
    let mus = [(0.15, 0.42), (0.4, 0.4), (0.0, 0.3), (0.9, 0.05)];
    let mut v = Vec::with_capacity(64);
    for tau in taus {
        for (mu0, mu) in mus {
            for lit in [0.0, 1.0] {
                v.push([tau, mu0, mu, lit]);
            }
        }
    }
    v
}

/// Shadow probe inputs: 64 ring points in the unit-ellipsoid space, the Sun 4.6° out of the ring
/// plane: at each of 8 radii (1.2–2.4) the azimuth where the CPU core gives ½ is found by
/// bisection, and 8 points sit around it, from deep umbra through both halves of the penumbra
/// to full sunlight. Pairs of (p, Sun angular radius), (d, 0).
fn shadow_probe_inputs() -> Vec<[f32; 4]> {
    let d = Vec3::new(1.0, 0.0, 0.08).normalize();
    let sun = SUN_ANGULAR_RADIUS_AT_SATURN_RAD;
    let at = |r: f32, phi: f32| Vec3::new(r * phi.cos(), r * phi.sin(), 0.0);
    let offsets = [-0.02, -6e-4, -3e-4, -1e-4, 1e-4, 3e-4, 6e-4, 0.02];
    let mut v = Vec::with_capacity(128);
    for k in 0..8 {
        let r = 1.2 + 1.2 * k as f32 / 7.0;
        // Lit at φ = 90°, umbra at φ = 180° (behind the planet); bisect the ½ point.
        let (mut lit, mut dark) = (std::f32::consts::FRAC_PI_2, std::f32::consts::PI);
        for _ in 0..40 {
            let mid = 0.5 * (lit + dark);
            if shadow_core(at(r, mid), d, sun) > 0.5 {
                lit = mid;
            } else {
                dark = mid;
            }
        }
        for off in offsets {
            let p = at(r, 0.5 * (lit + dark) + off);
            v.push([p.x, p.y, p.z, sun]);
            v.push([d.x, d.y, d.z, 0.0]);
        }
    }
    v
}

fn run_probe(
    gpu: &GpuContext,
    lib: &ShaderLib,
    entry: &str,
    input: &[[f32; 4]],
    cases: usize,
) -> anyhow::Result<Vec<[f32; 4]>> {
    let src = lib.compose(RING_PROBE_WGSL_PATH)?;
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
}

/// Worst error over pairs; NaN anywhere makes it NaN (which fails every check).
fn worst(errors: impl Iterator<Item = f32>) -> f32 {
    errors.fold(0.0_f32, |a, e| {
        if a.is_nan() || e.is_nan() {
            f32::NAN
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

/// B2's ring probes (`ring_phase`, `ring_layer_factor`, `planet_shadow_on_ring`), called from
/// [`crate::gpu_probes`] after it registered the crate's shaders.
pub(crate) fn ring_probes(gpu: &GpuContext, lib: &mut ShaderLib) -> Vec<ProbeResult> {
    lib.add(RING_PROBE_WGSL_PATH, RING_PROBE_WGSL);
    let lib = &*lib;
    let phase = || -> anyhow::Result<(bool, String)> {
        let input = phase_probe_inputs();
        let out = run_probe(gpu, lib, "cs_phase", &input, input.len())?;
        let err = worst(input.iter().zip(&out).map(|(c, g)| {
            let cpu = ring_phase(c[0], c[1] > 0.5);
            (g[0] - cpu).abs() / cpu.abs().max(1.0)
        }));
        Ok((
            err <= RING_TWIN_TOLERANCE,
            format!(
                "{} cases, max relative |GPU − CPU| {err:.1e} (tolerance {RING_TWIN_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let layer = || -> anyhow::Result<(bool, String)> {
        let input = layer_probe_inputs();
        let out = run_probe(gpu, lib, "cs_layer", &input, input.len())?;
        let err = worst(input.iter().zip(&out).map(|(c, g)| {
            let cpu = ring_layer_factor(c[0], c[1], c[2], c[3] > 0.5);
            (g[0] - cpu).abs() / cpu.abs().max(1.0)
        }));
        Ok((
            err <= RING_TWIN_TOLERANCE,
            format!(
                "{} cases, max relative |GPU − CPU| {err:.1e} (tolerance {RING_TWIN_TOLERANCE:.0e})",
                input.len()
            ),
        ))
    };
    let shadow = || -> anyhow::Result<(bool, String)> {
        let input = shadow_probe_inputs();
        let cases = input.len() / 2;
        let out = run_probe(gpu, lib, "cs_shadow", &input, cases)?;
        let err = worst(input.chunks(2).zip(&out).map(|(c, g)| {
            let cpu = shadow_core(
                Vec3::from_slice(&c[0][..3]),
                Vec3::from_slice(&c[1][..3]),
                c[0][3],
            );
            (g[0] - cpu).abs()
        }));
        Ok((
            err <= RING_SHADOW_TOLERANCE,
            format!(
                "{cases} ring points, max |GPU − CPU| {err:.1e} (tolerance {RING_SHADOW_TOLERANCE:.0e})"
            ),
        ))
    };
    vec![
        probe_result("ring_phase", phase()),
        probe_result("ring_layer_factor", layer()),
        probe_result("planet_shadow_on_ring", shadow()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;
    use uni_physics::photometry::Brdf;
    use uni_sim::snapshot::{BodyView, RingView};

    /// Saturn's equatorial and polar radii [m] (IAU WGCCRE 2015 report, Archinal et al. 2018):
    /// 60 268 km and 54 364 km.
    const SATURN_RADII: DVec3 = DVec3::new(60_268e3, 60_268e3, 54_364e3);

    #[test]
    fn ring_phase_normalised() {
        let n = 200_000;
        for lit in [true, false] {
            let mut sum = 0.0_f64;
            for k in 0..n {
                let x = -1.0 + 2.0 * (f64::from(k) + 0.5) / f64::from(n);
                sum += f64::from(ring_phase(x as f32, lit));
            }
            // ∫ P dΩ / 4π = ½ ∫₋₁¹ P(cos θ) d(cos θ).
            let mean = 0.5 * sum * 2.0 / f64::from(n);
            assert!((mean - 1.0).abs() <= 1e-3, "lit {lit}: {mean}");
        }
        // The lit face sends most light back toward the Sun (≈ 60 % into the back hemisphere;
        // its narrow forward lobe still peaks higher at exactly θ = 0), the unlit face forward.
        let hemisphere = |lit: bool, back: bool| {
            (0..n)
                .map(|k| {
                    let x = (f64::from(k) + 0.5) / f64::from(n);
                    f64::from(ring_phase(if back { -x } else { x } as f32, lit))
                })
                .sum::<f64>()
                / f64::from(n)
        };
        let back_fraction =
            hemisphere(true, true) / (hemisphere(true, true) + hemisphere(true, false));
        assert!((back_fraction - 0.596).abs() < 0.005, "{back_fraction}");
        assert!(hemisphere(false, false) > 6.0 * hemisphere(false, true));
        assert!(ring_phase(-1.0, true) > 10.0 * ring_phase(-1.0, false));
        // The plan's night-side arithmetic: HG(0.6, 30°) = 3.52.
        let hg30 = ring_phase(30_f32.to_radians().cos(), false);
        assert!((hg30 - 3.52).abs() < 0.01, "{hg30}");
    }

    #[test]
    fn planet_shadow_on_ring_cases() {
        let sun = DVec3::X;
        let a = SATURN_RADII.x;
        // Directly anti-sun behind Saturn: umbra.
        assert_eq!(
            planet_shadow_on_ring(DVec3::new(-2.0 * a, 0.0, 0.0), sun, SATURN_RADII),
            0.0
        );
        // 90° from the anti-sun line, and on the sunward side: full sunlight.
        assert_eq!(
            planet_shadow_on_ring(DVec3::new(0.0, 2.0 * a, 0.0), sun, SATURN_RADII),
            1.0
        );
        assert_eq!(
            planet_shadow_on_ring(DVec3::new(2.0 * a, 0.0, 0.0), sun, SATURN_RADII),
            1.0
        );
        // Across the shadow's edge at 2 a behind the planet: monotonic, with a penumbra.
        let mut prev = 0.0_f32;
        let mut partial = 0;
        for k in 0..=2000 {
            let y = a * (0.99 + 0.02 * f64::from(k) / 2000.0);
            let v = planet_shadow_on_ring(DVec3::new(-2.0 * a, y, 0.0), sun, SATURN_RADII);
            assert!(
                (0.0..=1.0).contains(&v) && v >= prev - 1e-6,
                "y/a {}: {v}",
                y / a
            );
            if v > 0.0 && v < 1.0 {
                partial += 1;
            }
            prev = v;
        }
        assert!(partial > 0, "no penumbra");
        // The oblate planet: with the Sun along the ring plane's normal tilt, the shadow edge in
        // z sits at the polar radius, not the equatorial one.
        let up = planet_shadow_on_ring(
            DVec3::new(-3.0 * a, 0.0, 0.95 * SATURN_RADII.z),
            sun,
            SATURN_RADII,
        );
        let up_out = planet_shadow_on_ring(
            DVec3::new(-3.0 * a, 0.0, 1.05 * SATURN_RADII.z),
            sun,
            SATURN_RADII,
        );
        assert_eq!((up, up_out), (0.0, 1.0));
    }

    #[test]
    fn ring_layer_factor_limits() {
        // An opaque layer seen from the Sun's own direction reflects the brief's ρ·P (factor 1).
        assert!((ring_layer_factor(50.0, 0.5, 0.5, true) - 1.0).abs() < 1e-6);
        // Sun in the ring plane (equinox): no light on either face.
        assert_eq!(ring_layer_factor(0.5, 0.0, 0.4, true), 0.0);
        assert_eq!(ring_layer_factor(0.5, 0.0, 0.4, false), 0.0);
        // Nothing there: nothing scattered.
        assert_eq!(ring_layer_factor(0.0, 0.3, 0.4, true), 0.0);
        assert_eq!(ring_layer_factor(0.0, 0.3, 0.4, false), 0.0);
        // An opaque layer transmits nothing to its unlit face.
        assert!(ring_layer_factor(50.0, 0.3, 0.4, false) < 1e-12);
        // The unlit face is brightest at intermediate optical depth.
        let t = |tau| ring_layer_factor(tau, 0.15, 0.42, false);
        assert!(t(0.5) > t(0.05) && t(0.5) > t(3.0));
        // Continuous through μ = μ₀ on the unlit face (the removable singularity).
        let at = ring_layer_factor(0.7, 0.4, 0.4, false);
        let near = ring_layer_factor(0.7, 0.4, 0.4 + 1e-3, false);
        assert!((at - near).abs() < 2e-3, "{at} vs {near}");
        // A thin layer brightens as it is seen more obliquely (longer path).
        assert!(ring_layer_factor(0.05, 0.5, 0.1, true) > ring_layer_factor(0.05, 0.5, 0.9, true));
    }

    fn saturn_view(rel: DVec3, sun_rel: DVec3) -> BodyView {
        let orient = DQuat::from_rotation_x(0.3);
        BodyView {
            id: 5,
            naif: Some(699),
            rel_pos_m: rel,
            orient,
            radii_m: SATURN_RADII,
            proj_radius_px: 2.5,
            sun_rel_m: sun_rel,
            sun_illuminance_lux: 1500.0,
            occluders: Vec::new(),
            ring: Some(RingView {
                inner_m: 7.0e7,
                outer_m: 1.4e8,
                normal: orient * DVec3::Z,
            }),
            has_atmosphere: true,
            brdf: Brdf::Lambert,
            geometric_albedo: 0.499,
            mean_radius_m: 58_232e3,
        }
    }

    #[test]
    fn ring_plan_from_the_view() {
        let sun_rel = DVec3::new(-1.4e12, 2.0e11, 1.0e11);
        let mut scene = BodyScene {
            bodies: vec![saturn_view(DVec3::new(1.0e9, 2.0e9, -3.0e8), sun_rel)],
            eye_adapted: true,
            ..Default::default()
        };
        let mut no_ring = saturn_view(DVec3::ONE, sun_rel);
        no_ring.ring = None;
        scene.bodies.push(no_ring);
        let rings = plan_rings(&scene, 69_933_828.0, 140_972_084.0);
        assert_eq!(rings.len(), 1);
        let r = &rings[0];
        let v = &scene.bodies[0];
        // Radii from the profile, normal from the view, exposure blend at 2.5 px.
        assert_eq!((r.axis_x[3], r.axis_y[3]), (69_933_828.0, 140_972_084.0));
        let normal = Vec3::from_slice(&r.normal[..3]).as_dvec3();
        assert!((normal - v.ring.unwrap().normal).length() < 1e-6);
        assert_eq!(r.normal[3], SATURN_RADII.x as f32);
        assert!((r.centre[3] - crate::lod_exposure_mix(2.5)).abs() < 1e-6);
        let sun = Vec3::from_slice(&r.sun[..3]).as_dvec3();
        assert!((sun - sun_rel.normalize()).length() < 1e-6);
        assert_eq!(r.sun[3], 1500.0);
        // The scaled Sun direction is the body-axes direction divided by the radii.
        let body = v.orient.conjugate() * sun_rel.normalize();
        let scaled = (body / SATURN_RADII).normalize();
        assert!((Vec3::from_slice(&r.sun_scaled[..3]).as_dvec3() - scaled).length() < 1e-6);
        let ang = uni_core::R_SUN_M / sun_rel.length();
        assert!((f64::from(r.sun_scaled[3]) / ang - 1.0).abs() < 1e-6);
        // Without eye-adapted stars the rings use the scene exposure.
        scene.eye_adapted = false;
        assert_eq!(plan_rings(&scene, 7.0e7, 1.4e8)[0].centre[3], 1.0);
        assert!(
            bytemuck::cast_slice::<RingGpu, f32>(&rings)
                .iter()
                .all(|x| x.is_finite())
        );
    }

    /// Lead ruling B6 I1 (task B10): the rings take their planet's snapped exposure weight
    /// ([`exposure_weight`]), so the share 1 − w they store in `lens_mask.a` is 0 or at least
    /// 2⁻¹⁴ (an f16 normal), and just below 4 px (3.995 px, where 1 − w < 2⁻¹⁴) they draw at
    /// scene exposure and store no share.
    #[test]
    fn ring_exposure_weight_is_the_planets_snapped() {
        use crate::body::{STAR_SHARE_MIN, star_share};
        let sun_rel = DVec3::new(-1.4e12, 2.0e11, 1.0e11);
        let mut scene = BodyScene {
            bodies: vec![saturn_view(DVec3::new(1.0e9, 2.0e9, -3.0e8), sun_rel)],
            eye_adapted: true,
            ..Default::default()
        };
        for r in [1.0, 1.5, 2.5, 3.95, 3.99, 3.995, 4.0, 50.0] {
            scene.bodies[0].proj_radius_px = r;
            let w = plan_rings(&scene, 7.0e7, 1.4e8)[0].centre[3];
            assert_eq!(w, exposure_weight(r, true), "{r} px");
            let s = star_share(w, 0.5);
            assert!(s == 0.0 || s >= STAR_SHARE_MIN, "{r} px: share {s}");
        }
        scene.bodies[0].proj_radius_px = 3.995;
        assert!(crate::lod_exposure_mix(3.995) < 1.0);
        assert_eq!(plan_rings(&scene, 7.0e7, 1.4e8)[0].centre[3], 1.0);
        scene.bodies[0].proj_radius_px = 2.0;
        let w = plan_rings(&scene, 7.0e7, 1.4e8)[0].centre[3];
        assert!(w > 0.0 && w < 1.0 && star_share(w, 0.5) == 1.0 - w);
    }

    /// The WGSL twin's constants equal the Rust ones (read from naga's module).
    #[test]
    fn ring_wgsl_constants_match_rust() {
        use uni_gfx::{ShaderLib, validate_wgsl};
        use wgpu::naga;
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(crate::RINGS_WGSL).unwrap()).unwrap();
        let scalar = |name: &str| {
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
        };
        assert_eq!(scalar("RING_SEGMENTS"), f64::from(RING_SEGMENTS));
        assert_eq!(scalar("RING_LIT_W"), f64::from(LIT_W));
        assert_eq!(scalar("RING_LIT_G1"), f64::from(LIT_G1));
        assert_eq!(scalar("RING_LIT_G2"), f64::from(LIT_G2));
        assert_eq!(scalar("RING_UNLIT_G"), f64::from(UNLIT_G));
        assert_eq!(scalar("RING_MU_MIN"), f64::from(MU_MIN));
        assert_eq!(scalar("RING_ALPHA_MAX"), f64::from(ALPHA_MAX));
        assert_eq!(scalar("RING_ALPHA_DRAWN_MIN"), f64::from(ALPHA_DRAWN_MIN));
    }

    /// Review I2: the single-scattering layer (as built) against the brief's opacity factors,
    /// with the committed profile, at the plan's two ring bookmarks and at equinox. Prints the
    /// per-region I/F that B6/B9 need to re-derive the `saturn-*` exposure rows.
    #[test]
    fn ring_photometry_with_the_real_profile() {
        let path = uni_data::data_dir().join("baked/saturn_rings.ron");
        let profile = uni_data::load_ring_profile(&path).expect("the committed saturn_rings.ron");
        // Region radii [km]: NASA NSSDCA Saturnian Rings Fact Sheet (D. R. Williams, last
        // updated 19 April 2022): C 74 658–91 975, B 91 975–117 507, A 122 340–136 780; the
        // Cassini division is the gap between the B ring's outer and the A ring's inner edge.
        let regions = [
            ("C ring", 74_658.0, 91_975.0),
            ("B ring", 91_975.0, 117_507.0),
            ("Cassini division", 117_507.0, 122_340.0),
            ("A ring", 122_340.0, 136_780.0),
        ];
        // Rec. 709 luminance of the profile's linear colour (the sunlight's colour has luminance 1).
        let lum = |s: &[f32; 4]| 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2];
        // The brief's form: lit ρ·α·P, unlit ρ·α(1 − α)·HG(+0.6), with no elevation dependence.
        let brief = |s: &[f32; 4], cos_scatter: f32, lit: bool| {
            let a = s[3];
            lum(s) * ring_phase(cos_scatter, lit) * if lit { a } else { a * (1.0 - a) }
        };
        // The plan's bookmark table (M1 plan, "M1 capture bookmarks", rows 8 and 10): the Sun
        // 8.5° south of the ring plane on 2026-11-20 (B′); `saturn-ring-plane` 4° above the lit
        // face at phase < 10° (scattering angle 175°); `saturn-night-side` 25° above the unlit
        // face at phase 150° (scattering angle 30°); equinox: the Sun 0.06° from the plane.
        let mu0 = 8.5_f32.to_radians().sin();
        let mu_plane = 4.0_f32.to_radians().sin();
        let mu_night = 25.0_f32.to_radians().sin();
        let back = 175.0_f32.to_radians().cos();
        let fwd = 30.0_f32.to_radians().cos();
        let n = profile.samples.len();
        let dr_km = (profile.outer_m - profile.inner_m) / 1e3 / n as f64;
        // Per region, the mean over its samples of: α, ρ (luminance), and I/F at ring-plane
        // (layer, brief), night side (layer, brief), equinox (layer).
        let mut rows = Vec::new();
        for (name, r0, r1) in regions {
            let mut sum = [0.0_f64; 7];
            let mut count = 0.0;
            for (i, s) in profile.samples.iter().enumerate() {
                let r = profile.inner_m / 1e3 + (i as f64 + 0.5) * dr_km;
                if !(r0..r1).contains(&r) {
                    continue;
                }
                let drawn = s[3] > ALPHA_DRAWN_MIN;
                let values = [
                    s[3],
                    lum(s),
                    ring_i_over_f(lum(s), s[3], mu0, mu_plane, back, true),
                    if drawn { brief(s, back, true) } else { 0.0 },
                    ring_i_over_f(lum(s), s[3], mu0, mu_night, fwd, false),
                    if drawn { brief(s, fwd, false) } else { 0.0 },
                    ring_i_over_f(lum(s), s[3], 1e-3, mu_plane, back, true),
                ];
                for (acc, v) in sum.iter_mut().zip(values) {
                    *acc += f64::from(v);
                }
                count += 1.0;
            }
            assert!(count >= 10.0, "{name}: {count} samples");
            let mean = sum.map(|v| v / count);
            eprintln!(
                "{name:>16}: α {:.3}, ρ {:.3} | ring-plane I/F layer {:.4}, brief {:.4} | \
                 night-side I/F layer {:.4}, brief {:.4} | equinox lit layer {:.5}",
                mean[0], mean[1], mean[2], mean[3], mean[4], mean[5], mean[6]
            );
            rows.push(mean);
        }
        let [c, b, cd, a] = [&rows[0], &rows[1], &rows[2], &rows[3]];
        // The plan's night-side premise ("unlit face ≈ 0.35 of E/π") holds under neither form.
        for row in &rows {
            assert!(row[4] < 0.1 && row[5] < 0.1, "{row:?}");
        }
        // The layer keeps the opaque B ring dark on its unlit face, below every other region;
        // the brief's α(1 − α) makes it brighter than the C ring there.
        assert!(b[4] < 0.25 * c[4].min(cd[4]) && b[4] < a[4], "{b:?}");
        assert!(b[5] > c[5], "{b:?} {c:?}");
        // Lit face at saturn-ring-plane: the B ring is the brightest region, and B9's HDR check
        // `ring_cassini / ring_b ≤ 0.5` holds on the region means.
        assert!(b[2] > a[2] && a[2] > cd[2] && a[2] > c[2]);
        assert!(cd[2] / b[2] <= 0.5, "{}", cd[2] / b[2]);
        // Equinox: the layer darkens the B ring's lit face ~48× (the brief's form would not
        // change at all, b[3]); the C ring, ~48× too.
        assert!(b[6] < 0.03 * b[2] && c[6] < 0.03 * c[2], "{b:?} {c:?}");
    }

    #[test]
    fn ring_i_over_f_composes_the_pass() {
        // An opaque ring seen from the Sun's direction reflects ρ·P (the brief's form at α = 1).
        let p = ring_phase(-1.0, true);
        assert!((ring_i_over_f(0.4, 1.0, 0.3, 0.3, -1.0, true) - 0.4 * p).abs() < 1e-5);
        // α is clamped as in the pass, and what the pass discards is 0.
        let tau_max = -(1.0 - ALPHA_MAX).ln();
        assert_eq!(
            ring_i_over_f(0.4, 1.0, 0.2, 0.5, 0.3, false),
            0.4 * ring_phase(0.3, false) * ring_layer_factor(tau_max, 0.2, 0.5, false)
        );
        assert_eq!(ring_i_over_f(0.4, 1e-5, 0.2, 0.5, 0.3, true), 0.0);
        assert_eq!(ring_i_over_f(0.4, f32::NAN, 0.2, 0.5, 0.3, true), 0.0);
        // The camera in the ring plane does not divide by zero.
        assert!(ring_i_over_f(0.4, 0.5, 0.2, 0.0, 0.3, true).is_finite());
    }

    /// The probe inputs exercise every regime on the CPU: umbra, partial penumbra, sunlight.
    #[test]
    fn ring_probe_inputs_cover_the_cases() {
        let shadow: Vec<f32> = shadow_probe_inputs()
            .chunks(2)
            .map(|c| {
                shadow_core(
                    Vec3::from_slice(&c[0][..3]),
                    Vec3::from_slice(&c[1][..3]),
                    c[0][3],
                )
            })
            .collect();
        assert_eq!(shadow.len(), 64);
        assert!(shadow.iter().any(|&v| v == 0.0));
        assert!(shadow.iter().any(|&v| v == 1.0));
        assert!(
            shadow.iter().filter(|&&v| v > 0.0 && v < 1.0).count() >= 32,
            "{shadow:?}"
        );
        assert_eq!(phase_probe_inputs().len(), 64);
        let layer = layer_probe_inputs();
        assert_eq!(layer.len(), 64);
        assert!(layer.iter().any(|c| c[1] == c[2] && c[3] == 0.0));
    }

    #[test]
    fn profile_mips_are_premultiplied_box_filtered() {
        let profile = RingProfile {
            inner_m: 1.0,
            outer_m: 2.0,
            samples: vec![
                [0.5, 0.4, 0.2, 1.0],
                [1.0, 1.0, 1.0, 0.0],
                [0.2, 0.2, 0.2, 0.5],
            ],
        };
        let mips = profile_mips(&profile);
        assert_eq!(mips.iter().map(Vec::len).collect::<Vec<_>>(), [3, 1]);
        // Level 0 is premultiplied; a transparent sample contributes no colour.
        assert_eq!(mips[0][0], [0.5, 0.4, 0.2, 1.0]);
        assert_eq!(mips[0][1], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(mips[0][2], [0.1, 0.1, 0.1, 0.5]);
        // The 1-texel level averages all three.
        let top = mips[1][0];
        assert!(
            (top[3] - 0.5).abs() < 1e-6 && (top[0] - 0.2).abs() < 1e-6,
            "{top:?}"
        );
    }
}
