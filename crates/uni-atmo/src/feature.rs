//! The atmosphere render feature, pass "atmosphere" (Task B3).
//!
//! Pass 8 of the M1 frame contract: after the opaque passes, before the meter. Per pixel it
//! intersects each atmosphere's shell, clips by the scene depth and the planet's ground, and
//! marches the inscattered light (world light × `scene_pre()`), writing `lens_mask.g =
//! 1 − luminance(T)` with `Max`. Two variants (review I7): dual-source blending (two pipelines,
//! `atmosphere_dual.wgsl` + `atmosphere_mask.wgsl`) or grey transmittance (one pipeline,
//! `atmosphere_grey.wgsl`).
//!
//! **Recorded deviation (frame contract):** a per-pixel march (32 steps, up to 48 on long limb
//! paths) through Hillaire's transmittance and multiple-scattering LUTs, not his sky-view and
//! aerial-perspective LUTs (design §5.1); those are the M2 upgrade. Ladder if over budget
//! (3.0 ms): `g` from the LUT ratio, fewer steps, half resolution.
//!
//! **The shell follows the body's ellipsoid.** Each draw maps root axes into the body-fixed frame
//! (`AtmoDraw::orient`) and scales it by R/(a, b, c), so the body's reference ellipsoid (WGCCRE
//! 2015 radii, [`body_shape_m`]) becomes the sphere of its mean radius R, which is the
//! atmosphere's bottom (Venus: R plus the cloud-top height). Heights are then measured from the
//! real ground at every latitude (Earth's ground is 21 km closer to the centre at the poles
//! than at the equator; a sphere would make the polar sky 2.75× too thick). Distances along a
//! ray stay true; angles are distorted by at most the flattening (0.6 % for Mars).

use glam::{DMat3, DQuat, DVec3, Mat4, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, HDR_FORMAT, LENS_MASK_FORMAT, PassTimer,
    RenderFeature, ShaderLib, SharedInput,
};

use crate::lut::{AtmosphereSet, March, cpu_luts, march, ray_sphere};
use crate::params::AtmosphereParams;

/// The atmospheres to draw this frame (set by the app before `Renderer::render`).
#[derive(Clone, Debug, Default)]
pub struct AtmoScene {
    pub atmos: Vec<AtmoDraw>,
}

/// One atmosphere to draw.
#[derive(Clone, Debug)]
pub struct AtmoDraw {
    /// NAIF id of the body; selects its preset and LUTs.
    pub naif: i32,
    /// Body centre − camera [m], root (ICRS) axes.
    pub rel_pos_m: DVec3,
    /// Body-fixed → ICRS rotation.
    pub orient: DQuat,
    /// Sun − body [m], root axes.
    pub sun_rel_m: DVec3,
    /// Solar illuminance at the body [lx].
    pub sun_illuminance_lux: f64,
}

/// Most atmospheres drawn per frame (M1 has four); extra draws are skipped.
pub const MAX_DRAWS: usize = 8;

/// Triaxial radii a ≥ b ≥ c [m] of the bodies with atmospheres: WGCCRE 2015 (Archinal et al.
/// 2018, Celest. Mech. Dyn. Astr. 130:22) as in NAIF `pck00011.tpc`, the values uni-physics's
/// registry holds for the bodies pass. `None` for other bodies (their shells stay spheres).
pub fn body_shape_m(naif: i32) -> Option<DVec3> {
    match naif {
        299 => Some(DVec3::splat(6_051.8e3)),
        399 => Some(DVec3::new(6_378.1366e3, 6_378.1366e3, 6_356.7519e3)),
        499 => Some(DVec3::new(3_396.19e3, 3_396.19e3, 3_376.20e3)),
        606 => Some(DVec3::new(2_575.15e3, 2_574.78e3, 2_574.47e3)),
        _ => None,
    }
}

/// Twin of `struct AtmoParamsGpu` in `atmosphere_common.wgsl` (nine vec4, 144 B;
/// `atmo_uniforms_layout_match_wgsl`). Coefficients [1/m] at (680, 550, 440) nm.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmoParamsGpu {
    /// bottom radius, top radius, Rayleigh scale height, Mie scale height [m].
    pub radii: [f32; 4],
    /// xyz Rayleigh scattering; w layer-2 tent centre [m].
    pub rayleigh_scattering: [f32; 4],
    /// xyz Mie scattering; w layer-2 tent full width [m].
    pub mie_scattering: [f32; 4],
    /// xyz Mie extinction; w 0.
    pub mie_extinction: [f32; 4],
    /// xyz Mie Cornette–Shanks g; w 0.
    pub mie_g: [f32; 4],
    /// xyz layer-2 scattering at the peak; w 0.
    pub layer2_scattering: [f32; 4],
    /// xyz layer-2 extinction at the peak; w 0.
    pub layer2_extinction: [f32; 4],
    /// xyz layer-2 Cornette–Shanks g; w 0.
    pub layer2_g: [f32; 4],
    /// xyz ground albedo; w 0.
    pub ground_albedo: [f32; 4],
}

impl AtmoParamsGpu {
    /// Packs `p` (radii and heights as f32: exact to 0.5 m at Earth's radius).
    pub fn from_params(p: &AtmosphereParams) -> Self {
        let v = |v: Vec3, w: f32| v.extend(w).to_array();
        Self {
            radii: [
                p.bottom_radius_m as f32,
                p.top_radius_m as f32,
                p.rayleigh_scale_height_m,
                p.mie_scale_height_m,
            ],
            rayleigh_scattering: v(p.rayleigh_scattering, p.layer2_center_m),
            mie_scattering: v(p.mie_scattering, p.layer2_width_m),
            mie_extinction: v(p.mie_extinction, 0.0),
            mie_g: v(p.mie_g, 0.0),
            layer2_scattering: v(p.layer2_scattering, 0.0),
            layer2_extinction: v(p.layer2_extinction, 0.0),
            layer2_g: v(p.layer2_g, 0.0),
            ground_albedo: v(p.ground_albedo, 0.0),
        }
    }

    #[cfg(test)]
    fn field_offsets() -> Vec<(String, usize)> {
        use core::mem::offset_of;
        [
            ("radii", offset_of!(Self, radii)),
            ("rayleigh_scattering", offset_of!(Self, rayleigh_scattering)),
            ("mie_scattering", offset_of!(Self, mie_scattering)),
            ("mie_extinction", offset_of!(Self, mie_extinction)),
            ("mie_g", offset_of!(Self, mie_g)),
            ("layer2_scattering", offset_of!(Self, layer2_scattering)),
            ("layer2_extinction", offset_of!(Self, layer2_extinction)),
            ("layer2_g", offset_of!(Self, layer2_g)),
            ("ground_albedo", offset_of!(Self, ground_albedo)),
        ]
        .map(|(n, o)| (n.to_owned(), o))
        .to_vec()
    }
}

/// Twin of `struct AtmoDrawGpu` in `atmosphere_common.wgsl` (seven vec4, 112 B), one per draw at
/// a 256-byte-aligned dynamic offset.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct AtmoDrawGpu {
    /// Columns x, y, z of the root → scaled body frame matrix (w 0).
    pub to_body_x: [f32; 4],
    pub to_body_y: [f32; 4],
    pub to_body_z: [f32; 4],
    /// xyz camera − body centre in the scaled body frame [m]; w the camera's height above the
    /// bottom [m].
    pub origin: [f32; 4],
    /// xyz unit direction to the Sun in the scaled body frame; w sun illuminance [lx].
    pub sun: [f32; 4],
    /// NDC x0, y0, x1, y1 of the screen quad.
    pub rect: [f32; 4],
    /// x exposure mix (1 = `scene_pre`); y 1 = fullscreen triangle; z the own-ground depth
    /// tolerance [m]; w 0.
    pub misc: [f32; 4],
}

impl AtmoDrawGpu {
    #[cfg(test)]
    fn field_offsets() -> Vec<(String, usize)> {
        use core::mem::offset_of;
        [
            ("to_body_x", offset_of!(Self, to_body_x)),
            ("to_body_y", offset_of!(Self, to_body_y)),
            ("to_body_z", offset_of!(Self, to_body_z)),
            ("origin", offset_of!(Self, origin)),
            ("sun", offset_of!(Self, sun)),
            ("rect", offset_of!(Self, rect)),
            ("misc", offset_of!(Self, misc)),
        ]
        .map(|(n, o)| (n.to_owned(), o))
        .to_vec()
    }
}

/// The camera position the march starts from: `ro` (camera − body centre), lifted to 1 m above
/// the bottom when it lies within 1 m of it (f32 cannot resolve less at planetary radii);
/// `None` when the camera is more than 1 m below the bottom (inside the body, or under Venus's
/// cloud deck: the pass draws nothing for that atmosphere) or `ro` is not finite.
pub(crate) fn camera_origin(params: &AtmosphereParams, ro: DVec3) -> Option<DVec3> {
    let r = ro.length();
    let b = params.bottom_radius_m;
    if !r.is_finite() || r < b - 1.0 {
        return None;
    }
    if r < b + 1.0 {
        let up = if r > 0.0 { ro / r } else { DVec3::Z };
        return Some(up * (b + 1.0));
    }
    Some(ro)
}

/// The march bounds of a ray from `ro` (already through [`camera_origin`]) along the unit `rd`:
/// from the camera or the shell entry to the shell exit, the ground or `t_max`, whichever comes
/// first. `None` when the segment is empty. Twin of `atmo_view_ray` with the depth given as a
/// distance.
pub(crate) fn segment_bounds(
    p: &AtmosphereParams,
    ro: DVec3,
    rd: DVec3,
    t_max: f64,
) -> Option<(f64, f64)> {
    let (near, far) = ray_sphere(ro, rd, p.top_radius_m)?;
    if far <= 0.0 {
        return None;
    }
    let t0 = near.max(0.0);
    let mut t1 = far.min(t_max);
    if let Some((g0, g1)) = ray_sphere(ro, rd, p.bottom_radius_m)
        && g1 > 0.0
    {
        if g0 > 0.0 {
            t1 = t1.min(g0);
        } else if ro.dot(rd) < 0.0 {
            t1 = t0;
        }
    }
    (t1 > t0).then_some((t0, t1))
}

/// The light a ray picks up through the atmosphere: from `ro_m` (camera − body centre [m]) along
/// the unit direction `rd` for at most `t_max_m` metres (the scene depth), with the Sun in the
/// unit direction `sun_dir` (all in the same axes; a sphere of the preset's radii, without the
/// ellipsoid scaling the pass applies). Returns (inscattered radiance per lux of sun illuminance
/// [cd/m² per lx], transmittance along the segment), per channel. CPU twin of the view march in
/// `atmosphere_common.wgsl` (B3). Non-finite input, an empty segment or a camera inside the body
/// gives (0, 1).
pub fn segment_radiance(
    params: &AtmosphereParams,
    ro_m: DVec3,
    rd: DVec3,
    t_max_m: f64,
    sun_dir: DVec3,
) -> (Vec3, Vec3) {
    let empty = (Vec3::ZERO, Vec3::ONE);
    let (rd, sun) = (rd.normalize_or_zero(), sun_dir.normalize_or_zero());
    if rd == DVec3::ZERO || sun == DVec3::ZERO || t_max_m.is_nan() {
        return empty;
    }
    let Some(ro) = camera_origin(params, ro_m) else {
        return empty;
    };
    let Some((t0, t1)) = segment_bounds(params, ro, rd, t_max_m) else {
        return empty;
    };
    let luts = cpu_luts(params);
    // Re-centred at the entry point, as the shader does.
    let m: March = march(&luts, ro + rd * t0, rd, 0.0, t1 - t0, sun);
    (m.inscatter.as_vec3(), m.transmittance.as_vec3())
}

/// The root → scaled-body-frame matrix of a body: R/(a, b, c) × (body-fixed ← ICRS), R = (abc)^⅓.
fn to_body_frame(naif: i32, orient: DQuat) -> DMat3 {
    let to_fixed = DMat3::from_quat(orient.normalize()).transpose();
    match body_shape_m(naif) {
        Some(abc) => {
            let mean = (abc.x * abc.y * abc.z).cbrt();
            DMat3::from_diagonal(DVec3::splat(mean) / abc) * to_fixed
        }
        None => to_fixed,
    }
}

/// The screen rectangle [NDC x0, y0, x1, y1] bounding a sphere of radius `radius` centred at
/// `center_view` (view space: camera at the origin looking down −Z), under a projection with
/// x_ndc = sx·x/(−z), y_ndc = sy·y/(−z). Exact per axis: the planes through the camera tangent
/// to the sphere (Mara & McGuire 2013, "2D Polyhedral Bounds of a Clipped, Perspective-Projected
/// 3D Sphere", JCGT 2(2)). `None` when the sphere reaches the camera plane (z ≥ −radius): draw
/// the full screen then.
pub(crate) fn sphere_screen_rect(
    center_view: DVec3,
    radius: f64,
    sx: f64,
    sy: f64,
) -> Option<[f64; 4]> {
    let depth = -center_view.z;
    if !(depth > radius) || !center_view.is_finite() {
        return None;
    }
    let bounds = |a: f64, scale: f64| {
        let l = (a * a + depth * depth).sqrt();
        let (centre, half) = (a.atan2(depth), (radius / l).asin());
        (scale * (centre - half).tan(), scale * (centre + half).tan())
    };
    let (x0, x1) = bounds(center_view.x, sx);
    let (y0, y1) = bounds(center_view.y, sy);
    Some([x0, y0, x1, y1])
}

/// The GPU data of one draw for this frame, or `None` when it draws nothing (camera inside the
/// body, shell off screen, bad input).
pub(crate) fn draw_uniform(
    params: &AtmosphereParams,
    draw: &AtmoDraw,
    view_rot: Mat4,
    proj: Mat4,
) -> Option<AtmoDrawGpu> {
    let finite = draw.rel_pos_m.is_finite()
        && draw.orient.is_finite()
        && draw.sun_rel_m.is_finite()
        && draw.sun_illuminance_lux.is_finite()
        && draw.sun_illuminance_lux >= 0.0;
    let sun_dir = draw.sun_rel_m.normalize_or_zero();
    if !finite || sun_dir == DVec3::ZERO {
        return None;
    }
    let m = to_body_frame(draw.naif, draw.orient);
    let origin = camera_origin(params, m * -draw.rel_pos_m)?;
    let sun = (m * sun_dir).normalize();
    let height = origin.length() - params.bottom_radius_m;
    // The shell in root space lies within top × max(a, b, c)/R of the centre.
    let stretch = body_shape_m(draw.naif).map_or(1.0, |abc| {
        abc.max_element() / (abc.x * abc.y * abc.z).cbrt()
    });
    let bound = params.top_radius_m * stretch * 1.001;
    let center_view = view_rot.as_dmat4().transform_vector3(draw.rel_pos_m);
    if center_view.z - bound >= 0.0 {
        // Entirely behind the camera.
        return None;
    }
    let (sx, sy) = (f64::from(proj.x_axis.x), f64::from(proj.y_axis.y));
    let (rect, full) = match sphere_screen_rect(center_view, bound, sx, sy) {
        Some([x0, y0, x1, y1]) => {
            if x0 >= 1.0 || x1 <= -1.0 || y0 >= 1.0 || y1 <= -1.0 {
                return None;
            }
            ([x0.max(-1.0), y0.max(-1.0), x1.min(1.0), y1.min(1.0)], 0.0)
        }
        None => ([-1.0, -1.0, 1.0, 1.0], 1.0),
    };
    // The bodies pass draws the ground on a proxy mesh up to ~1e-3 × the camera's height
    // outside the true surface (B0's cap-mesh gap; 2e-3 for the 32 × 64 ladder step); a depth
    // surface below this is the body's own ground. Capped at 2 % of the radius, far below any
    // moon.
    let own_ground = (200.0 + 2e-3 * height.max(0.0)).min(0.02 * params.bottom_radius_m);
    let col = |c: DVec3| c.as_vec3().extend(0.0).to_array();
    Some(AtmoDrawGpu {
        to_body_x: col(m.x_axis),
        to_body_y: col(m.y_axis),
        to_body_z: col(m.z_axis),
        origin: origin.as_vec3().extend(height as f32).to_array(),
        sun: sun
            .as_vec3()
            .extend(draw.sun_illuminance_lux as f32)
            .to_array(),
        rect: rect.map(|v| v as f32),
        // Exposure mix 1 (scene_pre): AtmoDraw carries no LOD blend yet (see the task report).
        misc: [1.0, full, own_ground as f32, 0.0],
    })
}

/// The pipelines of one variant.
enum Pipelines {
    /// Dual-source colour pipeline into `hdr`, then the `lens_mask.g` pipeline.
    Dual {
        color: wgpu::RenderPipeline,
        mask: wgpu::RenderPipeline,
    },
    /// One pipeline into `hdr` (grey transmittance) and `lens_mask.g`.
    Grey(wgpu::RenderPipeline),
}

/// One draw recorded in `prepare`.
struct DrawCmd {
    /// Byte offset of its `AtmoDrawGpu` in the draw buffer.
    offset: u32,
    /// Index into `AtmosphereFeature::atmos`.
    atmo: usize,
    /// 3 (fullscreen triangle) or 6 (quad).
    vertices: u32,
}

/// The atmosphere pass (`RenderFeature` named "atmosphere").
pub struct AtmosphereFeature {
    input: SharedInput<AtmoScene>,
    /// (NAIF id, parameters, group 2) per atmosphere of the set.
    atmos: Vec<(i32, AtmosphereParams, wgpu::BindGroup)>,
    pipelines: Pipelines,
    draw_bgl: wgpu::BindGroupLayout,
    draw_buffer: wgpu::Buffer,
    /// Byte stride between draws in `draw_buffer` (the device's uniform offset alignment).
    stride: u32,
    /// Group 1 (draw uniform + depth) for the current targets; rebuilt on resize.
    draw_group: Option<wgpu::BindGroup>,
    draws: Vec<DrawCmd>,
}

impl AtmosphereFeature {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`; `set` holds the
    /// LUTs; `dual_source` is `GpuContext::has_dual_source_blending()` and picks the variant.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        set: &AtmosphereSet,
        input: SharedInput<AtmoScene>,
        dual_source: bool,
    ) -> anyhow::Result<Self> {
        crate::register_shaders(lib);
        let device = &gpu.device;
        let dual_source = dual_source && gpu.has_dual_source_blending();
        let modules = if dual_source {
            (
                Some(lib.module(device, crate::ATMOSPHERE_DUAL_WGSL)?),
                Some(lib.module(device, crate::ATMOSPHERE_MASK_WGSL)?),
                None,
            )
        } else {
            (
                None,
                None,
                Some(lib.module(device, crate::ATMOSPHERE_GREY_WGSL)?),
            )
        };
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let align = device.limits().min_uniform_buffer_offset_alignment.max(1);
        let stride = (size_of::<AtmoDrawGpu>() as u32).div_ceil(align) * align;
        let draw_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("atmo draws"),
            size: u64::from(stride) * MAX_DRAWS as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let draw_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("atmo draw bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: wgpu::BufferSize::new(size_of::<AtmoDrawGpu>() as u64),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("atmo layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&draw_bgl), Some(set.view_bgl())],
            immediate_size: 0,
        });
        let pipeline = |label: &str,
                        module: &wgpu::ShaderModule,
                        fs: &str,
                        targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module,
                    entry_point: Some("vs_atmosphere"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: wgpu::PrimitiveState::default(),
                // No depth attachment: the pass reads the depth target as a texture.
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        // lens_mask.g with Max (WebGPU requires factor One for Min/Max).
        let mask_target = Some(wgpu::ColorTargetState {
            format: LENS_MASK_FORMAT,
            blend: Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Max,
                },
            }),
            write_mask: wgpu::ColorWrites::GREEN,
        });
        // hdr's alpha is never touched.
        let keep_alpha = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Zero,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipelines = match modules {
            (Some(dual), Some(mask), _) => Pipelines::Dual {
                // dst·T + src: source 1 is the per-channel transmittance.
                color: pipeline(
                    "atmosphere",
                    &dual,
                    "fs_dual",
                    &[Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::Src1,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: keep_alpha,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    })],
                ),
                mask: pipeline("atmosphere mask", &mask, "fs_mask", &[mask_target]),
            },
            (_, _, Some(grey)) => Pipelines::Grey(pipeline(
                "atmosphere",
                &grey,
                "fs_grey",
                &[
                    // dst·(1 − α) + L.
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: keep_alpha,
                        }),
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    mask_target,
                ],
            )),
            _ => anyhow::bail!("atmosphere: no shader variant"),
        };
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the atmosphere pipelines: {err}");
        }
        let atmos = set
            .entries()
            .iter()
            .map(|e| (e.naif, e.params.clone(), e.view_group.clone()))
            .collect();
        Ok(Self {
            input,
            atmos,
            pipelines,
            draw_bgl,
            draw_buffer,
            stride,
            draw_group: None,
            draws: Vec::new(),
        })
    }

    /// (Re)creates group 1 against `targets` (the depth view changes with the targets).
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.draw_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("atmo draw bg"),
            layout: &self.draw_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.draw_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(size_of::<AtmoDrawGpu>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&targets.depth_view),
                },
            ],
        }));
    }

    /// Records the draws of one pass (pipeline and group 0 already set).
    fn record(&self, pass: &mut wgpu::RenderPass<'_>, group1: &wgpu::BindGroup) {
        for d in &self.draws {
            pass.set_bind_group(1, group1, &[d.offset]);
            pass.set_bind_group(2, &self.atmos[d.atmo].2, &[]);
            pass.draw(0..d.vertices, 0..1);
        }
    }
}

impl RenderFeature for AtmosphereFeature {
    fn name(&self) -> &'static str {
        "atmosphere"
    }

    fn prepare(&mut self, gpu: &GpuContext, ctx: &FrameCtx, targets: &FrameTargets) {
        if self.draw_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        self.draws.clear();
        let mut uniforms: Vec<(u32, AtmoDrawGpu)> = Vec::new();
        self.input.with(|scene| {
            for draw in &scene.atmos {
                if uniforms.len() >= MAX_DRAWS {
                    break;
                }
                let Some(atmo) = self.atmos.iter().position(|a| a.0 == draw.naif) else {
                    continue;
                };
                let Some(u) = draw_uniform(&self.atmos[atmo].1, draw, ctx.view_rot, ctx.proj)
                else {
                    continue;
                };
                let offset = self.stride * uniforms.len() as u32;
                let vertices = if u.misc[1] > 0.5 { 3 } else { 6 };
                self.draws.push(DrawCmd {
                    offset,
                    atmo,
                    vertices,
                });
                uniforms.push((offset, u));
            }
        });
        for (offset, u) in uniforms {
            gpu.queue
                .write_buffer(&self.draw_buffer, u64::from(offset), bytemuck::bytes_of(&u));
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
        if self.draws.is_empty() {
            return;
        }
        if self.draw_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        let Some(group1) = self.draw_group.clone() else {
            return;
        };
        let attachment = |view| {
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
        match &self.pipelines {
            Pipelines::Dual { color, mask } => {
                for (label, pipeline, view) in [
                    ("atmosphere", color, &targets.hdr_view),
                    ("atmosphere mask", mask, &targets.lens_mask_view),
                ] {
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(label),
                        color_attachments: &[attachment(view)],
                        depth_stencil_attachment: None,
                        timestamp_writes: timer.render_writes(label),
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, frame.camera, &[]);
                    self.record(&mut pass, &group1);
                }
            }
            Pipelines::Grey(pipeline) => {
                let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("atmosphere"),
                    color_attachments: &[
                        attachment(&targets.hdr_view),
                        attachment(&targets.lens_mask_view),
                    ],
                    depth_stencil_attachment: None,
                    timestamp_writes: timer.render_writes("atmosphere"),
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, frame.camera, &[]);
                self.record(&mut pass, &group1);
            }
        }
    }

    fn resize(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_targets(gpu, targets);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lut::cpu_luts;
    use crate::params::{earth, mars, titan, venus};

    fn dir_elev(elev_deg: f64, azimuth_deg: f64) -> DVec3 {
        let (e, a) = (elev_deg.to_radians(), azimuth_deg.to_radians());
        DVec3::new(e.cos() * a.cos(), e.cos() * a.sin(), e.sin())
    }

    /// Zenith sky from the ground with the Sun 60° up: blue (B > G > R).
    #[test]
    fn earth_sky_blue_at_noon() {
        let p = earth();
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + 1.0);
        let (l, t) = segment_radiance(&p, ro, DVec3::Z, f64::INFINITY, dir_elev(60.0, 0.0));
        assert!(l.z > l.y && l.y > l.x, "zenith radiance per lux {l}");
        assert!(
            t.is_finite() && t.min_element() > 0.0 && t.max_element() < 1.0,
            "{t}"
        );
    }

    /// Review I6: with the per-channel dust g, the sky within 5° of a setting Sun is blue
    /// (B/R > 1: here 1–4° above a Sun 2° up) and the noon zenith is red (R/B > 1).
    #[test]
    fn mars_blue_sunset() {
        let p = mars();
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + 1.0);
        let sun = dir_elev(2.0, 0.0);
        for above in [1.0, 2.0, 3.0, 4.0] {
            let (near_sun, _) =
                segment_radiance(&p, ro, dir_elev(2.0 + above, 0.0), f64::INFINITY, sun);
            assert!(
                near_sun.z / near_sun.x > 1.0,
                "{above}° above the setting Sun: {near_sun}"
            );
        }
        let (zenith, _) = segment_radiance(&p, ro, DVec3::Z, f64::INFINITY, dir_elev(60.0, 0.0));
        assert!(zenith.x / zenith.z > 1.0, "noon zenith {zenith}");
    }

    /// Review Focus 2: a camera inside the shell (10 km up) marches from the camera (t_start 0),
    /// an upward ray is finite, a downward ray stops at the ground, a horizontal ray is fine.
    #[test]
    fn segment_from_inside_shell() {
        let p = earth();
        let h = 10e3;
        let ro = DVec3::new(0.0, 0.0, p.bottom_radius_m + h);
        let sun = dir_elev(30.0, 0.0);
        for rd in [DVec3::Z, -DVec3::Z, DVec3::X, dir_elev(-1.0, 45.0)] {
            let (l, t) = segment_radiance(&p, ro, rd, f64::INFINITY, sun);
            assert!(l.is_finite() && t.is_finite(), "{rd}: {l} {t}");
            assert!(l.min_element() > 0.0, "{rd}: {l}");
            assert!(t.min_element() > 0.0 && t.max_element() < 1.0, "{rd}: {t}");
        }
        // Downward: the ground at 10 km ends the march, so a longer t_max changes nothing.
        let ground = segment_radiance(&p, ro, -DVec3::Z, h, sun);
        let beyond = segment_radiance(&p, ro, -DVec3::Z, 1e9, sun);
        assert!((ground.0 - beyond.0).abs().max_element() <= 1e-6 * ground.0.max_element());
        assert!((ground.1 - beyond.1).abs().max_element() <= 1e-9);
        // t_start is the camera: a 100 m segment has the transmittance of 100 m of air there.
        let (short_l, short_t) = segment_radiance(&p, ro, DVec3::X, 100.0, sun);
        let ext = crate::lut::extinction_at(&p, h);
        for c in 0..3 {
            let expect = (-ext[c] * 100.0).exp();
            assert!(
                (f64::from(short_t[c]) - expect).abs() < 1e-6,
                "{c}: {short_t} vs {expect}"
            );
        }
        assert!(short_l.min_element() > 0.0);
        // Degenerate segments add nothing.
        assert_eq!(
            segment_radiance(&p, ro, DVec3::X, 0.0, sun),
            (Vec3::ZERO, Vec3::ONE)
        );
        let (l, t) = segment_radiance(&p, DVec3::NAN, DVec3::X, 1e3, sun);
        assert_eq!((l, t), (Vec3::ZERO, Vec3::ONE));
    }

    /// Every uniform struct matches its WGSL twin (size and member offsets, naga).
    #[test]
    fn atmo_uniforms_layout_match_wgsl() {
        use uni_gfx::{ShaderLib, validate_wgsl};
        use wgpu::naga;
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        let m = validate_wgsl(&lib.compose(crate::ATMOSPHERE_COMMON_WGSL).unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(m.to_ctx()).unwrap();
        let wgsl = |name: &str| -> (usize, Vec<(String, usize)>) {
            let (handle, ty) = m
                .types
                .iter()
                .find(|(_, t)| t.name.as_deref() == Some(name))
                .unwrap_or_else(|| panic!("struct {name}"));
            let naga::TypeInner::Struct { members, .. } = &ty.inner else {
                panic!("{name} is not a struct");
            };
            let members = members
                .iter()
                .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
                .collect();
            (layouter[handle].size as usize, members)
        };
        let (size, members) = wgsl("AtmoParamsGpu");
        assert_eq!(size, size_of::<AtmoParamsGpu>());
        let rust = AtmoParamsGpu::field_offsets();
        assert_eq!(members, rust);
        let (size, members) = wgsl("AtmoDrawGpu");
        assert_eq!(size, size_of::<AtmoDrawGpu>());
        assert_eq!(members, AtmoDrawGpu::field_offsets());
    }

    /// Review I7: the dual-source and mask variants validate with DUAL_SOURCE_BLENDING, the grey
    /// variant, the LUT passes and the probe kernels with m0.1's capabilities; every entry point
    /// the feature and the LUT builder use exists.
    #[test]
    fn atmo_shaders_validate() {
        use uni_gfx::{ShaderLib, validate_wgsl, validate_wgsl_with};
        use wgpu::naga::valid::Capabilities as Caps;
        let mut lib = ShaderLib::with_common();
        crate::register_shaders(&mut lib);
        crate::probes::register_probe_shaders(&mut lib);
        let caps = Caps::default() | Caps::SHADER_FLOAT16 | Caps::DUAL_SOURCE_BLENDING;
        let has = |m: &wgpu::naga::Module, entry: &str| {
            assert!(
                m.entry_points.iter().any(|e| e.name == entry),
                "missing entry point {entry}"
            );
        };
        for (path, entries) in [
            (
                crate::ATMOSPHERE_DUAL_WGSL,
                &["vs_atmosphere", "fs_dual"][..],
            ),
            (
                crate::ATMOSPHERE_MASK_WGSL,
                &["vs_atmosphere", "fs_mask"][..],
            ),
        ] {
            let m = validate_wgsl_with(&lib.compose(path).unwrap(), caps)
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            entries.iter().for_each(|e| has(&m, e));
        }
        for (path, entries) in [
            (
                crate::ATMOSPHERE_GREY_WGSL,
                &["vs_atmosphere", "fs_grey"][..],
            ),
            (crate::TRANSMITTANCE_LUT_WGSL, &["cs_transmittance"][..]),
            (crate::MULTISCATTER_LUT_WGSL, &["cs_multiscatter"][..]),
            (crate::IRRADIANCE_LUT_WGSL, &["cs_irradiance"][..]),
            (crate::probes::PROBE_WGSL, &crate::probes::PROBE_ENTRIES[..]),
        ] {
            let m = validate_wgsl(&lib.compose(path).unwrap())
                .unwrap_or_else(|e| panic!("{path}: {e}"));
            entries.iter().for_each(|e| has(&m, e));
        }
    }

    /// Rec. 709 luminance of a linear colour.
    fn lum(v: Vec3) -> f32 {
        0.2126 * v.x + 0.7152 * v.y + 0.0722 * v.z
    }

    /// Solar illuminance at 1 au [lx]: 1361 W/m² (Kopp & Lean 2011) × ≈ 93 lm/W.
    const E_SUN_1AU_LX: f32 = 1.27e5;

    /// The view-direction toward a limb point of a body seen from `cam` far along +x: the point
    /// at angle `phi` round the limb (0 toward +y, 90° toward +z) and tangent height `h`.
    fn limb_ray(p: &AtmosphereParams, cam: DVec3, phi_deg: f64, h: f64) -> DVec3 {
        let phi = phi_deg.to_radians();
        let target = DVec3::new(0.0, phi.cos(), phi.sin()) * (p.bottom_radius_m + h);
        (target - cam).normalize()
    }

    /// The 32–48-step march is within 1 % of a 1024-step reference (inscatter, relative to its
    /// largest channel) and within 1e-3 in transmittance (1 % where T > 0.01), on vertical,
    /// horizontal and limb paths.
    #[test]
    fn march_converges() {
        let p = earth();
        let b = p.bottom_radius_m;
        let luts = cpu_luts(&p);
        for (ro, rd, sun) in [
            (DVec3::new(0.0, 0.0, b + 1.0), DVec3::Z, dir_elev(30.0, 0.0)),
            (
                DVec3::new(0.0, 0.0, b + 1.0),
                dir_elev(1.0, 90.0),
                dir_elev(10.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                -DVec3::Z,
                dir_elev(40.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                dir_elev(-19.5, 0.0),
                dir_elev(10.0, 0.0),
            ),
            (
                DVec3::new(0.0, 0.0, b + 400e3),
                dir_elev(-20.1, 0.0),
                dir_elev(10.0, 0.0),
            ),
        ] {
            let ro = camera_origin(&p, ro).unwrap();
            let (t0, t1) = segment_bounds(&p, ro, rd, f64::INFINITY).unwrap();
            let o = ro + rd * t0;
            let fast = march(&luts, o, rd, 0.0, t1 - t0, sun);
            let reference = crate::lut::march_steps(&luts, o, rd, 0.0, t1 - t0, sun, 1024.0);
            let l_err = (fast.inscatter - reference.inscatter).abs().max_element()
                / reference.inscatter.max_element();
            let t_err = (fast.transmittance - reference.transmittance).abs();
            let t_ok = (0..3).all(|c| {
                t_err[c] < 1e-3
                    && (reference.transmittance[c] < 0.01
                        || t_err[c] < 0.01 * reference.transmittance[c])
            });
            assert!(
                l_err < 0.01 && t_ok,
                "{rd}: inscatter {l_err:.2e}, T {t_err}"
            );
        }
    }

    /// B9's `earth-limb-dawn` check on the twin: 420 km up, looking at the limb toward a Sun
    /// 1.8° below it; tangent heights 30–60 km are blue (B/R ≥ 1.5), 0–12 km red (R/B ≥ 1.5).
    #[test]
    fn earth_limb_dawn_arc_colours() {
        let p = earth();
        let b = p.bottom_radius_m;
        let cam = DVec3::new(0.0, 0.0, b + 420e3);
        let dip = (b / (b + 420e3)).acos().to_degrees();
        let sun = dir_elev(-dip - 1.8, 0.0);
        let at = |h: f64| {
            let elev = -((b + h) / (b + 420e3)).acos().to_degrees();
            segment_radiance(&p, cam, dir_elev(elev, 0.0), f64::INFINITY, sun).0
        };
        for h in [30e3, 45e3, 60e3] {
            let l = at(h);
            assert!(l.z / l.x >= 1.5, "{} km: {l}", h / 1e3);
        }
        for h in [0.5e3, 6e3, 12e3] {
            let l = at(h);
            assert!(l.x / l.z >= 1.5, "{} km: {l}", h / 1e3);
        }
    }

    /// B9's `venus-crescent` check (phase 155°, 150 000 km): the haze above the cloud tops is
    /// brighter than 183 cd/m² (3 stops above AgX black at EV 16.7) on the lit limb and where the
    /// limb crosses the terminator.
    #[test]
    fn venus_haze_ring_visible() {
        let p = venus();
        let e = E_SUN_1AU_LX / (0.7233_f32 * 0.7233);
        let phase = 155.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(150_000e3, 0.0, 0.0);
        for phi in [0.0, 45.0, 90.0] {
            let brightest = [2e3, 8e3, 15e3, 20e3, 30e3]
                .into_iter()
                .map(|h| {
                    lum(segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun).0)
                        * e
                })
                .fold(0.0, f32::max);
            assert!(brightest >= 183.0, "limb at {phi}°: {brightest} cd/m²");
        }
    }

    /// B9's `titan-backlit` check (phase 160°, 60 000 km): the main haze at the limb is orange
    /// (R/B ≥ 1.2) and brighter than 0.27 cd/m² (visible at EV 7.3).
    #[test]
    fn titan_backlit_haze_orange() {
        let p = titan();
        let e = E_SUN_1AU_LX / (9.58_f32 * 9.58);
        let phase = 160.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(60_000e3, 0.0, 0.0);
        for phi in [0.0, 90.0] {
            for h in [50e3, 100e3, 150e3] {
                let (l, _) =
                    segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun);
                assert!(
                    l.x / l.z >= 1.2 && lum(l) * e >= 0.27,
                    "limb {phi}° at {} km: {l}",
                    h / 1e3
                );
            }
        }
    }

    /// B9's `mars-reference-angle` check (phase 45°, 8 210 km): the limb haze is red (R/B ≥ 1).
    #[test]
    fn mars_limb_haze_red() {
        let p = mars();
        let phase = 45.0_f64.to_radians();
        let sun = DVec3::new(phase.cos(), phase.sin(), 0.0);
        let cam = DVec3::new(8_210e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            for h in [5e3, 20e3, 40e3] {
                let (l, _) =
                    segment_radiance(&p, cam, limb_ray(&p, cam, phi, h), f64::INFINITY, sun);
                assert!(
                    l.x / l.z >= 1.0 && l.min_element() > 0.0,
                    "limb {phi}° at {} km: {l}",
                    h / 1e3
                );
            }
        }
    }

    /// Ruling W-B1: Earth's disk-integrated geometric albedo at 550 nm (≈ V) with the
    /// atmosphere over A8's Lambert composite 0.302 is p = 0.210 (600-sample value; 0.201
    /// without the atmosphere), 0.79 mag below the catalogue p_V 0.434: the atmosphere adds
    /// light (0.063 over a black ground) but also dims the ground. Pinned so B9's measurement has
    /// a reference. p = (2π/R²) ∫ (L/E) b db over impact parameters out to the top (a Lambert
    /// sphere gives 2A/3).
    #[test]
    fn earth_disk_albedo_w_b1() {
        let p = earth();
        let (r, top) = (p.bottom_radius_m, p.top_radius_m);
        let ground = 0.302;
        let n = 200;
        let mut sum = 0.0;
        for k in 0..n {
            // Denser toward the limb: b = top·sin(π/2·s).
            let s = |k: usize| top * (k as f64 / n as f64 * std::f64::consts::FRAC_PI_2).sin();
            let (b0, b1) = (s(k), s(k + 1));
            let b = 0.5 * (b0 + b1);
            let (l, t) = segment_radiance(
                &p,
                DVec3::new(1e9, 0.0, b),
                -DVec3::X,
                f64::INFINITY,
                DVec3::X,
            );
            let mut radiance = f64::from(l.y);
            if b < r {
                let mu = (1.0 - (b / r).powi(2)).sqrt();
                let (t_sun, sky) = crate::ground::ground_lighting(Some(&p), r, mu);
                radiance += f64::from(t.y) * ground / std::f64::consts::PI
                    * (f64::from(t_sun.y) * mu + f64::from(sky.y));
            }
            sum += radiance * b * (b1 - b0);
        }
        let albedo = sum * std::f64::consts::TAU / (r * r);
        assert!((0.20..=0.22).contains(&albedo), "p_550 = {albedo}");
    }

    /// The screen rectangle contains the projection of every point of the sphere and touches
    /// it on all four sides.
    #[test]
    fn sphere_screen_rect_bounds_the_sphere() {
        let (fov, aspect) = (60f64.to_radians(), 16.0 / 9.0);
        let sy = 1.0 / (0.5 * fov).tan();
        let sx = sy / aspect;
        for (c, radius) in [
            (DVec3::new(0.0, 0.0, -10.0), 3.0),
            (DVec3::new(4.0, -2.0, -12.0), 2.0),
            (DVec3::new(-30.0, 9.0, -20.0), 5.0),
            (DVec3::new(1e7, 3e6, -4e7), 6.4e6),
        ] {
            let [x0, y0, x1, y1] = sphere_screen_rect(c, radius, sx, sy).unwrap();
            let (mut lo, mut hi) = (glam::DVec2::splat(f64::MAX), glam::DVec2::splat(f64::MIN));
            for i in 0..200 {
                for j in 0..400 {
                    let (th, ph) = (
                        std::f64::consts::PI * (i as f64 + 0.5) / 200.0,
                        std::f64::consts::TAU * j as f64 / 400.0,
                    );
                    let q =
                        c + DVec3::new(th.sin() * ph.cos(), th.sin() * ph.sin(), th.cos()) * radius;
                    let ndc = glam::DVec2::new(sx * q.x / -q.z, sy * q.y / -q.z);
                    lo = lo.min(ndc);
                    hi = hi.max(ndc);
                }
            }
            let tol = 1e-9 * (x1 - x0).abs().max(1.0);
            assert!(lo.x >= x0 - tol && lo.y >= y0 - tol && hi.x <= x1 + tol && hi.y <= y1 + tol);
            let slack = 1e-3 * (x1 - x0).max(y1 - y0);
            assert!(
                lo.x - x0 < slack && lo.y - y0 < slack && x1 - hi.x < slack && y1 - hi.y < slack
            );
        }
        assert!(sphere_screen_rect(DVec3::new(0.0, 0.0, -2.0), 3.0, sx, sy).is_none());
        assert!(sphere_screen_rect(DVec3::new(0.0, 0.0, 5.0), 3.0, sx, sy).is_none());
    }

    /// The draw's frame maps Earth's ground to the bottom radius at the pole and the equator,
    /// the camera inside the shell gets the fullscreen triangle, and a camera inside the body, a
    /// shell behind the camera or bad input draws nothing.
    #[test]
    fn draw_uniform_follows_the_ellipsoid() {
        let p = earth();
        let orient = DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(1.1);
        let m = to_body_frame(399, orient);
        let abc = body_shape_m(399).unwrap();
        for fixed in [
            DVec3::new(0.0, 0.0, abc.z),
            DVec3::new(abc.x, 0.0, 0.0),
            DVec3::new(0.0, -abc.y, 0.0),
        ] {
            let r = (m * (orient * fixed)).length();
            assert!((r - p.bottom_radius_m).abs() < 2.0, "{fixed}: {r}");
        }
        let proj = uni_gfx::reversed_z_infinite_proj(60f32.to_radians(), 16.0 / 9.0, 0.1);
        let draw = |rel: DVec3| AtmoDraw {
            naif: 399,
            rel_pos_m: rel,
            orient,
            sun_rel_m: DVec3::new(1.5e11, 0.0, 0.0),
            sun_illuminance_lux: 1.27e5,
        };
        // Body 5 radii ahead (−Z): a quad inside the screen, origin height ≈ 4 radii.
        let u = draw_uniform(
            &p,
            &draw(DVec3::new(0.0, 0.0, -5.0 * p.bottom_radius_m)),
            Mat4::IDENTITY,
            proj,
        )
        .unwrap();
        assert_eq!(u.misc[1], 0.0);
        assert!(
            u.rect[0] > -1.0 && u.rect[2] < 1.0 && u.rect[0] < u.rect[2] && u.rect[1] < u.rect[3]
        );
        assert!((u.origin[3] / p.bottom_radius_m as f32 - 4.0).abs() < 0.01);
        assert_eq!(u.sun[3], 1.27e5);
        // Inside the shell, 10 km up: fullscreen.
        let inside = draw(DVec3::new(0.0, -(p.bottom_radius_m + 10e3), 0.0));
        assert_eq!(
            draw_uniform(&p, &inside, Mat4::IDENTITY, proj)
                .unwrap()
                .misc[1],
            1.0
        );
        // Inside the body; behind the camera; off screen; non-finite.
        assert!(
            draw_uniform(&p, &draw(DVec3::new(0.0, -1e6, 0.0)), Mat4::IDENTITY, proj).is_none()
        );
        assert!(
            draw_uniform(
                &p,
                &draw(DVec3::new(0.0, 0.0, 5.0 * p.bottom_radius_m)),
                Mat4::IDENTITY,
                proj
            )
            .is_none()
        );
        assert!(
            draw_uniform(
                &p,
                &draw(DVec3::new(
                    9.0 * p.bottom_radius_m,
                    0.0,
                    -2.0 * p.bottom_radius_m
                )),
                Mat4::IDENTITY,
                proj
            )
            .is_none()
        );
        assert!(draw_uniform(&p, &draw(DVec3::NAN), Mat4::IDENTITY, proj).is_none());
    }

    /// Geometric albedo at phase 0 per channel over a Lambert ground (see
    /// `earth_disk_albedo_w_b1`), with `n` impact parameters.
    fn disk_albedo(p: &AtmosphereParams, ground: Vec3, n: usize) -> Vec3 {
        let (r, top) = (p.bottom_radius_m, p.top_radius_m);
        let mut sum = DVec3::ZERO;
        for k in 0..n {
            let s = |k: usize| top * (k as f64 / n as f64 * std::f64::consts::FRAC_PI_2).sin();
            let (b0, b1) = (s(k), s(k + 1));
            let b = 0.5 * (b0 + b1);
            let (l, t) = segment_radiance(
                p,
                DVec3::new(1e9 + 10.0 * top, 0.0, b),
                -DVec3::X,
                f64::INFINITY,
                DVec3::X,
            );
            let mut radiance = l.as_dvec3();
            if b < r {
                let mu = (1.0 - (b / r).powi(2)).sqrt();
                let (t_sun, sky) = crate::ground::ground_lighting(Some(p), r, mu);
                radiance += t.as_dvec3() * ground.as_dvec3() / std::f64::consts::PI
                    * (t_sun.as_dvec3() * mu + sky.as_dvec3());
            }
            sum += radiance * b * (b1 - b0);
        }
        (sum * std::f64::consts::TAU / (r * r)).as_vec3()
    }

    /// Prints every number the B3 report quotes (`cargo test -p uni-atmo --lib report_numbers
    /// -- --ignored --nocapture`; about 20 s unoptimised).
    #[test]
    #[ignore = "print-only: the numbers the B3 task report quotes"]
    fn report_numbers() {
        let e = earth();
        let b = e.bottom_radius_m;
        println!(
            "Earth T(ground, zenith) = {}",
            crate::transmittance(&e, b, 1.0)
        );
        let t = crate::transmittance(&e, b, 0.02);
        println!("Earth T(ground, mu_s 0.02) = {t}, R/B {:.1}", t.x / t.z);
        let mu = 60f64.to_radians().sin();
        let (l, _) = segment_radiance(
            &e,
            DVec3::new(0.0, 0.0, b + 1.0),
            DVec3::Z,
            f64::INFINITY,
            dir_elev(60.0, 0.0),
        );
        let sky = crate::sky_irradiance(&e, b, mu);
        let direct = crate::transmittance(&e, b, mu) * mu as f32;
        println!(
            "Earth Sun 60°: zenith L/E {l} = {:.0} cd/m² at {E_SUN_1AU_LX} lx; sky E/E0 {sky}; diffuse fraction (G) {:.3}",
            lum(l) * E_SUN_1AU_LX,
            sky.y / (sky.y + direct.y)
        );
        let m = mars();
        let ro = DVec3::new(0.0, 0.0, m.bottom_radius_m + 1.0);
        for above in [1.0, 2.0, 3.0, 4.0, 6.0, 10.0] {
            let (l, _) = segment_radiance(
                &m,
                ro,
                dir_elev(2.0 + above, 0.0),
                f64::INFINITY,
                dir_elev(2.0, 0.0),
            );
            println!("Mars {above}° above a Sun 2° up: B/R {:.3}", l.z / l.x);
        }
        for sun in [30.0, 45.0, 60.0] {
            let (l, _) = segment_radiance(&m, ro, DVec3::Z, f64::INFINITY, dir_elev(sun, 0.0));
            println!("Mars zenith, Sun {sun}° up: R/B {:.3}", l.x / l.z);
        }
        let cam = DVec3::new(0.0, 0.0, b + 420e3);
        let dip = (b / (b + 420e3)).acos().to_degrees();
        for h in [60e3, 45e3, 30e3, 20e3, 12e3, 6e3, 0.5e3] {
            let elev = -((b + h) / (b + 420e3)).acos().to_degrees();
            let (l, _) = segment_radiance(
                &e,
                cam,
                dir_elev(elev, 0.0),
                f64::INFINITY,
                dir_elev(-dip - 1.8, 0.0),
            );
            println!(
                "earth-limb-dawn tangent {:>4.1} km: B/R {:.2}, R/B {:.2}, {:.0} cd/m²",
                h / 1e3,
                l.z / l.x,
                l.x / l.z,
                lum(l) * E_SUN_1AU_LX
            );
        }
        let v = venus();
        let ev = E_SUN_1AU_LX / (0.7233 * 0.7233);
        let phase = 155f64.to_radians();
        let vc = DVec3::new(150_000e3, 0.0, 0.0);
        for phi in [0.0, 45.0, 90.0, 100.0, 110.0] {
            for h in [2e3, 8e3, 15e3, 20e3, 30e3] {
                let (l, _) = segment_radiance(
                    &v,
                    vc,
                    limb_ray(&v, vc, phi, h),
                    f64::INFINITY,
                    DVec3::new(phase.cos(), phase.sin(), 0.0),
                );
                println!(
                    "venus-crescent limb {phi:>5.1}° tangent {:>4.1} km: {:.3e} cd/m² (needs ≥ 183)",
                    h / 1e3,
                    lum(l) * ev
                );
            }
        }
        let ti = titan();
        let et = E_SUN_1AU_LX / (9.58 * 9.58);
        let phase = 160f64.to_radians();
        let tc = DVec3::new(60_000e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            for h in [50e3, 100e3, 150e3, 300e3, 500e3] {
                let (l, _) = segment_radiance(
                    &ti,
                    tc,
                    limb_ray(&ti, tc, phi, h),
                    f64::INFINITY,
                    DVec3::new(phase.cos(), phase.sin(), 0.0),
                );
                println!(
                    "titan-backlit limb {phi:>5.1}° tangent {:>5.1} km: {:.3} cd/m² (needs ≥ 0.27), R/B {:.2}",
                    h / 1e3,
                    lum(l) * et,
                    l.x / l.z
                );
            }
        }
        let phase = 45f64.to_radians();
        let mc = DVec3::new(8_210e3, 0.0, 0.0);
        for phi in [0.0, 90.0, 180.0] {
            let (l, _) = segment_radiance(
                &m,
                mc,
                limb_ray(&m, mc, phi, 20e3),
                f64::INFINITY,
                DVec3::new(phase.cos(), phase.sin(), 0.0),
            );
            println!(
                "mars-reference-angle limb {phi:>5.1}° tangent 20 km: R/B {:.2}",
                l.x / l.z
            );
        }
        for (name, p, ground) in [
            ("Earth, ground 0.302", earth(), Vec3::splat(0.302)),
            ("Earth, black ground", earth(), Vec3::ZERO),
            (
                "Earth, no atmosphere",
                AtmosphereParams {
                    rayleigh_scattering: Vec3::ZERO,
                    mie_scattering: Vec3::ZERO,
                    mie_extinction: Vec3::ZERO,
                    layer2_extinction: Vec3::ZERO,
                    ..earth()
                },
                Vec3::splat(0.302),
            ),
            ("Venus, cloud deck 0.76", venus(), Vec3::splat(0.76)),
            (
                "Mars, ground (0.25, 0.15, 0.08)",
                mars(),
                Vec3::new(0.25, 0.15, 0.08),
            ),
            // A Lambert ground calibrated to the catalogue p_V 0.17 without an atmosphere
            // (2A/3 = 0.17), as A8 calibrates Mars's map: what the dust does to that disk.
            (
                "Mars, ground 0.255 (p_V-calibrated)",
                mars(),
                Vec3::splat(0.255),
            ),
            ("Titan, ground 0.15", titan(), Vec3::splat(0.15)),
        ] {
            println!("geometric albedo, {name}: {}", disk_albedo(&p, ground, 600));
        }
        for (name, p) in [
            ("Earth", earth()),
            ("Mars", mars()),
            ("Venus", venus()),
            ("Titan", titan()),
        ] {
            let s = |h: f64| {
                crate::scintillation_strength(&p, DVec3::new(0.0, 0.0, p.bottom_radius_m + h))
            };
            println!(
                "scintillation {name}: ground {:.3}, 5 km {:.3}, 30 km {:.3}",
                s(0.0),
                s(5e3),
                s(30e3)
            );
        }
    }
}
