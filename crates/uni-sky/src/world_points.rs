//! Point sources drawn as world light (final review render I2).
//!
//! A light whose illuminance at the camera is at least [`ADAPTING_LUX`] light-adapts the eye: in
//! M1 that is the Sun from anywhere inside ~356 AU. When such a light projects below
//! `uni_sim::snapshot::POINT_LOD_PX` (1.5 px; the Sun from beyond ~3.2 AU at a 60° field, so
//! every view from Jupiter out), it used to be a sky-light point at the eye-adapted star
//! exposure, never metered: in `asteroid-belt-above` (6 AU, 3.5 × 10³ lx) its halo, clamped at
//! 64 px and renormalised inside, drew a hard-edged flat-white 129-px disk (final review I2, gate
//! look item 3, P36). Now it is world light: drawn at `scene_pre()` before the exposure meter, by
//! the star-surface pass ("sun", uni-planet), which owns a [`WorldPointDrawer`], so the meter
//! sees its glare and the eye adapts to it; the point pass skips its sprite
//! ([`is_adapting_light_sprite`]). Its PSF is the star pass's (`sky/stars3d.wgsl`), whose clamped
//! halo now tapers to black inside the clamp ([`crate::stars3d::halo_shape`]).
//!
//! [`ADAPTING_LUX`]: 1 lx, about 3 × the full Moon (−12.7 mag, 0.3 lx). The CIE disability-glare
//! veil a point source of illuminance E lays over the field θ degrees from it is L ≈ 10·E/θ²
//! cd/m² for 1° < θ < 30° (Stiles–Holladay; CIE 146:2002's general disability glare equation
//! reduces to it there, the equation B4L fitted the halo to). At 1 lx the veil 10° away is
//! 0.1 cd/m², twenty times the lower limit of mesopic vision (0.005 cd/m², CIE 191:2010): the eye
//! cannot stay dark-adapted. Fainter lights (from ~0.05 lx, where the veil at 10° reaches that
//! limit) partly adapt it too but stay sky light here, so the Sun seen from TestBH-1 (1000 AU,
//! 0.13 lx) and as a star from α Cen (1.7 × 10⁻⁶ lx) keep their star-exposure look. Only
//! `FrameCtx::lights` are candidates (the Sun in M1): a catalogue star the point pass draws
//! within ~400 AU of it would also pass 1 lx and stays sky light (stated in the report).

use glam::{DVec3, Mat4, Vec2, Vec3};
use uni_gfx::{GpuContext, HDR_FORMAT, LENS_MASK_FORMAT, LightView, ShaderLib};

use crate::stars3d::{
    HaloShape, MAX_SPLAT_ENERGY, QUAD_INDICES, STARS3D_HALO_FRACTION, halo_shape,
    quad_index_buffer, star_is_visible, vertex_buffer_entry,
};

/// `ShaderLib` path of the world-point splat.
pub const WORLD_POINTS_WGSL: &str = "sky/world_points.wgsl";
/// Group-1 binding of the instances (bindings 0–5 are declared by the imported star shaders).
const WORLD_POINTS_BINDING: u32 = 6;
/// Instances the buffer holds at first (it grows by powers of two).
const INITIAL_CAPACITY: usize = 4;

/// Illuminance at the camera [lx] from which a light light-adapts the eye and its point is world
/// light (module docs).
pub const ADAPTING_LUX: f64 = 1.0;

/// Illuminance of `light` at the camera [lx]: I/d² (NaN for a light at the camera).
pub fn light_illuminance_lux(light: &LightView) -> f64 {
    let d2 = light.rel_pos_m.length_squared();
    light.luminous_intensity_cd / d2
}

/// Whether `light` light-adapts the eye: its illuminance is finite and at least
/// [`ADAPTING_LUX`].
pub fn light_adapts_eye(light: &LightView) -> bool {
    let e = light_illuminance_lux(light);
    e.is_finite() && e >= ADAPTING_LUX
}

/// Whether a source at `rel_pos_m` (source − camera) is one of `lights` that light-adapts the eye
/// ([`light_adapts_eye`]); the point pass skips such a sprite, the star-surface pass draws it as
/// world light. Positions match to 1e-9 of the distance (both come from the same snapshot).
pub fn is_adapting_light_sprite(rel_pos_m: DVec3, lights: &[LightView]) -> bool {
    lights.iter().any(|l| {
        light_adapts_eye(l)
            && (rel_pos_m - l.rel_pos_m).length() <= 1e-9 * l.rel_pos_m.length().max(1.0)
    })
}

/// One point drawn as world light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPoint {
    /// Source centre − camera [m], root axes.
    pub rel_pos_m: DVec3,
    /// Illuminance at the camera [lx] this splat carries (the light's, or its share in the
    /// star-surface pass's point → disk cross-fade).
    pub illuminance_lux: f64,
    /// Linear Rec. 709 colour, luminance 1.
    pub color: Vec3,
    /// Distance [m] at which its depth is tested: in front of its own disk, so the disk never
    /// hides it (the star-surface pass uses d − 2R).
    pub depth_m: f64,
}

/// One world point on the GPU: twin of `struct WorldPointGpu` in `world_points.wgsl` (32 bytes,
/// two vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct WorldPointGpu {
    /// xyz: unit direction source − camera; w: the depth distance [m].
    pub dir_depth: [f32; 4],
    /// rgb: colour, luminance 1; w: illuminance [lx].
    pub color_lux: [f32; 4],
}

/// Packs `p`, or `None` when it cannot be drawn (no light, a zero or non-finite position or
/// depth distance, a non-finite colour). The direction is normalised in f64 before the cast.
pub(crate) fn world_point_gpu(p: &WorldPoint) -> Option<WorldPointGpu> {
    let d = p.rel_pos_m.length();
    if !(d > 0.0 && d.is_finite())
        || !(p.depth_m > 0.0 && p.depth_m.is_finite())
        || !(p.illuminance_lux > 0.0)
        || !p.color.is_finite()
    {
        return None;
    }
    let dir = (p.rel_pos_m / d).as_vec3();
    Some(WorldPointGpu {
        dir_depth: dir.extend(p.depth_m as f32).to_array(),
        color_lux: p
            .color
            .max(Vec3::ZERO)
            .extend((p.illuminance_lux as f32).min(f32::MAX))
            .to_array(),
    })
}

/// A world point's splat on a frame, as `vs_world_points` places it (the CPU twin of
/// `splat_vertex_at` at `scene_pre`, for the paper exposure).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldSplat {
    /// Centre in framebuffer pixels (x right, y down, pixel k spans [k, k + 1]).
    pub centre_px: Vec2,
    /// The world luminance [cd/m²] per unit PSF weight: E/Ω(θ).
    pub luminance_per_weight: f32,
    /// The halo share (`STARS3D_HALO_FRACTION`) and footprint.
    pub halo_fraction: f32,
    pub shape: HaloShape,
}

impl WorldSplat {
    /// The world luminance [cd/m²] the splat writes at pixel `pix` (integer index).
    pub fn luminance_at(&self, pix: Vec2) -> f32 {
        self.luminance_per_weight
            * crate::stars3d::psf_weight(pix, self.centre_px, self.halo_fraction, &self.shape)
    }
}

/// The splat of a source in unit direction `dir` (root axes) of illuminance `lux` on a `size`-px
/// target with vertical field `fov_y` [rad], root → camera rotation `view_rot` and projection
/// `proj`, written × `pre` (the scene exposure); `None` where the vertex stage culls it (behind
/// the camera, no light, below the visibility cutoff, off screen by more than its footprint).
/// Twin of `splat_vertex_at` (`stars3d.wgsl`).
pub fn world_splat(
    view_rot: Mat4,
    proj: Mat4,
    size: Vec2,
    fov_y: f32,
    dir: Vec3,
    lux: f32,
    pre: f32,
) -> Option<WorldSplat> {
    let view_dir = (view_rot * dir.extend(0.0)).truncate();
    let clip = proj * view_dir.extend(0.0);
    if !(clip.w > crate::stars::MIN_CLIP_W) || !(lux > 0.0) {
        return None;
    }
    let ndc = Vec2::new(clip.x, clip.y) / clip.w;
    let centre = Vec2::new(0.5 * (ndc.x + 1.0) * size.x, 0.5 * (1.0 - ndc.y) * size.y);
    let cos_theta = -view_dir.z / view_dir.length();
    let omega = crate::stars::pixel_solid_angle_off_axis(fov_y, size.y, cos_theta);
    let energy = (lux * pre / omega).min(MAX_SPLAT_ENERGY);
    let h = STARS3D_HALO_FRACTION;
    if !star_is_visible(energy, h) {
        return None;
    }
    let shape = halo_shape(energy, h);
    let reach = Vec2::splat(shape.radius_px + 1.0);
    if !(centre.cmpgt(-reach).all() && centre.cmplt(size + reach).all()) {
        return None;
    }
    Some(WorldSplat {
        centre_px: centre,
        luminance_per_weight: energy / pre,
        halo_fraction: h,
        shape,
    })
}

/// Draws [`WorldPoint`]s inside another feature's render pass: the star-surface pass's, whose
/// colour attachments are `hdr` (0) and `lens_mask` (1) with the depth target attached and
/// group 0 bound to the frame's camera. Additive into `hdr`'s colour, nothing into `lens_mask`,
/// depth tested `Greater` and never written (`world_points.wgsl`).
pub struct WorldPointDrawer {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    index_buffer: wgpu::Buffer,
    capacity: usize,
    count: u32,
}

fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("world points"),
        size: (capacity * size_of::<WorldPointGpu>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind_group(
    device: &wgpu::Device,
    bgl: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("world points bg"),
        layout: bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: WORLD_POINTS_BINDING,
            resource: buffer.as_entire_binding(),
        }],
    })
}

impl WorldPointDrawer {
    /// Builds the drawer. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, WORLD_POINTS_WGSL)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world points bgl"),
            entries: &[vertex_buffer_entry(
                WORLD_POINTS_BINDING,
                wgpu::BufferBindingType::Storage { read_only: true },
                size_of::<WorldPointGpu>(),
            )],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world points layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("world points"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_world_points"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_world_points"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: HDR_FORMAT,
                        blend: Some(wgpu::BlendState::ADDITIVE),
                        // Light adds to colour only; hdr's alpha is left alone.
                        write_mask: wgpu::ColorWrites::COLOR,
                    }),
                    // The host pass's lens_mask: untouched (glare occludes nothing).
                    Some(wgpu::ColorTargetState {
                        format: LENS_MASK_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::empty(),
                    }),
                ],
            }),
            // No culling: the quad's winding depends on the y flip.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: uni_gfx::DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        let buffer = instance_buffer(device, INITIAL_CAPACITY);
        let group = bind_group(device, &bgl, &buffer);
        let index_buffer = quad_index_buffer(device);
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the world-point pipeline: {err}");
        }
        Ok(Self {
            pipeline,
            bgl,
            buffer,
            bind_group: group,
            index_buffer,
            capacity: INITIAL_CAPACITY,
            count: 0,
        })
    }

    /// Uploads this frame's points (undrawable ones skipped), growing the buffer if needed.
    pub fn prepare(&mut self, gpu: &GpuContext, points: &[WorldPoint]) {
        let packed: Vec<WorldPointGpu> = points.iter().filter_map(world_point_gpu).collect();
        if packed.len() > self.capacity {
            self.capacity = packed.len().next_power_of_two();
            self.buffer = instance_buffer(&gpu.device, self.capacity);
            self.bind_group = bind_group(&gpu.device, &self.bgl, &self.buffer);
        }
        self.count = u32::try_from(packed.len()).unwrap_or(u32::MAX);
        if !packed.is_empty() {
            gpu.queue
                .write_buffer(&self.buffer, 0, bytemuck::cast_slice(&packed));
        }
    }

    /// Points prepared for this frame.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Draws the prepared points into `pass` (see the type docs; group 0 must be bound). Leaves
    /// its own pipeline and group 1 bound.
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if self.count == 0 {
            return;
        }
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        pass.draw_indexed(0..QUAD_INDICES.len() as u32, 0, 0..self.count);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stars3d::tests_support::{calls, entry, func, module, wgsl_struct};
    use crate::stars3d::twin::{ViewTwin, splat_at};
    use std::mem::offset_of;

    fn light(rel: DVec3, cd: f64) -> LightView {
        LightView {
            rel_pos_m: rel,
            radius_m: uni_core::R_SUN_M,
            luminous_intensity_cd: cd,
            color: Vec3::ONE,
        }
    }

    /// The Sun's luminous intensity [cd] (uni-sim's, as the app's lights carry it).
    fn sun_cd() -> f64 {
        uni_sim::world::sun_intensity_cd()
    }

    /// The threshold: the Sun light-adapts the eye out to ~356 AU, not from TestBH-1 (1000 AU) or
    /// α Cen (1.3 pc); a sprite is skipped only where it coincides with such a light.
    #[test]
    fn adapting_lights_and_sprites() {
        let au = uni_core::AU;
        let at = |d_au: f64| {
            light(
                DVec3::new(0.3, -0.4, 0.866).normalize() * d_au * au,
                sun_cd(),
            )
        };
        // 1 AU: 1.27e5 lx (the solar constant's illuminance, uni-sim).
        assert!((light_illuminance_lux(&at(1.0)) / 1.27e5 - 1.0).abs() < 0.02);
        for (d, adapts) in [(1.0, true), (6.0, true), (40.0, true), (300.0, true)] {
            assert_eq!(light_adapts_eye(&at(d)), adapts, "{d} AU");
        }
        for d in [400.0, 1000.0, 1.3 * uni_core::PC / au] {
            assert!(!light_adapts_eye(&at(d)), "{d} AU");
        }
        // The crossing: E = 1 lx at √(I/1 lx) ≈ 356 AU.
        let edge_au = sun_cd().sqrt() / au;
        assert!((edge_au - 356.0).abs() < 5.0, "{edge_au}");
        assert!(!light_adapts_eye(&light(DVec3::ZERO, sun_cd())));
        let lights = [at(6.0), at(1000.0)];
        assert!(is_adapting_light_sprite(lights[0].rel_pos_m, &lights));
        assert!(!is_adapting_light_sprite(lights[1].rel_pos_m, &lights));
        assert!(!is_adapting_light_sprite(
            lights[0].rel_pos_m + DVec3::X * 1e6,
            &lights
        ));
        assert!(!is_adapting_light_sprite(lights[0].rel_pos_m, &[]));
    }

    #[test]
    fn world_point_layout_matches_wgsl_and_packing_sanitises() {
        let m = module(WORLD_POINTS_WGSL);
        let (size, offsets, vec4_only) = wgsl_struct(&m, "WorldPointGpu");
        assert_eq!(size, 32);
        assert_eq!(size, size_of::<WorldPointGpu>());
        assert!(vec4_only);
        assert_eq!(
            offsets,
            vec![
                ("dir_depth".to_owned(), offset_of!(WorldPointGpu, dir_depth)),
                ("color_lux".to_owned(), offset_of!(WorldPointGpu, color_lux)),
            ]
        );
        let ok = WorldPoint {
            rel_pos_m: DVec3::new(3.0, 4.0, 0.0),
            illuminance_lux: 2.0,
            color: Vec3::new(1.0, 0.9, 0.8),
            depth_m: 4.0,
        };
        let g = world_point_gpu(&ok).unwrap();
        assert_eq!(g.dir_depth, [0.6, 0.8, 0.0, 4.0]);
        assert_eq!(g.color_lux, [1.0, 0.9, 0.8, 2.0]);
        for bad in [
            WorldPoint {
                rel_pos_m: DVec3::ZERO,
                ..ok
            },
            WorldPoint { depth_m: 0.0, ..ok },
            WorldPoint {
                illuminance_lux: f64::NAN,
                ..ok
            },
            WorldPoint {
                color: Vec3::splat(f32::NAN),
                ..ok
            },
        ] {
            assert_eq!(world_point_gpu(&bad), None, "{bad:?}");
        }
    }

    /// The shader: written at the scene exposure on the internal target, with the star pass's
    /// PSF; it binds only its instances (binding 6) besides group 0, and never the lens mask
    /// (which is the host pass's attachment).
    #[test]
    fn world_points_shader_uses_scene_exposure_and_the_star_psf() {
        let m = module(WORLD_POINTS_WGSL);
        let vs = entry(&m, "vs_world_points");
        let fs = entry(&m, "fs_world_points");
        assert!(calls(&vs.body, func(&m, "splat_vertex_at")));
        assert!(calls(&vs.body, func(&m, "scene_pre")));
        assert!(!calls(&vs.body, func(&m, "star_pre")));
        assert!(!calls(&vs.body, func(&m, "splat_vertex")));
        assert!(calls(&fs.body, func(&m, "psf_weight")));
        let reads = |f: &wgpu::naga::Function, name: &str| {
            f.expressions.iter().any(|(_, e)| match e {
                wgpu::naga::Expression::GlobalVariable(g) => {
                    m.global_variables[*g].name.as_deref() == Some(name)
                }
                _ => false,
            })
        };
        for f in [vs, fs] {
            assert!(!reads(f, "lens_mask"));
        }
        let (_, var) = m
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("world_points"))
            .unwrap();
        assert_eq!(
            var.binding,
            Some(wgpu::naga::ResourceBinding {
                group: 1,
                binding: WORLD_POINTS_BINDING
            })
        );
    }

    /// The CPU splat of a world point equals the star pass's twin at the same exposure (one
    /// PSF), and its luminance per weight is E/Ω.
    #[test]
    fn world_splat_is_the_star_splat_at_scene_exposure() {
        let axis = Vec3::new(0.3, -0.5, 0.8).normalize();
        let view = ViewTwin::looking(axis, 70.0, Vec2::new(1920.0, 1080.0), -6.5);
        let pre = uni_gfx::exposure_scale(18.0);
        for (lux, off) in [
            (3.5e3_f32, 0.0_f32),
            (3.5e3, 20.0),
            (1.2e4, 3.0),
            (1.0, 33.0),
        ] {
            let dir =
                (glam::Quat::from_axis_angle(axis.any_orthonormal_vector(), off.to_radians())
                    * axis)
                    .normalize();
            let w = world_splat(
                view.view_rot,
                view.proj,
                view.size,
                view.fov_y,
                dir,
                lux,
                pre,
            )
            .expect("on screen");
            let s = splat_at(&view, dir, lux, 1.0, Vec3::ONE, 0.5, pre).expect("on screen");
            assert_eq!(w.centre_px, s.centre_px);
            assert_eq!(w.shape.radius_px, s.radius_px);
            assert_eq!(w.shape.scale, s.halo_scale);
            assert_eq!(w.shape.taper, s.taper);
            assert!((w.luminance_per_weight * pre / s.energy - 1.0).abs() < 1e-6);
            // Its pixels sum to the light it keeps: within 2 % of E/Ω × (1 − h·dropped).
            let mut sum = 0.0_f64;
            let c = w.centre_px.floor();
            let r = w.shape.radius_px as i32 + 1;
            for dy in -r..=r {
                for dx in -r..=r {
                    sum += f64::from(w.luminance_at(c + Vec2::new(dx as f32, dy as f32)));
                }
            }
            let ratio = sum / f64::from(w.luminance_per_weight);
            assert!(ratio > 0.97 && ratio < 1.01, "{lux} at {off}°: {ratio}");
        }
        // Behind the camera: nothing.
        assert!(
            world_splat(
                view.view_rot,
                view.proj,
                view.size,
                view.fov_y,
                -axis,
                3.5e3,
                pre
            )
            .is_none()
        );
    }
}
