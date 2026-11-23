//! Unresolved bodies, the Sun and near stars drawn as points, pass "points" (Task B4).
//!
//! The same PSF as the star pass (`splat_vertex` in `stars3d.wgsl`, which `points.wgsl`
//! imports: M0's energy-normalised core and halo, the 2-px minimum footprint, the same cull),
//! from an instance buffer rebuilt per frame from the app's `Vec<PointSprite>`, written
//! × `star_pre()` × (1 − r) (review m6), with vertex depth from the true distance and compare
//! `Greater`, so bodies in front hide them. A point and a star of equal illuminance and direction
//! leave the same display energy and footprint (`point_sprite_energy_matches_star_path`), so a
//! near star handed from the star pass to this one does not pop.
//!
//! Points are not dimmed by `lens_mask.g` (sky occlusion): a point behind an atmosphere is not
//! dimmed by it in M1, the frame contract's accepted approximation. The app pre-tints star
//! colours through [`crate::vivid_color`]; this pass applies no colour change of its own.

use glam::{DVec3, Vec3};
use uni_gfx::{
    FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature, ShaderLib,
    SharedInput,
};

use crate::POINTS_WGSL;
use crate::stars3d::{
    QUAD_INDICES, begin_splat_pass, lens_mask_entry, quad_index_buffer, splat_pipeline,
    vertex_buffer_entry,
};

/// One point source (from uni-sim's `PointView`).
#[derive(Clone, Copy, Debug)]
pub struct PointSprite {
    /// Point − camera [m], ICRS axes, f64.
    pub rel_pos_m: DVec3,
    /// Illuminance at the camera [lx].
    pub illuminance_lux: f64,
    /// Linear Rec. 709 colour with luminance 1. Reflected light is the solar colour, tinted by
    /// `uni_planet::BodyRenderer::mean_albedo_rgb` where the body has a map (carry-over 6);
    /// star colours pass through [`crate::vivid_color`] while vivid colours are on.
    pub color: Vec3,
}

/// One point sprite as the GPU sees it: the twin of `struct PointGpu` in `points.wgsl` (32
/// bytes, two vec4s).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PointGpu {
    /// xyz: unit direction point − camera, root (ICRS) axes; w: distance [m].
    pub dir_dist: [f32; 4],
    /// rgb: colour (linear Rec. 709, luminance 1); w: illuminance at the camera [lx].
    pub color_lux: [f32; 4],
}

/// Packs `p` for the GPU, or `None` when it cannot be drawn (no light, a zero or non-finite
/// distance, a non-finite colour). The direction is normalised in f64 before the cast, and an
/// illuminance beyond f32's range becomes `f32::MAX`. Negative colour channels are clamped to 0.
pub(crate) fn point_gpu(p: &PointSprite) -> Option<PointGpu> {
    let d = p.rel_pos_m.length();
    if !(d > 0.0 && d.is_finite()) || !(p.illuminance_lux > 0.0) || !p.color.is_finite() {
        return None;
    }
    let dir = (p.rel_pos_m / d).as_vec3();
    let lux = (p.illuminance_lux as f32).min(f32::MAX);
    Some(PointGpu {
        dir_dist: dir.extend(d as f32).to_array(),
        color_lux: p.color.max(Vec3::ZERO).extend(lux).to_array(),
    })
}

/// Instances the point buffer holds at first; it grows by powers of two.
const INITIAL_CAPACITY: usize = 256;

/// The point-sprite pass (`RenderFeature` named "points").
///
/// Group 0 is the renderer's. Group 1: binding 4 the lens mask (the declaration it shares with
/// `stars3d.wgsl`), binding 5 the instances. The bind group is rebuilt when the instance buffer
/// grows and in `resize`.
pub struct PointSprites {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    instance_buffer: wgpu::Buffer,
    /// Instances `instance_buffer` holds.
    capacity: usize,
    index_buffer: wgpu::Buffer,
    /// Group 1 for the current targets and instance buffer.
    bind_group: Option<wgpu::BindGroup>,
    input: SharedInput<Vec<PointSprite>>,
    /// This frame's packed instances (reused between frames).
    scratch: Vec<PointGpu>,
    /// Instances to draw this frame.
    count: u32,
}

fn instance_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("point sprites"),
        size: (capacity * size_of::<PointGpu>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl PointSprites {
    /// Builds the pass. `camera_bgl` is `Renderer::camera_bind_group_layout()`.
    pub fn new(
        gpu: &GpuContext,
        lib: &mut ShaderLib,
        camera_bgl: &wgpu::BindGroupLayout,
        input: SharedInput<Vec<PointSprite>>,
    ) -> anyhow::Result<Self> {
        let device = &gpu.device;
        crate::register_shaders(lib);
        let module = lib.module(device, POINTS_WGSL)?;

        // Validation errors in the objects below come back as an Err, not a device panic.
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let instance_buffer = instance_buffer(device, INITIAL_CAPACITY);
        let index_buffer = quad_index_buffer(device);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("points bgl"),
            entries: &[
                lens_mask_entry(4),
                vertex_buffer_entry(
                    5,
                    wgpu::BufferBindingType::Storage { read_only: true },
                    size_of::<PointGpu>(),
                ),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("points layout"),
            bind_group_layouts: &[Some(camera_bgl), Some(&bgl)],
            immediate_size: 0,
        });
        // Points sit at their true distance: a body in front (greater reversed-Z depth) hides
        // them, and they draw over the cleared far plane and the sky.
        let pipeline = splat_pipeline(
            device,
            &module,
            &layout,
            ("vs_points", "fs_points"),
            wgpu::CompareFunction::Greater,
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("creating the point-sprite pipeline: {err}");
        }

        Ok(Self {
            pipeline,
            bgl,
            instance_buffer,
            capacity: INITIAL_CAPACITY,
            index_buffer,
            bind_group: None,
            input,
            scratch: Vec::new(),
            count: 0,
        })
    }

    /// (Re)creates group 1 against `targets` and the current instance buffer.
    fn bind_targets(&mut self, gpu: &GpuContext, targets: &FrameTargets) {
        self.bind_group = Some(gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("points bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&targets.lens_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.instance_buffer.as_entire_binding(),
                },
            ],
        }));
    }
}

impl RenderFeature for PointSprites {
    fn name(&self) -> &'static str {
        "points"
    }

    /// Packs this frame's sprites (skipping undrawable ones), grows the instance buffer when
    /// needed and uploads them.
    fn prepare(&mut self, gpu: &GpuContext, _ctx: &FrameCtx, targets: &FrameTargets) {
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.clear();
        self.input
            .with(|points| scratch.extend(points.iter().filter_map(point_gpu)));
        if scratch.len() > self.capacity {
            self.capacity = scratch.len().next_power_of_two();
            self.instance_buffer = instance_buffer(&gpu.device, self.capacity);
            self.bind_group = None;
        }
        if self.bind_group.is_none() {
            self.bind_targets(gpu, targets);
        }
        // More sprites than one draw can count is not a real scene; draw the first u32::MAX.
        self.count = u32::try_from(scratch.len()).unwrap_or(u32::MAX);
        if !scratch.is_empty() {
            gpu.queue
                .write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&scratch));
        }
        self.scratch = scratch;
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
        let mut pass = begin_splat_pass(enc, targets, timer, "points");
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

/// CPU twins of the point pass (test only).
#[cfg(test)]
pub(crate) mod twin {
    use super::*;
    use crate::stars3d::twin::{Splat, ViewTwin, splat};
    use glam::Vec4;

    /// The splat of point `p`, or `None` where `vs_points` culls it (behind the camera, nearer
    /// than the near plane, or culled by `splat_vertex`). Twin of `vs_points`.
    pub fn point_splat(view: &ViewTwin, p: &PointGpu) -> Option<Splat> {
        let dir = Vec3::from_slice(&p.dir_dist[..3]);
        let view_dir = (view.view_rot * dir.extend(0.0)).truncate();
        let clip = view.proj * (view_dir * p.dir_dist[3]).extend(1.0);
        let depth = clip.z / clip.w;
        if !(depth > 0.0 && depth <= 1.0) {
            return None;
        }
        splat(
            view,
            dir,
            p.color_lux[3],
            Vec3::from_slice(&p.color_lux[..3]),
            depth,
        )
    }

    /// The share of a point's light kept under `lens_mask` value `mask`: 1 − r (review m6).
    /// Twin of `point_sky_factor`.
    pub fn point_sky_factor(mask: Vec4) -> f32 {
        1.0 - mask.x.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::twin::*;
    use super::*;
    use crate::POINTS_WGSL;
    use crate::stars3d::tests_support::{calls, entry, func, module};
    use crate::stars3d::twin::{ViewTwin, splat, star_splat};
    use crate::stars3d::{StarInput, pack_stars, stars_params};
    use glam::{Vec2, Vec4};
    use uni_core::PC;
    use uni_data::Star3;

    /// A Star3 and a PointSprite with the same illuminance and direction give the same display
    /// energy (±1e-6 relative), footprint and halo, so a near star handed from the star pass to
    /// the point pass does not pop.
    #[test]
    fn point_sprite_energy_matches_star_path() {
        let axis = Vec3::new(0.3, -0.5, 0.8).normalize();
        let view = ViewTwin::looking(axis, 60.0, Vec2::new(1920.0, 1080.0), -4.0);
        // Rigel-like: M_V −7.0 at 264 pc, 5° off the axis.
        let dir = (glam::Quat::from_axis_angle(axis.any_orthonormal_vector(), 5f32.to_radians())
            * axis)
            .normalize();
        let pos = (dir * 264.0).to_array();
        let star = Star3::new(pos, -7.0, [0.0; 3], -0.03, 1, 0);
        let packed = pack_stars(&[star]);
        let params = stars_params(&StarInput::default(), DVec3::ZERO, &packed.slots);
        let rel = packed.table.centres_rel(DVec3::ZERO);
        let s = star_splat(&view, &packed, &rel, &params, 0).expect("star on screen");
        // The star path's illuminance agrees with uni_core's f64 formula.
        let d = glam::DVec3::from_array(pos.map(f64::from)).length();
        let lux_f64 = uni_core::star_illuminance_lux(-7.0 + 5.0 * d.log10() - 5.0);
        let star_lux = s.energy
            * crate::stars::pixel_solid_angle_off_axis(
                view.fov_y,
                view.size.y,
                crate::stars::off_axis_cos(view.view_rot, dir),
            )
            / view.star_pre;
        assert!(
            (f64::from(star_lux) / lux_f64 - 1.0).abs() < 1e-5,
            "{star_lux} vs {lux_f64}"
        );

        let point = PointSprite {
            rel_pos_m: glam::DVec3::from_array(pos.map(f64::from)) * PC,
            illuminance_lux: f64::from(crate::stars3d::star_lux(
                crate::stars3d::star_apparent_mag(-7.0, Vec3::from(pos).length()),
            )),
            color: Vec3::ONE,
        };
        let p = point_splat(&view, &point_gpu(&point).expect("drawable")).expect("on screen");
        assert!(
            (p.energy / s.energy - 1.0).abs() < 1e-6,
            "{} vs {}",
            p.energy,
            s.energy
        );
        assert!(
            (p.radius_px - s.radius_px).abs() < 1e-4,
            "{} vs {}",
            p.radius_px,
            s.radius_px
        );
        assert_eq!(p.halo_fraction, s.halo_fraction);
        assert!((p.centre_px - s.centre_px).length() < 1e-3);
        // The star sits on the far plane; the point at its true distance, in front of it.
        assert_eq!(s.depth, 0.0);
        assert!(p.depth > 0.0);
    }

    /// Review m6: points honour the black hole's capture (r) but not sky occlusion (g); they are
    /// depth-tested against bodies instead.
    #[test]
    fn points_respect_capture_mask() {
        assert_eq!(point_sky_factor(Vec4::new(1.0, 0.0, 0.0, 0.0)), 0.0);
        assert_eq!(point_sky_factor(Vec4::new(0.25, 0.0, 0.0, 0.0)), 0.75);
        assert_eq!(point_sky_factor(Vec4::new(0.0, 1.0, 1.0, 1.0)), 1.0);
        assert_eq!(point_sky_factor(Vec4::ZERO), 1.0);
        // The CPU model of fs_points: radiance × PSF weight × (1 − r).
        let energy = 3.0;
        for (r, want) in [(1.0, 0.0), (0.25, 0.75 * energy), (0.0, energy)] {
            let mask = Vec4::new(r, 0.6, 0.0, 0.0);
            assert_eq!(energy * point_sky_factor(mask), want);
        }
        // The shader's fragment stage uses that factor, not the star pass's (1 − r)(1 − g).
        let m = module(POINTS_WGSL);
        let fs = entry(&m, "fs_points");
        assert!(calls(&fs.body, func(&m, "point_sky_factor")));
        assert!(!calls(&fs.body, func(&m, "star_sky_factor")));
    }

    /// Vertex depth comes from the true distance: nearer points have larger reversed-Z depth, the
    /// same value a body's vertex at that distance gets, so compare `Greater` lets bodies in
    /// front hide a point and never the reverse.
    #[test]
    fn point_depth_from_true_distance() {
        let view = ViewTwin::looking(Vec3::Y, 60.0, Vec2::new(800.0, 600.0), -4.0);
        let depth = |d: f64| {
            let p = PointSprite {
                rel_pos_m: DVec3::Y * d,
                illuminance_lux: 1e-3,
                color: Vec3::ONE,
            };
            point_splat(&view, &point_gpu(&p).unwrap())
                .expect("on screen")
                .depth
        };
        let (near, far) = (depth(1.0e9), depth(4.5e12));
        assert!(near > far && far > 0.0, "{near} {far}");
        // proj × (0, 0, −d, 1): depth = near / d with the projection's 0.1 m near plane.
        assert!((near / (0.1 / 1.0e9) - 1.0).abs() < 1e-6, "{near}");
        // A point behind the camera is culled.
        let behind = PointSprite {
            rel_pos_m: DVec3::NEG_Y * 1e9,
            illuminance_lux: 1e-3,
            color: Vec3::ONE,
        };
        assert!(point_splat(&view, &point_gpu(&behind).unwrap()).is_none());
        // And the splat matches the shared vertex twin at that depth.
        let p = point_gpu(&PointSprite {
            rel_pos_m: DVec3::Y * 1e9,
            illuminance_lux: 1e-3,
            color: Vec3::ONE,
        })
        .unwrap();
        let s = splat(&view, Vec3::Y, 1e-3, Vec3::ONE, near).unwrap();
        assert_eq!(point_splat(&view, &p), Some(s));
    }

    #[test]
    fn point_gpu_sanitises() {
        let ok = PointSprite {
            rel_pos_m: DVec3::new(3.0, 4.0, 0.0),
            illuminance_lux: 2.0,
            color: Vec3::new(1.0, 0.9, 0.8),
        };
        let g = point_gpu(&ok).unwrap();
        assert_eq!(g.dir_dist, [0.6, 0.8, 0.0, 5.0]);
        assert_eq!(g.color_lux, [1.0, 0.9, 0.8, 2.0]);
        for bad in [
            PointSprite {
                rel_pos_m: DVec3::ZERO,
                ..ok
            },
            PointSprite {
                rel_pos_m: DVec3::new(f64::NAN, 0.0, 1.0),
                ..ok
            },
            PointSprite {
                illuminance_lux: 0.0,
                ..ok
            },
            PointSprite {
                illuminance_lux: f64::NAN,
                ..ok
            },
            PointSprite {
                color: Vec3::new(f32::NAN, 1.0, 1.0),
                ..ok
            },
        ] {
            assert_eq!(point_gpu(&bad), None, "{bad:?}");
        }
        // Too bright for f32 stays finite.
        let blinding = PointSprite {
            illuminance_lux: 1e300,
            ..ok
        };
        assert!(point_gpu(&blinding).unwrap().color_lux[3].is_finite());
    }
}
