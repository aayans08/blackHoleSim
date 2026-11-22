//! Per-frame context handed to every feature, and its GPU twins `CameraUniform`,
//! `ExposureUniform` and `LightUniform` (group 0 of `shaders/common/camera.wgsl`).

use glam::{DVec3, Mat4, UVec2, Vec2, Vec3};
use uni_core::{DAY, Epoch, UniversePos};

/// Exposure values (EV at ISO 100) for the scene and for point-like stars.
///
/// The manual exposure (the HUD sliders and bookmarks). The renderer turns it into the GPU
/// [`ExposureUniform`] every frame; M1's auto-exposure will write that buffer on the GPU and use
/// this as its target or override.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Exposure {
    pub scene_ev100: f32,
    pub star_ev100: f32,
}

impl Exposure {
    /// True when stars are exposed differently from the scene: the "Eye-adapted stars" dual
    /// exposure of design §7 item 4, which is non-physical and must be labelled on screen.
    pub fn eye_adapted_stars(&self) -> bool {
        self.star_ev100 != self.scene_ev100
    }
}

/// A lens as seen from the camera.
#[derive(Clone, Debug)]
pub struct LensView {
    /// Lens − camera [m], root axes.
    pub rel_pos_m: DVec3,
    /// Mass [kg].
    pub mass_kg: f64,
    /// Dimensionless spin a/M.
    pub spin: f64,
}

/// Most lights the GPU sees per frame ([`LightUniform`]); `FrameCtx::lights` beyond the first
/// `MAX_LIGHTS` finite ones are dropped.
pub const MAX_LIGHTS: usize = 4;

/// Period [s] of the simulation time sent to the GPU, 2^22 s ≈ 48.5 d. Below 2^22 an f32 steps
/// by at most 0.25 s, so shader animation driven by `sim_time_s()` stays smooth at any epoch.
pub const SIM_TIME_WRAP_S: f64 = 4_194_304.0;

/// A light source (a star) as seen from the camera.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightView {
    /// Light centre − camera [m], root axes.
    pub rel_pos_m: DVec3,
    /// Photosphere radius [m].
    pub radius_m: f64,
    /// Luminous intensity [cd]: the illuminance at distance d is I/d² lux.
    pub luminous_intensity_cd: f64,
    /// Colour in linear Rec. 709, normalised to luminance 1.
    pub color: Vec3,
}

/// Everything a feature needs to know about the current frame.
pub struct FrameCtx {
    pub frame_index: u64,
    pub epoch: uni_core::Epoch,
    pub dt_real_s: f64,
    pub camera: uni_core::CameraView,
    /// Root→camera rotation only (camera at the origin).
    pub view_rot: Mat4,
    /// Reversed-Z infinite projection ([`crate::reversed_z_infinite_proj`]).
    pub proj: Mat4,
    /// Internal render resolution [px]: the size of the frame targets every feature renders
    /// into, decoupled from the window. `Renderer::render` keeps the targets at this size.
    pub viewport: UVec2,
    /// Size [px] of the output view the tonemap writes (the swapchain or a capture). The
    /// tonemap upscales `viewport` to it; captures render with both equal.
    pub output_size: UVec2,
    pub exposure: Exposure,
    pub lenses: Vec<LensView>,
    /// World lights, brightest first; the GPU gets the first [`MAX_LIGHTS`] finite ones.
    pub lights: Vec<LightView>,
    /// Simulated seconds this frame advanced the epoch by (0 when paused).
    pub sim_dt_s: f64,
    /// Sub-pixel projection jitter [px] for M2's TAA; 0 in M1.
    pub jitter_px: Vec2,
}

/// Twin of `struct CameraUniform` in `shaders/common/camera.wgsl`; group(0) binding(0),
/// written once per frame by the `Renderer` and bound by every feature.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_rot: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    /// inverse(proj * view_rot): pixel → root-axes direction.
    pub inv_view_proj: [[f32; 4]; 4],
    /// The previous frame's proj * view_rot (the current one on the first frame); M2 TAA.
    pub prev_view_proj: [[f32; 4]; 4],
    /// Internal resolution: w, h, 1/w, 1/h.
    pub viewport: [f32; 4],
    /// Output view: w, h, 1/w, 1/h.
    pub output: [f32; 4],
    /// frame_index as f32 (modulo 2^24), dt_real_s, fov_y, aspect.
    pub time: [f32; 4],
    /// TDB seconds since J2000 modulo [`SIM_TIME_WRAP_S`], sim dt [s], jitter x, y [px].
    pub sim: [f32; 4],
    /// Camera translation since the previous frame [m], root axes (f64 difference, then f32);
    /// w 0. Zero on the first frame.
    pub cam_delta: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<CameraUniform>() == 336);

/// `frame_index` is sent modulo 2^24, the largest range in which every integer is exact in f32.
const FRAME_INDEX_WRAP: u64 = 1 << 24;

/// (w, h, 1/w, 1/h) of `size`, a zero axis treated as 1 so the reciprocals stay finite.
fn size_vec4(size: UVec2) -> [f32; 4] {
    let w = size.x.max(1) as f32;
    let h = size.y.max(1) as f32;
    [w, h, 1.0 / w, 1.0 / h]
}

/// TDB seconds since J2000.0 modulo [`SIM_TIME_WRAP_S`], in [0, 2^22).
///
/// 2^15 days = 2^22 · 675 s, a whole number of periods, so the day count is reduced first and
/// the f64 sum stays below 2.9e9 s: the result keeps sub-microsecond resolution at any epoch.
fn wrapped_sim_time_s(epoch: Epoch) -> f64 {
    const WRAP_DAYS: i64 = 1 << 15;
    let days = epoch.day.rem_euclid(WRAP_DAYS) as f64;
    (days * DAY + epoch.sec).rem_euclid(SIM_TIME_WRAP_S)
}

impl CameraUniform {
    /// Packs the frame context for the GPU, as a first frame: `prev_view_proj` is the current
    /// `proj * view_rot` and `cam_delta` is 0. The `Renderer` fills both from the previous
    /// frame.
    ///
    /// A zero viewport or output axis is treated as 1 so `1/w`, `1/h` and the aspect stay
    /// finite. `frame_index` wraps modulo 2^24 so it keeps counting exactly in f32.
    pub fn from_ctx(ctx: &FrameCtx) -> Self {
        let viewport = size_vec4(ctx.viewport);
        let view_proj = ctx.proj * ctx.view_rot;
        Self {
            view_rot: ctx.view_rot.to_cols_array_2d(),
            proj: ctx.proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: view_proj.to_cols_array_2d(),
            viewport,
            output: size_vec4(ctx.output_size),
            time: [
                (ctx.frame_index % FRAME_INDEX_WRAP) as f32,
                ctx.dt_real_s as f32,
                ctx.camera.fov_y_rad as f32,
                viewport[0] / viewport[1],
            ],
            sim: [
                wrapped_sim_time_s(ctx.epoch) as f32,
                ctx.sim_dt_s as f32,
                ctx.jitter_px.x,
                ctx.jitter_px.y,
            ],
            cam_delta: [0.0; 4],
        }
    }
}

/// The previous frame's camera, which the `Renderer` keeps to fill
/// [`CameraUniform::prev_view_proj`] and [`CameraUniform::cam_delta`].
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CameraHistory {
    /// proj * view_rot and the camera position of the last frame packed.
    prev: Option<(Mat4, UniversePos)>,
}

impl CameraHistory {
    /// The camera uniform for `ctx`, relative to the previous call's frame; remembers this one.
    /// The translation is taken between the two `UniversePos` in f64 and only then cast to f32,
    /// so a 1.5 m step stays 1.5 m a gigaparsec from the origin. A non-finite step gives 0.
    pub(crate) fn uniform(&mut self, ctx: &FrameCtx) -> CameraUniform {
        let mut u = CameraUniform::from_ctx(ctx);
        if let Some((view_proj, pos)) = self.prev {
            u.prev_view_proj = view_proj.to_cols_array_2d();
            let d = ctx.camera.pos.delta_from(pos).as_vec3();
            if d.is_finite() {
                u.cam_delta = d.extend(0.0).to_array();
            }
        }
        self.prev = Some((ctx.proj * ctx.view_rot, ctx.camera.pos));
        u
    }
}

/// One light on the GPU: twin of `struct LightGpu` in `shaders/common/camera.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightGpu {
    /// Light centre − camera [m] (root axes), photosphere radius [m].
    pub pos_radius: [f32; 4],
    /// Linear Rec. 709 colour (luminance 1), luminous intensity [cd].
    pub color_intensity: [f32; 4],
}

impl LightGpu {
    /// `l` in f32, or `None` if any value is not finite once in f32.
    fn from_view(l: &LightView) -> Option<Self> {
        let p = l.rel_pos_m.as_vec3();
        let g = Self {
            pos_radius: [p.x, p.y, p.z, l.radius_m as f32],
            color_intensity: [
                l.color.x,
                l.color.y,
                l.color.z,
                l.luminous_intensity_cd as f32,
            ],
        };
        g.pos_radius
            .iter()
            .chain(&g.color_intensity)
            .all(|v| v.is_finite())
            .then_some(g)
    }
}

/// Twin of `struct LightUniform` in `shaders/common/camera.wgsl`; group(0) binding(2).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightUniform {
    /// x: number of valid entries in `lights` (≤ [`MAX_LIGHTS`]); yzw 0.
    pub info: [u32; 4],
    /// Valid entries first; the rest are zero.
    pub lights: [LightGpu; MAX_LIGHTS],
}

const _: () = assert!(core::mem::size_of::<LightUniform>() == 144);

impl LightUniform {
    /// The first [`MAX_LIGHTS`] lights of `ctx` whose values are finite in f32, in order. A
    /// non-finite light is dropped rather than put on the GPU as NaN.
    pub fn from_ctx(ctx: &FrameCtx) -> Self {
        let mut u: Self = bytemuck::Zeroable::zeroed();
        let finite = ctx.lights.iter().filter_map(LightGpu::from_view);
        for (slot, light) in u.lights.iter_mut().zip(finite) {
            *slot = light;
            u.info[0] += 1;
        }
        u
    }
}

/// Scale from scene luminance [cd/m²] to normalised exposure for a camera at `ev100`:
/// 1/(1.2·2^EV100). Saturation-based sensitivity, L_max = (78/(S·q))·2^EV100 with S = 100 and
/// q = 0.65: Lagarde & de Rousiers 2014, "Moving Frostbite to Physically Based Rendering 3.0",
/// §4.1.1 (after ISO 12232:2006). Twin of `exposure_scale` in `shaders/common/exposure.wgsl`.
pub fn exposure_scale(ev100: f32) -> f32 {
    1.0 / (1.2 * ev100.exp2())
}

/// Twin of `struct ExposureUniform` in `shaders/common/exposure.wgsl`, bound as `exposure` at
/// group(0) binding(1) by `shaders/common/camera.wgsl`.
///
/// The pre-exposure contract: features write light × exposure into hdr, so the tonemap does
/// not rescale. World light uses `pre[0]`, catalogue stars `pre[1]`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ExposureUniform {
    /// Scene pre-exposure, star pre-exposure ([`exposure_scale`] of each EV), 0, 0.
    pub pre: [f32; 4],
    /// Scene EV100, star EV100, 0, 0.
    pub ev100: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<ExposureUniform>() == 32);

impl ExposureUniform {
    /// The pre-exposure factors for a manual exposure.
    pub fn from_exposure(e: Exposure) -> Self {
        Self {
            pre: [
                exposure_scale(e.scene_ev100),
                exposure_scale(e.star_ev100),
                0.0,
                0.0,
            ],
            ev100: [e.scene_ev100, e.star_ev100, 0.0, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::reversed_z_infinite_proj;
    use crate::shader::{ShaderLib, validate_wgsl};
    use core::mem::{offset_of, size_of};
    use glam::{DQuat, Vec4};

    #[test]
    fn camera_uniform_layout_matches_wgsl() {
        let lib = ShaderLib::with_common();
        let module = validate_wgsl(&lib.compose("common/camera.wgsl").unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();

        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("CameraUniform"))
            .expect("struct CameraUniform in camera.wgsl");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("CameraUniform is not a struct");
        };

        assert_eq!(size_of::<CameraUniform>(), 336);
        assert_eq!(layouter[handle].size as usize, size_of::<CameraUniform>());

        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust: Vec<(String, usize)> = [
            ("view_rot", offset_of!(CameraUniform, view_rot)),
            ("proj", offset_of!(CameraUniform, proj)),
            ("inv_view_proj", offset_of!(CameraUniform, inv_view_proj)),
            ("prev_view_proj", offset_of!(CameraUniform, prev_view_proj)),
            ("viewport", offset_of!(CameraUniform, viewport)),
            ("output", offset_of!(CameraUniform, output)),
            ("time", offset_of!(CameraUniform, time)),
            ("sim", offset_of!(CameraUniform, sim)),
            ("cam_delta", offset_of!(CameraUniform, cam_delta)),
        ]
        .into_iter()
        .map(|(n, o)| (n.to_owned(), o))
        .collect();
        assert_eq!(wgsl, rust);

        // `camera` is the uniform at group(0) binding(0).
        let (_, var) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("camera"))
            .expect("var camera in camera.wgsl");
        assert_eq!(var.space, naga::AddressSpace::Uniform);
        assert_eq!(
            var.binding,
            Some(naga::ResourceBinding {
                group: 0,
                binding: 0
            })
        );
        assert_eq!(var.ty, handle);
    }

    #[test]
    fn exposure_uniform_layout_matches_wgsl() {
        let lib = ShaderLib::with_common();
        let module = validate_wgsl(&lib.compose("common/camera.wgsl").unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("ExposureUniform"))
            .expect("struct ExposureUniform in camera.wgsl");
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("ExposureUniform is not a struct");
        };
        assert_eq!(size_of::<ExposureUniform>(), 32);
        assert_eq!(layouter[handle].size as usize, size_of::<ExposureUniform>());
        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        assert_eq!(
            wgsl,
            [
                ("pre".to_owned(), offset_of!(ExposureUniform, pre)),
                ("ev100".to_owned(), offset_of!(ExposureUniform, ev100)),
            ]
        );
        // `exposure` is the uniform at group(0) binding(1).
        let (_, var) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("exposure"))
            .expect("var exposure in camera.wgsl");
        assert_eq!(var.space, naga::AddressSpace::Uniform);
        assert_eq!(
            var.binding,
            Some(naga::ResourceBinding {
                group: 0,
                binding: 1
            })
        );
        assert_eq!(var.ty, handle);
    }

    /// The composed `common/camera.wgsl` and its layouter.
    fn camera_module() -> (naga::Module, naga::proc::Layouter) {
        let lib = ShaderLib::with_common();
        let module = validate_wgsl(&lib.compose("common/camera.wgsl").unwrap()).unwrap();
        let mut layouter = naga::proc::Layouter::default();
        layouter.update(module.to_ctx()).unwrap();
        (module, layouter)
    }

    /// (name, offset) of every member of WGSL struct `name`, and its handle.
    fn wgsl_members(
        module: &naga::Module,
        name: &str,
    ) -> (naga::Handle<naga::Type>, Vec<(String, usize)>) {
        let (handle, ty) = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("struct {name} in camera.wgsl"));
        let naga::TypeInner::Struct { members, .. } = &ty.inner else {
            panic!("{name} is not a struct");
        };
        let members = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        (handle, members)
    }

    #[test]
    fn light_uniform_layout_matches_wgsl() {
        let (module, layouter) = camera_module();
        let (light, members) = wgsl_members(&module, "LightGpu");
        assert_eq!(size_of::<LightGpu>(), 32);
        assert_eq!(layouter[light].size as usize, size_of::<LightGpu>());
        assert_eq!(
            members,
            [
                ("pos_radius".to_owned(), offset_of!(LightGpu, pos_radius)),
                (
                    "color_intensity".to_owned(),
                    offset_of!(LightGpu, color_intensity)
                ),
            ]
        );

        let (handle, members) = wgsl_members(&module, "LightUniform");
        assert_eq!(size_of::<LightUniform>(), 144);
        assert_eq!(layouter[handle].size as usize, size_of::<LightUniform>());
        assert_eq!(
            members,
            [
                ("info".to_owned(), offset_of!(LightUniform, info)),
                ("lights".to_owned(), offset_of!(LightUniform, lights)),
            ]
        );
        assert_eq!(offset_of!(LightUniform, lights), 16);
        // `lights` is an array of MAX_LIGHTS LightGpu with a 32-byte stride, as in Rust.
        let naga::TypeInner::Struct { members, .. } = &module.types[handle].inner else {
            unreachable!()
        };
        let naga::TypeInner::Array { base, size, stride } = module.types[members[1].ty].inner
        else {
            panic!("LightUniform.lights is not an array");
        };
        assert_eq!(base, light);
        assert_eq!(stride as usize, size_of::<LightGpu>());
        assert_eq!(
            size,
            naga::ArraySize::Constant(core::num::NonZeroU32::new(MAX_LIGHTS as u32).unwrap())
        );

        // `lights` is the uniform at group(0) binding(2).
        let (_, var) = module
            .global_variables
            .iter()
            .find(|(_, v)| v.name.as_deref() == Some("lights"))
            .expect("var lights in camera.wgsl");
        assert_eq!(var.space, naga::AddressSpace::Uniform);
        assert_eq!(
            var.binding,
            Some(naga::ResourceBinding {
                group: 0,
                binding: 2
            })
        );
        assert_eq!(var.ty, handle);
    }

    fn light(x: f64) -> LightView {
        LightView {
            rel_pos_m: DVec3::new(x, -2.0 * x, 0.5),
            radius_m: 6.957e8,
            luminous_intensity_cd: 2.8e27,
            color: Vec3::new(1.05, 0.98, 0.93),
        }
    }

    #[test]
    fn light_uniform_drops_non_finite_and_caps_count() {
        let mut ctx = test_ctx(UVec2::new(4, 4));
        ctx.lights = (1..=6).map(|i| light(f64::from(i) * 1.0e11)).collect();
        ctx.lights[1].rel_pos_m.y = f64::NAN;
        let u = LightUniform::from_ctx(&ctx);
        assert_eq!(u.info, [4, 0, 0, 0]);
        // The first four finite lights, in order: 1, 3, 4, 5 (1-based); the NaN one is absent.
        for (slot, i) in [0usize, 2, 3, 4].into_iter().enumerate() {
            let l = &ctx.lights[i];
            let p = l.rel_pos_m.as_vec3();
            assert_eq!(
                u.lights[slot].pos_radius,
                [p.x, p.y, p.z, l.radius_m as f32],
                "slot {slot}"
            );
            assert_eq!(
                u.lights[slot].color_intensity,
                [
                    l.color.x,
                    l.color.y,
                    l.color.z,
                    l.luminous_intensity_cd as f32
                ],
                "slot {slot}"
            );
        }
        assert!(
            u.lights
                .iter()
                .flat_map(|l| l.pos_radius.iter().chain(&l.color_intensity))
                .all(|v| v.is_finite())
        );

        // A value finite in f64 but not in f32 is dropped too; no lights gives count 0.
        ctx.lights = vec![light(1.0), light(1.0)];
        ctx.lights[0].luminous_intensity_cd = 1e300;
        let u = LightUniform::from_ctx(&ctx);
        assert_eq!(u.info[0], 1);
        assert_eq!(u.lights[0].pos_radius[0], 1.0);
        ctx.lights.clear();
        assert_eq!(LightUniform::from_ctx(&ctx).info, [0; 4]);
    }

    #[test]
    fn sim_time_wraps_and_keeps_quarter_second() {
        let mut ctx = test_ctx(UVec2::new(4, 4));
        ctx.epoch = uni_core::Epoch::J2000.add_seconds(1e9);
        ctx.sim_dt_s = 3600.0;
        ctx.jitter_px = glam::Vec2::new(0.25, -0.5);
        let u = CameraUniform::from_ctx(&ctx);
        let want = 1e9_f64.rem_euclid(SIM_TIME_WRAP_S);
        assert_eq!(want, 1_755_648.0);
        assert!((f64::from(u.sim[0]) - want).abs() <= 0.25, "{}", u.sim[0]);
        assert_eq!(&u.sim[1..], &[3600.0, 0.25, -0.5]);

        // Before J2000 it still lands in [0, wrap): 1 s before is wrap − 1.
        ctx.epoch = uni_core::Epoch::J2000.add_seconds(-1.0);
        let u = CameraUniform::from_ctx(&ctx);
        assert_eq!(f64::from(u.sim[0]), SIM_TIME_WRAP_S - 1.0);

        // Far from J2000 (about 10 million years) the quarter second is kept.
        let far = 3.2e14 + 0.75;
        ctx.epoch = uni_core::Epoch::J2000.add_seconds(far);
        let u = CameraUniform::from_ctx(&ctx);
        let want = far.rem_euclid(SIM_TIME_WRAP_S);
        assert!(
            (f64::from(u.sim[0]) - want).abs() <= 0.25,
            "{} vs {want}",
            u.sim[0]
        );
        // And 0.5 s later the wrapped time moves by 0.5 ± 0.25 s, not by 0 or 1.
        ctx.epoch = ctx.epoch.add_seconds(0.5);
        let later = CameraUniform::from_ctx(&ctx);
        let step = later.sim[0] - u.sim[0];
        assert!((step - 0.5).abs() <= 0.25, "{step}");
    }

    #[test]
    fn cam_delta_is_f64_difference() {
        let mut history = CameraHistory::default();
        let mut ctx = test_ctx(UVec2::new(4, 4));
        let gpc = 1e9 * uni_core::PC;
        ctx.camera.pos = uni_core::UniversePos::from_meters(DVec3::new(gpc, -gpc, 0.0));
        let first = history.uniform(&ctx);
        let first_view_proj = (ctx.proj * ctx.view_rot).to_cols_array_2d();
        // First frame: no motion, and the previous view_proj is the current one.
        assert_eq!(first.cam_delta, [0.0; 4]);
        assert_eq!(first.prev_view_proj, first_view_proj);

        ctx.camera.pos = ctx.camera.pos.offset(DVec3::new(1.5, 0.0, 0.0));
        ctx.view_rot = Mat4::from_rotation_y(0.01) * ctx.view_rot;
        let second = history.uniform(&ctx);
        let d = Vec4::from_array(second.cam_delta);
        assert!(
            (d - Vec4::new(1.5, 0.0, 0.0, 0.0)).abs().max_element() < 1e-3,
            "{d}"
        );
        assert_eq!(second.prev_view_proj, first_view_proj);
        assert_ne!(
            second.prev_view_proj,
            (ctx.proj * ctx.view_rot).to_cols_array_2d()
        );
    }

    #[test]
    fn exposure_wgsl_has_no_bindings() {
        let lib = ShaderLib::with_common();
        let src = lib.compose("common/exposure.wgsl").unwrap();
        let module = validate_wgsl(&src).unwrap();
        assert!(
            module
                .global_variables
                .iter()
                .all(|(_, v)| v.binding.is_none()),
            "{src}"
        );
        assert!(
            module
                .constants
                .iter()
                .any(|(_, c)| c.name.as_deref() == Some("HDR_MAX"))
        );
        assert!(
            module
                .functions
                .iter()
                .any(|(_, f)| f.name.as_deref() == Some("exposure_scale"))
        );
        assert!(
            module
                .types
                .iter()
                .any(|(_, t)| t.name.as_deref() == Some("ExposureUniform"))
        );
    }

    #[test]
    fn hdr_max_matches_wgsl_and_fits_f16() {
        let lib = ShaderLib::with_common();
        let module = validate_wgsl(&lib.compose("common/camera.wgsl").unwrap()).unwrap();
        let (_, c) = module
            .constants
            .iter()
            .find(|(_, c)| c.name.as_deref() == Some("HDR_MAX"))
            .expect("const HDR_MAX in camera.wgsl");
        let naga::Expression::Literal(naga::Literal::F32(v)) = module.global_expressions[c.init]
        else {
            panic!("HDR_MAX is not an f32 literal");
        };
        assert_eq!(v, crate::HDR_MAX);
        // IEEE 754 binary16's largest finite value.
        assert!(crate::HDR_MAX < 65504.0);
    }

    /// Review I2: pre-exposed HDR keeps the brightest M1 content inside f16 at the exposures
    /// it is seen at, where absolute luminance would overflow.
    #[test]
    fn pre_exposure_keeps_the_sun_inside_f16() {
        // The Sun's disk: ~1.6e9 cd/m² (mean visible radiance; design review I2).
        let sun_cd_m2 = 1.6e9_f32;
        assert!(sun_cd_m2 > 65504.0);
        // Sunlit planet exposure (EV100 15, "sunny 16"): the disk itself fits in f16.
        let at_ev15 = sun_cd_m2 * exposure_scale(15.0);
        assert!(at_ev15 < crate::HDR_MAX, "{at_ev15}");
        // Exposure scale is the Frostbite form 1/(1.2·2^EV).
        assert!((exposure_scale(0.0) - 1.0 / 1.2).abs() < 1e-7);
        assert!((exposure_scale(-4.0) - 16.0 / 1.2).abs() < 1e-5);
    }

    #[test]
    fn exposure_uniform_packs_both_pre_exposures() {
        let e = Exposure {
            scene_ev100: 3.0,
            star_ev100: -2.0,
        };
        let u = ExposureUniform::from_exposure(e);
        assert_eq!(u.pre, [exposure_scale(3.0), exposure_scale(-2.0), 0.0, 0.0]);
        assert_eq!(u.ev100, [3.0, -2.0, 0.0, 0.0]);
        assert!(e.eye_adapted_stars());
        assert!(
            !Exposure {
                scene_ev100: 1.5,
                star_ev100: 1.5
            }
            .eye_adapted_stars()
        );
    }

    /// Camera→root rotation used by the tests: yawed and pitched so no axis is trivial.
    fn orient() -> DQuat {
        DQuat::from_rotation_y(0.7) * DQuat::from_rotation_x(-0.3)
    }

    fn test_ctx(viewport: UVec2) -> FrameCtx {
        let fov_y = 60f32.to_radians();
        let aspect = viewport.x as f32 / viewport.y as f32;
        FrameCtx {
            frame_index: 7,
            epoch: uni_core::Epoch::J2000,
            dt_real_s: 1.0 / 60.0,
            camera: uni_core::CameraView {
                pos: uni_core::UniversePos::default(),
                orient: orient(),
                fov_y_rad: f64::from(fov_y),
                velocity_mps: DVec3::ZERO,
            },
            view_rot: Mat4::from_quat(orient().as_quat().inverse()),
            proj: reversed_z_infinite_proj(fov_y, aspect, 0.1),
            viewport,
            output_size: viewport * 2,
            exposure: Exposure {
                scene_ev100: 3.0,
                star_ev100: -2.0,
            },
            lenses: Vec::new(),
            lights: Vec::new(),
            sim_dt_s: 0.0,
            jitter_px: glam::Vec2::ZERO,
        }
    }

    /// CPU copy of `pixel_dir` in camera.wgsl (same NDC convention).
    fn pixel_dir(u: &CameraUniform, pix: glam::Vec2) -> Vec3 {
        let uv = (pix + 0.5) * glam::Vec2::new(u.viewport[2], u.viewport[3]);
        let ndc = Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
        let p = Mat4::from_cols_array_2d(&u.inv_view_proj) * ndc;
        (p.truncate() / p.w).normalize()
    }

    #[test]
    fn from_ctx_packs_fields() {
        let ctx = test_ctx(UVec2::new(101, 61));
        let u = CameraUniform::from_ctx(&ctx);
        assert_eq!(u.view_rot, ctx.view_rot.to_cols_array_2d());
        assert_eq!(u.proj, ctx.proj.to_cols_array_2d());
        assert_eq!(u.viewport, [101.0, 61.0, 1.0 / 101.0, 1.0 / 61.0]);
        assert_eq!(u.output, [202.0, 122.0, 1.0 / 202.0, 1.0 / 122.0]);
        assert_eq!(u.time[0], 7.0);
        assert!((u.time[1] - 1.0 / 60.0).abs() < 1e-7);
        assert!((u.time[2] - 60f32.to_radians()).abs() < 1e-6);
        assert!((u.time[3] - 101.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn pixel_dir_convention_centre_forward_row0_top() {
        let ctx = test_ctx(UVec2::new(101, 61));
        let u = CameraUniform::from_ctx(&ctx);
        let forward = orient().as_quat() * Vec3::NEG_Z;

        // The centre pixel of an odd-sized viewport looks straight ahead, in root axes.
        let c = pixel_dir(&u, glam::Vec2::new(50.0, 30.0));
        assert!((c - forward).length() < 1e-5, "{c} vs {forward}");

        // Row 0 is the top of the image, column 0 the left; check the exact elevation angle.
        let to_cam = |d: Vec3| ctx.view_rot.transform_vector3(d);
        let top = to_cam(pixel_dir(&u, glam::Vec2::new(50.0, 0.0)));
        let ndc_y = 1.0 - 2.0 * (0.5 / 61.0);
        let want = (ndc_y * (30f32.to_radians()).tan()).atan();
        assert!((top.y.atan2(-top.z) - want).abs() < 1e-5, "top = {top}");
        assert!(top.x.abs() < 1e-5);
        let left = to_cam(pixel_dir(&u, glam::Vec2::new(0.0, 30.0)));
        assert!(
            left.x < 0.0 && left.y.abs() < 1e-5 && left.z < 0.0,
            "left = {left}"
        );
    }

    #[test]
    fn frame_index_wraps_before_losing_f32_precision() {
        let mut ctx = test_ctx(UVec2::new(4, 4));
        ctx.frame_index = (1 << 24) + 5;
        assert_eq!(CameraUniform::from_ctx(&ctx).time[0], 5.0);
    }
}
