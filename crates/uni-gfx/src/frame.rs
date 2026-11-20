//! Per-frame context handed to every feature, and its GPU twin `CameraUniform` (camera.wgsl).

use glam::{DVec3, Mat4, UVec2};

/// Exposure values (EV at ISO 100) for the scene and for point-like stars.
#[derive(Clone, Copy, Debug)]
pub struct Exposure {
    pub scene_ev100: f32,
    pub star_ev100: f32,
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
    pub viewport: UVec2,
    pub exposure: Exposure,
    pub lenses: Vec<LensView>,
}

/// Twin of `struct CameraUniform` in `shaders/common/camera.wgsl`;
/// bound at group(0) binding(0) by every feature.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_rot: [[f32; 4]; 4],
    pub proj: [[f32; 4]; 4],
    /// inverse(proj * view_rot): pixel → root-axes direction.
    pub inv_view_proj: [[f32; 4]; 4],
    /// w, h, 1/w, 1/h.
    pub viewport: [f32; 4],
    /// scene_ev100, star_ev100, 0, 0.
    pub exposure: [f32; 4],
    /// frame_index as f32 (modulo 2^24), dt_real_s, fov_y, aspect.
    pub time: [f32; 4],
}

const _: () = assert!(core::mem::size_of::<CameraUniform>() == 240);

/// `frame_index` is sent modulo 2^24, the largest range in which every integer is exact in f32.
const FRAME_INDEX_WRAP: u64 = 1 << 24;

impl CameraUniform {
    /// Packs the frame context for the GPU.
    ///
    /// A zero viewport axis is treated as 1 so `1/w`, `1/h` and the aspect stay finite.
    /// `frame_index` wraps modulo 2^24 so it keeps counting exactly in f32.
    pub fn from_ctx(ctx: &FrameCtx) -> Self {
        let w = ctx.viewport.x.max(1) as f32;
        let h = ctx.viewport.y.max(1) as f32;
        Self {
            view_rot: ctx.view_rot.to_cols_array_2d(),
            proj: ctx.proj.to_cols_array_2d(),
            inv_view_proj: (ctx.proj * ctx.view_rot).inverse().to_cols_array_2d(),
            viewport: [w, h, 1.0 / w, 1.0 / h],
            exposure: [ctx.exposure.scene_ev100, ctx.exposure.star_ev100, 0.0, 0.0],
            time: [
                (ctx.frame_index % FRAME_INDEX_WRAP) as f32,
                ctx.dt_real_s as f32,
                ctx.camera.fov_y_rad as f32,
                w / h,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proj::reversed_z_infinite_proj;
    use crate::shader::{ShaderLib, validate_wgsl};
    use core::mem::{offset_of, size_of};
    use glam::{DQuat, Vec3, Vec4};

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

        assert_eq!(size_of::<CameraUniform>(), 240);
        assert_eq!(layouter[handle].size as usize, size_of::<CameraUniform>());

        let wgsl: Vec<(String, usize)> = members
            .iter()
            .map(|m| (m.name.clone().unwrap_or_default(), m.offset as usize))
            .collect();
        let rust: Vec<(String, usize)> = [
            ("view_rot", offset_of!(CameraUniform, view_rot)),
            ("proj", offset_of!(CameraUniform, proj)),
            ("inv_view_proj", offset_of!(CameraUniform, inv_view_proj)),
            ("viewport", offset_of!(CameraUniform, viewport)),
            ("exposure", offset_of!(CameraUniform, exposure)),
            ("time", offset_of!(CameraUniform, time)),
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
            },
            view_rot: Mat4::from_quat(orient().as_quat().inverse()),
            proj: reversed_z_infinite_proj(fov_y, aspect, 0.1),
            viewport,
            exposure: Exposure {
                scene_ev100: 3.0,
                star_ev100: -2.0,
            },
            lenses: Vec::new(),
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
        assert_eq!(u.exposure, [3.0, -2.0, 0.0, 0.0]);
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
