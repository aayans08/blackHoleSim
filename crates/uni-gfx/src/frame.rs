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
    /// frame_index as f32, dt_real_s, fov_y, aspect.
    pub time: [f32; 4],
}

impl CameraUniform {
    /// Packs the frame context for the GPU.
    pub fn from_ctx(ctx: &FrameCtx) -> Self {
        todo!()
    }
}
