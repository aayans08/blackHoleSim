//! Per-frame render targets: HDR colour, reversed-Z depth and the lens-capture mask.

use glam::UVec2;

/// Scene-linear HDR colour.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Reversed-Z depth (near = 1, far = 0; compare `Greater`).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Lens mask: 1.0 where the ray was captured by a black hole.
pub const LENS_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// The textures every feature renders into.
pub struct FrameTargets {
    /// Never 0 in either axis.
    pub size: UVec2,
    /// `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`.
    pub hdr: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// 1.0 = ray captured.
    pub lens_mask: wgpu::Texture,
    pub lens_mask_view: wgpu::TextureView,
}

impl FrameTargets {
    /// Allocates all targets at `size`.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        todo!()
    }
}
