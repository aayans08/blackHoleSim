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
    /// `RENDER_ATTACHMENT | TEXTURE_BINDING`; cleared to 0.0 (infinitely far) every frame.
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// 1.0 = ray captured. `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`.
    pub lens_mask: wgpu::Texture,
    pub lens_mask_view: wgpu::TextureView,
}

impl FrameTargets {
    /// Allocates all targets at `size`. Each axis is clamped to at least 1 and at most the
    /// device's `max_texture_dimension_2d`, so a minimised window or an oversized Retina
    /// surface never produces an invalid texture.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let max = device.limits().max_texture_dimension_2d.max(1);
        let size = Self::clamp_size(size).min(UVec2::splat(max));
        let target = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.x,
                    height: size.y,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        // Written by compute (storage) and raster (attachment) passes, sampled by later ones.
        let shared = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT;
        let (hdr, hdr_view) = target("hdr", HDR_FORMAT, shared);
        let (depth, depth_view) = target(
            "depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let (lens_mask, lens_mask_view) = target("lens_mask", LENS_MASK_FORMAT, shared);
        Self {
            size,
            hdr,
            hdr_view,
            depth,
            depth_view,
            lens_mask,
            lens_mask_view,
        }
    }
}

impl FrameTargets {
    /// Clamps each axis to at least 1 texel, so a minimised (0×0) window still gets valid targets.
    pub(crate) fn clamp_size(size: UVec2) -> UVec2 {
        size.max(UVec2::ONE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Review Focus 1: a minimised or 0×0 window must never create a zero-sized texture.
    #[test]
    fn targets_clamp_zero_size() {
        assert_eq!(FrameTargets::clamp_size(UVec2::ZERO), UVec2::ONE);
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(0, 720)),
            UVec2::new(1, 720)
        );
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(1280, 0)),
            UVec2::new(1280, 1)
        );
        assert_eq!(
            FrameTargets::clamp_size(UVec2::new(1920, 1080)),
            UVec2::new(1920, 1080)
        );
    }
}
