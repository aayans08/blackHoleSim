//! Per-frame render targets: HDR colour, reversed-Z depth and the lens-capture mask, all at
//! the internal render resolution.
//!
//! # The background / lens / star contract (design §5.3)
//!
//! Features run in a fixed order, and each knows which pixels it owns:
//! 1. **Background** (no depth): writes every pixel it covers, pre-exposed (see `hdr`). In M0
//!    that is uni-rel's debug grid, drawn only while a lens is traced; elsewhere the cleared
//!    black is space. M1's Milky Way band goes here.
//! 2. **Opaque** geometry (target, M1+).
//! 3. **Lens pass** (compute): overwrites `hdr` and writes `lens_mask` **only on the pixels it
//!    traces** (its screen rectangle; the whole frame from inside a horizon). Its escaped rays
//!    must show the same environment the background draws, so the seam at the rectangle's edge
//!    is invisible: in M0 both draw the debug grid. Pixels outside the rectangle keep the
//!    background untouched.
//! 4. **Stars** (additive): multiply by `1 − lens_mask`, so no star shines through a shadow.
//!    Any background-like feature that must run after the lens does the same.
//! 5. **Tonemap** (and post): reads `hdr`, writes the output view.

use glam::UVec2;

/// Pre-exposed scene-linear HDR colour (Rec. 709 primaries): light × exposure, see
/// `shaders/common/camera.wgsl`. The tonemap applies AgX to it with no further exposure.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Reversed-Z depth (near = 1, far = 0; compare `Greater`, or `GreaterEqual` for things drawn
/// exactly on the far plane, such as stars).
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Lens mask, `r` channel: the fraction of the pixel whose light from behind the lens is
/// captured by a black hole (1 = inside the shadow, 0 = escaped or not traced). Written only by
/// the lens pass, only on its traced pixels; cleared to 0 every frame. Target (M3): per-pixel
/// magnification for the star pass (design §5.3 item 6) needs a second channel, so this
/// format widens then.
pub const LENS_MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

/// Largest value any feature writes into `hdr` (below f16's largest finite value, 65504).
/// Twin of `HDR_MAX` in `shaders/common/camera.wgsl`.
pub const HDR_MAX: f32 = 60000.0;

/// The textures every feature renders into, at the internal render resolution.
pub struct FrameTargets {
    /// Internal render resolution. Never 0 in either axis.
    pub size: UVec2,
    /// Pre-exposed HDR colour. `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`.
    pub hdr: wgpu::Texture,
    pub hdr_view: wgpu::TextureView,
    /// `RENDER_ATTACHMENT | TEXTURE_BINDING`; cleared to 0.0 (infinitely far) every frame.
    pub depth: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    /// See [`LENS_MASK_FORMAT`]. `STORAGE_BINDING | TEXTURE_BINDING | RENDER_ATTACHMENT`.
    pub lens_mask: wgpu::Texture,
    pub lens_mask_view: wgpu::TextureView,
}

impl FrameTargets {
    /// Allocates all targets at `size`. Each axis is clamped to at least 1 and at most the
    /// device's `max_texture_dimension_2d`, so a minimised window or an oversized Retina
    /// surface never produces an invalid texture.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let size = Self::fit(device, size);
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
    /// The size [`FrameTargets::new`] allocates for a request of `size` on `device`: each axis
    /// clamped to 1..=`max_texture_dimension_2d`.
    pub fn fit(device: &wgpu::Device, size: UVec2) -> UVec2 {
        let max = device.limits().max_texture_dimension_2d.max(1);
        Self::clamp_size(size).min(UVec2::splat(max))
    }

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
