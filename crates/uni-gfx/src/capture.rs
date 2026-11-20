//! Headless capture: render into an sRGB texture and read it back as RGBA8 or PNG.

use glam::UVec2;

use crate::context::GpuContext;

/// Format of capture images (8-bit sRGB).
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// An offscreen colour target that can be read back.
pub struct CaptureTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: UVec2,
}

impl CaptureTarget {
    /// Allocates a `CAPTURE_FORMAT` texture with `RENDER_ATTACHMENT | COPY_SRC`.
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        todo!()
    }

    /// Blocking read-back of tightly packed RGBA8 rows (row padding stripped).
    pub fn read_rgba8(&self, gpu: &GpuContext) -> anyhow::Result<Vec<u8>> {
        todo!()
    }

    /// Reads back and writes a PNG to `path`.
    pub fn save_png(&self, gpu: &GpuContext, path: &std::path::Path) -> anyhow::Result<()> {
        todo!()
    }
}
