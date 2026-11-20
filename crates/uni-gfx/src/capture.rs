//! Headless capture: render into an sRGB texture and read it back as RGBA8 or PNG.

use anyhow::Context as _;
use glam::UVec2;

use crate::context::GpuContext;
use crate::targets::FrameTargets;

/// Format of capture images (8-bit sRGB).
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Bytes per texel of [`CAPTURE_FORMAT`].
const BYTES_PER_TEXEL: u32 = 4;

/// An offscreen colour target that can be read back.
pub struct CaptureTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// Never 0 in either axis.
    pub size: UVec2,
}

impl CaptureTarget {
    /// Allocates a `CAPTURE_FORMAT` texture with `RENDER_ATTACHMENT | COPY_SRC`.
    /// A zero axis is clamped to 1, as for [`FrameTargets`].
    pub fn new(device: &wgpu::Device, size: UVec2) -> Self {
        let size = FrameTargets::clamp_size(size);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("capture"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: CAPTURE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            texture,
            view,
            size,
        }
    }

    /// Blocking read-back of tightly packed RGBA8 rows (row padding stripped).
    ///
    /// Submits a texture→buffer copy after any work already submitted (the queue runs in
    /// order), then waits for the GPU. The bytes are sRGB-encoded, top row first.
    pub fn read_rgba8(&self, gpu: &GpuContext) -> anyhow::Result<Vec<u8>> {
        let (w, h) = (self.size.x, self.size.y);
        let padded_row = padded_bytes_per_row(w);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("capture read-back"),
            size: u64::from(padded_row) * u64::from(h),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("capture read-back"),
            });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        gpu.queue.submit(std::iter::once(enc.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            // The receiver outlives the wait below; a send error cannot be acted on here.
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the capture copy")?;
        // The wait above runs the callback; `try_recv` turns a missing callback into an
        // error instead of blocking forever.
        rx.try_recv()
            .context("capture map callback did not run")?
            .context("mapping the capture buffer")?;

        let pixels = {
            let view = buffer
                .get_mapped_range(..)
                .context("reading the mapped capture buffer")?;
            strip_row_padding(&view, w, h, padded_row)
        };
        buffer.unmap();
        Ok(pixels)
    }

    /// Reads back and writes a PNG to `path`.
    pub fn save_png(&self, gpu: &GpuContext, path: &std::path::Path) -> anyhow::Result<()> {
        let pixels = self.read_rgba8(gpu)?;
        let image = image::RgbaImage::from_raw(self.size.x, self.size.y, pixels)
            .context("capture read-back has the wrong length for its size")?;
        image
            .save_with_format(path, image::ImageFormat::Png)
            .with_context(|| format!("writing {}", path.display()))
    }
}

/// Bytes per row of a tightly packed RGBA8 image of `width` texels, rounded up to
/// `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256) as texture→buffer copies require.
pub(crate) fn padded_bytes_per_row(width: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (width * BYTES_PER_TEXEL).div_ceil(align) * align
}

/// Copies `height` rows of `width * 4` bytes out of rows that are `padded_row` bytes apart.
pub(crate) fn strip_row_padding(
    padded: &[u8],
    width: u32,
    height: u32,
    padded_row: u32,
) -> Vec<u8> {
    let row = (width * BYTES_PER_TEXEL) as usize;
    let mut out = Vec::with_capacity(row * height as usize);
    for r in padded.chunks(padded_row as usize).take(height as usize) {
        out.extend_from_slice(&r[..row.min(r.len())]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_row_rounds_up_to_256() {
        assert_eq!(padded_bytes_per_row(1), 256);
        assert_eq!(padded_bytes_per_row(64), 256);
        assert_eq!(padded_bytes_per_row(65), 512);
        assert_eq!(padded_bytes_per_row(1920), 7680); // already a multiple of 256
        assert_eq!(padded_bytes_per_row(1921), 7936);
    }

    #[test]
    fn strip_row_padding_keeps_only_pixels() {
        let (w, h, row) = (3u32, 2u32, 256u32);
        let mut padded = vec![0xEEu8; (row * h) as usize];
        for y in 0..h {
            for x in 0..w * 4 {
                padded[(y * row + x) as usize] = (y * 100 + x) as u8;
            }
        }
        let tight = strip_row_padding(&padded, w, h, row);
        assert_eq!(tight.len(), (w * h * 4) as usize);
        assert_eq!(&tight[..12], &(0u8..12).collect::<Vec<_>>()[..]);
        assert_eq!(&tight[12..], &(100u8..112).collect::<Vec<_>>()[..]);
    }
}
