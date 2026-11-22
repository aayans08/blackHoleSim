//! Headless capture: render into an sRGB texture and read it back as RGBA8 or PNG; read the
//! pre-exposed `hdr` target back as f32 for photometric checks (the `--hdr` sidecar).

use anyhow::Context as _;
use glam::UVec2;

use crate::context::GpuContext;
use crate::targets::FrameTargets;

/// Format of capture images (8-bit sRGB).
pub const CAPTURE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

/// Bytes per texel of [`CAPTURE_FORMAT`].
const BYTES_PER_TEXEL: u32 = 4;
/// Bytes per texel of `hdr` (`Rgba16Float`).
const HDR_BYTES_PER_TEXEL: u32 = 8;

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
        read_texture(gpu, &self.texture, self.size, BYTES_PER_TEXEL)
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

/// Blocking read-back of the pre-exposed `hdr` target as RGBA f32, rows top-down, 4 floats per
/// texel (light × exposure, exactly as the tonemap reads it). The f16 texels are copied out and
/// converted on the CPU. Like [`CaptureTarget::read_rgba8`], it reads after any work already
/// submitted.
pub fn read_hdr(gpu: &GpuContext, targets: &FrameTargets) -> anyhow::Result<Vec<f32>> {
    let bytes = read_texture(gpu, &targets.hdr, targets.size, HDR_BYTES_PER_TEXEL)?;
    // Texel copies keep each component's little-endian byte order.
    Ok(bytes
        .chunks_exact(2)
        .map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]])))
        .collect())
}

/// Blocking copy of `texture` (mip 0, layer 0, `size` texels of `bytes_per_texel` bytes) into
/// tightly packed rows, top row first. Submits after any work already submitted (the queue runs
/// in order), then waits for the GPU. The texture needs `COPY_SRC`.
fn read_texture(
    gpu: &GpuContext,
    texture: &wgpu::Texture,
    size: UVec2,
    bytes_per_texel: u32,
) -> anyhow::Result<Vec<u8>> {
    let (w, h) = (size.x, size.y);
    let padded_row = padded_bytes_per_row(w, bytes_per_texel);
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
            texture,
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
        strip_row_padding(&view, w * bytes_per_texel, h, padded_row)
    };
    buffer.unmap();
    Ok(pixels)
}

/// Bytes per row of a tightly packed image of `width` texels of `bytes_per_texel` bytes,
/// rounded up to `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256) as texture→buffer copies require.
pub(crate) fn padded_bytes_per_row(width: u32, bytes_per_texel: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (width * bytes_per_texel).div_ceil(align) * align
}

/// Copies `height` rows of `row_bytes` bytes out of rows that are `padded_row` bytes apart.
pub(crate) fn strip_row_padding(
    padded: &[u8],
    row_bytes: u32,
    height: u32,
    padded_row: u32,
) -> Vec<u8> {
    let row = row_bytes as usize;
    let mut out = Vec::with_capacity(row * height as usize);
    for r in padded.chunks(padded_row as usize).take(height as usize) {
        out.extend_from_slice(&r[..row.min(r.len())]);
    }
    out
}

/// An IEEE 754 binary16 value (bits) as f32, exactly: every f16 is representable in f32.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let mant = u32::from(h & 0x3ff);
    let magnitude = match exp {
        // Zero and subnormals: mant × 2^-24, exact in f32.
        0 => mant as f32 * f32::from_bits(0x3380_0000),
        // Infinity (mant 0) and NaN (payload kept in the high mantissa bits).
        0x1f => f32::from_bits(0x7f80_0000 | (mant << 13)),
        // Normal: rebias the exponent from 15 to 127, widen the mantissa from 10 to 23 bits.
        _ => f32::from_bits(((exp + 127 - 15) << 23) | (mant << 13)),
    };
    f32::from_bits(magnitude.to_bits() | sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_row_rounds_up_to_256() {
        assert_eq!(padded_bytes_per_row(1, 4), 256);
        assert_eq!(padded_bytes_per_row(64, 4), 256);
        assert_eq!(padded_bytes_per_row(65, 4), 512);
        assert_eq!(padded_bytes_per_row(1920, 4), 7680); // already a multiple of 256
        assert_eq!(padded_bytes_per_row(1921, 4), 7936);
        // Rgba16Float hdr: 8 bytes per texel.
        assert_eq!(padded_bytes_per_row(32, 8), 256);
        assert_eq!(padded_bytes_per_row(33, 8), 512);
        assert_eq!(padded_bytes_per_row(1920, 8), 15_360);
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
        let tight = strip_row_padding(&padded, w * 4, h, row);
        assert_eq!(tight.len(), (w * h * 4) as usize);
        assert_eq!(&tight[..12], &(0u8..12).collect::<Vec<_>>()[..]);
        assert_eq!(&tight[12..], &(100u8..112).collect::<Vec<_>>()[..]);
    }

    #[test]
    fn f16_to_f32_known_values() {
        assert_eq!(f16_to_f32(0x0000).to_bits(), 0.0_f32.to_bits());
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0_f32).to_bits());
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x7bff), 65504.0); // largest finite f16
        assert_eq!(f16_to_f32(0x0400), 2_f32.powi(-14)); // smallest normal
        assert_eq!(f16_to_f32(0x0001), 2_f32.powi(-24)); // smallest subnormal
        assert_eq!(f16_to_f32(0x83ff), -1023.0 * 2_f32.powi(-24)); // largest subnormal
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan() && f16_to_f32(0xfd01).is_nan());
    }

    /// Every finite f16 against the IEEE 754 binary16 definition, evaluated in f64.
    #[test]
    fn f16_to_f32_exhaustive() {
        for h in 0..=u16::MAX {
            let (sign, exp, mant) = (h >> 15, i32::from((h >> 10) & 0x1f), f64::from(h & 0x3ff));
            if exp == 0x1f {
                continue; // infinities and NaNs: f16_to_f32_known_values
            }
            let magnitude = if exp == 0 {
                mant * 2_f64.powi(-24)
            } else {
                (1.0 + mant / 1024.0) * 2_f64.powi(exp - 15)
            };
            let want = if sign == 1 { -magnitude } else { magnitude };
            assert_eq!(f64::from(f16_to_f32(h)), want, "{h:#06x}");
        }
    }
}
