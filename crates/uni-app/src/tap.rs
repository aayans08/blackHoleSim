//! HDR taps: a render feature that copies the pre-exposed `hdr` target into a read-back buffer
//! at its place in the pass order, on the frames it is armed for (captures only).
//!
//! Two taps sit in the capture stack (M1 Task B9 Step 2):
//! - **pre-bloom**, between the last sky pass and bloom (pass 15): the `--hdr` sidecar, so the
//!   photometric checks read unbloomed energy (re-review r3 minor 1: bloom otherwise moves
//!   3–4 % of a point's energy outside a star's footprint);
//! - **world**, right after the exposure meter (pass 9), before any sky light or point source
//!   is written: the world light the meter read, for the capture JSON's `lit_px`.
//!
//! Disarmed, a tap encodes nothing, so the window and `--bench` pay nothing for it.

use std::cell::RefCell;
use std::rc::Rc;

use anyhow::Context as _;
use glam::UVec2;
use uni_gfx::{FrameBindings, FrameCtx, FrameTargets, GpuContext, PassTimer, RenderFeature};

/// Bytes per texel of `hdr` (`Rgba16Float`).
const HDR_BYTES_PER_TEXEL: u32 = 8;

/// A copy of `hdr` taken during a frame: linear pre-exposed RGBA, rows top-down.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    pub size: UVec2,
    /// `size.x × size.y` texels, RGBA f32 (converted exactly from the target's f16).
    pub pixels: Vec<[f32; 4]>,
}

impl HdrImage {
    /// Texel (x, y), or `None` outside the image.
    pub fn at(&self, x: i64, y: i64) -> Option<[f32; 4]> {
        if x < 0 || y < 0 || x >= i64::from(self.size.x) || y >= i64::from(self.size.y) {
            return None;
        }
        self.pixels
            .get(y as usize * self.size.x as usize + x as usize)
            .copied()
    }

    /// Little-endian f32 RGBA bytes, rows top-down: the `.hdr.bin` sidecar layout.
    pub fn to_le_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.pixels.len() * 16);
        for p in &self.pixels {
            for c in p {
                out.extend_from_slice(&c.to_le_bytes());
            }
        }
        out
    }
}

/// What the feature and its owner share.
#[derive(Default)]
struct TapState {
    armed: bool,
    /// The read-back buffer of the last copy, its size and padded row length.
    pending: Option<(wgpu::Buffer, UVec2, u32)>,
}

/// The app's handle on a tap in the renderer.
#[derive(Clone)]
pub struct HdrTap {
    state: Rc<RefCell<TapState>>,
}

impl HdrTap {
    /// A tap and the feature to push into the renderer at its place in the pass order.
    pub fn new(name: &'static str) -> (Self, TapFeature) {
        let state = Rc::new(RefCell::new(TapState::default()));
        (
            Self {
                state: Rc::clone(&state),
            },
            TapFeature { name, state },
        )
    }

    /// Copy `hdr` on the next frame(s) rendered until disarmed.
    pub fn arm(&self, on: bool) {
        self.state.borrow_mut().armed = on;
    }

    /// Reads the last copy back (blocking: waits for the GPU) and clears it; `None` if no copy
    /// was taken since the last call.
    pub fn take(&self, gpu: &GpuContext) -> anyhow::Result<Option<HdrImage>> {
        let Some((buffer, size, padded_row)) = self.state.borrow_mut().pending.take() else {
            return Ok(None);
        };
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.map_async(wgpu::MapMode::Read, .., move |r| {
            // The receiver outlives the wait below; a send error cannot be acted on here.
            let _ = tx.send(r);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("waiting for the HDR tap")?;
        rx.try_recv()
            .context("HDR tap map callback did not run")?
            .context("mapping the HDR tap buffer")?;
        let pixels = {
            let view = buffer
                .get_mapped_range(..)
                .context("reading the mapped HDR tap buffer")?;
            let row = (size.x * HDR_BYTES_PER_TEXEL) as usize;
            let mut out = Vec::with_capacity((size.x * size.y) as usize);
            for r in view.chunks(padded_row as usize).take(size.y as usize) {
                for t in r[..row.min(r.len())].chunks_exact(HDR_BYTES_PER_TEXEL as usize) {
                    out.push(std::array::from_fn(|c| {
                        uni_data::f16_to_f32(u16::from_le_bytes([t[2 * c], t[2 * c + 1]]))
                    }));
                }
            }
            out
        };
        buffer.unmap();
        Ok(Some(HdrImage { size, pixels }))
    }
}

/// The render feature of an [`HdrTap`].
pub struct TapFeature {
    name: &'static str,
    state: Rc<RefCell<TapState>>,
}

/// Bytes per row of a texture → buffer copy, rounded up to `COPY_BYTES_PER_ROW_ALIGNMENT`.
fn padded_row(width: u32) -> u32 {
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    (width * HDR_BYTES_PER_TEXEL).div_ceil(align) * align
}

impl RenderFeature for TapFeature {
    fn name(&self) -> &'static str {
        self.name
    }

    fn prepare(&mut self, _gpu: &GpuContext, _ctx: &FrameCtx, _targets: &FrameTargets) {}

    fn encode(
        &mut self,
        gpu: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        _frame: &FrameBindings<'_>,
        targets: &FrameTargets,
        _output: &wgpu::TextureView,
        _timer: &mut PassTimer,
    ) {
        let mut state = self.state.borrow_mut();
        if !state.armed {
            return;
        }
        let size = targets.size;
        let row = padded_row(size.x);
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.name),
            size: u64::from(row) * u64::from(size.y),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &targets.hdr,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        state.pending = Some((buffer, size, row));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_rows_and_sidecar_bytes() {
        assert_eq!(padded_row(1), 256);
        assert_eq!(padded_row(32), 256);
        assert_eq!(padded_row(33), 512);
        assert_eq!(padded_row(1920), 15_360);
        let img = HdrImage {
            size: UVec2::new(2, 1),
            pixels: vec![[1.0, 2.0, 3.0, 4.0], [0.5, 0.0, -1.0, 65504.0]],
        };
        let b = img.to_le_bytes();
        assert_eq!(b.len(), 32);
        assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), 2.0);
        assert_eq!(f32::from_le_bytes(b[28..32].try_into().unwrap()), 65504.0);
        assert_eq!(img.at(1, 0), Some([0.5, 0.0, -1.0, 65504.0]));
        assert_eq!(img.at(2, 0), None);
        assert_eq!(img.at(0, -1), None);
    }
}
