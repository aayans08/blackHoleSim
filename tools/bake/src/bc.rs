//! Block compression for the surface maps (M1 Task A8): BC7 albedo and BC4 single-channel maps
//! through `intel_tex_2` (Intel ISPC Texture Compressor, MIT OR Apache-2.0), and the bake-side
//! BC4 decoder for the PSNR test. `intel_tex_2` linked and ran on the M3 (A0b Step 1 and A8
//! Step 1), so the own-encoder BC1 path is not built.
//!
//! Both encoders split the image into horizontal strips of whole block rows and compress the
//! strips in parallel with rayon; the output is the usual BC layout (rows of blocks top to bottom,
//! blocks left to right), so strip outputs concatenate.

use anyhow::{Result, bail};
use rayon::prelude::*;

/// Checks that a `width` × `height` image with `channels` bytes per texel is block-aligned and
/// that `data` holds exactly that many bytes.
fn check_surface(data: &[u8], width: u32, height: u32, channels: usize) -> Result<()> {
    if width == 0 || height == 0 || width % 4 != 0 || height % 4 != 0 {
        bail!("BC encoding needs a non-empty size that is a multiple of 4, got {width}×{height}");
    }
    let expected = width as usize * height as usize * channels;
    if data.len() != expected {
        bail!(
            "{width}×{height}×{channels} image needs {expected} B, got {}",
            data.len()
        );
    }
    Ok(())
}

/// BC7-encodes an sRGB RGBA8 image (`width`, `height` multiples of 4) with `intel_tex_2`'s
/// `opaque_basic` preset (A8 Step 1: 29.5 s for an 8192×4096 mip 0 on the M3's 8 cores; the
/// `slow` preset was not needed). The GPU decodes it as `Bc7RgbaUnormSrgb`.
pub fn encode_bc7_srgb(rgba: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    check_surface(rgba, width, height, 4)?;
    let settings = intel_tex_2::bc7::opaque_basic_settings();
    Ok(encode_strips(
        rgba,
        width,
        height,
        4,
        16,
        |strip, rows, out| {
            let surface = intel_tex_2::RgbaSurface {
                data: strip,
                width,
                height: rows,
                stride: width * 4,
            };
            intel_tex_2::bc7::compress_blocks_into(&settings, &surface, out);
        },
    ))
}

/// BC4-encodes a single-channel image (`width`, `height` multiples of 4).
pub fn encode_bc4(r: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    check_surface(r, width, height, 1)?;
    Ok(encode_strips(r, width, height, 1, 8, |strip, rows, out| {
        let surface = intel_tex_2::RSurface {
            data: strip,
            width,
            height: rows,
            stride: width,
        };
        intel_tex_2::bc4::compress_blocks_into(&surface, out);
    }))
}

/// Block rows per parallel strip (64 texel rows).
const STRIP_BLOCK_ROWS: usize = 16;

/// Runs `encode(strip texels, strip height, strip blocks)` over strips of whole block rows in
/// parallel and returns the concatenated blocks.
fn encode_strips(
    data: &[u8],
    width: u32,
    height: u32,
    texel_bytes: usize,
    block_bytes: usize,
    encode: impl Fn(&[u8], u32, &mut [u8]) + Sync,
) -> Vec<u8> {
    let blocks_x = width as usize / 4;
    let block_rows = height as usize / 4;
    let mut out = vec![0_u8; blocks_x * block_rows * block_bytes];
    let strip_out = STRIP_BLOCK_ROWS * blocks_x * block_bytes;
    let strip_in = STRIP_BLOCK_ROWS * 4 * width as usize * texel_bytes;
    out.par_chunks_mut(strip_out)
        .zip(data.par_chunks(strip_in))
        .for_each(|(blocks, texels)| {
            let rows = (blocks.len() / (blocks_x * block_bytes) * 4) as u32;
            encode(texels, rows, blocks);
        });
    out
}

/// Decodes BC4 (`Bc4RUnorm`) blocks to 8-bit texels, as a GPU does, rounded to the nearest code.
///
/// Block layout (Microsoft, "BC4" in the Direct3D 11 block-compression reference): red_0, red_1,
/// then sixteen 3-bit indices, little-endian, texels in row order. red_0 > red_1: indices 2–7
/// interpolate six values, ((7 − k)·red_0 + k·red_1)/7 for index k + 1; otherwise indices 2–5
/// interpolate four, ((5 − k)·red_0 + k·red_1)/5, index 6 is 0 and index 7 is 255.
/// Test-only: the runtime has no BC decoder (review I17), and the bake does not decode.
#[cfg(test)]
pub fn decode_bc4(blocks: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
    if width % 4 != 0 || height % 4 != 0 {
        bail!("BC4 decoding needs a size that is a multiple of 4, got {width}×{height}");
    }
    let (blocks_x, w) = (width as usize / 4, width as usize);
    if blocks.len() != blocks_x * (height as usize / 4) * 8 {
        bail!(
            "{width}×{height} BC4 needs {} B of blocks, got {}",
            w * height as usize / 2,
            blocks.len()
        );
    }
    let mut out = vec![0_u8; w * height as usize];
    for (b, block) in blocks.chunks_exact(8).enumerate() {
        let (r0, r1) = (f32::from(block[0]), f32::from(block[1]));
        let palette: [f32; 8] = if block[0] > block[1] {
            std::array::from_fn(|i| match i {
                0 => r0,
                1 => r1,
                _ => ((8 - i) as f32 * r0 + (i - 1) as f32 * r1) / 7.0,
            })
        } else {
            std::array::from_fn(|i| match i {
                0 => r0,
                1 => r1,
                6 => 0.0,
                7 => 255.0,
                _ => ((6 - i) as f32 * r0 + (i - 1) as f32 * r1) / 5.0,
            })
        };
        let mut bits = [0_u8; 8];
        bits[..6].copy_from_slice(&block[2..]);
        let indices = u64::from_le_bytes(bits);
        let (bx, by) = (b % blocks_x, b / blocks_x);
        for t in 0..16 {
            let index = ((indices >> (3 * t)) & 7) as usize;
            out[(4 * by + t / 4) * w + 4 * bx + t % 4] = palette[index].round() as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// PSNR [dB] of `b` against `a` (8-bit, peak 255).
    fn psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mse = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (f64::from(x) - f64::from(y)).powi(2))
            .sum::<f64>()
            / a.len() as f64;
        if mse == 0.0 {
            f64::INFINITY
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        }
    }

    /// A 256×128 smooth single-channel image: low-frequency waves across the full 0–255 range,
    /// like a cloud-coverage map at its coarser mips.
    fn smooth_image() -> (Vec<u8>, u32, u32) {
        let (w, h) = (256_u32, 128_u32);
        let data = (0..h)
            .flat_map(|y| {
                (0..w).map(move |x| {
                    let (u, v) = (f64::from(x) / f64::from(w), f64::from(y) / f64::from(h));
                    let t = 0.5
                        + 0.3 * (std::f64::consts::TAU * 2.0 * u).sin()
                        + 0.15 * (std::f64::consts::TAU * 3.0 * v + 1.0).cos()
                        + 0.05 * (std::f64::consts::TAU * 5.0 * (u + v)).sin();
                    (t * 255.0).round().clamp(0.0, 255.0) as u8
                })
            })
            .collect();
        (data, w, h)
    }

    #[test]
    fn bc4_roundtrip_psnr() {
        let (image, w, h) = smooth_image();
        let blocks = encode_bc4(&image, w, h).expect("encode");
        assert_eq!(blocks.len(), (w / 4 * (h / 4) * 8) as usize);
        let decoded = decode_bc4(&blocks, w, h).expect("decode");
        let db = psnr(&image, &decoded);
        eprintln!("BC4 PSNR on the smooth test image: {db:.2} dB");
        assert!(db >= 42.0, "BC4 PSNR {db:.2} dB < 42 dB");
    }

    /// On the real bake (skipped where the git-ignored maps are not baked): Earth's BC4 cloud mip
    /// of width 2048, decoded, against the uncompressed R8 fallback, which holds the same mip.
    /// Real clouds have sharp edges, so the bar is B1's 38 dB for maps (`bc_matches_fallback`),
    /// not the 42 dB of the smooth test image (measured on the M3: 40.22 dB).
    #[test]
    fn baked_earth_clouds_bc4_psnr() {
        let dir = uni_data::data_dir().join("baked/maps");
        let (Ok(bc), Ok(r8)) = (
            uni_data::load_tex(&dir.join("earth_clouds.big")),
            uni_data::load_tex(&dir.join("earth_clouds.r2k.big")),
        ) else {
            eprintln!("earth_clouds not baked here: skipped");
            return;
        };
        let level = (0..bc.mips.len())
            .find(|&l| bc.mip_size(l).0 == r8.width)
            .expect("2k mip");
        let decoded = decode_bc4(&bc.mips[level], r8.width, r8.height).expect("decode");
        let db = psnr(&r8.mips[0], &decoded);
        eprintln!("Earth clouds BC4 mip {level} vs R8 fallback: {db:.2} dB");
        assert!(db >= 38.0, "{db:.2} dB");
    }

    #[test]
    fn bc4_solid_blocks_are_exact() {
        for value in [0_u8, 1, 77, 128, 254, 255] {
            let image = vec![value; 8 * 4];
            let blocks = encode_bc4(&image, 8, 4).expect("encode");
            assert_eq!(decode_bc4(&blocks, 8, 4).expect("decode"), image, "{value}");
        }
    }

    /// Both BC4 modes (Microsoft's BC4 reference: r0 > r1 → 6 interpolated values; r0 ≤ r1 → 4
    /// interpolated values plus 0 and 255), with every 3-bit index used once.
    #[test]
    fn decode_bc4_both_modes() {
        // Indices 0..8 in texel order, packed little-endian, 3 bits each.
        let indices: u64 = (0..16_u64).map(|i| (i % 8) << (3 * i)).sum();
        let block = |r0: u8, r1: u8| {
            let mut b = vec![r0, r1];
            b.extend_from_slice(&indices.to_le_bytes()[..6]);
            b
        };
        let eight = decode_bc4(&block(210, 70), 4, 4).expect("decode");
        assert_eq!(&eight[..8], &[210, 70, 190, 170, 150, 130, 110, 90]);
        let six = decode_bc4(&block(50, 200), 4, 4).expect("decode");
        assert_eq!(&six[..8], &[50, 200, 80, 110, 140, 170, 0, 255]);
    }

    #[test]
    fn bc7_encodes_whole_blocks() {
        let (w, h) = (16_u32, 8_u32);
        let rgba: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i * 3) as u8, 90, 200, 255])
            .collect();
        let blocks = encode_bc7_srgb(&rgba, w, h).expect("encode");
        assert_eq!(blocks.len(), (w / 4 * (h / 4) * 16) as usize);
        assert!(blocks.iter().any(|&b| b != 0));
    }

    #[test]
    fn encoders_reject_unaligned_or_short_input() {
        assert!(encode_bc4(&[0; 6 * 4], 6, 4).is_err());
        assert!(encode_bc4(&[0; 15], 4, 4).is_err());
        assert!(encode_bc7_srgb(&[0; 4 * 4 * 4], 4, 2).is_err());
        assert!(decode_bc4(&[0; 7], 4, 4).is_err());
    }

    /// A0b Step 1: `intel_tex_2`'s packaged aarch64-apple-darwin ISPC kernels link into this
    /// binary and run (a solid block encodes to a non-zero block of the right size).
    #[test]
    fn intel_tex_2_kernels_link_and_run() {
        let rgba = [200_u8, 100, 50, 255].repeat(16);
        let surface = intel_tex_2::RgbaSurface {
            data: &rgba,
            width: 4,
            height: 4,
            stride: 16,
        };
        let bc7 = intel_tex_2::bc7::compress_blocks(
            &intel_tex_2::bc7::opaque_ultra_fast_settings(),
            &surface,
        );
        assert_eq!(bc7.len(), 16);
        assert!(bc7.iter().any(|&b| b != 0), "BC7 kernel wrote nothing");

        let r = [128_u8; 16];
        let bc4 = intel_tex_2::bc4::compress_blocks(&intel_tex_2::RSurface {
            data: &r,
            width: 4,
            height: 4,
            stride: 4,
        });
        assert_eq!(bc4.len(), 8);
        assert!(bc4.iter().any(|&b| b != 0), "BC4 kernel wrote nothing");
    }
}
