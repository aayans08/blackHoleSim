//! Block compression for the surface maps (M1 Task A8): BC7 albedo and BC4 single-channel maps
//! through `intel_tex_2`, and the bake-side BC4 decoder for the PSNR test. If `intel_tex_2`
//! fails to link, this module holds A8's own BC1/BC4 encoders and decoders instead.
//! A0b stub; A8 implements.

#[cfg(test)]
mod tests {
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
