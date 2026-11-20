//! Measures the black-hole shadow radius in a capture PNG (M0 Task 12).
//!
//! `uni-verify shadow --png <file> --expected-px <f64> --tol <f64>` finds the 4-connected
//! region of black pixels (luma below 2/255, see [`is_black`]) that contains the image centre
//! and reports the radius of a disc of equal area, √(area/π). The expected value comes from the
//! capture's bookmark: `bh-shadow-30M` (Task 11) puts the Schwarzschild shadow, critical impact
//! parameter b_c = 3√3 M (Synge 1966, MNRAS 131, 463), at 158.745 px at 1920×1080.
//!
//! Captures trace one ray per pixel centre (resolution scale 1.0,
//! crates/uni-rel/src/tracer.rs), so the shadow's edge is hard and its pixel count estimates
//! the disc's area without bias. On an antialiased edge the partly covered pixels are brighter
//! than the threshold, so the radius reads up to about half a pixel small.

use anyhow::Context;
use image::{ImageReader, Rgb, RgbImage};
use std::path::Path;

/// Rec. 709 luma weights for R′, G′, B′, times 10⁴ so they are exact integers summing to 10⁴:
/// E′Y = 0.2126 E′R + 0.7152 E′G + 0.0722 E′B (ITU-R BT.709-6, Part 1, item 3.2).
const REC709_LUMA_E4: [u32; 3] = [2126, 7152, 722];

/// A pixel is black when its luma is below this many 8-bit code values, i.e. below 2/255.
const BLACK_LUMA_CODES: u32 = 2;

/// Loads a PNG (any bit depth or channel layout) as 8-bit RGB; alpha is dropped.
pub fn load_rgb8(path: &Path) -> anyhow::Result<RgbImage> {
    let image = ImageReader::open(path)
        .with_context(|| format!("cannot open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("cannot read {}", path.display()))?
        .decode()
        .with_context(|| format!("cannot decode {}", path.display()))?;
    Ok(image.into_rgb8())
}

/// True when the pixel's Rec. 709 luma is below 2/255.
///
/// Luma is taken on the raw 8-bit (sRGB-encoded) code values, not on sRGB-decoded linear
/// light. Luma, unlike luminance, is by definition a weighted sum of the gamma-encoded
/// components, and on code values 2/255 is two quantisation steps: only true black, plus the
/// tonemapper's ±0.5/255 dither (crates/uni-post), falls under it. The comparison is exact
/// in integers.
fn is_black(px: &Rgb<u8>) -> bool {
    let weighted: u32 = REC709_LUMA_E4
        .iter()
        .zip(px.0)
        .map(|(w, c)| w * u32::from(c))
        .sum();
    weighted < BLACK_LUMA_CODES * 10_000
}

/// Equivalent-disc radius [px] of the black region that contains the image centre:
/// √(area/π), or 0 if the centre is not black.
pub fn shadow_radius(img: &RgbImage) -> f64 {
    (black_area_at_centre(img) as f64 / std::f64::consts::PI).sqrt()
}

/// Pixel count of the 4-connected black region that contains the image centre.
///
/// The centre is the pixel (odd size) or the 2×2 block of pixels (even size) that the
/// geometric centre lies in or touches; the fill starts from each of them that is black.
fn black_area_at_centre(img: &RgbImage) -> u64 {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return 0;
    }
    let index = |x: u32, y: u32| y as usize * w as usize + x as usize;
    let mut visited = vec![false; w as usize * h as usize];
    let mut stack = Vec::new();
    for y in [(h - 1) / 2, h / 2] {
        for x in [(w - 1) / 2, w / 2] {
            stack.push((x, y));
        }
    }
    let mut area = 0;
    while let Some((x, y)) = stack.pop() {
        let i = index(x, y);
        if visited[i] {
            continue;
        }
        visited[i] = true;
        if !is_black(img.get_pixel(x, y)) {
            continue;
        }
        area += 1;
        if x > 0 {
            stack.push((x - 1, y));
        }
        if x + 1 < w {
            stack.push((x + 1, y));
        }
        if y > 0 {
            stack.push((x, y - 1));
        }
        if y + 1 < h {
            stack.push((x, y + 1));
        }
    }
    area
}

/// True when `measured` is within `tol` of `expected` [px], inclusive.
pub fn passes(measured: f64, expected: f64, tol: f64) -> bool {
    (measured - expected).abs() <= tol
}

/// The result line: `measured=<r> expected=<e> PASS|FAIL`, radii in px to 3 decimals.
pub fn report_line(measured: f64, expected: f64, pass: bool) -> String {
    let verdict = if pass { "PASS" } else { "FAIL" };
    format!("measured={measured:.3} expected={expected:.3} {verdict}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Expected shadow radius of the `bh-shadow-30M` bookmark at 1920×1080 (Task 11 brief).
    const R: f64 = 158.745;
    /// Background level: mid-grey, as an 8-bit code value.
    const GREY: f64 = 128.0;

    /// 1920×1080 grey image with a black disc of radius `r` centred on the image centre.
    /// `coverage` maps how far a pixel's centre lies inside the disc edge [px] (negative
    /// outside) to the fraction of the pixel the disc covers; the pixel blends grey toward
    /// black by that fraction.
    fn grey_with_disc(r: f64, coverage: impl Fn(f64) -> f64) -> RgbImage {
        let (w, h) = (1920, 1080);
        let (cx, cy) = (f64::from(w) / 2.0, f64::from(h) / 2.0);
        RgbImage::from_fn(w, h, |x, y| {
            let d = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
            let v = (GREY * (1.0 - coverage(r - d))).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// The usual one-pixel antialiased edge: coverage ramps from 0 to 1 across the pixel
    /// whose centre is on the edge.
    fn antialiased(inside: f64) -> f64 {
        (inside + 0.5).clamp(0.0, 1.0)
    }

    /// One sample per pixel centre, as captures trace one ray per pixel (resolution scale 1.0).
    fn hard(inside: f64) -> f64 {
        if inside > 0.0 { 1.0 } else { 0.0 }
    }

    /// The brief's case. The 2/255 threshold keeps only the pixels this ramp blends at least
    /// 98.8% black, whose centres lie more than 0.488 px inside the edge, so the radius reads
    /// 0.490 px small: within the 0.5 px bound by construction, but only just.
    /// `hard_edged_disc_is_unbiased` is the case captures produce.
    #[test]
    fn synthetic_disc_radius() {
        let measured = shadow_radius(&grey_with_disc(R, antialiased));
        assert!(
            (measured - R).abs() < 0.5,
            "measured {measured}, expected {R}"
        );
    }

    #[test]
    fn ignores_black_outside_centre_region() {
        let mut img = grey_with_disc(R, antialiased);
        let alone = shadow_radius(&img);
        for y in 20..220 {
            for x in 20..220 {
                img.put_pixel(x, y, Rgb([0, 0, 0]));
            }
        }
        assert_eq!(shadow_radius(&img), alone);
    }

    #[test]
    fn hard_edged_disc_is_unbiased() {
        let measured = shadow_radius(&grey_with_disc(R, hard));
        assert!(
            (measured - R).abs() < 0.05,
            "measured {measured}, expected {R}"
        );
    }

    #[test]
    fn grey_centre_measures_zero() {
        let img = RgbImage::from_pixel(64, 48, Rgb([128, 128, 128]));
        assert_eq!(shadow_radius(&img), 0.0);
    }

    #[test]
    fn black_means_rec709_luma_below_two_codes() {
        // Luma ×255 = 0.2126 R + 0.7152 G + 0.0722 B on the 8-bit code values.
        assert!(is_black(&Rgb([0, 0, 0])));
        assert!(is_black(&Rgb([1, 1, 1])));
        assert!(!is_black(&Rgb([2, 2, 2])));
        assert!(is_black(&Rgb([9, 0, 0])));
        assert!(!is_black(&Rgb([10, 0, 0])));
        assert!(is_black(&Rgb([0, 2, 0])));
        assert!(!is_black(&Rgb([0, 3, 0])));
        assert!(is_black(&Rgb([0, 0, 27])));
        assert!(!is_black(&Rgb([0, 0, 28])));
    }

    #[test]
    fn verdict_and_report_line() {
        assert!(passes(158.2, 158.745, 1.0));
        assert!(!passes(157.5, 158.745, 1.0));
        assert_eq!(
            report_line(158.74491, 158.745, true),
            "measured=158.745 expected=158.745 PASS"
        );
        assert_eq!(
            report_line(150.0, 158.745, false),
            "measured=150.000 expected=158.745 FAIL"
        );
    }

    #[test]
    fn loads_png_from_disk() {
        let path = std::env::temp_dir().join(format!(
            "uni-verify-{}-loads_png_from_disk.png",
            std::process::id()
        ));
        RgbImage::from_pixel(5, 3, Rgb([0, 0, 0]))
            .save(&path)
            .expect("write test PNG");
        let loaded = load_rgb8(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load test PNG");
        assert_eq!(loaded.dimensions(), (5, 3));
        assert!(load_rgb8(&path).is_err(), "a missing file is an error");
    }
}
