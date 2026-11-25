//! Checks projected positions against brightness centroids in a capture PNG (M1 Task B8).
//!
//! `uni-verify points --png <f> --json <capture.json> --names <a,b,…> --tol <px>`
//!
//! For each name, the capture JSON's `projected` list (B9 Step 2: `{name, x, y, radius_px}`,
//! framebuffer pixels with x right, y down and pixel k spanning [k, k + 1]) gives the expected
//! position. The tool takes the square window of side 2·⌈2·radius_px + 3⌉ + 1 px (at least 7)
//! centred on the expected pixel, so a resolved body such as the 4.2-px Moon at `start` is not
//! truncated (review m25), clipped to the image. Brightness is the Rec. 709 luminance of the
//! sRGB-decoded pixel (IEC 61966-2-1 §5.2; ITU-R BT.709-6 item 3.2). The background is the median
//! brightness of the window's border pixels; the centroid weights each pixel centre by its
//! brightness above that background (negative values count as 0). The error is the distance
//! from the centroid to the expected position.
//!
//! **Signal** (final review, app I3). On a lit sky with the tonemap's dither about half the
//! window lies above the border median, and that noise alone has a centroid near the window's
//! centre, so a missing star could "pass". The point must therefore stand out: its SNR, the
//! brightest window pixel above the background over the border's noise σ, must be at least
//! [`MIN_SNR`]. σ comes from the differences of neighbouring pixels along the border (a loop
//! around the window): 1.4826 × their median absolute value ÷ √2. 1.4826 = 1/Φ⁻¹(3/4) turns a
//! median absolute deviation into a normal σ (Rousseeuw & Croux 1993, JASA 88, 1273), and the
//! difference of two pixels with independent noise has σ√2. A smooth gradient across the
//! window (a comet's coma, a planet's glare) barely changes neighbouring pixels, so it does not
//! count as noise, while the spread of the border about its median would: at
//! `comet-neowise-2020` that spread is 0.15 against a nucleus 0.51 above the median. σ is never
//! less than one 8-bit code step at the background's level, the smallest difference a PNG can
//! hold (on black or a flat border the median difference is 0).
//!
//! The check passes when every name is in the list, has an SNR of at least [`MIN_SNR`], and
//! its centroid is within `--tol`.

use crate::{Verdict, pass_word};
use anyhow::Context;
use image::{Rgb, RgbImage};
use serde::Deserialize;
use std::fs;
use std::path::Path;

/// One `projected` entry of a capture JSON.
#[derive(Debug, Deserialize)]
struct Projected {
    name: String,
    x: f64,
    y: f64,
    #[serde(default)]
    radius_px: f64,
}

/// The part of a capture JSON this check reads.
#[derive(Debug, Deserialize)]
struct CaptureJson {
    #[serde(default)]
    projected: Vec<Projected>,
}

/// Least SNR of a point (see the module docs). For Gaussian noise the brightest of a 7 × 7
/// window's 49 pixels exceeds 8 σ with a probability below 10⁻¹³, and the dither's noise is
/// bounded (about ±1 code), so noise alone never reaches it (a dithered band without the point
/// measures about 1); on black a point needs about 8 codes. At the M1 gate captures (HEAD
/// 335b309) the least SNR is 14.2, the NEOWISE nucleus inside its coma (row 20); the stars,
/// moons and the Sun measure 72–790.
pub const MIN_SNR: f64 = 8.0;

/// The consistency factor that turns a median absolute deviation into a normal σ, 1/Φ⁻¹(3/4)
/// (Rousseeuw & Croux 1993).
const MAD_TO_SIGMA: f64 = 1.4826;

/// A point's measured centroid [px] and its signal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub x: f64,
    pub y: f64,
    /// Brightest window pixel above the background (linear luminance).
    pub peak: f64,
    /// The border's noise σ (linear luminance), at least one code step.
    pub noise: f64,
}

impl Measured {
    /// peak / noise.
    pub fn snr(&self) -> f64 {
        self.peak / self.noise
    }
}

/// Half-side [px] of the centroid window for an item of projected radius `radius_px`:
/// ⌈2·radius_px + 3⌉, so the side 2·half + 1 is at least 7. A negative or non-finite radius
/// counts as 0.
pub fn window_half(radius_px: f64) -> u32 {
    let r = if radius_px.is_finite() && radius_px > 0.0 {
        radius_px
    } else {
        0.0
    };
    (2.0 * r + 3.0).ceil() as u32
}

/// sRGB code value → linear light, IEC 61966-2-1:1999 §5.2.
fn srgb_to_linear(code: u8) -> f64 {
    let c = f64::from(code) / 255.0;
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Rec. 709 relative luminance of an 8-bit sRGB pixel (ITU-R BT.709-6 item 3.2).
fn brightness(px: &Rgb<u8>) -> f64 {
    let [r, g, b] = px.0.map(srgb_to_linear);
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Median of `values` (the mean of the middle two for an even count); 0 for none.
fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    }
}

/// The linear-light step of one 8-bit code at linear level `level`: the gap between the two
/// codes around it (codes 254 and 255 at the top).
fn code_step(level: f64) -> f64 {
    let code = (0u8..=254)
        .rev()
        .find(|&c| srgb_to_linear(c) <= level)
        .unwrap_or(0);
    srgb_to_linear(code + 1) - srgb_to_linear(code)
}

/// The window's border pixels [x0, x1] × [y0, y1] as one loop, each next to the one before:
/// the top row left to right, the right column down, the bottom row back and the left column up.
/// It is closed (the last pixel touches the first) unless the window is one row or one column.
fn border_loop(x0: i64, x1: i64, y0: i64, y1: i64) -> Vec<(i64, i64)> {
    let mut ring: Vec<(i64, i64)> = (x0..=x1).map(|i| (i, y0)).collect();
    ring.extend((y0 + 1..=y1).map(|j| (x1, j)));
    if y1 > y0 {
        ring.extend((x0..x1).rev().map(|i| (i, y1)));
    }
    if x1 > x0 {
        ring.extend((y0 + 1..y1).rev().map(|j| (x0, j)));
    }
    ring
}

/// The border's noise σ from the brightness along its loop (`closed` when the last pixel
/// touches the first): 1.4826 × the median of the absolute differences of neighbouring pixels
/// ÷ √2, at least one code step at `background` (see the module docs).
fn border_noise(ring: &[f64], closed: bool, background: f64) -> f64 {
    let mut steps: Vec<f64> = ring.windows(2).map(|p| (p[1] - p[0]).abs()).collect();
    if closed && let (Some(first), Some(last)) = (ring.first(), ring.last()) {
        steps.push((first - last).abs());
    }
    (MAD_TO_SIGMA * median(&mut steps) / std::f64::consts::SQRT_2).max(code_step(background))
}

/// The background-subtracted brightness centroid [px] of the point around (x, y) and its signal
/// (see the module docs), or `None` when the window is off the image or holds no light above its
/// border.
pub fn measure(img: &RgbImage, x: f64, y: f64, radius_px: f64) -> Option<Measured> {
    let (w, h) = img.dimensions();
    if !(x.is_finite() && y.is_finite()) || w == 0 || h == 0 {
        return None;
    }
    let half = i64::from(window_half(radius_px));
    let (px, py) = (x.floor() as i64, y.floor() as i64);
    let x0 = (px - half).max(0);
    let x1 = (px + half).min(i64::from(w) - 1);
    let y0 = (py - half).max(0);
    let y1 = (py + half).min(i64::from(h) - 1);
    if x0 > x1 || y0 > y1 {
        return None;
    }
    let at = |i: i64, j: i64| brightness(img.get_pixel(i as u32, j as u32));
    let ring: Vec<f64> = border_loop(x0, x1, y0, y1)
        .into_iter()
        .map(|(i, j)| at(i, j))
        .collect();
    let background = median(&mut ring.clone());
    let noise = border_noise(&ring, x1 > x0 && y1 > y0, background);
    let (mut sum, mut sum_x, mut sum_y, mut peak) = (0.0, 0.0, 0.0, 0.0_f64);
    for j in y0..=y1 {
        for i in x0..=x1 {
            let weight = (at(i, j) - background).max(0.0);
            sum += weight;
            sum_x += weight * (i as f64 + 0.5);
            sum_y += weight * (j as f64 + 0.5);
            peak = peak.max(weight);
        }
    }
    (sum > 0.0).then(|| Measured {
        x: sum_x / sum,
        y: sum_y / sum,
        peak,
        noise,
    })
}

/// Loads the PNG and the capture JSON and checks every name against `tol` [px].
pub fn run(png: &Path, json: &Path, names: &[String], tol: f64) -> anyhow::Result<Verdict> {
    let img = crate::shadow::load_rgb8(png)?;
    let text =
        fs::read_to_string(json).with_context(|| format!("cannot read {}", json.display()))?;
    let capture: CaptureJson =
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", json.display()))?;
    let mut lines = Vec::new();
    let (mut within, mut worst, mut least_snr) = (0, None::<f64>, f64::INFINITY);
    for name in names {
        let Some(item) = capture.projected.iter().find(|p| &p.name == name) else {
            lines.push(format!(
                "{name}: not in the capture's `projected` list FAIL"
            ));
            continue;
        };
        let side = 2 * window_half(item.radius_px) + 1;
        match measure(&img, item.x, item.y, item.radius_px) {
            None => lines.push(format!(
                "{name}: expected ({:.3}, {:.3}): no light above the background in the \
                 {side}-px window FAIL",
                item.x, item.y
            )),
            Some(m) => {
                let error = (m.x - item.x).hypot(m.y - item.y);
                let snr = m.snr();
                least_snr = least_snr.min(snr);
                let bright = snr >= MIN_SNR;
                let ok = bright && error <= tol;
                within += usize::from(ok);
                if bright {
                    worst = Some(worst.map_or(error, |w| w.max(error)));
                }
                let signal = format!(
                    "SNR {snr:.1} (peak {:.3e} over noise {:.3e})",
                    m.peak, m.noise
                );
                lines.push(if bright {
                    format!(
                        "{name}: expected ({:.3}, {:.3}) centroid ({:.3}, {:.3}) error {error:.3} \
                         px, {signal} (radius {:.2} px, window {side} px) {}",
                        item.x,
                        item.y,
                        m.x,
                        m.y,
                        item.radius_px,
                        if ok { "ok" } else { "FAIL" }
                    )
                } else {
                    format!(
                        "{name}: expected ({:.3}, {:.3}): {signal} < {MIN_SNR}: nothing stands \
                         above the noise in the {side}-px window FAIL",
                        item.x, item.y
                    )
                });
            }
        }
    }
    let pass = within == names.len();
    let snr = if least_snr.is_finite() {
        format!("; least SNR {least_snr:.1}, need ≥ {MIN_SNR}")
    } else {
        String::new()
    };
    let worst = worst.map_or_else(
        || "no point above the noise".to_owned(),
        |w| format!("worst measured error {w:.3} px"),
    );
    lines.push(format!(
        "points: {within} of {} within {tol} px ({worst}{snr}) {}",
        names.len(),
        pass_word(pass)
    ));
    Ok(Verdict { lines, pass })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;
    use std::fs;
    use std::path::PathBuf;

    /// A fresh, empty directory under the system temp dir, unique to this process and test.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uni-verify-{}-{test}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// Background code value of the synthetic frame (a dim band behind the points).
    const BACKGROUND: f64 = 12.0;

    /// A 240×160 frame: two Gaussian blobs (σ = 0.9 px, 1.3 px) at sub-pixel positions and a
    /// uniformly lit 4.2-px-radius disk with antialiased edges, written as 8-bit codes.
    fn frame() -> RgbImage {
        let blobs = [(40.3, 50.7, 0.9, 220.0), (120.85, 60.15, 1.3, 160.0)];
        let (dx, dy, dr) = (180.4, 110.6, 4.2);
        RgbImage::from_fn(240, 160, |x, y| {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let mut v = BACKGROUND;
            for (bx, by, sigma, peak) in blobs {
                let r2 = (px - bx).powi(2) + (py - by).powi(2);
                v += peak * (-r2 / (2.0 * sigma * sigma)).exp();
            }
            // Disk coverage by 8×8 supersampling.
            let mut inside = 0;
            for sy in 0..8 {
                for sx in 0..8 {
                    let qx = f64::from(x) + (f64::from(sx) + 0.5) / 8.0;
                    let qy = f64::from(y) + (f64::from(sy) + 0.5) / 8.0;
                    if (qx - dx).hypot(qy - dy) < dr {
                        inside += 1;
                    }
                }
            }
            v += 180.0 * f64::from(inside) / 64.0;
            let c = v.round().clamp(0.0, 255.0) as u8;
            Rgb([c, c, c])
        })
    }

    #[test]
    fn window_side_follows_radius() {
        // Side 2·⌈2·radius_px + 3⌉ + 1, at least 7 (review m25).
        assert_eq!(2 * window_half(0.0) + 1, 7);
        assert_eq!(2 * window_half(0.5) + 1, 9);
        assert_eq!(2 * window_half(4.2) + 1, 25);
        assert_eq!(2 * window_half(f64::NAN) + 1, 7);
    }

    #[test]
    fn points_centroid_subpixel() {
        let dir = temp_dir("points_centroid_subpixel");
        let png = dir.join("start.png");
        let json = dir.join("start.json");
        frame().save(&png).expect("write PNG");
        fs::write(
            &json,
            r#"{"name": "start", "projected": [
                {"name": "Io", "x": 40.3, "y": 50.7, "radius_px": 0.3},
                {"name": "Europa", "x": 120.85, "y": 60.15, "radius_px": 0.6},
                {"name": "Moon", "x": 180.4, "y": 110.6, "radius_px": 4.2},
                {"name": "Far", "x": 10.0, "y": 10.0, "radius_px": 0.5}
            ]}"#,
        )
        .expect("write JSON");

        let img = crate::shadow::load_rgb8(&png).expect("load PNG");
        for (x, y, r, tol) in [
            (40.3, 50.7, 0.3, 0.1),
            (120.85, 60.15, 0.6, 0.1),
            (180.4, 110.6, 4.2, 0.2),
        ] {
            let m = measure(&img, x, y, r).expect("centroid");
            let (cx, cy) = (m.x, m.y);
            let err = (cx - x).hypot(cy - y);
            assert!(err <= tol, "({x}, {y}): centroid ({cx}, {cy}), error {err}");
        }

        let names = |list: &[&str]| list.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
        let pass = run(&png, &json, &names(&["Io", "Europa", "Moon"]), 0.2).expect("run");
        // A name missing from `projected` fails, and so does an item with no light at its pixel.
        let missing = run(&png, &json, &names(&["Io", "Ganymede"]), 0.2).expect("run");
        let dark = run(&png, &json, &names(&["Far"]), 0.2).expect("run");
        let _ = fs::remove_dir_all(&dir);

        assert!(pass.pass, "{pass:?}");
        assert!(
            pass.lines.last().is_some_and(|l| l.ends_with("PASS")),
            "{pass:?}"
        );
        assert_eq!(pass.lines.len(), 4, "{pass:?}");
        assert!(!missing.pass, "{missing:?}");
        assert!(missing.lines.join("\n").contains("Ganymede"), "{missing:?}");
        assert!(!dark.pass, "{dark:?}");
    }

    /// The tonemap's dither on a lit background: −1, 0 or +1 code with probabilities 1/4, 1/2,
    /// 1/4 (a triangular PDF), from a fixed integer hash of the pixel (splitmix64's finaliser,
    /// Steele, Lea & Flood 2014).
    fn dither(x: u32, y: u32) -> f64 {
        let mut z = (u64::from(x) << 32 | u64::from(y)).wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        match z & 3 {
            0 => -1.0,
            3 => 1.0,
            _ => 0.0,
        }
    }

    /// A 160×80 frame: the band at code 40 with ±1-code dither and, when `blobs`, two Gaussian
    /// points (σ = 0.9 px) of the given peaks [codes] at (40.3, 40.7) and (120.85, 40.15).
    fn dithered(blobs: Option<[f64; 2]>) -> RgbImage {
        let points = [(40.3, 40.7), (120.85, 40.15)];
        RgbImage::from_fn(160, 80, |x, y| {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let mut v = 40.0 + dither(x, y);
            if let Some(peaks) = blobs {
                for ((bx, by), peak) in points.into_iter().zip(peaks) {
                    let r2 = (px - bx).powi(2) + (py - by).powi(2);
                    v += peak * (-r2 / (2.0 * 0.9 * 0.9)).exp();
                }
            }
            let c = v.round().clamp(0.0, 255.0) as u8;
            Rgb([c, c, c])
        })
    }

    /// Final review, app I3: with the points removed from a dithered band, the noise's
    /// centroid still lands near the window centre, but the check must FAIL on the SNR. Bright
    /// points pass with a high SNR; a point 4 codes above the band does not stand out.
    #[test]
    fn points_need_signal_above_the_noise() {
        let dir = temp_dir("points_need_signal_above_the_noise");
        let json = dir.join("sky.json");
        fs::write(
            &json,
            r#"{"name": "sky", "projected": [
                {"name": "Bellatrix", "x": 40.3, "y": 40.7, "radius_px": 0.0},
                {"name": "Alnilam", "x": 120.85, "y": 40.15, "radius_px": 0.0}
            ]}"#,
        )
        .expect("write JSON");
        let names = vec!["Bellatrix".to_owned(), "Alnilam".to_owned()];
        let mut verdicts = Vec::new();
        for (file, frame) in [
            ("lit.png", dithered(Some([180.0, 30.0]))),
            ("empty.png", dithered(None)),
            ("faint.png", dithered(Some([4.0, 4.0]))),
        ] {
            let png = dir.join(file);
            frame.save(&png).expect("write PNG");
            verdicts.push(run(&png, &json, &names, 1.0).expect("run"));
        }
        let empty = dithered(None);
        let _ = fs::remove_dir_all(&dir);
        let [lit, missing, faint] = <[Verdict; 3]>::try_from(verdicts).expect("three verdicts");

        assert!(lit.pass, "{lit:#?}");
        for l in &lit.lines[..2] {
            let snr: f64 = l
                .split("SNR ")
                .nth(1)
                .and_then(|s| s.split(' ').next())
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| panic!("no SNR in {l}"));
            assert!(snr >= 2.0 * MIN_SNR, "{l}");
        }
        assert!(lit.lines[2].contains("least SNR"), "{lit:#?}");

        // Without the points the noise's centroid lands within the 1-px tolerance of both
        // expected positions (0.91 and 0.40 px with this dither), so the check without the SNR
        // test passed them; the SNR is about 1.
        for (x, y) in [(40.3, 40.7), (120.85, 40.15)] {
            let m = measure(&empty, x, y, 0.0).expect("noise above the border median");
            assert!(m.snr() < 2.0, "{m:?}");
            assert!((m.x - x).hypot(m.y - y) <= 1.0, "{m:?}");
        }
        for v in [&missing, &faint] {
            assert!(!v.pass, "{v:#?}");
            assert_eq!(
                v.lines
                    .iter()
                    .filter(|l| l.contains("nothing stands above the noise"))
                    .count(),
                2,
                "{v:#?}"
            );
            assert!(
                v.lines
                    .last()
                    .is_some_and(|l| l.starts_with("points: 0 of 2") && l.ends_with("FAIL")),
                "{v:#?}"
            );
        }
    }

    #[test]
    fn noise_floor_is_one_code_step() {
        // On black the floor is code 1's linear value, 1/(255 × 12.92) (IEC 61966-2-1 §5.2).
        assert!((code_step(0.0) - 1.0 / (255.0 * 12.92)).abs() < 1e-12);
        // At code 128 (linear 0.2158) one step is about 0.0033.
        let mid = srgb_to_linear(128);
        assert!((code_step(mid) - (srgb_to_linear(129) - mid)).abs() < 1e-12);
        // At white the step below it.
        assert!((code_step(1.0) - (1.0 - srgb_to_linear(254))).abs() < 1e-12);
        // A flat border has no differences: the floor holds.
        assert_eq!(border_noise(&[0.0; 24], true, 0.0), code_step(0.0));
        // A ramp along the border counts only its steps: 2e-2 each gives σ = 1.4826 × 2e-2 / √2
        // (above the floor, 6.0e-3 at 0.5), not the ramp's spread of ±0.23 about its median.
        let ramp: Vec<f64> = (0..24).map(|k| 0.27 + 2e-2 * f64::from(k)).collect();
        let open = border_noise(&ramp, false, 0.5);
        assert!(
            (open - 1.4826 * 2e-2 / std::f64::consts::SQRT_2).abs() < 1e-9,
            "{open}"
        );
        // Closed, the wrap from the top of the ramp back to its foot is one outlier step.
        let closed = border_noise(&ramp, true, 0.5);
        assert!((closed - open).abs() < 1e-9, "{closed}");
    }

    #[test]
    fn border_loop_walks_neighbours() {
        // 7 × 7 window: 24 border pixels, each next to the one before, and closed.
        let ring = border_loop(10, 16, 20, 26);
        assert_eq!(ring.len(), 24);
        let mut sorted = ring.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 24);
        let touch = |a: (i64, i64), b: (i64, i64)| (a.0 - b.0).abs() + (a.1 - b.1).abs() == 1;
        assert!(ring.windows(2).all(|p| touch(p[0], p[1])), "{ring:?}");
        assert!(touch(ring[0], ring[23]));
        // Clipped to one row or one column: a line, every pixel once.
        assert_eq!(border_loop(0, 6, 5, 5).len(), 7);
        assert_eq!(border_loop(3, 3, 0, 6).len(), 7);
        assert_eq!(border_loop(3, 3, 4, 4), vec![(3, 4)]);
    }
}
