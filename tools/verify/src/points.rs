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
//! from the centroid to the expected position. The check passes when every name is in the list,
//! has light above the background, and is within `--tol`.

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

/// Background-subtracted brightness centroid [px] in the window around (x, y) (see the module
/// docs), or `None` when the window is off the image or holds no light above its border.
pub fn centroid(img: &RgbImage, x: f64, y: f64, radius_px: f64) -> Option<(f64, f64)> {
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
    let mut border: Vec<f64> = (y0..=y1)
        .flat_map(|j| (x0..=x1).map(move |i| (i, j)))
        .filter(|&(i, j)| i == x0 || i == x1 || j == y0 || j == y1)
        .map(|(i, j)| at(i, j))
        .collect();
    let background = median(&mut border);
    let (mut sum, mut sum_x, mut sum_y) = (0.0, 0.0, 0.0);
    for j in y0..=y1 {
        for i in x0..=x1 {
            let weight = (at(i, j) - background).max(0.0);
            sum += weight;
            sum_x += weight * (i as f64 + 0.5);
            sum_y += weight * (j as f64 + 0.5);
        }
    }
    (sum > 0.0).then(|| (sum_x / sum, sum_y / sum))
}

/// Loads the PNG and the capture JSON and checks every name against `tol` [px].
pub fn run(png: &Path, json: &Path, names: &[String], tol: f64) -> anyhow::Result<Verdict> {
    let img = crate::shadow::load_rgb8(png)?;
    let text =
        fs::read_to_string(json).with_context(|| format!("cannot read {}", json.display()))?;
    let capture: CaptureJson =
        serde_json::from_str(&text).with_context(|| format!("cannot parse {}", json.display()))?;
    let mut lines = Vec::new();
    let (mut within, mut worst) = (0, 0.0_f64);
    for name in names {
        let Some(item) = capture.projected.iter().find(|p| &p.name == name) else {
            lines.push(format!(
                "{name}: not in the capture's `projected` list FAIL"
            ));
            continue;
        };
        let side = 2 * window_half(item.radius_px) + 1;
        match centroid(&img, item.x, item.y, item.radius_px) {
            None => lines.push(format!(
                "{name}: expected ({:.3}, {:.3}): no light above the background in the \
                 {side}-px window FAIL",
                item.x, item.y
            )),
            Some((cx, cy)) => {
                let error = (cx - item.x).hypot(cy - item.y);
                let ok = error <= tol;
                within += usize::from(ok);
                worst = worst.max(error);
                lines.push(format!(
                    "{name}: expected ({:.3}, {:.3}) centroid ({cx:.3}, {cy:.3}) error {error:.3} \
                     px (radius {:.2} px, window {side} px) {}",
                    item.x,
                    item.y,
                    item.radius_px,
                    if ok { "ok" } else { "FAIL" }
                ));
            }
        }
    }
    let pass = within == names.len();
    lines.push(format!(
        "points: {within} of {} within {tol} px (worst measured error {worst:.3} px) {}",
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
            let (cx, cy) = centroid(&img, x, y, r).expect("centroid");
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
}
