//! Fits an ellipse to a body's limb in a capture PNG (M1 Task B8).
//!
//! `uni-verify disk --png <f> --center <x>,<y> [--threshold 0.008] --expected-a <px>
//! [--expected-b <px>] --tol <px> [--ratio <b/a> --ratio-tol <t>]
//! [--exclude-angle <deg> --exclude-width <deg>]`
//!
//! 1. **Bright region.** A pixel is bright when its Rec. 709 luma, taken on the 8-bit code values
//!    as in `shadow` (ITU-R BT.709-6 item 3.2), is at least `--threshold` (default 0.008, about
//!    2/255: a pre-exposed 4.49 × 10⁻⁴ through AgX). The region is the 4-connected set of bright
//!    pixels that contains the `--center` pixel, so other bright things that do not touch it (a
//!    moon, a star) are ignored.
//! 2. **Edges.** From the region's area centroid, [`RAY_COUNT`] rays (every 0.5°) walk the pixel
//!    grid exactly (Amanatides & Woo 1987, "A Fast Voxel Traversal Algorithm for Ray Tracing",
//!    Eurographics '87). A ray's edge point is where it enters the first pixel below the
//!    threshold: the boundary between that pixel and the last bright one. A pixel is bright when
//!    its centre lies inside the limb (body.wgsl shades the antialiased rim like the limb, so
//!    rim pixels outside it stay dark), so the staircase of pixel boundaries lies on the limb on
//!    average and the fit is unbiased; each point is off by up to half a pixel, which the 720
//!    points average out.
//! 3. **Exclusion.** `--exclude-angle` and `--exclude-width` drop the rays within half the width
//!    of the angle and of the opposite direction (angle + 180°), e.g. a ring line across the
//!    disk. A width of 180° or more drops only the wedge around the angle itself, a half plane
//!    such as a terminator side: with its opposite it would drop every ray. Angles are degrees
//!    counter-clockwise from the image's +x axis with y up.
//! 4. **Sky check.** If fewer than half the rays that are not excluded find a drop before
//!    leaving the frame, the background is above the threshold, and the tool fails with
//!    "no limb edge: background above threshold" instead of fitting the sky.
//! 5. **Fit.** Direct least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999, "Direct Least
//!    Square Fitting of Ellipses", IEEE TPAMI 21(5), 476) in the numerically stable form of
//!    Halíř & Flusser 1998 ("Numerically Stable Direct Least Squares Fitting of Ellipses",
//!    WSCG '98), on coordinates centred and scaled to unit RMS radius.
//!
//! The fitted semi-axes are a ≥ b; the result passes when |a − expected a| ≤ tol, and, when
//! given, |b − expected b| ≤ tol and |b/a − ratio| ≤ ratio tol.

use crate::{Verdict, pass_word};
use image::RgbImage;
use std::path::PathBuf;

/// Default luma threshold, as a fraction of the 8-bit code range: ≈ 2/255, which AgX maps from
/// a pre-exposed 4.49 × 10⁻⁴ (brief, re-review r1 NP2).
pub const DEFAULT_THRESHOLD: f64 = 0.008;
/// Number of rays cast from the bright region's centroid, one every 0.5°.
pub const RAY_COUNT: usize = 720;

/// Rec. 709 luma weights for R′, G′, B′ (ITU-R BT.709-6, Part 1, item 3.2).
const REC709_LUMA: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// Rays to leave out of the fit: within half of `width_deg` of `angle_deg` and, for widths below
/// 180°, of the opposite direction too. Degrees, counter-clockwise from +x with y up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Exclusion {
    pub angle_deg: f64,
    pub width_deg: f64,
}

impl Exclusion {
    /// True when the ray at `ray_deg` is excluded (the wedge's edges are excluded too).
    pub fn excludes(&self, ray_deg: f64) -> bool {
        let half = self.width_deg / 2.0;
        let d = angular_distance(ray_deg, self.angle_deg);
        if self.width_deg >= 180.0 {
            d <= half
        } else {
            d <= half || 180.0 - d <= half
        }
    }
}

/// The smaller angle between two directions [deg], in [0, 180].
fn angular_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// The `disk` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct DiskArgs {
    pub png: PathBuf,
    /// A pixel on the body, the seed of its bright region [px].
    pub center: (f64, f64),
    /// Luma threshold, a fraction of the 8-bit code range.
    pub threshold: f64,
    /// Expected semi-major axis [px].
    pub expected_a: f64,
    /// Expected semi-minor axis [px], if checked.
    pub expected_b: Option<f64>,
    /// Tolerance on a and b [px].
    pub tol: f64,
    /// Expected b/a and its tolerance, if checked.
    pub ratio: Option<(f64, f64)>,
    pub exclusion: Option<Exclusion>,
}

/// An ellipse in image pixels: centre (x right, y down, pixel k spans [k, k + 1]), semi-axes
/// a ≥ b, and the major axis's direction in degrees counter-clockwise from +x with y up, in
/// (−90, 90].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    pub cx: f64,
    pub cy: f64,
    pub a: f64,
    pub b: f64,
    pub angle_deg: f64,
}

/// What [`measure`] found.
#[derive(Debug, Clone, PartialEq)]
pub enum Measurement {
    /// An ellipse fitted to `edges` edge points from `unexcluded` rays.
    Fitted {
        ellipse: Ellipse,
        centroid: (f64, f64),
        unexcluded: usize,
        edges: usize,
    },
    /// Fewer than half the unexcluded rays found a drop below the threshold: the background
    /// is above it. Nothing was fitted.
    NoLimb { unexcluded: usize, edges: usize },
    /// No region, a centroid off the region, every ray excluded, or a degenerate fit.
    Failed(String),
}

/// Which pixels of `img` have a luma of at least `threshold`, row-major.
fn bright_mask(img: &RgbImage, threshold: f64) -> Vec<bool> {
    img.pixels()
        .map(|px| {
            let luma: f64 = REC709_LUMA
                .iter()
                .zip(px.0)
                .map(|(w, c)| w * f64::from(c))
                .sum();
            luma / 255.0 >= threshold
        })
        .collect()
}

/// Area centroid [px] of the 4-connected `bright` region that contains pixel `seed`, or `None`
/// if the seed is outside the image or not bright.
fn region_centroid(
    bright: &[bool],
    width: u32,
    height: u32,
    seed: (i64, i64),
) -> Option<(f64, f64)> {
    let (w, h) = (i64::from(width), i64::from(height));
    let inside = |x: i64, y: i64| (0..w).contains(&x) && (0..h).contains(&y);
    if !inside(seed.0, seed.1) || !bright[(seed.1 * w + seed.0) as usize] {
        return None;
    }
    let mut visited = vec![false; bright.len()];
    let mut stack = vec![seed];
    let (mut count, mut sum_x, mut sum_y) = (0u64, 0.0, 0.0);
    while let Some((x, y)) = stack.pop() {
        if !inside(x, y) {
            continue;
        }
        let i = (y * w + x) as usize;
        if visited[i] || !bright[i] {
            continue;
        }
        visited[i] = true;
        count += 1;
        sum_x += x as f64 + 0.5;
        sum_y += y as f64 + 0.5;
        stack.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]);
    }
    Some((sum_x / count as f64, sum_y / count as f64))
}

/// Walks the pixel grid from `origin` along the unit direction `dir` (image axes, y down) and
/// returns the point where the ray enters the first pixel that is not bright, or `None` if it
/// leaves the image first. Amanatides & Woo 1987.
fn edge_along_ray(
    bright: &[bool],
    width: u32,
    height: u32,
    origin: (f64, f64),
    dir: (f64, f64),
) -> Option<(f64, f64)> {
    let (ox, oy) = origin;
    let (dx, dy) = dir;
    let (mut ix, mut iy) = (ox.floor() as i64, oy.floor() as i64);
    let step_x: i64 = if dx > 0.0 { 1 } else { -1 };
    let step_y: i64 = if dy > 0.0 { 1 } else { -1 };
    // Ray parameter at the next vertical / horizontal grid line, and between two of them.
    let first = |i: i64, o: f64, d: f64| {
        if d > 0.0 {
            (i as f64 + 1.0 - o) / d
        } else if d < 0.0 {
            (i as f64 - o) / d
        } else {
            f64::INFINITY
        }
    };
    let delta = |d: f64| {
        if d != 0.0 {
            1.0 / d.abs()
        } else {
            f64::INFINITY
        }
    };
    let (mut t_x, mut t_y) = (first(ix, ox, dx), first(iy, oy, dy));
    let (dt_x, dt_y) = (delta(dx), delta(dy));
    let (w, h) = (i64::from(width), i64::from(height));
    loop {
        let t = if t_x < t_y {
            ix += step_x;
            t_x += dt_x;
            t_x - dt_x
        } else {
            iy += step_y;
            t_y += dt_y;
            t_y - dt_y
        };
        if !(0..w).contains(&ix) || !(0..h).contains(&iy) {
            return None;
        }
        if !bright[(iy * w + ix) as usize] {
            return Some((ox + t * dx, oy + t * dy));
        }
    }
}

/// Measures the limb of the bright region that contains `seed` (see the module docs).
pub fn measure(
    img: &RgbImage,
    seed: (f64, f64),
    threshold: f64,
    exclusion: Option<Exclusion>,
) -> Measurement {
    let (w, h) = img.dimensions();
    let bright = bright_mask(img, threshold);
    let seed_px = (seed.0.floor() as i64, seed.1.floor() as i64);
    let Some(centroid) = region_centroid(&bright, w, h, seed_px) else {
        return Measurement::Failed(format!(
            "no bright region at --center ({}, {}): luma below {threshold} or outside the image",
            seed.0, seed.1
        ));
    };
    let (cx, cy) = (centroid.0.floor() as i64, centroid.1.floor() as i64);
    if !bright[(cy * i64::from(w) + cx) as usize] {
        return Measurement::Failed(format!(
            "the bright region's centroid ({:.1}, {:.1}) is not on the region",
            centroid.0, centroid.1
        ));
    }
    let mut points = Vec::with_capacity(RAY_COUNT);
    let mut unexcluded = 0;
    for k in 0..RAY_COUNT {
        let deg = k as f64 * 360.0 / RAY_COUNT as f64;
        if exclusion.is_some_and(|e| e.excludes(deg)) {
            continue;
        }
        unexcluded += 1;
        let (s, c) = deg.to_radians().sin_cos();
        // Counter-clockwise with y up is (cos, −sin) in image axes.
        if let Some(p) = edge_along_ray(&bright, w, h, centroid, (c, -s)) {
            points.push(p);
        }
    }
    let edges = points.len();
    if unexcluded == 0 {
        return Measurement::Failed("the exclusion leaves no ray".to_owned());
    }
    if 2 * edges < unexcluded {
        return Measurement::NoLimb { unexcluded, edges };
    }
    match fit_ellipse(&points) {
        Some(ellipse) => Measurement::Fitted {
            ellipse,
            centroid,
            unexcluded,
            edges,
        },
        None => Measurement::Failed(format!("the {edges} edge points do not fit an ellipse")),
    }
}

type M3 = [[f64; 3]; 3];

fn mat_mul(a: &M3, b: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
}

fn transpose(a: &M3) -> M3 {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

fn mat_vec(a: &M3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|k| a[i][k] * v[k]).sum())
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn det3(a: &M3) -> f64 {
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
}

/// Inverse by the adjugate, or `None` for a singular matrix.
fn inverse3(a: &M3) -> Option<M3> {
    let det = det3(a);
    if !det.is_finite() || det == 0.0 {
        return None;
    }
    // Rows of the inverse are the cross products of the columns, over the determinant.
    let col = |j: usize| [a[0][j], a[1][j], a[2][j]];
    let rows = [
        cross(col(1), col(2)),
        cross(col(2), col(0)),
        cross(col(0), col(1)),
    ];
    Some(rows.map(|r| r.map(|v| v / det)))
}

/// Real roots of λ³ + a·λ² + b·λ + c (Cardano; the trigonometric form for three real roots,
/// Viète), each polished by two Newton steps.
fn real_cubic_roots(a: f64, b: f64, c: f64) -> Vec<f64> {
    let p = b - a * a / 3.0;
    let q = 2.0 * a * a * a / 27.0 - a * b / 3.0 + c;
    let shift = -a / 3.0;
    let disc = (q / 2.0).powi(2) + (p / 3.0).powi(3);
    let mut roots = if p >= 0.0 && disc <= 0.0 {
        // p = q = 0: a triple root.
        vec![shift]
    } else if disc <= 0.0 {
        let r = (-p / 3.0).sqrt();
        let phi = (-q / (2.0 * r * r * r)).clamp(-1.0, 1.0).acos();
        (0..3)
            .map(|k| 2.0 * r * ((phi - std::f64::consts::TAU * f64::from(k)) / 3.0).cos() + shift)
            .collect()
    } else {
        let s = disc.sqrt();
        vec![(-q / 2.0 + s).cbrt() + (-q / 2.0 - s).cbrt() + shift]
    };
    for x in &mut roots {
        for _ in 0..2 {
            let f = ((*x + a) * *x + b) * *x + c;
            let df = (3.0 * *x + 2.0 * a) * *x + b;
            if df != 0.0 && (f / df).is_finite() {
                *x -= f / df;
            }
        }
    }
    roots
}

/// A unit vector spanning the null space of `m − λI` (rank 2): the longest cross product of
/// two of its rows.
fn null_vector(m: &M3, lambda: f64) -> Option<[f64; 3]> {
    let mut r = *m;
    for (i, row) in r.iter_mut().enumerate() {
        row[i] -= lambda;
    }
    let candidates = [cross(r[0], r[1]), cross(r[0], r[2]), cross(r[1], r[2])];
    let norm = |v: &[f64; 3]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
    let best = candidates
        .iter()
        .max_by(|u, v| norm(u).total_cmp(&norm(v)))?;
    let n = norm(best);
    (n > 0.0 && n.is_finite()).then(|| best.map(|x| x / n))
}

/// Direct least-squares ellipse fit to `points` [px] (Fitzgibbon, Pilu & Fisher 1999), in
/// Halíř & Flusser's (1998) stable form. `None` with fewer than six points or when no ellipse
/// fits (collinear or degenerate points).
pub fn fit_ellipse(points: &[(f64, f64)]) -> Option<Ellipse> {
    if points.len() < 6 {
        return None;
    }
    // Centre and scale the coordinates (unit RMS radius) so the scatter matrices are well
    // conditioned.
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let scale = (points
        .iter()
        .map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2))
        .sum::<f64>()
        / n)
        .sqrt();
    if !(scale > 0.0 && scale.is_finite()) {
        return None;
    }
    // Design matrix split into the quadratic part D1 = [x², xy, y²] and the linear part
    // D2 = [x, y, 1]; S1 = D1ᵀD1, S2 = D1ᵀD2, S3 = D2ᵀD2 (Halíř & Flusser §3).
    let (mut s1, mut s2, mut s3) = ([[0.0; 3]; 3], [[0.0; 3]; 3], [[0.0; 3]; 3]);
    for p in points {
        let (x, y) = ((p.0 - mx) / scale, (p.1 - my) / scale);
        let d1 = [x * x, x * y, y * y];
        let d2 = [x, y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                s1[i][j] += d1[i] * d1[j];
                s2[i][j] += d1[i] * d2[j];
                s3[i][j] += d2[i] * d2[j];
            }
        }
    }
    // T = −S3⁻¹S2ᵀ gives the linear coefficients from the quadratic ones; M = S1 + S2·T is the
    // reduced scatter matrix, and C1⁻¹M with C1 = [[0, 0, 2], [0, −1, 0], [2, 0, 0]] the reduced
    // eigensystem, whose eigenvector with 4ac − b² > 0 is the ellipse.
    let t = mat_mul(&inverse3(&s3)?, &transpose(&s2)).map(|row| row.map(|v| -v));
    let m = {
        let st = mat_mul(&s2, &t);
        let mut m = s1;
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += st[i][j];
            }
        }
        m
    };
    let reduced: M3 = [
        m[2].map(|v| v / 2.0),
        m[1].map(|v| -v),
        m[0].map(|v| v / 2.0),
    ];
    // Characteristic polynomial λ³ − tr·λ² + (sum of principal 2×2 minors)·λ − det.
    let r = &reduced;
    let trace = r[0][0] + r[1][1] + r[2][2];
    let minors = r[0][0] * r[1][1] - r[0][1] * r[1][0] + r[0][0] * r[2][2] - r[0][2] * r[2][0]
        + r[1][1] * r[2][2]
        - r[1][2] * r[2][1];
    let a1 = real_cubic_roots(-trace, minors, -det3(r))
        .into_iter()
        .filter_map(|lambda| null_vector(r, lambda))
        .map(|v| (4.0 * v[0] * v[2] - v[1] * v[1], v))
        .filter(|(constraint, _)| *constraint > 0.0)
        .max_by(|a, b| a.0.total_cmp(&b.0))?
        .1;
    let a2 = mat_vec(&t, a1);
    conic_to_ellipse([a1[0], a1[1], a1[2], a2[0], a2[1], a2[2]]).map(|e| Ellipse {
        cx: e.cx * scale + mx,
        cy: e.cy * scale + my,
        a: e.a * scale,
        b: e.b * scale,
        angle_deg: e.angle_deg,
    })
}

/// Centre, semi-axes and major-axis angle of the conic A·x² + B·xy + C·y² + D·x + E·y + F = 0
/// in image axes (y down), or `None` if it is not a real ellipse.
fn conic_to_ellipse([a, b, c, d, e, f]: [f64; 6]) -> Option<Ellipse> {
    let den = b * b - 4.0 * a * c;
    if !(den < 0.0) {
        return None;
    }
    let x0 = (2.0 * c * d - b * e) / den;
    let y0 = (2.0 * a * e - b * d) / den;
    let f0 = a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0 + f;
    // Eigenvalues of the quadratic form [[A, B/2], [B/2, C]]: the semi-axis along each
    // eigenvector is √(−F0/λ).
    let mid = (a + c) / 2.0;
    let rad = ((a - c) / 2.0).hypot(b / 2.0);
    let axes = [mid - rad, mid + rad].map(|l| (-f0 / l, l));
    if !axes.iter().all(|(s, _)| *s > 0.0 && s.is_finite()) {
        return None;
    }
    let (major, minor) = if axes[0].0 >= axes[1].0 {
        (axes[0], axes[1])
    } else {
        (axes[1], axes[0])
    };
    let l = major.1;
    let (u, v) = ((b / 2.0, l - a), (l - c, b / 2.0));
    let dir = if u.0.hypot(u.1) >= v.0.hypot(v.1) {
        u
    } else {
        v
    };
    // Image y points down; report counter-clockwise with y up, in (−90, 90].
    let mut angle = if dir == (0.0, 0.0) {
        0.0
    } else {
        (-dir.1).atan2(dir.0).to_degrees()
    };
    while angle > 90.0 {
        angle -= 180.0;
    }
    while angle <= -90.0 {
        angle += 180.0;
    }
    Some(Ellipse {
        cx: x0,
        cy: y0,
        a: major.0.sqrt(),
        b: minor.0.sqrt(),
        angle_deg: angle,
    })
}

/// One `  label measured expected ± tol ok|FAIL` line, and whether it passed.
fn criterion(label: &str, measured: f64, expected: f64, tol: f64, digits: usize) -> (String, bool) {
    let ok = (measured - expected).abs() <= tol;
    let line = format!(
        "  {label} {measured:.digits$} expected {expected:.digits$} ± {tol} (error {:+.digits$}) {}",
        measured - expected,
        if ok { "ok" } else { "FAIL" }
    );
    (line, ok)
}

/// Judges a measurement against the command line's expectations.
pub fn judge(m: &Measurement, args: &DiskArgs) -> Verdict {
    match m {
        Measurement::Failed(reason) => Verdict {
            lines: vec![format!("disk: {reason} FAIL")],
            pass: false,
        },
        Measurement::NoLimb { unexcluded, edges } => Verdict {
            lines: vec![format!(
                "disk: no limb edge: background above threshold (only {edges} of {unexcluded} \
                 rays dropped below luma {} before leaving the frame) FAIL",
                args.threshold
            )],
            pass: false,
        },
        Measurement::Fitted {
            ellipse: e,
            centroid,
            unexcluded,
            edges,
        } => {
            let mut lines = vec![format!(
                "disk: centre ({:.3}, {:.3}) a {:.3} px b {:.3} px b/a {:.5} major axis {:.2}° \
                 ({edges} edge points from {unexcluded} rays, {} excluded; rays from ({:.2}, \
                 {:.2}); luma threshold {})",
                e.cx,
                e.cy,
                e.a,
                e.b,
                e.b / e.a,
                e.angle_deg,
                RAY_COUNT - unexcluded,
                centroid.0,
                centroid.1,
                args.threshold
            )];
            let mut checks = vec![criterion("a", e.a, args.expected_a, args.tol, 3)];
            if let Some(b) = args.expected_b {
                checks.push(criterion("b", e.b, b, args.tol, 3));
            }
            if let Some((ratio, tol)) = args.ratio {
                checks.push(criterion("b/a", e.b / e.a, ratio, tol, 5));
            }
            let pass = checks.iter().all(|(_, ok)| *ok);
            lines.extend(checks.into_iter().map(|(line, _)| line));
            lines.push(format!("disk {}", pass_word(pass)));
            Verdict { lines, pass }
        }
    }
}

/// Loads the PNG, measures and judges.
pub fn run(args: &DiskArgs) -> anyhow::Result<Verdict> {
    let img = crate::shadow::load_rgb8(&args.png)?;
    let m = measure(&img, args.center, args.threshold, args.exclusion);
    Ok(judge(&m, args))
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgb;

    // ---- Test fixture: a copy of uni-post's AgX CPU twin (crates/uni-post/src/agx.rs at
    // 58374d2), in plain arrays because uni-verify does not depend on uni-post or glam. Troy
    // Sobotka's AgX "base" with the default-contrast look in Benjamin Wrensch's real-time form
    // ("Minimal AgX Implementation", iolite engine blog, 2023); every constant is verbatim. ----

    /// Inset matrix, by columns (Wrensch 2023, `agx_mat`).
    #[allow(clippy::excessive_precision)]
    const AGX_INSET: [[f32; 3]; 3] = [
        [0.842479062253094, 0.0423282422610123, 0.0423756549057051],
        [0.0784335999999992, 0.878468636469772, 0.0784336],
        [0.0792237451477643, 0.0791661274605434, 0.879142973793104],
    ];
    /// Outset matrix, by columns (Wrensch 2023, `agx_mat_inv`).
    #[allow(clippy::excessive_precision)]
    const AGX_OUTSET: [[f32; 3]; 3] = [
        [1.19687900512017, -0.0528968517574562, -0.0529716355144438],
        [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
        [-0.0990297440797205, -0.0989611768448433, 1.15107367264116],
    ];
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;
    const AGX_LOG_FLOOR: f32 = 1e-10;

    fn mul_cols(cols: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        std::array::from_fn(|i| cols[0][i] * v[0] + cols[1][i] * v[1] + cols[2][i] * v[2])
    }

    /// Wrensch 2023 `agxDefaultContrastApprox`, coefficients and term order verbatim.
    fn agx_default_contrast(x: f32) -> f32 {
        let x2 = x * x;
        let x4 = x2 * x2;
        15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
            - 0.00232
    }

    /// IEC 61966-2-1:1999 §5.2, sRGB code value → display-linear.
    fn srgb_to_linear(c: f32) -> f32 {
        let x = c.max(0.0);
        if x <= 0.04045 {
            x / 12.92
        } else {
            ((x + 0.055) / 1.055).powf(2.4)
        }
    }

    /// `uni_post::agx`: pre-exposed scene-linear Rec. 709 → display-linear.
    fn agx(rgb: [f32; 3]) -> [f32; 3] {
        let inset = mul_cols(&AGX_INSET, rgb);
        let encoded = inset.map(|c| {
            let ev = c.max(AGX_LOG_FLOOR).log2().clamp(AGX_MIN_EV, AGX_MAX_EV);
            agx_default_contrast((ev - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV))
        });
        mul_cols(&AGX_OUTSET, encoded).map(|c| srgb_to_linear(c.clamp(0.0, 1.0)))
    }

    /// The sRGB view's hardware encoder: IEC 61966-2-1:1999 §5.2 (linear → code), then 8 bits.
    fn to_code(linear: f32) -> u8 {
        let c = linear.clamp(0.0, 1.0);
        let encoded = if c <= 0.003_130_8 {
            12.92 * c
        } else {
            1.055 * c.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u8
    }

    /// The 8-bit pixel the tonemap writes for a grey pre-exposed value (no dither).
    fn tonemap_grey(pre_exposed: f32) -> Rgb<u8> {
        Rgb(agx([pre_exposed; 3]).map(to_code))
    }

    // ---- Synthetic limbs ----

    /// Normalised radius ρ (1 on the limb) and the first-order signed distance from the limb [px]
    /// (positive outside) of the point (x, y) for ellipse `e`.
    fn ellipse_coords(e: &Ellipse, x: f64, y: f64) -> (f64, f64) {
        let (s, c) = e.angle_deg.to_radians().sin_cos();
        let (dx, dy) = (x - e.cx, y - e.cy);
        // Major-axis direction in the image is (cos, −sin) (angles counter-clockwise, y up).
        let u = dx * c - dy * s;
        let v = dx * s + dy * c;
        let rho = ((u / e.a).powi(2) + (v / e.b).powi(2)).sqrt();
        let grad = ((u / (e.a * e.a)).powi(2) + (v / (e.b * e.b)).powi(2)).sqrt();
        let d = if rho > 1e-9 {
            (rho - 1.0) * rho / grad
        } else {
            -e.a
        };
        (rho, d)
    }

    /// body.wgsl's silhouette coverage: clamp(0.5 − d, 0, 1) for the signed distance d [px].
    fn coverage(d: f64) -> f64 {
        (0.5 - d).clamp(0.0, 1.0)
    }

    /// A white-on-black ellipse antialiased by coverage, written straight as code values.
    fn aa_ellipse(w: u32, h: u32, e: &Ellipse) -> RgbImage {
        RgbImage::from_fn(w, h, |x, y| {
            let (_, d) = ellipse_coords(e, f64::from(x) + 0.5, f64::from(y) + 0.5);
            let v = (255.0 * coverage(d)).round() as u8;
            Rgb([v, v, v])
        })
    }

    /// Pre-exposed value of a Lambert disk at zero phase: coverage · peak · μ with
    /// μ = √(1 − ρ²) (the emission cosine on a sphere seen face-on; the AA rim pixels are shaded
    /// like the limb, μ = 0, as body.wgsl does).
    fn lambert(e: &Ellipse, peak: f64, x: f64, y: f64) -> f64 {
        let (rho, d) = ellipse_coords(e, x, y);
        coverage(d) * peak * (1.0 - rho * rho).max(0.0).sqrt()
    }

    /// The tonemapped capture of a Lambert disk (`peak` pre-exposed) on a uniform `sky`.
    fn lambert_disk(w: u32, h: u32, e: &Ellipse, peak: f64, sky: f64) -> RgbImage {
        let sky_px = tonemap_grey(sky as f32);
        RgbImage::from_fn(w, h, |x, y| {
            let v = lambert(e, peak, f64::from(x) + 0.5, f64::from(y) + 0.5);
            if v == 0.0 {
                sky_px
            } else {
                tonemap_grey((v + sky) as f32)
            }
        })
    }

    fn fitted(m: &Measurement) -> Ellipse {
        match m {
            Measurement::Fitted { ellipse, .. } => *ellipse,
            other => panic!("no fit: {other:?}"),
        }
    }

    /// Black sky far below the threshold (4e-7 pre-exposed: Saturn's band at scene EV 11.1).
    const BLACK_SKY: f64 = 4e-7;
    /// Lambert sub-solar peak, pre-exposed (Jupiter in "Exposure on paper").
    const PEAK: f64 = 0.123;

    #[test]
    fn agx_fixture_matches_the_threshold_arithmetic() {
        // The brief: 0.008 luma (≈ 2/255) is a pre-exposed 4.49e-4 through AgX.
        let px = tonemap_grey(4.49e-4);
        assert!(px.0.iter().all(|&c| c == 2), "{px:?}");
        assert_eq!(tonemap_grey(BLACK_SKY as f32), Rgb([0, 0, 0]));
        // The eye-adapted band at 20 mag/arcsec² (1.44e-2) is ≈ 33/255.
        let band = tonemap_grey(1.44e-2);
        assert!((30..=36).contains(&band.0[1]), "{band:?}");
    }

    #[test]
    fn fit_ellipse_recovers_exact_points() {
        let e = Ellipse {
            cx: 812.25,
            cy: 433.5,
            a: 300.0,
            b: 120.0,
            angle_deg: -35.0,
        };
        let (s, c) = e.angle_deg.to_radians().sin_cos();
        let points: Vec<(f64, f64)> = (0..90)
            .map(|k| {
                let t = f64::from(k) * 4.0_f64.to_radians();
                let (u, v) = (e.a * t.cos(), e.b * t.sin());
                // Inverse of ellipse_coords' rotation.
                (e.cx + u * c + v * s, e.cy - u * s + v * c)
            })
            .collect();
        let f = fit_ellipse(&points).expect("fit");
        for (got, want) in [
            (f.cx, e.cx),
            (f.cy, e.cy),
            (f.a, e.a),
            (f.b, e.b),
            (f.angle_deg, e.angle_deg),
        ] {
            assert!((got - want).abs() < 1e-6, "{f:?} vs {e:?}");
        }
        assert!(
            fit_ellipse(&points[..5]).is_none(),
            "five points do not fix an ellipse"
        );
    }

    #[test]
    fn exclusion_wedge_and_opposite() {
        let ring = Exclusion {
            angle_deg: 0.0,
            width_deg: 10.0,
        };
        for ray in [0.0, 4.5, 5.0, 355.5, 175.0, 180.0, 184.5, -3.0] {
            assert!(ring.excludes(ray), "{ray}");
        }
        for ray in [5.5, 90.0, 174.5, 185.5, 270.0] {
            assert!(!ring.excludes(ray), "{ray}");
        }
        // 180° or more: the wedge alone (a terminator side), not its opposite.
        let terminator = Exclusion {
            angle_deg: 0.0,
            width_deg: 180.0,
        };
        for ray in [0.0, 89.5, 90.0, 270.0, 300.0] {
            assert!(terminator.excludes(ray), "{ray}");
        }
        for ray in [90.5, 180.0, 269.5] {
            assert!(!terminator.excludes(ray), "{ray}");
        }
    }

    #[test]
    fn ellipse_fit_synthetic() {
        // A uniform antialiased ellipse. Its true edge is the half-coverage contour, so the
        // edge threshold sits at half the plateau; the Lambert tests below use the default.
        let e = Ellipse {
            cx: 550.37,
            cy: 499.81,
            a: 425.0,
            b: 380.0,
            angle_deg: 20.0,
        };
        let img = aa_ellipse(1100, 1000, &e);
        let f = fitted(&measure(&img, (550.0, 500.0), 0.5, None));
        assert!((f.a - e.a).abs() <= 0.3, "a {f:?}");
        assert!((f.b - e.b).abs() <= 0.3, "b {f:?}");
        assert!((f.angle_deg - e.angle_deg).abs() <= 0.2, "angle {f:?}");
        assert!(
            (f.cx - e.cx).abs() <= 0.3 && (f.cy - e.cy).abs() <= 0.3,
            "{f:?}"
        );
    }

    #[test]
    fn ellipse_fit_small_agx_limb() {
        // Review C4: a 60-px Lambert limb through the AgX twin at the default threshold.
        let e = Ellipse {
            cx: 100.3,
            cy: 90.6,
            a: 60.0,
            b: 54.0,
            angle_deg: 0.0,
        };
        let img = lambert_disk(200, 180, &e, PEAK, BLACK_SKY);
        let f = fitted(&measure(&img, (100.0, 90.0), DEFAULT_THRESHOLD, None));
        let ratio_error = f.b / f.a - 0.90;
        eprintln!(
            "ellipse_fit_small_agx_limb: a {:.4} b {:.4} b/a {:.5} ratio error {ratio_error:+.5}",
            f.a,
            f.b,
            f.b / f.a
        );
        assert!(ratio_error.abs() <= 0.004, "{f:?}");
    }

    #[test]
    fn ellipse_fit_lambert_limb_black_sky() {
        // Re-review r1 NP2: Jupiter's disk at fov 3.5° (a = 421.3 px, b/a = 0.9351).
        let e = Ellipse {
            cx: 480.4,
            cy: 430.2,
            a: 421.3,
            b: 421.3 * 0.9351,
            angle_deg: 0.0,
        };
        let img = lambert_disk(960, 860, &e, PEAK, BLACK_SKY);
        let f = fitted(&measure(&img, (480.0, 430.0), DEFAULT_THRESHOLD, None));
        eprintln!(
            "ellipse_fit_lambert_limb_black_sky: a {:.4} (error {:+.4}) b/a {:.5} (error {:+.5})",
            f.a,
            f.a - e.a,
            f.b / f.a,
            f.b / f.a - 0.9351
        );
        assert!((f.a - e.a).abs() <= 0.1, "{f:?}");
        assert!((f.b / f.a - 0.9351).abs() <= 0.001, "{f:?}");
    }

    #[test]
    fn disk_fails_when_sky_above_threshold() {
        // The same disk on the eye-adapted band at 20 mag/arcsec² (1.44e-2 at star EV −4).
        let e = Ellipse {
            cx: 480.4,
            cy: 430.2,
            a: 421.3,
            b: 421.3 * 0.9351,
            angle_deg: 0.0,
        };
        let img = lambert_disk(960, 860, &e, PEAK, 1.44e-2);
        let m = measure(&img, (480.0, 430.0), DEFAULT_THRESHOLD, None);
        assert!(matches!(m, Measurement::NoLimb { edges: 0, .. }), "{m:?}");
        let args = DiskArgs {
            png: PathBuf::from("jupiter-disk.png"),
            center: (480.0, 430.0),
            threshold: DEFAULT_THRESHOLD,
            expected_a: 421.3,
            expected_b: None,
            tol: 1.0,
            ratio: Some((0.9351, 0.004)),
            exclusion: None,
        };
        let v = judge(&m, &args);
        assert!(!v.pass);
        let text = v.lines.join("\n");
        assert!(
            text.contains("no limb edge: background above threshold"),
            "{text}"
        );
        assert!(text.ends_with("FAIL"), "{text}");
    }

    /// Saturn edge-on (a = 433.6 px, b/a = 0.902) with a 12-px bright ring line through it at
    /// 0°, out to the A ring's outer edge (2.27 equatorial radii), inside the frame so that rays
    /// running along the line end on it rather than leaving the frame.
    fn saturn_with_ring_line() -> (Ellipse, RgbImage) {
        let e = Ellipse {
            cx: 1050.3,
            cy: 450.6,
            a: 433.6,
            b: 433.6 * 0.902,
            angle_deg: 0.0,
        };
        let sky_px = tonemap_grey(BLACK_SKY as f32);
        let img = RgbImage::from_fn(2100, 900, |x, y| {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let on_ring = (py - e.cy).abs() < 6.0 && (px - e.cx).abs() < 2.27 * e.a;
            let ring = if on_ring { 0.05 } else { 0.0 };
            let v = lambert(&e, PEAK, px, py) + ring;
            if v == 0.0 {
                sky_px
            } else {
                tonemap_grey((v + BLACK_SKY) as f32)
            }
        });
        (e, img)
    }

    #[test]
    fn disk_exclusion_wedge() {
        let (e, img) = saturn_with_ring_line();
        let wedge = Exclusion {
            angle_deg: 0.0,
            width_deg: 10.0,
        };
        let f = fitted(&measure(
            &img,
            (1050.0, 450.0),
            DEFAULT_THRESHOLD,
            Some(wedge),
        ));
        assert!((f.b / f.a - 0.902).abs() <= 0.003, "{f:?}");
        assert!((f.a - e.a).abs() <= 0.5, "{f:?}");
        // Without the wedge the rays that leave the limb inside the line end out on the ring,
        // and the fit is visibly wrong: the exclusion is what makes the check pass.
        let bad = fitted(&measure(&img, (1050.0, 450.0), DEFAULT_THRESHOLD, None));
        eprintln!("disk_exclusion_wedge: with the wedge {f:?}, without {bad:?}");
        assert!((bad.b / bad.a - 0.902).abs() > 0.003, "{bad:?}");
    }

    #[test]
    fn disk_half_arc() {
        // Only the left half lit, terminator on the vertical centre line.
        let e = Ellipse {
            cx: 400.2,
            cy: 350.7,
            a: 300.0,
            b: 300.0,
            angle_deg: 0.0,
        };
        let sky_px = tonemap_grey(BLACK_SKY as f32);
        let img = RgbImage::from_fn(800, 700, |x, y| {
            let (px, py) = (f64::from(x) + 0.5, f64::from(y) + 0.5);
            let v = if px < e.cx {
                lambert(&e, PEAK, px, py)
            } else {
                0.0
            };
            if v == 0.0 {
                sky_px
            } else {
                tonemap_grey((v + BLACK_SKY) as f32)
            }
        });
        let terminator = Exclusion {
            angle_deg: 0.0,
            width_deg: 180.0,
        };
        let f = fitted(&measure(
            &img,
            (250.0, 350.0),
            DEFAULT_THRESHOLD,
            Some(terminator),
        ));
        eprintln!("disk_half_arc: {f:?}");
        assert!((f.a - e.a).abs() <= 0.5, "{f:?}");
    }

    #[test]
    fn disk_ignores_unconnected_bright() {
        let e = Ellipse {
            cx: 400.6,
            cy: 380.3,
            a: 300.0,
            b: 270.0,
            angle_deg: 0.0,
        };
        let alone = lambert_disk(800, 760, &e, PEAK, BLACK_SKY);
        let mut with_corner = alone.clone();
        // A second bright disk (radius 40 px) in the top-left corner, not touching the first.
        for y in 0..120 {
            for x in 0..120 {
                let d = (f64::from(x) + 0.5 - 60.0).hypot(f64::from(y) + 0.5 - 60.0);
                if d < 40.0 {
                    with_corner.put_pixel(x, y, tonemap_grey(0.1));
                }
            }
        }
        let a = fitted(&measure(&alone, (400.0, 380.0), DEFAULT_THRESHOLD, None));
        let b = fitted(&measure(
            &with_corner,
            (400.0, 380.0),
            DEFAULT_THRESHOLD,
            None,
        ));
        assert!(
            (a.a - b.a).abs() <= 0.1 && (a.b - b.b).abs() <= 0.1,
            "{a:?} vs {b:?}"
        );
        assert!((b.a - e.a).abs() <= 0.5, "{b:?}");
    }

    #[test]
    fn judge_reports_numbers_and_verdict() {
        let m = Measurement::Fitted {
            ellipse: Ellipse {
                cx: 960.0,
                cy: 540.0,
                a: 421.5,
                b: 394.1,
                angle_deg: 0.1,
            },
            centroid: (960.0, 540.0),
            unexcluded: 720,
            edges: 720,
        };
        let mut args = DiskArgs {
            png: PathBuf::from("jupiter-disk.png"),
            center: (960.0, 540.0),
            threshold: DEFAULT_THRESHOLD,
            expected_a: 421.3,
            expected_b: Some(394.0),
            tol: 1.0,
            ratio: Some((0.9351, 0.004)),
            exclusion: None,
        };
        let v = judge(&m, &args);
        assert!(v.pass, "{v:?}");
        assert!(v.lines.last().is_some_and(|l| l.ends_with("PASS")), "{v:?}");
        args.ratio = Some((0.9, 0.004));
        let v = judge(&m, &args);
        assert!(!v.pass, "{v:?}");
        assert!(v.lines.last().is_some_and(|l| l.ends_with("FAIL")), "{v:?}");
    }
}
