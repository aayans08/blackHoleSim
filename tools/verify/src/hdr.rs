//! Photometric checks on the `--hdr` capture sidecar (M1 Task B8; review I16, round 2).
//!
//! `universe --capture --hdr` writes `<name>.hdr.bin` (little-endian f32 RGBA of the pre-exposed
//! HDR target, rows top-down, copied before bloom) and, in the capture JSON (B9 Step 2):
//! - `hdr: {file, width, height, fov_y_rad, pre: [scene, star], ev100: [scene, star], layout}`;
//!   `file` is relative to the JSON's directory, `pre` the exposure factors 1/(1.2·2^EV);
//! - `regions: {name: [[x, y], …]}`, pixel samples (column, row; fractional values are floored);
//! - `samples: [{x, y, expected_cd_m2: [r, g, b], what}]`;
//! - `features: {name: {x, y, r_min, r_max, angle_deg?, expected_slope?}}`;
//! - `flux: [{name, x, y, radius_px, expected_lux, mask: [[x, y, r], …]}]`.
//!
//! Positions (x, y) of features and flux entries are framebuffer pixels: x right, y down, pixel
//! k spans [k, k + 1], so a pixel's centre is k + 0.5. Luminance is Rec. 709,
//! Y = 0.2126 R + 0.7152 G + 0.0722 B (ITU-R BT.709-6 item 3.2), on the linear HDR values. Each
//! check divides by `pre[0]` (world light) or `pre[1]` (sky light) where it states so; ratios,
//! slopes and angles do not depend on the exposure.
//!
//! - `ratio`: stat(region `num`, channel) / stat(region `den`, channel), with the channel pair
//!   r/b (R over B), b/r, g (G over G) or luma (Y over Y) and the stat mean or median (default
//!   median). A colour ratio of one region names it as both `--num` and `--den`.
//! - `stats`: a region's relative RMS of luminance (population standard deviation over the
//!   mean) and its mean pre-exposed luminance.
//! - `samples`: each sample pixel's luminance ÷ `pre[0]` against the luminance of
//!   `expected_cd_m2`, as a relative error.
//! - `profile`: the least-squares slope of ln(mean Y) against ln(mean ρ) over 1-px annuli from
//!   `r_min` to `r_max` around the feature, against its `expected_slope`.
//! - `angle`: the direction of Σ Y·ρ̂ over the pixels in the annulus [`r_min`, `r_max`) whose
//!   colour passes the test B > R (bluer than neutral, as the CO⁺ ion tail is against the
//!   solar-coloured dust and coma), against `angle_deg`. Angles are degrees counter-clockwise
//!   from the image's +x axis with y up.
//! - `flux`: E = Σ_{ρ_i < R} (Y_i − Y_bg)·Ω_i ÷ `pre[1]` against `expected_lux` (re-review r2
//!   NPA). Ω_i = Ω_c·cos³θ_i is the solid angle of pixel i, the `uni_sky::stars` convention
//!   (Ω_c = s², s = 2·tan(fov_y/2)/H; cos θ_i = 1/√(1 + (s·d_i)²), d_i the pixel centre's distance
//!   from the image centre (W/2, H/2); dω = dA·cosθ/r², Pharr, Jakob & Humphreys 2016 §5.5.3).
//!   R is the entry's `radius_px`, the footprint the star shader used, which holds all of the
//!   PSF's energy. Y_bg is the median of Y over the annulus R ≤ ρ < 1.5 R, skipping pixels inside
//!   any `mask` disk, so the band behind the star is subtracted.

use crate::{Verdict, pass_word};
use anyhow::{Context, bail};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// The sidecar layout this reader understands (B9 Step 2).
const LAYOUT: &str = "f32 LE RGBA";

/// Rec. 709 luminance weights (ITU-R BT.709-6, Part 1, item 3.2).
const REC709_Y: [f64; 3] = [0.2126, 0.7152, 0.0722];

/// The `hdr` header of a capture JSON, with `file` resolved against the JSON's directory.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrHeader {
    pub file: PathBuf,
    pub width: u32,
    pub height: u32,
    /// The capture camera's vertical field of view [rad].
    pub fov_y_rad: f64,
    /// Exposure factors [scene, star]: pre-exposed value = luminance [cd/m²] × pre.
    pub pre: [f64; 2],
    pub ev100: Option<[f64; 2]>,
}

/// The `hdr` header as the JSON spells it.
#[derive(Debug, Deserialize)]
struct HeaderJson {
    file: String,
    width: u32,
    height: u32,
    fov_y_rad: f64,
    pre: [f64; 2],
    #[serde(default)]
    ev100: Option<[f64; 2]>,
    #[serde(default)]
    layout: Option<String>,
}

/// One `samples` entry: a pixel and the luminance the CPU twins expect there.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    /// Expected linear colour [cd/m²], not pre-exposed.
    pub expected_cd_m2: [f64; 3],
    #[serde(default)]
    pub what: String,
}

/// One `features` entry: a centre and an annulus [px].
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Feature {
    pub x: f64,
    pub y: f64,
    pub r_min: f64,
    pub r_max: f64,
    #[serde(default)]
    pub angle_deg: Option<f64>,
    #[serde(default)]
    pub expected_slope: Option<f64>,
}

/// One `flux` entry: a star's position [px], its footprint radius [px], its expected
/// illuminance [lx] and the other footprints [x, y, r] to leave out of the background annulus.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FluxEntry {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub radius_px: f64,
    pub expected_lux: f64,
    #[serde(default)]
    pub mask: Vec<[f64; 3]>,
}

/// The part of a capture JSON the HDR checks read.
#[derive(Debug, Deserialize)]
struct CaptureJson {
    #[serde(default)]
    hdr: Option<HeaderJson>,
    #[serde(default)]
    regions: BTreeMap<String, Vec<[f64; 2]>>,
    #[serde(default)]
    samples: Vec<Sample>,
    #[serde(default)]
    features: BTreeMap<String, Feature>,
    #[serde(default)]
    flux: Vec<FluxEntry>,
}

/// A capture's HDR image (pre-exposed linear RGBA, row-major, top row first) and the
/// measurement geometry the app wrote beside it.
#[derive(Debug, Clone, PartialEq)]
pub struct HdrCapture {
    pub header: HdrHeader,
    pub pixels: Vec<[f32; 4]>,
    pub regions: BTreeMap<String, Vec<[f64; 2]>>,
    pub samples: Vec<Sample>,
    pub features: BTreeMap<String, Feature>,
    pub flux: Vec<FluxEntry>,
}

impl HdrCapture {
    /// Loads a capture JSON and the HDR sidecar its `hdr` header names.
    pub fn load(json_path: &Path) -> anyhow::Result<Self> {
        let text = fs::read_to_string(json_path)
            .with_context(|| format!("cannot read {}", json_path.display()))?;
        let json: CaptureJson = serde_json::from_str(&text)
            .with_context(|| format!("cannot parse {}", json_path.display()))?;
        let Some(h) = json.hdr else {
            bail!(
                "{} has no `hdr` header: capture with --hdr",
                json_path.display()
            );
        };
        if let Some(layout) = &h.layout
            && !layout
                .to_ascii_lowercase()
                .starts_with(&LAYOUT.to_ascii_lowercase())
        {
            bail!("unsupported HDR layout {layout:?}, expected {LAYOUT:?}");
        }
        if !(h.fov_y_rad > 0.0 && h.fov_y_rad < std::f64::consts::PI) {
            bail!("hdr.fov_y_rad {} is not in (0, π)", h.fov_y_rad);
        }
        if !h.pre.iter().all(|p| p.is_finite() && *p > 0.0) {
            bail!("hdr.pre {:?} must be finite and positive", h.pre);
        }
        let dir = json_path.parent().unwrap_or(Path::new(""));
        let file = dir.join(&h.file);
        let bytes = fs::read(&file).with_context(|| format!("cannot read {}", file.display()))?;
        let expected = u64::from(h.width) * u64::from(h.height) * 16;
        if bytes.len() as u64 != expected {
            bail!(
                "{} holds {} bytes, but {}×{} f32 RGBA needs {expected}",
                file.display(),
                bytes.len(),
                h.width,
                h.height
            );
        }
        let pixels = bytes
            .chunks_exact(16)
            .map(|px| {
                std::array::from_fn(|c| {
                    let b = &px[4 * c..4 * c + 4];
                    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
                })
            })
            .collect();
        Ok(Self {
            header: HdrHeader {
                file,
                width: h.width,
                height: h.height,
                fov_y_rad: h.fov_y_rad,
                pre: h.pre,
                ev100: h.ev100,
            },
            pixels,
            regions: json.regions,
            samples: json.samples,
            features: json.features,
            flux: json.flux,
        })
    }

    /// Linear RGB of pixel (x, y), pre-exposed.
    fn rgb(&self, x: u32, y: u32) -> [f64; 3] {
        let p = self.pixels[y as usize * self.header.width as usize + x as usize];
        [f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]
    }

    /// Rec. 709 luminance of pixel (x, y), pre-exposed.
    pub fn luminance(&self, x: u32, y: u32) -> f64 {
        luminance(self.rgb(x, y))
    }

    /// The pixel containing the point (x, y), or `None` outside the image.
    fn pixel_at(&self, x: f64, y: f64) -> Option<(u32, u32)> {
        let (fx, fy) = (x.floor(), y.floor());
        let inside = fx >= 0.0
            && fy >= 0.0
            && fx < f64::from(self.header.width)
            && fy < f64::from(self.header.height);
        inside.then_some((fx as u32, fy as u32))
    }

    /// Region `name`'s pixels; an error if it is missing, empty or leaves the image.
    fn region(&self, name: &str) -> anyhow::Result<Vec<(u32, u32)>> {
        let samples = self
            .regions
            .get(name)
            .with_context(|| format!("no region {name:?} in the capture JSON"))?;
        if samples.is_empty() {
            bail!("region {name:?} is empty");
        }
        samples
            .iter()
            .map(|&[x, y]| {
                self.pixel_at(x, y)
                    .with_context(|| format!("region {name:?} sample ({x}, {y}) is off the image"))
            })
            .collect()
    }

    fn feature(&self, name: &str) -> anyhow::Result<&Feature> {
        let f = self
            .features
            .get(name)
            .with_context(|| format!("no feature {name:?} in the capture JSON"))?;
        if !(f.r_min >= 0.0 && f.r_max > f.r_min && f.r_max.is_finite()) {
            bail!(
                "feature {name:?} needs 0 ≤ r_min < r_max, not {} and {}",
                f.r_min,
                f.r_max
            );
        }
        Ok(f)
    }

    /// The image pixels whose centres lie within `r` of (x, y), with their offsets
    /// (centre − (x, y)) and distances.
    fn pixels_within(
        &self,
        x: f64,
        y: f64,
        r: f64,
    ) -> impl Iterator<Item = (u32, u32, f64, f64, f64)> {
        let (w, h) = (i64::from(self.header.width), i64::from(self.header.height));
        let x0 = ((x - r).floor() as i64).max(0);
        let x1 = ((x + r).ceil() as i64).min(w - 1);
        let y0 = ((y - r).floor() as i64).max(0);
        let y1 = ((y + r).ceil() as i64).min(h - 1);
        (y0..=y1)
            .flat_map(move |j| (x0..=x1).map(move |i| (i, j)))
            .filter_map(move |(i, j)| {
                let (dx, dy) = (i as f64 + 0.5 - x, j as f64 + 0.5 - y);
                let rho = dx.hypot(dy);
                (rho < r).then_some((i as u32, j as u32, dx, dy, rho))
            })
    }
}

/// Rec. 709 luminance of linear RGB.
pub(crate) fn luminance(rgb: [f64; 3]) -> f64 {
    REC709_Y.iter().zip(rgb).map(|(w, c)| w * c).sum()
}

/// Median of `values` (the mean of the middle two for an even count); NaN for none.
pub(crate) fn median(values: &mut [f64]) -> f64 {
    let n = values.len();
    if n == 0 {
        return f64::NAN;
    }
    let (lower, mid, _) = values.select_nth_unstable_by(n / 2, f64::total_cmp);
    let upper = *mid;
    if n % 2 == 1 {
        upper
    } else {
        let below = lower
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap_or(upper);
        (below + upper) / 2.0
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Which channels a ratio compares.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    /// R of `num` over B of `den`.
    RedOverBlue,
    /// B of `num` over R of `den`.
    BlueOverRed,
    /// G over G.
    Green,
    /// Rec. 709 luminance over luminance.
    Luma,
}

/// One channel of a pixel.
#[derive(Debug, Clone, Copy)]
enum Channel {
    R,
    G,
    B,
    Y,
}

impl Channel {
    fn of(self, rgb: [f64; 3]) -> f64 {
        match self {
            Self::R => rgb[0],
            Self::G => rgb[1],
            Self::B => rgb[2],
            Self::Y => luminance(rgb),
        }
    }
}

impl Channels {
    /// Parses `r/b`, `b/r`, `g` or `luma`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "r/b" => Some(Self::RedOverBlue),
            "b/r" => Some(Self::BlueOverRed),
            "g" => Some(Self::Green),
            "luma" => Some(Self::Luma),
            _ => None,
        }
    }

    /// (numerator channel, denominator channel).
    fn pair(self) -> (Channel, Channel) {
        match self {
            Self::RedOverBlue => (Channel::R, Channel::B),
            Self::BlueOverRed => (Channel::B, Channel::R),
            Self::Green => (Channel::G, Channel::G),
            Self::Luma => (Channel::Y, Channel::Y),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::RedOverBlue => "R over B",
            Self::BlueOverRed => "B over R",
            Self::Green => "G",
            Self::Luma => "luminance",
        }
    }
}

/// The statistic taken over a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    Mean,
    Median,
}

impl Stat {
    /// Parses `mean` or `median`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "mean" => Some(Self::Mean),
            "median" => Some(Self::Median),
            _ => None,
        }
    }

    fn of(self, values: &mut [f64]) -> f64 {
        match self {
            Self::Mean => mean(values),
            Self::Median => median(values),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
        }
    }
}

/// Acceptance bounds for a measured number: at least `min`, at most `max`, and within the
/// relative tolerance `expect.1` of `expect.0`, each when given. NaN never passes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub expect: Option<(f64, f64)>,
}

impl Bounds {
    pub fn is_empty(&self) -> bool {
        self.min.is_none() && self.max.is_none() && self.expect.is_none()
    }

    pub fn contains(&self, value: f64) -> bool {
        self.min.is_none_or(|m| value >= m)
            && self.max.is_none_or(|m| value <= m)
            && self
                .expect
                .is_none_or(|(x, t)| (value - x).abs() <= t * x.abs())
            && !value.is_nan()
    }

    /// e.g. "≥ 1.5", "≤ 0.5", "= 1 ± 10 %".
    fn describe(&self) -> String {
        let mut parts = Vec::new();
        if let Some(m) = self.min {
            parts.push(format!("≥ {m}"));
        }
        if let Some(m) = self.max {
            parts.push(format!("≤ {m}"));
        }
        if let Some((x, t)) = self.expect {
            parts.push(format!("= {x} ± {} %", t * 100.0));
        }
        parts.join(", ")
    }
}

/// The `ratio` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct RatioArgs {
    pub num: String,
    pub den: String,
    pub channels: Channels,
    pub stat: Stat,
    pub bounds: Bounds,
}

/// The `stats` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct StatsArgs {
    pub region: String,
    pub rms_min: Option<f64>,
    pub rms_max: Option<f64>,
    pub mean_pre_min: Option<f64>,
}

/// A measured ratio: the two statistics (pre-exposed) and their quotient.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RatioResult {
    pub num: f64,
    pub den: f64,
    pub ratio: f64,
}

/// stat(`num`, numerator channel) / stat(`den`, denominator channel).
pub fn ratio(cap: &HdrCapture, args: &RatioArgs) -> anyhow::Result<RatioResult> {
    let (cn, cd) = args.channels.pair();
    let values = |region: &str, c: Channel| -> anyhow::Result<Vec<f64>> {
        Ok(cap
            .region(region)?
            .into_iter()
            .map(|(x, y)| c.of(cap.rgb(x, y)))
            .collect())
    };
    let num = args.stat.of(&mut values(&args.num, cn)?);
    let den = args.stat.of(&mut values(&args.den, cd)?);
    Ok(RatioResult {
        num,
        den,
        ratio: num / den,
    })
}

/// `ratio`: measures and judges.
pub fn check_ratio(cap: &HdrCapture, args: &RatioArgs) -> anyhow::Result<Verdict> {
    let r = ratio(cap, args)?;
    let pass = args.bounds.contains(r.ratio);
    let stat = args.stat.label();
    Ok(Verdict {
        lines: vec![
            format!(
                "ratio {} / {} ({}, {stat} of {} / {} px): {:.6e} / {:.6e} (pre-exposed)",
                args.num,
                args.den,
                args.channels.label(),
                cap.region(&args.num)?.len(),
                cap.region(&args.den)?.len(),
                r.num,
                r.den
            ),
            format!(
                "ratio {:.5} {} {}",
                r.ratio,
                args.bounds.describe(),
                pass_word(pass)
            ),
        ],
        pass,
    })
}

/// Statistics of one region's luminance: pixel count, mean pre-exposed value and relative RMS
/// (population standard deviation over the mean).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionStats {
    pub count: usize,
    pub mean_pre: f64,
    pub rel_rms: f64,
}

pub fn region_stats(cap: &HdrCapture, region: &str) -> anyhow::Result<RegionStats> {
    let ys: Vec<f64> = cap
        .region(region)?
        .into_iter()
        .map(|(x, y)| cap.luminance(x, y))
        .collect();
    let m = mean(&ys);
    let variance = ys.iter().map(|y| (y - m).powi(2)).sum::<f64>() / ys.len() as f64;
    Ok(RegionStats {
        count: ys.len(),
        mean_pre: m,
        rel_rms: variance.sqrt() / m,
    })
}

/// One `  label value op bound ok|FAIL` line, and whether it passed.
fn bound_line(label: &str, value: f64, bound: f64, at_least: bool) -> (String, bool) {
    let ok = if at_least {
        value >= bound
    } else {
        value <= bound
    };
    let op = if at_least { "≥" } else { "≤" };
    (
        format!(
            "  {label} {value:.5e} {op} {bound} {}",
            if ok { "ok" } else { "FAIL" }
        ),
        ok,
    )
}

/// `stats`: measures and judges.
pub fn check_stats(cap: &HdrCapture, args: &StatsArgs) -> anyhow::Result<Verdict> {
    let s = region_stats(cap, &args.region)?;
    let mut checks = Vec::new();
    if let Some(min) = args.rms_min {
        checks.push(bound_line("relative RMS", s.rel_rms, min, true));
    }
    if let Some(max) = args.rms_max {
        checks.push(bound_line("relative RMS", s.rel_rms, max, false));
    }
    if let Some(min) = args.mean_pre_min {
        checks.push(bound_line("mean pre-exposed", s.mean_pre, min, true));
    }
    let pass = checks.iter().all(|(_, ok)| *ok);
    let mut lines = vec![format!(
        "stats {}: {} px, mean pre-exposed luminance {:.5e}, relative RMS {:.5}",
        args.region, s.count, s.mean_pre, s.rel_rms
    )];
    lines.extend(checks.into_iter().map(|(line, _)| line));
    lines.push(format!("stats {}", pass_word(pass)));
    Ok(Verdict { lines, pass })
}

/// One sample's comparison: luminance ÷ `pre[0]` [cd/m²] against the expectation's luminance.
#[derive(Debug, Clone, PartialEq)]
pub struct SampleResult {
    pub what: String,
    pub x: f64,
    pub y: f64,
    pub measured_cd_m2: f64,
    pub expected_cd_m2: f64,
    /// measured / expected − 1.
    pub rel_err: f64,
}

pub fn samples(cap: &HdrCapture) -> anyhow::Result<Vec<SampleResult>> {
    if cap.samples.is_empty() {
        bail!("no `samples` in the capture JSON");
    }
    let pre = cap.header.pre[0];
    cap.samples
        .iter()
        .map(|s| {
            let (x, y) = cap.pixel_at(s.x, s.y).with_context(|| {
                format!("sample {:?} ({}, {}) is off the image", s.what, s.x, s.y)
            })?;
            let measured = cap.luminance(x, y) / pre;
            let expected = luminance(s.expected_cd_m2);
            Ok(SampleResult {
                what: s.what.clone(),
                x: s.x,
                y: s.y,
                measured_cd_m2: measured,
                expected_cd_m2: expected,
                rel_err: measured / expected - 1.0,
            })
        })
        .collect()
}

/// `samples`: every sample within the relative tolerance `tol`.
pub fn check_samples(cap: &HdrCapture, tol: f64) -> anyhow::Result<Verdict> {
    let results = samples(cap)?;
    let mut lines = Vec::new();
    let mut within = 0;
    for r in &results {
        let ok = r.rel_err.abs() <= tol;
        within += usize::from(ok);
        lines.push(format!(
            "  {} ({}, {}): {:.5e} cd/m² vs {:.5e} cd/m² ({:+.2} %) {}",
            r.what,
            r.x,
            r.y,
            r.measured_cd_m2,
            r.expected_cd_m2,
            100.0 * r.rel_err,
            if ok { "ok" } else { "FAIL" }
        ));
    }
    let pass = within == results.len();
    lines.push(format!(
        "samples: {within} of {} within ±{} % (luminance ÷ pre[0] = {:.5e}) {}",
        results.len(),
        100.0 * tol,
        cap.header.pre[0],
        pass_word(pass)
    ));
    Ok(Verdict { lines, pass })
}

/// A radial profile's log–log slope and the number of 1-px annuli it was fitted over.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfileResult {
    pub slope: f64,
    pub bins: usize,
}

pub fn profile_slope(cap: &HdrCapture, feature: &str) -> anyhow::Result<ProfileResult> {
    let f = cap.feature(feature)?;
    let n = (f.r_max - f.r_min).ceil() as usize;
    let (mut sum_y, mut sum_r, mut count) = (vec![0.0; n], vec![0.0; n], vec![0u32; n]);
    for (x, y, _, _, rho) in cap.pixels_within(f.x, f.y, f.r_max) {
        if rho < f.r_min {
            continue;
        }
        let k = ((rho - f.r_min).floor() as usize).min(n - 1);
        sum_y[k] += cap.luminance(x, y);
        sum_r[k] += rho;
        count[k] += 1;
    }
    let points: Vec<(f64, f64)> = (0..n)
        .filter(|&k| count[k] > 0 && sum_y[k] > 0.0)
        .map(|k| {
            let c = f64::from(count[k]);
            ((sum_r[k] / c).ln(), (sum_y[k] / c).ln())
        })
        .collect();
    if points.len() < 3 {
        bail!(
            "feature {feature:?}: only {} annuli with light between {} and {} px",
            points.len(),
            f.r_min,
            f.r_max
        );
    }
    // Ordinary least squares slope.
    let m = points.len() as f64;
    let (mx, my) = (
        points.iter().map(|p| p.0).sum::<f64>() / m,
        points.iter().map(|p| p.1).sum::<f64>() / m,
    );
    let sxy: f64 = points.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let sxx: f64 = points.iter().map(|p| (p.0 - mx).powi(2)).sum();
    Ok(ProfileResult {
        slope: sxy / sxx,
        bins: points.len(),
    })
}

/// `profile`: the slope within `tol` of the feature's `expected_slope`.
pub fn check_profile(cap: &HdrCapture, feature: &str, tol: f64) -> anyhow::Result<Verdict> {
    let f = cap.feature(feature)?;
    let expected = f
        .expected_slope
        .with_context(|| format!("feature {feature:?} has no expected_slope"))?;
    let p = profile_slope(cap, feature)?;
    let pass = (p.slope - expected).abs() <= tol;
    Ok(Verdict {
        lines: vec![format!(
            "profile {feature}: log–log slope {:.4} over {} annuli ({}–{} px) vs {expected} ± {tol} {}",
            p.slope,
            p.bins,
            f.r_min,
            f.r_max,
            pass_word(pass)
        )],
        pass,
    })
}

/// A feature's brightness-weighted direction [deg] and how many pixels passed the colour test
/// (none: NaN).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleResult {
    pub angle_deg: f64,
    pub pixels: usize,
}

pub fn feature_angle(cap: &HdrCapture, feature: &str) -> anyhow::Result<AngleResult> {
    let f = cap.feature(feature)?;
    let (mut sx, mut sy, mut pixels) = (0.0, 0.0, 0);
    for (x, y, dx, dy, rho) in cap.pixels_within(f.x, f.y, f.r_max) {
        let rgb = cap.rgb(x, y);
        if rho < f.r_min || rho == 0.0 || !(rgb[2] > rgb[0]) {
            continue;
        }
        let w = luminance(rgb);
        if w > 0.0 {
            // Image y points down; angles are counter-clockwise with y up.
            sx += w * dx / rho;
            sy -= w * dy / rho;
            pixels += 1;
        }
    }
    let angle_deg = if pixels > 0 {
        sy.atan2(sx).to_degrees()
    } else {
        f64::NAN
    };
    Ok(AngleResult { angle_deg, pixels })
}

/// `angle`: the direction within `tol_deg` of the feature's `angle_deg`.
pub fn check_angle(cap: &HdrCapture, feature: &str, tol_deg: f64) -> anyhow::Result<Verdict> {
    let f = cap.feature(feature)?;
    let expected = f
        .angle_deg
        .with_context(|| format!("feature {feature:?} has no angle_deg"))?;
    let a = feature_angle(cap, feature)?;
    if a.pixels == 0 {
        return Ok(Verdict {
            lines: vec![format!(
                "angle {feature}: no pixel between {} and {} px passes the colour test B > R FAIL",
                f.r_min, f.r_max
            )],
            pass: false,
        });
    }
    let error = (a.angle_deg - expected + 180.0).rem_euclid(360.0) - 180.0;
    let pass = error.abs() <= tol_deg;
    Ok(Verdict {
        lines: vec![format!(
            "angle {feature}: {:.3}° from {} px with B > R ({}–{} px) vs {expected}° ± {tol_deg}° \
             (error {error:+.3}°) {}",
            a.angle_deg,
            a.pixels,
            f.r_min,
            f.r_max,
            pass_word(pass)
        )],
        pass,
    })
}

/// A star's measured illuminance and the terms behind it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluxResult {
    /// E = Σ (Y_i − Y_bg)·Ω_i ÷ pre[1] [lx].
    pub lux: f64,
    /// The annulus median, pre-exposed.
    pub y_bg: f64,
    /// Σ Y_i·Ω_i ÷ pre[1] [lx], before the background is subtracted.
    pub raw_lux: f64,
    /// The background's share of the raw sum: Y_bg·ΣΩ_i / ΣY_i·Ω_i.
    pub bg_share: f64,
    /// Pixels inside R.
    pub pixels: usize,
    /// Unmasked annulus pixels behind the median.
    pub annulus_pixels: usize,
}

pub fn flux(cap: &HdrCapture, name: &str) -> anyhow::Result<FluxResult> {
    let e = cap
        .flux
        .iter()
        .find(|e| e.name == name)
        .with_context(|| format!("no flux entry {name:?} in the capture JSON"))?;
    if !(e.radius_px > 0.0 && e.radius_px.is_finite()) {
        bail!("flux entry {name:?} has radius_px {}", e.radius_px);
    }
    let h = &cap.header;
    let s = 2.0 * (h.fov_y_rad / 2.0).tan() / f64::from(h.height);
    let omega_c = s * s;
    let (icx, icy) = (f64::from(h.width) / 2.0, f64::from(h.height) / 2.0);
    let (mut sum_y_omega, mut sum_omega, mut pixels) = (0.0, 0.0, 0);
    let mut annulus = Vec::new();
    for (x, y, dx, dy, rho) in cap.pixels_within(e.x, e.y, 1.5 * e.radius_px) {
        let (cx, cy) = (e.x + dx, e.y + dy);
        if rho < e.radius_px {
            let d = (cx - icx).hypot(cy - icy);
            let cos_theta = 1.0 / (1.0 + (s * d).powi(2)).sqrt();
            let omega = omega_c * cos_theta.powi(3);
            sum_y_omega += cap.luminance(x, y) * omega;
            sum_omega += omega;
            pixels += 1;
        } else if !e.mask.iter().any(|m| (cx - m[0]).hypot(cy - m[1]) < m[2]) {
            annulus.push(cap.luminance(x, y));
        }
    }
    if annulus.is_empty() {
        bail!("flux entry {name:?}: no unmasked pixel in the annulus R ≤ ρ < 1.5 R");
    }
    let y_bg = median(&mut annulus);
    let pre = h.pre[1];
    let raw_lux = sum_y_omega / pre;
    let bg_lux = y_bg * sum_omega / pre;
    Ok(FluxResult {
        lux: raw_lux - bg_lux,
        y_bg,
        raw_lux,
        bg_share: bg_lux / raw_lux,
        pixels,
        annulus_pixels: annulus.len(),
    })
}

/// `flux`: E within the relative tolerance `tol` of the entry's `expected_lux`.
pub fn check_flux(cap: &HdrCapture, name: &str, tol: f64) -> anyhow::Result<Verdict> {
    let f = flux(cap, name)?;
    let e = cap
        .flux
        .iter()
        .find(|e| e.name == name)
        .with_context(|| format!("no flux entry {name:?} in the capture JSON"))?;
    let rel = f.lux / e.expected_lux - 1.0;
    let pass = rel.abs() <= tol;
    let pre = cap.header.pre[1];
    Ok(Verdict {
        lines: vec![
            format!(
                "flux {name}: E {:.5e} lx, expected {:.5e} lx ({:+.2} %); {} px inside R = {} px",
                f.lux,
                e.expected_lux,
                100.0 * rel,
                f.pixels,
                e.radius_px
            ),
            format!(
                "  Y_bg {:.5e} pre-exposed ({:.5e} cd/m², median of {} annulus px, {} masked \
                 footprints); background share of the sum {:.2} %",
                f.y_bg,
                f.y_bg / pre,
                f.annulus_pixels,
                e.mask.len(),
                100.0 * f.bg_share
            ),
            format!("flux within ±{} % {}", 100.0 * tol, pass_word(pass)),
        ],
        pass,
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::fs;

    /// A fresh, empty directory under the system temp dir, unique to this process and test.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uni-verify-{}-{test}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// An in-memory capture of `width`×`height` black pixels with the given header numbers.
    pub(crate) fn blank(width: u32, height: u32, fov_y_rad: f64, pre: [f64; 2]) -> HdrCapture {
        HdrCapture {
            header: HdrHeader {
                file: PathBuf::from("blank.hdr.bin"),
                width,
                height,
                fov_y_rad,
                pre,
                ev100: None,
            },
            pixels: vec![[0.0, 0.0, 0.0, 1.0]; width as usize * height as usize],
            regions: BTreeMap::new(),
            samples: Vec::new(),
            features: BTreeMap::new(),
            flux: Vec::new(),
        }
    }

    fn put(cap: &mut HdrCapture, x: u32, y: u32, rgb: [f32; 3]) {
        let i = y as usize * cap.header.width as usize + x as usize;
        cap.pixels[i] = [rgb[0], rgb[1], rgb[2], 1.0];
    }

    fn add(cap: &mut HdrCapture, x: i64, y: i64, rgb: [f64; 3]) {
        let (w, h) = (i64::from(cap.header.width), i64::from(cap.header.height));
        if (0..w).contains(&x) && (0..h).contains(&y) {
            let p = &mut cap.pixels[(y * w + x) as usize];
            for c in 0..3 {
                p[c] += rgb[c] as f32;
            }
        }
    }

    /// Rec. 709 luminance (ITU-R BT.709-6 item 3.2).
    fn y709(rgb: [f64; 3]) -> f64 {
        0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    }

    fn close(got: f64, want: f64, tol: f64, what: &str) {
        assert!((got - want).abs() <= tol, "{what}: {got} vs {want}");
    }

    #[test]
    fn hdr_ratio_stats_samples() {
        // An 8×4 sidecar on disk, read through the capture JSON's `hdr` header.
        let dir = temp_dir("hdr_ratio_stats_samples");
        let (w, h) = (8u32, 4u32);
        let mut px = vec![[0.0f32, 0.0, 0.0, 1.0]; (w * h) as usize];
        let mut set = |x: u32, y: u32, rgb: [f32; 3]| {
            px[(y * w + x) as usize] = [rgb[0], rgb[1], rgb[2], 1.0];
        };
        // Region "a": three pixels; region "b": four.
        set(0, 0, [0.30, 0.20, 0.10]);
        set(1, 0, [0.60, 0.40, 0.20]);
        set(2, 0, [0.90, 0.80, 0.30]);
        set(0, 1, [0.10, 0.10, 0.40]);
        set(1, 1, [0.20, 0.10, 0.20]);
        set(2, 1, [0.10, 0.30, 0.50]);
        set(3, 1, [0.40, 0.20, 0.60]);
        // Two samples, world light at pre[0] = 0.25.
        set(5, 3, [0.25, 0.50, 0.75]);
        set(6, 2, [1.00, 1.00, 1.00]);
        let bytes: Vec<u8> = px
            .iter()
            .flat_map(|p| p.iter().flat_map(|c| c.to_le_bytes()))
            .collect();
        fs::write(dir.join("t.hdr.bin"), bytes).expect("write sidecar");
        let json = dir.join("t.json");
        fs::write(
            &json,
            r#"{"name": "t", "width": 8, "height": 4,
                "hdr": {"file": "t.hdr.bin", "width": 8, "height": 4, "fov_y_rad": 1.0471975511965976,
                        "pre": [0.25, 2.0], "ev100": [1.0, -4.0], "layout": "f32 LE RGBA, rows top-down"},
                "regions": {"a": [[0, 0], [1, 0], [2, 0]], "b": [[0, 1], [1, 1], [2, 1], [3, 1]]},
                "samples": [{"x": 5, "y": 3, "expected_cd_m2": [1.0, 2.0, 3.0], "what": "exact"},
                            {"x": 6.5, "y": 2.5, "expected_cd_m2": [4.4, 4.4, 4.4], "what": "10% high"}]}"#,
        )
        .expect("write JSON");
        let cap = HdrCapture::load(&json);
        let missing = HdrCapture::load(&dir.join("absent.json"));
        let _ = fs::remove_dir_all(&dir);
        let cap = cap.expect("load capture");
        assert!(missing.is_err());
        assert_eq!((cap.header.width, cap.header.height), (8, 4));
        assert_eq!(cap.header.pre, [0.25, 2.0]);

        let ya = [
            y709([0.30, 0.20, 0.10]),
            y709([0.60, 0.40, 0.20]),
            y709([0.90, 0.80, 0.30]),
        ];
        let yb = [
            y709([0.10, 0.10, 0.40]),
            y709([0.20, 0.10, 0.20]),
            y709([0.10, 0.30, 0.50]),
            y709([0.40, 0.20, 0.60]),
        ];
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let args = |channels, stat| RatioArgs {
            num: "a".into(),
            den: "b".into(),
            channels,
            stat,
            bounds: Bounds {
                min: Some(0.0),
                max: None,
                expect: None,
            },
        };
        let r = ratio(&cap, &args(Channels::Luma, Stat::Mean)).expect("ratio");
        close(r.num, mean(&ya), 1e-6, "mean luma a");
        close(r.ratio, mean(&ya) / mean(&yb), 1e-6, "luma mean ratio");
        // Median of an even count is the mean of the middle two: B over b is 0.20, 0.40, 0.50,
        // 0.60 → 0.45; R over a is 0.30, 0.60, 0.90 → 0.60.
        let r = ratio(&cap, &args(Channels::RedOverBlue, Stat::Median)).expect("ratio");
        close(r.num, 0.60, 1e-6, "median R of a");
        close(r.den, 0.45, 1e-6, "median B of b");
        close(r.ratio, 0.60 / 0.45, 1e-6, "R/B median ratio");
        let r = ratio(&cap, &args(Channels::Green, Stat::Mean)).expect("ratio");
        close(r.ratio, (1.4 / 3.0) / (0.7 / 4.0), 1e-6, "G mean ratio");
        // The same region over itself in colour: B/R of b by medians, 0.45 / 0.15.
        let self_ratio = RatioArgs {
            num: "b".into(),
            den: "b".into(),
            ..args(Channels::BlueOverRed, Stat::Median)
        };
        close(
            ratio(&cap, &self_ratio).expect("ratio").ratio,
            0.45 / 0.15,
            1e-6,
            "B/R of b",
        );
        assert!(check_ratio(&cap, &self_ratio).expect("check").pass);
        let strict = RatioArgs {
            bounds: Bounds {
                min: None,
                max: None,
                expect: Some((3.2, 0.05)),
            },
            ..self_ratio.clone()
        };
        let v = check_ratio(&cap, &strict).expect("check");
        assert!(
            !v.pass && v.lines.last().is_some_and(|l| l.ends_with("FAIL")),
            "{v:?}"
        );
        let unknown = RatioArgs {
            num: "nope".into(),
            ..self_ratio
        };
        assert!(ratio(&cap, &unknown).is_err());

        // Stats: relative RMS (population) of luminance, mean of pre-exposed luminance.
        let s = region_stats(&cap, "b").expect("stats");
        let m = mean(&yb);
        let rms = (yb.iter().map(|y| (y - m).powi(2)).sum::<f64>() / 4.0).sqrt() / m;
        assert_eq!(s.count, 4);
        close(s.mean_pre, m, 1e-6, "mean pre-exposed");
        close(s.rel_rms, rms, 1e-6, "relative RMS");
        let v = check_stats(
            &cap,
            &StatsArgs {
                region: "b".into(),
                rms_min: Some(rms - 0.01),
                rms_max: Some(rms + 0.01),
                mean_pre_min: Some(m - 0.01),
            },
        )
        .expect("check");
        assert!(v.pass, "{v:?}");
        let v = check_stats(
            &cap,
            &StatsArgs {
                region: "b".into(),
                rms_min: None,
                rms_max: None,
                mean_pre_min: Some(m + 0.01),
            },
        )
        .expect("check");
        assert!(!v.pass, "{v:?}");

        // Samples: luminance ÷ pre[0] against the expected colour's luminance.
        let s = samples(&cap).expect("samples");
        assert_eq!(s.len(), 2);
        close(s[0].measured_cd_m2, y709([1.0, 2.0, 3.0]), 1e-6, "sample 0");
        close(
            s[0].expected_cd_m2,
            y709([1.0, 2.0, 3.0]),
            1e-6,
            "sample 0 expected",
        );
        close(s[0].rel_err, 0.0, 1e-6, "sample 0 error");
        close(s[1].measured_cd_m2, 4.0, 1e-6, "sample 1");
        close(s[1].rel_err, 4.0 / 4.4 - 1.0, 1e-6, "sample 1 error");
        assert!(check_samples(&cap, 0.10).expect("check").pass);
        assert!(!check_samples(&cap, 0.05).expect("check").pass);
    }

    #[test]
    fn hdr_profile_and_angle() {
        let (w, h) = (241u32, 241u32);
        let (cx, cy) = (120.0, 120.0);
        let coma = |x: u32, y: u32| {
            let rho = (f64::from(x) + 0.5 - cx).hypot(f64::from(y) + 0.5 - cy);
            (100.0 / rho) as f32
        };
        // Profile: a neutral 1/ρ coma, slope −1.
        let mut cap = blank(w, h, 0.35, [1.0, 1.0]);
        for y in 0..h {
            for x in 0..w {
                let v = coma(x, y);
                put(&mut cap, x, y, [v, v, v]);
            }
        }
        cap.features.insert(
            "coma".into(),
            Feature {
                x: cx,
                y: cy,
                r_min: 8.0,
                r_max: 40.0,
                angle_deg: None,
                expected_slope: Some(-1.0),
            },
        );
        let p = profile_slope(&cap, "coma").expect("profile");
        close(p.slope, -1.0, 0.02, "coma slope");
        assert!(check_profile(&cap, "coma", 0.1).expect("check").pass);
        assert!(profile_slope(&cap, "tail").is_err());

        // Angle: the same coma (neutral, so it fails the colour test), a straight blue streak at
        // 37° (counter-clockwise from +x, y up) and a yellow dust tail at 140° that must be
        // ignored.
        let streak = |angle_deg: f64, rgb: [f64; 3], cap: &mut HdrCapture| {
            let (s, c) = angle_deg.to_radians().sin_cos();
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (f64::from(x) + 0.5 - cx, -(f64::from(y) + 0.5 - cy));
                    let along = dx * c + dy * s;
                    let across = -dx * s + dy * c;
                    if along > 2.0 {
                        let k = (-across * across / (2.0 * 1.5 * 1.5)).exp();
                        add(cap, i64::from(x), i64::from(y), rgb.map(|v| v * k));
                    }
                }
            }
        };
        streak(37.0, [0.05, 0.3, 2.0], &mut cap);
        streak(140.0, [3.0, 2.5, 0.6], &mut cap);
        cap.features.insert(
            "ion_tail".into(),
            Feature {
                x: cx,
                y: cy,
                r_min: 10.0,
                r_max: 110.0,
                angle_deg: Some(37.0),
                expected_slope: None,
            },
        );
        let a = feature_angle(&cap, "ion_tail").expect("angle");
        close(a.angle_deg, 37.0, 0.2, "ion tail angle");
        assert!(check_angle(&cap, "ion_tail", 1.0).expect("check").pass);
        // A feature without its expectation is an error, not a pass.
        assert!(check_angle(&cap, "coma", 1.0).is_err());
    }

    // ---- Test fixture: a copy of uni-sky's PSF CPU twin (crates/uni-sky/src/psf.rs at
    // 58374d2): Gaussian core σ 0.65 px integrated over pixels with erf differences
    // (Abramowitz & Stegun 1964, eq. 7.1.26), and a Moffat (1969) halo carrying 2 % of the
    // energy of stars above 1e-6 lx, lowered to 0 at the footprint radius and renormalised
    // inside it. ----

    const CORE_SIGMA_PX: f32 = 0.65;
    const HALO_ENERGY_FRACTION: f32 = 0.02;
    const HALO_MIN_LUX: f32 = 1.0e-6;
    const HALO_ALPHA_PX: f32 = 4.0;
    const HALO_BETA: f32 = 2.5;
    const MIN_RADIUS_PX: f32 = 3.0;
    const MAX_RADIUS_PX: f32 = 64.0;
    const VISIBILITY_CUTOFF: f32 = 3.0e-5;

    #[allow(clippy::excessive_precision)]
    fn erf(x: f32) -> f32 {
        const P: f32 = 0.327_591_1;
        const A1: f32 = 0.254_829_592;
        const A2: f32 = -0.284_496_736;
        const A3: f32 = 1.421_413_741;
        const A4: f32 = -1.453_152_027;
        const A5: f32 = 1.061_405_429;
        let a = x.abs();
        let t = 1.0 / (1.0 + P * a);
        let poly = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5))));
        let y = 1.0 - poly * (-a * a).exp();
        if x >= 0.0 { y } else { -y }
    }

    fn pixel_coverage(edge_px: f32, sigma_px: f32) -> f32 {
        let k = 1.0 / (std::f32::consts::SQRT_2 * sigma_px);
        (0.5 * (erf((edge_px + 1.0) * k) - erf(edge_px * k))).max(0.0)
    }

    fn halo_fraction(lux: f32) -> f32 {
        if lux > HALO_MIN_LUX {
            HALO_ENERGY_FRACTION
        } else {
            0.0
        }
    }

    fn moffat(r_px: f32) -> f32 {
        let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
        (HALO_BETA - 1.0) / (std::f32::consts::PI * a2) * (1.0 + r_px * r_px / a2).powf(-HALO_BETA)
    }

    fn halo_weight(r_px: f32, radius_px: f32) -> f32 {
        if !(r_px < radius_px) {
            return 0.0;
        }
        let a2 = HALO_ALPHA_PX * HALO_ALPHA_PX;
        let edge = moffat(radius_px);
        let inside = 1.0
            - (1.0 + radius_px * radius_px / a2).powf(1.0 - HALO_BETA)
            - std::f32::consts::PI * radius_px * radius_px * edge;
        (moffat(r_px) - edge) / inside
    }

    fn footprint_radius_px(energy: f32, halo_fraction: f32) -> f32 {
        let mut r = MIN_RADIUS_PX;
        let core_peak = (1.0 - halo_fraction) * energy
            / (std::f32::consts::TAU * CORE_SIGMA_PX * CORE_SIGMA_PX);
        if core_peak > VISIBILITY_CUTOFF {
            let r_core = CORE_SIGMA_PX * (2.0 * (core_peak / VISIBILITY_CUTOFF).ln()).sqrt();
            r = r.max(r_core);
        }
        let halo_peak = halo_fraction * energy * moffat(0.0);
        if halo_peak > VISIBILITY_CUTOFF {
            let q = (halo_peak / VISIBILITY_CUTOFF).powf(1.0 / HALO_BETA);
            r = r.max(HALO_ALPHA_PX * (q - 1.0).sqrt());
        }
        r.min(MAX_RADIUS_PX)
    }

    /// Ω_c = s², s = 2·tan(fov_y/2)/H [sr] (uni_sky::stars::pixel_solid_angle).
    fn omega_c(cap: &HdrCapture) -> f64 {
        let s = 2.0 * (cap.header.fov_y_rad / 2.0).tan() / f64::from(cap.header.height);
        s * s
    }

    /// cos θ of the direction through the image point (x, y) (rectilinear projection).
    fn cos_theta(cap: &HdrCapture, x: f64, y: f64) -> f64 {
        let s = omega_c(cap).sqrt();
        let d =
            (x - f64::from(cap.header.width) / 2.0).hypot(y - f64::from(cap.header.height) / 2.0);
        1.0 / (1.0 + (s * d).powi(2)).sqrt()
    }

    /// Splats a star of `lux` [lx] at (x, y) the way `stars.wgsl` does: display energy
    /// E·pre[1]/(Ω_c·cos³θ) with θ at the star's centre, a Gaussian core integrated over each
    /// pixel plus the halo, over its footprint radius R. Returns R.
    fn splat(cap: &mut HdrCapture, x: f64, y: f64, lux: f64, colour: [f64; 3]) -> f32 {
        let ct = cos_theta(cap, x, y);
        let energy = lux * cap.header.pre[1] / (omega_c(cap) * ct.powi(3));
        let h = halo_fraction(lux as f32);
        let radius = footprint_radius_px(energy as f32, h);
        let n = radius.ceil() as i64;
        let (px, py) = (x.floor() as i64, y.floor() as i64);
        let sub = ((x - x.floor()) as f32, (y - y.floor()) as f32);
        for dy in -n..=n {
            for dx in -n..=n {
                let core = pixel_coverage(dx as f32 - sub.0, CORE_SIGMA_PX)
                    * pixel_coverage(dy as f32 - sub.1, CORE_SIGMA_PX);
                let centre = ((px + dx) as f64 + 0.5, (py + dy) as f64 + 0.5);
                let r = (centre.0 - x).hypot(centre.1 - y) as f32;
                let w = (1.0 - h) * core + h * halo_weight(r, radius);
                add(
                    cap,
                    px + dx,
                    py + dy,
                    colour.map(|c| c * energy * f64::from(w)),
                );
            }
        }
        radius
    }

    /// A Planck-like orange with Y = 1, as the star colours have.
    fn star_colour() -> [f64; 3] {
        let rgb = [1.3, 0.95, 0.7];
        let y = y709(rgb);
        rgb.map(|c| c / y)
    }

    /// Rigel's illuminance [lx] (V 0.13; `uni_core::star_illuminance_lux`, as the brief quotes).
    const RIGEL_LUX: f64 = 2.249e-6;
    /// Star pre-exposure at EV −4: 1/(1.2 × 2⁻⁴).
    const STAR_PRE: f64 = 1.0 / (1.2 * 0.0625);

    #[test]
    fn hdr_flux() {
        // On the axis, on black: the summed energy is the star's illuminance.
        let mut cap = blank(1920, 1080, 60f64.to_radians(), [1e-4, STAR_PRE]);
        let (x, y) = (960.37, 540.81);
        let radius = splat(&mut cap, x, y, RIGEL_LUX, star_colour());
        cap.flux.push(FluxEntry {
            name: "Rigel".into(),
            x,
            y,
            radius_px: f64::from(radius),
            expected_lux: RIGEL_LUX,
            mask: Vec::new(),
        });
        let f = flux(&cap, "Rigel").expect("flux");
        close(f.lux / RIGEL_LUX, 1.0, 0.01, "on-axis flux");
        assert_eq!(f.y_bg, 0.0);
        assert!(check_flux(&cap, "Rigel", 0.05).expect("check").pass);
        assert!(flux(&cap, "Betelgeuse").is_err());
    }

    /// The flux with Ω_c in place of Ω_c·cos³θ (the mistake the brief guards against).
    fn flux_with_central_omega(cap: &HdrCapture, e: &FluxEntry, y_bg: f64) -> f64 {
        let mut sum = 0.0;
        for (i, p) in cap.pixels.iter().enumerate() {
            let x = (i % cap.header.width as usize) as f64 + 0.5;
            let y = (i / cap.header.width as usize) as f64 + 0.5;
            if (x - e.x).hypot(y - e.y) < e.radius_px {
                let lum = y709([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
                sum += (lum - y_bg) * omega_c(cap);
            }
        }
        sum / cap.header.pre[1]
    }

    #[test]
    fn hdr_flux_sky_and_off_axis() {
        // Re-review r2 NPA: fov 60°, 1920×1080, a uniform 1.4e-2 sky (20 mag/arcsec² at star
        // EV −4), Rigel-like splats on the axis and 9° off it, and a fainter star in the annulus
        // whose own footprint stays outside R (re-review r3 minor 3).
        let off_axis_px = 9f64.to_radians().tan() / (2.0 * 30f64.to_radians().tan() / 1080.0);
        close(off_axis_px, 148.1, 0.05, "9° off the axis");
        for (label, x, y) in [
            ("on axis", 960.37, 540.81),
            (
                "9° off axis",
                960.0 + off_axis_px * 0.6,
                540.0 - off_axis_px * 0.8,
            ),
        ] {
            let mut cap = blank(1920, 1080, 60f64.to_radians(), [1e-4, STAR_PRE]);
            for p in &mut cap.pixels {
                *p = [1.4e-2, 1.4e-2, 1.4e-2, 1.0];
            }
            let radius = f64::from(splat(&mut cap, x, y, RIGEL_LUX, star_colour()));
            close(radius, 13.5, 0.15, "Rigel's footprint");
            // The neighbour: 1e-7 lx (no halo, the 3-px minimum footprint), 18 px away, so its
            // footprint spans 15–21 px from Rigel: in the annulus, outside R.
            let (nx, ny) = (x + 18.0 * 0.8, y + 18.0 * 0.6);
            let n_radius = f64::from(splat(&mut cap, nx, ny, 1e-7, star_colour()));
            assert!(18.0 - n_radius >= radius, "neighbour reaches inside R");
            let entry = FluxEntry {
                name: "Rigel".into(),
                x,
                y,
                radius_px: radius,
                expected_lux: RIGEL_LUX,
                mask: vec![[nx, ny, n_radius]],
            };
            cap.flux.push(entry.clone());
            let f = flux(&cap, "Rigel").expect("flux");
            eprintln!(
                "hdr_flux_sky_and_off_axis {label}: E/expected {:.5}, Y_bg {:.6}, background share {:.4}",
                f.lux / RIGEL_LUX,
                f.y_bg,
                f.bg_share
            );
            close(f.lux / RIGEL_LUX, 1.0, 0.01, label);
            close(f.y_bg, 1.4e-2, 1e-6, "annulus median");
            assert!(check_flux(&cap, "Rigel", 0.05).expect("check").pass);
            // Guards. No subtraction reads ≈ +29.5 % (1.4e-2 × π × 13.5² / 27.2).
            close(f.raw_lux / RIGEL_LUX, 1.295, 0.02, "unsubtracted");
            if label == "9° off axis" {
                // Ω_c alone reads 1/cos³ 9° = 1.0379 high.
                let wrong = flux_with_central_omega(&cap, &entry, f.y_bg);
                close(wrong / RIGEL_LUX, 1.0379, 0.003, "Ω_c without cos³θ");
            }
        }
    }
}
