//! Photometric checks on the `--hdr` capture sidecar (M1 Task B8; review I16, round 2).
//!
//! `universe --capture --hdr` writes `<name>.hdr.bin` (little-endian f32 RGBA of the pre-exposed
//! HDR target, rows top-down, copied before bloom) and, in the capture JSON (B9 Step 2):
//! - `hdr: {file, width, height, fov_y_rad, pre: [scene, star], ev100: [scene, star], layout}`;
//!   `file` is relative to the JSON's directory, `pre` the exposure factors 1/(1.2·2^EV);
//! - `regions: {name: [[x, y], …]}`, pixel samples (column, row; fractional values are floored);
//! - `samples: [{x, y, expected_cd_m2: [r, g, b], what, footprint?: [[x, y], …]}]`;
//! - `features: {name: {x, y, r_min, r_max, angle_deg?, expected_slope?}}`;
//! - `flux: [{name, x, y, radius_px, expected_lux, neighbour_lux, mask: [[x, y, r], …]}]`;
//! - `limbs: [{name, bins: [{psi_deg: [from, to], strip: [[x, y, expected_cd_m2, sky_t], …],
//!   sky: [[x, y], …]}]}]`.
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
//! - `samples`: each sample's luminance ÷ `pre[0]` against the luminance of `expected_cd_m2`,
//!   as a relative error. With a `footprint` (the pixels that see the map patch the expectation
//!   averages: 3 × 3 texels of the 2k copy), the measured value is the mean over those pixels,
//!   since the GPU samples finer mips that resolve detail the patch averages away (lead ruling
//!   "row 5 Mars samples"); without one, the sample's own pixel.
//! - `profile`: the least-squares slope of ln(mean Y) against ln(mean ρ) over 1-px annuli from
//!   `r_min` to `r_max` around the feature, against its `expected_slope`.
//! - `angle`: the direction of a straight blue feature from the centre, against `angle_deg`
//!   (Task C6). For each ray from the centre (every 0.25°) the median of the blue excess B − R
//!   over the 1-px steps from `r_min` to `r_max`; the feature is the ray whose median stands
//!   highest above the median of all rays (the CO⁺ ion tail is bluer than the solar-coloured
//!   dust, coma and band), refined by the excess-weighted mean of the rays within 1.5° of it. A
//!   field star crosses a ray at a few radii only, so the median ignores it; B8's first
//!   estimator, the direction of Σ Y·ρ̂ over the annulus's B > R pixels, is still printed, but at
//!   `comet-neowise-2020` the field's blue stars (9 000 pixels) outweighed the 1.5-px ion tail
//!   and put it 40° off. Angles are degrees counter-clockwise from the image's +x axis with y
//!   up.
//! - `flux`: E = Σ_{ρ_i < R} (Y_i − Y_bg)·Ω_i ÷ `pre[1]` against `expected_lux` (re-review r2
//!   NPA). Ω_i = Ω_c·cos³θ_i is the solid angle of pixel i, the `uni_sky::stars` convention
//!   (Ω_c = s², s = 2·tan(fov_y/2)/H; cos θ_i = 1/√(1 + (s·d_i)²), d_i the pixel centre's distance
//!   from the image centre (W/2, H/2); dω = dA·cosθ/r², Pharr, Jakob & Humphreys 2016 §5.5.3).
//!   R is the entry's `radius_px`, the footprint the star shader used, which holds all of the
//!   PSF's energy. Y_bg is the median of Y over the annulus R ≤ ρ < 1.5 R, skipping pixels inside
//!   any `mask` disk, so the band behind the star is subtracted. The expectation is the star's
//!   own `expected_lux` plus `neighbour_lux` (lead ruling "row 15 flux", Wave B gate): the drawn
//!   illuminance of every other star or point whose footprint centre lies inside R, which B4L's
//!   wider halo footprints put inside R (18 % of Rigel's light at `sky-orion-3d`). A capture JSON
//!   without `neighbour_lux` reads 0.
//! - `limb`: the world light integrated across a limb (lead ruling "row 13 Venus haze"; fix
//!   round 1, review I1): E = Σ_bins Σ_strip (Y_i − Y_sky,bin·t_i)·Ω_i ÷ `pre[0]` against the CPU
//!   twin's Σ L_i·Ω_i, both in lux. The entry holds only the azimuth bins where the twin expects
//!   light (Venus: the haze's two cusp arcs), so the subtracted sky is a minor part of the sum.
//!   Y_sky,bin is the median of Y over the bin's own `sky` pixels (the band and stars just
//!   beyond the limb, pre-exposed at the star exposure; a bin without one takes the nearest
//!   bin's), so a patchy band is estimated where it lies; t_i, the pixel's sky share, is what
//!   the passes keep of the sky behind it (the view transmittance's luminance, 0 on the body).
//!   Ω_i as for `flux`. Integrating every pixel across the limb counts a haze band thinner than
//!   a pixel by its light, not by where its sub-pixel peak lands. The output gives the sky's
//!   share of the measured sum and the twin's share on the body (t = 0).

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
    /// The pixels to average (see the module docs); empty: the sample's own pixel.
    #[serde(default)]
    pub footprint: Vec<[f64; 2]>,
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
/// illuminance [lx], the illuminance of the other sources whose footprint centre lies inside R
/// [lx], and the other footprints [x, y, r] to leave out of the background annulus.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct FluxEntry {
    pub name: String,
    pub x: f64,
    pub y: f64,
    pub radius_px: f64,
    pub expected_lux: f64,
    /// Light of the other stars and points centred inside R [lx] (0 when absent).
    #[serde(default)]
    pub neighbour_lux: f64,
    #[serde(default)]
    pub mask: Vec<[f64; 3]>,
}

impl FluxEntry {
    /// What the sum inside R should read [lx]: the star's own illuminance plus its neighbours'
    /// (lead ruling "row 15 flux").
    pub fn expected_total_lux(&self) -> f64 {
        self.expected_lux + self.neighbour_lux
    }
}

/// One `limbs` entry: the azimuth bins of a limb where the twin expects light.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LimbEntry {
    pub name: String,
    pub bins: Vec<LimbBin>,
}

/// One azimuth bin: the pixels across the limb with the twin's world light and their sky
/// share, and the bin's sky pixels beyond it.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct LimbBin {
    /// The bin's image angle range [deg] (reported only).
    #[serde(default)]
    pub psi_deg: Option<[f64; 2]>,
    /// [x, y, expected luminance [cd/m²] (world light), sky share t].
    pub strip: Vec<[f64; 4]>,
    #[serde(default)]
    pub sky: Vec<[f64; 2]>,
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
    #[serde(default)]
    limbs: Vec<LimbEntry>,
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
    pub limbs: Vec<LimbEntry>,
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
            limbs: json.limbs,
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
    /// Pixels averaged (1 without a footprint).
    pub pixels: usize,
}

pub fn samples(cap: &HdrCapture) -> anyhow::Result<Vec<SampleResult>> {
    if cap.samples.is_empty() {
        bail!("no `samples` in the capture JSON");
    }
    let pre = cap.header.pre[0];
    cap.samples
        .iter()
        .map(|s| {
            let own = [[s.x, s.y]];
            let points: &[[f64; 2]] = if s.footprint.is_empty() {
                &own
            } else {
                &s.footprint
            };
            let lums = points
                .iter()
                .map(|&[px, py]| {
                    let (x, y) = cap.pixel_at(px, py).with_context(|| {
                        format!("sample {:?} pixel ({px}, {py}) is off the image", s.what)
                    })?;
                    Ok(cap.luminance(x, y))
                })
                .collect::<anyhow::Result<Vec<f64>>>()?;
            let measured = mean(&lums) / pre;
            let expected = luminance(s.expected_cd_m2);
            Ok(SampleResult {
                what: s.what.clone(),
                x: s.x,
                y: s.y,
                measured_cd_m2: measured,
                expected_cd_m2: expected,
                rel_err: measured / expected - 1.0,
                pixels: lums.len(),
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
            "  {} ({}, {}): {:.5e} cd/m² (mean of {} px) vs {:.5e} cd/m² ({:+.2} %) {}",
            r.what,
            r.x,
            r.y,
            r.measured_cd_m2,
            r.pixels,
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

/// Ray step of [`feature_angle`] [deg].
const RAY_STEP_DEG: f64 = 0.25;
/// Half-width of the refinement around the best ray [deg].
const RAY_REFINE_DEG: f64 = 1.5;

/// The direction of the straight blue feature `feature` (module docs: the ray-median
/// estimator) and the number of samples on its ray that stand above the rays' median; NaN and 0
/// when no ray stands above it.
pub fn feature_angle(cap: &HdrCapture, feature: &str) -> anyhow::Result<AngleResult> {
    let f = cap.feature(feature)?;
    let n = (360.0 / RAY_STEP_DEG).round() as usize;
    let steps = (f.r_max - f.r_min).floor().max(1.0) as usize;
    let excess = |rgb: [f64; 3]| rgb[2] - rgb[0];
    let mut scores = vec![f64::NAN; n];
    let mut on_ray: Vec<Vec<f64>> = vec![Vec::new(); n];
    for (k, score) in scores.iter_mut().enumerate() {
        let (sn, cs) = (k as f64 * RAY_STEP_DEG).to_radians().sin_cos();
        let mut v: Vec<f64> = (0..steps)
            .filter_map(|i| {
                let r = f.r_min + 0.5 + i as f64;
                // Image y points down; angles are counter-clockwise with y up.
                let (x, y) = cap.pixel_at(f.x + r * cs, f.y - r * sn)?;
                Some(excess(cap.rgb(x, y)))
            })
            .collect();
        if 2 * v.len() < steps {
            continue;
        }
        v.sort_by(f64::total_cmp);
        *score = v[v.len() / 2];
        on_ray[k] = v;
    }
    let mut valid: Vec<f64> = scores.iter().copied().filter(|s| s.is_finite()).collect();
    if valid.is_empty() {
        return Ok(AngleResult {
            angle_deg: f64::NAN,
            pixels: 0,
        });
    }
    valid.sort_by(f64::total_cmp);
    let base = valid[valid.len() / 2];
    let Some((best, peak)) = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(k, &s)| (k, s - base))
    else {
        return Ok(AngleResult {
            angle_deg: f64::NAN,
            pixels: 0,
        });
    };
    if !(peak > 0.0) {
        return Ok(AngleResult {
            angle_deg: f64::NAN,
            pixels: 0,
        });
    }
    // Excess-weighted mean of the rays near the best one (on the circle, about it).
    let half = (RAY_REFINE_DEG / RAY_STEP_DEG).round() as i64;
    let (mut sw, mut swa) = (0.0, 0.0);
    for d in -half..=half {
        let k = (best as i64 + d).rem_euclid(n as i64) as usize;
        let e = scores[k] - base;
        if e.is_finite() && e > 0.0 {
            sw += e;
            swa += e * d as f64 * RAY_STEP_DEG;
        }
    }
    let angle = best as f64 * RAY_STEP_DEG + swa / sw;
    let pixels = on_ray[best].iter().filter(|&&v| v > base).count();
    Ok(AngleResult {
        angle_deg: (angle + 180.0).rem_euclid(360.0) - 180.0,
        pixels,
    })
}

/// B8's first `angle` estimator, printed beside [`feature_angle`]'s: the direction of Σ Y·ρ̂
/// over the annulus's pixels with B > R.
pub fn weighted_angle(cap: &HdrCapture, feature: &str) -> anyhow::Result<AngleResult> {
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
    let old = weighted_angle(cap, feature)?;
    let first = format!(
        "  B8's Σ Y·ρ̂ over the B > R pixels: {:.3}° from {} px (not the verdict; module docs)",
        old.angle_deg, old.pixels
    );
    if a.pixels == 0 {
        return Ok(Verdict {
            lines: vec![
                first,
                format!(
                    "angle {feature}: no ray between {} and {} px stands above the rays' median \
                     blue excess FAIL",
                    f.r_min, f.r_max
                ),
            ],
            pass: false,
        });
    }
    let error = (a.angle_deg - expected + 180.0).rem_euclid(360.0) - 180.0;
    let pass = error.abs() <= tol_deg;
    Ok(Verdict {
        lines: vec![
            first,
            format!(
                "angle {feature}: {:.3}° (ray median of B − R, {} samples above the field, {}–{} \
                 px) vs {expected}° ± {tol_deg}° (error {error:+.3}°) {}",
                a.angle_deg,
                a.pixels,
                f.r_min,
                f.r_max,
                pass_word(pass)
            ),
        ],
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

/// `flux`: E within the relative tolerance `tol` of the entry's expected total, `expected_lux`
/// + `neighbour_lux`.
pub fn check_flux(cap: &HdrCapture, name: &str, tol: f64) -> anyhow::Result<Verdict> {
    let f = flux(cap, name)?;
    let e = cap
        .flux
        .iter()
        .find(|e| e.name == name)
        .with_context(|| format!("no flux entry {name:?} in the capture JSON"))?;
    let expected = e.expected_total_lux();
    let rel = f.lux / expected - 1.0;
    let pass = rel.abs() <= tol;
    let pre = cap.header.pre[1];
    Ok(Verdict {
        lines: vec![
            format!(
                "flux {name}: E {:.5e} lx, expected {:.5e} lx = own {:.5e} + neighbours {:.5e} \
                 ({:+.2} %); {} px inside R = {} px",
                f.lux,
                expected,
                e.expected_lux,
                e.neighbour_lux,
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

/// A limb's integrated world light and the terms behind it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LimbResult {
    /// E = Σ (Y_i − Y_sky,bin·t_i)·Ω_i ÷ pre[0] [lx].
    pub lux: f64,
    /// The twin's Σ L_i·Ω_i [lx].
    pub expected_lux: f64,
    /// The twin's light on the body (pixels with t = 0) [lx].
    pub expected_body_lux: f64,
    /// The sky light subtracted, Σ Y_sky,bin·t_i·Ω_i ÷ pre[0] [lx].
    pub sky_lux: f64,
    /// The smallest and largest bin sky median, pre-exposed.
    pub y_sky_range: [f64; 2],
    pub bins: usize,
    /// Bins without sky pixels of their own (they took the nearest bin's).
    pub borrowed_sky: usize,
    pub pixels: usize,
    pub sky_pixels: usize,
}

/// Solid angle [sr] of the pixel centred at (x, y): Ω_c·cos³θ, as `flux` uses.
fn pixel_omega(h: &HdrHeader, x: f64, y: f64) -> f64 {
    let s = 2.0 * (h.fov_y_rad / 2.0).tan() / f64::from(h.height);
    let d = (x - f64::from(h.width) / 2.0).hypot(y - f64::from(h.height) / 2.0);
    let cos_theta = 1.0 / (1.0 + (s * d).powi(2)).sqrt();
    s * s * cos_theta.powi(3)
}

pub fn limb_flux(cap: &HdrCapture, name: &str) -> anyhow::Result<LimbResult> {
    let e = cap
        .limbs
        .iter()
        .find(|e| e.name == name)
        .with_context(|| format!("no limbs entry {name:?} in the capture JSON"))?;
    if e.bins.iter().all(|b| b.strip.is_empty()) {
        bail!("limbs entry {name:?} has no strip pixels");
    }
    // Each bin's own sky median, pre-exposed (None without sky pixels).
    let own: Vec<Option<f64>> = e
        .bins
        .iter()
        .map(|bin| {
            let mut sky = bin
                .sky
                .iter()
                .map(|&[x, y]| {
                    let (i, j) = cap.pixel_at(x, y).with_context(|| {
                        format!("limbs {name:?} sky pixel ({x}, {y}) is off the image")
                    })?;
                    Ok(cap.luminance(i, j))
                })
                .collect::<anyhow::Result<Vec<f64>>>()?;
            Ok((!sky.is_empty()).then(|| median(&mut sky)))
        })
        .collect::<anyhow::Result<_>>()?;
    let nearest = |k: usize| -> Option<f64> {
        (0..e.bins.len()).find_map(|d| {
            own.get(k + d)
                .copied()
                .flatten()
                .or_else(|| k.checked_sub(d).and_then(|j| own[j]))
        })
    };
    let pre = cap.header.pre[0];
    let mut r = LimbResult {
        lux: 0.0,
        expected_lux: 0.0,
        expected_body_lux: 0.0,
        sky_lux: 0.0,
        y_sky_range: [f64::INFINITY, 0.0],
        bins: e.bins.len(),
        borrowed_sky: own.iter().filter(|o| o.is_none()).count(),
        pixels: 0,
        sky_pixels: e.bins.iter().map(|b| b.sky.len()).sum(),
    };
    for (k, bin) in e.bins.iter().enumerate() {
        let y_sky = nearest(k)
            .with_context(|| format!("limbs entry {name:?} has no sky pixels in any bin"))?;
        r.y_sky_range = [r.y_sky_range[0].min(y_sky), r.y_sky_range[1].max(y_sky)];
        for &[x, y, expected, t] in &bin.strip {
            let (i, j) = cap.pixel_at(x, y).with_context(|| {
                format!("limbs {name:?} strip pixel ({x}, {y}) is off the image")
            })?;
            let omega = pixel_omega(&cap.header, x, y);
            r.lux += (cap.luminance(i, j) - y_sky * t) * omega / pre;
            r.sky_lux += y_sky * t * omega / pre;
            r.expected_lux += expected * omega;
            if t == 0.0 {
                r.expected_body_lux += expected * omega;
            }
            r.pixels += 1;
        }
    }
    if !(r.expected_lux > 0.0) {
        bail!(
            "limbs entry {name:?}: the twin expects no light ({} lx)",
            r.expected_lux
        );
    }
    Ok(r)
}

/// `limb`: the integrated light within the relative tolerance `tol` of the twin's.
pub fn check_limb(cap: &HdrCapture, name: &str, tol: f64) -> anyhow::Result<Verdict> {
    let r = limb_flux(cap, name)?;
    let rel = r.lux / r.expected_lux - 1.0;
    let pass = rel.abs() <= tol;
    Ok(Verdict {
        lines: vec![
            format!(
                "limb {name}: E {:.5e} lx vs the twin's {:.5e} lx ({:+.2} %) over {} px in {} \
                 lit bins; the twin's light on the body {:.1} %",
                r.lux,
                r.expected_lux,
                100.0 * rel,
                r.pixels,
                r.bins,
                100.0 * r.expected_body_lux / r.expected_lux
            ),
            format!(
                "  sky medians {:.5e}–{:.5e} pre-exposed (of {} px; {} bins borrowed a \
                 neighbour's); sky light subtracted {:.5e} lx ({:.1} % of the measured sum)",
                r.y_sky_range[0],
                r.y_sky_range[1],
                r.sky_pixels,
                r.borrowed_sky,
                r.sky_lux,
                100.0 * r.sky_lux / (r.lux + r.sky_lux)
            ),
            format!("limb within ±{} % {}", 100.0 * tol, pass_word(pass)),
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
            limbs: Vec::new(),
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

        // Task C6: a faint 1.5-px blue line in a field of bright blue stars (as at
        // comet-neowise-2020): the ray median finds the line; B8's weighted sum is pulled off.
        let mut field = cap.clone();
        for p in field.pixels.iter_mut() {
            *p = [0.0035, 0.0027, 0.0022, 1.0]; // a reddish band
        }
        let mut seed = 12345_u64;
        let mut rnd = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / f64::from(1u32 << 31)
        };
        for _ in 0..400 {
            let (x, y) = ((rnd() * f64::from(w)) as i64, (rnd() * f64::from(h)) as i64);
            let b = 0.05 + 0.5 * rnd();
            add(&mut field, x, y, [0.2 * b, 0.5 * b, b]);
        }
        streak(37.0, [0.0002, 0.0004, 0.0015], &mut field);
        let a = feature_angle(&field, "ion_tail").expect("angle");
        close(a.angle_deg, 37.0, 0.5, "faint ion tail among blue stars");
        let old = weighted_angle(&field, "ion_tail").expect("weighted");
        assert!((old.angle_deg - 37.0).abs() > 1.0, "{}", old.angle_deg);
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
            neighbour_lux: 0.0,
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
                neighbour_lux: 0.0,
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

    /// `captures/b9-dev/sky-orion-3d.json` (B9's capture at 6cd81e9): Rigel's own illuminance,
    /// the light of the other sources centred inside R, R, and its pixel.
    const B9_RIGEL_LUX: f64 = 2.270_823_967_084_235_5e-6;
    const B9_NEIGHBOUR_LUX: f64 = 4.148_540_357_670_285e-7;
    const B9_RIGEL_R_PX: f64 = 47.702_751_159_667_97;
    const B9_RIGEL_XY: (f64, f64) = (1_044.315_375_018_001_8, 655.415_269_513_219_8);
    /// B9's measured Rigel flux at the gate, against its own illuminance alone (report §6).
    const B9_MEASURED_OVER_OWN: f64 = 1.177;

    /// Lead ruling "row 15 flux": the expectation is the star's own illuminance plus
    /// `neighbour_lux`. A synthetic frame with B9's numbers (Rigel's own light, 18.3 % more from
    /// twelve neighbours centred inside its 47.7-px R, on a uniform sky): the old expectation
    /// reads +18 % and fails ±5 %, the new one passes. B9's measured +17.7 % becomes −0.5 %.
    /// When the b9-dev capture is present (git-ignored), its entry parses with these numbers.
    #[test]
    fn hdr_flux_counts_neighbours_inside_r() {
        let json =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../captures/b9-dev/sky-orion-3d.json");
        match fs::read_to_string(&json) {
            Ok(text) => {
                let c: CaptureJson = serde_json::from_str(&text).expect("b9-dev capture parses");
                let e = c.flux.iter().find(|e| e.name == "Rigel").expect("Rigel");
                assert_eq!(e.expected_lux, B9_RIGEL_LUX);
                assert_eq!(e.neighbour_lux, B9_NEIGHBOUR_LUX);
                assert_eq!(e.radius_px, B9_RIGEL_R_PX);
                assert_eq!((e.x, e.y), B9_RIGEL_XY);
                assert!(!e.mask.is_empty());
            }
            Err(_) => eprintln!("{} absent: synthetic part only", json.display()),
        }
        let rel = B9_MEASURED_OVER_OWN * B9_RIGEL_LUX / (B9_RIGEL_LUX + B9_NEIGHBOUR_LUX) - 1.0;
        close(
            rel,
            -0.0048,
            0.0005,
            "B9's measurement against own + neighbours",
        );

        let mut cap = blank(1920, 1080, 60f64.to_radians(), [1e-4, STAR_PRE]);
        for p in &mut cap.pixels {
            *p = [2.0e-3, 2.0e-3, 2.0e-3, 1.0];
        }
        let (x, y) = B9_RIGEL_XY;
        let r_rigel = f64::from(splat(&mut cap, x, y, B9_RIGEL_LUX, star_colour()));
        let n = 12;
        for k in 0..n {
            let a = std::f64::consts::TAU * f64::from(k) / f64::from(n);
            let d = 20.0 + 1.5 * f64::from(k);
            let (nx, ny) = (x + d * a.cos(), y + d * a.sin());
            let r = f64::from(splat(
                &mut cap,
                nx,
                ny,
                B9_NEIGHBOUR_LUX / f64::from(n),
                star_colour(),
            ));
            assert!(d + r < B9_RIGEL_R_PX, "neighbour {k} reaches outside R");
        }
        assert!(r_rigel < B9_RIGEL_R_PX);
        let mut entry = FluxEntry {
            name: "Rigel".into(),
            x,
            y,
            radius_px: B9_RIGEL_R_PX,
            expected_lux: B9_RIGEL_LUX,
            neighbour_lux: 0.0,
            mask: Vec::new(),
        };
        cap.flux.push(entry.clone());
        let f = flux(&cap, "Rigel").expect("flux");
        close(f.lux / B9_RIGEL_LUX, 1.183, 0.01, "measured over own");
        assert!(
            !check_flux(&cap, "Rigel", 0.05).expect("check").pass,
            "own only"
        );
        entry.neighbour_lux = B9_NEIGHBOUR_LUX;
        cap.flux = vec![entry];
        close(
            f.lux / (B9_RIGEL_LUX + B9_NEIGHBOUR_LUX),
            1.0,
            0.01,
            "own + neighbours",
        );
        let v = check_flux(&cap, "Rigel", 0.05).expect("check");
        assert!(v.pass, "{v:?}");
        assert!(v.lines[0].contains("+ neighbours 4.14854e-7"), "{v:?}");
        // A capture JSON without the key reads 0.
        let old: FluxEntry = serde_json::from_str(
            r#"{"name": "Sun", "x": 1, "y": 2, "radius_px": 3, "expected_lux": 4}"#,
        )
        .expect("old entry");
        assert_eq!(old.neighbour_lux, 0.0);
        assert_eq!(old.expected_total_lux(), 4.0);
    }

    /// Lead ruling "row 5 Mars samples": with a footprint, `samples` reads the GPU's mean over
    /// the pixels that see the 2k twin's 3 × 3 texel patch, not one pixel. The samples are
    /// `captures/b9-dev/mars-reference-angle.json`'s (B9's capture at 6cd81e9, written before
    /// footprints existed; when the git-ignored file is absent, its two disk-centre samples
    /// below), each given the 3 × 3 px footprint that Mars's 2k texels (≈ 1.3 px at 425 px
    /// radius) cover. In the synthetic HDR every footprint averages to the expectation, but the
    /// two disk-centre pixels read +13 %, as B9 measured them: the single-pixel check fails, the
    /// footprint check passes.
    #[test]
    fn hdr_samples_average_the_texel_footprint() {
        let json = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../captures/b9-dev/mars-reference-angle.json");
        let (pre, mut samples) = match fs::read_to_string(&json) {
            Ok(text) => {
                let c: CaptureJson = serde_json::from_str(&text).expect("b9-dev capture parses");
                let pre = c.hdr.expect("hdr header").pre;
                assert_eq!(c.samples.len(), 16);
                (pre, c.samples)
            }
            Err(_) => {
                eprintln!("{} absent: two embedded samples", json.display());
                let sample = |x, y, e| Sample {
                    x,
                    y,
                    expected_cd_m2: e,
                    what: "Mars".into(),
                    footprint: Vec::new(),
                };
                (
                    [5.939_643e-5, 75.424_72],
                    vec![
                        sample(900.5, 588.5, [2945.3857, 915.43335, 506.91324]),
                        sample(876.5, 588.5, [2990.1443, 949.09564, 524.1765]),
                    ],
                )
            }
        };
        let mut cap = blank(1920, 1080, 60f64.to_radians(), pre);
        // The disk-centre samples (lat ≈ 10°, the two B9 saw at +13 %).
        let bright = |s: &Sample| s.y == 588.5 && (s.x == 900.5 || s.x == 876.5);
        for s in &samples {
            let k = if bright(s) { 1.13 } else { 1.0 };
            // The other 8 pixels make the footprint's mean the expectation.
            let rest = (9.0 - k) / 8.0;
            for dy in -1..=1_i64 {
                for dx in -1..=1_i64 {
                    let f = if dx == 0 && dy == 0 { k } else { rest };
                    let rgb = s.expected_cd_m2.map(|c| (c * f * pre[0]) as f32);
                    put(
                        &mut cap,
                        (s.x.floor() as i64 + dx) as u32,
                        (s.y.floor() as i64 + dy) as u32,
                        rgb,
                    );
                }
            }
        }
        cap.samples = samples.clone();
        let single = check_samples(&cap, 0.05).expect("samples");
        assert!(!single.pass, "{single:?}");
        let failed = samples_failing(&single);
        assert_eq!(failed, 2, "{single:?}");
        for s in &mut samples {
            let (x, y) = (s.x, s.y);
            s.footprint = (-1..=1)
                .flat_map(|dy| (-1..=1).map(move |dx| [x + f64::from(dx), y + f64::from(dy)]))
                .collect();
        }
        cap.samples = samples;
        let mean = check_samples(&cap, 0.05).expect("samples");
        assert!(mean.pass, "{mean:?}");
        assert!(mean.lines[0].contains("(mean of 9 px)"), "{mean:?}");
        for r in super::samples(&cap).expect("results") {
            close(r.rel_err, 0.0, 1e-5, &r.what);
        }
    }

    /// Sample lines of a verdict that end in FAIL.
    fn samples_failing(v: &Verdict) -> usize {
        v.lines
            .iter()
            .filter(|l| l.starts_with("  ") && l.ends_with("FAIL"))
            .count()
    }

    /// Lead ruling "row 13 Venus haze", fix round 1 (review I1): `limb` integrates the world
    /// light across the limb in the lit azimuth bins only, each with its own sky. The frame is
    /// `venus-crescent`'s (`captures/b9-dev/venus-crescent.json`, B9's capture at 6cd81e9: fov
    /// 20°, pre = (5.35e-6, 75.4); embedded when the git-ignored file is absent): the deck's
    /// limb at 125.1 px, the shell's top 1.2 px beyond, and the twin's haze on the night half:
    /// a band 0.25 px deep, lit on the two cusp arcs (|ψ| 90°–105°) with the twin's total,
    /// 1.27e-2 lx (uni-app's `capture_measurements_cover_the_plan_checks`). The sky behind it
    /// is B9's level (its old `night_limb_haze` region read 1.28e-3 pre-exposed) with a
    /// gradient across the frame, ±30 % patches 50–70 px across and stray stars in the sky
    /// rings; the GPU's sky share is the twin's smeared over 2 px. With the haze right the
    /// check stays within ±10 %; 20 % too bright, 20 % too dark or missing, it fails.
    #[test]
    fn hdr_limb_integrates_the_haze_across_the_limb() {
        let json =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../captures/b9-dev/venus-crescent.json");
        let (fov, pre) = match fs::read_to_string(&json) {
            Ok(text) => {
                let c: CaptureJson = serde_json::from_str(&text).expect("b9-dev capture parses");
                let h = c.hdr.expect("hdr header");
                (h.fov_y_rad, h.pre)
            }
            Err(_) => {
                eprintln!("{} absent: embedded header", json.display());
                (20f64.to_radians(), [5.354_771e-6, 75.424_72])
            }
        };
        const SKY_PRE: f64 = 1.28e-3;
        const BIN_DEG: f64 = 2.0;
        let (cx, cy, r_deck, r_top) = (960.0, 540.0, 125.1, 126.3);
        // The Sun toward +y (image down): ψ is the angle from it, the night half |ψ| ≥ 90°.
        let psi_of = |x: f64, y: f64| (-(x - cx)).atan2(y - cy).to_degrees();
        // The twin's haze [cd/m²]: lit from each cusp (|ψ| = 90°) fading to nothing 15° past it.
        let haze = |rho: f64, psi: f64| -> f64 {
            let h = rho - r_deck;
            let past = psi.abs() - 90.0;
            if !(0.0..=r_top - r_deck).contains(&h) || !(0.0..15.0).contains(&past) {
                return 0.0;
            }
            1.5e4 * (1.0 - past / 15.0) * (-h / 0.25).exp()
        };
        // The twin's sky share: opaque at the deck, clear 0.3 px above it; the GPU's, the same
        // smeared over 2 px (a box in ρ, which keeps its integral across the edge).
        let t_twin = |rho: f64| ((rho - r_deck) / 0.3).clamp(0.0, 1.0);
        let t_gpu = |rho: f64| {
            (0..20)
                .map(|k| t_twin(rho - 1.0 + 0.1 * (f64::from(k) + 0.5)))
                .sum::<f64>()
                / 20.0
        };
        // A patchy band: a gradient across the frame and ±30 % patches 50–70 px across.
        let sky = |x: f64, y: f64| {
            SKY_PRE
                * (1.0 + 0.5 * (x - cx) / 400.0)
                * (1.0 + 0.3 * (x / 17.0).sin() * (y / 21.0).cos())
        };
        let build = |gpu_haze: f64| {
            let mut cap = blank(1920, 1080, fov, pre);
            let mut bins: BTreeMap<i64, LimbBin> = BTreeMap::new();
            for j in 380..700 {
                for i in 800..1120 {
                    let (x, y) = (f64::from(i) + 0.5, f64::from(j) + 0.5);
                    let rho = (x - cx).hypot(y - cy);
                    let psi = psi_of(x, y);
                    let t = if rho < r_deck { 0.0 } else { t_gpu(rho) };
                    // Stray stars in the sky rings, every 37th pixel.
                    let star = if (i * 31 + j * 17) % 37 == 0 && rho > r_top + 3.0 {
                        0.05
                    } else {
                        0.0
                    };
                    let y_pre = gpu_haze * haze(rho, psi) * pre[0] + sky(x, y) * t + star;
                    put(&mut cap, i, j, [y_pre as f32; 3]);
                    // The entry: lit bins and one margin bin each side (|ψ| 88°–107°).
                    if !(88.0..107.0).contains(&psi.abs()) {
                        continue;
                    }
                    let k = (psi / BIN_DEG).floor() as i64;
                    let bin = bins.entry(k).or_insert_with(|| LimbBin {
                        psi_deg: Some([k as f64 * BIN_DEG, (k + 1) as f64 * BIN_DEG]),
                        strip: Vec::new(),
                        sky: Vec::new(),
                    });
                    if rho >= r_deck - 2.0 && rho <= r_top + 2.0 {
                        let tw = if rho < r_deck { 0.0 } else { t_twin(rho) };
                        bin.strip.push([x, y, haze(rho, psi), tw]);
                    } else if rho >= r_top + 4.0 && rho <= r_top + 10.0 {
                        bin.sky.push([x, y]);
                    }
                }
            }
            cap.limbs.push(LimbEntry {
                name: "venus_night_limb".into(),
                bins: bins.into_values().collect(),
            });
            cap
        };
        let right = build(1.0);
        let r = limb_flux(&right, "venus_night_limb").expect("limb");
        eprintln!("limb: {r:?}, E/twin {:.4}", r.lux / r.expected_lux);
        close(r.expected_lux, 1.27e-2, 0.1, "the twin's total");
        assert!(r.sky_lux < r.lux, "the sky is the minor part: {r:?}");
        assert_eq!(r.expected_body_lux, 0.0);
        let v = check_limb(&right, "venus_night_limb", 0.1).expect("check");
        assert!(v.pass, "{v:?}");
        for (k, what) in [
            (1.2, "20 % too bright"),
            (0.8, "20 % too dark"),
            (0.0, "missing"),
        ] {
            let v = check_limb(&build(k), "venus_night_limb", 0.1).expect("check");
            assert!(!v.pass, "{what}: {v:?}");
        }
        // One sky median for all bins (the review's leverage): with the patches it misses.
        let mut one = build(1.0);
        let all_sky: Vec<[f64; 2]> = one.limbs[0]
            .bins
            .iter()
            .flat_map(|b| b.sky.clone())
            .collect();
        let mut merged = one.limbs[0].bins.clone();
        for b in &mut merged {
            b.sky = all_sky.clone();
        }
        one.limbs[0].bins = merged;
        let single = limb_flux(&one, "venus_night_limb").expect("limb");
        let err = |x: &LimbResult| (x.lux / x.expected_lux - 1.0).abs();
        eprintln!(
            "one sky median: E/twin {:.4}",
            single.lux / single.expected_lux
        );
        assert!(err(&single) > 0.1 && err(&r) < 0.1, "{single:?}");
    }
}
