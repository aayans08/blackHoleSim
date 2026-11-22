//! `uni-bake rings` (M1 Task A8): Saturn's ring alpha PNG → the radial profile `data/baked/saturn_rings.ron`.
//!
//! `uni-bake rings --input data/cache/sss-8k/8k_saturn_ring_alpha.png [--inner-km <r> --outer-km <r>]
//! [--row <y>] --out data/baked/saturn_rings.ron`
//!
//! One row of the PNG (default: the middle one) is the profile from the inner edge (left) to the
//! outer edge (right): colour = reflectance (sRGB → linear), alpha = opacity. It is box-filtered
//! to [`SAMPLES`] samples uniform in radius. Solar System Scope does not document the PNG's radii
//! (its XMP metadata is Photoshop history only), so without `--inner-km`/`--outer-km` they are
//! fitted: the steepest opacity drop is the B ring's outer edge at the Cassini division (assigned
//! the Huygens gap's radius, [`HUYGENS_GAP_KM`]) and the last texel at ≥ 25 % opacity is the A
//! ring's outer edge ([`A_RING_OUTER_KM`]); the texel scale and the end radii follow linearly.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, bail, ensure};

/// Samples in the baked profile.
const SAMPLES: usize = 1024;
/// Huygens gap at the inner edge of the Cassini division [km] (the plan's value; the B ring's
/// outer edge is 117 507 km in the NSSDCA Saturnian Rings Fact Sheet).
const HUYGENS_GAP_KM: f64 = 117_680.0;
/// Outer edge of the A ring [km] (the plan's value; 136 780 km in the NSSDCA Saturnian Rings Fact
/// Sheet, https://nssdc.gsfc.nasa.gov/planetary/factsheet/satringfact.html).
const A_RING_OUTER_KM: f64 = 136_775.0;

/// The parsed command line.
#[derive(Debug, PartialEq)]
struct RingSpec {
    input: PathBuf,
    radii_km: Option<(f64, f64)>,
    row: Option<u32>,
    out: PathBuf,
}

/// Runs `uni-bake rings` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let spec = parse(options)?;
    let png = image::open(&spec.input)
        .with_context(|| format!("decoding {}", spec.input.display()))?
        .into_rgba8();
    let (width, height) = png.dimensions();
    let row = spec.row.unwrap_or(height / 2);
    ensure!(
        row < height,
        "--row {row} is outside the {height}-row image"
    );
    let texels: Vec<[f32; 4]> = (0..width)
        .map(|x| {
            let [r, g, b, a] = png.get_pixel(x, row).0;
            [
                srgb_to_linear(r),
                srgb_to_linear(g),
                srgb_to_linear(b),
                f32::from(a) / 255.0,
            ]
        })
        .collect();
    let identical_rows =
        (0..height).all(|y| (0..width).all(|x| png.get_pixel(x, y) == png.get_pixel(x, row)));
    let (inner_km, outer_km) = match spec.radii_km {
        Some(radii) => radii,
        None => {
            let opacity: Vec<f32> = texels.iter().map(|t| t[3]).collect();
            let (b, a) = find_edges(&opacity)?;
            let (inner, outer) = fit_radii((b, a), texels.len());
            println!(
                "rings: fitted B-ring/Cassini edge at texel {b:.0} = {HUYGENS_GAP_KM} km, A-ring \
                 outer edge at texel {a:.0} = {A_RING_OUTER_KM} km: {:.3} km/texel, row spans \
                 {inner:.0}–{outer:.0} km",
                (A_RING_OUTER_KM - HUYGENS_GAP_KM) / (a - b)
            );
            (inner, outer)
        }
    };
    let samples = resample(&texels, SAMPLES)
        .into_iter()
        .map(|s| s.map(|v| (v * 1e4).round() / 1e4))
        .collect();
    let profile = uni_data::RingProfile {
        inner_m: inner_km * 1e3,
        outer_m: outer_km * 1e3,
        samples,
    };
    uni_data::rings::write_ring_profile(&spec.out, &profile)?;
    println!(
        "rings: {width}×{height} PNG, row {row} (all rows identical: {identical_rows}) -> {SAMPLES} \
         samples, {inner_km:.0}–{outer_km:.0} km -> {}",
        spec.out.display()
    );
    Ok(())
}

/// IEC 61966-2-1:1999 (sRGB) decoding of an 8-bit code.
fn srgb_to_linear(code: u8) -> f32 {
    let c = f64::from(code) / 255.0;
    let l = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    l as f32
}

fn parse(options: &[OsString]) -> anyhow::Result<RingSpec> {
    let (mut input, mut out, mut inner, mut outer, mut row) = (None, None, None, None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value"))?;
        let text = || {
            value
                .to_str()
                .with_context(|| format!("{flag:?} {value:?} is not UTF-8"))
        };
        let km = || -> anyhow::Result<f64> {
            let v: f64 = text()?
                .parse()
                .with_context(|| format!("{flag:?} is not a number"))?;
            ensure!(
                v.is_finite() && v > 0.0,
                "{flag:?} must be a positive radius"
            );
            Ok(v)
        };
        match flag.to_str() {
            Some("--input") => input = Some(PathBuf::from(value)),
            Some("--out") => out = Some(PathBuf::from(value)),
            Some("--inner-km") => inner = Some(km()?),
            Some("--outer-km") => outer = Some(km()?),
            Some("--row") => row = Some(text()?.parse().context("--row is not a row number")?),
            _ => bail!("unknown option {flag:?}"),
        }
    }
    let radii_km = match (inner, outer) {
        (Some(inner), Some(outer)) => {
            ensure!(inner < outer, "--inner-km must be below --outer-km");
            Some((inner, outer))
        }
        (None, None) => None,
        _ => bail!("give both --inner-km and --outer-km, or neither (fit)"),
    };
    Ok(RingSpec {
        input: input.context("missing --input")?,
        radii_km,
        row,
        out: out.context("missing --out")?,
    })
}

/// The fitted pixel positions (as texel-edge coordinates, 0 = the row's left edge) of the B
/// ring's outer edge and the A ring's outer edge in an opacity row.
///
/// The A ring's outer edge is the right end of the last run of ≥ 64 texels at ≥ 25 % opacity
/// (beyond it are only a faint haze and the narrow F ring, a few texels wide). The B ring's outer
/// edge is the steepest drop of the mean opacity across a texel edge (8 texels each side) in the
/// inner 85 % of the rings, which leaves out the Encke and Keeler gaps near the A ring's outer
/// edge and that edge itself.
fn find_edges(opacity: &[f32]) -> anyhow::Result<(f64, f64)> {
    const HALF: usize = 8;
    const MIN_RUN: usize = 64;
    let mut a_edge = None;
    let mut run = 0;
    for (x, &o) in opacity.iter().enumerate() {
        run = if o >= 0.25 { run + 1 } else { 0 };
        if run >= MIN_RUN && opacity.get(x + 1).is_none_or(|&next| next < 0.25) {
            a_edge = Some(x + 1);
        }
    }
    let a_edge = a_edge.context("no run of 64 texels at ≥ 25 % opacity: not a ring profile")?;
    let limit = (a_edge as f64 * 0.85) as usize;
    ensure!(limit > 2 * HALF, "ring profile too short to fit");
    let mean = |range: std::ops::Range<usize>| {
        opacity[range.clone()]
            .iter()
            .map(|&o| f64::from(o))
            .sum::<f64>()
            / range.len() as f64
    };
    let b_edge = (HALF..limit)
        .map(|x| (x, mean(x - HALF..x) - mean(x..x + HALF)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(x, _)| x)
        .context("no opacity drop found")?;
    Ok((b_edge as f64, a_edge as f64))
}

/// Radii [km] of the row's left and right edges, given the two edges' pixel positions in a row
/// `width` texels wide.
fn fit_radii(edges_px: (f64, f64), width: usize) -> (f64, f64) {
    let (b, a) = edges_px;
    let km_per_px = (A_RING_OUTER_KM - HUYGENS_GAP_KM) / (a - b);
    (
        HUYGENS_GAP_KM - b * km_per_px,
        A_RING_OUTER_KM + (width as f64 - a) * km_per_px,
    )
}

/// Box-filters per-texel `[r, g, b, opacity]` (linear colour) down to `n` samples: opacity is
/// averaged, colour is the opacity-weighted mean (the plain mean where nothing is opaque).
/// Sample `i` covers texels [i·len/n, (i + 1)·len/n), with fractional texels weighted by overlap.
fn resample(texels: &[[f32; 4]], n: usize) -> Vec<[f32; 4]> {
    let step = texels.len() as f64 / n as f64;
    (0..n)
        .map(|i| {
            let (start, end) = (i as f64 * step, (i + 1) as f64 * step);
            let (mut colour, mut plain, mut opacity) = ([0.0_f64; 3], [0.0_f64; 3], 0.0_f64);
            for j in start.floor() as usize..(end.ceil() as usize).min(texels.len()) {
                let w = (end.min(j as f64 + 1.0) - start.max(j as f64)) / step;
                let [r, g, b, o] = texels[j].map(f64::from);
                for (k, c) in [r, g, b].into_iter().enumerate() {
                    colour[k] += w * o * c;
                    plain[k] += w * c;
                }
                opacity += w * o;
            }
            let rgb = if opacity > 0.0 {
                colour.map(|c| c / opacity)
            } else {
                plain
            };
            [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, opacity as f32]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_rings_command() {
        let spec = parse(&args(&["--input", "r.png", "--out", "r.ron"])).expect("valid");
        assert_eq!(
            spec,
            RingSpec {
                input: PathBuf::from("r.png"),
                radii_km: None,
                row: None,
                out: PathBuf::from("r.ron"),
            }
        );
        let spec = parse(&args(&[
            "--input",
            "r.png",
            "--inner-km",
            "74500",
            "--outer-km",
            "140220",
            "--row",
            "3",
            "--out",
            "r.ron",
        ]))
        .expect("valid");
        assert_eq!(spec.radii_km, Some((74_500.0, 140_220.0)));
        assert_eq!(spec.row, Some(3));
        assert!(parse(&args(&["--input", "r.png"])).is_err());
        assert!(
            parse(&args(&[
                "--input",
                "r.png",
                "--out",
                "r.ron",
                "--inner-km",
                "1"
            ]))
            .is_err()
        );
        assert!(
            parse(&args(&[
                "--input",
                "r.png",
                "--out",
                "r.ron",
                "--inner-km",
                "9",
                "--outer-km",
                "1"
            ]))
            .is_err()
        );
        assert!(parse(&args(&["--input", "r.png", "--out", "r.ron", "--x", "1"])).is_err());
    }

    /// A synthetic row like the real one: a C ring, a B ring ending at texel edge 5400, a
    /// division, an A ring with an Encke-like gap ending at 7700, a faint haze and a narrow
    /// F-ring-like spike: the fit finds both edges within a texel and maps them to the reference
    /// radii.
    #[test]
    fn fit_recovers_synthetic_edges() {
        let opacity: Vec<f32> = (0..8192)
            .map(|x| match x {
                0..400 => 0.0,
                400..2600 => 0.3,
                2600..5400 => 0.9 + 0.05 * ((x as f32) * 0.37).sin(),
                5400..5800 => 0.25,
                7300..7330 => 0.05,
                5800..7700 => 0.75,
                8100..8106 => 0.33,
                _ => 0.03,
            })
            .collect();
        let (b, a) = find_edges(&opacity).expect("edges");
        assert!((b - 5400.0).abs() <= 1.0, "B edge {b}");
        assert!((a - 7700.0).abs() <= 1.0, "A edge {a}");
        let (inner, outer) = fit_radii((5400.0, 7700.0), 8192);
        let km_per_px = (A_RING_OUTER_KM - HUYGENS_GAP_KM) / 2300.0;
        assert!((inner - (HUYGENS_GAP_KM - 5400.0 * km_per_px)).abs() < 1e-6);
        assert!((outer - (A_RING_OUTER_KM + 492.0 * km_per_px)).abs() < 1e-6);
    }

    #[test]
    fn resample_weights_colour_by_opacity() {
        let texels = [
            [1.0, 0.0, 0.0, 0.75],
            [0.0, 1.0, 0.0, 0.25],
            [0.2, 0.2, 0.2, 0.0],
            [0.4, 0.4, 0.4, 0.0],
        ];
        let out = resample(&texels, 2);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], [0.75, 0.25, 0.0, 0.5]);
        assert_eq!(out[1], [0.3, 0.3, 0.3, 0.0]);
    }
}
