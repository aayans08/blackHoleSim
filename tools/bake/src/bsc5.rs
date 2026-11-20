//! Parses the raw Yale Bright Star Catalogue, 5th revised ed. (Hoffleit & Warren 1991, CDS V/50,
//! file `catalog`) and writes data/baked/bsc5.bin (M0 Task 7).
//!
//! Field positions are the byte-by-byte description in the CDS V/50 ReadMe (1-based, inclusive).
//! The file's lines have trailing blanks trimmed, so a field past the end of a line is blank.

use anyhow::{Context, bail, ensure};
use flate2::read::GzDecoder;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::str::FromStr;
use uni_core::{bv_to_teff, radec_to_icrs};
use uni_data::CatalogStar;

/// Teff [K] given to stars with no B−V in the catalogue: a placeholder close to the Sun's
/// nominal 5772 K (IAU 2015 Resolution B3), so these stars render near white.
const MISSING_BV_TEFF_K: f64 = 5800.0;

/// Catalogue B−V is clamped to [BV_MIN, BV_MAX] before the Ballesteros (2012, EPL 97 34008)
/// formula in `uni_core::bv_to_teff`, which diverges at B−V = −0.674 (lead's ruling, M0 Task 7).
/// BSC5 colours run from −0.28 to +5.74, so in practice only very red stars (carbon stars,
/// Miras, a few supergiants) are clamped, to about 3170 K.
const BV_MIN: f64 = -0.4;
const BV_MAX: f64 = 2.0;

/// What the bake did, for the console.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// Lines read from the catalogue.
    pub records: usize,
    /// Stars written.
    pub stars: usize,
    /// Records skipped because their J2000 position is blank.
    pub no_position: usize,
    /// Stars with no B−V, given `MISSING_BV_TEFF_K`.
    pub missing_bv: usize,
    /// Stars whose B−V was clamped to `BV_MIN..=BV_MAX`.
    pub clamped_bv: usize,
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} stars from {} records; {} records without a J2000 position skipped; \
             {} stars without B−V set to {MISSING_BV_TEFF_K} K; \
             {} stars with B−V clamped to [{BV_MIN:.1}, {BV_MAX:.1}]",
            self.stars, self.records, self.no_position, self.missing_bv, self.clamped_bv
        )
    }
}

/// How a star's Teff was obtained from its catalogue B−V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Colour {
    Measured,
    Clamped,
    Missing,
}

/// One parsed line of the catalogue.
#[derive(Debug, PartialEq)]
enum Record {
    Star { star: CatalogStar, colour: Colour },
    NoPosition { hr: u32 },
}

/// Reads the gzipped catalogue at `input`, writes the baked file to `out` and checks that it
/// reads back unchanged.
pub fn bake(input: &Path, out: &Path) -> anyhow::Result<Summary> {
    let file = File::open(input).with_context(|| format!("opening {}", input.display()))?;
    let (stars, summary) = parse_catalog(BufReader::new(GzDecoder::new(file)))
        .with_context(|| format!("parsing {}", input.display()))?;
    uni_data::write_bsc5(out, &stars).with_context(|| format!("writing {}", out.display()))?;
    let written =
        uni_data::load_bsc5(out).with_context(|| format!("reading back {}", out.display()))?;
    ensure!(
        written == stars,
        "{} does not read back as written",
        out.display()
    );
    Ok(summary)
}

fn parse_catalog(reader: impl BufRead) -> anyhow::Result<(Vec<CatalogStar>, Summary)> {
    let mut stars = Vec::new();
    let mut summary = Summary::default();
    for (index, line) in reader.lines().enumerate() {
        let number = index + 1;
        let line = line.with_context(|| format!("reading line {number}"))?;
        summary.records += 1;
        match parse_record(&line).with_context(|| format!("line {number}"))? {
            Record::NoPosition { hr } => {
                log::info!("HR {hr}: no J2000 position (nova or non-stellar entry), skipped");
                summary.no_position += 1;
            }
            Record::Star { star, colour } => {
                match colour {
                    Colour::Measured => {}
                    Colour::Clamped => summary.clamped_bv += 1,
                    Colour::Missing => summary.missing_bv += 1,
                }
                stars.push(star);
            }
        }
    }
    summary.stars = stars.len();
    if summary.missing_bv > 0 {
        log::warn!(
            "{} stars have no B−V; their Teff is set to {MISSING_BV_TEFF_K} K \
             (RUST_LOG=debug lists them)",
            summary.missing_bv
        );
    }
    if summary.clamped_bv > 0 {
        log::warn!(
            "{} stars have B−V outside [{BV_MIN:.1}, {BV_MAX:.1}] and were clamped \
             (RUST_LOG=debug lists them)",
            summary.clamped_bv
        );
    }
    Ok((stars, summary))
}

fn parse_record(line: &str) -> anyhow::Result<Record> {
    ensure!(line.is_ascii(), "line is not ASCII");
    let hr: u32 = required(line, 1, 4, "HR")?;
    // ReadMe note (1): the J2000 position (bytes 76-90) is all blank for the 14 entries kept only
    // to preserve the numbering (novae, clusters, galaxies).
    if field(line, 76, 90).is_empty() {
        return Ok(Record::NoPosition { hr });
    }
    parse_star(line, hr).with_context(|| format!("HR {hr}"))
}

fn parse_star(line: &str, hr: u32) -> anyhow::Result<Record> {
    let ra_h: u32 = required(line, 76, 77, "RAh")?;
    let ra_m: u32 = required(line, 78, 79, "RAm")?;
    let ra_s: f64 = required(line, 80, 83, "RAs")?;
    let dec_sign = match field(line, 84, 84) {
        "+" => 1.0,
        "-" => -1.0,
        other => bail!("DE- (byte 84) is {other:?}, not + or -"),
    };
    let dec_d: u32 = required(line, 85, 86, "DEd")?;
    let dec_m: u32 = required(line, 87, 88, "DEm")?;
    let dec_s: u32 = required(line, 89, 90, "DEs")?;
    ensure!(
        ra_h < 24 && ra_m < 60 && (0.0..60.0).contains(&ra_s),
        "RA {ra_h}h {ra_m}m {ra_s}s is out of range"
    );
    ensure!(
        dec_d <= 90 && dec_m < 60 && dec_s < 60,
        "Dec {dec_d}° {dec_m}′ {dec_s}″ is out of range"
    );
    let ra_deg = 15.0 * (f64::from(ra_h) + f64::from(ra_m) / 60.0 + ra_s / 3600.0);
    let dec_deg =
        dec_sign * (f64::from(dec_d) + f64::from(dec_m) / 60.0 + f64::from(dec_s) / 3600.0);
    ensure!(dec_deg.abs() <= 90.0, "Dec {dec_deg}° is beyond a pole");

    let v_mag: f64 = required(line, 103, 107, "Vmag")?;
    let bv: Option<f64> = optional(line, 110, 114, "B-V")?;
    ensure!(
        v_mag.is_finite() && bv.is_none_or(f64::is_finite),
        "Vmag {v_mag} or B-V {bv:?} is not finite"
    );
    let (teff_k, colour) = teff_from_bv(hr, bv);

    // The positions are FK5, equinox J2000, epoch 2000.0. The FK5 axes match ICRS to better than
    // 0.1″ (Feissel & Mignard 1998, A&A 331, L33), well under the catalogue's 0.1 s / 1″
    // rounding, so they are used as ICRS directly. Proper motion is not applied.
    let dir_icrs = radec_to_icrs(ra_deg.to_radians(), dec_deg.to_radians()).as_vec3();
    let star = CatalogStar {
        dir_icrs,
        v_mag: v_mag as f32,
        teff_k: teff_k as f32,
        hr,
    };
    Ok(Record::Star { star, colour })
}

/// Teff [K] from catalogue B−V via Ballesteros (2012), with the fallback and clamp above.
fn teff_from_bv(hr: u32, bv: Option<f64>) -> (f64, Colour) {
    match bv {
        None => {
            log::debug!("HR {hr}: no B−V; Teff = {MISSING_BV_TEFF_K} K");
            (MISSING_BV_TEFF_K, Colour::Missing)
        }
        Some(bv) if (BV_MIN..=BV_MAX).contains(&bv) => (bv_to_teff(bv), Colour::Measured),
        Some(bv) => {
            let clamped = bv.clamp(BV_MIN, BV_MAX);
            log::debug!("HR {hr}: B−V {bv:+.2} clamped to {clamped:+.2}");
            (bv_to_teff(clamped), Colour::Clamped)
        }
    }
}

/// Bytes `first..=last` of `line` (1-based, as in the ReadMe), trimmed; blank past the line's end.
fn field(line: &str, first: usize, last: usize) -> &str {
    line.get(first - 1..last.min(line.len()))
        .unwrap_or("")
        .trim()
}

/// The number in bytes `first..=last` of `line`, or `None` when the field is blank.
fn optional<T>(line: &str, first: usize, last: usize, label: &str) -> anyhow::Result<Option<T>>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    let text = field(line, first, last);
    if text.is_empty() {
        return Ok(None);
    }
    let value = text
        .parse()
        .with_context(|| format!("{label} (bytes {first}-{last}) is {text:?}"))?;
    Ok(Some(value))
}

/// The number in bytes `first..=last` of `line`; an error when the field is blank.
fn required<T>(line: &str, first: usize, last: usize, label: &str) -> anyhow::Result<T>
where
    T: FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    optional(line, first, last, label)?
        .with_context(|| format!("{label} (bytes {first}-{last}) is blank"))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real lines of CDS V/50 `catalog`, copied byte for byte (the file trims trailing blanks).
    /// HR 2491, Sirius.
    const SIRIUS: &str = "2491  9Alp CMaBD-16 1591  48915151881 257I   5423           064044.6-163444064508.9-164258227.22-08.88-1.46   0.00 -0.05 -0.03   A1Vm               -0.553-1.205 +.375-008SBO    13 10.3  11.2AB   4*";
    /// HR 92, Nova 1572 (Tycho's supernova): kept for numbering, no position.
    const TYCHO_NOVA: &str = "  92 NOVA 1572                                     B Cas                                                                                                                                            *";
    /// HR 52: no B−V.
    const NO_BV: &str = "  52          BD+30   26   1075 53755    I                  000954.9+305848001507.0+313209113.93-30.70 6.45R                     K5                 +0.035-0.005      +002";
    /// HR 977, TW Hor, a carbon star with B−V = +2.28.
    const TW_HOR: &str = " 977          CP-57  513  20234233037 118          TW Hor   031001.1-574145031233.2-571918273.30-50.90 5.74  +2.28 +2.83 +1.38   C5II               +0.019+0.009      +014                          *";

    fn star(record: Record) -> (CatalogStar, Colour) {
        match record {
            Record::Star { star, colour } => (star, colour),
            other => panic!("expected a star, got {other:?}"),
        }
    }

    /// `line` with 1-based bytes `first..first + text.len()` replaced by `text`.
    fn patched(line: &str, first: usize, text: &str) -> String {
        let mut bytes = line.as_bytes().to_vec();
        bytes[first - 1..first - 1 + text.len()].copy_from_slice(text.as_bytes());
        String::from_utf8(bytes).expect("ascii")
    }

    #[test]
    fn parse_sirius_line() {
        let (sirius, colour) = star(parse_record(SIRIUS).expect("parse"));
        let expected = radec_to_icrs(101.2872_f64.to_radians(), (-16.7161_f64).to_radians());
        let dir = sirius.dir_icrs.as_dvec3();
        let angle_deg = dir
            .cross(expected)
            .length()
            .atan2(dir.dot(expected))
            .to_degrees();
        assert!(angle_deg < 0.01, "{angle_deg}° from the expected direction");
        assert!((sirius.dir_icrs.length() - 1.0).abs() < 1e-6);
        assert_eq!(sirius.v_mag, -1.46);
        assert_eq!(sirius.teff_k, bv_to_teff(0.0) as f32);
        assert_eq!(sirius.hr, 2491);
        assert_eq!(colour, Colour::Measured);
    }

    #[test]
    fn blank_position_is_no_position() {
        assert_eq!(
            parse_record(TYCHO_NOVA).expect("parse"),
            Record::NoPosition { hr: 92 }
        );
    }

    #[test]
    fn missing_bv_gets_5800_k() {
        let (s, colour) = star(parse_record(NO_BV).expect("parse"));
        assert_eq!((s.hr, s.teff_k, colour), (52, 5800.0, Colour::Missing));
    }

    #[test]
    fn red_bv_is_clamped_to_2() {
        let (s, colour) = star(parse_record(TW_HOR).expect("parse"));
        assert_eq!(s.teff_k, bv_to_teff(2.0) as f32);
        assert_eq!(colour, Colour::Clamped);
    }

    #[test]
    fn malformed_position_is_an_error() {
        let bad_minutes = patched(SIRIUS, 78, "75");
        assert!(parse_record(&bad_minutes).is_err());
        let blank_degrees = patched(SIRIUS, 85, "  ");
        assert!(parse_record(&blank_degrees).is_err());
        let bad_sign = patched(SIRIUS, 84, "x");
        assert!(parse_record(&bad_sign).is_err());
    }

    #[test]
    fn catalog_counts_skips_and_fallbacks() {
        let text = [NO_BV, TYCHO_NOVA, TW_HOR, SIRIUS].join("\n");
        let (stars, summary) = parse_catalog(text.as_bytes()).expect("parse");
        let hrs: Vec<u32> = stars.iter().map(|s| s.hr).collect();
        assert_eq!(hrs, [52, 977, 2491]);
        assert_eq!(
            summary,
            Summary {
                records: 4,
                stars: 3,
                no_position: 1,
                missing_bv: 1,
                clamped_bv: 1,
            }
        );
    }

    #[test]
    fn bad_line_reports_its_line_number() {
        let text = [SIRIUS, "garbage"].join("\n");
        let err = parse_catalog(text.as_bytes()).expect_err("garbage line");
        assert!(format!("{err:#}").contains("line 2"), "{err:#}");
    }
}
