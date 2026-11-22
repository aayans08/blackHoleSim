//! `uni-bake athyg` (M1 Task A7): AT-HYG v4.0 subset (V ≤ 10) → `data/baked/athyg_m10.bin` + `athyg_names.ron`.
//!
//! Command line:
//! `uni-bake athyg --input data/cache/athyg-m10/athyg_40_reduced_m10.csv.gz --out data/baked/athyg_m10.bin
//!  --names data/baked/athyg_names.ron --bsc5 data/cache/bsc5/catalog.gz`.
//! `--bsc5` (the raw CDS V/50 catalogue, see `data/manifest.toml` entry `bsc5`) is an addition to
//! the plan's command line: it supplies Johnson V and B−V for the brightest stars, whose AT-HYG
//! magnitude is a Hipparcos Hp with no colour (see "Photometry" below). The star file is written
//! without `FLAG_EXO_HOST`; run `uni-bake exo` afterwards, which sets it in place.
//!
//! Column names and meanings are AT-HYG's own (`README.md`, `version-info.md` "Version 2.x
//! Data Format", `data/subsets/README.md`, all v4.0, read 2026-11-22):
//! - Position: `ra` [h] and `dec` [deg] ("epoch + equinox 2000.0") with `dist` [pc] give the
//!   stored f64 → f32 position; `x0, y0, z0` [pc] (x toward RA 0 Dec 0, y toward RA 6 h, z toward
//!   Dec +90°) are printed to only 1e-4 pc, so they serve as a cross-check, which the bake
//!   enforces to [`POS_CHECK_TOL_PC`].
//! - Velocity: `vx, vy, vz` [km/s], same axes. "Cartesian velocities were not calculated if
//!   either the distance or the proper motions were missing" → stored as 0 with
//!   `FLAG_NO_VELOCITY`. "When the radial velocity was missing or equal to zero, the Cartesian
//!   velocities were still calculated, but assuming no radial component" → `FLAG_NO_RV`.
//! - Rows with an empty `dist` (Gaia parallax errors larger than the parallax, set to NULL in
//!   v2.4) have no position and are skipped and counted. So are rows at `dist` ≥ 100 000 pc:
//!   HYG's placeholder, "A value >= 100000 indicates missing or dubious (e.g., negative)
//!   parallax data in Hipparcos" (HYG `data/hyg/README.md`, field 9), which AT-HYG keeps for
//!   some Hipparcos distances (their velocities reach 65 000 km/s). The Sun's row (`Sol`, id 1
//!   at 4.85e-6 pc in v4.0) is skipped: uni-planet and uni-sky draw the Sun.
//!
//! Photometry (`mag`, `absmag`, `ci`, `mag_src`):
//! - `mag_src = T`, `ci` present: `mag` is Tycho VT and `ci` is BT−VT. Converted to Johnson
//!   with ESA (1997), *The Hipparcos and Tycho Catalogues*, ESA SP-1200 vol. 1, §1.3 Appendix 4,
//!   Table 1.3.4 ([`tycho_to_johnson`]): the "more accurate transformation" that ESA used for
//!   the Tycho Catalogue's own B−V (field T37), linear between the table's nodes at BT−VT −0.2,
//!   0.1, 0.5, 1.4 and 1.8, and beyond the end nodes "the values tabulated at the ends of the
//!   given interval are used for extrapolation". (The simple form V = VT − 0.090 (BT−VT),
//!   B−V = 0.850 (BT−VT) of §1.3 eq. 1.3.20 and §2.2 eqs. 2.2.1–2.2.2 is the table without its
//!   corrections of up to 0.046 mag.) M_V = absmag + (V − VT). Measured on v4.0 against BSC5
//!   Johnson B−V (the bake enforces |median| ≤ [`BSC5_BV_CHECK_MAX`]): inside the table's
//!   interval the median of BSC5 − converted is +0.001 (7,740 HR stars); redder than 1.8 it is
//!   +0.005 (724 stars to BT−VT 2.6) and −0.18 for the 21 beyond 2.6 (17 of them carbon stars).
//!   Clamping BT−VT to 1.8, as this bake first did, made those stars 0.07 and 1.07 too blue.
//!   Exceptions:
//!   - Red dwarfs. ESA: the transformations "ignore variations due to luminosity class. They
//!     should not be applied to M-type stars". A row redder than the last node (BT−VT > 1.8)
//!     whose `absmag` is [`DWARF_MIN_ABS_MAG`] or fainter is a K/M dwarf. These stars are faint
//!     in BT, so their Tycho colour is noisy (the Tycho Catalogue's median BT−VT error is 0.12
//!     mag at VT 10–11 and 0.20 mag beyond, ESA SP-1200 vol. 1 §2.2; Ross 128, VT 11.45, has
//!     BT−VT 4.0, where CNS3 gives B−V 1.75). They take the B−V of
//!     the Pecaut & Mamajek (2013) dwarf sequence at their M_V ([`dwarf_bv_from_abs_mag`]) and
//!     V from Table 1.3.4 at its last node. Measured against the Johnson B−V of the Gliese–
//!     Jahreiss CNS3 (CDS V/70A) for 96 of them (CNS3 − baked): median +0.021, rms 0.042, largest
//!     0.11, against median −0.14, rms 0.37, largest 1.6 for the extrapolated table and median
//!     −0.03, rms 0.08, largest 0.22 for the clamp.
//!   - A B−V bluer than any star: a BT−VT bluer than [`BLUEST_BT_VT`] would convert to a B−V
//!     bluer than O3V (−0.33, Pecaut & Mamajek 2013), and is raised to it. In v4.0 these are
//!     5 faint (VT 12.7–13.2) white dwarfs or untyped stars with BT−VT down to −0.955.
//!   - Stars with an HR number whose BSC5 V is brighter than [`TYCHO_SATURATED_V`] take BSC5's
//!     Johnson V and B−V (same identity check as below), because there Tycho photometry is
//!     biased: measured on v4.0 against BSC5, the 8 such stars (Canopus, Rigel, Altair,
//!     Aldebaran, Antares, Spica, Achernar, Betelgeuse) come out 0.06–0.17 mag too faint
//!     (median +0.13), against a median of −0.001 ± 0.015 for V ≥ 4.
//! - `mag_src = T`, `ci` empty: a Tycho-2 supplement-1 star whose "VT" is the Hipparcos Hp, or
//!   one with only VT (Tycho-2 ReadMe notes 3–4). When it has an HR number and its `mag` is
//!   within [`BSC5_MATCH_MAX_MAG`] of BSC5's V, BSC5's Johnson V and B−V are used (Hoffleit &
//!   Warren 1991; this fixes Sirius, Vega, Arcturus, Capella, Procyon, α Cen A and B, …).
//!   Otherwise `mag` is kept as V.
//! - `mag_src = GJ` (Gliese) or `HIP`: `mag` is Johnson V and `ci` Johnson B−V.
//! - No colour left: B−V from the spectral type, using the dwarf sequence of Pecaut & Mamajek
//!   (2013, ApJS 208, 9, Table 5; E. Mamajek's version 2022.04.16); with no usable type (empty,
//!   white dwarfs), [`PLACEHOLDER_BV`].

use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail, ensure};
use flate2::read::GzDecoder;
use uni_data::{
    FLAG_EXO_HOST, FLAG_FROM_ARCHIVE, FLAG_NAMED, FLAG_NO_RV, FLAG_NO_VELOCITY, STAR3_SIZE, Star3,
    StarName,
};

const USAGE: &str = "usage: uni-bake athyg --input <athyg_40_reduced_m10.csv.gz> --out <athyg_m10.bin> \
--names <athyg_names.ron> --bsc5 <BSC5 catalog.gz>";

/// Rows of `athyg_40_reduced_m10.csv` ("AT-HYG v4.0 to magnitude 10 (332,178 stars)",
/// AT-HYG `data/subsets/README.md`).
const SUBSET_ROWS: usize = 332_178;

/// Tycho → Johnson slopes, ESA SP-1200 vol. 1 §1.3 Appendix 4 eq. 1.3.20 (= §2.2 eqs.
/// 2.2.1–2.2.2): V = VT − 0.090 (BT−VT), B−V = 0.850 (BT−VT). Table 1.3.4 tabulates the
/// corrections to these.
const TYCHO_V_PER_BT_VT: f64 = 0.090;
const TYCHO_BV_PER_BT_VT: f64 = 0.850;
/// ESA SP-1200 vol. 1 §1.3 Appendix 4, Table 1.3.4 ("Transformation from the Tycho photometric
/// system to Johnson. The points given can be used in a linear interpolation"), one node per
/// column: (BT−VT, V − VT + 0.090 (BT−VT), B−V − 0.85 (BT−VT)). The table's first and last
/// columns read "≤ −0.2" and "≥ 1.8": the end values hold beyond them (its G factor, the local
/// slope of B−V against BT−VT, is 0.85 on both sides).
const TYCHO_TABLE_1_3_4: [(f64, f64, f64); 5] = [
    (-0.2, 0.014, -0.006),
    (0.1, 0.000, 0.000),
    (0.5, -0.005, 0.046),
    (1.4, -0.005, -0.008),
    (1.8, -0.015, -0.032),
];
/// The reddest node of Table 1.3.4 [mag].
const TYCHO_TABLE_REDDEST: f64 = TYCHO_TABLE_1_3_4[4].0;

/// The bluest B−V of a star: O3V, the first entry of the Pecaut & Mamajek (2013) dwarf
/// sequence ([`DWARF_BV`]). The hottest white dwarfs are no bluer (G191-B2B, −0.33).
const BLUEST_BV: f64 = -0.330;
/// The BT−VT that Table 1.3.4 maps to [`BLUEST_BV`] (−0.381); bluer Tycho colours are raised
/// to it.
const BLUEST_BT_VT: f64 = (BLUEST_BV - TYCHO_TABLE_1_3_4[0].2) / TYCHO_BV_PER_BT_VT;

/// AT-HYG `absmag` (M_VT) [mag] from which a star redder than [`TYCHO_TABLE_REDDEST`] is taken
/// to be a main-sequence dwarf. At that colour (B−V ≳ 1.5) dwarfs are K7V or later, M_V ≥ 8.16
/// (Pecaut & Mamajek 2013, Mamajek v2022.04.16), and giants are far brighter. Measured on
/// v4.0: of the 25,819 rows redder than 1.8, 25,465 have M_VT < 5, 354 have M_VT ≥ 7 and
/// none lie between, so any limit from 5 to 7 gives the same split.
const DWARF_MIN_ABS_MAG: f64 = 6.0;

/// Largest accepted |median| [mag] of BSC5 B−V − converted Tycho B−V, checked by the bake
/// separately inside and redder than Table 1.3.4's interval: ESA's stated B−V error of the
/// transformation (SP-1200 vol. 1 §1.3 Appendix 4: "errors generally below 0.015 mag in V_J
/// and below 0.05 mag in (B−V)_J").
const BSC5_BV_CHECK_MAX: f64 = 0.05;
/// Fewest BSC5 comparison stars that make that check meaningful.
const BSC5_BV_CHECK_MIN_STARS: usize = 100;

/// BSC5 V [mag] brighter than which a Tycho-converted HR star takes BSC5's Johnson photometry
/// instead (Tycho photometry of the brightest stars is biased faint; module doc).
const TYCHO_SATURATED_V: f64 = 1.0;

/// Largest |V − BSC5 V| [mag] for which a row is taken to be the BSC5 star it names.
/// Measured on v4.0: the 15 colourless HR stars differ by Hp − V ≤ 0.37 (Sirius) and the 8
/// saturated Tycho stars by ≤ 0.17, while a V 7.7 companion that carries κ Tuc's HR 377
/// differs by 2.9.
const BSC5_MATCH_MAX_MAG: f64 = 0.5;

/// B−V given to a star with neither a colour nor a usable spectral type: G2V, 0.650, the Sun's
/// class in Pecaut & Mamajek (2013) Table 5.
const PLACEHOLDER_BV: f64 = 0.650;

/// A row closer than this [pc] is the Sun (AT-HYG v4.0 lists Sol at 4.85e-6 pc = 1 au).
const SUN_MAX_DIST_PC: f64 = 1e-3;

/// HYG's "missing or dubious parallax" distance [pc] (HYG `data/hyg/README.md`, field 9).
const PLACEHOLDER_DIST_PC: f64 = 100_000.0;

/// Largest finite binary16 [km/s]: a larger velocity component cannot be stored.
const MAX_F16: f64 = 65_504.0;

/// Largest accepted disagreement [pc] between `x0, y0, z0` and the position from
/// `ra, dec, dist`. The columns are printed to 1e-4 pc; on v4.0 the largest gap is 1.33e-4 pc.
const POS_CHECK_TOL_PC: f64 = 2e-4;

/// The id field of `Star3::id_flags` holds 22 bits.
const MAX_ATHYG_ID: u32 = (1 << 22) - 1;

/// Runs `uni-bake athyg` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    let summary = bake(&options)?;
    println!("athyg: {summary}");
    Ok(())
}

#[derive(Debug, PartialEq)]
struct Options {
    input: PathBuf,
    out: PathBuf,
    names: PathBuf,
    bsc5: PathBuf,
}

fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut input, mut out, mut names, mut bsc5) = (None, None, None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--input") => &mut input,
            Some("--out") => &mut out,
            Some("--names") => &mut names,
            Some("--bsc5") => &mut bsc5,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    match (input, out, names, bsc5) {
        (Some(input), Some(out), Some(names), Some(bsc5)) => Ok(Options {
            input,
            out,
            names,
            bsc5,
        }),
        _ => bail!("athyg needs --input, --out, --names and --bsc5\n{USAGE}"),
    }
}

/// Reads the inputs, writes both outputs and checks that they read back unchanged.
fn bake(options: &Options) -> anyhow::Result<Summary> {
    let bsc5_file =
        File::open(&options.bsc5).with_context(|| format!("opening {}", options.bsc5.display()))?;
    let bsc5 = parse_bsc5_johnson(BufReader::new(GzDecoder::new(bsc5_file)))
        .with_context(|| format!("parsing {}", options.bsc5.display()))?;
    let input = File::open(&options.input)
        .with_context(|| format!("opening {}", options.input.display()))?;
    let Baked {
        stars,
        names,
        summary,
    } = parse_catalog(BufReader::new(GzDecoder::new(input)), &bsc5)
        .with_context(|| format!("parsing {}", options.input.display()))?;
    ensure!(
        summary.rows == SUBSET_ROWS,
        "{} has {} rows; the AT-HYG subsets README documents {SUBSET_ROWS}",
        options.input.display(),
        summary.rows
    );
    ensure!(
        summary.written + summary.null_distance + summary.placeholder_distance + summary.sun
            == summary.rows,
        "row accounting: {} written + {} NULL distance + {} placeholder distance + {} Sun != {} rows",
        summary.written,
        summary.null_distance,
        summary.placeholder_distance,
        summary.sun,
        summary.rows
    );
    for (residuals, which) in [
        (
            &summary.bsc5_bv_residuals_tabulated,
            "inside ESA Table 1.3.4",
        ),
        (
            &summary.bsc5_bv_residuals_red,
            "redder than ESA Table 1.3.4",
        ),
    ] {
        let mid = median(residuals);
        ensure!(
            residuals.len() >= BSC5_BV_CHECK_MIN_STARS && mid.abs() <= BSC5_BV_CHECK_MAX,
            "Tycho → Johnson B−V {which}: median BSC5 − converted {mid:+.4} over {} stars \
             (need ≥ {BSC5_BV_CHECK_MIN_STARS} stars within ±{BSC5_BV_CHECK_MAX})",
            residuals.len()
        );
    }

    uni_data::write_athyg(&options.out, &stars)
        .with_context(|| format!("writing {}", options.out.display()))?;
    let size = std::fs::metadata(&options.out)?.len();
    ensure!(
        size == (16 + STAR3_SIZE * stars.len()) as u64,
        "{} is {size} B",
        options.out.display()
    );
    let back = uni_data::load_athyg(&options.out)
        .with_context(|| format!("reading back {}", options.out.display()))?;
    ensure!(
        back == stars,
        "{} does not read back as written",
        options.out.display()
    );

    let text =
        ron::ser::to_string_pretty(&names, ron::ser::PrettyConfig::new().compact_structs(true))?;
    std::fs::write(&options.names, text)
        .with_context(|| format!("writing {}", options.names.display()))?;
    let back = uni_data::load_star_names(&options.names)
        .with_context(|| format!("reading back {}", options.names.display()))?;
    ensure!(
        back.entries.len() == names.len(),
        "{} does not read back as written",
        options.names.display()
    );
    Ok(summary)
}

/// Johnson V and B−V of one BSC5 star.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Johnson {
    v: f64,
    bv: f64,
}

/// Johnson V and B−V of every BSC5 star that has both, by HR number. Byte columns from the
/// CDS V/50 ReadMe (1-based, inclusive): HR 1–4, Vmag 103–107, B−V 110–114.
fn parse_bsc5_johnson(reader: impl BufRead) -> anyhow::Result<HashMap<u32, Johnson>> {
    let mut table = HashMap::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("reading BSC5 line {}", index + 1))?;
        let field = |first: usize, last: usize| {
            line.get(first - 1..last.min(line.len()))
                .unwrap_or("")
                .trim()
        };
        let hr: u32 = field(1, 4)
            .parse()
            .with_context(|| format!("BSC5 line {}: HR", index + 1))?;
        let (v, bv) = (field(103, 107), field(110, 114));
        if v.is_empty() || bv.is_empty() {
            continue;
        }
        let v = v.parse().with_context(|| format!("HR {hr}: Vmag {v:?}"))?;
        let bv = bv.parse().with_context(|| format!("HR {hr}: B-V {bv:?}"))?;
        table.insert(hr, Johnson { v, bv });
    }
    Ok(table)
}

/// Splits one CSV line into fields (RFC 4180 quoting: `"a,b"` is one field, `""` inside quotes
/// is a literal quote).
pub(crate) fn split_csv(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match (c, quoted) {
            ('"', true) if chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            ('"', _) => quoted = !quoted,
            (',', false) => fields.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    fields.push(field);
    fields
}

/// Where a Tycho colour lies relative to ESA Table 1.3.4.
#[derive(Clone, Copy, Debug, PartialEq)]
enum TychoColour {
    /// Inside the table's nodes, −0.2 ≤ BT−VT ≤ 1.8: interpolated.
    Tabulated,
    /// Redder than 1.8 (not a dwarf): the table's end values, as ESA prescribes.
    RedExtrapolated,
    /// Bluer than −0.2: the table's end values.
    BlueExtrapolated,
    /// Bluer than [`BLUEST_BT_VT`]: raised to it, B−V = [`BLUEST_BV`].
    BluestStar,
}

/// How a star's V and B−V were obtained.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Photometry {
    /// Johnson V and B−V as catalogued (Gliese, Hipparcos).
    Johnson,
    /// Tycho VT, BT−VT converted to Johnson with ESA Table 1.3.4.
    Tycho { colour: TychoColour },
    /// Tycho BT−VT redder than Table 1.3.4's last node on a main-sequence dwarf: B−V from the
    /// dwarf sequence at M_V, V from the table at its last node.
    TychoRedDwarf,
    /// BSC5's Johnson V and B−V, for a row without colour (Hp or VT only) or a Tycho star
    /// brighter than [`TYCHO_SATURATED_V`].
    Bsc5 { hr: u32 },
    /// No colour: V as catalogued, B−V from the spectral type.
    SpectralType,
    /// No colour and no usable spectral type: V as catalogued, B−V = [`PLACEHOLDER_BV`].
    Placeholder,
}

#[derive(Debug)]
struct ParsedStar {
    /// The record, with `FLAG_NAMED` set when `name` is `Some`.
    star: Star3,
    /// Names, with `index` still 0 (set when the star's place in the file is known).
    name: Option<StarName>,
    photometry: Photometry,
    /// Apparent Johnson V.
    v_mag: f64,
    /// |(x0, y0, z0) − position from ra, dec, dist| [pc].
    pos_dev_pc: f64,
    /// For a Tycho-converted (not dwarf) row that is a BSC5 star: BSC5 B−V − converted B−V.
    bsc5_bv_residual: Option<f64>,
}

#[derive(Debug)]
enum Row {
    Sun,
    NullDistance,
    PlaceholderDistance,
    Star(Box<ParsedStar>),
}

/// Column positions in the AT-HYG header.
struct Columns {
    count: usize,
    id: usize,
    hip: usize,
    hr: usize,
    bayer: usize,
    flam: usize,
    con: usize,
    proper: usize,
    ra: usize,
    dec: usize,
    dist: usize,
    x0: usize,
    y0: usize,
    z0: usize,
    mag: usize,
    absmag: usize,
    ci: usize,
    mag_src: usize,
    rv: usize,
    vx: usize,
    vy: usize,
    vz: usize,
    spect: usize,
}

impl Columns {
    fn from_header(header: &str) -> anyhow::Result<Self> {
        let names = split_csv(header);
        let find = |name: &str| {
            names
                .iter()
                .position(|n| n == name)
                .with_context(|| format!("AT-HYG header has no {name:?} column"))
        };
        Ok(Self {
            count: names.len(),
            id: find("id")?,
            hip: find("hip")?,
            hr: find("hr")?,
            bayer: find("bayer")?,
            flam: find("flam")?,
            con: find("con")?,
            proper: find("proper")?,
            ra: find("ra")?,
            dec: find("dec")?,
            dist: find("dist")?,
            x0: find("x0")?,
            y0: find("y0")?,
            z0: find("z0")?,
            mag: find("mag")?,
            absmag: find("absmag")?,
            ci: find("ci")?,
            mag_src: find("mag_src")?,
            rv: find("rv")?,
            vx: find("vx")?,
            vy: find("vy")?,
            vz: find("vz")?,
            spect: find("spect")?,
        })
    }
}

/// Parses an optional number: `None` for an empty field.
fn optional<T: std::str::FromStr>(field: &str, name: &str) -> anyhow::Result<Option<T>>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    if field.is_empty() {
        return Ok(None);
    }
    field
        .parse()
        .map(Some)
        .with_context(|| format!("{name} = {field:?}"))
}

fn required<T: std::str::FromStr>(field: &str, name: &str) -> anyhow::Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    optional(field, name)?.with_context(|| format!("{name} is empty"))
}

fn non_empty(field: &str) -> Option<String> {
    (!field.is_empty()).then(|| field.to_owned())
}

fn parse_row(
    cols: &Columns,
    fields: &[String],
    bsc5: &HashMap<u32, Johnson>,
) -> anyhow::Result<Row> {
    ensure!(
        fields.len() == cols.count,
        "{} fields; the header has {}",
        fields.len(),
        cols.count
    );
    let get = |column: usize| fields[column].as_str();
    let id: u32 = required(get(cols.id), "id")?;
    ensure!(id <= MAX_ATHYG_ID, "id {id} does not fit 22 bits");
    let Some(dist) = optional::<f64>(get(cols.dist), "dist")? else {
        return Ok(Row::NullDistance);
    };
    if get(cols.proper) == "Sol" || dist < SUN_MAX_DIST_PC {
        return Ok(Row::Sun);
    }
    if dist >= PLACEHOLDER_DIST_PC {
        return Ok(Row::PlaceholderDistance);
    }
    ensure!(dist.is_finite() && dist > 0.0, "dist = {dist}");

    // Position from ra [h], dec [deg] and dist [pc]; x0, y0, z0 as a check.
    let ra = (15.0 * required::<f64>(get(cols.ra), "ra")?).to_radians();
    let dec = required::<f64>(get(cols.dec), "dec")?.to_radians();
    let pos = glam::DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()) * dist;
    let printed = glam::DVec3::new(
        required(get(cols.x0), "x0")?,
        required(get(cols.y0), "y0")?,
        required(get(cols.z0), "z0")?,
    );
    let pos_dev_pc = pos.distance(printed);

    let mag: f64 = required(get(cols.mag), "mag")?;
    let absmag: f64 = required(get(cols.absmag), "absmag")?;
    let ci: Option<f64> = optional(get(cols.ci), "ci")?;
    let hr: Option<u32> = optional(get(cols.hr), "hr")?;
    let spect = get(cols.spect);
    // The BSC5 star this row names, if its V agrees with `v` (companions carry their
    // primary's HR number).
    let bsc5_star = |v: f64| {
        let hr = hr?;
        let johnson = bsc5.get(&hr)?;
        ((johnson.v - v).abs() <= BSC5_MATCH_MAX_MAG).then_some((hr, *johnson))
    };
    let mut bsc5_bv_residual = None;
    let (v_mag, bv, photometry) = match (get(cols.mag_src), ci) {
        ("T" | "TYC", Some(bt_vt)) => {
            let (v, bv, photometry) = tycho_photometry(mag, bt_vt, absmag);
            match bsc5_star(v) {
                Some((hr, johnson)) if johnson.v < TYCHO_SATURATED_V => {
                    (johnson.v, johnson.bv, Photometry::Bsc5 { hr })
                }
                bsc5 => {
                    if let (Some((_, johnson)), Photometry::Tycho { .. }) = (bsc5, photometry) {
                        bsc5_bv_residual = Some(johnson.bv - bv);
                    }
                    (v, bv, photometry)
                }
            }
        }
        ("T" | "TYC", None) => match bsc5_star(mag) {
            Some((hr, johnson)) => (johnson.v, johnson.bv, Photometry::Bsc5 { hr }),
            _ => colour_from_type(mag, spect),
        },
        ("GJ" | "GLIESE" | "HIP" | "OTHER", Some(bv)) => (mag, bv, Photometry::Johnson),
        ("GJ" | "GLIESE" | "HIP" | "OTHER", None) => colour_from_type(mag, spect),
        (other, _) => bail!("unknown mag_src {other:?}"),
    };
    let abs_mag_v = absmag + (v_mag - mag);

    let velocity: [Option<f64>; 3] = [
        optional(get(cols.vx), "vx")?,
        optional(get(cols.vy), "vy")?,
        optional(get(cols.vz), "vz")?,
    ];
    let rv: Option<f64> = optional(get(cols.rv), "rv")?;
    let (vel_kms, motion_flags) = match velocity {
        [Some(x), Some(y), Some(z)] => {
            ensure!(
                [x, y, z].iter().all(|c| c.abs() < MAX_F16),
                "velocity ({x}, {y}, {z}) km/s does not fit binary16"
            );
            let no_rv = if rv.is_none_or(|rv| rv == 0.0) {
                FLAG_NO_RV
            } else {
                0
            };
            ([x as f32, y as f32, z as f32], no_rv)
        }
        [None, None, None] => ([0.0; 3], FLAG_NO_VELOCITY),
        partial => bail!("velocity is only partly given: {partial:?}"),
    };

    let (bayer, flam, con, proper) = (
        get(cols.bayer),
        get(cols.flam),
        get(cols.con),
        get(cols.proper),
    );
    let designation = bayer_flamsteed(bayer, flam, con);
    let hip: Option<u32> = optional(get(cols.hip), "hip")?;
    let name = (!proper.is_empty() || designation.is_some()).then(|| StarName {
        index: 0,
        proper: non_empty(proper),
        bayer_flamsteed: designation,
        con: non_empty(con),
        hip,
        spect: non_empty(spect),
    });
    let named = if name.is_some() { FLAG_NAMED } else { 0 };
    let star = Star3::new(
        pos.as_vec3().to_array(),
        abs_mag_v as f32,
        vel_kms,
        bv as f32,
        id,
        motion_flags | named,
    );
    Ok(Row::Star(Box::new(ParsedStar {
        star,
        name,
        photometry,
        v_mag,
        pos_dev_pc,
        bsc5_bv_residual,
    })))
}

/// Johnson (V, B−V) from Tycho VT and BT−VT by ESA SP-1200 vol. 1 §1.3 Appendix 4,
/// Table 1.3.4: linear interpolation between the nodes; outside them the end node's values.
fn tycho_to_johnson(vt: f64, bt_vt: f64) -> (f64, f64) {
    let table = &TYCHO_TABLE_1_3_4;
    let (dv, dbv) = match table.iter().position(|&(node, _, _)| bt_vt < node) {
        Some(0) => (table[0].1, table[0].2),
        None => (table[4].1, table[4].2),
        Some(i) => {
            let ((x0, v0, b0), (x1, v1, b1)) = (table[i - 1], table[i]);
            let t = (bt_vt - x0) / (x1 - x0);
            (v0 + (v1 - v0) * t, b0 + (b1 - b0) * t)
        }
    };
    (
        vt - TYCHO_V_PER_BT_VT * bt_vt + dv,
        TYCHO_BV_PER_BT_VT * bt_vt + dbv,
    )
}

/// Johnson V, B−V and their provenance for a Tycho row with VT `vt`, colour `bt_vt` and
/// AT-HYG `absmag` (M_VT); see the module doc.
fn tycho_photometry(vt: f64, bt_vt: f64, absmag: f64) -> (f64, f64, Photometry) {
    if bt_vt > TYCHO_TABLE_REDDEST && absmag >= DWARF_MIN_ABS_MAG {
        let (v, _) = tycho_to_johnson(vt, TYCHO_TABLE_REDDEST);
        let bv = dwarf_bv_from_abs_mag(absmag + (v - vt));
        return (v, bv, Photometry::TychoRedDwarf);
    }
    let colour = if bt_vt > TYCHO_TABLE_REDDEST {
        TychoColour::RedExtrapolated
    } else if bt_vt < BLUEST_BT_VT {
        TychoColour::BluestStar
    } else if bt_vt < TYCHO_TABLE_1_3_4[0].0 {
        TychoColour::BlueExtrapolated
    } else {
        TychoColour::Tabulated
    };
    let (v, bv) = tycho_to_johnson(vt, bt_vt.max(BLUEST_BT_VT));
    (v, bv, Photometry::Tycho { colour })
}

/// V as catalogued and B−V from the spectral type, or the placeholder.
fn colour_from_type(mag: f64, spect: &str) -> (f64, f64, Photometry) {
    match bv_from_spectral_type(spect) {
        Some(bv) => (mag, bv, Photometry::SpectralType),
        None => (mag, PLACEHOLDER_BV, Photometry::Placeholder),
    }
}

/// What the bake did, for the console and the task report.
#[derive(Debug, Default)]
struct Summary {
    rows: usize,
    written: usize,
    sun: usize,
    null_distance: usize,
    placeholder_distance: usize,
    no_velocity: usize,
    no_rv: usize,
    named: usize,
    johnson: usize,
    /// Tycho rows converted with ESA Table 1.3.4, by [`TychoColour`].
    tycho_tabulated: usize,
    tycho_red_extrapolated: usize,
    tycho_blue_extrapolated: usize,
    tycho_bluest_star: usize,
    /// Tycho rows redder than the table on main-sequence dwarfs.
    tycho_red_dwarf: usize,
    bsc5_override: usize,
    spectral_colour: usize,
    placeholder_colour: usize,
    max_pos_dev_pc: f64,
    /// BSC5 B−V − converted B−V of the Tycho-converted BSC5 stars with −0.2 ≤ BT−VT ≤ 1.8.
    bsc5_bv_residuals_tabulated: Vec<f64>,
    /// The same for those redder than 1.8 (giants; dwarfs are not converted).
    bsc5_bv_residuals_red: Vec<f64>,
}

impl Summary {
    fn tycho_converted(&self) -> usize {
        self.tycho_tabulated
            + self.tycho_red_extrapolated
            + self.tycho_blue_extrapolated
            + self.tycho_bluest_star
            + self.tycho_red_dwarf
    }
}

/// Median of `values`, or NaN when empty.
fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    match sorted.len() {
        0 => f64::NAN,
        n if n % 2 == 1 => sorted[n / 2],
        n => 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]),
    }
}

impl fmt::Display for Summary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} rows: {} stars written, {} NULL-distance rows skipped, \
             {} placeholder-distance (≥ {PLACEHOLDER_DIST_PC} pc) rows skipped, {} Sun row skipped; \
             {} named; {} without velocity (FLAG_NO_VELOCITY), {} without radial velocity (FLAG_NO_RV); \
             photometry: {} Tycho converted to Johnson by ESA Table 1.3.4 ({} inside its nodes, \
             {} redder than {TYCHO_TABLE_REDDEST} extrapolated, {} bluer than {} extrapolated, \
             {} bluer than {BLUEST_BT_VT:.3} raised to B−V {BLUEST_BV}, {} red dwarfs (M_VT ≥ {DWARF_MIN_ABS_MAG}) \
             with B−V from the dwarf sequence at M_V), \
             {} Johnson as catalogued, {} from BSC5, {} B−V from spectral type, {} B−V placeholder {PLACEHOLDER_BV}; \
             BSC5 − converted B−V: median {:+.4} over {} stars inside the table, {:+.4} over {} redder; \
             x0/y0/z0 agree with ra/dec/dist to {:.2e} pc",
            self.rows,
            self.written,
            self.null_distance,
            self.placeholder_distance,
            self.sun,
            self.named,
            self.no_velocity,
            self.no_rv,
            self.tycho_converted(),
            self.tycho_tabulated,
            self.tycho_red_extrapolated,
            self.tycho_blue_extrapolated,
            TYCHO_TABLE_1_3_4[0].0,
            self.tycho_bluest_star,
            self.tycho_red_dwarf,
            self.johnson,
            self.bsc5_override,
            self.spectral_colour,
            self.placeholder_colour,
            median(&self.bsc5_bv_residuals_tabulated),
            self.bsc5_bv_residuals_tabulated.len(),
            median(&self.bsc5_bv_residuals_red),
            self.bsc5_bv_residuals_red.len(),
            self.max_pos_dev_pc,
        )
    }
}

struct Baked {
    stars: Vec<Star3>,
    names: Vec<StarName>,
    summary: Summary,
}

fn parse_catalog(reader: impl BufRead, bsc5: &HashMap<u32, Johnson>) -> anyhow::Result<Baked> {
    let mut lines = reader.lines();
    let header = lines
        .next()
        .context("empty file")?
        .context("reading the header")?;
    let cols = Columns::from_header(&header)?;
    let mut stars = Vec::new();
    let mut names = Vec::new();
    let mut summary = Summary::default();
    for (index, line) in lines.enumerate() {
        let number = index + 2;
        let line = line.with_context(|| format!("reading line {number}"))?;
        summary.rows += 1;
        match parse_row(&cols, &split_csv(&line), bsc5).with_context(|| format!("line {number}"))? {
            Row::Sun => summary.sun += 1,
            Row::NullDistance => summary.null_distance += 1,
            Row::PlaceholderDistance => summary.placeholder_distance += 1,
            Row::Star(parsed) => {
                let ParsedStar {
                    star,
                    name,
                    photometry,
                    v_mag,
                    pos_dev_pc,
                    bsc5_bv_residual,
                } = *parsed;
                ensure!(
                    pos_dev_pc <= POS_CHECK_TOL_PC,
                    "line {number} (id {}): x0/y0/z0 differ from ra/dec/dist by {pos_dev_pc:.2e} pc",
                    star.athyg_id()
                );
                summary.max_pos_dev_pc = summary.max_pos_dev_pc.max(pos_dev_pc);
                match photometry {
                    Photometry::Johnson => summary.johnson += 1,
                    Photometry::Tycho { colour } => {
                        let (count, residuals) = match colour {
                            TychoColour::Tabulated => (
                                &mut summary.tycho_tabulated,
                                Some(&mut summary.bsc5_bv_residuals_tabulated),
                            ),
                            TychoColour::RedExtrapolated => (
                                &mut summary.tycho_red_extrapolated,
                                Some(&mut summary.bsc5_bv_residuals_red),
                            ),
                            TychoColour::BlueExtrapolated => {
                                (&mut summary.tycho_blue_extrapolated, None)
                            }
                            TychoColour::BluestStar => (&mut summary.tycho_bluest_star, None),
                        };
                        *count += 1;
                        if let (Some(residuals), Some(residual)) = (residuals, bsc5_bv_residual) {
                            residuals.push(residual);
                        }
                    }
                    Photometry::TychoRedDwarf => summary.tycho_red_dwarf += 1,
                    Photometry::Bsc5 { hr } => {
                        log::info!(
                            "AT-HYG id {}: V {v_mag:.2} and B−V {:.2} from BSC5 HR {hr}",
                            star.athyg_id(),
                            star.bv()
                        );
                        summary.bsc5_override += 1;
                    }
                    Photometry::SpectralType => summary.spectral_colour += 1,
                    Photometry::Placeholder => summary.placeholder_colour += 1,
                }
                summary.no_velocity += usize::from(star.flags() & FLAG_NO_VELOCITY != 0);
                summary.no_rv += usize::from(star.flags() & FLAG_NO_RV != 0);
                debug_assert_eq!(star.flags() & (FLAG_EXO_HOST | FLAG_FROM_ARCHIVE), 0);
                if let Some(mut name) = name {
                    name.index = u32::try_from(stars.len())?;
                    names.push(name);
                }
                stars.push(star);
            }
        }
    }
    summary.written = stars.len();
    summary.named = names.len();
    Ok(Baked {
        stars,
        names,
        summary,
    })
}

/// Mean B−V of main-sequence dwarfs at integer subtypes O0…M9, Pecaut & Mamajek (2013, ApJS
/// 208, 9) Table 5 as updated by E. Mamajek, "A Modern Mean Dwarf Stellar Color and Effective
/// Temperature Sequence", version 2022.04.16. The table starts at O3V; O0–O2 repeat O3V.
const DWARF_BV: [[f64; 10]; 7] = [
    [
        -0.330, -0.330, -0.330, -0.330, -0.326, -0.323, -0.321, -0.318, -0.315, -0.312,
    ], // O
    [
        -0.301, -0.278, -0.215, -0.178, -0.165, -0.156, -0.140, -0.128, -0.109, -0.070,
    ], // B
    [
        0.000, 0.035, 0.070, 0.100, 0.140, 0.160, 0.185, 0.210, 0.250, 0.270,
    ], // A
    [
        0.295, 0.330, 0.370, 0.390, 0.410, 0.440, 0.486, 0.500, 0.530, 0.560,
    ], // F
    [
        0.595, 0.622, 0.650, 0.660, 0.670, 0.680, 0.700, 0.710, 0.730, 0.775,
    ], // G
    [
        0.816, 0.857, 0.884, 0.990, 1.090, 1.150, 1.240, 1.340, 1.363, 1.400,
    ], // K
    [
        1.420, 1.485, 1.505, 1.530, 1.650, 1.830, 2.010, 2.120, 2.150, 2.170,
    ], // M
];
const CLASSES: &str = "OBAFGKM";

/// (M_V, B−V) of the main-sequence dwarf sequence from K4V to M9V, in steps of the table's
/// subtypes: E. Mamajek, "A Modern Mean Dwarf Stellar Color and Effective Temperature
/// Sequence", version 2022.04.16, columns `Mv` and `B-V` (Pecaut & Mamajek 2013, ApJS 208, 9).
const DWARF_MV_BV: [(f64, f64); 25] = [
    (6.98, 1.090),  // K4V
    (7.28, 1.150),  // K5V
    (7.64, 1.240),  // K6V
    (8.16, 1.340),  // K7V
    (8.43, 1.363),  // K8V
    (8.56, 1.400),  // K9V
    (8.80, 1.420),  // M0V
    (9.20, 1.445),  // M0.5V
    (9.64, 1.485),  // M1V
    (9.85, 1.495),  // M1.5V
    (10.21, 1.505), // M2V
    (10.61, 1.522), // M2.5V
    (11.15, 1.530), // M3V
    (12.10, 1.600), // M3.5V
    (12.61, 1.650), // M4V
    (13.58, 1.690), // M4.5V
    (14.15, 1.830), // M5V
    (15.30, 1.940), // M5.5V
    (16.32, 2.010), // M6V
    (17.10, 2.070), // M6.5V
    (17.70, 2.120), // M7V
    (18.16, 2.140), // M7.5V
    (18.60, 2.150), // M8V
    (19.20, 2.160), // M8.5V
    (19.40, 2.170), // M9V
];

/// B−V of a main-sequence dwarf of absolute magnitude `abs_mag_v`, interpolated linearly in
/// [`DWARF_MV_BV`]; brighter than K4V gives K4V's, fainter than M9V gives M9V's.
fn dwarf_bv_from_abs_mag(abs_mag_v: f64) -> f64 {
    let table = &DWARF_MV_BV;
    match table.iter().position(|&(mv, _)| abs_mag_v < mv) {
        Some(0) => table[0].1,
        None => table[table.len() - 1].1,
        Some(i) => {
            let ((m0, b0), (m1, b1)) = (table[i - 1], table[i]);
            b0 + (b1 - b0) * (abs_mag_v - m0) / (m1 - m0)
        }
    }
}

/// B−V from a spectral type string, read as a dwarf of its class and subtype: the first
/// upper-case O, B, A, F, G, K or M (so luminosity prefixes such as "dM4", "sdM4", "gK0" and
/// Am-star notation "kA2hA5mA7V" work), or, when the type is a lower-case Gliese photometric
/// estimate ("m", "k-m"), its first letter. The subtype may be fractional ("M3.5", interpolated)
/// or missing (5 is used). White dwarfs ("DA2") and unknown types give `None`.
fn bv_from_spectral_type(spect: &str) -> Option<f64> {
    let spect = spect.trim();
    if spect.starts_with('D') {
        return None;
    }
    let (class, rest) = match spect.char_indices().find(|&(_, c)| CLASSES.contains(c)) {
        Some((at, c)) => (c, &spect[at + c.len_utf8()..]),
        None => {
            let first = spect.chars().next()?.to_ascii_uppercase();
            (CLASSES.contains(first).then_some(first)?, &spect[1..])
        }
    };
    let row = CLASSES.find(class)?;
    let digits: String = rest
        .trim_start()
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let subtype: f64 = digits.trim_end_matches('.').parse().unwrap_or(5.0);
    // Index into the flattened O0…M9 sequence; fractional subtypes interpolate to the next one.
    let at = (row as f64 * 10.0 + subtype.min(9.0)).min(69.0);
    let lo = at.floor() as usize;
    let hi = (lo + 1).min(69);
    let value = |i: usize| DWARF_BV[i / 10][i % 10];
    Some(value(lo) + (value(hi) - value(lo)) * (at - lo as f64))
}

/// AT-HYG's three-letter Bayer abbreviations and their Greek letters.
const GREEK: [(&str, &str); 24] = [
    ("Alp", "α"),
    ("Bet", "β"),
    ("Gam", "γ"),
    ("Del", "δ"),
    ("Eps", "ε"),
    ("Zet", "ζ"),
    ("Eta", "η"),
    ("The", "θ"),
    ("Iot", "ι"),
    ("Kap", "κ"),
    ("Lam", "λ"),
    ("Mu", "μ"),
    ("Nu", "ν"),
    ("Xi", "ξ"),
    ("Omi", "ο"),
    ("Pi", "π"),
    ("Rho", "ρ"),
    ("Sig", "σ"),
    ("Tau", "τ"),
    ("Ups", "υ"),
    ("Phi", "φ"),
    ("Chi", "χ"),
    ("Psi", "ψ"),
    ("Ome", "ω"),
];
const SUPERSCRIPT_DIGITS: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];

/// The display designation: Bayer (Greek, with a superscript component number for "Alp-1")
/// when present, else Flamsteed; followed by the constellation abbreviation when known.
/// Latin-letter Bayer designations ("p") and unknown abbreviations are kept as written.
fn bayer_flamsteed(bayer: &str, flam: &str, con: &str) -> Option<String> {
    let designation = if !bayer.is_empty() {
        let (letter, component) = bayer.split_once('-').unwrap_or((bayer, ""));
        let greek = GREEK
            .iter()
            .find(|(abbr, _)| *abbr == letter)
            .map_or(letter, |(_, g)| *g);
        let superscript: String = component
            .chars()
            .map(|c| c.to_digit(10).map_or(c, |d| SUPERSCRIPT_DIGITS[d as usize]))
            .collect();
        format!("{greek}{superscript}")
    } else if !flam.is_empty() {
        flam.to_owned()
    } else {
        return None;
    };
    Some(if con.is_empty() {
        designation
    } else {
        format!("{designation} {con}")
    })
}

/// Catalogue cross-identifications of one AT-HYG star, for `uni-bake exo`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CrossIds {
    pub gaia_dr3: Option<u64>,
    pub hip: Option<u32>,
    pub hd: Option<u32>,
}

/// Reads the Gaia DR3 source id (`gaia`), Hipparcos (`hip`) and Henry Draper (`hd`) numbers of
/// every AT-HYG row, keyed by AT-HYG `id`.
pub(crate) fn read_cross_ids(path: &Path) -> anyhow::Result<HashMap<u32, CrossIds>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut lines = BufReader::new(GzDecoder::new(file)).lines();
    let header = split_csv(&lines.next().context("empty file")??);
    let find = |name: &str| {
        header
            .iter()
            .position(|n| n == name)
            .with_context(|| format!("no {name:?} column"))
    };
    let (id, gaia, hip, hd) = (find("id")?, find("gaia")?, find("hip")?, find("hd")?);
    let mut ids = HashMap::new();
    for (index, line) in lines.enumerate() {
        let fields = split_csv(&line?);
        ensure!(
            fields.len() == header.len(),
            "{}: line {} has {} fields",
            path.display(),
            index + 2,
            fields.len()
        );
        let parse = || -> anyhow::Result<(u32, CrossIds)> {
            Ok((
                required(&fields[id], "id")?,
                CrossIds {
                    gaia_dr3: optional(&fields[gaia], "gaia")?,
                    hip: optional(&fields[hip], "hip")?,
                    hd: optional(&fields[hd], "hd")?,
                },
            ))
        };
        let (key, xids) =
            parse().with_context(|| format!("{}: line {}", path.display(), index + 2))?;
        ids.insert(key, xids);
    }
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_data::{FLAG_NAMED, FLAG_NO_RV, FLAG_NO_VELOCITY};

    /// The AT-HYG v4.0 header line (athyg_40_reduced_m10.csv).
    const HEADER: &str = "id,tyc,gaia,hyg,hip,hd,hr,gl,bayer,flam,con,proper,ra,dec,pos_src,dist,x0,y0,z0,\
dist_src,mag,absmag,ci,mag_src,rv,rv_src,pmra,pmdec,pm_src,vx,vy,vz,spect,spect_src";
    /// Real AT-HYG v4.0 rows.
    const SOL: &str = "1,,,0,,,,,,,,Sol,0.0,0.0,OTHER,4.85e-06,4.85e-06,0.0,0.0,OTHER,-26.74,4.831,,OTHER,\
0.0,OTHER,0.0,0.0,OTHER,0.0,0.0,0.0,G2 V,OTHER";
    /// Sirius: a Tycho-2 supplement-1 star, so `mag` is its Hipparcos Hp (−1.088), not VT, and
    /// `ci` is empty (Tycho-2 ReadMe, CDS I/259, notes 3 and 4).
    const SIRIUS: &str = "586693,5949-2777-1,,32263,32349,48915,2491,244A,Alp,9,CMa,Sirius,6.7525694,\
-16.71314306,T,2.6371,-0.4944,2.4768,-0.7584,H,-1.088,1.806,,T,-5.5,OTHER,-546.01,-1223.07,HIP,8.59,-8.14,\
-13.06,A0m...,HYG";
    /// Hamal (α Ari, HR 617, K2 III): Tycho photometry, VT 2.125, BT−VT 1.362.
    const HAMAL: &str = "160733,1758-2416-1,,9861,9884,12929,617,84.3,Alp,13,Ari,Hamal,2.119555,23.462423,\
HIP_X,20.1776,15.7321,9.7518,8.0336,H,2.125,0.601,1.362,T,-14.412,G_R2,188.55,-148.08,HIP,-15.95,11.33,-18.73,\
K2III,HYG";
    /// Barnard's Star (Gl 699, sdM4): BT−VT 1.907, redder than ESA Table 1.3.4; M_VT 13.466.
    const BARNARD: &str = "1794252,425-2502-1,4472832130942575872,87665,87937,,,699,,,Oph,Barnard's Star,\
17.96347159,4.69327996,T,1.8282,-0.0174,-1.822,0.1496,G_R3,9.776,13.466,1.907,T,-110.468,G_R3,-801.55,\
10362.39,G_R3,-5.82,117.51,80.47,sdM4,HYG";
    /// Ross 128 (Gl 447, M4.5V): BT−VT 4.006 at VT 11.45, near Tycho's BT limit.
    const ROSS_128: &str = "1219330,272-1051-1,3796072592206250624,57375,57548,,,447,,,Vir,Ross 128,\
11.795666,0.804563,HIP_X,3.3749,-3.3697,0.1804,0.0474,G_R3,11.451,13.81,4.006,T,-31.1,HYG,607.3,-1223.03,\
G_R3,30.26,-11.35,-20.0,M4.5V,HYG";
    /// Proxima Centauri (Gl 551, M5Ve): BT−VT 2.099, M_VT 15.614.
    const PROXIMA: &str = "1440825,9010-4949-1,5853498713190525696,70666,70890,,,551,,,Cen,Proxima Centauri,\
14.495985,-62.679485,HIP_X,1.302,-0.4745,-0.3633,-1.1567,G_R3,11.187,15.614,2.099,T,-21.943,G_R3,-3781.74,\
769.47,G_R3,-9.54,22.09,21.67,M5Ve,HYG";
    /// Menkar (α Cet, HR 911, M1.5 IIIa): BT−VT 1.971, a red giant beyond Table 1.3.4.
    const MENKAR: &str = "227696,58-1618-1,,14100,14135,18884,911,,Alp,92,Cet,Menkar,3.03799228,4.08973824,T,\
76.3942,53.3427,54.4145,5.4483,H,2.716,-1.699,1.971,T,-25.47,OTHER,-10.41,-76.85,HIP,-13.7,-19.36,-29.58,\
M2 III,T";
    /// Mirach (β And, HR 337, M0 IIIa): BT−VT 1.911.
    const MIRACH: &str = "89288,2286-1329-1,,5436,5447,6860,337,53.3,Bet,43,And,Mirach,1.16220082,35.62055764,\
T,60.5327,46.9464,14.7418,35.2551,H,2.244,-1.666,1.911,T,0.609,OTHER,175.9,-112.2,HIP,3.24,53.92,-25.82,M0III,T";
    /// A DA white dwarf (Gl 838.4) with BT−VT −0.955 at VT 13.2: bluer than any star.
    const BLUE_WHITE_DWARF: &str = "2367353,548-105-1,2693940725141960192,107625,107968,,,838.4,,,Peg,,\
21.87371626,2.38878597,T,22.5603,19.1376,-11.9095,0.9403,G_R3,13.198,11.431,-0.955,T,0.0,HYG,15.32,-300.53,\
G_R3,2.0,0.68,-32.11,DA:,HYG";
    /// Canopus (α Car, HR 2326): Tycho VT −0.608, BT−VT 0.191, so converted V −0.625; BSC5 has
    /// V −0.72 (Tycho photometry of the brightest stars is biased faint).
    const CANOPUS: &str = "532909,8534-2277-1,,30365,30438,45348,2326,,Alp,,Car,Canopus,6.39919184,-52.69571799,\
T,94.7867,-5.9926,57.1319,-75.396,H,-0.608,-5.492,0.191,T,20.3,OTHER,19.93,23.24,HIP,-11.06,19.56,-9.82,F0 II,T";
    /// HIP 58897: HYG's placeholder distance 100 000 pc ("missing or dubious parallax"), which
    /// gives it a velocity of 10 686 km/s.
    const PLACEHOLDER_DISTANCE: &str = "1243398,8237-2039-1,,58722,58897,104895,,,,,Cen,,12.07867924,-50.56000224,\
T,100000.0,-63513.5035,-1308.4474,-77229.0285,H,8.986,-11.014,0.941,T,0.0,HYG,-22.3,3.3,T,-1425.61,10544.12,993.79,\
K2/3 III + F/G,T";
    /// A V = 7.7 companion of κ Tuc that carries the primary's HR 377 (BSC5 V 4.86).
    const KAP_TUC_COMPANION: &str = "96489,9136-1759-1,4692263904164105856,5884,5896,7788,377,55.3A,Kap,,Tuc,,\
1.26248899,-68.8750854,T,20.9864,7.1541,2.4546,-19.576,G_R3,7.743,6.133,,T,7.84,G_R3,386.26,82.07,G_R3,-2.59,\
39.74,-4.37,F6IV,HYG";
    /// Real BSC5 lines (CDS V/50 `catalog`) for HR 337, 377, 617, 911, 2326 and 2491.
    const BSC5_LINES: &str = concat!(
        " 337 43Bet AndBD+34  198   6860 54471  42I    949    414    010407.8+350526010943.9+353714127.10-27.10 2.06  +1.58 +1.96 +1.00   M0+IIIa           e+0.178-0.114 +.049+003V         10.1  80.4AD   5*\n",
        " 377   Kap TucCP-69   45   7788248346      W     A   454    011222.7-692426011546.1-685234299.66-48.10 4.86  +0.47 +0.03 +0.22   F6IV               +0.399+0.107 +.061+009      111  2.2   5.2AB   4*\n",
        " 617 13Alp AriBD+22  306  12929 75151  74I           725    020132.0+225923020710.4+232745144.57-36.20 2.00  +1.15 +1.12 +0.62   K2-IIICa-1         +0.190-0.148 +.049-014SB   < 17                 *\n",
        " 911 92Alp CetBD+03  419  18884110920 107I         Alp Cet  025703.0+034151030216.8+040523173.32-45.59 2.53  +1.64 +1.94 +1.16   M1.5IIIa          v-0.009-0.078 +.009-026V                         *\n",
        "2326   Alp CarCP-52  914  45348234480 245I                  062143.9-523827062357.1-524145261.21-25.29-0.72  +0.15 +0.10 +0.18   F0II               +0.022+0.021 +.028+021        0                 *\n",
        "2491  9Alp CMaBD-16 1591  48915151881 257I   5423           064044.6-163444064508.9-164258227.22-08.88-1.46   0.00 -0.05 -0.03   A1Vm               -0.553-1.205 +.375-008SBO    13 10.3  11.2AB   4*\n",
    );

    fn bsc5() -> HashMap<u32, Johnson> {
        parse_bsc5_johnson(BSC5_LINES.as_bytes()).expect("BSC5 lines parse")
    }

    fn parse(line: &str) -> Row {
        let cols = Columns::from_header(HEADER).expect("header");
        parse_row(&cols, &split_csv(line), &bsc5()).expect("row parses")
    }

    fn parse_star(line: &str) -> ParsedStar {
        match parse(line) {
            Row::Star(star) => *star,
            other => panic!("expected a star, got {other:?}"),
        }
    }

    /// Apparent V from the stored absolute magnitude and distance.
    fn apparent_v(star: &Star3) -> f64 {
        let d = glam::Vec3::from(star.pos_pc).as_dvec3().length();
        f64::from(star.abs_mag_v) + 5.0 * d.log10() - 5.0
    }

    fn ra_dec_deg(star: &Star3) -> (f64, f64) {
        let p = glam::Vec3::from(star.pos_pc).as_dvec3();
        (
            p.y.atan2(p.x).to_degrees().rem_euclid(360.0),
            (p.z / p.length()).asin().to_degrees(),
        )
    }

    #[test]
    fn bsc5_johnson_columns() {
        let table = bsc5();
        assert_eq!(table.len(), 6);
        assert_eq!(table[&337], Johnson { v: 2.06, bv: 1.58 });
        assert_eq!(table[&911], Johnson { v: 2.53, bv: 1.64 });
        assert_eq!(table[&2326], Johnson { v: -0.72, bv: 0.15 });
        assert_eq!(table[&2491], Johnson { v: -1.46, bv: 0.0 });
        assert_eq!(table[&617], Johnson { v: 2.00, bv: 1.15 });
        assert_eq!(table[&377], Johnson { v: 4.86, bv: 0.47 });
    }

    #[test]
    fn parse_sirius_row() {
        let parsed = parse_star(SIRIUS);
        let star = parsed.star;
        let d = glam::Vec3::from(star.pos_pc).length();
        assert!((d - 2.6371).abs() < 1e-4, "distance {d}");
        let (ra, dec) = ra_dec_deg(&star);
        assert!((ra - 101.287).abs() < 0.01, "RA {ra}");
        assert!((dec - -16.716).abs() < 0.01, "Dec {dec}");
        assert!(
            parsed.pos_dev_pc < 1e-4,
            "x0/y0/z0 disagree with ra/dec/dist by {} pc",
            parsed.pos_dev_pc
        );
        // Hp −1.088 replaced by BSC5's Johnson V −1.46: M_V = 1.806 + (−1.46 + 1.088) = 1.434.
        assert_eq!(parsed.photometry, Photometry::Bsc5 { hr: 2491 });
        assert!((star.abs_mag_v - 1.4).abs() < 0.1, "M_V {}", star.abs_mag_v);
        assert!(
            (star.abs_mag_v - 1.434).abs() < 1e-4,
            "M_V {}",
            star.abs_mag_v
        );
        assert!((parsed.v_mag - -1.46).abs() < 1e-9);
        assert!(star.bv().abs() < 0.02, "B−V {}", star.bv());
        assert_eq!(star.athyg_id(), 586_693);
        assert_eq!(star.flags(), FLAG_NAMED);
        let v = star.velocity_kms();
        assert!(
            (v - glam::Vec3::new(8.59, -8.14, -13.06)).length() < 0.02,
            "{v}"
        );
        let name = parsed.name.expect("Sirius is named");
        assert_eq!(name.proper.as_deref(), Some("Sirius"));
        assert_eq!(name.bayer_flamsteed.as_deref(), Some("α CMa"));
        assert_eq!(name.con.as_deref(), Some("CMa"));
        assert_eq!(name.hip, Some(32349));
        assert_eq!(name.spect.as_deref(), Some("A0m..."));
    }

    #[test]
    fn tycho_row_converted_to_johnson() {
        // Hamal (α Ari, HR 617, K2 III). Catalogued Johnson photometry: V = 2.00, B−V = +1.15
        // (Hoffleit & Warren 1991, BSC5). Unconverted, V would be 2.125 and B−V 1.362.
        let parsed = parse_star(HAMAL);
        assert_eq!(
            parsed.photometry,
            Photometry::Tycho {
                colour: TychoColour::Tabulated
            }
        );
        let star = parsed.star;
        assert!((star.bv() - 1.15).abs() < 0.05, "B−V {}", star.bv());
        assert!(
            (apparent_v(&star) - 2.00).abs() < 0.05,
            "V {}",
            apparent_v(&star)
        );
        // ESA 1997 Table 1.3.4, BT−VT 1.362 between the nodes 0.5 and 1.4:
        // V = VT − 0.090 (BT−VT) − 0.005 = 1.99742; B−V = 0.85 (BT−VT) + 0.046 − 0.054 t.
        let t = (1.362 - 0.5) / 0.9;
        assert!((parsed.v_mag - 1.99742).abs() < 1e-9);
        let bv = 0.85 * 1.362 + 0.046 + (-0.008 - 0.046) * t;
        assert!(
            (f64::from(star.bv()) - bv).abs() < 1e-3,
            "B−V {}",
            star.bv()
        );
        // M_V = absmag + (V − VT).
        assert!((star.abs_mag_v - (0.601 + 1.99742 - 2.125) as f32).abs() < 1e-5);
        // A BSC5 star converted from Tycho: its residual feeds the bake's B−V check.
        let residual = parsed.bsc5_bv_residual.expect("Hamal is HR 617");
        assert!((residual - (1.15 - f64::from(star.bv()))).abs() < 1e-3);
    }

    #[test]
    fn table_1_3_4_nodes_interpolation_and_extrapolation() {
        // ESA SP-1200 vol. 1 §1.3 Appendix 4, Table 1.3.4, read as (BT−VT, V − VT, B−V).
        let check = |bt_vt: f64, dv: f64, bv: f64| {
            let (v, got_bv) = tycho_to_johnson(10.0, bt_vt);
            assert!(
                (v - (10.0 + dv)).abs() < 1e-12,
                "{bt_vt}: V − VT {}",
                v - 10.0
            );
            assert!((got_bv - bv).abs() < 1e-12, "{bt_vt}: B−V {got_bv} vs {bv}");
        };
        check(-0.2, 0.018 + 0.014, -0.17 - 0.006);
        check(0.1, -0.009, 0.085);
        check(0.5, -0.045 - 0.005, 0.425 + 0.046);
        check(1.4, -0.126 - 0.005, 1.19 - 0.008);
        check(1.8, -0.162 - 0.015, 1.53 - 0.032);
        // Halfway between 0.1 and 0.5: half of each correction.
        check(0.3, -0.027 - 0.0025, 0.255 + 0.023);
        // Beyond the end nodes, the end values (slope 0.85 continues).
        check(2.2, -0.198 - 0.015, 1.87 - 0.032);
        check(-0.3, 0.027 + 0.014, -0.255 - 0.006);
        // Continuous at every node.
        for (node, _, _) in TYCHO_TABLE_1_3_4 {
            let (a, b) = (
                tycho_to_johnson(5.0, node - 1e-9),
                tycho_to_johnson(5.0, node + 1e-9),
            );
            assert!(
                (a.0 - b.0).abs() < 1e-8 && (a.1 - b.1).abs() < 1e-8,
                "{node}"
            );
        }
    }

    #[test]
    fn red_giant_tycho_colour_extrapolated_per_esa() {
        // Redder than the table's last node (1.8): ESA extrapolates with the end values. Clamping
        // BT−VT to 1.8 would give both B−V 1.53, 0.11 and 0.05 too blue.
        for (row, hr, v_bsc5, bv_bsc5, bt_vt, vt) in [
            (MENKAR, 911, 2.53, 1.64, 1.971, 2.716),
            (MIRACH, 337, 2.06, 1.58, 1.911, 2.244),
        ] {
            let parsed = parse_star(row);
            assert_eq!(
                parsed.photometry,
                Photometry::Tycho {
                    colour: TychoColour::RedExtrapolated
                },
                "HR {hr}"
            );
            let bv = f64::from(parsed.star.bv());
            assert!(
                (bv - (0.85 * bt_vt - 0.032)).abs() < 1e-3,
                "HR {hr}: B−V {bv}"
            );
            assert!(
                (bv - bv_bsc5).abs() < 0.02,
                "HR {hr}: B−V {bv} vs BSC5 {bv_bsc5}"
            );
            assert!((parsed.v_mag - (vt - 0.090 * bt_vt - 0.015)).abs() < 1e-9);
            assert!(
                (parsed.v_mag - v_bsc5).abs() < 0.02,
                "HR {hr}: V {} vs BSC5 {v_bsc5}",
                parsed.v_mag
            );
            assert!(parsed.bsc5_bv_residual.is_some(), "HR {hr}");
        }
    }

    #[test]
    fn red_dwarf_colour_from_dwarf_sequence() {
        // M dwarfs redder than Table 1.3.4 take the Pecaut & Mamajek dwarf B−V at their M_V,
        // with V from the table's last node (V = VT − 0.177). Catalogued Johnson B−V: CNS3
        // (Gliese & Jahreiss 1991, CDS V/70A) Gl 699 +1.74, Gl 447 +1.75, Gl 551 +1.83:.
        for (name, row, vt, absmag, cns3_bv, tolerance) in [
            ("Barnard's Star", BARNARD, 9.776, 13.466, 1.74, 0.07),
            ("Ross 128", ROSS_128, 11.451, 13.81, 1.75, 0.05),
            ("Proxima", PROXIMA, 11.187, 15.614, 1.83, 0.13),
        ] {
            let parsed = parse_star(row);
            assert_eq!(parsed.photometry, Photometry::TychoRedDwarf, "{name}");
            let v = vt - 0.090 * 1.8 - 0.015;
            assert!((parsed.v_mag - v).abs() < 1e-9);
            let abs_mag_v = absmag + (v - vt);
            assert!((f64::from(parsed.star.abs_mag_v) - abs_mag_v).abs() < 1e-5);
            let bv = f64::from(parsed.star.bv());
            assert!((bv - dwarf_bv_from_abs_mag(abs_mag_v)).abs() < 1e-3);
            assert!(
                (bv - cns3_bv).abs() < tolerance,
                "{name}: B−V {bv} vs CNS3 {cns3_bv}"
            );
            assert_eq!(parsed.bsc5_bv_residual, None);
        }
        // Ross 128's BT−VT 4.006 would give B−V 3.37 by the table and 1.53 by the old clamp.
        let ross = f64::from(parse_star(ROSS_128).star.bv());
        assert!((ross - 1.703).abs() < 2e-3, "Ross 128 B−V {ross}");
    }

    #[test]
    fn dwarf_sequence_by_absolute_magnitude() {
        // Mamajek v2022.04.16: M0V M_V 8.80 B−V 1.420, M4V 12.61 1.65, M4.5V 13.58 1.69.
        assert!((dwarf_bv_from_abs_mag(8.80) - 1.420).abs() < 1e-12);
        assert!((dwarf_bv_from_abs_mag(12.61) - 1.650).abs() < 1e-12);
        assert!((dwarf_bv_from_abs_mag(13.095) - 1.670).abs() < 1e-12);
        // Outside the table: its end values.
        assert!((dwarf_bv_from_abs_mag(5.0) - 1.090).abs() < 1e-12);
        assert!((dwarf_bv_from_abs_mag(25.0) - 2.170).abs() < 1e-12);
        // Monotonic: redder as dwarfs get fainter.
        for pair in DWARF_MV_BV.windows(2) {
            assert!(pair[1].0 > pair[0].0 && pair[1].1 >= pair[0].1, "{pair:?}");
        }
    }

    #[test]
    fn colour_bluer_than_any_star_is_raised_to_o3v() {
        let parsed = parse_star(BLUE_WHITE_DWARF);
        assert_eq!(
            parsed.photometry,
            Photometry::Tycho {
                colour: TychoColour::BluestStar
            }
        );
        assert!((f64::from(parsed.star.bv()) - BLUEST_BV).abs() < 1e-3);
        assert!((BLUEST_BT_VT - -0.381_176).abs() < 1e-6);
        assert!((parsed.v_mag - (13.198 - 0.090 * BLUEST_BT_VT + 0.014)).abs() < 1e-9);
        // Between the floor and the first node: extrapolated, not raised.
        let (_, _, photometry) = tycho_photometry(10.0, -0.3, 1.0);
        assert_eq!(
            photometry,
            Photometry::Tycho {
                colour: TychoColour::BlueExtrapolated
            }
        );
    }

    #[test]
    fn bsc5_residuals_are_collected_by_colour() {
        let csv = format!("{HEADER}\n{HAMAL}\n{MENKAR}\n{BARNARD}\n{CANOPUS}\n");
        let baked = parse_catalog(csv.as_bytes(), &bsc5()).expect("parse");
        let summary = &baked.summary;
        assert_eq!(
            (
                summary.tycho_tabulated,
                summary.tycho_red_extrapolated,
                summary.tycho_red_dwarf,
                summary.bsc5_override,
                summary.tycho_converted()
            ),
            (1, 1, 1, 1, 3)
        );
        // Hamal (HR 617) inside the table; Menkar (HR 911) redder; Barnard's Star has no HR and
        // Canopus takes BSC5's photometry, so neither is compared.
        assert_eq!(summary.bsc5_bv_residuals_tabulated.len(), 1);
        assert_eq!(summary.bsc5_bv_residuals_red.len(), 1);
        assert!(summary.bsc5_bv_residuals_red[0].abs() < 0.02);
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-12);
        assert!((median(&[4.0, 1.0, 2.0, 3.0]) - 2.5).abs() < 1e-12);
        assert!(median(&[]).is_nan());
    }

    #[test]
    fn brightest_tycho_stars_take_bsc5_photometry() {
        // Canopus: converted Tycho V −0.625 would be 0.10 mag too faint; BSC5 V −0.72, B−V 0.15.
        let parsed = parse_star(CANOPUS);
        assert_eq!(parsed.photometry, Photometry::Bsc5 { hr: 2326 });
        assert!((parsed.v_mag - -0.72).abs() < 1e-9);
        assert!((parsed.star.bv() - 0.15).abs() < 1e-3);
        assert!(
            (apparent_v(&parsed.star) - -0.72).abs() < 1e-3,
            "V {}",
            apparent_v(&parsed.star)
        );
        // Hamal (BSC5 V 2.00) is fainter than the saturation limit and keeps the conversion.
        assert_eq!(
            parse_star(HAMAL).photometry,
            Photometry::Tycho {
                colour: TychoColour::Tabulated
            }
        );
        assert_eq!(parsed.bsc5_bv_residual, None);
    }

    #[test]
    fn placeholder_distance_rows_skipped_and_counted() {
        assert!(matches!(
            parse(PLACEHOLDER_DISTANCE),
            Row::PlaceholderDistance
        ));
        let csv = format!("{HEADER}\n{PLACEHOLDER_DISTANCE}\n{HAMAL}\n");
        let baked = parse_catalog(csv.as_bytes(), &bsc5()).expect("parse");
        assert_eq!(baked.stars.len(), 1);
        assert_eq!(
            (
                baked.summary.placeholder_distance,
                baked.summary.null_distance
            ),
            (1, 0)
        );
    }

    #[test]
    fn velocity_beyond_f16_is_an_error() {
        let cols = Columns::from_header(HEADER).expect("header");
        let line = HAMAL.replace(",-15.95,11.33,-18.73,", ",-15.95,70000.0,-18.73,");
        assert!(parse_row(&cols, &split_csv(&line), &bsc5()).is_err());
    }

    #[test]
    fn companion_with_primary_hr_keeps_its_own_magnitude() {
        // Its V 7.74 differs from BSC5's 4.86 for HR 377 by far more than Hp − V, so no override;
        // with no colour, B−V comes from its spectral type F6 (Pecaut & Mamajek 2013: 0.486).
        let parsed = parse_star(KAP_TUC_COMPANION);
        assert_eq!(parsed.photometry, Photometry::SpectralType);
        assert!((parsed.v_mag - 7.743).abs() < 1e-9);
        assert!(
            (parsed.star.bv() - 0.486).abs() < 1e-3,
            "B−V {}",
            parsed.star.bv()
        );
        assert!((parsed.star.abs_mag_v - 6.133).abs() < 1e-5);
    }

    #[test]
    fn sol_is_excluded() {
        assert!(matches!(parse(SOL), Row::Sun));
        let csv = format!("{HEADER}\n{SOL}\n{HAMAL}\n");
        let baked = parse_catalog(csv.as_bytes(), &bsc5()).expect("parse");
        assert_eq!(baked.stars.len(), 1);
        assert_eq!(baked.stars[0].athyg_id(), 160_733);
        assert_eq!(
            (baked.summary.rows, baked.summary.sun, baked.summary.written),
            (2, 1, 1)
        );
    }

    #[test]
    fn null_distance_rows_skipped_and_counted() {
        // Review Focus 3. One row with an empty distance (and so no absmag, position or
        // velocity), one with a distance but no velocity (no proper motion).
        let no_distance = "21,4-1-1,,,,,,,,,And,,0.004,1.5,T,,,,,N,9.5,,0.5,T,,N,,,N,,,,G5,T";
        let no_velocity = "22,4-2-1,,,,,,,,,And,,0.004,1.6,T,100.0,99.961,0.1047,2.7922,G_R3,9.5,4.5,0.5,T,,N,,,N,,,,G5,T";
        let csv = format!("{HEADER}\n{no_distance}\n{HAMAL}\n{no_velocity}\n");
        let baked = parse_catalog(csv.as_bytes(), &bsc5()).expect("parse");
        let summary = &baked.summary;
        assert_eq!(summary.rows, 3);
        assert_eq!(summary.null_distance, 1);
        assert_eq!(summary.no_velocity, 1);
        assert_eq!(baked.stars.len(), 2);
        let stored = baked
            .stars
            .iter()
            .find(|s| s.athyg_id() == 22)
            .expect("row 22 is kept");
        assert_eq!(stored.flags() & FLAG_NO_VELOCITY, FLAG_NO_VELOCITY);
        assert_eq!(
            stored.flags() & FLAG_NO_RV,
            0,
            "no velocity at all is not 'tangential only'"
        );
        assert_eq!(stored.velocity_kms(), glam::Vec3::ZERO);
        for star in &baked.stars {
            let finite = glam::Vec3::from(star.pos_pc).is_finite()
                && star.abs_mag_v.is_finite()
                && star.velocity_kms().is_finite()
                && star.bv().is_finite();
            assert!(finite, "{star:?}");
        }
    }

    #[test]
    fn missing_rv_flagged() {
        // AT-HYG computed the velocity "assuming no radial component" when RV is missing or 0.
        let with_rv = parse_star(HAMAL).star;
        assert_eq!(with_rv.flags() & FLAG_NO_RV, 0);
        let empty_rv = HAMAL.replace(",-14.412,G_R2,", ",,N,");
        assert_eq!(parse_star(&empty_rv).star.flags() & FLAG_NO_RV, FLAG_NO_RV);
        let zero_rv = HAMAL.replace(",-14.412,G_R2,", ",0.0,HYG,");
        assert_eq!(parse_star(&zero_rv).star.flags() & FLAG_NO_RV, FLAG_NO_RV);
    }

    #[test]
    fn names_follow_written_indices() {
        let unnamed = "22,4-2-1,,,,,,,,,And,,0.004,1.6,T,100.0,99.961,0.1047,2.7922,G_R3,9.5,4.5,0.5,T,1.0,G_R3,1.0,1.0,G_R3,1.0,1.0,1.0,G5,T";
        let csv = format!("{HEADER}\n{SOL}\n{unnamed}\n{SIRIUS}\n{HAMAL}\n");
        let baked = parse_catalog(csv.as_bytes(), &bsc5()).expect("parse");
        assert_eq!(baked.stars.len(), 3);
        assert_eq!(baked.summary.named, 2);
        let indices: Vec<u32> = baked.names.iter().map(|n| n.index).collect();
        assert_eq!(indices, [1, 2]);
        assert_eq!(baked.names[0].proper.as_deref(), Some("Sirius"));
        assert_eq!(baked.names[1].proper.as_deref(), Some("Hamal"));
        assert!(!baked.stars[0].is_named());
        assert!(baked.stars[1].is_named() && baked.stars[2].is_named());
    }

    #[test]
    fn unknown_magnitude_source_is_an_error() {
        let cols = Columns::from_header(HEADER).expect("header");
        let line = HAMAL.replace(",1.362,T,", ",1.362,XYZ,");
        assert!(parse_row(&cols, &split_csv(&line), &bsc5()).is_err());
    }

    #[test]
    fn missing_column_is_an_error() {
        assert!(Columns::from_header(&HEADER.replace(",ci,", ",colour,")).is_err());
    }

    #[test]
    fn csv_split_handles_quotes() {
        assert_eq!(split_csv("a,,b"), ["a", "", "b"]);
        assert_eq!(split_csv("1,\"M1-M3,5IB\",T"), ["1", "M1-M3,5IB", "T"]);
        assert_eq!(split_csv("\"say \"\"hi\"\"\",x"), ["say \"hi\"", "x"]);
        assert_eq!(
            split_csv("\"Kepler-257 d\",,1.5"),
            ["Kepler-257 d", "", "1.5"]
        );
        assert_eq!(split_csv(""), [""]);
    }

    #[test]
    fn spectral_type_colour_fallback() {
        // Pecaut & Mamajek (2013, ApJS 208, 9) Table 5, dwarf sequence (Mamajek 2022.04.16).
        let close = |spect: &str, want: f64| {
            let got =
                bv_from_spectral_type(spect).unwrap_or_else(|| panic!("{spect:?} has a class"));
            assert!((got - want).abs() < 1e-9, "{spect:?}: {got} vs {want}");
        };
        close("G2V", 0.650);
        close("dM4", 1.65);
        close("sdM4", 1.65);
        close("M3.5", 1.59); // midway between M3V 1.53 and M4V 1.65
        close("K3/4 V", 0.990);
        close("B 9 V", -0.070);
        close("A 0 P(SR,EU)", 0.000);
        close("kA2hA5mA7V", 0.070); // Am notation: the first upper-case class (A2) is used
        close("gK0", 0.816);
        close("O3V", -0.330);
        close("O1", -0.330); // earlier than the table's O3 → O3
        // No subtype → the class's subtype 5; lower-case Gliese types are photometric estimates.
        close("m", 1.83);
        close("k-m", 1.15);
        close("g-k", 0.680);
        close("M:", 1.83);
        // White dwarfs and empty types have no main-sequence colour.
        assert_eq!(bv_from_spectral_type("DA2"), None);
        assert_eq!(bv_from_spectral_type("DC8"), None);
        assert_eq!(bv_from_spectral_type(""), None);
        assert_eq!(bv_from_spectral_type("C6,3"), None);
    }

    #[test]
    fn bayer_flamsteed_designations() {
        assert_eq!(bayer_flamsteed("Alp", "9", "CMa").as_deref(), Some("α CMa"));
        assert_eq!(
            bayer_flamsteed("Alp-1", "", "Cen").as_deref(),
            Some("α¹ Cen")
        );
        assert_eq!(
            bayer_flamsteed("Psi-9", "", "Aqr").as_deref(),
            Some("ψ⁹ Aqr")
        );
        assert_eq!(bayer_flamsteed("", "61", "Cyg").as_deref(), Some("61 Cyg"));
        assert_eq!(bayer_flamsteed("p", "", "Eri").as_deref(), Some("p Eri"));
        assert_eq!(bayer_flamsteed("Ome", "", "").as_deref(), Some("ω"));
        assert_eq!(bayer_flamsteed("", "", "Cyg"), None);
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&[
            "--input",
            "in.csv.gz",
            "--out",
            "s.bin",
            "--names",
            "n.ron",
            "--bsc5",
            "catalog.gz",
        ]))
        .expect("valid");
        assert_eq!(
            options,
            Options {
                input: "in.csv.gz".into(),
                out: "s.bin".into(),
                names: "n.ron".into(),
                bsc5: "catalog.gz".into(),
            }
        );
        assert!(parse_options(&args(&["--input", "in.csv.gz", "--out", "s.bin"])).is_err());
        assert!(parse_options(&args(&["--input"])).is_err());
        assert!(parse_options(&args(&["--bogus", "x"])).is_err());
    }
}
