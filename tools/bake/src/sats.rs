//! `uni-bake sats` (M1 Task A9 Step 3): JPL SSD satellite elements and physical parameters → `sat_elements.ron`.
//!
//! `uni-bake sats --out data/baked/sat_elements.ron [--cache <dir>]`
//!
//! Downloads <https://ssd.jpl.nasa.gov/sats/elem/> and <https://ssd.jpl.nasa.gov/sats/phys_par/>
//! with `curl` into `--cache` (default `data/cache/jpl-sats/`; delete it after the bake), parses
//! their HTML tables (`sat_elem`, `sat_phys_par`) and joins them on the JPL satellite code (the
//! NAIF id). Each table's header must read as expected, or the bake stops before writing.
//!
//! Every moon of the elements table is kept. Where JPL lists a moon twice (Puck: URA182 at
//! 2000-01-01.5, URA184 at 2025-01-01.0), the row with the later epoch, the newer fit, is kept.
//! The reference-plane pole of each frame (see `uni_data::sats`):
//! - "Laplace": JPL's R.A. and Dec. columns.
//! - "ecliptic": the pole of the J2000 mean ecliptic, RA 270°, Dec 90° − ε₀, with ε₀ =
//!   84381.448″ (IAU 1976, Lieske et al. 1977, A&A 58, 1), the obliquity JPL uses for its
//!   "ecliptic and mean equinox of J2000" frame.
//! - "equatorial": the planet's pole from the IAU WGCCRE 2015 report (Archinal et al. 2018, CMDA
//!   130:22), evaluated at the row's epoch. JPL does not tabulate it. For Uranus the pole is the
//!   one about which the moons orbit prograde, the opposite of the IAU north pole (Uranus rotates
//!   retrograde). JPL's own Laplace poles for Uranus's inner moons, 77.3°/15.2° with a 180° tilt
//!   to the equator, confirm that sign.
//!
//! The mean motion is 360° over JPL's sidereal period P. A precession period that JPL gives as
//! "-" (not determined) or as 0 (none tabulated: those rows are circular or uninclined) is
//! `f64::INFINITY`. A GM that JPL gives only as an upper limit ("<0.0002") is `None`.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, bail, ensure};
use uni_data::SatRecord;

use crate::sbdb::{curl_get, utc_now, write_ron};

const USAGE: &str = "usage: uni-bake sats --out <sat_elements.ron> [--cache <dir>]";

const ELEM_URL: &str = "https://ssd.jpl.nasa.gov/sats/elem/";
const PHYS_URL: &str = "https://ssd.jpl.nasa.gov/sats/phys_par/";

/// Runs `uni-bake sats` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    std::fs::create_dir_all(&options.cache)
        .with_context(|| format!("creating {}", options.cache.display()))?;
    let fetched = utc_now();
    let mut pages = Vec::new();
    for (url, file) in [(ELEM_URL, "elem.html"), (PHYS_URL, "phys_par.html")] {
        let raw = options.cache.join(file);
        log::info!("sats: {url} -> {}", raw.display());
        curl_get(url, &[], &raw)?;
        pages.push(
            std::fs::read_to_string(&raw).with_context(|| format!("reading {}", raw.display()))?,
        );
    }
    let elem_table = html_table(&pages[0], "sat_elem").context(ELEM_URL)?;
    let rows = elem_table.rows.len();
    let mut sats = elem_records(&elem_table).context(ELEM_URL)?;
    let phys =
        phys_rows(&html_table(&pages[1], "sat_phys_par").context(PHYS_URL)?).context(PHYS_URL)?;
    merge(&mut sats, &phys)?;

    let mut by_planet = BTreeMap::new();
    for sat in &sats {
        *by_planet.entry(sat.planet_naif).or_insert(0usize) += 1;
    }
    let planets: Vec<String> = by_planet.iter().map(|(p, n)| format!("{p}: {n}")).collect();
    let header = vec![
        "JPL SSD planetary-satellite mean elements and physical parameters (M1 Task A9 Step 3).".to_owned(),
        format!("Written by `uni-bake sats` from {ELEM_URL} and {PHYS_URL},"),
        format!("fetched {fetched} UTC (`date -u`). Public JPL Solar System Dynamics data; see ATTRIBUTION.md and"),
        "data/manifest.toml (ids \"jpl-sats-elem\", \"jpl-sats-phys\").".to_owned(),
        String::new(),
        "JPL: these mean elements are \"not intended for ephemeris computation\"; they describe a".to_owned(),
        "precessing ellipse fitted to each numerically integrated orbit. Use Horizons for accuracy.".to_owned(),
        String::new(),
        "a_km, e, argp_deg, m_deg, i_deg, node_deg: JPL's a, e, omega, M, i, node at epoch_jd_tdb (JPL's".to_owned(),
        "TDB calendar epoch as a Julian Date). The node is measured from the reference plane's ascending".to_owned(),
        "node on the ICRF equator. n_deg_per_day = 360 / P (JPL's sidereal period, days). p_apsides_yr,".to_owned(),
        "p_node_yr: JPL's precession periods; inf where JPL gives \"-\" or 0 (no precession).".to_owned(),
        "laplace_ra_deg, laplace_dec_deg: the pole of each row's reference plane (JPL's Frame column):".to_owned(),
        "  Laplace: JPL's R.A./Dec. columns.".to_owned(),
        "  ecliptic: J2000 mean ecliptic pole, RA 270, Dec 90 - 84381.448\" (IAU 1976 obliquity).".to_owned(),
        "  equatorial: the planet's WGCCRE 2015 pole (Archinal et al. 2018; NAIF pck00011) at the row's".to_owned(),
        "  epoch; for Uranus the prograde pole 77.311/15.175, opposite to the IAU north pole.".to_owned(),
        "radius_km, gm_km3_s2: JPL's mean radius and GM (/sats/phys_par/); None where not listed, or".to_owned(),
        "where the GM is only an upper limit.".to_owned(),
        format!(
            "{rows} table rows; a moon listed twice keeps its later-epoch row (Puck: URA184 2025); {} moons.",
            sats.len()
        ),
        format!("Moons per planet: {}.", planets.join(", ")),
    ];
    write_ron(&options.out, &header, &sats, true)?;
    let back =
        uni_data::load_sat_elements(&options.out).context("reading back the written file")?;
    ensure!(
        back.len() == sats.len(),
        "read back {} of {} moons",
        back.len(),
        sats.len()
    );
    println!(
        "sats: {} moons ({} with radius, {} with GM) from {rows} rows -> {} ({})",
        sats.len(),
        sats.iter().filter(|s| s.radius_km.is_some()).count(),
        sats.iter().filter(|s| s.gm_km3_s2.is_some()).count(),
        options.out.display(),
        planets.join(", ")
    );
    Ok(())
}

/// One HTML table: header cell texts and body rows of cell texts (tags stripped, entities
/// decoded, whitespace collapsed).
#[derive(Debug)]
struct HtmlTable {
    header: Vec<String>,
    rows: Vec<Vec<String>>,
}

/// The table with `id`. A cell runs from its `<td>`/`<th>` start tag to the next cell or row tag,
/// so unclosed `<td>`s parse too; `<br>` is a space and every other tag is dropped.
fn html_table(html: &str, id: &str) -> anyhow::Result<HtmlTable> {
    let marker = format!("id=\"{id}\"");
    let at = html
        .find(&marker)
        .with_context(|| format!("no table with id {id:?}"))?;
    let open = html[..at].rfind('<').context("no tag around the id")?;
    ensure!(
        html[open..].starts_with("<table"),
        "{marker} is not on a <table> tag"
    );
    let end = html[open..]
        .find("</table>")
        .with_context(|| format!("table {id:?} never ends"))?;
    let body = &html[open..open + end];

    let (mut header, mut rows) = (Vec::new(), Vec::new());
    let mut in_head = false;
    let mut row: Option<Vec<String>> = None;
    let mut cell: Option<String> = None;
    let finish_cell = |row: &mut Option<Vec<String>>, cell: &mut Option<String>| {
        if let (Some(row), Some(text)) = (row.as_mut(), cell.take()) {
            row.push(clean(&text));
        }
    };
    let mut rest = body;
    while let Some(lt) = rest.find('<') {
        // As in a browser, '<' starts a tag only before a letter, '/' or '!': the live
        // physical-parameters page writes upper limits as a bare "<0.0002".
        let starts_tag =
            rest[lt + 1..].starts_with(|c: char| c.is_ascii_alphabetic() || c == '/' || c == '!');
        let text_end = if starts_tag { lt } else { lt + 1 };
        if let Some(text) = cell.as_mut() {
            text.push_str(&rest[..text_end]);
        }
        if !starts_tag {
            rest = &rest[text_end..];
            continue;
        }
        let gt = rest[lt..].find('>').map_or(rest.len(), |g| lt + g + 1);
        let tag = &rest[lt + 1..gt.saturating_sub(1).max(lt + 1)];
        let closing = tag.starts_with('/');
        let name: String = tag
            .trim_start_matches('/')
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>()
            .to_ascii_lowercase();
        match (name.as_str(), closing) {
            ("thead", false) => in_head = true,
            ("thead", true) => in_head = false,
            ("tr", false) => {
                finish_cell(&mut row, &mut cell);
                row = Some(Vec::new());
            }
            ("tr", true) | ("tbody", _) | ("table", _) => {
                finish_cell(&mut row, &mut cell);
                if let Some(done) = row.take().filter(|r| !r.is_empty()) {
                    if in_head {
                        header.extend(done);
                    } else {
                        rows.push(done);
                    }
                }
            }
            ("td" | "th", false) => {
                finish_cell(&mut row, &mut cell);
                cell = Some(String::new());
            }
            ("td" | "th", true) => finish_cell(&mut row, &mut cell),
            ("br", _) => {
                if let Some(text) = cell.as_mut() {
                    text.push(' ');
                }
            }
            _ => {}
        }
        rest = &rest[gt..];
    }
    finish_cell(&mut row, &mut cell);
    if let Some(done) = row.take().filter(|r| !r.is_empty()) {
        rows.push(done);
    }
    Ok(HtmlTable { header, rows })
}

/// Decodes the HTML entities these pages use and collapses whitespace (also used by `facts`).
pub(crate) fn clean(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        let entity = after
            .find(';')
            .filter(|&semi| semi <= 8)
            .map(|semi| &after[..semi]);
        let decoded = entity.and_then(|e| match e {
            "nbsp" => Some(' '),
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "omega" => Some('ω'),
            "pm" => Some('±'),
            "deg" => Some('°'),
            "minus" => Some('−'),
            _ => e
                .strip_prefix("#x")
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| e.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                .and_then(char::from_u32),
        });
        match (entity, decoded) {
            (Some(entity), Some(c)) => {
                out.push(c);
                rest = &after[entity.len() + 1..];
            }
            _ => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The elements table's header, as `clean` reads it.
#[rustfmt::skip]
const ELEM_HEADER: [&str; 20] = [
    "ID", "Planet", "Satellite", "Code", "Ephemeris", "Frame", "Epoch (TDB)", "a (km)", "e",
    "ω (deg)", "M (deg)", "i (deg)", "node (deg)", "P (days)", "Papsis (yr)", "Pnode (yr)",
    "R.A. (deg)", "Dec. (deg)", "Tilt (deg)", "Ref.",
];

/// NAIF ids of the planets (body centres) by the name JPL's tables use.
const PLANETS: [(&str, i32); 7] = [
    ("Earth", 399),
    ("Mars", 499),
    ("Jupiter", 599),
    ("Saturn", 699),
    ("Uranus", 799),
    ("Neptune", 899),
    ("Pluto", 999),
];

/// A finite number from a table cell.
fn number(field: &str, what: &str) -> anyhow::Result<f64> {
    field
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .with_context(|| format!("{what} {field:?} is not a number"))
}

/// The elements table's rows as records, without radius or GM. Where a moon is listed twice, the
/// row with the later epoch is kept, in its own place; two rows at one epoch are an error.
fn elem_records(table: &HtmlTable) -> anyhow::Result<Vec<SatRecord>> {
    ensure!(
        table.header == ELEM_HEADER,
        "the elements table's columns changed: {:?}",
        table.header
    );
    let mut kept: Vec<SatRecord> = Vec::new();
    for (index, cells) in table.rows.iter().enumerate() {
        let sat = elem_record(cells).with_context(|| format!("elements row {index}: {cells:?}"))?;
        match kept.iter_mut().find(|k| k.naif == sat.naif) {
            None => kept.push(sat),
            Some(old) if old.epoch_jd_tdb == sat.epoch_jd_tdb => {
                bail!("moon {} is listed twice at one epoch", sat.naif)
            }
            Some(old) if old.epoch_jd_tdb < sat.epoch_jd_tdb => *old = sat,
            Some(_) => {}
        }
    }
    Ok(kept)
}

fn elem_record(cells: &[String]) -> anyhow::Result<SatRecord> {
    ensure!(
        cells.len() == ELEM_HEADER.len(),
        "{} cells, not {}",
        cells.len(),
        ELEM_HEADER.len()
    );
    let planet_naif = PLANETS
        .iter()
        .find(|(name, _)| *name == cells[1])
        .map(|&(_, naif)| naif)
        .with_context(|| format!("unknown planet {:?}", cells[1]))?;
    ensure!(!cells[2].is_empty(), "no satellite name");
    let naif: i32 = cells[3]
        .parse()
        .ok()
        .filter(|&n| n > 0)
        .context("code is not a NAIF id")?;
    let epoch_jd_tdb = calendar_jd(&cells[6])?;
    let a_km = number(&cells[7], "a")?;
    let e = number(&cells[8], "e")?;
    let i_deg = number(&cells[11], "i")?;
    let period_d = number(&cells[13], "P")?;
    ensure!(a_km > 0.0 && period_d > 0.0, "a and P must be > 0");
    ensure!((0.0..1.0).contains(&e), "e = {e} is outside [0, 1)");
    ensure!(
        (0.0..=180.0).contains(&i_deg),
        "i = {i_deg} is outside [0, 180]"
    );
    let (laplace_ra_deg, laplace_dec_deg) =
        reference_pole(&cells[5], planet_naif, &cells[16], &cells[17], epoch_jd_tdb)?;
    Ok(SatRecord {
        naif,
        name: cells[2].clone(),
        planet_naif,
        a_km,
        e,
        argp_deg: number(&cells[9], "omega")?,
        m_deg: number(&cells[10], "M")?,
        i_deg,
        node_deg: number(&cells[12], "node")?,
        n_deg_per_day: 360.0 / period_d,
        p_apsides_yr: period_yr(&cells[14])?,
        p_node_yr: period_yr(&cells[15])?,
        epoch_jd_tdb,
        laplace_ra_deg,
        laplace_dec_deg,
        radius_km: None,
        gm_km3_s2: None,
    })
}

/// One row of the physical-parameters table: code, GM [km³/s²] (`None` for an upper limit),
/// mean radius [km].
#[derive(Debug, PartialEq)]
struct Phys {
    naif: i32,
    gm_km3_s2: Option<f64>,
    radius_km: f64,
}

fn phys_rows(table: &HtmlTable) -> anyhow::Result<Vec<Phys>> {
    let h = &table.header;
    let layout_ok = h.len() == 15
        && h[..3] == ["Planet", "Satellite", "Code"]
        && h[3].starts_with("GM")
        && h[4].starts_with("Mean Radius")
        && h[5].starts_with("Mean Density");
    ensure!(
        layout_ok,
        "the physical-parameters table's columns changed: {h:?}"
    );
    table
        .rows
        .iter()
        .enumerate()
        .map(|(index, cells)| {
            let row = || format!("physical row {index}: {cells:?}");
            ensure!(
                cells.len() == 12,
                "{}: {} cells, not 12",
                row(),
                cells.len()
            );
            let naif = cells[2]
                .parse()
                .ok()
                .filter(|&n: &i32| n > 0)
                .with_context(row)?;
            let gm = &cells[3];
            let gm_km3_s2 = if gm.starts_with('<') || gm == "n/a" {
                None
            } else {
                Some(number(gm, "GM").with_context(row)?).filter(|&g| g > 0.0)
            };
            let radius_km = number(&cells[6], "mean radius").with_context(row)?;
            ensure!(radius_km > 0.0, "{}: radius <= 0", row());
            Ok(Phys {
                naif,
                gm_km3_s2,
                radius_km,
            })
        })
        .collect()
}

/// Fills each record's radius and GM; every physical row must name a moon of the elements table,
/// once.
fn merge(sats: &mut [SatRecord], phys: &[Phys]) -> anyhow::Result<()> {
    for (index, p) in phys.iter().enumerate() {
        ensure!(
            !phys[..index].iter().any(|q| q.naif == p.naif),
            "moon {} has two physical rows",
            p.naif
        );
        let sat = sats
            .iter_mut()
            .find(|s| s.naif == p.naif)
            .with_context(|| format!("moon {} has physical parameters but no elements", p.naif))?;
        sat.radius_km = Some(p.radius_km);
        sat.gm_km3_s2 = p.gm_km3_s2;
    }
    Ok(())
}

/// Julian Date of a calendar date `YYYY-MM-DD.ddd` (Gregorian, the day's fraction after the dot):
/// Meeus, Astronomical Algorithms (2nd ed. 1998), ch. 7, eq. 7.1.
fn calendar_jd(date: &str) -> anyhow::Result<f64> {
    let bad = || format!("epoch {date:?} is not YYYY-MM-DD.d");
    let mut parts = date.splitn(3, '-');
    let (Some(y), Some(m), Some(d)) = (parts.next(), parts.next(), parts.next()) else {
        bail!(bad());
    };
    let (day, fraction) = d.split_once('.').with_context(bad)?;
    let digits = |s: &str, n: usize| s.len() == n && s.bytes().all(|b| b.is_ascii_digit());
    ensure!(
        digits(y, 4)
            && digits(m, 2)
            && digits(day, 2)
            && !fraction.is_empty()
            && fraction.bytes().all(|b| b.is_ascii_digit()),
        bad()
    );
    let (year, month, day): (i64, i64, i64) = (y.parse()?, m.parse()?, day.parse()?);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    ensure!(
        (1..=12).contains(&month) && (1..=month_days[month as usize - 1]).contains(&day),
        bad()
    );
    let fraction: f64 = format!("0.{fraction}").parse()?;
    let (y, m) = if month <= 2 {
        (year - 1, month + 12)
    } else {
        (year, month)
    };
    let a = y.div_euclid(100);
    let b = 2 - a + a.div_euclid(4);
    let whole = (365.25 * (y + 4716) as f64).floor() + (30.6001 * (m + 1) as f64).floor();
    Ok(whole + day as f64 + b as f64 - 1524.5 + fraction)
}

/// J2000 mean obliquity of the ecliptic ε₀ = 84381.448″ (IAU 1976; Lieske et al. 1977).
const OBLIQUITY_J2000_ARCSEC: f64 = 84_381.448;

/// The reference-plane pole (RA, Dec) [deg] of a row: `frame` is JPL's frame name, `ra`/`dec`
/// the row's own columns (empty unless "Laplace"), `jd` the row's epoch (TDB).
///
/// Equatorial poles: IAU WGCCRE 2015 (Archinal et al. 2018, CMDA 130:22), as NAIF `pck00011.tpc`
/// gives them (checked 2026-11-22), T in Julian centuries (TDB) from J2000:
/// Saturn 40.589 − 0.036 T, 83.537 − 0.004 T; Uranus 257.311, −15.175, negated here to the
/// prograde pole 77.311, +15.175; Neptune 299.36 + 0.70 sin N, 43.46 − 0.51 cos N, with
/// N = 357.85° + 52.316° T; Pluto 132.993, −6.163.
fn reference_pole(
    frame: &str,
    planet_naif: i32,
    ra: &str,
    dec: &str,
    jd: f64,
) -> anyhow::Result<(f64, f64)> {
    let t = (jd - 2_451_545.0) / 36_525.0;
    let (ra, dec) = match frame {
        "Laplace" => (
            number(ra, "Laplace pole R.A.")?,
            number(dec, "Laplace pole Dec.")?,
        ),
        "ecliptic" => (270.0, 90.0 - OBLIQUITY_J2000_ARCSEC / 3600.0),
        "equatorial" => match planet_naif {
            699 => (40.589 - 0.036 * t, 83.537 - 0.004 * t),
            // The IAU north pole (257.311, −15.175) negated: RA + 180°, −Dec.
            799 => (77.311, 15.175),
            899 => {
                let n = (357.85 + 52.316 * t).to_radians();
                (299.36 + 0.70 * n.sin(), 43.46 - 0.51 * n.cos())
            }
            999 => (132.993, -6.163),
            other => bail!("no equatorial pole for planet {other}"),
        },
        other => bail!("unknown frame {other:?}"),
    };
    ensure!(
        (-90.0..=90.0).contains(&dec),
        "pole Dec {dec} is outside [-90, 90]"
    );
    Ok((ra.rem_euclid(360.0), dec))
}

/// A precession period [yr]; "-" (not determined) or 0 (none tabulated) is infinite.
fn period_yr(field: &str) -> anyhow::Result<f64> {
    if field == "-" {
        return Ok(f64::INFINITY);
    }
    let period = number(field, "precession period")?;
    ensure!(period >= 0.0, "precession period {period} < 0");
    Ok(if period == 0.0 { f64::INFINITY } else { period })
}

#[derive(Debug)]
struct Options {
    out: PathBuf,
    cache: PathBuf,
}

/// `--out` (required) and `--cache` (default `<data>/cache/jpl-sats`).
fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut out, mut cache) = (None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--out") => &mut out,
            Some("--cache") => &mut cache,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    let out = out.with_context(|| format!("sats needs --out\n{USAGE}"))?;
    let cache = cache.unwrap_or_else(|| uni_data::data_dir().join("cache").join("jpl-sats"));
    Ok(Options { out, cache })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = r#"<table id="sat_elem" class="sat-elem table">
  <thead>
    <tr>
      <th>ID</th>
      <th>Planet<br>&nbsp;</th>
      <th>Satellite<br>&nbsp;</th>
      <th>Code<br>&nbsp;</th>
      <th class="text-center">Ephemeris<br>&nbsp;</th>
      <th class="text-center">Frame<br>&nbsp;</th>
      <th>Epoch<br>(TDB)</th>
      <th class="text-center">a<br>(km)</th>
      <th class="text-center">e<br>&nbsp;</th>
      <th class="text-center">&omega;<br>(deg)</th>
      <th class="text-center">M<br>(deg)</th>
      <th class="text-center">i<br>(deg)</th>
      <th class="text-center">node<br>(deg)</th>
      <th class="text-center">P<br>(days)</th>
      <th class="text-center">P<sub>apsis</sub><br>(yr)</th>
      <th class="text-center">P<sub>node</sub><br>(yr)</th>
      <th class="text-center">R.A.<br>(deg)</th>
      <th class="text-center">Dec.<br>(deg)</th>
      <th class="text-center">Tilt<br>(deg)</th>
      <th class="text-center">Ref.<br>&nbsp;</th>
    </tr>
  </thead>
  <tbody>
"#;

    /// One row as the live page writes it (anchors, whitespace, empty cells).
    fn row(cells: [&str; 20]) -> String {
        let mut html = String::from("    <tr>\n");
        for (i, cell) in cells.iter().enumerate() {
            let cell = match i {
                4 if !cell.starts_with("DE") => {
                    format!("<a href=\"#{cell}\" ng-click=\"x\">{cell}</a>")
                }
                5 => format!("<a href=\"#LP\" ng-click=\"x\">\n          {cell}\n        </a>"),
                19 if *cell != "n/a" => format!("<a href=\"#refs\">{cell}</a>"),
                _ => (*cell).to_owned(),
            };
            html.push_str(&format!(
                "      <td class=\"text-center\">\n        {cell}\n      </td>\n"
            ));
        }
        html.push_str("    </tr>\n");
        html
    }

    fn elem_page(rows: &[[&str; 20]]) -> String {
        let mut html = String::from("<html><body><h1>Planetary Satellite Mean Elements</h1>\n");
        html.push_str(HEAD);
        for cells in rows {
            html.push_str(&row(*cells));
        }
        html.push_str(
            "  </tbody>\n</table>\n<table id=\"other\"><tr><td>9</td></tr></table></body></html>",
        );
        html
    }

    #[rustfmt::skip]
    const MOON: [&str; 20] = [
        "0", "Earth", "Moon", "301", "DE405/LE405", "ecliptic", "2000-01-01.5", "384400.",
        "0.0554", "318.15", "135.27", "5.16", "125.08", "27.322", "5.997", "18.600", "", "", "",
        "n/a",
    ];
    #[rustfmt::skip]
    const PHOBOS: [&str; 20] = [
        "1", "Mars", "Phobos", "401", "MAR099", "Laplace", "2000-01-01.5", "9375.", "0.015",
        "216.3", "189.7", "1.1", "169.2", "0.3187", "1.1", "2.3", "317.7", "52.9", "0.0", "7",
    ];
    #[rustfmt::skip]
    const PUCK_OLD: [&str; 20] = [
        "415", "Uranus", "Puck", "715", "URA182", "equatorial", "2000-01-01.5", "86004.", "0.000",
        "0.0", "50.1", "0.3", "216.1", "0.761833", "2.226", "4.454", "", "", "", "45",
    ];
    #[rustfmt::skip]
    const PUCK_NEW: [&str; 20] = [
        "425", "Uranus", "Puck", "715", "URA184", "Laplace", "2025-01-01.0", "86007.", "0.009",
        "337.0", "264.1", "1.1", "111.2", "0.7618", "1.9", "3.9", "77.3", "15.2", "180.0", "45",
    ];
    #[rustfmt::skip]
    const ARIEL: [&str; 20] = [
        "410", "Uranus", "Ariel", "701", "URA182", "equatorial", "2000-01-01.5", "190929.",
        "0.001", "9.6", "193.5", "0.0", "0.0", "2.520379", "28.901", "0.000", "", "", "", "45",
    ];
    #[rustfmt::skip]
    const TRITON: [&str; 20] = [
        "440", "Neptune", "Triton", "801", "NEP098", "equatorial", "2000-01-01.5", "354800.",
        "0.000", "0.0", "63.0", "157.3", "177.7", "5.8792", "675.2", "299.4", "", "", "", "3",
    ];
    #[rustfmt::skip]
    const KERBEROS: [&str; 20] = [
        "459", "Pluto", "Kerberos", "904", "PLU060", "equatorial", "2000-01-01.5", "58300.",
        "0.010", "32.1", "276.1", "0.4", "314.3", "32.17", "-", "9", "", "", "", "6",
    ];
    #[rustfmt::skip]
    const HYDRA: [&str; 20] = [
        "460", "Pluto", "Hydra", "903", "PLU060", "equatorial", "2000-01-01.5", "65200.", "0.009",
        "139.3", "335.0", "0.3", "114.3", "38.20", "-", "14", "", "", "", "6",
    ];

    fn records(rows: &[[&str; 20]]) -> anyhow::Result<Vec<SatRecord>> {
        elem_records(&html_table(&elem_page(rows), "sat_elem")?)
    }

    #[test]
    fn reads_html_tables() {
        let page = elem_page(&[MOON, PHOBOS]);
        let table = html_table(&page, "sat_elem").expect("table");
        assert_eq!(table.header.len(), 20);
        assert_eq!(table.header[9], "ω (deg)");
        assert_eq!(table.header[14], "Papsis (yr)");
        assert_eq!(table.rows.len(), 2);
        assert_eq!(table.rows[1][2], "Phobos");
        assert_eq!(table.rows[1][4], "MAR099");
        assert_eq!(table.rows[0][5], "ecliptic");
        assert_eq!(table.rows[0][16], "");
        assert!(html_table(&page, "sat_phys_par").is_err(), "no such table");
        // Unclosed cells, as on /sats/phys_par/.
        let loose = "<table id=\"t\"><thead><tr><th>A<th>B</tr></thead><tbody><tr><td>1 <td> x &amp; y\n<td><a href=\"#n\">*</a></td></tr></tbody></table>";
        let loose = html_table(loose, "t").expect("loose");
        assert_eq!(loose.header, ["A", "B"]);
        assert_eq!(loose.rows, [["1", "x & y", "*"]]);
    }

    #[test]
    fn parses_element_rows() {
        let sats = records(&[MOON, PHOBOS, ARIEL, TRITON, HYDRA]).expect("records");
        let moon = &sats[0];
        assert_eq!(
            (moon.naif, moon.name.as_str(), moon.planet_naif),
            (301, "Moon", 399)
        );
        assert_eq!(
            (moon.a_km, moon.e, moon.argp_deg, moon.m_deg),
            (384_400.0, 0.0554, 318.15, 135.27)
        );
        assert_eq!((moon.i_deg, moon.node_deg), (5.16, 125.08));
        assert_eq!(moon.n_deg_per_day, 360.0 / 27.322);
        assert_eq!((moon.p_apsides_yr, moon.p_node_yr), (5.997, 18.6));
        assert_eq!(moon.epoch_jd_tdb, 2_451_545.0);
        // J2000 ecliptic pole: Dec = 90° − 84381.448″.
        assert_eq!(moon.laplace_ra_deg, 270.0);
        assert!(
            (moon.laplace_dec_deg - 66.560_708_888_9).abs() < 1e-9,
            "{}",
            moon.laplace_dec_deg
        );
        assert_eq!((moon.radius_km, moon.gm_km3_s2), (None, None));
        let phobos = &sats[1];
        assert_eq!(
            (phobos.laplace_ra_deg, phobos.laplace_dec_deg),
            (317.7, 52.9)
        );
        assert_eq!(phobos.planet_naif, 499);
        let ariel = &sats[2];
        // Uranus's prograde pole: −(IAU north 257.311°, −15.175°).
        assert!(
            (ariel.laplace_ra_deg - 77.311).abs() < 1e-9
                && (ariel.laplace_dec_deg - 15.175).abs() < 1e-9
        );
        assert_eq!(ariel.p_node_yr, f64::INFINITY, "0.000 means none tabulated");
        let triton = &sats[3];
        // Neptune at J2000: 299.36 + 0.70 sin N, 43.46 − 0.51 cos N, N = 357.85°.
        let n = 357.85_f64.to_radians();
        assert!((triton.laplace_ra_deg - (299.36 + 0.70 * n.sin())).abs() < 1e-9);
        assert!((triton.laplace_dec_deg - (43.46 - 0.51 * n.cos())).abs() < 1e-9);
        assert_eq!(triton.i_deg, 157.3);
        let hydra = &sats[4];
        assert_eq!((hydra.p_apsides_yr, hydra.p_node_yr), (f64::INFINITY, 14.0));
        assert_eq!(
            (
                hydra.planet_naif,
                hydra.laplace_ra_deg,
                hydra.laplace_dec_deg
            ),
            (999, 132.993, -6.163)
        );
    }

    #[test]
    fn keeps_the_newer_row_of_a_moon_listed_twice() {
        for rows in [[PUCK_OLD, PUCK_NEW], [PUCK_NEW, PUCK_OLD]] {
            let sats = records(&rows).expect("records");
            assert_eq!(sats.len(), 1);
            assert_eq!(
                (sats[0].a_km, sats[0].epoch_jd_tdb),
                (86_007.0, 2_460_676.5)
            );
            assert_eq!(
                (sats[0].laplace_ra_deg, sats[0].laplace_dec_deg),
                (77.3, 15.2)
            );
        }
        let mut same_epoch = PUCK_NEW;
        same_epoch[7] = "86010.";
        assert!(
            records(&[PUCK_NEW, same_epoch]).is_err(),
            "two rows, one epoch"
        );
    }

    #[test]
    fn rejects_bad_element_rows() {
        let bad = |index: usize, value: &'static str| {
            let mut cells = PHOBOS;
            cells[index] = value;
            records(&[cells])
        };
        assert!(bad(1, "Vulcan").is_err(), "unknown planet");
        assert!(bad(3, "4x1").is_err(), "code");
        assert!(bad(5, "galactic").is_err(), "unknown frame");
        assert!(bad(6, "2000-13-01.5").is_err(), "epoch");
        assert!(bad(7, "far").is_err(), "a");
        assert!(bad(13, "0").is_err(), "P = 0");
        assert!(bad(14, "-3").is_err(), "negative precession period");
        assert!(bad(16, "").is_err(), "Laplace row without a pole");
        assert!(bad(8, "1.2").is_err(), "e >= 1");
        assert!(records(&[PHOBOS]).is_ok(), "the unedited row parses");
        let renamed = elem_page(&[PHOBOS]).replacen("node<br>(deg)", "Omega<br>(deg)", 1);
        assert!(elem_records(&html_table(&renamed, "sat_elem").expect("table")).is_err());
        let short = elem_page(&[PHOBOS]).replacen(
            "<td class=\"text-center\">\n        0.0\n      </td>\n",
            "",
            1,
        );
        assert!(records_from(&short).is_err(), "a row with 19 cells");
    }

    fn records_from(page: &str) -> anyhow::Result<Vec<SatRecord>> {
        elem_records(&html_table(page, "sat_elem")?)
    }

    #[test]
    fn calendar_dates_become_julian_dates() {
        assert_eq!(calendar_jd("2000-01-01.5").expect("J2000"), 2_451_545.0);
        assert_eq!(calendar_jd("2020-01-01.0").expect("2020"), 2_458_849.5);
        assert_eq!(calendar_jd("2025-01-01.0").expect("2025"), 2_460_676.5);
        // Meeus, Astronomical Algorithms (1998), example 7.a: 1957 October 4.81 = JD 2436116.31.
        assert!((calendar_jd("1957-10-04.81").expect("Sputnik") - 2_436_116.31).abs() < 1e-9);
        assert!((calendar_jd("2009-02-26.03").expect("S2009") - 2_454_888.53).abs() < 1e-9);
        for bad in [
            "2000-01-01",
            "2000-1-01.5",
            "2000-02-30.0",
            "2000-01-01.5x",
            "x",
        ] {
            assert!(calendar_jd(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn reference_poles_by_frame() {
        let j2000 = 2_451_545.0;
        assert_eq!(
            reference_pole("Laplace", 599, "268.1", "64.5", j2000).expect("L"),
            (268.1, 64.5)
        );
        assert!(reference_pole("Laplace", 599, "", "", j2000).is_err());
        let (ra, dec) = reference_pole("ecliptic", 599, "", "", j2000).expect("ecliptic");
        assert_eq!(ra, 270.0);
        assert!((dec - (90.0 - 84_381.448 / 3600.0)).abs() < 1e-12);
        // Saturn: 40.589 − 0.036 T, 83.537 − 0.004 T, T in Julian centuries from J2000.
        let (ra, dec) =
            reference_pole("equatorial", 699, "", "", j2000 + 36_525.0).expect("Saturn");
        assert!(
            (ra - 40.553).abs() < 1e-9 && (dec - 83.533).abs() < 1e-9,
            "{ra} {dec}"
        );
        assert_eq!(
            reference_pole("equatorial", 999, "", "", j2000).expect("Pluto"),
            (132.993, -6.163)
        );
        assert!(
            reference_pole("equatorial", 599, "", "", j2000).is_err(),
            "no equatorial rows for Jupiter"
        );
        assert!(reference_pole("galactic", 599, "", "", j2000).is_err());
    }

    #[test]
    fn precession_periods() {
        assert_eq!(period_yr("18.600").expect("value"), 18.6);
        assert_eq!(period_yr("-").expect("dash"), f64::INFINITY);
        assert_eq!(period_yr("0.000").expect("zero"), f64::INFINITY);
        assert!(period_yr("-3").is_err());
        assert!(period_yr("").is_err());
    }

    /// The physical-parameters page: two header rows and unclosed cells.
    const PHYS: &str = r##"<table id="sat_phys_par" class="sat-phys-par table">
  <thead>
    <tr>
      <th rowspan="2" class="text-left">&nbsp;<br/>Planet</th>
      <th rowspan="2" class="text-left">&nbsp;<br/>Satellite</th>
      <th rowspan="2" class="text-center">&nbsp;<br/>Code</th>
      <th class="text-center" colspan="3">GM&nbsp; (km<sup>3</sup>/s<sup>2</sup>)</th>
      <th class="text-center" colspan="3">Mean Radius&nbsp; (km)</th>
      <th class="text-center" colspan="3">Mean Density&nbsp; (g/cm<sup>3</sup>)</th>
    </tr>
    <tr>
    <th class="text-center">value</th>
    <th class="text-center">sigma</th>
    <th class="text-center">ref</th>
    <th class="text-center">value</th>
    <th class="text-center">sigma</th>
    <th class="text-center">ref</th>
    <th class="text-center">value</th>
    <th class="text-center">sigma</th>
    <th class="text-center">ref</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="text-left">Earth</td>
      <td class="text-left">Moon</td>
      <td class="text-center">301</td>
      <td class="text-center">
            4902.800
      <td class="text-center">
            0.001
      <td class="text-center">
                DE440
      </td>
      <td class="text-center">
            1737.4
      <td class="text-center">
            0.1
      <td class="text-center">
          <a href="#refs">1</a>
      </td>
      <td class="text-center">
            3.344
      <td class="text-center">
            0.001
      <td class="text-center">
          <a href="#notes">*</a>
      </td>
    </tr>
    <tr>
      <td class="text-left">Mars</td>
      <td class="text-left">Phobos</td>
      <td class="text-center">401</td>
      <td class="text-center">&lt;0.0002<td class="text-center">n/a<td class="text-center">MAR097</td>
      <td class="text-center">11.08<td class="text-center">0.04<td class="text-center">1</td>
      <td class="text-center">n/a<td class="text-center">n/a<td class="text-center">*</td>
    </tr>
    <tr>
      <td class="text-left">Pluto</td>
      <td class="text-left">Kerberos</td>
      <td class="text-center">904</td>
      <td class="text-center">
            <0.0002
      <td class="text-center">
            n/a
      <td class="text-center">
          <a href="#PLU060" ng-click="$event.preventDefault();open('#PLU060')">PLU060</a>
      </td>
      <td class="text-center">6.0<td class="text-center">1.0<td class="text-center">3</td>
      <td class="text-center">n/a<td class="text-center">n/a<td class="text-center">*</td>
    </tr>
  </tbody>
</table>"##;

    #[test]
    fn parses_physical_rows_and_merges_them() {
        let phys = phys_rows(&html_table(PHYS, "sat_phys_par").expect("table")).expect("rows");
        assert_eq!(
            phys,
            [
                Phys {
                    naif: 301,
                    gm_km3_s2: Some(4902.8),
                    radius_km: 1737.4
                },
                Phys {
                    naif: 401,
                    gm_km3_s2: None,
                    radius_km: 11.08
                },
                // The live page writes this upper limit with a bare '<' (not "&lt;").
                Phys {
                    naif: 904,
                    gm_km3_s2: None,
                    radius_km: 6.0
                },
            ]
        );
        let mut sats = records(&[MOON, PHOBOS, KERBEROS]).expect("records");
        merge(&mut sats, &phys).expect("merge");
        assert_eq!(
            (sats[0].radius_km, sats[0].gm_km3_s2),
            (Some(1737.4), Some(4902.8))
        );
        assert_eq!((sats[1].radius_km, sats[1].gm_km3_s2), (Some(11.08), None));
        let mut only_moon = records(&[MOON]).expect("records");
        assert!(
            merge(&mut only_moon, &phys).is_err(),
            "Phobos has no elements row"
        );
        let renamed = PHYS.replacen("Mean Radius", "Radius", 1);
        assert!(phys_rows(&html_table(&renamed, "sat_phys_par").expect("table")).is_err());
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&["--out", "o.ron", "--cache", "c"])).expect("valid");
        assert_eq!(
            (options.out, options.cache),
            (PathBuf::from("o.ron"), PathBuf::from("c"))
        );
        let options = parse_options(&args(&["--out", "o.ron"])).expect("cache optional");
        assert!(
            options.cache.ends_with("jpl-sats"),
            "{}",
            options.cache.display()
        );
        assert!(parse_options(&args(&[])).is_err());
        assert!(parse_options(&args(&["--out"])).is_err());
        assert!(parse_options(&args(&["--in", "x"])).is_err());
    }
}
