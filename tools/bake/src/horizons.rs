//! `uni-bake horizons` (M1 Task A9 Step 1): JPL Horizons spot values → `data/golden/horizons_spots.ron`
//! (geometric state vectors) and `data/golden/sky_spots.ron` (observer tables), the goldens that
//! the uni-physics ephemeris and photometry tests compare against (review I9).
//!
//! `uni-bake horizons --plan m1 --out data/golden/horizons_spots.ron --sky data/golden/sky_spots.ron [--cache <dir>]`
//!
//! `--plan` names a built-in request list; `m1`, A9 Step 1's, is the only one. Each request is one
//! call of the Horizons API (<https://ssd-api.jpl.nasa.gov/doc/horizons.html>), made with
//! `curl -L --fail` through `std::process::Command`. The raw JSON answers are kept in `--cache`
//! (default `data/cache/horizons/`, git-ignored; delete it after the bake). From each answer the
//! CSV table between `$$SOE` and `$$EOE` is parsed. Each answer's header must name the requested
//! target and centre (vectors also: ICRF, km and km/s, geometric states), and each requested epoch
//! must come back exactly once, or the bake stops before writing anything.
//!
//! State vectors: `EPHEM_TYPE=VECTORS`, `REF_SYSTEM=ICRF`, `REF_PLANE=FRAME`, `VEC_CORR=NONE`
//! (geometric), `OUT_UNITS=KM-S`, `TIME_TYPE=TDB`, and the epochs as a `TLIST` of TDB Julian Dates
//! (`TLIST_TYPE=JD`), never calendar dates, because Horizons reads dates before 1582-10-15 as
//! Julian (review I2). Observer tables: `CENTER=500@399` (the geocentre), `TIME_TYPE=UT`, a `TLIST`
//! of UTC Julian Dates, angles in degrees, `QUANTITIES` from 1 (astrometric RA/Dec, ICRF),
//! 9 (APmag), 19 (r), 20 (Δ) and 24 (phase angle S-T-O).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, bail, ensure};
use uni_data::{EphemSpot, SkySpot};

const USAGE: &str = "usage: uni-bake horizons --plan m1 --out <horizons_spots.ron> --sky <sky_spots.ron> [--cache <dir>]";

/// The Horizons API endpoint (<https://ssd-api.jpl.nasa.gov/doc/horizons.html>).
const API_URL: &str = "https://ssd.jpl.nasa.gov/api/horizons.api";

// A9 Step 1's epochs as Julian Dates: TDB for the state vectors, UTC for the observer tables.
const JD_1000: f64 = 2_086_307.5; // AD 1000-01-01 0h (Julian calendar)
const JD_1800: f64 = 2_378_496.5; // 1800-01-01 0h
const JD_1900: f64 = 2_415_020.5; // 1900-01-01 0h
const JD_1950: f64 = 2_433_282.5; // 1950-01-01 0h
const JD_2000: f64 = 2_451_545.0; // J2000.0 = 2000-01-01 12h
/// 2026-11-20 00:00 UTC as TDB: UTC + 37 s (leap seconds) + 32.184 s (TT − TAI) = +69.184 s
/// = +0.000 800 7 d, which the plan rounds to 0.0008 d (TDB − TT < 2 ms is below the rounding).
const JD_2026_TDB: f64 = 2_461_364.500_8;
const JD_2026_UTC: f64 = 2_461_364.5; // 2026-11-20 00:00 UTC
const JD_2050: f64 = 2_469_807.5; // 2050-01-01 0h
const JD_2090: f64 = 2_484_417.5; // 2090-01-01 0h
const JD_2500: f64 = 2_634_166.5; // AD 2500-01-01 0h
const JD_HALLEY: f64 = 2_446_502.5; // 1986-03-13 0h
const JD_NEOWISE_UTC: f64 = 2_459_045.625; // 2020-07-15 03:00 UTC

/// NAIF ids written as the target of the small-body spots: 1P/Halley's comet code (also its SBDB
/// SPK-ID) and (1) Ceres's original-schema asteroid id, 2 000 000 + number, which NAIF's name
/// table uses. SBDB's SPK-ID for Ceres is the extended-schema 20000001 (NAIF "NAIF Integer ID
/// codes" Required Reading, §Asteroids); `uni_data::smallbody::asteroid_spkid_from_naif` maps one
/// to the other (A9 review r0 I1).
const HALLEY_NAIF: i32 = 1_000_036;
const CERES_NAIF: i32 = 2_000_001;

/// The moons with Horizons spots, each with the planet whose body centre is the centre.
const MOONS: [(i32, i32); 19] = [
    (301, 399),
    (401, 499),
    (402, 499),
    (501, 599),
    (502, 599),
    (503, 599),
    (504, 599),
    (601, 699),
    (602, 699),
    (603, 699),
    (604, 699),
    (605, 699),
    (606, 699),
    (607, 699),
    (608, 699),
    (703, 799),
    (704, 799),
    (801, 899),
    (901, 999),
];

/// Observer-table quantities for the planets: astrometric RA/Dec, APmag, r, Δ, phase angle.
const PLANET_QUANTITIES: &str = "1,9,19,20,24";

/// Largest difference between an answer row's JD and the asked JD [d]; Horizons prints JDs to 1e-9 d.
const JD_MATCH_D: f64 = 1e-6;

/// One state-vector request: one target and centre, several TDB epochs.
#[derive(Debug)]
struct VectorRequest {
    /// Horizons `COMMAND`, unquoted.
    command: String,
    /// The id written as each spot's target.
    target: i32,
    /// NAIF id of the centre; the query asks for its body centre (`500@<id>`).
    center: i32,
    /// Accepted endings of the answer's "Target body name" (without its source).
    target_names: Vec<String>,
    /// TDB Julian Dates, ascending.
    jds: Vec<f64>,
}

/// One observer-table request: one target at one UTC epoch, seen from a body's centre.
#[derive(Debug)]
struct SkyRequest {
    /// Horizons `COMMAND`, unquoted.
    command: String,
    /// The text written as the spot's target.
    target: String,
    /// NAIF id of the observer's body (`CENTER='500@<id>'`).
    observer: i32,
    /// Accepted endings of the answer's "Target body name" (without its source).
    target_names: Vec<String>,
    jd_utc: f64,
    /// Horizons `QUANTITIES`, comma-separated codes.
    quantities: &'static str,
}

/// A built-in request list.
struct Plan {
    vectors: Vec<VectorRequest>,
    sky: Vec<SkyRequest>,
}

/// The useful part of one Horizons JSON answer.
#[derive(Debug)]
struct Answer {
    /// The `result` text (header, `$$SOE` table, footnotes).
    text: String,
    /// `signature.version` of the API.
    api_version: String,
}

/// `uni-bake horizons` options.
struct Options {
    plan: String,
    out: PathBuf,
    sky: PathBuf,
    cache: PathBuf,
}

/// Runs `uni-bake horizons` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    let plan = match options.plan.as_str() {
        "m1" => m1_plan(),
        other => bail!("unknown plan {other:?}: the built-in plan is \"m1\"\n{USAGE}"),
    };
    std::fs::create_dir_all(&options.cache)
        .with_context(|| format!("creating {}", options.cache.display()))?;

    let mut calls = 0usize;
    let mut raw_bytes = 0u64;
    let mut fetched = None;
    let mut api_version = None;
    let mut ask = |label: String, params: Vec<(&'static str, String)>| -> anyhow::Result<Answer> {
        let raw = options.cache.join(format!("{calls:03}-{label}.json"));
        log::info!("horizons: {label}");
        let answer = fetch(&params, &raw).with_context(|| format!("Horizons request {label}"))?;
        calls += 1;
        raw_bytes += std::fs::metadata(&raw).map(|m| m.len()).unwrap_or(0);
        fetched.get_or_insert_with(|| fetch_stamp(&answer.text));
        api_version.get_or_insert_with(|| answer.api_version.clone());
        Ok(answer)
    };

    let mut ephem = Vec::new();
    let mut ephem_sources = Vec::new();
    for request in &plan.vectors {
        let label = format!("vectors-{}@{}", request.target, request.center);
        let answer = ask(label.clone(), vector_params(request))?;
        let (spots, source) = parse_vectors(&answer.text, request)
            .with_context(|| format!("Horizons answer {label}"))?;
        ephem_sources.push(format!(
            "  {:>7}@{:<3} {source}, {} epoch(s)",
            request.target,
            request.center,
            spots.len()
        ));
        ephem.extend(spots);
    }
    let mut sky = Vec::new();
    let mut sky_sources = Vec::new();
    for request in &plan.sky {
        let label = format!(
            "observer-{}@{}",
            request.target.replace(['/', ' '], "_"),
            request.observer
        );
        let answer = ask(label.clone(), sky_params(request))?;
        let (spot, source) =
            parse_sky(&answer.text, request).with_context(|| format!("Horizons answer {label}"))?;
        sky_sources.push(format!(
            "  {:<10} quantities {:<13} {source}",
            request.target, request.quantities
        ));
        sky.push(spot);
    }
    ephem.sort_by(|a, b| {
        (a.center, a.target)
            .cmp(&(b.center, b.target))
            .then(a.jd_tdb.total_cmp(&b.jd_tdb))
    });

    let fetched = fetched
        .flatten()
        .unwrap_or_else(|| "unknown time".to_owned());
    let api_version = api_version.unwrap_or_default();
    let provenance = |manifest_id: &str| {
        [
            "Written by `uni-bake horizons --plan m1` (M1 Task A9 Step 1) from the JPL Horizons API".to_owned(),
            format!("{API_URL} (NASA/JPL Horizons API {api_version}), fetched {fetched} (Horizons' stamp)."),
            format!("Public JPL Solar System Dynamics data; see ATTRIBUTION.md and data/manifest.toml (id \"{manifest_id}\")."),
        ]
    };
    let mut header =
        vec!["JPL Horizons geometric state vectors: test goldens (review I9).".to_owned()];
    header.extend(provenance("horizons"));
    header.extend([
        String::new(),
        "Query: EPHEM_TYPE=VECTORS REF_SYSTEM=ICRF REF_PLANE=FRAME VEC_CORR=NONE (geometric) OUT_UNITS=KM-S".to_owned(),
        "TIME_TYPE=TDB TLIST_TYPE=JD VEC_TABLE=2 CSV_FORMAT=YES; CENTER=500@<center> (its body centre).".to_owned(),
        "target, center: NAIF ids. Small bodies: 1000036 = 1P/Halley with CAP<1987 (also its SBDB SPK-ID); 2000001 = Ceres,".to_owned(),
        "NAIF's original asteroid id 2000000 + number (SBDB and asteroids.bin: extended SPK-ID 20000001).".to_owned(),
        "jd_tdb: the Julian Date (TDB) asked for; pos_km, vel_km_s: target minus centre, ICRF axes.".to_owned(),
        String::new(),
        "Ephemeris source of each answer, as its header names it (target [source] from centre [source]).".to_owned(),
        "Horizons labels barycentres 1 and 2 with the body ids 199 and 299: moonless, they are the bodies.".to_owned(),
    ]);
    header.extend(ephem_sources);
    write_ron(&options.out, &header, &ephem)?;

    let mut header = vec!["JPL Horizons observer tables: test goldens (review I4).".to_owned()];
    header.extend(provenance("horizons-sky"));
    header.extend([
        String::new(),
        "Query: EPHEM_TYPE=OBSERVER CENTER=500@399 (geocentre) REF_SYSTEM=ICRF TIME_TYPE=UT TLIST_TYPE=JD".to_owned(),
        "ANG_FORMAT=DEG CAL_FORMAT=JD EXTRA_PREC=YES CSV_FORMAT=YES; airless, no refraction.".to_owned(),
        "target: NAIF id or SBDB designation; observer: NAIF id; jd_utc: Julian Date (UTC) asked for.".to_owned(),
        "ra_deg, dec_deg: astrometric ICRF (quantity 1); delta_au: Δ (20); apmag: APmag (9); r_au: r (19);".to_owned(),
        "phase_deg: Sun-target-observer angle (24); None where the quantity was not asked for.".to_owned(),
        "Jupiter's spot of the plan's quantities '1,20' is the 599 row below (same target, observer and time).".to_owned(),
        String::new(),
        "Target, quantities and ephemeris source (target [source] from observer [source]):".to_owned(),
    ]);
    header.extend(sky_sources);
    write_ron(&options.sky, &header, &sky)?;

    println!(
        "horizons: {} state vectors -> {}, {} observer rows -> {} ({calls} calls, {raw_bytes} B raw in {})",
        ephem.len(),
        options.out.display(),
        sky.len(),
        options.sky.display(),
        options.cache.display()
    );
    Ok(())
}

/// Parses `--plan`, `--out`, `--sky` (required) and `--cache` (default `<data>/cache/horizons`).
fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut plan, mut out, mut sky, mut cache) = (None, None, None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--plan") => &mut plan,
            Some("--out") => &mut out,
            Some("--sky") => &mut sky,
            Some("--cache") => &mut cache,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(value.clone());
    }
    let (Some(plan), Some(out), Some(sky)) = (plan, out, sky) else {
        bail!("horizons needs --plan, --out and --sky\n{USAGE}");
    };
    Ok(Options {
        plan: plan
            .into_string()
            .map_err(|p| anyhow::anyhow!("--plan {p:?} is not UTF-8"))?,
        out: out.into(),
        sky: sky.into(),
        cache: cache.map_or_else(
            || uni_data::data_dir().join("cache").join("horizons"),
            PathBuf::from,
        ),
    })
}

/// A vector request for a major body or barycentre, named by its NAIF id.
fn major(target: i32, center: i32, jds: &[f64]) -> VectorRequest {
    let mut jds = jds.to_vec();
    jds.sort_by(f64::total_cmp);
    let target_names = if (1..=9).contains(&target) {
        // Horizons labels the barycentres of moonless Mercury and Venus with the body ids,
        // "Mercury Barycenter (199)" (measured 2026-11-22): DE441 has no separate point for them.
        vec![
            format!("Barycenter ({target})"),
            format!("Barycenter ({target}99)"),
        ]
    } else {
        vec![format!("({target})")]
    };
    VectorRequest {
        command: target.to_string(),
        target,
        center,
        target_names,
        jds,
    }
}

/// The geocentric observer request for planet `naif` at 2026-11-20 00:00 UTC.
fn planet_sky(naif: i32) -> SkyRequest {
    SkyRequest {
        command: naif.to_string(),
        target: naif.to_string(),
        observer: 399,
        target_names: vec![format!("({naif})")],
        jd_utc: JD_2026_UTC,
        quantities: PLANET_QUANTITIES,
    }
}

/// A9 Step 1's request list (the brief's plan, one request per target).
fn m1_plan() -> Plan {
    // Standish's elements describe the planetary-system barycentres 1–8 (review I2); AD 1000 and
    // AD 2500 are for the 3000 BC – AD 3000 table (Table 2).
    let standish = [
        JD_1800,
        JD_1900,
        JD_1950,
        JD_2000,
        JD_2026_TDB,
        JD_2050,
        JD_1000,
        JD_2500,
    ];
    let mut vectors: Vec<_> = (1..=8).map(|target| major(target, 10, &standish)).collect();
    vectors.push(major(
        9,
        10,
        &[JD_1900, JD_1950, JD_2000, JD_2026_TDB, JD_2050, JD_2090],
    ));
    vectors.push(major(10, 0, &[JD_1950, JD_2000, JD_2026_TDB]));
    let moon_epochs = [JD_1950, JD_2000, JD_2026_TDB, JD_2050];
    vectors.extend(
        MOONS
            .iter()
            .map(|&(moon, planet)| major(moon, planet, &moon_epochs)),
    );
    vectors.push(VectorRequest {
        command: "DES=1P;CAP<1987;".to_owned(),
        target: HALLEY_NAIF,
        center: 10,
        target_names: vec!["1P/Halley".to_owned()],
        jds: vec![JD_HALLEY],
    });
    vectors.push(VectorRequest {
        command: "DES=2000001;".to_owned(),
        target: CERES_NAIF,
        center: 10,
        target_names: vec!["1 Ceres (A801 AA)".to_owned()],
        jds: vec![JD_2026_TDB],
    });

    // The brief also lists Jupiter at 2026-11-20 00:00 UTC with quantities '1,20'; the 599 row
    // asks for a superset of them at the same time, so it is that spot too.
    let mut sky: Vec<_> = [199, 299, 499, 599, 699, 799, 899].map(planet_sky).into();
    sky.push(SkyRequest {
        command: "DES=C/2020 F3;CAP;".to_owned(),
        target: "C/2020 F3".to_owned(),
        observer: 399,
        target_names: vec!["(C/2020 F3)".to_owned()],
        jd_utc: JD_NEOWISE_UTC,
        quantities: "1,20",
    });
    Plan { vectors, sky }
}

fn quoted(value: &str) -> String {
    format!("'{value}'")
}

/// A `TLIST` of Julian Dates: each quoted, space-separated, printed exactly (`{:?}`).
fn tlist(jds: &[f64]) -> String {
    jds.iter()
        .map(|jd| format!("'{jd:?}'"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Query parameters of a state-vector request (A9 Step 1's settings).
fn vector_params(request: &VectorRequest) -> Vec<(&'static str, String)> {
    vec![
        ("format", "json".to_owned()),
        ("COMMAND", quoted(&request.command)),
        ("CENTER", quoted(&format!("500@{}", request.center))),
        ("EPHEM_TYPE", quoted("VECTORS")),
        ("REF_SYSTEM", quoted("ICRF")),
        ("REF_PLANE", quoted("FRAME")),
        ("VEC_CORR", quoted("NONE")),
        ("OUT_UNITS", quoted("KM-S")),
        ("TIME_TYPE", quoted("TDB")),
        ("TLIST_TYPE", quoted("JD")),
        ("TLIST", tlist(&request.jds)),
        ("VEC_TABLE", quoted("2")),
        ("CSV_FORMAT", quoted("YES")),
        ("OBJ_DATA", quoted("NO")),
        ("MAKE_EPHEM", quoted("YES")),
    ]
}

/// Query parameters of an observer-table request.
fn sky_params(request: &SkyRequest) -> Vec<(&'static str, String)> {
    vec![
        ("format", "json".to_owned()),
        ("COMMAND", quoted(&request.command)),
        ("CENTER", quoted(&format!("500@{}", request.observer))),
        ("EPHEM_TYPE", quoted("OBSERVER")),
        ("REF_SYSTEM", quoted("ICRF")),
        ("TIME_TYPE", quoted("UT")),
        ("TLIST_TYPE", quoted("JD")),
        ("TLIST", tlist(&[request.jd_utc])),
        ("QUANTITIES", quoted(request.quantities)),
        ("ANG_FORMAT", quoted("DEG")),
        ("CAL_FORMAT", quoted("JD")),
        ("EXTRA_PREC", quoted("YES")),
        ("CSV_FORMAT", quoted("YES")),
        ("OBJ_DATA", quoted("NO")),
        ("MAKE_EPHEM", quoted("YES")),
    ]
}

/// Calls the API with `curl` (GET, parameters URL-encoded by curl), keeps the raw answer at
/// `raw_path` and returns its parsed JSON.
fn fetch(params: &[(&'static str, String)], raw_path: &Path) -> anyhow::Result<Answer> {
    let mut curl = Command::new("curl");
    curl.args([
        "-sS",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--max-time",
        "120",
    ])
    .arg("--get")
    .arg("-o")
    .arg(raw_path);
    for (key, value) in params {
        curl.arg("--data-urlencode").arg(format!("{key}={value}"));
    }
    curl.arg(API_URL);
    let output = curl.output().context("running curl")?;
    ensure!(
        output.status.success(),
        "curl failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let body = std::fs::read_to_string(raw_path)
        .with_context(|| format!("reading {}", raw_path.display()))?;
    horizons_answer(&body)
}

/// Reads a Horizons JSON answer. Horizons reports input errors with HTTP 200 and an `error` field.
fn horizons_answer(body: &str) -> anyhow::Result<Answer> {
    let json: serde_json::Value =
        serde_json::from_str(body).context("the Horizons answer is not JSON")?;
    if let Some(error) = json.get("error") {
        let error = error
            .as_str()
            .map_or_else(|| error.to_string(), str::to_owned);
        bail!("Horizons error: {}", error.trim());
    }
    let text = json
        .get("result")
        .and_then(serde_json::Value::as_str)
        .context("the Horizons answer has no \"result\"")?;
    let api_version = json
        .pointer("/signature/version")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("unknown");
    Ok(Answer {
        text: text.to_owned(),
        api_version: api_version.to_owned(),
    })
}

/// The time stamp on a Horizons answer, from its line
/// "Ephemeris / API_USER Sun Nov 22 08:34:47 2026 Pasadena, USA      / Horizons".
fn fetch_stamp(text: &str) -> Option<String> {
    let line = text.lines().find(|l| l.contains("API_USER"))?;
    let (_, rest) = line.split_once("API_USER")?;
    Some(rest.split('/').next()?.trim().to_owned())
}

/// The value of the header line `<key> : <value>` (any spaces before the colon).
fn header_value<'a>(text: &'a str, key: &str) -> anyhow::Result<&'a str> {
    text.lines()
        .find_map(|line| line.strip_prefix(key)?.trim_start().strip_prefix(':'))
        .map(str::trim)
        .with_context(|| format!("no \"{key}\" line in the Horizons answer"))
}

/// Splits "Saturn (699)   {source: sat441l}" into the name and the source.
fn name_and_source(value: &str) -> (&str, &str) {
    match value.split_once("{source:") {
        Some((name, source)) => (name.trim(), source.trim().trim_end_matches('}').trim()),
        None => (value.trim(), "unknown"),
    }
}

/// Checks the answer's target and centre lines and returns
/// "<target> [<source>] from <centre> [<source>]".
fn check_bodies(text: &str, target_names: &[String], center: i32) -> anyhow::Result<String> {
    let (target, target_source) = name_and_source(header_value(text, "Target body name")?);
    ensure!(
        target_names
            .iter()
            .any(|name| target.ends_with(name.as_str())),
        "the answer is for {target:?}, not {target_names:?}"
    );
    let (centre, centre_source) = name_and_source(header_value(text, "Center body name")?);
    ensure!(
        centre.ends_with(&format!("({center})")),
        "the answer is centred on {centre:?}, not {center}"
    );
    let site = header_value(text, "Center-site name")?;
    ensure!(
        matches!(site, "BODY CENTER" | "GEOCENTRIC"),
        "the answer's centre site is {site:?}, not the body centre"
    );
    Ok(format!(
        "{target} [{target_source}] from {centre} [{centre_source}]"
    ))
}

/// The CSV table between `$$SOE` and `$$EOE`: its column names (the last header line before
/// `$$SOE`) and its rows, each with as many fields as there are columns.
fn soe_table(text: &str) -> anyhow::Result<(Vec<String>, Vec<Vec<String>>)> {
    let lines: Vec<&str> = text.lines().collect();
    let soe = lines
        .iter()
        .position(|l| l.trim() == "$$SOE")
        .context("no $$SOE in the Horizons answer")?;
    let eoe = lines
        .iter()
        .position(|l| l.trim() == "$$EOE")
        .context("no $$EOE in the Horizons answer")?;
    ensure!(soe < eoe, "$$EOE comes before $$SOE");
    let header = lines[..soe]
        .iter()
        .rev()
        .map(|l| l.trim())
        .find(|l| !l.is_empty() && !l.chars().all(|c| c == '*'))
        .context("no column header before $$SOE")?;
    let columns = csv_fields(header);
    let rows: Vec<Vec<String>> = lines[soe + 1..eoe]
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| csv_fields(l))
        .collect();
    for row in &rows {
        ensure!(
            row.len() == columns.len(),
            "a row has {} fields for {} columns",
            row.len(),
            columns.len()
        );
    }
    Ok((columns, rows))
}

/// The fields of one CSV line, trimmed; Horizons ends every line with a comma.
fn csv_fields(line: &str) -> Vec<String> {
    let line = line.trim();
    let line = line.strip_suffix(',').unwrap_or(line);
    line.split(',')
        .map(|field| field.trim().to_owned())
        .collect()
}

fn number(field: &str) -> anyhow::Result<f64> {
    let value: f64 = field
        .parse()
        .with_context(|| format!("not a number: {field:?}"))?;
    ensure!(value.is_finite(), "not a finite number: {field:?}");
    Ok(value)
}

/// Parses a state-vector answer into one spot per asked epoch (in row order), plus its source line.
fn parse_vectors(text: &str, request: &VectorRequest) -> anyhow::Result<(Vec<EphemSpot>, String)> {
    let source = check_bodies(text, &request.target_names, request.center)?;
    for (key, want) in [("Output units", "KM-S"), ("Reference frame", "ICRF")] {
        let value = header_value(text, key)?;
        ensure!(value == want, "{key} is {value:?}, not {want}");
    }
    let output_type = header_value(text, "Output type")?;
    ensure!(
        output_type.starts_with("GEOMETRIC"),
        "Output type is {output_type:?}, not geometric"
    );
    let (columns, rows) = soe_table(text)?;
    let index = |name: &str| {
        columns
            .iter()
            .position(|c| c == name)
            .with_context(|| format!("no {name} column"))
    };
    let fields = [
        index("JDTDB")?,
        index("X")?,
        index("Y")?,
        index("Z")?,
        index("VX")?,
        index("VY")?,
        index("VZ")?,
    ];
    ensure!(
        rows.len() == request.jds.len(),
        "{} rows for {} asked epochs",
        rows.len(),
        request.jds.len()
    );
    let mut used = vec![false; request.jds.len()];
    let mut spots = Vec::with_capacity(rows.len());
    for row in &rows {
        let v = fields.map(|i| number(&row[i]));
        let [jd, x, y, z, vx, vy, vz] = v;
        let jd = jd?;
        let asked = request
            .jds
            .iter()
            .position(|&want| (jd - want).abs() < JD_MATCH_D)
            .with_context(|| format!("row at JD {jd} was not asked for"))?;
        ensure!(!used[asked], "two rows at JD {jd}");
        used[asked] = true;
        spots.push(EphemSpot {
            target: request.target,
            center: request.center,
            jd_tdb: request.jds[asked],
            pos_km: [x?, y?, z?],
            vel_km_s: [vx?, vy?, vz?],
        });
    }
    Ok((spots, source))
}

/// Parses an observer answer of one row into its spot, plus its source line.
fn parse_sky(text: &str, request: &SkyRequest) -> anyhow::Result<(SkySpot, String)> {
    let source = check_bodies(text, &request.target_names, request.observer)?;
    let (columns, rows) = soe_table(text)?;
    let [row] = rows.as_slice() else {
        bail!("{} rows where one epoch was asked", rows.len());
    };
    let value = |what: &str, matches: &dyn Fn(&str) -> bool| -> anyhow::Result<f64> {
        let index = columns
            .iter()
            .position(|c| matches(c))
            .with_context(|| format!("no {what} column"))?;
        number(&row[index]).with_context(|| format!("column {what}"))
    };
    let asked = |code: &str| request.quantities.split(',').any(|q| q == code);
    let optional = |code: &str, name: &str| -> anyhow::Result<Option<f64>> {
        asked(code).then(|| value(name, &|c| c == name)).transpose()
    };
    let jd = value("Date (JDUT)", &|c| {
        c.starts_with("Date") && c.ends_with("JDUT")
    })?;
    ensure!(
        (jd - request.jd_utc).abs() < JD_MATCH_D,
        "row at JD {jd} UT, asked {:?}",
        request.jd_utc
    );
    let spot = SkySpot {
        target: request.target.clone(),
        observer: request.observer,
        jd_utc: request.jd_utc,
        ra_deg: value("R.A. (ICRF)", &|c| {
            c.starts_with("R.A.") && c.ends_with("(ICRF)")
        })?,
        dec_deg: value("DEC (ICRF)", &|c| {
            c.starts_with("DEC") && c.ends_with("(ICRF)")
        })?,
        delta_au: value("delta", &|c| c == "delta")?,
        apmag: optional("9", "APmag")?,
        r_au: optional("19", "r")?,
        phase_deg: optional("24", "S-T-O")?,
    };
    Ok((spot, source))
}

/// Writes `items` as a RON sequence, one compact struct per line, after `header` as `//` comments.
fn write_ron<T: serde::Serialize>(
    path: &Path,
    header: &[String],
    items: &[T],
) -> anyhow::Result<()> {
    let config = ron::ser::PrettyConfig::new().compact_structs(true);
    let body = ron::ser::to_string_pretty(items, config).context("serialising RON")?;
    let mut text = String::new();
    for line in header {
        text.push_str(if line.is_empty() { "//" } else { "// " });
        text.push_str(line);
        text.push('\n');
    }
    text.push_str(&body);
    text.push('\n');
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A Horizons vector answer, trimmed from a real one (Jupiter barycentre @ Sun, 2026-11-22).
    const VECTOR_ANSWER: &str = "\
*******************************************************************************
Ephemeris / API_USER Sun Nov 22 08:34:47 2026 Pasadena, USA      / Horizons
*******************************************************************************
Target body name: Jupiter Barycenter (5)          {source: DE441}
Center body name: Sun (10)                        {source: DE441}
Center-site name: BODY CENTER
*******************************************************************************
Output units    : KM-S
Calendar mode   : Mixed Julian/Gregorian
Output type     : GEOMETRIC cartesian states
Output format   : 2 (position and velocity)
Reference frame : ICRF
*******************************************************************************
            JDTDB,            Calendar Date (TDB),                      X,                      Y,                      Z,                     VX,                     VY,                     VZ,
**************************************************************************************************
$$SOE
2086307.500000000, A.D. 1000-Jan-01 00:00:00.0000,  1.509304939266923E+08, -6.993177567170761E+08, -3.040250409851395E+08,  1.269645296174122E+01,  2.994897101507841E+00,  9.695506401570733E-01,
2451545.000000000, A.D. 2000-Jan-01 12:00:00.0000,  5.985675835979289E+08,  4.093863501165546E+08,  1.608943471230070E+08, -7.909837688567292E+00,  1.018349781536609E+01,  4.557719053468124E+00,
2461364.500800000, A.D. 2026-Nov-20 00:01:09.1200, -5.640475942392980E+08,  5.115198534827855E+08,  2.329813975741199E+08, -9.381734722133515E+00, -8.037685946387459E+00, -3.216792381573927E+00,
$$EOE
**************************************************************************************************
";

    /// A Horizons observer answer, trimmed from a real one (Saturn from the geocentre).
    const SATURN_ANSWER: &str = "\
*******************************************************************************
Ephemeris / API_USER Sun Nov 22 08:35:20 2026 Pasadena, USA      / Horizons
*******************************************************************************
Target body name: Saturn (699)                    {source: sat441l}
Center body name: Earth (399)                     {source: DE441}
Center-site name: GEOCENTRIC
*******************************************************************************
Date_________JDUT, , , R.A.___(ICRF), DEC____(ICRF),    APmag,  S-brt,                r,       rdot,             delta,     deldot,     S-T-O,
**********************************************************************************************************************************************
$$SOE
2461364.500000000, , ,   8.331486146,   0.734877993,    0.555,  6.683,   9.419475012161, -0.5235721,  8.74653339379041, 21.7768700,    4.5638,
$$EOE
";

    /// A Horizons observer answer with quantities 1 and 20 only (C/2020 F3 from the geocentre).
    const NEOWISE_ANSWER: &str = "\
Target body name: NEOWISE (C/2020 F3)             {source: JPL#31}
Center body name: Earth (399)                     {source: DE441}
Center-site name: GEOCENTRIC
***********************************************************************************
Date_________JDUT, , , R.A.___(ICRF), DEC____(ICRF),             delta,     deldot,
***********************************************************************************
$$SOE
2459045.625000000, , , 115.934519652,  46.763967450,  0.78434409858666,-39.4983616,
$$EOE
";

    fn jupiter(jds: &[f64]) -> VectorRequest {
        major(5, 10, jds)
    }

    fn saturn_sky() -> SkyRequest {
        planet_sky(699)
    }

    fn param<'a>(params: &'a [(&'static str, String)], key: &str) -> &'a str {
        let mut values = params.iter().filter(|(k, _)| *k == key);
        let (_, value) = values.next().unwrap_or_else(|| panic!("no {key}"));
        assert!(values.next().is_none(), "{key} given twice");
        value
    }

    /// A per-process scratch path (no tempfile crate).
    fn scratch(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-bake-horizons-{}-{name}", std::process::id()))
    }

    #[test]
    fn m1_plan_covers_step_1_once() {
        let plan = m1_plan();
        let mut keys: Vec<(i32, i32, u64)> = plan
            .vectors
            .iter()
            .flat_map(|r| {
                r.jds
                    .iter()
                    .map(move |jd| (r.target, r.center, jd.to_bits()))
            })
            .collect();
        assert_eq!(
            keys.len(),
            151,
            "8×8 Standish + 6 Pluto + 3 Sun + 19×4 moons + Halley + Ceres"
        );
        keys.sort_unstable();
        keys.dedup();
        assert_eq!(keys.len(), 151, "a (target, center, JD) is asked twice");
        let mut sky: Vec<_> = plan
            .sky
            .iter()
            .map(|r| (r.target.clone(), r.observer, r.jd_utc.to_bits()))
            .collect();
        assert_eq!(
            sky.len(),
            8,
            "7 planets (Jupiter's '1,20' row included) + C/2020 F3"
        );
        sky.sort_unstable();
        sky.dedup();
        assert_eq!(sky.len(), 8, "an observer spot is asked twice");
        // Small bodies are stored under their NAIF ids: Halley's comet code is its SBDB SPK-ID;
        // Ceres's original-schema id maps to the SBDB SPK-ID that asteroids.bin holds (review r0 I1).
        let ids: Vec<i32> = plan.vectors.iter().map(|r| r.target).collect();
        assert!(ids.contains(&HALLEY_NAIF) && ids.contains(&CERES_NAIF));
        use uni_data::smallbody::asteroid_spkid_from_naif;
        assert_eq!(asteroid_spkid_from_naif(CERES_NAIF), Some(20_000_001));
        assert_eq!(
            asteroid_spkid_from_naif(HALLEY_NAIF),
            None,
            "a comet, not an asteroid"
        );
    }

    #[test]
    fn vector_queries_use_the_briefs_settings() {
        let plan = m1_plan();
        for request in &plan.vectors {
            let params = vector_params(request);
            for (key, value) in [
                ("format", "json"),
                ("EPHEM_TYPE", "'VECTORS'"),
                ("REF_SYSTEM", "'ICRF'"),
                ("REF_PLANE", "'FRAME'"),
                ("VEC_CORR", "'NONE'"),
                ("OUT_UNITS", "'KM-S'"),
                ("TIME_TYPE", "'TDB'"),
                ("TLIST_TYPE", "'JD'"),
                ("VEC_TABLE", "'2'"),
                ("CSV_FORMAT", "'YES'"),
            ] {
                assert_eq!(param(&params, key), value, "{} {key}", request.command);
            }
            assert_eq!(
                param(&params, "CENTER"),
                format!("'500@{}'", request.center)
            );
        }
        let jupiter = plan
            .vectors
            .iter()
            .find(|r| r.target == 5)
            .expect("Jupiter");
        let params = vector_params(jupiter);
        assert_eq!(param(&params, "COMMAND"), "'5'");
        // TDB Julian Dates only, never calendar dates (review I2), in ascending order.
        assert_eq!(
            param(&params, "TLIST"),
            "'2086307.5' '2378496.5' '2415020.5' '2433282.5' '2451545.0' '2461364.5008' '2469807.5' '2634166.5'"
        );
        let command = |target| {
            let request = plan
                .vectors
                .iter()
                .find(|r| r.target == target)
                .expect("planned");
            param(&vector_params(request), "COMMAND").to_owned()
        };
        assert_eq!(command(HALLEY_NAIF), "'DES=1P;CAP<1987;'");
        assert_eq!(command(CERES_NAIF), "'DES=2000001;'");
        assert_eq!(command(901), "'901'");
    }

    #[test]
    fn sky_queries_use_the_briefs_settings() {
        let plan = m1_plan();
        for request in &plan.sky {
            let params = sky_params(request);
            for (key, value) in [
                ("format", "json"),
                ("EPHEM_TYPE", "'OBSERVER'"),
                ("CENTER", "'500@399'"),
                ("REF_SYSTEM", "'ICRF'"),
                ("TIME_TYPE", "'UT'"),
                ("TLIST_TYPE", "'JD'"),
                ("ANG_FORMAT", "'DEG'"),
                ("CAL_FORMAT", "'JD'"),
                ("CSV_FORMAT", "'YES'"),
            ] {
                assert_eq!(param(&params, key), value, "{} {key}", request.command);
            }
        }
        let neowise = plan
            .sky
            .iter()
            .find(|r| r.target == "C/2020 F3")
            .expect("NEOWISE");
        let params = sky_params(neowise);
        assert_eq!(param(&params, "COMMAND"), "'DES=C/2020 F3;CAP;'");
        assert_eq!(param(&params, "QUANTITIES"), "'1,20'");
        assert_eq!(param(&params, "TLIST"), "'2459045.625'");
        for planet in ["199", "299", "499", "599", "699", "799", "899"] {
            let request = plan
                .sky
                .iter()
                .find(|r| r.target == planet)
                .expect("planet");
            let params = sky_params(request);
            assert_eq!(param(&params, "COMMAND"), format!("'{planet}'"));
            assert_eq!(param(&params, "QUANTITIES"), "'1,9,19,20,24'");
            assert_eq!(param(&params, "TLIST"), "'2461364.5'");
        }
    }

    #[test]
    fn parses_vector_table() {
        // Asked out of order: rows are matched to the asked epochs by value.
        let request = jupiter(&[JD_2026_TDB, JD_1000, JD_2000]);
        let (spots, source) = parse_vectors(VECTOR_ANSWER, &request).expect("valid answer");
        assert_eq!(
            source,
            "Jupiter Barycenter (5) [DE441] from Sun (10) [DE441]"
        );
        assert_eq!(spots.len(), 3);
        let s = &spots[2];
        assert_eq!((s.target, s.center), (5, 10));
        assert_eq!(
            s.jd_tdb.to_bits(),
            JD_2026_TDB.to_bits(),
            "the asked JD is stored"
        );
        assert_eq!(
            s.pos_km,
            [
                -5.640475942392980e8,
                5.115198534827855e8,
                2.329813975741199e8
            ]
        );
        assert_eq!(
            s.vel_km_s,
            [-9.381734722133515, -8.037685946387459, -3.216792381573927]
        );
        assert_eq!(spots[0].jd_tdb, JD_1000);
        assert_eq!(spots[1].pos_km[0], 5.985675835979289e8);
        assert_eq!(
            fetch_stamp(VECTOR_ANSWER).as_deref(),
            Some("Sun Nov 22 08:34:47 2026 Pasadena, USA")
        );
        assert_eq!(fetch_stamp(NEOWISE_ANSWER), None);
    }

    /// Horizons labels the barycentres of moonless Mercury and Venus with the body ids:
    /// "Mercury Barycenter (199)" (measured 2026-11-22).
    #[test]
    fn accepts_moonless_barycentre_labels() {
        let all = [JD_1000, JD_2000, JD_2026_TDB];
        let mercury = |label: &str| VECTOR_ANSWER.replace("Jupiter Barycenter (5)", label);
        let (spots, _) = parse_vectors(&mercury("Mercury Barycenter (199)"), &major(1, 10, &all))
            .expect("barycentre 1");
        assert!(spots.iter().all(|s| s.target == 1));
        assert!(parse_vectors(&mercury("Mercury Barycenter (1)"), &major(1, 10, &all)).is_ok());
        assert!(parse_vectors(&mercury("Mercury (199)"), &major(1, 10, &all)).is_err());
        assert!(parse_vectors(&mercury("Venus Barycenter (299)"), &major(1, 10, &all)).is_err());
    }

    #[test]
    fn rejects_mismatched_vector_answers() {
        let all = [JD_1000, JD_2000, JD_2026_TDB];
        let wrong_target = VectorRequest {
            target_names: vec!["(6)".into()],
            ..jupiter(&all)
        };
        assert!(parse_vectors(VECTOR_ANSWER, &wrong_target).is_err());
        let wrong_center = VectorRequest {
            center: 0,
            ..jupiter(&all)
        };
        assert!(parse_vectors(VECTOR_ANSWER, &wrong_center).is_err());
        assert!(parse_vectors(VECTOR_ANSWER, &jupiter(&[JD_1000, JD_2050, JD_2026_TDB])).is_err());
        assert!(parse_vectors(VECTOR_ANSWER, &jupiter(&[JD_1000, JD_2000])).is_err());
        for (from, to) in [
            ("KM-S", "AU-D"),
            (
                "Reference frame : ICRF",
                "Reference frame : Ecliptic of J2000.0",
            ),
            ("GEOMETRIC", "ASTROMETRIC"),
            ("$$SOE", "$$S0E"),
            ("-8.037685946387459E+00", "n.a."),
        ] {
            let answer = VECTOR_ANSWER.replace(from, to);
            assert!(
                parse_vectors(&answer, &jupiter(&all)).is_err(),
                "{from} -> {to}"
            );
        }
    }

    #[test]
    fn parses_observer_tables() {
        let (spot, source) = parse_sky(SATURN_ANSWER, &saturn_sky()).expect("valid answer");
        assert_eq!(source, "Saturn (699) [sat441l] from Earth (399) [DE441]");
        assert_eq!((spot.target.as_str(), spot.observer), ("699", 399));
        assert_eq!(spot.jd_utc, JD_2026_UTC);
        assert_eq!(
            (spot.ra_deg, spot.dec_deg, spot.delta_au),
            (8.331486146, 0.734877993, 8.74653339379041)
        );
        assert_eq!(
            (spot.apmag, spot.r_au, spot.phase_deg),
            (Some(0.555), Some(9.419475012161), Some(4.5638))
        );

        let plan = m1_plan();
        let neowise = plan
            .sky
            .iter()
            .find(|r| r.target == "C/2020 F3")
            .expect("NEOWISE");
        let (spot, _) = parse_sky(NEOWISE_ANSWER, neowise).expect("valid answer");
        assert_eq!(
            (spot.ra_deg, spot.dec_deg, spot.delta_au),
            (115.934519652, 46.76396745, 0.78434409858666)
        );
        assert_eq!((spot.apmag, spot.r_au, spot.phase_deg), (None, None, None));
    }

    #[test]
    fn rejects_mismatched_observer_answers() {
        for (from, to) in [
            ("Saturn (699)", "Jupiter (599)"),
            ("Earth (399)", "Mars (499)"),
            ("2461364.500000000", "2461365.500000000"),
            ("   0.555", "    n.a."),
            (
                "    4.5638,",
                "    4.5638,\n2461364.600000000, , , 1, 1, 1, 1, 1, 1, 1, 1, 1,",
            ),
        ] {
            let answer = SATURN_ANSWER.replace(from, to);
            assert!(parse_sky(&answer, &saturn_sky()).is_err(), "{from} -> {to}");
        }
        // A planet row must carry the quantities 9, 19 and 24 it asked for.
        let answer = SATURN_ANSWER.replace("S-T-O", "S-T-O?");
        assert!(parse_sky(&answer, &saturn_sky()).is_err());
    }

    #[test]
    fn reads_horizons_json() {
        let ok = r#"{"signature":{"source":"NASA/JPL Horizons API","version":"1.2"},"result":"line\n$$SOE\n"}"#;
        let answer = horizons_answer(ok).expect("valid");
        assert_eq!(
            (answer.text.as_str(), answer.api_version.as_str()),
            ("line\n$$SOE\n", "1.2")
        );
        let error = r#"{"error":"No ephemeris for target","signature":{"source":"NASA/JPL Horizons API","version":"1.2"},"result":"..."}"#;
        let why = horizons_answer(error)
            .expect_err("error answer")
            .to_string();
        assert!(why.contains("No ephemeris for target"), "{why}");
        assert!(horizons_answer("<html>busy</html>").is_err());
        assert!(horizons_answer(r#"{"signature":{}}"#).is_err());
    }

    #[test]
    fn written_files_load_back() {
        let (ephem, _) = parse_vectors(VECTOR_ANSWER, &jupiter(&[JD_1000, JD_2000, JD_2026_TDB]))
            .expect("vectors");
        let (sky, _) = parse_sky(SATURN_ANSWER, &saturn_sky()).expect("sky");
        let header = vec![
            "provenance".to_owned(),
            String::new(),
            "  5@10 source".to_owned(),
        ];
        let (ephem_path, sky_path) = (scratch("ephem.ron"), scratch("sky.ron"));
        write_ron(&ephem_path, &header, &ephem).expect("write ephem");
        write_ron(&sky_path, &header, std::slice::from_ref(&sky)).expect("write sky");
        let text = std::fs::read_to_string(&ephem_path).expect("read");
        assert!(
            text.starts_with("// provenance\n//\n//   5@10 source\n["),
            "{text}"
        );
        assert_eq!(
            text.lines().count(),
            3 + 2 + ephem.len(),
            "one spot per line:\n{text}"
        );
        let ephem_back = uni_data::load_ephem_spots(&ephem_path).expect("load ephem");
        let sky_back = uni_data::load_sky_spots(&sky_path).expect("load sky");
        let _ = std::fs::remove_file(&ephem_path);
        let _ = std::fs::remove_file(&sky_path);
        let bits = |s: &EphemSpot| {
            (
                s.target,
                s.center,
                s.jd_tdb.to_bits(),
                s.pos_km.map(f64::to_bits),
                s.vel_km_s.map(f64::to_bits),
            )
        };
        assert_eq!(
            ephem_back.iter().map(bits).collect::<Vec<_>>(),
            ephem.iter().map(bits).collect::<Vec<_>>()
        );
        let sky_bits = |s: &SkySpot| {
            (
                s.target.clone(),
                s.observer,
                [s.jd_utc, s.ra_deg, s.dec_deg, s.delta_au].map(f64::to_bits),
                [s.apmag, s.r_au, s.phase_deg].map(|v| v.map(f64::to_bits)),
            )
        };
        assert_eq!(
            sky_back.iter().map(sky_bits).collect::<Vec<_>>(),
            vec![sky_bits(&sky)]
        );
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&["--plan", "m1", "--out", "a.ron", "--sky", "b.ron"]))
            .expect("valid");
        assert_eq!(options.plan, "m1");
        assert_eq!(
            (options.out, options.sky),
            (PathBuf::from("a.ron"), PathBuf::from("b.ron"))
        );
        assert_eq!(
            options.cache,
            uni_data::data_dir().join("cache").join("horizons")
        );
        let options = parse_options(&args(&[
            "--sky", "b", "--cache", "c", "--out", "a", "--plan", "m1",
        ]))
        .expect("any order");
        assert_eq!(options.cache, PathBuf::from("c"));
        assert!(
            parse_options(&args(&["--plan", "m1", "--out", "a.ron"])).is_err(),
            "--sky missing"
        );
        assert!(
            parse_options(&args(&["--plan", "m1", "--out", "a", "--sky"])).is_err(),
            "no value"
        );
        assert!(
            parse_options(&args(&[
                "--plan", "m1", "--out", "a", "--sky", "b", "--x", "y"
            ]))
            .is_err()
        );
    }
}
