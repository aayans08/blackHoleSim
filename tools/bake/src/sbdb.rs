//! `uni-bake sbdb` (M1 Task A9 Step 2): JPL Small-Body Database query results → `comets.ron`,
//! `asteroids.bin` + `asteroid_names.txt`.
//!
//! `uni-bake sbdb --out-dir data/baked [--cache <dir>]`
//!
//! Every request is one GET of the SBDB Query API
//! (<https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html>; filter syntax in
//! <https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html>) made with `curl -L --fail` through
//! `std::process::Command`, with `full-prec=1`. The raw JSON answers are kept in `--cache`
//! (default `data/cache/`, in `sbdb-asteroids/` and `sbdb-comets/`; git-ignored, deleted after
//! the bake):
//! 1. Counts (`limit=1`) of asteroids (`sb-kind=a`) with H < 14, H ≤ 14, H ≤ 14.5 and H ≤ 15.
//!    Lead ruling 11: H ≤ 14 stands unless that count exceeds 60 000; then the bake stops.
//! 2. Asteroids with H ≤ 14 (`sb-cdata={"AND":["H|LE|14"]}`), fields
//!    `spkid,full_name,a,e,i,om,w,ma,epoch,H,G,diameter,albedo`. A missing G becomes 0.15, the
//!    default the SBDB documentation names for the H–G law; a missing diameter or albedo, NaN.
//!    A row whose orbit is not closed and complete (an element missing, or e ≥ 1) is skipped and
//!    named; more than 0.1 % of such rows stops the bake.
//! 3. Numbered periodic comets (`sb-kind=c`, `sb-ns=n`) without fragments (`sb-xfrag=1`), fields
//!    `spkid,full_name,e,q,i,om,w,tp,epoch,M1,K1,diameter`.
//! 4. Unnumbered comets (`sb-kind=c`, `sb-ns=u`), same fields, of which only the six named
//!    long-period comets of A9 Step 2 are kept (the API cannot filter on designations).
//!
//! Columns are looked up by the answer's `fields` names, not by position. SBDB sends numbers as
//! strings and missing values as `null`. Nothing is written unless every request and row passes.
//!
//! This module also holds the small helpers the other A9 bakes share (`curl_get`, `utc_now`,
//! `write_ron`).

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail, ensure};
use uni_data::{AsteroidRecord, Comet};

const USAGE: &str = "usage: uni-bake sbdb --out-dir <data/baked> [--cache <dir>]";

/// The SBDB Query API endpoint.
const API_URL: &str = "https://ssd-api.jpl.nasa.gov/sbdb_query.api";

/// Asteroid fields, in the brief's order.
const ASTEROID_FIELDS: &str = "spkid,full_name,a,e,i,om,w,ma,epoch,H,G,diameter,albedo";
/// Comet fields, in the brief's order.
const COMET_FIELDS: &str = "spkid,full_name,e,q,i,om,w,tp,epoch,M1,K1,diameter";

/// Lead ruling 11: the catalogue limit, and the count above which the lead picks instead.
const H_MAX: &str = "14";
const MAX_ASTEROIDS: u64 = 60_000;

/// The six long-period comets of A9 Step 2, by designation.
const NAMED_COMETS: [&str; 6] = [
    "C/1995 O1",
    "C/2020 F3",
    "C/1996 B2",
    "C/2006 P1",
    "C/2023 A3",
    "C/2011 L4",
];

/// SBDB's default H–G slope parameter, used where an asteroid has no G ("magnitude slope
/// parameter (default is 0.15)", SBDB Query API documentation, field G).
const DEFAULT_G: f32 = 0.15;

/// Runs `uni-bake sbdb` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    let asteroid_dir = options.cache.join("sbdb-asteroids");
    let comet_dir = options.cache.join("sbdb-comets");
    for dir in [&asteroid_dir, &comet_dir, &options.out_dir] {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let fetch = |params: Vec<(&'static str, String)>, raw: PathBuf| -> anyhow::Result<Table> {
        log::info!("sbdb: {}", raw.display());
        curl_get(API_URL, &params, &raw)?;
        let body =
            std::fs::read_to_string(&raw).with_context(|| format!("reading {}", raw.display()))?;
        sbdb_table(&body).with_context(|| format!("SBDB answer {}", raw.display()))
    };

    // Step 2's counts first (ruling 11).
    let mut counts = Vec::new();
    for (op, h, label) in [
        ("LT", "14", "H < 14"),
        ("LE", "14", "H <= 14"),
        ("LE", "14.5", "H <= 14.5"),
        ("LE", "15", "H <= 15"),
    ] {
        let raw = asteroid_dir.join(format!("count-H{op}{h}.json"));
        let count = fetch(count_params(op, h), raw)?.count;
        println!("sbdb: asteroids with {label}: {count}");
        counts.push((label, count));
    }
    let wanted = counts[1].1;
    ensure!(
        wanted <= MAX_ASTEROIDS,
        "{wanted} asteroids have H <= {H_MAX}, more than {MAX_ASTEROIDS}: lead ruling 11 says to \
         report this and wait for the lead's pick"
    );

    let table = fetch(
        asteroid_params(),
        asteroid_dir.join(format!("asteroids-HLE{H_MAX}.json")),
    )?;
    ensure!(
        table.data.len() as u64 == table.count && table.count == wanted,
        "the asteroid query returned {} rows for a count of {} (count query: {wanted})",
        table.data.len(),
        table.count
    );
    let (records, names, skipped) = asteroids_from(&table)?;
    for name in &skipped {
        println!("sbdb: skipped {name}: its orbit lacks an element or is unbound");
    }
    ensure!(
        skipped.len() * 1000 <= table.data.len(),
        "{} of {} asteroids lack an orbital element or are unbound: more than 0.1 %, so the \
         answer looks wrong",
        skipped.len(),
        table.data.len()
    );

    let numbered = fetch(comet_params("n"), comet_dir.join("numbered-nofrag.json"))?;
    let unnumbered = fetch(comet_params("u"), comet_dir.join("unnumbered.json"))?;
    for (what, t) in [("numbered", &numbered), ("unnumbered", &unnumbered)] {
        ensure!(
            t.data.len() as u64 == t.count,
            "the {what} comet query returned {} rows for a count of {}",
            t.data.len(),
            t.count
        );
    }
    let mut comets = comets_from(&numbered)?;
    let periodic = comets.len();
    comets.extend(pick_named(&comets_from(&unnumbered)?, &NAMED_COMETS)?);
    let mut header = vec![
        "JPL Small-Body Database: numbered periodic comets (fragments excluded) and six named".to_owned(),
        "long-period comets (M1 Task A9 Step 2). Written by `uni-bake sbdb` from the SBDB Query API".to_owned(),
        format!("{API_URL}, fetched {} UTC (`date -u`).", utc_now()),
        "Public JPL Solar System Dynamics data; see ATTRIBUTION.md and data/manifest.toml (id \"sbdb-comets\").".to_owned(),
        String::new(),
        format!("Queries: sb-kind=c sb-ns=n sb-xfrag=1 ({} rows) and sb-kind=c sb-ns=u ({} rows, kept: {}),", numbered.count, unnumbered.count, NAMED_COMETS.join(", ")),
        format!("fields={COMET_FIELDS}, full-prec=1."),
        "Heliocentric osculating elements, ecliptic and mean equinox of J2000; angles in degrees; tp and".to_owned(),
        "epoch are TDB Julian Dates; m1/k1 the total-magnitude law m = M1 + 5 log10(delta) + K1 log10(r);".to_owned(),
        "None where SBDB has no value.".to_owned(),
    ];
    header.push(format!(
        "{periodic} numbered periodic + {} named = {} comets.",
        comets.len() - periodic,
        comets.len()
    ));
    // Every request and row has passed: write.
    let bin = options.out_dir.join("asteroids.bin");
    let names_path = options.out_dir.join("asteroid_names.txt");
    uni_data::smallbody::write_asteroids(&bin, &names_path, &records, &names)?;
    let comets_path = options.out_dir.join("comets.ron");
    write_ron(&comets_path, &header, &comets, true)?;
    uni_data::load_comets(&comets_path).context("reading back comets.ron")?;
    uni_data::load_asteroids(&bin, &names_path).context("reading back asteroids.bin")?;

    println!(
        "sbdb: {} asteroids (H <= {H_MAX}; {} skipped) -> {} + {}; {} comets ({periodic} numbered periodic + {} named) -> {}",
        records.len(),
        skipped.len(),
        bin.display(),
        names_path.display(),
        comets.len(),
        comets.len() - periodic,
        comets_path.display()
    );
    Ok(())
}

/// One SBDB Query API answer: the field names, the rows and the total count of matches.
#[derive(Debug)]
struct Table {
    fields: Vec<String>,
    data: Vec<Vec<serde_json::Value>>,
    count: u64,
}

/// Reads an SBDB Query API JSON answer. A count-only answer has no fields or data.
fn sbdb_table(body: &str) -> anyhow::Result<Table> {
    let json: serde_json::Value =
        serde_json::from_str(body).context("the SBDB answer is not JSON")?;
    let source = json
        .pointer("/signature/source")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    ensure!(
        source.contains("SBDB"),
        "not an SBDB Query API answer (signature source {source:?})"
    );
    let count = json
        .get("count")
        .and_then(serde_json::Value::as_u64)
        .context("no numeric \"count\"")?;
    let fields = match json.get("fields") {
        None => Vec::new(),
        Some(fields) => fields
            .as_array()
            .context("\"fields\" is not an array")?
            .iter()
            .map(|f| {
                f.as_str()
                    .map(str::to_owned)
                    .context("a field name is not a string")
            })
            .collect::<anyhow::Result<_>>()?,
    };
    let data = match json.get("data") {
        None => Vec::new(),
        Some(data) => data
            .as_array()
            .context("\"data\" is not an array")?
            .iter()
            .map(|row| row.as_array().cloned().context("a row is not an array"))
            .collect::<anyhow::Result<Vec<_>>>()?,
    };
    if let Some((index, row)) = data
        .iter()
        .enumerate()
        .find(|(_, r)| r.len() != fields.len())
    {
        bail!(
            "row {index} has {} values for {} fields",
            row.len(),
            fields.len()
        );
    }
    Ok(Table {
        fields,
        data,
        count,
    })
}

/// Reads one table's rows by field name.
struct Rows<'a> {
    table: &'a Table,
    columns: Vec<usize>,
}

impl<'a> Rows<'a> {
    /// Looks up the columns of `names` (comma-separated), which must all be present.
    fn new(table: &'a Table, names: &str) -> anyhow::Result<Self> {
        let columns = names
            .split(',')
            .map(|name| {
                table
                    .fields
                    .iter()
                    .position(|f| f == name)
                    .with_context(|| format!("the answer has no {name:?} field"))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self { table, columns })
    }

    /// The value of the `k`-th requested field in row `row`.
    fn get(&self, row: usize, k: usize) -> &'a serde_json::Value {
        &self.table.data[row][self.columns[k]]
    }

    /// A number, sent as a string or a JSON number; `None` for `null`.
    fn number(&self, row: usize, k: usize) -> anyhow::Result<Option<f64>> {
        let field = &self.table.fields[self.columns[k]];
        match self.get(row, k) {
            serde_json::Value::Null => Ok(None),
            serde_json::Value::String(text) => text
                .trim()
                .parse::<f64>()
                .ok()
                .filter(|v| v.is_finite())
                .map(Some)
                .with_context(|| format!("row {row}: {field} {text:?} is not a number")),
            serde_json::Value::Number(n) => Ok(n.as_f64()),
            other => bail!("row {row}: {field} {other} is not a number"),
        }
    }

    fn required(&self, row: usize, k: usize) -> anyhow::Result<f64> {
        self.number(row, k)?
            .with_context(|| format!("row {row}: no {} value", self.table.fields[self.columns[k]]))
    }

    /// The SPK-ID, a JSON integer.
    fn spkid(&self, row: usize, k: usize) -> anyhow::Result<u64> {
        self.get(row, k).as_u64().with_context(|| {
            format!(
                "row {row}: SPK-ID {} is not a positive integer",
                self.get(row, k)
            )
        })
    }

    /// The trimmed full name.
    fn name(&self, row: usize, k: usize) -> anyhow::Result<String> {
        let name = self.get(row, k).as_str().map(str::trim).unwrap_or_default();
        ensure!(!name.is_empty(), "row {row}: no name");
        Ok(name.to_owned())
    }
}

/// Asteroid records and names from an asteroid table, in its order (`name_index` is the record's
/// own index), and the names of the rows skipped because their orbit is not a closed one that can
/// be placed: an element (a, e, i, node, argument of perihelion, mean anomaly or epoch) is
/// missing, or the orbit is unbound (e ≥ 1, as for some "A/" objects).
fn asteroids_from(
    table: &Table,
) -> anyhow::Result<(Vec<AsteroidRecord>, Vec<String>, Vec<String>)> {
    let rows = Rows::new(table, ASTEROID_FIELDS)?;
    let mut records = Vec::with_capacity(table.data.len());
    let mut names = Vec::with_capacity(table.data.len());
    let mut skipped = Vec::new();
    for row in 0..table.data.len() {
        let spkid = rows.spkid(row, 0)?;
        // Fields 2..=8: a, e, i, om, w, ma, epoch. A malformed value is an error; a missing one
        // skips the row.
        let elements = (2..=8)
            .map(|k| rows.number(row, k))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let Some([a_au, e, i_deg, node_deg, argp_deg, m0_deg, epoch_jd_tdb]) = elements
            .into_iter()
            .collect::<Option<Vec<f64>>>()
            .and_then(|orbit| <[f64; 7]>::try_from(orbit).ok())
            .filter(|orbit| orbit[1] < 1.0)
        else {
            skipped.push(rows.name(row, 1)?);
            continue;
        };
        let f = |k| rows.required(row, k);
        let optional =
            |k| -> anyhow::Result<f32> { Ok(rows.number(row, k)?.map_or(f32::NAN, |v| v as f32)) };
        let name_index = u32::try_from(records.len()).context("too many rows")?;
        records.push(AsteroidRecord {
            a_au,
            e,
            i_deg,
            node_deg,
            argp_deg,
            m0_deg,
            epoch_jd_tdb,
            h: f(9)? as f32,
            g: rows.number(row, 10)?.map_or(DEFAULT_G, |g| g as f32),
            diameter_km: optional(11)?,
            albedo: optional(12)?,
            spkid: u32::try_from(spkid)
                .with_context(|| format!("row {row}: SPK-ID {spkid} does not fit a u32"))?,
            name_index,
        });
        names.push(rows.name(row, 1)?);
    }
    Ok((records, names, skipped))
}

/// Comets from a comet table.
fn comets_from(table: &Table) -> anyhow::Result<Vec<Comet>> {
    let rows = Rows::new(table, COMET_FIELDS)?;
    (0..table.data.len())
        .map(|row| {
            let f = |k| rows.required(row, k);
            let spkid = rows.spkid(row, 0)?;
            Ok(Comet {
                name: rows.name(row, 1)?,
                spkid: i32::try_from(spkid)
                    .with_context(|| format!("row {row}: SPK-ID {spkid} does not fit an i32"))?,
                e: f(2)?,
                q_au: f(3)?,
                i_deg: f(4)?,
                node_deg: f(5)?,
                argp_deg: f(6)?,
                tp_jd_tdb: f(7)?,
                epoch_jd_tdb: f(8)?,
                m1: rows.number(row, 9)?,
                k1: rows.number(row, 10)?,
                diameter_km: rows.number(row, 11)?,
            })
        })
        .collect()
}

/// The comets named by `designations`, in that order. Each designation must match exactly one
/// comet: its name is the designation alone or the designation followed by " (" and a name.
fn pick_named(comets: &[Comet], designations: &[&str]) -> anyhow::Result<Vec<Comet>> {
    designations
        .iter()
        .map(|designation| {
            let named = format!("{designation} (");
            let matches: Vec<&Comet> = comets
                .iter()
                .filter(|c| c.name == *designation || c.name.starts_with(&named))
                .collect();
            match matches.as_slice() {
                [comet] => Ok((*comet).clone()),
                [] => bail!("comet {designation} is not in the answer"),
                more => bail!("comet {designation} matches {} rows", more.len()),
            }
        })
        .collect()
}

/// `sb-cdata` for `H <op> h`, compact JSON as the filter documentation recommends.
fn h_filter(op: &str, h: &str) -> String {
    format!(r#"{{"AND":["H|{op}|{h}"]}}"#)
}

/// The query of a count of asteroids with `H <op> h` (`op` is an SBDB operator: LT, LE).
fn count_params(op: &str, h: &str) -> Vec<(&'static str, String)> {
    vec![
        ("fields", "spkid".to_owned()),
        ("sb-kind", "a".to_owned()),
        ("sb-cdata", h_filter(op, h)),
        ("limit", "1".to_owned()),
    ]
}

/// The query of every asteroid with H ≤ [`H_MAX`].
fn asteroid_params() -> Vec<(&'static str, String)> {
    vec![
        ("fields", ASTEROID_FIELDS.to_owned()),
        ("sb-kind", "a".to_owned()),
        ("sb-cdata", h_filter("LE", H_MAX)),
        ("full-prec", "1".to_owned()),
    ]
}

/// The query of the comets with numbered status `ns` ("n" without fragments, or "u").
fn comet_params(ns: &str) -> Vec<(&'static str, String)> {
    let mut params = vec![
        ("fields", COMET_FIELDS.to_owned()),
        ("sb-kind", "c".to_owned()),
        ("sb-ns", ns.to_owned()),
    ];
    if ns == "n" {
        params.push(("sb-xfrag", "1".to_owned()));
    }
    params.push(("full-prec", "1".to_owned()));
    params
}

/// `--out-dir` (required) and `--cache` (default `<data>/cache`).
#[derive(Debug)]
struct Options {
    out_dir: PathBuf,
    cache: PathBuf,
}

fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut out_dir, mut cache) = (None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--out-dir") => &mut out_dir,
            Some("--cache") => &mut cache,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    let out_dir = out_dir.with_context(|| format!("sbdb needs --out-dir\n{USAGE}"))?;
    let cache = cache.unwrap_or_else(|| uni_data::data_dir().join("cache"));
    Ok(Options { out_dir, cache })
}

/// Downloads `url` to `out` with `curl -L --fail`, as a GET with `params` URL-encoded by curl.
pub(crate) fn curl_get(url: &str, params: &[(&str, String)], out: &Path) -> anyhow::Result<()> {
    let mut curl = std::process::Command::new("curl");
    curl.args([
        "-sS",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--max-time",
        "600",
    ])
    .arg("-o")
    .arg(out);
    if !params.is_empty() {
        curl.arg("--get");
        for (key, value) in params {
            curl.arg("--data-urlencode").arg(format!("{key}={value}"));
        }
    }
    curl.arg(url);
    let output = curl.output().context("running curl")?;
    ensure!(
        output.status.success(),
        "curl {url} failed ({}): {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    Ok(())
}

/// The current UTC time from `date -u` ("2026-11-22 17:30"), for provenance headers.
pub(crate) fn utc_now() -> String {
    std::process::Command::new("date")
        .args(["-u", "+%Y-%m-%d %H:%M"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| "at an unknown time".to_owned())
}

/// Writes `items` as one RON sequence after `header` as `//` comment lines; `compact` puts each
/// struct on one line.
pub(crate) fn write_ron<T: serde::Serialize>(
    path: &Path,
    header: &[String],
    items: &[T],
    compact: bool,
) -> anyhow::Result<()> {
    let config = ron::ser::PrettyConfig::new().compact_structs(compact);
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

    /// A trimmed live answer (2026-11-22), fields reordered to prove lookup by name: (1) Ceres as
    /// served, and an object with no G, diameter or albedo.
    const ASTEROIDS: &str = r#"{"signature":{"source":"NASA/JPL SBDB (Small-Body DataBase) Query API","version":"1.0"},
        "fields":["full_name","spkid","a","e","i","om","w","ma","epoch","H","G","diameter","albedo"],
        "data":[["     1 Ceres (A801 AA)",20000001,"2.765552595034094",".07969229514816586","10.58802780183462","80.24862682043221","73.29421453021587","274.4193463761342","2461200.5","3.34","0.12","939.4","0.090"],
                ["  (2001 XY123)",54321098,"3.1","0.25","5.5","100.","200.","300.","2461200.5","13.9",null,null,null]],
        "count":2}"#;

    const COMETS: &str = r#"{"signature":{"source":"NASA/JPL SBDB (Small-Body DataBase) Query API","version":"1.0"},
        "fields":["spkid","full_name","e","q","i","om","w","tp","epoch","M1","K1","diameter"],
        "data":[[1000036,"    1P/Halley",".9679359956953211",".5748638313743413","162.1905300439129","59.09894720612437","112.2414314637764","2446469.973616146677","2439875.5","5.5","8.0","11.0"],
                [1000589,"     C/-146 P1","1.0","0.43","71","330","261","1667909.5","1667909.5",null,null,null]],
        "count":3479}"#;

    fn table(body: &str) -> Table {
        sbdb_table(body).expect("valid table")
    }

    #[test]
    fn reads_sbdb_tables() {
        let t = table(COMETS);
        assert_eq!(t.count, 3479);
        assert_eq!(t.fields[1], "full_name");
        assert_eq!(t.data.len(), 2);
        assert!(sbdb_table("not json").is_err());
        assert!(
            sbdb_table(r#"{"fields":[],"data":[],"count":0}"#).is_err(),
            "no signature"
        );
        assert!(
            sbdb_table(r#"{"signature":{"source":"NASA/JPL SBDB (Small-Body DataBase) Query API"},"count":"x"}"#)
                .is_err(),
            "count is not a number"
        );
        // A count-only answer has no fields or data.
        let counted = table(
            r#"{"signature":{"source":"NASA/JPL SBDB (Small-Body DataBase) Query API","version":"1.0"},"count":33681}"#,
        );
        assert_eq!((counted.count, counted.data.len()), (33_681, 0));
    }

    /// An object whose orbit lacks an element (SBDB had one on 2026-11-22: (2002 PD153), e = 0
    /// and no mean anomaly) or is unbound (two "A/" objects with e > 1 that day) cannot be kept as
    /// a closed orbit: it is skipped and named, not an error.
    #[test]
    fn skips_rows_without_a_closed_orbit() {
        let unbound = ASTEROIDS
            .replacen("\"2.765552595034094\"", "\"-3067.54692202654\"", 1)
            .replacen("\".07969229514816586\"", "\"1.002666334453656\"", 1);
        let (records, _, skipped) = asteroids_from(&table(&unbound)).expect("parses");
        assert_eq!(
            (records.len(), skipped.as_slice()),
            (1, &["1 Ceres (A801 AA)".to_owned()][..])
        );
        for field in [
            "\"2.765552595034094\"",
            "\"274.4193463761342\"",
            "\"2461200.5\"",
        ] {
            let body = ASTEROIDS.replacen(field, "null", 1);
            let (records, names, skipped) = asteroids_from(&table(&body)).expect("parses");
            assert_eq!(skipped, ["1 Ceres (A801 AA)"], "{field}");
            assert_eq!(names, ["(2001 XY123)"]);
            assert_eq!((records.len(), records[0].name_index), (1, 0));
        }
    }

    #[test]
    fn parses_asteroid_rows() {
        let (records, names, skipped) = asteroids_from(&table(ASTEROIDS)).expect("asteroids");
        assert!(skipped.is_empty());
        assert_eq!(names, ["1 Ceres (A801 AA)", "(2001 XY123)"]);
        let ceres = records[0];
        assert_eq!(ceres.spkid, 20_000_001);
        assert_eq!(ceres.name_index, 0);
        assert_eq!(ceres.a_au, 2.765_552_595_034_094);
        assert_eq!(ceres.e, 0.079_692_295_148_165_86);
        assert_eq!(ceres.i_deg, 10.588_027_801_834_62);
        assert_eq!(ceres.node_deg, 80.248_626_820_432_21);
        assert_eq!(ceres.argp_deg, 73.294_214_530_215_87);
        assert_eq!(ceres.m0_deg, 274.419_346_376_134_2);
        assert_eq!(ceres.epoch_jd_tdb, 2_461_200.5);
        assert_eq!(
            (ceres.h, ceres.g, ceres.diameter_km, ceres.albedo),
            (3.34, 0.12, 939.4, 0.09)
        );
        let unknown = records[1];
        assert_eq!((unknown.spkid, unknown.name_index), (54_321_098, 1));
        assert_eq!(unknown.g, DEFAULT_G);
        assert!(unknown.diameter_km.is_nan() && unknown.albedo.is_nan());
    }

    #[test]
    fn rejects_bad_asteroid_rows() {
        let swap = |from: &str, to: &str| {
            assert!(ASTEROIDS.contains(from), "fixture lacks {from}");
            asteroids_from(&table(&ASTEROIDS.replacen(from, to, 1)))
        };
        assert!(
            swap("\"2.765552595034094\"", "\"2.7x\"").is_err(),
            "a not a number"
        );
        assert!(swap("\"3.34\"", "null").is_err(), "H missing");
        assert!(swap("20000001", "-5").is_err(), "negative SPK-ID");
        assert!(
            swap("20000001", "\"20000001\"").is_err(),
            "SPK-ID not a number"
        );
        assert!(
            swap("\"albedo\"", "\"rho\"").is_err(),
            "albedo column missing"
        );
        assert!(
            swap("\"     1 Ceres (A801 AA)\"", "\"  \"").is_err(),
            "blank name"
        );
    }

    /// An orbit the loader would reject parses, but the bake's writer (uni-data's) refuses it.
    #[test]
    fn writer_refuses_impossible_orbits() {
        let body = ASTEROIDS.replacen("\"10.58802780183462\"", "\"-5\"", 1);
        let (records, names, _) = asteroids_from(&table(&body)).expect("parses");
        let dir = std::env::temp_dir();
        let pid = std::process::id();
        let (bin, txt) = (
            dir.join(format!("uni-bake-a9-{pid}-e.bin")),
            dir.join(format!("uni-bake-a9-{pid}-e.txt")),
        );
        let written = uni_data::smallbody::write_asteroids(&bin, &txt, &records, &names);
        assert!(written.is_err(), "i = -5 deg was written");
        assert!(!bin.exists(), "nothing is written");
    }

    #[test]
    fn parses_comet_rows() {
        let comets = comets_from(&table(COMETS)).expect("comets");
        let halley = &comets[0];
        assert_eq!(
            (halley.name.as_str(), halley.spkid),
            ("1P/Halley", 1_000_036)
        );
        assert_eq!(halley.q_au, 0.574_863_831_374_341_3);
        assert_eq!(halley.e, 0.967_935_995_695_321_1);
        assert_eq!(halley.i_deg, 162.190_530_043_912_9);
        assert_eq!(halley.node_deg, 59.098_947_206_124_37);
        assert_eq!(halley.argp_deg, 112.241_431_463_776_4);
        assert_eq!(halley.tp_jd_tdb, 2_446_469.973_616_146_677);
        assert_eq!(halley.epoch_jd_tdb, 2_439_875.5);
        assert_eq!(
            (halley.m1, halley.k1, halley.diameter_km),
            (Some(5.5), Some(8.0), Some(11.0))
        );
        let ancient = &comets[1];
        assert_eq!(ancient.name, "C/-146 P1");
        assert_eq!((ancient.e, ancient.q_au), (1.0, 0.43));
        assert_eq!(
            (ancient.m1, ancient.k1, ancient.diameter_km),
            (None, None, None)
        );
        let no_q = COMETS.replacen("\".5748638313743413\"", "null", 1);
        assert!(comets_from(&table(&no_q)).is_err(), "q missing");
    }

    fn comet(name: &str) -> Comet {
        Comet {
            name: name.to_owned(),
            spkid: 1,
            q_au: 1.0,
            e: 0.99,
            i_deg: 10.0,
            node_deg: 0.0,
            argp_deg: 0.0,
            tp_jd_tdb: 2_451_545.0,
            epoch_jd_tdb: 2_451_545.0,
            m1: None,
            k1: None,
            diameter_km: None,
        }
    }

    #[test]
    fn keeps_only_the_named_comets() {
        let pool: Vec<Comet> = [
            "C/1995 O10 (Decoy)",
            "C/1995 O1 (Hale-Bopp)",
            "C/2020 F3 (NEOWISE)",
            "C/2020 F3-B",
            "C/1996 B2 (Hyakutake)",
            "C/2006 P1 (McNaught)",
            "C/2023 A3 (Tsuchinshan-ATLAS)",
            "C/2011 L4 (PANSTARRS)",
        ]
        .into_iter()
        .map(comet)
        .collect();
        let picked = pick_named(&pool, &NAMED_COMETS).expect("all six");
        let names: Vec<&str> = picked.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "C/1995 O1 (Hale-Bopp)",
                "C/2020 F3 (NEOWISE)",
                "C/1996 B2 (Hyakutake)",
                "C/2006 P1 (McNaught)",
                "C/2023 A3 (Tsuchinshan-ATLAS)",
                "C/2011 L4 (PANSTARRS)",
            ]
        );
        assert!(
            pick_named(&pool[..3], &NAMED_COMETS).is_err(),
            "a named comet is missing"
        );
        let mut twice = pool.clone();
        twice.push(comet("C/2011 L4"));
        assert!(
            pick_named(&twice, &NAMED_COMETS).is_err(),
            "a designation matches twice"
        );
    }

    #[test]
    fn queries_use_the_briefs_fields_and_filters() {
        let find = |params: &[(&'static str, String)], key: &str| {
            params
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, v)| v.clone())
        };
        let asteroids = asteroid_params();
        assert_eq!(find(&asteroids, "fields").as_deref(), Some(ASTEROID_FIELDS));
        assert_eq!(find(&asteroids, "sb-kind").as_deref(), Some("a"));
        assert_eq!(
            find(&asteroids, "sb-cdata").as_deref(),
            Some(r#"{"AND":["H|LE|14"]}"#)
        );
        assert_eq!(find(&asteroids, "full-prec").as_deref(), Some("1"));
        assert_eq!(find(&asteroids, "limit"), None);
        let count = count_params("LT", "14");
        assert_eq!(
            find(&count, "sb-cdata").as_deref(),
            Some(r#"{"AND":["H|LT|14"]}"#)
        );
        assert_eq!(find(&count, "limit").as_deref(), Some("1"));
        let numbered = comet_params("n");
        assert_eq!(find(&numbered, "fields").as_deref(), Some(COMET_FIELDS));
        assert_eq!(find(&numbered, "sb-kind").as_deref(), Some("c"));
        assert_eq!(find(&numbered, "sb-ns").as_deref(), Some("n"));
        assert_eq!(find(&numbered, "sb-xfrag").as_deref(), Some("1"));
        let unnumbered = comet_params("u");
        assert_eq!(find(&unnumbered, "sb-ns").as_deref(), Some("u"));
        assert_eq!(find(&unnumbered, "sb-xfrag"), None);
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&["--out-dir", "o", "--cache", "c"])).expect("valid");
        assert_eq!(
            (options.out_dir, options.cache),
            (PathBuf::from("o"), PathBuf::from("c"))
        );
        let options = parse_options(&args(&["--out-dir", "o"])).expect("cache is optional");
        assert!(
            options.cache.ends_with("cache"),
            "{}",
            options.cache.display()
        );
        assert!(parse_options(&args(&[])).is_err());
        assert!(parse_options(&args(&["--out-dir"])).is_err());
        assert!(parse_options(&args(&["--out", "o"])).is_err());
    }

    #[test]
    fn write_ron_puts_the_header_first() {
        let path = std::env::temp_dir().join(format!("uni-bake-a9-{}-w.ron", std::process::id()));
        write_ron(
            &path,
            &["line one".to_owned(), String::new()],
            &[1.5_f64, 2.0],
            true,
        )
        .expect("write");
        let text = std::fs::read_to_string(&path).expect("read");
        let _ = std::fs::remove_file(&path);
        assert!(text.starts_with("// line one\n//\n"), "{text}");
        let back: Vec<f64> = ron::from_str(&text).expect("RON");
        assert_eq!(back, [1.5, 2.0]);
    }
}
