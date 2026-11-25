//! Builds a self-contained HTML contact sheet of captures and their JSON metrics (M0 Task 12;
//! v2 with the Checks and Performance tables, M1 Task B8).
//!
//! `uni-verify sheet --captures <dir> --out <dir>/index.html [--bench <bench.json>]` lists every
//! `*.png` in the captures directory, sorted by file name, as a thumbnail that links to the full
//! image and a table of its metrics, under the git commit and the time the sheet was made. The
//! page has inline styles and no scripts; images are linked relative to the sheet, so the sheet
//! and the captures move together.
//!
//! Task 11 writes `<name>.png` with `<name>.json` beside it:
//! `{name, width, height, gpu_ms: {pass: ms | null}, gpu_frame_ms: ms | null, cpu_ms,
//! star_count, lens_scale, epoch_utc}`. A GPU time of `null` is a pass the device did not
//! measure validly and shows as "n/a" (review I5: never as 0). Otherwise the JSON is read
//! tolerantly: a missing, null or empty value shows as "—", a value of an unexpected type is
//! shown as it is, and unknown keys get rows of their own after the known ones. A missing or
//! unparsable JSON file gives a note and a table of dashes.
//!
//! The capture JSON v2 keys (M1 B9 Step 2) are summarised, one row each and only when present:
//! `ev100` (scene and star EV), `honesty` (the labels), `metrics` (one row per key; a `*_ms`
//! timing of `null` is "n/a"), `hdr` (the sidecar and its size), `hud` (one row per key), and
//! the names or counts of `projected`, `regions`, `samples`, `features` and `flux` rather than
//! their pixel lists.
//!
//! Above the captures:
//! - **Checks**, from `<captures>/checks.json` (written by `uni-verify gate`, see `gate`): an
//!   array of `{row, bookmark, check, output, pass}` or `{bookmark, check, measured, expected,
//!   pass}` (or `{checks: [...]}`), one row each, with `output` shown as the measured value when
//!   there is no `measured`, and `pass` shown as PASS or FAIL with a count of each.
//! - **Performance**, from the bench JSON (`--bench`, else `<captures>/bench.json`; schema in
//!   `bench`): per bookmark, the first and last 10 s side by side for p95 GPU frame, p95 CPU and
//!   p95 frame interval, the mean frame rate, and the passes over 1.25 × their budget in the
//!   last 10 s. Last-10-s cells over the plan's FPS gate ([`bench::GATE`]) are marked. A timing
//!   with no valid measurement is "n/a", a missing window "—".
//!
//! Either table is left out when its default file is absent; an unreadable one gives a note.
//! An explicit `--bench` file that cannot be read is an error.

use crate::bench::{self, BenchFile, BenchWindow, Stats, Window};
use anyhow::Context;
use serde::Deserialize;
use serde_json::{Map, Value};
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Shown for a metric that is missing, null or empty.
const MISSING: &str = "—";

/// One capture's metrics file, read tolerantly (see the module docs). Values stay as JSON so
/// that a value of an unexpected type is shown rather than failing the whole file.
#[derive(Debug, Default, Deserialize)]
struct Metrics {
    name: Option<Value>,
    width: Option<Value>,
    height: Option<Value>,
    /// GPU time per render pass [ms], keyed by pass name; `null` for an invalid measurement.
    gpu_ms: Option<Value>,
    /// Whole-frame GPU time [ms]; `null` when not measured validly (kept as
    /// `Some(Value::Null)`, unlike a missing key).
    #[serde(default, deserialize_with = "present")]
    gpu_frame_ms: Option<Value>,
    cpu_ms: Option<Value>,
    star_count: Option<Value>,
    lens_scale: Option<Value>,
    epoch_utc: Option<Value>,
    // Capture JSON v2 (M1 B9 Step 2), summarised by `v2_rows`.
    ev100: Option<Value>,
    honesty: Option<Value>,
    metrics: Option<Value>,
    hdr: Option<Value>,
    hud: Option<Value>,
    projected: Option<Value>,
    regions: Option<Value>,
    samples: Option<Value>,
    features: Option<Value>,
    flux: Option<Value>,
    /// Keys the capture format does not (yet) document.
    #[serde(flatten)]
    extra: Map<String, Value>,
}

/// Deserialises a present key as `Some`, even when its value is `null`; a missing key falls back
/// to `None` through `#[serde(default)]`.
fn present<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Value>, D::Error> {
    Value::deserialize(d).map(Some)
}

/// What the sheet shows for one capture.
struct Capture {
    /// File stem of the PNG, e.g. `bh-shadow-30M`.
    stem: String,
    /// Link to the PNG, relative to the sheet's directory and percent-encoded.
    href: String,
    metrics: Metrics,
    /// Why the metrics are missing, if they are.
    note: Option<String>,
}

/// The checks file the sheet reads from the captures directory.
const CHECKS_FILE: &str = "checks.json";
/// The bench file the sheet reads from the captures directory when `--bench` is not given.
const BENCH_FILE: &str = "bench.json";

/// A table read from a file beside the captures.
enum Loaded<T> {
    /// The file does not exist: the table is left out.
    Absent,
    /// The file exists but cannot be read or parsed: the table is a note with the reason.
    Broken(String),
    Read(T),
}

/// One row of `checks.json`, read tolerantly: every field may be missing or of any type.
#[derive(Debug, Default, Deserialize)]
struct CheckRow {
    /// The plan row (`gate`).
    #[serde(default)]
    row: Option<Value>,
    #[serde(default)]
    bookmark: Option<Value>,
    #[serde(default)]
    check: Option<Value>,
    #[serde(default)]
    measured: Option<Value>,
    /// The check's last output line (`gate`), shown when there is no `measured`.
    #[serde(default)]
    output: Option<Value>,
    #[serde(default)]
    expected: Option<Value>,
    #[serde(default)]
    pass: Option<Value>,
}

/// The bench results the Performance table shows, and the file they came from.
struct Perf {
    source: String,
    file: BenchFile,
}

/// Everything on the page.
struct Page<'a> {
    captures: &'a [Capture],
    dir: String,
    commit: String,
    generated: String,
    checks: Loaded<Vec<CheckRow>>,
    perf: Loaded<Perf>,
}

/// Writes the sheet for every PNG in `captures` to `out`, creating `out`'s directory if
/// needed, with the Checks and Performance tables (see the module docs). `bench` is the bench
/// JSON; without it, `<captures>/bench.json` is used if it exists. Returns the number of
/// captures on the sheet.
pub fn write_sheet(captures: &Path, out: &Path, bench: Option<&Path>) -> anyhow::Result<usize> {
    let pngs = list_pngs(captures)?;
    let perf = match bench {
        Some(path) => Loaded::Read(Perf {
            source: path.display().to_string(),
            file: bench::load(path)?,
        }),
        None => {
            let path = captures.join(BENCH_FILE);
            load_optional(&path, |text| {
                serde_json::from_str(text).map_err(|e| e.to_string())
            })
            .map(|file| Perf {
                source: path.display().to_string(),
                file,
            })
        }
    };
    let checks = load_optional(&captures.join(CHECKS_FILE), parse_checks);
    let out_dir = match out.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir,
        _ => Path::new("."),
    };
    fs::create_dir_all(out_dir).with_context(|| format!("cannot create {}", out_dir.display()))?;
    let link_prefix = relative_segments(&canonical(out_dir)?, &canonical(captures)?);
    let entries: Vec<Capture> = pngs
        .iter()
        .map(|png| load_capture(png, &link_prefix))
        .collect();
    let generated = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or_else(|_| "unknown".to_owned(), |t| format_utc(t.as_secs()));
    let html = render(&Page {
        captures: &entries,
        dir: captures.display().to_string(),
        commit: git_commit(),
        generated,
        checks,
        perf,
    });
    fs::write(out, html).with_context(|| format!("cannot write {}", out.display()))?;
    Ok(entries.len())
}

impl<T> Loaded<T> {
    fn map<U>(self, f: impl FnOnce(T) -> U) -> Loaded<U> {
        match self {
            Self::Absent => Loaded::Absent,
            Self::Broken(reason) => Loaded::Broken(reason),
            Self::Read(value) => Loaded::Read(f(value)),
        }
    }
}

/// Reads `path` and parses it with `parse`: absent when the file does not exist, broken (with
/// the reason) when it cannot be read or parsed.
fn load_optional<T>(path: &Path, parse: impl Fn(&str) -> Result<T, String>) -> Loaded<T> {
    match fs::read_to_string(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Loaded::Absent,
        Err(e) => Loaded::Broken(format!("{} unreadable: {e}", path.display())),
        Ok(text) => match parse(&text) {
            Ok(value) => Loaded::Read(value),
            Err(e) => Loaded::Broken(format!("{} unreadable: {e}", path.display())),
        },
    }
}

/// Parses `checks.json`: an array of rows, or an object holding one under `checks`.
fn parse_checks(text: &str) -> Result<Vec<CheckRow>, String> {
    let value: Value = serde_json::from_str(text).map_err(|e| e.to_string())?;
    let rows = match value {
        Value::Object(mut map) => map
            .remove("checks")
            .ok_or("expected an array of checks or an object with a `checks` array")?,
        other => other,
    };
    serde_json::from_value(rows).map_err(|e| e.to_string())
}

fn canonical(path: &Path) -> anyhow::Result<PathBuf> {
    fs::canonicalize(path).with_context(|| format!("cannot resolve {}", path.display()))
}

/// The `*.png` files directly in `dir` (extension in any case), sorted by file name.
pub(crate) fn list_pngs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
    let read_error = || format!("cannot read {}", dir.display());
    let mut pngs = Vec::new();
    for entry in fs::read_dir(dir).with_context(read_error)? {
        let path = entry.with_context(read_error)?.path();
        let is_png = path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("png"));
        if is_png && path.is_file() {
            pngs.push(path);
        }
    }
    pngs.sort();
    Ok(pngs)
}

/// The sheet entry for `png`, whose directory is reached from the sheet by `link_prefix`.
fn load_capture(png: &Path, link_prefix: &[OsString]) -> Capture {
    let file_name = png.file_name().unwrap_or_default();
    let href = link_prefix
        .iter()
        .map(OsString::as_os_str)
        .chain([file_name])
        .map(|segment| encode_segment(segment.as_encoded_bytes()))
        .collect::<Vec<_>>()
        .join("/");
    let json = png.with_extension("json");
    let (metrics, note) = match fs::read_to_string(&json) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            (Metrics::default(), Some("no metrics file".to_owned()))
        }
        Err(e) => (Metrics::default(), Some(format!("metrics unreadable: {e}"))),
        Ok(text) => match serde_json::from_str(&text) {
            Ok(metrics) => (metrics, None),
            Err(e) => (Metrics::default(), Some(format!("metrics unreadable: {e}"))),
        },
    };
    Capture {
        stem: png
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        href,
        metrics,
        note,
    }
}

/// Short hash of the checked-out commit (`git rev-parse --short HEAD` in the current
/// directory), or "unknown" if git or the repository is unavailable.
fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|hash| hash.trim().to_owned())
        .filter(|hash| !hash.is_empty())
        .unwrap_or_else(|| "unknown".to_owned())
}

/// A JSON value as table text: "—" for none or null, strings unquoted, anything else as JSON.
fn text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => MISSING.to_owned(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Shown for a timing with no valid measurement (JSON `null`), never 0 (review I5).
const NOT_MEASURED: &str = "n/a";

/// A timing [ms] to 3 decimals, "n/a" for `null` (not validly measured), "—" when the key is
/// missing.
fn timing_millis(value: Option<&Value>) -> String {
    match value {
        Some(Value::Null) => NOT_MEASURED.to_owned(),
        other => millis(other),
    }
}

/// A duration [ms] to 3 decimals, or the value as it is if it is not a number.
fn millis(value: Option<&Value>) -> String {
    match value.and_then(Value::as_f64) {
        Some(ms) => format!("{ms:.3} ms"),
        None => text(value),
    }
}

/// The table rows (label, value) for one capture: the documented keys in order, missing ones
/// as "—", with one `gpu_ms · <pass>` row per pass; then any other keys.
fn metric_rows(m: &Metrics) -> Vec<(String, String)> {
    let mut rows = vec![
        ("name".to_owned(), text(m.name.as_ref())),
        (
            "size".to_owned(),
            format!("{} × {}", text(m.width.as_ref()), text(m.height.as_ref())),
        ),
    ];
    match &m.gpu_ms {
        Some(Value::Object(passes)) if !passes.is_empty() => {
            for (pass, ms) in passes {
                rows.push((format!("gpu_ms · {pass}"), timing_millis(Some(ms))));
            }
        }
        Some(Value::Object(_)) => rows.push(("gpu_ms".to_owned(), MISSING.to_owned())),
        other => rows.push(("gpu_ms".to_owned(), text(other.as_ref()))),
    }
    rows.push((
        "gpu_frame_ms".to_owned(),
        timing_millis(m.gpu_frame_ms.as_ref()),
    ));
    rows.extend([
        ("cpu_ms".to_owned(), millis(m.cpu_ms.as_ref())),
        ("star_count".to_owned(), text(m.star_count.as_ref())),
        ("lens_scale".to_owned(), text(m.lens_scale.as_ref())),
        ("epoch_utc".to_owned(), text(m.epoch_utc.as_ref())),
    ]);
    rows.extend(v2_rows(m));
    rows.extend(
        m.extra
            .iter()
            .map(|(key, value)| (key.clone(), text(Some(value)))),
    );
    rows
}

/// A summary row lists at most this many names, then counts the rest.
const MAX_NAMES: usize = 12;

/// "a, b, c", at most [`MAX_NAMES`] of them, then "… (+n more)"; "none" for no names.
fn name_list<'a>(names: impl Iterator<Item = &'a str>) -> String {
    let names: Vec<&str> = names.collect();
    if names.is_empty() {
        return "none".to_owned();
    }
    let shown = names[..names.len().min(MAX_NAMES)].join(", ");
    match names.len().saturating_sub(MAX_NAMES) {
        0 => shown,
        more => format!("{shown}, … (+{more} more)"),
    }
}

/// "n thing" or "n things".
fn count(n: usize, thing: &str) -> String {
    if n == 1 {
        format!("1 {thing}")
    } else {
        format!("{n} {thing}s")
    }
}

/// The `name` of each object in a JSON list.
fn entry_names(list: &[Value]) -> impl Iterator<Item = &str> {
    list.iter()
        .filter_map(|entry| entry.get("name").and_then(Value::as_str))
}

/// "n things: a, b, …" for a list of named objects or a map keyed by name.
fn named_entries(value: &Value, thing: &str) -> String {
    match value {
        Value::Array(list) => format!(
            "{}: {}",
            count(list.len(), thing),
            name_list(entry_names(list))
        ),
        Value::Object(map) => format!(
            "{}: {}",
            count(map.len(), thing),
            name_list(map.keys().map(String::as_str))
        ),
        other => text(Some(other)),
    }
}

/// A JSON value as one line: a list of strings joined by `sep`, anything else as [`text`].
fn joined(value: &Value, sep: &str) -> String {
    match value.as_array() {
        Some(list) if list.is_empty() => "none".to_owned(),
        Some(list) => list
            .iter()
            .map(|item| text(Some(item)))
            .collect::<Vec<_>>()
            .join(sep),
        None => text(Some(value)),
    }
}

/// The summary rows of the capture JSON v2 keys present in `m` (see the module docs).
fn v2_rows(m: &Metrics) -> Vec<(String, String)> {
    let mut rows = Vec::new();
    if let Some(v) = &m.ev100 {
        let ev: Option<Vec<f64>> = v
            .as_array()
            .and_then(|list| list.iter().map(Value::as_f64).collect());
        let shown = match ev.as_deref() {
            Some([scene, star]) => format!("scene {scene:.2} · star {star:.2}"),
            _ => text(Some(v)),
        };
        rows.push(("ev100".to_owned(), shown));
    }
    if let Some(v) = &m.honesty {
        rows.push(("honesty".to_owned(), joined(v, "; ")));
    }
    // One row per key of an object; a single row for anything else.
    let mut per_key =
        |label: &str, value: &Value, show: &dyn Fn(&str, &Value) -> String| match value.as_object()
        {
            Some(map) => rows.extend(
                map.iter()
                    .map(|(key, v)| (format!("{label} · {key}"), show(key, v))),
            ),
            None => rows.push((label.to_owned(), text(Some(value)))),
        };
    if let Some(v) = &m.metrics {
        per_key("metrics", v, &|key, v| {
            if key.ends_with("_ms") {
                timing_millis(Some(v))
            } else {
                text(Some(v))
            }
        });
    }
    if let Some(v) = &m.hud {
        per_key("hud", v, &|_, v| joined(v, ", "));
    }
    if let Some(v) = &m.hdr {
        let size = format!("{} × {}", text(v.get("width")), text(v.get("height")));
        let shown = match v.get("file") {
            Some(file) => format!("{} ({size})", text(Some(file))),
            None => text(Some(v)),
        };
        rows.push(("hdr".to_owned(), shown));
    }
    for (label, value, thing) in [
        ("projected", &m.projected, "item"),
        ("regions", &m.regions, "region"),
        ("features", &m.features, "feature"),
        ("flux", &m.flux, "star"),
    ] {
        if let Some(v) = value {
            rows.push((label.to_owned(), named_entries(v, thing)));
        }
    }
    if let Some(v) = &m.samples {
        let shown = match v.as_array() {
            Some(list) => count(list.len(), "point"),
            None => text(Some(v)),
        };
        rows.push(("samples".to_owned(), shown));
    }
    rows
}

/// Escapes text for HTML element content and quoted attribute values.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

/// Inline style sheet: dark page, responsive grid of capture cards.
const STYLE: &str = "
:root { color-scheme: dark; --bg: #0d0f12; --card: #161a20; --line: #262c35;
  --text: #d8dde3; --dim: #8a94a1; --accent: #7fb2ff; --warn: #e0b45c;
  --ok: #7fd18b; --bad: #ff7b72; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
header { max-width: 1600px; margin: 0 auto 20px; }
h1 { margin: 0 0 4px; font-size: 20px; font-weight: 600; }
.meta { margin: 0; color: var(--dim); }
code, td { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
.grid { max-width: 1600px; margin: 0 auto; display: grid; gap: 16px;
  grid-template-columns: repeat(auto-fill, minmax(min(100%, 360px), 1fr)); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  overflow: hidden; }
.card a { display: block; background: #000; }
.card img { display: block; width: 100%; height: auto; }
.card h2 { margin: 10px 12px 6px; font-size: 15px; font-weight: 600; overflow-wrap: anywhere; }
.card h2 a { display: inline; background: none; color: var(--accent); text-decoration: none; }
.card h2 a:hover { text-decoration: underline; }
.note { margin: 0 12px 6px; color: var(--warn); }
table { width: 100%; border-collapse: collapse; margin-bottom: 8px; }
th, td { padding: 3px 12px; border-top: 1px solid var(--line); vertical-align: top; }
th { text-align: left; font-weight: 400; color: var(--dim); white-space: nowrap; }
td { text-align: right; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.empty { max-width: 1600px; margin: 0 auto; color: var(--warn); }
.panel { max-width: 1600px; margin: 0 auto 24px; }
.panel h2 { margin: 0 0 4px; font-size: 16px; font-weight: 600; }
.panel .meta, .panel .note { margin: 0 0 8px; }
.scroll { overflow-x: auto; background: var(--card); border: 1px solid var(--line);
  border-radius: 8px; }
.wide { margin: 0; }
.wide thead th { border-top: none; }
.wide td { text-align: left; }
.wide td.num { text-align: right; }
.pass { color: var(--ok); }
.fail { color: var(--bad); font-weight: 600; }
";

/// The whole page.
fn render(page: &Page) -> String {
    let mut html = String::new();
    let _ = write!(
        html,
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>Capture sheet</title>\n<style>{STYLE}</style>\n</head>\n<body>\n\
         <header>\n<h1>Capture sheet</h1>\n\
         <p class=\"meta\"><code>{dir}</code> · {count} · commit <code>{commit}</code> · \
         generated {generated}</p>\n</header>\n",
        dir = escape(&page.dir),
        count = count(page.captures.len(), "capture"),
        commit = escape(&page.commit),
        generated = escape(&page.generated),
    );
    match &page.checks {
        Loaded::Absent => {}
        Loaded::Broken(reason) => render_note(&mut html, "Checks", reason),
        Loaded::Read(rows) => render_checks(&mut html, rows),
    }
    match &page.perf {
        Loaded::Absent => {}
        Loaded::Broken(reason) => render_note(&mut html, "Performance", reason),
        Loaded::Read(perf) => render_performance(&mut html, perf),
    }
    if page.captures.is_empty() {
        html.push_str("<p class=\"empty\">No PNG files in this directory.</p>\n");
    } else {
        html.push_str("<main class=\"grid\">\n");
        for capture in page.captures {
            render_card(&mut html, capture);
        }
        html.push_str("</main>\n");
    }
    html.push_str("</body>\n</html>\n");
    html
}

/// A section whose file could not be read: its title and the reason.
fn render_note(html: &mut String, title: &str, reason: &str) {
    let _ = write!(
        html,
        "<section class=\"panel\">\n<h2>{}</h2>\n<p class=\"note\">{}</p>\n</section>\n",
        escape(title),
        escape(reason)
    );
}

/// PASS or FAIL (with its class) for a check's `pass`; anything else as [`text`], unclassed.
fn verdict(pass: Option<&Value>) -> (String, &'static str) {
    match pass {
        Some(Value::Bool(true)) => ("PASS".to_owned(), "pass"),
        Some(Value::Bool(false)) => ("FAIL".to_owned(), "fail"),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("pass") => ("PASS".to_owned(), "pass"),
        Some(Value::String(s)) if s.eq_ignore_ascii_case("fail") => ("FAIL".to_owned(), "fail"),
        other => (text(other), ""),
    }
}

/// The Checks table: one row per check, in file order, with a count of each verdict.
fn render_checks(html: &mut String, rows: &[CheckRow]) {
    let verdicts: Vec<(String, &str)> = rows.iter().map(|r| verdict(r.pass.as_ref())).collect();
    let passed = verdicts.iter().filter(|(_, c)| *c == "pass").count();
    let failed = verdicts.iter().filter(|(_, c)| *c == "fail").count();
    let other = rows.len() - passed - failed;
    let mut summary = format!(
        "<code>{CHECKS_FILE}</code> · {} · {passed} PASS · {failed} FAIL",
        count(rows.len(), "check")
    );
    if other > 0 {
        let _ = write!(summary, " · {other} without a verdict");
    }
    let _ = write!(
        html,
        "<section class=\"panel\">\n<h2>Checks</h2>\n<p class=\"meta\">{summary}</p>\n\
         <div class=\"scroll\"><table class=\"wide\">\n<thead><tr><th>row</th><th>bookmark</th>\
         <th>check</th><th>measured</th><th>expected</th><th>result</th></tr></thead>\n<tbody>\n"
    );
    for (row, (word, class)) in rows.iter().zip(&verdicts) {
        let _ = writeln!(
            html,
            "<tr><td class=\"num\">{}</td><td>{}</td><td>{}</td><td class=\"num\">{}</td>\
             <td class=\"num\">{}</td><td class=\"{class}\">{}</td></tr>",
            escape(&text(row.row.as_ref())),
            escape(&text(row.bookmark.as_ref())),
            escape(&text(row.check.as_ref())),
            escape(&text(row.measured.as_ref().or(row.output.as_ref()))),
            escape(&text(row.expected.as_ref())),
            escape(word),
        );
    }
    html.push_str("</tbody>\n</table></div>\n</section>\n");
}

/// One p95 cell [ms] of `window`: "—" without the window, "n/a" without a valid timing, and
/// marked when over `gate` [ms] (last 10 s only).
fn p95_cell(
    window: Option<&BenchWindow>,
    timing: fn(&BenchWindow) -> Option<&Stats>,
    gate: Option<f64>,
) -> String {
    let Some(window) = window else {
        return format!("<td class=\"num\">{MISSING}</td>");
    };
    match timing(window).and_then(|s| s.p95).filter(|p| p.is_finite()) {
        None => format!("<td class=\"num\">{NOT_MEASURED}</td>"),
        Some(p95) => {
            let class = if gate.is_some_and(|g| p95 > g) {
                "num fail"
            } else {
                "num"
            };
            format!("<td class=\"{class}\">{p95:.3}</td>")
        }
    }
}

/// The mean frame rate of `window` (1000 / mean frame interval [ms]), as a cell.
fn fps_cell(window: Option<&BenchWindow>) -> String {
    let fps = match window {
        None => MISSING.to_owned(),
        Some(w) => match w
            .interval_ms
            .and_then(|s| s.mean)
            .filter(|m| *m > 0.0 && m.is_finite())
        {
            Some(mean) => format!("{:.1}", 1000.0 / mean),
            None => NOT_MEASURED.to_owned(),
        },
    };
    format!("<td class=\"num\">{fps}</td>")
}

/// The passes (and CPU sub-timings) over 1.25 × their budget in `window`.
fn budget_cell(window: Option<&BenchWindow>) -> String {
    let shown = match window {
        None => MISSING.to_owned(),
        Some(w) if w.passes.is_empty() => "no pass timings".to_owned(),
        Some(w) => {
            let over: Vec<String> = bench::budget_lines(w)
                .into_iter()
                .filter(|(_, flagged)| *flagged)
                .map(|(line, _)| line.trim().to_owned())
                .collect();
            if over.is_empty() {
                "none".to_owned()
            } else {
                over.join("; ")
            }
        }
    };
    let class = if shown.contains("OVER BUDGET") {
        " class=\"fail\""
    } else {
        ""
    };
    format!("<td{class}>{}</td>", escape(&shown))
}

/// The Performance table: per bookmark, the first and last 10 s side by side.
fn render_performance(html: &mut String, perf: &Perf) {
    let file = &perf.file;
    let mut meta = vec![format!("<code>{}</code>", escape(&perf.source))];
    if let Some(size) = &file.size {
        let shown = match size.as_array().map(Vec::as_slice) {
            Some([w, h]) => format!("{} × {}", text(Some(w)), text(Some(h))),
            _ => text(Some(size)),
        };
        meta.push(format!("internal size {}", escape(&shown)));
    }
    if let Some(s) = file.sustained_s.as_ref().filter(|s| !s.is_null()) {
        meta.push(format!(
            "sustained {} s per bookmark",
            escape(&text(Some(s)))
        ));
    }
    if let Some(f) = file.frames.as_ref().filter(|f| !f.is_null()) {
        meta.push(format!("{} frames per bookmark", escape(&text(Some(f)))));
    }
    let gate = bench::GATE;
    meta.push(format!(
        "p95 per window [ms]; the FPS gate reads the last 10 s: GPU frame ≤ {:.1} ms, CPU ≤ \
         {:.1} ms, frame interval ≤ {:.1} ms (marked when over)",
        gate.max_gpu_ms, gate.max_cpu_ms, gate.max_interval_ms
    ));
    let windows = "<th>first 10 s</th><th>last 10 s</th>".repeat(4);
    let _ = write!(
        html,
        "<section class=\"panel\">\n<h2>Performance</h2>\n<p class=\"meta\">{}</p>\n\
         <div class=\"scroll\"><table class=\"wide\">\n<thead>\n<tr><th rowspan=\"2\">bookmark</th>\
         <th rowspan=\"2\">lens scale</th><th colspan=\"2\">GPU frame p95</th>\
         <th colspan=\"2\">CPU p95</th><th colspan=\"2\">frame interval p95</th>\
         <th colspan=\"2\">mean fps</th><th rowspan=\"2\">over 1.25 × budget (last 10 s)</th></tr>\n\
         <tr>{windows}</tr>\n</thead>\n<tbody>\n",
        meta.join(" · ")
    );
    let timings: [(fn(&BenchWindow) -> Option<&Stats>, f64); 3] = [
        (|w| w.gpu_frame_ms.as_ref(), gate.max_gpu_ms),
        (|w| w.cpu_ms.as_ref(), gate.max_cpu_ms),
        (|w| w.interval_ms.as_ref(), gate.max_interval_ms),
    ];
    for bookmark in &file.bookmarks {
        let first = bookmark.window(Window::First10s);
        let last = bookmark.window(Window::Last10s);
        let _ = write!(
            html,
            "<tr><td>{}</td><td class=\"num\">{}</td>",
            escape(&bookmark.name),
            escape(&text(bookmark.lens_scale.as_ref()))
        );
        for (timing, limit) in timings {
            html.push_str(&p95_cell(first, timing, None));
            html.push_str(&p95_cell(last, timing, Some(limit)));
        }
        html.push_str(&fps_cell(first));
        html.push_str(&fps_cell(last));
        html.push_str(&budget_cell(last));
        html.push_str("</tr>\n");
    }
    html.push_str("</tbody>\n</table></div>\n</section>\n");
}

/// One capture's card: thumbnail linking to the full PNG, name, note, metrics table.
fn render_card(html: &mut String, capture: &Capture) {
    let href = escape(&capture.href);
    let stem = escape(&capture.stem);
    let _ = write!(
        html,
        "<article class=\"card\">\n<a href=\"{href}\"><img src=\"{href}\" alt=\"{stem}\" \
         loading=\"lazy\"></a>\n<h2><a href=\"{href}\">{stem}</a></h2>\n"
    );
    if let Some(note) = &capture.note {
        let _ = writeln!(html, "<p class=\"note\">{}</p>", escape(note));
    }
    html.push_str("<table>\n");
    for (label, value) in metric_rows(&capture.metrics) {
        let _ = writeln!(
            html,
            "<tr><th>{}</th><td>{}</td></tr>",
            escape(&label),
            escape(&value)
        );
    }
    html.push_str("</table>\n</article>\n");
}

/// Path segments leading from directory `from` to directory `to` (both absolute and
/// canonical): a ".." for each of `from`'s components past the common prefix, then the rest
/// of `to`.
fn relative_segments(from: &Path, to: &Path) -> Vec<OsString> {
    let from: Vec<_> = from.components().collect();
    let to: Vec<_> = to.components().collect();
    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let up = from[common..].iter().map(|_| OsString::from(".."));
    let down = to[common..].iter().map(|c| c.as_os_str().to_owned());
    up.chain(down).collect()
}

/// Percent-encodes one URL path segment: every byte except the unreserved characters
/// A–Z a–z 0–9 - . _ ~ (RFC 3986 §2.3) becomes %XX.
fn encode_segment(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len());
    for &b in bytes {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

/// `YYYY-MM-DD HH:MM:SS UTC` for a Unix time [s] (proleptic Gregorian calendar, no leap
/// seconds, as Unix time counts).
///
/// Days to date: H. Hinnant, "chrono-Compatible Low-Level Date Algorithms",
/// `civil_from_days` (howardhinnant.github.io/date_algorithms.html), in unsigned form since
/// Unix times here are never before 1970.
fn format_utc(unix_s: u64) -> String {
    let (days, secs) = (unix_s / 86_400, unix_s % 86_400);
    // Shift the epoch to 0000-03-01 so each 400-year era starts just after a leap day.
    let z = days + 719_468;
    let era = z / 146_097;
    let day_of_era = z - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_from_march = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_from_march + 2) / 5 + 1;
    let month = if month_from_march < 10 {
        month_from_march + 3
    } else {
        month_from_march - 9
    };
    let year = era * 400 + year_of_era + u64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} UTC",
        secs / 3_600,
        secs % 3_600 / 60,
        secs % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use std::fs;
    use std::path::PathBuf;

    fn rows_of(json: &str) -> Vec<(String, String)> {
        metric_rows(&serde_json::from_str(json).expect("test JSON parses"))
    }

    fn row<'a>(rows: &'a [(String, String)], label: &str) -> &'a str {
        rows.iter()
            .find(|(l, _)| l == label)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("no row {label:?} in {rows:?}"))
    }

    #[test]
    fn metric_rows_show_every_documented_key() {
        let rows = rows_of(
            r#"{"name": "bh-shadow-30M", "width": 1920, "height": 1080,
                "gpu_ms": {"lens": 3.25, "stars": 0.5}, "cpu_ms": 1.0, "star_count": 9096,
                "lens_scale": 1.0, "epoch_utc": "2026-11-20T00:00:00Z"}"#,
        );
        assert_eq!(row(&rows, "name"), "bh-shadow-30M");
        assert_eq!(row(&rows, "size"), "1920 × 1080");
        assert_eq!(row(&rows, "gpu_ms · lens"), "3.250 ms");
        assert_eq!(row(&rows, "gpu_ms · stars"), "0.500 ms");
        assert_eq!(row(&rows, "cpu_ms"), "1.000 ms");
        assert_eq!(row(&rows, "star_count"), "9096");
        assert_eq!(row(&rows, "lens_scale"), "1.0");
        assert_eq!(row(&rows, "epoch_utc"), "2026-11-20T00:00:00Z");
        // Written before the whole-frame time existed: a dash.
        assert_eq!(row(&rows, "gpu_frame_ms"), MISSING);
    }

    /// Review I5: a GPU time the device did not measure validly is `null` in the JSON and
    /// "n/a" on the sheet, never 0.
    #[test]
    fn null_gpu_times_show_as_na() {
        let rows = rows_of(r#"{"gpu_ms": {"lens": 140.5, "tonemap": null}, "gpu_frame_ms": null}"#);
        assert_eq!(row(&rows, "gpu_ms · lens"), "140.500 ms");
        assert_eq!(row(&rows, "gpu_ms · tonemap"), "n/a");
        assert_eq!(row(&rows, "gpu_frame_ms"), "n/a");
        let rows = rows_of(r#"{"gpu_frame_ms": 16.25}"#);
        assert_eq!(row(&rows, "gpu_frame_ms"), "16.250 ms");
    }

    #[test]
    fn metric_rows_dash_missing_keys_and_keep_extra_ones() {
        let rows = rows_of(r#"{"width": 640, "gpu_ms": {}, "lens_scale": null, "seed": 7}"#);
        for label in [
            "name",
            "gpu_ms",
            "cpu_ms",
            "star_count",
            "lens_scale",
            "epoch_utc",
        ] {
            assert_eq!(row(&rows, label), "—", "{label}");
        }
        assert_eq!(row(&rows, "size"), "640 × —");
        assert_eq!(row(&rows, "seed"), "7");
        // Unexpected types are shown as they are, not dropped.
        let rows = rows_of(r#"{"cpu_ms": "slow", "gpu_ms": 4}"#);
        assert_eq!(row(&rows, "cpu_ms"), "slow");
        assert_eq!(row(&rows, "gpu_ms"), "4");
    }

    #[test]
    fn link_segments_are_percent_encoded() {
        assert_eq!(
            encode_segment(b"bh-shadow_30M.v2~.png"),
            "bh-shadow_30M.v2~.png"
        );
        assert_eq!(encode_segment(b"a b#c?.png"), "a%20b%23c%3F.png");
        assert_eq!(encode_segment("é".as_bytes()), "%C3%A9");
    }

    #[test]
    fn relative_segments_between_directories() {
        let rel = |a: &str, b: &str| relative_segments(Path::new(a), Path::new(b));
        assert!(rel("/r/captures/m0", "/r/captures/m0").is_empty());
        assert_eq!(rel("/r/sheets", "/r/captures/m0"), ["..", "captures", "m0"]);
        assert_eq!(rel("/r/captures/m0/html", "/r/captures/m0"), [".."]);
    }

    #[test]
    fn formats_utc_dates() {
        assert_eq!(format_utc(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(format_utc(951_782_400), "2000-02-29 00:00:00 UTC");
        assert_eq!(format_utc(1_795_176_245), "2026-11-20 12:04:05 UTC");
    }

    /// A fresh, empty directory under the system temp dir, unique to this process and test.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uni-verify-{}-{test}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn writes_sheet_for_a_capture_directory() {
        let root = temp_dir("writes_sheet");
        let caps = root.join("captures");
        fs::create_dir_all(&caps).expect("create captures dir");
        for name in ["bh-shadow-30M", "no-json", "bad-json"] {
            RgbImage::from_pixel(4, 2, Rgb([9, 9, 9]))
                .save(caps.join(format!("{name}.png")))
                .expect("write PNG");
        }
        fs::write(
            caps.join("bh-shadow-30M.json"),
            r#"{"name": "bh-shadow-30M", "width": 4, "height": 2, "gpu_ms": {"lens": 1.25},
                "note": "<b>&</b>"}"#,
        )
        .expect("write JSON");
        fs::write(caps.join("bad-json.json"), "{not json").expect("write JSON");
        fs::write(caps.join("readme.txt"), "not a capture").expect("write text");

        let out = caps.join("index.html");
        let count = write_sheet(&caps, &out, None).expect("sheet");
        let sibling = root.join("sheets").join("m0.html");
        let sibling_count = write_sheet(&caps, &sibling, None).expect("sheet in another dir");
        let html = fs::read_to_string(&out).expect("read sheet");
        let sibling_html = fs::read_to_string(&sibling).expect("read sheet");
        let _ = fs::remove_dir_all(&root);

        assert_eq!((count, sibling_count), (3, 3));
        assert!(html.starts_with("<!doctype html>"));
        // Sorted by file name, each linking its PNG relative to the sheet.
        let order: Vec<_> = ["bad-json", "bh-shadow-30M", "no-json"]
            .iter()
            .map(|n| html.find(&format!(r#"<a href="{n}.png""#)).expect("link"))
            .collect();
        assert!(order.is_sorted(), "{order:?}");
        assert!(sibling_html.contains(r#"<a href="../captures/bh-shadow-30M.png""#));
        assert!(html.contains(r#"<img src="bh-shadow-30M.png""#));
        // Metrics, notes for missing and broken JSON, escaping.
        assert!(html.contains("1.250 ms"));
        assert!(html.contains("no metrics file"));
        assert!(html.contains("metrics unreadable"));
        assert!(html.contains("&lt;b&gt;&amp;&lt;/b&gt;"));
        assert!(!html.contains("<b>&</b>"));
        assert!(!html.contains("readme"));
        // Provenance: a short hash (or "unknown" without git) and "YYYY-MM-DD HH:MM:SS UTC".
        let after = |marker: &str, end: &str| {
            let rest = html.split(marker).nth(1).expect(marker);
            rest.split(end).next().unwrap_or_default().to_owned()
        };
        let commit = after("commit <code>", "</code>");
        assert!(
            commit == "unknown"
                || (commit.len() >= 4 && commit.chars().all(|c| c.is_ascii_hexdigit())),
            "commit {commit:?}"
        );
        let generated = after("generated ", "</p>");
        assert!(
            generated.len() == 23 && generated.ends_with(" UTC"),
            "generated {generated:?}"
        );
    }

    /// `uni-verify gate` writes `{row, bookmark, check, pass, output}`: the row and the output
    /// line are shown (as the measured value), with each verdict.
    #[test]
    fn sheet_shows_gate_checks() {
        let caps = temp_dir("sheet_shows_gate_checks");
        RgbImage::from_pixel(4, 2, Rgb([9, 9, 9]))
            .save(caps.join("sun-closeup.png"))
            .expect("write PNG");
        fs::write(
            caps.join("checks.json"),
            r#"[{"row": 11, "bookmark": "sun-closeup", "check": "JSON ev100 within limits",
                 "pass": true, "output": "ev100 [33.97, -6.5] within [-30, 40], no NaN PASS"},
                {"row": 27, "bookmark": "bh-lens-seam", "check": "ratio seam",
                 "pass": false, "output": "error: cannot read bh-lens-seam.hdr.bin FAIL"}]"#,
        )
        .expect("write checks");
        let out = caps.join("index.html");
        write_sheet(&caps, &out, None).expect("sheet");
        let html = fs::read_to_string(&out).expect("read sheet");
        let _ = fs::remove_dir_all(&caps);
        assert!(html.contains("<th>row</th>"), "{html}");
        for text in [
            "<td class=\"num\">11</td>",
            "<td class=\"num\">27</td>",
            "ev100 [33.97, -6.5] within [-30, 40], no NaN PASS",
            "error: cannot read bh-lens-seam.hdr.bin FAIL",
            "1 PASS · 1 FAIL",
        ] {
            assert!(html.contains(text), "missing {text:?}");
        }
    }

    #[test]
    fn sheet_includes_checks_and_perf() {
        let caps = temp_dir("sheet_includes_checks_and_perf");
        RgbImage::from_pixel(4, 2, Rgb([9, 9, 9]))
            .save(caps.join("sky-orion-3d.png"))
            .expect("write PNG");
        // A capture JSON v2 (B9 Step 2): the bulky keys are summarised, not dumped.
        fs::write(
            caps.join("sky-orion-3d.json"),
            r#"{"name": "sky-orion-3d", "width": 4, "height": 2, "gpu_ms": {"stars": 0.25},
                "gpu_frame_ms": 1.5, "ev100": [-4.0, -4.0],
                "honesty": ["Eye-adapted stars (dual exposure, non-physical)"],
                "metrics": {"star_count_drawn": 331842, "lit_px": 12},
                "projected": [{"name": "Rigel", "x": 1.0, "y": 1.0, "radius_px": 0.5}],
                "regions": {"band_plane": [[0, 0], [1, 0], [2, 0], [3, 0]]},
                "hdr": {"file": "sky-orion-3d.hdr.bin", "width": 4, "height": 2}}"#,
        )
        .expect("write JSON");
        fs::write(
            caps.join("checks.json"),
            r#"[{"bookmark": "sky-orion-3d", "check": "points Rigel", "measured": "0.31 px",
                 "expected": "≤ 1 px", "pass": true},
                {"bookmark": "sky-orion-3d", "check": "flux Rigel", "measured": 1.071,
                 "expected": "1 ± 0.05", "pass": false}]"#,
        )
        .expect("write checks");
        let window = |gpu: &str, cpu: f64, interval: f64| {
            format!(
                r#"{{"frame_ms": {{"mean": 15.0, "p95": 15.5, "count": 600}},
                    "interval_ms": {{"mean": 16.6, "p95": {interval}, "count": 600}},
                    "gpu_frame_ms": {gpu},
                    "cpu_ms": {{"mean": 2.0, "p95": {cpu}, "count": 600}},
                    "passes": {{"bodies": {{"mean": 1.25, "valid": 600, "runs": 600}}}}}}"#
            )
        };
        let bench = format!(
            r#"{{"size": [1920, 1080], "sustained_s": 60, "bookmarks": [
                {{"name": "start", "lens_scale": 0.5, "windows": {{
                    "first10s": {a}, "last10s": {b}}}}},
                {{"name": "mars-reference-angle", "lens_scale": 0.5, "windows": {{
                    "first10s": {c}, "last10s": {d}}}}}]}}"#,
            a = window(r#"{"mean": 10.0, "p95": 11.125, "count": 600}"#, 2.5, 16.7),
            b = window(r#"{"mean": 13.0, "p95": 14.375, "count": 600}"#, 3.25, 16.8),
            c = window("null", 2.0, 16.6),
            d = window(r#"{"mean": 15.0, "p95": 15.875, "count": 600}"#, 3.5, 16.9),
        );
        fs::write(caps.join("bench.json"), bench).expect("write bench");

        let out = caps.join("index.html");
        let count = write_sheet(&caps, &out, None).expect("sheet");
        let html = fs::read_to_string(&out).expect("read sheet");
        // An explicit bench file elsewhere wins over the default.
        let other = caps.join("other-bench.json");
        fs::write(
            &other,
            r#"{"bookmarks": [{"name": "earth-limb-dawn", "windows": {}}]}"#,
        )
        .expect("write bench");
        let explicit = caps.join("explicit.html");
        write_sheet(&caps, &explicit, Some(&other)).expect("sheet");
        let explicit_html = fs::read_to_string(&explicit).expect("read sheet");
        // An explicit bench file that cannot be read is an error; a broken checks.json is a note.
        let absent_bench = write_sheet(&caps, &explicit, Some(&caps.join("absent.json")));
        fs::write(caps.join("checks.json"), "{not json").expect("write checks");
        write_sheet(&caps, &explicit, Some(&other)).expect("sheet");
        let broken_html = fs::read_to_string(&explicit).expect("read sheet");
        let _ = fs::remove_dir_all(&caps);

        assert_eq!(count, 1);
        // Checks: every row, with its verdict.
        assert!(html.contains("Checks"), "{html}");
        for text in [
            "points Rigel",
            "0.31 px",
            "≤ 1 px",
            "flux Rigel",
            "1.071",
            "1 ± 0.05",
            "PASS",
            "FAIL",
        ] {
            assert!(html.contains(&escape(text)), "missing {text:?}");
        }
        // Performance: every bookmark, first and last 10 s side by side, n/a for null.
        assert!(html.contains("Performance"), "{html}");
        for text in [
            "start",
            "mars-reference-angle",
            "first 10 s",
            "last 10 s",
            "11.125",
            "14.375",
            "15.875",
            "n/a",
        ] {
            assert!(html.contains(text), "missing {text:?}");
        }
        // Verdict counts; last-10-s cells over the plan's gate are marked (interval 16.8 and
        // 16.9 > 16.7 ms), first-10-s cells are not (16.7 is within it anyway).
        assert!(html.contains("1 PASS · 1 FAIL"), "check counts");
        assert!(
            html.contains(r#"<td class="num fail">16.800</td>"#),
            "gate mark"
        );
        assert!(
            html.contains(r#"<td class="num">16.700</td>"#),
            "first 10 s unmarked"
        );
        assert!(
            html.contains("GPU frame ≤ 16.0 ms, CPU ≤ 4.0 ms"),
            "gate caption"
        );
        // bodies 1.25 ms is within 1.25 × 1.5 ms: nothing over budget.
        assert!(html.contains("<td>none</td>"), "budget column");
        assert!(explicit_html.contains("earth-limb-dawn"));
        assert!(!explicit_html.contains("mars-reference-angle"));
        assert!(absent_bench.is_err(), "unreadable --bench");
        assert!(
            broken_html.contains("checks.json unreadable"),
            "broken checks note"
        );
        // The capture card summarises the v2 keys instead of dumping the region samples.
        assert!(html.contains("331842"), "metrics row");
        assert!(html.contains("Eye-adapted stars"), "honesty row");
        assert!(!html.contains("[[0,0],[1,0]"), "region samples dumped");
    }
}
