//! Builds a self-contained HTML contact sheet of captures and their JSON metrics (M0 Task 12).
//!
//! `uni-verify sheet --captures <dir> --out <dir>/index.html` lists every `*.png` in the
//! captures directory, sorted by file name, as a thumbnail that links to the full image and a
//! table of its metrics, under the git commit and the time the sheet was made. The page has
//! inline styles and no scripts; images are linked relative to the sheet, so the sheet and the
//! captures move together.
//!
//! Task 11 writes `<name>.png` with `<name>.json` beside it:
//! `{name, width, height, gpu_ms: {pass: ms}, cpu_ms, star_count, lens_scale, epoch_utc}`.
//! The JSON is read tolerantly: a missing, null or empty value shows as "—", a value of an
//! unexpected type is shown as it is, and unknown keys get rows of their own after the known
//! ones. A missing or unparsable JSON file gives a note and a table of dashes.

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
    /// GPU time per render pass [ms], keyed by pass name.
    gpu_ms: Option<Value>,
    cpu_ms: Option<Value>,
    star_count: Option<Value>,
    lens_scale: Option<Value>,
    epoch_utc: Option<Value>,
    /// Keys the capture format does not (yet) document.
    #[serde(flatten)]
    extra: Map<String, Value>,
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

/// Writes the sheet for every PNG in `captures` to `out`, creating `out`'s directory if
/// needed. Returns the number of captures on the sheet.
pub fn write_sheet(captures: &Path, out: &Path) -> anyhow::Result<usize> {
    let pngs = list_pngs(captures)?;
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
    let html = render(
        &entries,
        &captures.display().to_string(),
        &git_commit(),
        &generated,
    );
    fs::write(out, html).with_context(|| format!("cannot write {}", out.display()))?;
    Ok(entries.len())
}

fn canonical(path: &Path) -> anyhow::Result<PathBuf> {
    fs::canonicalize(path).with_context(|| format!("cannot resolve {}", path.display()))
}

/// The `*.png` files directly in `dir` (extension in any case), sorted by file name.
fn list_pngs(dir: &Path) -> anyhow::Result<Vec<PathBuf>> {
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
                rows.push((format!("gpu_ms · {pass}"), millis(Some(ms))));
            }
        }
        Some(Value::Object(_)) => rows.push(("gpu_ms".to_owned(), MISSING.to_owned())),
        other => rows.push(("gpu_ms".to_owned(), text(other.as_ref()))),
    }
    rows.extend([
        ("cpu_ms".to_owned(), millis(m.cpu_ms.as_ref())),
        ("star_count".to_owned(), text(m.star_count.as_ref())),
        ("lens_scale".to_owned(), text(m.lens_scale.as_ref())),
        ("epoch_utc".to_owned(), text(m.epoch_utc.as_ref())),
    ]);
    rows.extend(
        m.extra
            .iter()
            .map(|(key, value)| (key.clone(), text(Some(value)))),
    );
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
  --text: #d8dde3; --dim: #8a94a1; --accent: #7fb2ff; --warn: #e0b45c; }
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
";

/// The whole page.
fn render(captures: &[Capture], dir: &str, commit: &str, generated: &str) -> String {
    let mut html = String::new();
    let count = match captures.len() {
        1 => "1 capture".to_owned(),
        n => format!("{n} captures"),
    };
    let _ = write!(
        html,
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n\
         <title>Capture sheet</title>\n<style>{STYLE}</style>\n</head>\n<body>\n\
         <header>\n<h1>Capture sheet</h1>\n\
         <p class=\"meta\"><code>{dir}</code> · {count} · commit <code>{commit}</code> · \
         generated {generated}</p>\n</header>\n",
        dir = escape(dir),
        commit = escape(commit),
        generated = escape(generated),
    );
    if captures.is_empty() {
        html.push_str("<p class=\"empty\">No PNG files in this directory.</p>\n");
    } else {
        html.push_str("<main class=\"grid\">\n");
        for capture in captures {
            render_card(&mut html, capture);
        }
        html.push_str("</main>\n");
    }
    html.push_str("</body>\n</html>\n");
    html
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
        let count = write_sheet(&caps, &out).expect("sheet");
        let sibling = root.join("sheets").join("m0.html");
        let sibling_count = write_sheet(&caps, &sibling).expect("sheet in another dir");
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
}
