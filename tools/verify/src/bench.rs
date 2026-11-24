//! Checks `universe --bench --json` timings against the frame limits and the per-pass budgets
//! (M1 Task B8).
//!
//! `uni-verify bench --json <f> --max-gpu-ms <ms> --max-cpu-ms <ms> --max-interval-ms <ms>
//! [--window last10s|first10s|all] [--budgets]`
//!
//! The bench JSON is B9 Step 3's: `{size, frames, sustained_s, bookmarks: [{name, lens_scale,
//! windows: {all|first10s|last10s: {frame_ms, interval_ms, gpu_frame_ms, cpu_ms:
//! {mean, p95, count} | null, passes: {pass: {mean, valid, runs}}, snapshot_cpu_ms,
//! comets_cpu_ms}}}]}`. For each bookmark, the chosen window's (default `last10s`, the FPS gate's
//! window: plan "Performance budget", Gate rule) p95 GPU frame, p95 CPU and p95 frame interval
//! must be within the limits. A `null` timing has no valid measurement (CLAUDE.md §2, Timing)
//! and fails with the reason: "n/a" never passes.
//!
//! `--budgets` also flags every pass whose mean in the window exceeds 1.25 × its budget in the
//! plan's per-pass table ([`PASS_BUDGETS_MS`], compiled in), and the two CPU sub-budgets
//! ([`CPU_BUDGETS_MS`]). A feature timed as several passes is budgeted as their sum
//! ([`FEATURE_PASSES`], B9t: the atmosphere's march, composite and mask; the bloom's pyramid,
//! downsample and upsample in one compute pass, and its composite); its parts are listed. A pass that ran but never timed validly is flagged as unmeasured; one
//! that did not run, or has no budget (the lens, the M0 debug grid), is listed and not flagged.
//! Any flag fails the run, since the gate turns each into a perf-fix task.

use crate::{Verdict, pass_word};
use anyhow::Context;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Which part of a sustained run to judge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    Last10s,
    First10s,
    All,
}

impl Window {
    /// Parses `last10s`, `first10s` or `all`.
    pub fn parse(text: &str) -> Option<Self> {
        match text {
            "last10s" => Some(Self::Last10s),
            "first10s" => Some(Self::First10s),
            "all" => Some(Self::All),
            _ => None,
        }
    }

    /// The window's key in the bench JSON.
    pub fn key(self) -> &'static str {
        match self {
            Self::Last10s => "last10s",
            Self::First10s => "first10s",
            Self::All => "all",
        }
    }
}

/// The frame limits [ms].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    pub max_gpu_ms: f64,
    pub max_cpu_ms: f64,
    pub max_interval_ms: f64,
}

/// The plan's FPS gate on the last 10 s of a sustained run (plan "Performance budget", Gate
/// rule): p95 GPU frame ≤ 16.0 ms, p95 CPU ≤ 4.0 ms, p95 frame interval ≤ 16.7 ms (≥ 60 FPS).
/// The `bench` command takes its limits on the command line; the sheet marks cells over these.
pub const GATE: Limits = Limits {
    max_gpu_ms: 16.0,
    max_cpu_ms: 4.0,
    max_interval_ms: 16.7,
};

/// The `bench` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchArgs {
    pub json: PathBuf,
    pub limits: Limits,
    pub window: Window,
    pub budgets: bool,
}

/// GPU budget per pass at 1920×1080 internal, sustained [ms]: the plan's "Performance budget"
/// table (ruling, review I15), keyed by the passes' timer labels. The lens pass is not budgeted
/// (M0 ground truth). `band-update` is 0 in steady state and ≤ 0.5 ms on a refresh frame; a
/// pass's mean is over the frames it ran, so the refresh-frame figure is its budget here.
pub const PASS_BUDGETS_MS: &[(&str, f64)] = &[
    ("exposure", 0.05),
    ("band-update", 0.5),
    ("sun", 0.5),
    ("bodies", 1.5),
    ("rings", 0.4),
    ("comets", 0.5),
    ("atmosphere", 3.0),
    // Lead ruling (Task C6 context, item 13): 0.3 → 0.6 ms. Every separate pass boundary costs
    // ≈ 0.35–0.45 ms on this GPU (a timestamp-only mark measured 0.42–0.46 ms; progress.md at
    // dc61b36), so the meter's own work is a small part of its time.
    ("histogram", 0.6),
    // Lead ruling "band budget" (Wave B gate): re-baselined 0.3 → 0.6 ms, the measured floor
    // of a full-screen pass on the M3 (≈ 0.34 ms) plus the faithful band's own work (≈ 0.55).
    ("band", 0.6),
    ("stars", 2.5),
    ("points", 0.2),
    ("asteroids", 1.0),
    ("bloom", 1.0),
    ("tonemap", 0.6),
    ("hud", 0.5),
];

/// Features the renderer times as several passes, budgeted together under the first name's
/// budget (B9t): (budget key, the timer labels that make it up). uni-atmo's half-resolution
/// march (task B9p) is "atmosphere" + "atmosphere composite" + "atmosphere mask"; uni-post's
/// bloom is "bloom" (the pyramid: every downsample and upsample dispatch, one compute pass) +
/// "bloom composite". Wave C's two passes are single timed passes with no sub-pass labels (Task
/// C6, item 11): "comets" (C1: nuclei, comae, ion and dust tails in one render pass) and
/// "asteroids" (C2: one indexed draw), budgeted under their own names above.
pub const FEATURE_PASSES: &[(&str, &[&str])] = &[
    (
        "atmosphere",
        &["atmosphere", "atmosphere composite", "atmosphere mask"],
    ),
    ("bloom", &["bloom", "bloom composite"]),
];

/// The feature that pass `name` is a part of, if any.
fn feature_of(name: &str) -> Option<&'static (&'static str, &'static [&'static str])> {
    FEATURE_PASSES
        .iter()
        .find(|(_, parts)| parts.contains(&name))
}

/// CPU sub-budgets per frame [ms] (plan "Performance budget", ruling 3): the World snapshot, the
/// comet tails and the asteroid mean anomalies, keyed by their bench JSON fields.
pub const CPU_BUDGETS_MS: &[(&str, f64)] = &[
    ("snapshot_cpu_ms", 2.0),
    ("comets_cpu_ms", 1.0),
    ("asteroids_cpu_ms", 0.5),
];

/// A pass is flagged when its mean exceeds this many times its budget (plan, Gate rule).
pub const BUDGET_FACTOR: f64 = 1.25;

/// The budget of pass `name` [ms], if it has one.
pub fn budget_ms(name: &str) -> Option<f64> {
    PASS_BUDGETS_MS
        .iter()
        .find(|(pass, _)| *pass == name)
        .map(|(_, ms)| *ms)
}

/// The bench JSON (B9 Step 3), read tolerantly: missing fields are `None` or empty.
#[derive(Debug, Default, Deserialize)]
pub struct BenchFile {
    #[serde(default)]
    pub size: Option<Value>,
    #[serde(default)]
    pub frames: Option<Value>,
    #[serde(default)]
    pub sustained_s: Option<Value>,
    #[serde(default)]
    pub bookmarks: Vec<BenchBookmark>,
}

/// One bench bookmark.
#[derive(Debug, Default, Deserialize)]
pub struct BenchBookmark {
    pub name: String,
    #[serde(default)]
    pub lens_scale: Option<Value>,
    /// `all`, `first10s`, `last10s`; a window may be `null`.
    #[serde(default)]
    pub windows: BTreeMap<String, Option<BenchWindow>>,
}

impl BenchBookmark {
    /// The window `w`, if the JSON has it.
    pub fn window(&self, w: Window) -> Option<&BenchWindow> {
        self.windows.get(w.key()).and_then(Option::as_ref)
    }
}

/// The timings of one window (`frame_ms`, the serial frame time, is not judged and not read).
#[derive(Debug, Default, Deserialize)]
pub struct BenchWindow {
    #[serde(default)]
    pub interval_ms: Option<Stats>,
    #[serde(default)]
    pub gpu_frame_ms: Option<Stats>,
    #[serde(default)]
    pub cpu_ms: Option<Stats>,
    #[serde(default)]
    pub passes: BTreeMap<String, PassStats>,
    /// `{mean, p95, count}` or a plain number [ms].
    #[serde(default)]
    pub snapshot_cpu_ms: Option<Value>,
    /// As `snapshot_cpu_ms`.
    #[serde(default)]
    pub comets_cpu_ms: Option<Value>,
    /// As `snapshot_cpu_ms`.
    #[serde(default)]
    pub asteroids_cpu_ms: Option<Value>,
}

impl BenchWindow {
    /// The mean of a CPU sub-timing given as `{mean, …}` or a plain number [ms].
    fn cpu_mean(&self, field: &str) -> Option<f64> {
        let value = match field {
            "snapshot_cpu_ms" => self.snapshot_cpu_ms.as_ref(),
            "comets_cpu_ms" => self.comets_cpu_ms.as_ref(),
            "asteroids_cpu_ms" => self.asteroids_cpu_ms.as_ref(),
            _ => None,
        }?;
        value
            .as_f64()
            .or_else(|| value.get("mean").and_then(Value::as_f64))
    }
}

/// Mean and 95th percentile of one timing [ms]; `null` values are `None` (the sample `count`
/// is not read).
#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct Stats {
    #[serde(default)]
    pub mean: Option<f64>,
    #[serde(default)]
    pub p95: Option<f64>,
}

/// One pass's GPU time: the mean over its valid measurements [ms], how many were valid and how
/// many frames it ran in.
#[derive(Debug, Default, Clone, Copy, Deserialize)]
pub struct PassStats {
    #[serde(default)]
    pub mean: Option<f64>,
    #[serde(default)]
    pub valid: Option<f64>,
    #[serde(default)]
    pub runs: Option<f64>,
}

/// Reads a bench JSON file.
pub fn load(path: &Path) -> anyhow::Result<BenchFile> {
    let text =
        fs::read_to_string(path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// `<label> p95 <ms> ≤ <max>` or the reason it has none, and whether it is within `max`.
fn limit_part(label: &str, stats: Option<&Stats>, max: f64, missing: &str) -> (String, bool) {
    match stats.and_then(|s| s.p95) {
        Some(p95) if p95.is_finite() => {
            let ok = p95 <= max;
            let sign = if ok { "≤" } else { ">" };
            (format!("{label} p95 {p95:.3} ms {sign} {max} ms"), ok)
        }
        _ => (format!("{label} p95 n/a ({missing})"), false),
    }
}

/// `  <what>: mean <m> ms ≤ 1.25 × <budget> ms = <limit> ms ok` (or `> … OVER BUDGET`), and
/// whether it is flagged.
fn budget_line(what: &str, mean: f64, budget: f64) -> (String, bool) {
    let limit = BUDGET_FACTOR * budget;
    let over = mean > limit;
    let (sign, word) = if over {
        (">", "OVER BUDGET")
    } else {
        ("≤", "ok")
    };
    (
        format!(
            "  {what}: mean {mean:.3} ms {sign} {BUDGET_FACTOR} × {budget} ms = {limit:.4} ms {word}"
        ),
        over,
    )
}

/// The budget lines of one window, each with whether it is flagged.
pub(crate) fn budget_lines(win: &BenchWindow) -> Vec<(String, bool)> {
    let mut lines = Vec::new();
    for (feature, parts) in FEATURE_PASSES {
        let ran: Vec<(&str, &PassStats)> = parts
            .iter()
            .filter_map(|p| win.passes.get(*p).map(|s| (*p, s)))
            .filter(|(_, s)| s.runs.unwrap_or(0.0) > 0.0)
            .collect();
        let Some(budget) = budget_ms(feature) else {
            continue;
        };
        if ran.is_empty() {
            continue;
        }
        let means: Vec<Option<f64>> = ran
            .iter()
            .map(|(_, s)| {
                s.mean
                    .filter(|m| m.is_finite() && s.valid.unwrap_or(0.0) > 0.0)
            })
            .collect();
        let names = ran.iter().map(|(p, _)| *p).collect::<Vec<_>>().join(" + ");
        lines.push(if means.iter().all(Option::is_some) {
            let sum = means.iter().flatten().sum();
            budget_line(&format!("feature {feature} ({names})"), sum, budget)
        } else {
            (
                format!(
                    "  feature {feature} ({names}): a part has no valid timing, budget {budget} \
                     ms: OVER BUDGET (unmeasured)"
                ),
                true,
            )
        });
    }
    for (pass, stats) in &win.passes {
        if let Some((feature, _)) = feature_of(pass) {
            let mean = stats.mean.filter(|m| m.is_finite());
            lines.push((
                format!(
                    "  pass {pass}: mean {} (part of feature {feature})",
                    mean.map_or_else(|| "n/a".to_owned(), |m| format!("{m:.3} ms"))
                ),
                false,
            ));
            continue;
        }
        let runs = stats.runs.unwrap_or(0.0);
        let valid = stats.valid.unwrap_or(0.0);
        let mean = stats.mean.filter(|m| m.is_finite() && valid > 0.0);
        lines.push(match (budget_ms(pass), mean) {
            (None, Some(m)) => (format!("  pass {pass}: mean {m:.3} ms (no budget)"), false),
            (None, None) => (format!("  pass {pass}: mean n/a (no budget)"), false),
            (Some(budget), _) if runs <= 0.0 => (
                format!("  pass {pass}: did not run (budget {budget} ms)"),
                false,
            ),
            (Some(budget), Some(m)) => budget_line(&format!("pass {pass}"), m, budget),
            (Some(budget), None) => (
                format!(
                    "  pass {pass}: mean n/a ({valid} valid timings in {runs} runs), budget \
                     {budget} ms: OVER BUDGET (unmeasured)"
                ),
                true,
            ),
        });
    }
    for (field, budget) in CPU_BUDGETS_MS {
        lines.push(match win.cpu_mean(field) {
            Some(m) => budget_line(&format!("cpu {field}"), m, *budget),
            None => (
                format!("  cpu {field}: not reported (budget {budget} ms)"),
                false,
            ),
        });
    }
    lines
}

/// Judges every bookmark of `file` (see the module docs).
pub fn evaluate(file: &BenchFile, args: &BenchArgs) -> Verdict {
    let key = args.window.key();
    let limits = &args.limits;
    let mut lines = Vec::new();
    let (mut within, mut flags) = (0, 0);
    for bookmark in &file.bookmarks {
        let Some(win) = bookmark.window(args.window) else {
            lines.push(format!(
                "{} [{key}]: no {key} window in the JSON FAIL",
                bookmark.name
            ));
            continue;
        };
        let parts = [
            limit_part(
                "GPU frame",
                win.gpu_frame_ms.as_ref(),
                limits.max_gpu_ms,
                "no valid GPU timing",
            ),
            limit_part(
                "CPU",
                win.cpu_ms.as_ref(),
                limits.max_cpu_ms,
                "not measured",
            ),
            limit_part(
                "interval",
                win.interval_ms.as_ref(),
                limits.max_interval_ms,
                "not measured",
            ),
        ];
        let ok = parts.iter().all(|(_, ok)| *ok);
        within += usize::from(ok);
        let text: Vec<&str> = parts.iter().map(|(t, _)| t.as_str()).collect();
        lines.push(format!(
            "{} [{key}]: {} {}",
            bookmark.name,
            text.join(", "),
            pass_word(ok)
        ));
        if args.budgets {
            for (line, flagged) in budget_lines(win) {
                flags += usize::from(flagged);
                lines.push(line);
            }
        }
    }
    let n = file.bookmarks.len();
    let pass = n > 0 && within == n && flags == 0;
    let budgets = if args.budgets {
        format!("; {flags} over 1.25 × budget")
    } else {
        String::new()
    };
    lines.push(format!(
        "bench [{key}]: {within} of {n} bookmarks within GPU frame p95 ≤ {} ms, CPU p95 ≤ {} ms, \
         interval p95 ≤ {} ms{budgets} {}",
        limits.max_gpu_ms,
        limits.max_cpu_ms,
        limits.max_interval_ms,
        pass_word(pass)
    ));
    Verdict { lines, pass }
}

/// Loads and judges.
pub fn run(args: &BenchArgs) -> anyhow::Result<Verdict> {
    Ok(evaluate(&load(&args.json)?, args))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One window of the B9 Step 3 schema with the given p95s (`None` → `null`) and pass means.
    fn window(gpu_p95: Option<f64>, cpu_p95: f64, interval_p95: f64, passes: &str) -> String {
        let stats = |p95: Option<f64>| match p95 {
            Some(p) => format!(r#"{{"mean": {}, "p95": {p}, "count": 600}}"#, p * 0.9),
            None => "null".to_owned(),
        };
        format!(
            r#"{{"frame_ms": {f}, "interval_ms": {i}, "gpu_frame_ms": {g}, "cpu_ms": {c},
                "passes": {{{passes}}}, "snapshot_cpu_ms": {{"mean": 0.8, "p95": 1.1, "count": 600}},
                "comets_cpu_ms": {{"mean": 0.3, "p95": 0.4, "count": 600}}}}"#,
            f = stats(Some(interval_p95 - 0.2)),
            i = stats(Some(interval_p95)),
            g = stats(gpu_p95),
            c = stats(Some(cpu_p95)),
        )
    }

    /// Three bookmarks: last 10 s GPU p95 15.9 (PASS), 16.4 (FAIL) and n/a (FAIL). Every first
    /// 10 s is far over the limits, so judging the last 10 s must ignore it.
    fn fixture() -> BenchFile {
        let slow_start = window(Some(30.0), 9.0, 40.0, "");
        let bodies_ok = r#""bodies": {"mean": 1.2, "valid": 600, "runs": 600},
            "lens": {"mean": 140.0, "valid": 600, "runs": 600}"#;
        let bodies_over = r#""bodies": {"mean": 2.0, "valid": 600, "runs": 600},
            "stars": {"mean": null, "valid": 0, "runs": 600}"#;
        let json = format!(
            r#"{{"size": [1920, 1080], "frames": null, "sustained_s": 60, "bookmarks": [
                {{"name": "start", "lens_scale": 0.5, "windows": {{
                    "first10s": {slow_start}, "last10s": {a}, "all": {a}}}}},
                {{"name": "mars-reference-angle", "lens_scale": 0.5, "windows": {{
                    "first10s": {slow_start}, "last10s": {b}, "all": {b}}}}},
                {{"name": "sky-orion-3d", "lens_scale": 0.5, "windows": {{
                    "first10s": {slow_start}, "last10s": {c}, "all": {c}}}}}
            ]}}"#,
            a = window(Some(15.9), 3.1, 16.5, bodies_ok),
            b = window(Some(16.4), 3.2, 16.6, bodies_over),
            c = window(None, 2.0, 16.4, ""),
        );
        serde_json::from_str(&json).expect("fixture parses")
    }

    fn args(window: Window, budgets: bool) -> BenchArgs {
        BenchArgs {
            json: PathBuf::from("bench.json"),
            limits: Limits {
                max_gpu_ms: 16.0,
                max_cpu_ms: 4.0,
                max_interval_ms: 16.7,
            },
            window,
            budgets,
        }
    }

    fn line<'a>(v: &'a Verdict, starts: &str) -> &'a str {
        v.lines
            .iter()
            .find(|l| l.starts_with(starts))
            .unwrap_or_else(|| panic!("no line starting {starts:?} in {v:#?}"))
    }

    #[test]
    fn bench_threshold_and_na() {
        let file = fixture();
        let v = evaluate(&file, &args(Window::Last10s, false));
        assert!(!v.pass);
        assert!(line(&v, "start").ends_with("PASS"), "{v:#?}");
        assert!(line(&v, "start").contains("15.900"), "{v:#?}");
        assert!(line(&v, "mars-reference-angle").ends_with("FAIL"), "{v:#?}");
        assert!(
            line(&v, "mars-reference-angle").contains("16.400"),
            "{v:#?}"
        );
        let na = line(&v, "sky-orion-3d");
        assert!(na.ends_with("FAIL") && na.contains("n/a"), "{v:#?}");
        assert!(
            v.lines.last().is_some_and(|l| l.ends_with("FAIL")),
            "{v:#?}"
        );

        // The first 10 s fail everything; judging them shows the window is honoured.
        let first = evaluate(&file, &args(Window::First10s, false));
        assert!(line(&first, "start").ends_with("FAIL"), "{first:#?}");
        assert!(line(&first, "start").contains("30.000"), "{first:#?}");

        // Only the passing bookmark: PASS overall.
        let only_start = BenchFile {
            bookmarks: file.bookmarks.into_iter().take(1).collect(),
            ..BenchFile::default()
        };
        let v = evaluate(&only_start, &args(Window::Last10s, false));
        assert!(v.pass, "{v:#?}");
        assert!(
            v.lines.last().is_some_and(|l| l.ends_with("PASS")),
            "{v:#?}"
        );
    }

    #[test]
    fn budgets_flag_passes_over_one_and_a_quarter() {
        let v = evaluate(&fixture(), &args(Window::Last10s, true));
        let text = v.lines.join("\n");
        // bodies 2.0 ms > 1.25 × 1.5 ms is flagged; 1.2 ms is not; lens has no budget; a pass
        // that ran but never timed validly is flagged as n/a.
        assert!(
            v.lines
                .iter()
                .any(|l| l.contains("bodies") && l.contains("2.000") && l.contains("OVER")),
            "{text}"
        );
        assert!(
            v.lines
                .iter()
                .any(|l| l.contains("bodies") && l.contains("1.200") && !l.contains("OVER")),
            "{text}"
        );
        assert!(
            v.lines
                .iter()
                .any(|l| l.contains("stars") && l.contains("n/a") && l.contains("OVER")),
            "{text}"
        );
        assert!(
            v.lines
                .iter()
                .any(|l| l.contains("lens") && l.contains("no budget")),
            "{text}"
        );
        assert!(!v.pass);

        // The one passing bookmark with budgets on: bodies 1.2 ms is within 1.25 × 1.5 ms.
        let file = BenchFile {
            bookmarks: fixture().bookmarks.into_iter().take(1).collect(),
            ..BenchFile::default()
        };
        assert!(evaluate(&file, &args(Window::Last10s, true)).pass);
    }

    #[test]
    fn budget_table_matches_the_plan() {
        // Plan "Performance budget": the budgeted passes sum to 12.35 ms (band-update counted
        // at 0 steady, the HUD at 0.5), 12.65 ms since the lead's band re-baseline (0.3 → 0.6),
        // 12.95 ms since the histogram's (0.3 → 0.6, Task C6 item 13): still ≤ 13 ms.
        let sum: f64 = PASS_BUDGETS_MS
            .iter()
            .filter(|(name, _)| *name != "band-update")
            .map(|(_, ms)| ms)
            .sum();
        assert!((sum - 12.95).abs() < 1e-9, "{sum}");
        assert!(sum <= 13.0);
        assert_eq!(budget_ms("band"), Some(0.6));
        assert_eq!(budget_ms("histogram"), Some(0.6));
        assert_eq!(budget_ms("comets"), Some(0.5));
        assert_eq!(budget_ms("asteroids"), Some(1.0));
        assert!(CPU_BUDGETS_MS.contains(&("asteroids_cpu_ms", 0.5)));
        assert_eq!(budget_ms("bodies"), Some(1.5));
        assert_eq!(budget_ms("lens"), None);
    }

    /// B9t (c): a feature timed as several passes is judged on their sum, and each part is
    /// listed without a budget of its own: the atmosphere's march 1.3 + composite 0.9 + mask
    /// 0.45 = 2.65 ms is within 1.25 × 3.0 ms though the march alone would read 1.3; the bloom's
    /// pyramid 0.6 + composite 0.8 = 1.4 ms is over 1.25 × 1.0 ms, which the pyramid alone
    /// (0.6) would hide.
    #[test]
    fn features_are_budgeted_with_all_their_passes() {
        let win: BenchWindow = serde_json::from_str(
            r#"{"passes": {
                "atmosphere": {"mean": 1.3, "valid": 10, "runs": 10},
                "atmosphere composite": {"mean": 0.9, "valid": 10, "runs": 10},
                "atmosphere mask": {"mean": 0.45, "valid": 10, "runs": 10},
                "bloom": {"mean": 0.6, "valid": 10, "runs": 10},
                "bloom composite": {"mean": 0.8, "valid": 10, "runs": 10},
                "tonemap": {"mean": 0.26, "valid": 10, "runs": 10}
            }}"#,
        )
        .expect("window");
        let lines = budget_lines(&win);
        let find = |what: &str| {
            lines
                .iter()
                .find(|(l, _)| l.contains(what))
                .unwrap_or_else(|| panic!("{what}: {lines:#?}"))
        };
        let (atmo, flagged) = find("feature atmosphere");
        assert!(!flagged && atmo.contains("2.650"), "{atmo}");
        assert!(atmo.contains("atmosphere + atmosphere composite + atmosphere mask"));
        let (bloom, flagged) = find("feature bloom");
        assert!(*flagged && bloom.contains("1.400"), "{bloom}");
        let (part, flagged) = find("pass atmosphere composite");
        assert!(
            !flagged && part.contains("part of feature atmosphere"),
            "{part}"
        );
        let (part, flagged) = find("pass bloom:");
        assert!(!flagged && part.contains("part of feature bloom"), "{part}");
        assert!(!find("pass tonemap").1);
        assert_eq!(lines.iter().filter(|(_, f)| *f).count(), 1);
        // A part without a valid timing makes the feature unmeasured.
        let mut win = win;
        if let Some(s) = win.passes.get_mut("atmosphere mask") {
            s.valid = Some(0.0);
            s.mean = None;
        }
        let lines = budget_lines(&win);
        assert!(
            lines
                .iter()
                .any(|(l, f)| *f && l.contains("feature atmosphere") && l.contains("unmeasured")),
            "{lines:#?}"
        );
    }

    #[test]
    fn missing_window_or_file_is_reported() {
        let file: BenchFile = serde_json::from_str(
            r#"{"bookmarks": [{"name": "start", "lens_scale": 1.0, "windows": {"all": null}}]}"#,
        )
        .expect("parses");
        let v = evaluate(&file, &args(Window::Last10s, false));
        assert!(!v.pass);
        assert!(line(&v, "start").contains("no last10s window"), "{v:#?}");
        assert!(load(Path::new("/nonexistent/bench.json")).is_err());
    }
}
