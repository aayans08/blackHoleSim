//! The M1 gate's numeric checks (final review, app I4).
//!
//! `uni-verify gate --captures <dir> --hud <dir> [--out <checks.json>]`
//!
//! Runs every numeric check of the plan's table "M1 capture bookmarks"
//! (`docs/superpowers/plans/2026-11-22-m1-solar-system.md`) on the captures of
//! `universe --capture bookmarks/m1.ron --out <dir> --size 1920x1080 --hdr`, and the card row
//! (row 24) on those of the same run with `--hud` instead of `--hdr`. [`M1_CHECKS`] is the table,
//! one entry per check with its plan row. It ports the lead's C6/M1-gate harness (a Python script
//! that lived outside the repository) check for check: the same `uni-verify` command lines,
//! with the same tolerances and the lead's rulings noted in the check text, run in-process, and
//! the same JSON checks. It adds row 11's "JSON ev100 within limits, no NaN", which the script
//! left out. Row 19 has no numeric check (lead rulings "row 19 shadow" and "row 19 seam"; its
//! seam is row 27).
//!
//! Each check's result is the last line of its output, which ends in PASS or FAIL; an error (a
//! missing PNG, JSON or `.hdr.bin` sidecar, a missing field) is a FAIL with the reason. The
//! results go to `--out` (default `<captures>/checks.json`, which `sheet` shows) as an array of
//! `{row, bookmark, check, pass, output}`, and one line per check is printed. The run passes
//! when every check passes.
//!
//! The HDR checks need the `.hdr.bin` sidecars, which are large (33 MB each at 1920×1080): run
//! the gate before deleting them.

use crate::{Verdict, pass_word};
use anyhow::Context;
use serde::Serialize;
use serde_json::Value;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

/// The `gate` command line.
#[derive(Debug, Clone, PartialEq)]
pub struct GateArgs {
    /// The `--hdr` capture directory.
    pub captures: PathBuf,
    /// The `--hud` capture directory (row 24's card rows).
    pub hud: PathBuf,
    /// Where to write the results; `None`: `<captures>/checks.json`.
    pub out: Option<PathBuf>,
}

/// The file the results go to when `--out` is not given, in the captures directory.
pub const CHECKS_FILE: &str = "checks.json";

/// The lowest and highest exposure [EV100] the app writes: `uni_post::EV100_MIN` and
/// `uni_post::EV100_MAX` (crates/uni-post/src/exposure.rs), also `uni_app::scene::EV100_LIMITS`
/// (checked against the source by a test).
pub const EV100_LIMITS: [f64; 2] = [-30.0, 40.0];

/// Which capture directory a JSON check reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dir {
    /// `--captures` (the `--hdr` run).
    Captures,
    /// `--hud`.
    Hud,
}

/// How a check is run.
#[derive(Debug, Clone, Copy)]
pub enum How {
    /// A `uni-verify` command line (without the program name), run in-process. The values of
    /// `--png`, `--json` and `--hdr-json` are file names in the captures directory.
    Cli(&'static [&'static str]),
    /// `disk` with `--center` set to the projected centre of `body` in the bookmark's capture
    /// JSON (`projected`), then the other arguments as for [`How::Cli`].
    Disk {
        body: &'static str,
        args: &'static [&'static str],
    },
    /// A judgement of the bookmark's capture JSON in `dir`: (pass, the measured text).
    Json {
        dir: Dir,
        judge: fn(&Value) -> (bool, String),
    },
}

/// One check of the gate.
#[derive(Debug, Clone, Copy)]
pub struct Check {
    /// The plan's row ("M1 capture bookmarks").
    pub row: u32,
    pub bookmark: &'static str,
    /// What is checked and against what (rulings in brackets).
    pub check: &'static str,
    pub how: How,
}

const fn cli(
    row: u32,
    bookmark: &'static str,
    check: &'static str,
    args: &'static [&'static str],
) -> Check {
    Check {
        row,
        bookmark,
        check,
        how: How::Cli(args),
    }
}

const fn json(
    row: u32,
    bookmark: &'static str,
    check: &'static str,
    judge: fn(&Value) -> (bool, String),
) -> Check {
    Check {
        row,
        bookmark,
        check,
        how: How::Json {
            dir: Dir::Captures,
            judge,
        },
    }
}

/// The M1 gate: the 42 checks of the lead's harness, in its order, plus row 11's ev100 check.
#[rustfmt::skip]
pub const M1_CHECKS: &[Check] = &[
    cli(1, "start", "points Moon ±2px",
        &["points", "--png", "start.png", "--json", "start.json", "--names", "Moon", "--tol", "2"]),
    cli(2, "earth-limb-dawn", "ratio limb_arc_high B/R ≥1.5",
        &["ratio", "--hdr-json", "earth-limb-dawn.json", "--num", "limb_arc_high", "--den",
          "limb_arc_high", "--channels", "b/r", "--min", "1.5"]),
    cli(2, "earth-limb-dawn", "ratio limb_arc_low R/B ≥1.5",
        &["ratio", "--hdr-json", "earth-limb-dawn.json", "--num", "limb_arc_low", "--den",
          "limb_arc_low", "--channels", "r/b", "--min", "1.5"]),
    cli(3, "earth-map-closeup", "samples 16 pts ±10%",
        &["samples", "--hdr-json", "earth-map-closeup.json", "--tol", "0.10"]),
    Check {
        row: 4,
        bookmark: "moon-full",
        check: "disk a=267.0±1, b/a=1.000±0.003 (threshold 0.05, ruling \"disk fits\")",
        how: How::Disk {
            body: "Moon",
            args: &["disk", "--png", "moon-full.png", "--threshold", "0.05", "--expected-a",
                    "267.0", "--tol", "1", "--ratio", "1.0", "--ratio-tol", "0.003"],
        },
    },
    cli(4, "moon-full", "samples 16 pts ±10%",
        &["samples", "--hdr-json", "moon-full.json", "--tol", "0.10"]),
    cli(5, "mars-reference-angle", "samples 16 pts ±5% (footprint averages)",
        &["samples", "--hdr-json", "mars-reference-angle.json", "--tol", "0.05"]),
    cli(5, "mars-reference-angle", "ratio limb_haze R/B ≥1.0",
        &["ratio", "--hdr-json", "mars-reference-angle.json", "--num", "limb_haze", "--den",
          "limb_haze", "--channels", "r/b", "--min", "1.0"]),
    cli(6, "jupiter-galileans", "points Io,Europa,Ganymede,Callisto ±2px",
        &["points", "--png", "jupiter-galileans.png", "--json", "jupiter-galileans.json",
          "--names", "Io,Europa,Ganymede,Callisto", "--tol", "2"]),
    Check {
        row: 7,
        bookmark: "jupiter-disk",
        check: "disk a=421.3±1, b/a 0.9351±0.004 (threshold 0.05)",
        how: How::Disk {
            body: "Jupiter",
            args: &["disk", "--png", "jupiter-disk.png", "--threshold", "0.05", "--expected-a",
                    "421.3", "--tol", "1", "--ratio", "0.9351", "--ratio-tol", "0.004"],
        },
    },
    cli(8, "saturn-ring-plane", "ratio ring_cassini/ring_b luma ≤0.5",
        &["ratio", "--hdr-json", "saturn-ring-plane.json", "--num", "ring_cassini", "--den",
          "ring_b", "--max", "0.5"]),
    cli(8, "saturn-ring-plane", "ratio globe_ring_shadow/globe_lit_ref luma ≤0.5",
        &["ratio", "--hdr-json", "saturn-ring-plane.json", "--num", "globe_ring_shadow", "--den",
          "globe_lit_ref", "--max", "0.5"]),
    Check {
        row: 9,
        bookmark: "saturn-edge-on",
        check: "disk a=433.6±1, b/a 0.9020±0.004, exclude ring line (threshold 0.05)",
        how: How::Disk {
            body: "Saturn",
            args: &["disk", "--png", "saturn-edge-on.png", "--threshold", "0.05", "--expected-a",
                    "433.6", "--tol", "1", "--ratio", "0.9020", "--ratio-tol", "0.004",
                    "--exclude-angle", "0", "--exclude-width", "10"],
        },
    },
    cli(10, "saturn-night-side", "ratio ring_in_planet_shadow/ring_lit_same_radius luma ≤0.2",
        &["ratio", "--hdr-json", "saturn-night-side.json", "--num", "ring_in_planet_shadow",
          "--den", "ring_lit_same_radius", "--max", "0.2"]),
    cli(11, "sun-closeup", "ratio mu_0p5/disk_centre G = 0.703705 ±3%",
        &["ratio", "--hdr-json", "sun-closeup.json", "--num", "mu_0p5", "--den", "disk_centre",
          "--channels", "g", "--expect", "0.703705", "--rel-tol", "0.03"]),
    json(11, "sun-closeup",
        "JSON ev100 [scene, star] within EV100 limits [-30, 40], no NaN (null) (plan row 11; \
         the harness left it out, review I4)",
        judge_ev100),
    cli(12, "sun-granulation", "stats granulation_patch rel RMS ∈[0.08,0.20]",
        &["stats", "--hdr-json", "sun-granulation.json", "--region", "granulation_patch",
          "--rms-min", "0.08", "--rms-max", "0.20"]),
    cli(13, "venus-crescent", "limb venus_night_limb ±10%",
        &["limb", "--hdr-json", "venus-crescent.json", "--name", "venus_night_limb", "--tol",
          "0.1"]),
    cli(14, "titan-backlit", "ratio haze_ring R/B ≥1.2",
        &["ratio", "--hdr-json", "titan-backlit.json", "--num", "haze_ring", "--den", "haze_ring",
          "--channels", "r/b", "--min", "1.2"]),
    cli(14, "titan-backlit", "stats haze_ring mean pre ≥1.4e-3",
        &["stats", "--hdr-json", "titan-backlit.json", "--region", "haze_ring",
          "--mean-pre-min", "1.4e-3"]),
    cli(15, "sky-orion-3d", "points Betelgeuse,Rigel,Bellatrix,Alnilam ±1px",
        &["points", "--png", "sky-orion-3d.png", "--json", "sky-orion-3d.json", "--names",
          "Betelgeuse,Rigel,Bellatrix,Alnilam", "--tol", "1"]),
    cli(15, "sky-orion-3d", "flux Rigel ±5% (own + neighbour_lux)",
        &["flux", "--hdr-json", "sky-orion-3d.json", "--name", "Rigel", "--tol", "0.05"]),
    cli(16, "sky-galactic-center", "ratio band_plane/band_high_lat median ≥2",
        &["ratio", "--hdr-json", "sky-galactic-center.json", "--num", "band_plane", "--den",
          "band_high_lat", "--min", "2"]),
    json(16, "sky-galactic-center", "JSON band_gc_over_anticentre ≥2 (CPU twin)", judge_band_gc),
    cli(17, "alpha-cen-sun", "points Sun ±1px",
        &["points", "--png", "alpha-cen-sun.png", "--json", "alpha-cen-sun.json", "--names", "Sun",
          "--tol", "1"]),
    cli(17, "alpha-cen-sun", "flux Sun ±5%",
        &["flux", "--hdr-json", "alpha-cen-sun.json", "--name", "Sun", "--tol", "0.05"]),
    json(17, "alpha-cen-sun", "JSON sun_v_mag = 0.3955 ±0.01 (M_V☉ + 5 log10(d/10 pc), plan)",
        judge_sun_v_mag),
    cli(18, "bh-shadow-30M-grid", "shadow PNG 158.745±1.0",
        &["shadow", "--png", "bh-shadow-30M-grid.png", "--expected-px", "158.745", "--tol",
          "1.0"]),
    // Row 19: visual and seam only (lead rulings "row 19 shadow", "row 19 seam"); the seam is
    // row 27.
    cli(20, "comet-neowise-2020", "points nucleus at the Horizons RA/Dec ±5px",
        &["points", "--png", "comet-neowise-2020.png", "--json", "comet-neowise-2020.json",
          "--names", "C/2020 F3 Horizons", "--tol", "5"]),
    cli(20, "comet-neowise-2020", "angle ion_tail ±1° of ion_tail_dir projected",
        &["angle", "--hdr-json", "comet-neowise-2020.json", "--feature", "ion_tail", "--tol-deg",
          "1"]),
    cli(21, "comet-halley-1986", "profile coma slope ±0.1 of −1 − ⟨ρ/ℓ⟩ (8–40 px)",
        &["profile", "--hdr-json", "comet-halley-1986.json", "--feature", "coma", "--tol", "0.1"]),
    json(22, "asteroid-belt-above", "JSON asteroid_counts (real, procedural)",
        judge_asteroid_counts),
    json(22, "asteroid-belt-above", "JSON belt_radial_hist: Kirkwood 3:1 dip (a) < 0.3",
        judge_kirkwood),
    json(22, "asteroid-belt-above",
        "JSON belt_radial_hist: L4 and L5 peaks > 3× the λ−λJ background", judge_trojans),
    cli(22, "asteroid-belt-above", "ratio belt_annulus/belt_outer mean ≥3",
        &["ratio", "--hdr-json", "asteroid-belt-above.json", "--num", "belt_annulus", "--den",
          "belt_outer", "--stat", "mean", "--min", "3"]),
    cli(22, "asteroid-belt-above",
        "ratio belt_annulus/belt_outer median ≥3 (logged beside the mean, round-3 ruling)",
        &["ratio", "--hdr-json", "asteroid-belt-above.json", "--num", "belt_annulus", "--den",
          "belt_outer", "--stat", "median", "--min", "3"]),
    json(22, "asteroid-belt-above", "JSON honesty has the amber belt label", judge_belt_label),
    json(23, "exo-trappist1", "JSON overlay orbit count == 7 (dashed, Ω unknown)",
        judge_trappist_orbits),
    Check {
        row: 24,
        bookmark: "hud-target-mars",
        check: "JSON hud.card_rows == the mockup Mars order (--hud run)",
        how: How::Json {
            dir: Dir::Hud,
            judge: judge_card_rows,
        },
    },
    json(25, "hud-cruise", "JSON honesty contains the cruise label", judge_cruise_label),
    cli(26, "sky-orion-50kyr",
        "points Betelgeuse,Rigel,Bellatrix,Alnilam ±1px (+50 kyr positions)",
        &["points", "--png", "sky-orion-50kyr.png", "--json", "sky-orion-50kyr.json", "--names",
          "Betelgeuse,Rigel,Bellatrix,Alnilam", "--tol", "1"]),
    json(26, "sky-orion-50kyr",
        "JSON honesty has the planets-extrapolated label, not the stars one",
        judge_extrapolation_labels),
    cli(27, "bh-lens-seam",
        "ratio lens_rect_inner_edge/lens_rect_outer_edge median = 1±0.1 (row 19 seam ruling)",
        &["ratio", "--hdr-json", "bh-lens-seam.json", "--num", "lens_rect_inner_edge", "--den",
          "lens_rect_outer_edge", "--expect", "1", "--rel-tol", "0.1"]),
];

/// The amber label of the belt highlight (uni-ui).
const BELT_LABEL: &str = "Asteroid belt brightness exaggerated (non-physical)";
/// The amber cruise label (uni-ui).
const CRUISE_LABEL: &str = "CRUISE · FTL · NON-PHYSICAL · GRAVITY IGNORED";
/// The planets-extrapolated label (uni-sim honesty).
const PLANETS_LABEL: &str = "Planet positions extrapolated outside 3000 BC – AD 3000 (approximate)";
/// The stars-extrapolated label (uni-sim honesty).
const STARS_LABEL: &str = "Star positions: linear motion beyond ±100 000 yr (approximate)";
/// The Mars card's rows in the mockup's order (Task C4).
const MARS_CARD_ROWS: [&str; 15] = [
    "Distance",
    "Closing speed",
    "Light time",
    "Mean radius",
    "Mass",
    "Surface gravity",
    "Escape velocity",
    "Sidereal day",
    "Orbital period",
    "Semi-major axis",
    "Axial tilt",
    "Surface pressure",
    "Atmosphere",
    "Mean temperature",
    "Moons",
];

/// `metrics.<key>` of a capture JSON.
fn metric<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    v.get("metrics")?.get(key)
}

/// A value printed compactly: `[a, b]`, `{"k": v}`, strings quoted.
fn show(v: &Value) -> String {
    match v {
        Value::Array(items) => format!(
            "[{}]",
            items.iter().map(show).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(map) => format!(
            "{{{}}}",
            map.iter()
                .map(|(k, v)| format!("{k:?}: {}", show(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

/// True when the JSON's `honesty` list holds `label` exactly.
fn honesty_has(v: &Value, label: &str) -> bool {
    v.get("honesty")
        .and_then(Value::as_array)
        .is_some_and(|labels| labels.iter().any(|l| l.as_str() == Some(label)))
}

/// Row 11: the capture's `ev100` (and the HDR header's, when there is one) is two numbers
/// within [`EV100_LIMITS`]. serde_json writes a NaN or ±∞ as `null`, so a `null` is a NaN.
fn judge_ev100(v: &Value) -> (bool, String) {
    let pair = |key: &str, value: Option<&Value>| -> Result<String, String> {
        let items = value
            .and_then(Value::as_array)
            .ok_or_else(|| format!("no {key} array"))?;
        if items.len() != 2 {
            return Err(format!(
                "{key} {} has {} entries, not 2",
                show(&Value::from(items.clone())),
                items.len()
            ));
        }
        for (i, item) in items.iter().enumerate() {
            match item.as_f64() {
                None if item.is_null() => return Err(format!("{key}[{i}] is null (NaN)")),
                None => return Err(format!("{key}[{i}] {item} is not a number")),
                Some(ev) if !(EV100_LIMITS[0]..=EV100_LIMITS[1]).contains(&ev) => {
                    return Err(format!(
                        "{key}[{i}] {ev} outside [{}, {}]",
                        EV100_LIMITS[0], EV100_LIMITS[1]
                    ));
                }
                Some(_) => {}
            }
        }
        Ok(format!("{key} {}", show(&Value::from(items.clone()))))
    };
    let mut parts = vec![pair("ev100", v.get("ev100"))];
    if let Some(hdr) = v.get("hdr") {
        parts.push(pair("hdr.ev100", hdr.get("ev100")));
    }
    let pass = parts.iter().all(Result::is_ok);
    let text: Vec<String> = parts.into_iter().map(|p| p.unwrap_or_else(|e| e)).collect();
    let limits = format!("[{}, {}]", EV100_LIMITS[0], EV100_LIMITS[1]);
    let tail = if pass {
        format!(" within {limits}, no NaN")
    } else {
        String::new()
    };
    (pass, format!("{}{tail}", text.join(", ")))
}

/// Row 16: the CPU twin's band ratio, galactic centre over anticentre, ≥ 2.
fn judge_band_gc(v: &Value) -> (bool, String) {
    match metric(v, "band_gc_over_anticentre").and_then(Value::as_f64) {
        Some(g) => (g >= 2.0, format!("band_gc_over_anticentre {g:.4} ≥ 2")),
        None => (false, "no metrics.band_gc_over_anticentre".to_owned()),
    }
}

/// Row 17: the Sun's V magnitude from α Cen, 0.3955 ± 0.01 (M_V☉ + 5 log10(d/10 pc), plan).
fn judge_sun_v_mag(v: &Value) -> (bool, String) {
    match metric(v, "sun_v_mag_from_camera").and_then(Value::as_f64) {
        Some(m) => (
            (m - 0.3955).abs() <= 0.01,
            format!("sun_v_mag {m:.4} vs 0.3955"),
        ),
        None => (false, "no metrics.sun_v_mag_from_camera".to_owned()),
    }
}

/// Row 22: both asteroid counts (registry, procedural) are positive.
fn judge_asteroid_counts(v: &Value) -> (bool, String) {
    match metric(v, "asteroid_counts") {
        Some(c) => {
            let n = |i: usize| c.get(i).and_then(Value::as_f64).unwrap_or(0.0);
            (
                n(0) > 0.0 && n(1) > 0.0,
                format!("asteroid_counts {}", show(c)),
            )
        }
        None => (false, "no metrics.asteroid_counts".to_owned()),
    }
}

/// `metrics.belt_radial_hist.<key>` as a number.
fn belt(v: &Value, key: &str) -> Option<f64> {
    metric(v, "belt_radial_hist")?.get(key)?.as_f64()
}

/// Row 22: the Kirkwood 3:1 gap, by semi-major axis, dips below 0.3 of its surroundings.
fn judge_kirkwood(v: &Value) -> (bool, String) {
    match belt(v, "kirkwood_dip_a") {
        Some(a) => {
            let r =
                belt(v, "kirkwood_dip_r").map_or_else(|| "n/a".to_owned(), |r| format!("{r:.3}"));
            (
                a < 0.3,
                format!("kirkwood_dip_a {a:.3} (by projected r {r})"),
            )
        }
        None => (
            false,
            "no metrics.belt_radial_hist.kirkwood_dip_a".to_owned(),
        ),
    }
}

/// Row 22: the L4 and L5 Trojan peaks stand > 3× above the λ − λJ background.
fn judge_trojans(v: &Value) -> (bool, String) {
    let l4 = belt(v, "l4_over_bg").unwrap_or(0.0);
    let l5 = belt(v, "l5_over_bg").unwrap_or(0.0);
    (l4.min(l5) > 3.0, format!("L4 {l4:.1}×, L5 {l5:.1}×"))
}

/// Row 22: the amber belt-highlight label.
fn judge_belt_label(v: &Value) -> (bool, String) {
    let has = honesty_has(v, BELT_LABEL);
    (has, format!("\"{BELT_LABEL}\" in honesty: {has}"))
}

/// Row 23: the 7 TRAPPIST-1 orbits are drawn dashed (Ω unknown).
fn judge_trappist_orbits(v: &Value) -> (bool, String) {
    match metric(v, "overlay_orbits") {
        Some(o) => (
            o.get("dashed").and_then(Value::as_f64) == Some(7.0),
            format!("overlay_orbits {}", show(o)),
        ),
        None => (false, "no metrics.overlay_orbits".to_owned()),
    }
}

/// Row 24: the Mars card's rows in the mockup's order (the `--hud` run).
fn judge_card_rows(v: &Value) -> (bool, String) {
    match v.get("hud").and_then(|h| h.get("card_rows")) {
        Some(rows) => {
            let want = Value::from(MARS_CARD_ROWS.to_vec());
            (*rows == want, format!("card_rows {}", show(rows)))
        }
        None => (false, "no hud.card_rows: capture with --hud".to_owned()),
    }
}

/// Row 25: the amber cruise label.
fn judge_cruise_label(v: &Value) -> (bool, String) {
    let has = honesty_has(v, CRUISE_LABEL);
    (has, format!("\"{CRUISE_LABEL}\" in honesty: {has}"))
}

/// Row 26: 50 kyr ahead, the planets are labelled extrapolated, the stars not yet.
fn judge_extrapolation_labels(v: &Value) -> (bool, String) {
    let (planets, stars) = (honesty_has(v, PLANETS_LABEL), honesty_has(v, STARS_LABEL));
    (
        planets && !stars,
        format!("planets {planets}, stars {stars}"),
    )
}

/// One check's result, as `checks.json` holds it.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Outcome {
    pub row: u32,
    pub bookmark: String,
    pub check: String,
    pub pass: bool,
    /// The check's last output line (ending in PASS or FAIL), or the error.
    pub output: String,
}

/// The values of these flags are file names in the captures directory.
const FILE_FLAGS: [&str; 3] = ["--png", "--json", "--hdr-json"];

/// The command line of a [`How::Cli`] check with its file arguments in `dir`.
fn resolve(args: &[&str], dir: &Path) -> Vec<OsString> {
    let mut out = Vec::with_capacity(args.len());
    let mut file_next = false;
    for &arg in args {
        out.push(if file_next {
            dir.join(arg).into_os_string()
        } else {
            OsString::from(arg)
        });
        file_next = FILE_FLAGS.contains(&arg);
    }
    out
}

/// Reads `<dir>/<bookmark>.json`.
fn load_json(dir: &Path, bookmark: &str) -> anyhow::Result<Value> {
    let path = dir.join(format!("{bookmark}.json"));
    let text =
        fs::read_to_string(&path).with_context(|| format!("cannot read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("cannot parse {}", path.display()))
}

/// The projected centre `<x>,<y>` of `body` in `<captures>/<bookmark>.json`.
fn projected_centre(captures: &Path, bookmark: &str, body: &str) -> anyhow::Result<String> {
    let v = load_json(captures, bookmark)?;
    let item = v
        .get("projected")
        .and_then(Value::as_array)
        .and_then(|list| {
            list.iter()
                .find(|p| p.get("name").and_then(Value::as_str) == Some(body))
        })
        .with_context(|| format!("{body} is not in {bookmark}.json's `projected` list"))?;
    let coord = |k: &str| {
        item.get(k)
            .and_then(Value::as_f64)
            .with_context(|| format!("{body} has no projected {k}"))
    };
    Ok(format!("{},{}", coord("x")?, coord("y")?))
}

/// Runs a command line in-process: whether it passed (its verdict, and a last line ending in
/// PASS, as the harness judged a run) and its last line.
fn run_cli(args: &[OsString]) -> anyhow::Result<(bool, String)> {
    let verdict = crate::verdict(crate::parse_args(args)?)?;
    let last = verdict.lines.last().cloned().unwrap_or_default();
    Ok((verdict.pass && last.ends_with("PASS"), last))
}

/// Runs one check against the capture directories.
pub fn run_check(check: &Check, captures: &Path, hud: &Path) -> Outcome {
    let result: anyhow::Result<(bool, String)> = match check.how {
        How::Cli(args) => run_cli(&resolve(args, captures)),
        How::Disk { body, args } => {
            projected_centre(captures, check.bookmark, body).and_then(|c| {
                let mut line = resolve(args, captures);
                line.extend([OsString::from("--center"), OsString::from(c)]);
                run_cli(&line)
            })
        }
        How::Json { dir, judge } => {
            let dir = match dir {
                Dir::Captures => captures,
                Dir::Hud => hud,
            };
            load_json(dir, check.bookmark).map(|v| {
                let (pass, text) = judge(&v);
                (pass, format!("{text} {}", pass_word(pass)))
            })
        }
    };
    let (pass, output) = result.unwrap_or_else(|e| {
        let reason = format!("{e:#}").replace('\n', " ");
        (false, format!("error: {reason} FAIL"))
    });
    Outcome {
        row: check.row,
        bookmark: check.bookmark.to_owned(),
        check: check.check.to_owned(),
        pass,
        output,
    }
}

/// Runs `checks`, writes the results to `out` and returns the verdict: one line per check, then
/// the count.
pub fn run_checks(
    checks: &[Check],
    captures: &Path,
    hud: &Path,
    out: &Path,
) -> anyhow::Result<Verdict> {
    let outcomes: Vec<Outcome> = checks.iter().map(|c| run_check(c, captures, hud)).collect();
    let text = serde_json::to_string_pretty(&outcomes).context("cannot encode the results")?;
    fs::write(out, text + "\n").with_context(|| format!("cannot write {}", out.display()))?;
    let passed = outcomes.iter().filter(|o| o.pass).count();
    let pass = !outcomes.is_empty() && passed == outcomes.len();
    let mut lines: Vec<String> = outcomes
        .iter()
        .map(|o| {
            format!(
                "{} {} {} | {} | {}",
                pass_word(o.pass),
                o.row,
                o.bookmark,
                o.check,
                o.output
            )
        })
        .collect();
    lines.push(format!("results -> {}", out.display()));
    lines.push(format!(
        "gate: {passed}/{} checks {}",
        outcomes.len(),
        pass_word(pass)
    ));
    Ok(Verdict { lines, pass })
}

/// `gate`: the M1 checks on the two capture directories.
pub fn run(args: &GateArgs) -> anyhow::Result<Verdict> {
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| args.captures.join(CHECKS_FILE));
    run_checks(M1_CHECKS, &args.captures, &args.hud, &out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use std::collections::BTreeMap;

    /// A fresh, empty directory under the system temp dir, unique to this process and test.
    fn temp_dir(test: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("uni-verify-{}-{test}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    /// The plan's rows with a numeric check, and how many checks each has: the lead's harness
    /// (42) plus row 11's ev100 check. Row 19 has none (lead rulings).
    #[test]
    fn table_covers_the_plan_rows() {
        let mut per_row: BTreeMap<u32, usize> = BTreeMap::new();
        for c in M1_CHECKS {
            *per_row.entry(c.row).or_default() += 1;
        }
        let want: BTreeMap<u32, usize> = [
            (1, 1),
            (2, 2),
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 1),
            (7, 1),
            (8, 2),
            (9, 1),
            (10, 1),
            (11, 2),
            (12, 1),
            (13, 1),
            (14, 2),
            (15, 2),
            (16, 2),
            (17, 3),
            (18, 1),
            (20, 2),
            (21, 1),
            (22, 6),
            (23, 1),
            (24, 1),
            (25, 1),
            (26, 2),
            (27, 1),
        ]
        .into_iter()
        .collect();
        assert_eq!(per_row, want);
        assert_eq!(M1_CHECKS.len(), 43);
        // In plan order.
        assert!(M1_CHECKS.windows(2).all(|w| w[0].row <= w[1].row));
        // Only row 24 reads the --hud run.
        for c in M1_CHECKS {
            let hud = matches!(c.how, How::Json { dir: Dir::Hud, .. });
            assert_eq!(hud, c.row == 24, "{c:?}");
        }
    }

    /// Every command line of the table parses, its files are the bookmark's, and the bounds are
    /// the harness's (spot checks of each kind).
    #[test]
    fn table_command_lines_parse() {
        let dir = Path::new("caps");
        for c in M1_CHECKS {
            let line = match c.how {
                How::Cli(args) => resolve(args, dir),
                How::Disk { args, .. } => {
                    let mut line = resolve(args, dir);
                    line.extend([OsString::from("--center"), OsString::from("960.5,540")]);
                    line
                }
                How::Json { .. } => continue,
            };
            if let Err(e) = crate::parse_args(&line) {
                panic!("row {} {}: {e:#}", c.row, c.check);
            }
            // File arguments name the bookmark's own capture, in the captures directory.
            for (flag, value) in line.iter().zip(line.iter().skip(1)) {
                if FILE_FLAGS.iter().any(|f| flag == *f) {
                    let p = Path::new(value);
                    assert_eq!(p.parent(), Some(dir), "{c:?}");
                    let stem = p.file_stem().and_then(|s| s.to_str());
                    assert_eq!(stem, Some(c.bookmark), "{c:?}");
                }
            }
        }
        let parsed = |row: u32, check: &str| {
            let c = M1_CHECKS
                .iter()
                .find(|c| c.row == row && c.check.starts_with(check))
                .unwrap_or_else(|| panic!("row {row} {check}"));
            let How::Cli(args) = c.how else {
                panic!("row {row} {check} is not a command line")
            };
            crate::parse_args(&resolve(args, dir)).expect("parses")
        };
        match parsed(11, "ratio") {
            crate::Command::Ratio { hdr_json, args } => {
                assert_eq!(hdr_json, dir.join("sun-closeup.json"));
                assert_eq!(args.channels, crate::hdr::Channels::Green);
                assert_eq!(args.bounds.expect, Some((0.703705, 0.03)));
            }
            other => panic!("{other:?}"),
        }
        match parsed(22, "ratio belt_annulus/belt_outer mean") {
            crate::Command::Ratio { args, .. } => {
                assert_eq!(args.stat, crate::hdr::Stat::Mean);
                assert_eq!(args.bounds.min, Some(3.0));
            }
            other => panic!("{other:?}"),
        }
        match parsed(20, "points") {
            crate::Command::Points { names, tol, .. } => {
                assert_eq!(names, vec!["C/2020 F3 Horizons".to_owned()]);
                assert_eq!(tol, 5.0);
            }
            other => panic!("{other:?}"),
        }
        match parsed(18, "shadow") {
            crate::Command::Shadow {
                expected_px, tol, ..
            } => assert_eq!((expected_px, tol), (158.745, 1.0)),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn ev100_limits_match_uni_post() {
        // uni-verify does not link uni-post; the constants are read from its source.
        let src = include_str!("../../../crates/uni-post/src/exposure.rs");
        assert!(
            src.contains("pub const EV100_MIN: f32 = -30.0;"),
            "EV100_MIN changed"
        );
        assert!(
            src.contains("pub const EV100_MAX: f32 = 40.0;"),
            "EV100_MAX changed"
        );
        assert_eq!(EV100_LIMITS, [-30.0, 40.0]);
    }

    #[test]
    fn row_11_ev100_within_limits_and_no_nan() {
        let judge = |text: &str| judge_ev100(&serde_json::from_str(text).expect("JSON"));
        let (pass, out) =
            judge(r#"{"ev100": [33.974155, -6.5], "hdr": {"ev100": [33.974155, -6.5]}}"#);
        assert!(pass, "{out}");
        assert_eq!(
            out,
            "ev100 [33.974155, -6.5], hdr.ev100 [33.974155, -6.5] within [-30, 40], no NaN"
        );
        // Without --hdr only the capture's own pair is read.
        assert!(judge(r#"{"ev100": [12.0, -4.0]}"#).0);
        // serde_json writes NaN as null.
        let (pass, out) = judge(r#"{"ev100": [null, -6.5]}"#);
        assert!(!pass && out.contains("ev100[0] is null (NaN)"), "{out}");
        let (pass, out) = judge(r#"{"ev100": [33.0, -6.5], "hdr": {"ev100": [33.0, null]}}"#);
        assert!(!pass && out.contains("hdr.ev100[1] is null (NaN)"), "{out}");
        for bad in [
            r#"{"ev100": [40.5, -6.5]}"#,
            r#"{"ev100": [10.0, -31.0]}"#,
            r#"{"ev100": [10.0]}"#,
            r#"{"ev100": "high"}"#,
            r#"{}"#,
        ] {
            assert!(!judge(bad).0, "{bad}");
        }
        // The limits themselves pass.
        assert!(judge(r#"{"ev100": [40.0, -30.0]}"#).0);
    }

    #[test]
    fn json_judges_pass_and_fail() {
        let v = |text: &str| -> Value { serde_json::from_str(text).expect("JSON") };
        let belt = v(r#"{"metrics": {"asteroid_counts": [24705, 100000],
            "belt_radial_hist": {"kirkwood_dip_a": 0.089, "kirkwood_dip_r": 1.005,
                                 "l4_over_bg": 385.2, "l5_over_bg": 242.4}},
            "honesty": ["Asteroid belt brightness exaggerated (non-physical)"]}"#);
        assert_eq!(
            judge_asteroid_counts(&belt),
            (true, "asteroid_counts [24705, 100000]".to_owned())
        );
        assert_eq!(
            judge_kirkwood(&belt),
            (
                true,
                "kirkwood_dip_a 0.089 (by projected r 1.005)".to_owned()
            )
        );
        assert_eq!(
            judge_trojans(&belt),
            (true, "L4 385.2×, L5 242.4×".to_owned())
        );
        assert!(judge_belt_label(&belt).0);
        let empty = v("{}");
        for judge in [
            judge_ev100,
            judge_band_gc,
            judge_sun_v_mag,
            judge_asteroid_counts,
            judge_kirkwood,
            judge_trojans,
            judge_belt_label,
            judge_trappist_orbits,
            judge_card_rows,
            judge_cruise_label,
            judge_extrapolation_labels,
        ] {
            assert!(!judge(&empty).0);
        }
        assert!(!judge_asteroid_counts(&v(r#"{"metrics": {"asteroid_counts": [0, 100000]}}"#)).0);
        assert!(
            !judge_kirkwood(&v(
                r#"{"metrics": {"belt_radial_hist": {"kirkwood_dip_a": 0.3}}}"#
            ))
            .0
        );
        assert!(
            !judge_trojans(&v(
                r#"{"metrics": {"belt_radial_hist": {"l4_over_bg": 9, "l5_over_bg": 3}}}"#
            ))
            .0
        );
        assert_eq!(
            judge_band_gc(&v(
                r#"{"metrics": {"band_gc_over_anticentre": 2.507760882163044}}"#
            )),
            (true, "band_gc_over_anticentre 2.5078 ≥ 2".to_owned())
        );
        assert!(!judge_band_gc(&v(r#"{"metrics": {"band_gc_over_anticentre": 1.99}}"#)).0);
        assert_eq!(
            judge_sun_v_mag(&v(
                r#"{"metrics": {"sun_v_mag_from_camera": 0.3948353437036243}}"#
            )),
            (true, "sun_v_mag 0.3948 vs 0.3955".to_owned())
        );
        assert!(!judge_sun_v_mag(&v(r#"{"metrics": {"sun_v_mag_from_camera": 0.41}}"#)).0);
        let orbits = |n: u32| {
            v(&format!(
                r#"{{"metrics": {{"overlay_orbits": {{"lines": 7, "dashed": {n}, "dots": 7}}}}}}"#
            ))
        };
        assert!(judge_trappist_orbits(&orbits(7)).0);
        assert!(!judge_trappist_orbits(&orbits(6)).0);
        let rows = |list: &[&str]| {
            v(&format!(
                r#"{{"hud": {{"card_rows": {}}}}}"#,
                Value::from(list.to_vec())
            ))
        };
        assert!(judge_card_rows(&rows(&MARS_CARD_ROWS)).0);
        let mut swapped = MARS_CARD_ROWS;
        swapped.swap(3, 4);
        assert!(!judge_card_rows(&rows(&swapped)).0);
        assert!(!judge_card_rows(&rows(&MARS_CARD_ROWS[..14])).0);
        let honesty = |labels: &[&str]| {
            v(&format!(
                r#"{{"honesty": {}}}"#,
                Value::from(labels.to_vec())
            ))
        };
        assert!(judge_cruise_label(&honesty(&[CRUISE_LABEL])).0);
        assert!(!judge_cruise_label(&honesty(&["CRUISE"])).0);
        assert!(judge_extrapolation_labels(&honesty(&[PLANETS_LABEL])).0);
        assert!(!judge_extrapolation_labels(&honesty(&[PLANETS_LABEL, STARS_LABEL])).0);
        assert!(!judge_extrapolation_labels(&honesty(&[])).0);
    }

    /// End to end on a small capture directory: a points check, JSON checks in both directories, a
    /// missing capture, a missing `.hdr.bin` (then present), a disk body missing from
    /// `projected`; `checks.json` holds every result.
    #[test]
    fn runs_checks_and_writes_checks_json() {
        let caps = temp_dir("gate_caps");
        let hud = temp_dir("gate_hud");
        // A 32 × 32 black frame with one bright pixel at (10, 12).
        let mut img = RgbImage::new(32, 32);
        img.put_pixel(10, 12, Rgb([255, 255, 255]));
        img.save(caps.join("sky.png")).expect("write PNG");
        fs::write(
            caps.join("sky.json"),
            r#"{"ev100": [-4.0, -6.5], "projected": [{"name": "Vega", "x": 10.5, "y": 12.5, "radius_px": 0.0}],
                "hdr": {"file": "sky.hdr.bin", "width": 32, "height": 32, "fov_y_rad": 1.0,
                        "pre": [1.0, 1.0], "ev100": [-4.0, -6.5]},
                "regions": {"a": [[0, 0]]}}"#,
        )
        .expect("write JSON");
        fs::write(
            hud.join("card.json"),
            format!(
                r#"{{"hud": {{"card_rows": {}}}}}"#,
                Value::from(MARS_CARD_ROWS.to_vec())
            ),
        )
        .expect("write HUD JSON");
        let checks = [
            cli(
                1,
                "sky",
                "points Vega ±1px",
                &[
                    "points", "--png", "sky.png", "--json", "sky.json", "--names", "Vega", "--tol",
                    "1",
                ],
            ),
            json(11, "sky", "ev100", judge_ev100),
            Check {
                row: 24,
                bookmark: "card",
                check: "card rows",
                how: How::Json {
                    dir: Dir::Hud,
                    judge: judge_card_rows,
                },
            },
            cli(
                2,
                "sky",
                "ratio a/a ≥ 1 (no sidecar)",
                &[
                    "ratio",
                    "--hdr-json",
                    "sky.json",
                    "--num",
                    "a",
                    "--den",
                    "a",
                    "--min",
                    "1",
                ],
            ),
            json(25, "absent", "no such capture", judge_cruise_label),
            cli(
                6,
                "sky",
                "points Deneb",
                &[
                    "points", "--png", "sky.png", "--json", "sky.json", "--names", "Deneb",
                    "--tol", "1",
                ],
            ),
            Check {
                row: 7,
                bookmark: "sky",
                check: "disk of a body not projected",
                how: How::Disk {
                    body: "Jupiter",
                    args: &[
                        "disk",
                        "--png",
                        "sky.png",
                        "--expected-a",
                        "5",
                        "--tol",
                        "1",
                    ],
                },
            },
        ];
        let out = caps.join("out.json");
        let verdict = run_checks(&checks, &caps, &hud, &out);
        let written = fs::read_to_string(&out);
        // With the sidecar written (pixel (0, 0) lit), the HDR check runs and passes.
        let sidecar: Vec<u8> = (0..32 * 32)
            .flat_map(|i| {
                let v: f32 = if i == 0 { 1.0 } else { 0.0 };
                [v, v, v, 1.0]
            })
            .flat_map(f32::to_le_bytes)
            .collect();
        fs::write(caps.join("sky.hdr.bin"), sidecar).expect("write sidecar");
        let with_sidecar = run_check(&checks[3], &caps, &hud);
        let unwritable = run_checks(
            &checks[..1],
            &caps,
            &hud,
            &caps.join("no-dir").join("x.json"),
        );
        let _ = fs::remove_dir_all(&caps);
        let _ = fs::remove_dir_all(&hud);
        let verdict = verdict.expect("gate runs");
        assert!(unwritable.is_err());

        let results: Vec<Value> = serde_json::from_str(&written.expect("checks.json written"))
            .expect("checks.json parses");
        assert_eq!(results.len(), checks.len());
        let pass: Vec<bool> = results
            .iter()
            .map(|r| r["pass"].as_bool().expect("pass"))
            .collect();
        assert_eq!(
            pass,
            [true, true, true, false, false, false, false],
            "{results:#?}"
        );
        for (r, c) in results.iter().zip(&checks) {
            assert_eq!(r["row"], c.row);
            assert_eq!(r["bookmark"], c.bookmark);
            assert_eq!(r["check"], c.check);
            let output = r["output"].as_str().expect("output");
            assert!(
                output.ends_with(if r["pass"] == true { "PASS" } else { "FAIL" }),
                "{output}"
            );
        }
        assert!(
            results[0]["output"]
                .as_str()
                .is_some_and(|o| o.starts_with("points: 1 of 1 within 1 px"))
        );
        assert!(
            results[3]["output"]
                .as_str()
                .is_some_and(|o| o.starts_with("error: cannot read") && o.contains("sky.hdr.bin"))
        );
        assert!(
            results[4]["output"]
                .as_str()
                .is_some_and(|o| o.contains("absent.json"))
        );
        assert!(
            results[6]["output"]
                .as_str()
                .is_some_and(|o| o.contains("Jupiter is not in sky.json")),
            "{results:#?}"
        );
        assert!(with_sidecar.pass, "{with_sidecar:?}");
        assert_eq!(with_sidecar.output, "ratio 1.00000 ≥ 1 PASS");
        assert!(!verdict.pass);
        assert_eq!(
            verdict.lines.last().map(String::as_str),
            Some("gate: 3/7 checks FAIL")
        );
        assert!(
            verdict.lines[0].starts_with("PASS 1 sky | points Vega ±1px | points: 1 of 1"),
            "{verdict:#?}"
        );
    }
}
