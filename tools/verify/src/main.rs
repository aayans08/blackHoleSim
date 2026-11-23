//! uni-verify: checks headless captures (contact sheets, shadow, disk, points, HDR photometry,
//! bench timings).
//!
//! - `sheet --captures <dir> --out <dir>/index.html [--bench <json>]` writes an HTML sheet of
//!   every PNG in `<dir>` with its JSON metrics, plus a "Checks" table from
//!   `<dir>/checks.json` and a "Performance" table from the bench JSON (default
//!   `<dir>/bench.json`; see `sheet`).
//! - `contact --captures <dir> --out <png> [--width 2400]` tiles every PNG into one labelled grid
//!   PNG and prints the legend (see `contact`).
//! - `shadow (--png <file> | --hdr-json <capture.json>) --expected-px <r> --tol <px>` measures a
//!   black-hole shadow's equivalent-disc radius (see `shadow`).
//! - `disk --png <f> --center <x>,<y> [--threshold 0.008] --expected-a <px> [--expected-b <px>]
//!   --tol <px> [--ratio <b/a> --ratio-tol <t>] [--exclude-angle <deg> --exclude-width <deg>]`
//!   fits an ellipse to a body's limb (see `disk`).
//! - `points --png <f> --json <capture.json> --names <a,b,…> --tol <px>` checks projected
//!   positions against brightness centroids (see `points`).
//! - `bench --json <bench.json> --max-gpu-ms <ms> --max-cpu-ms <ms> --max-interval-ms <ms>
//!   [--window last10s|first10s|all] [--budgets]` checks bench p95 timings (see `bench`).
//! - HDR checks on the `--hdr` capture sidecar (see `hdr`): `ratio`, `stats`, `samples`,
//!   `profile`, `angle`, `flux`, `limb`, each with `--hdr-json <capture.json>`. They may also
//!   be written `hdr <check> …`.
//!
//! Every check prints its numbers and a last line ending in PASS or FAIL. Exit codes: 0 on
//! success or PASS, 1 on FAIL, 2 on a bad command line or an I/O or decode error (the error goes
//! to stderr).

mod bench;
mod contact;
mod disk;
mod hdr;
mod points;
mod shadow;
mod sheet;

use anyhow::{Context, bail};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  uni-verify sheet --captures <dir> --out <dir>/index.html [--bench <bench.json>]
  uni-verify contact --captures <dir> --out <png> [--width 2400]
  uni-verify shadow (--png <file> | --hdr-json <capture.json>) --expected-px <f64> --tol <f64>
  uni-verify disk --png <file> --center <x>,<y> [--threshold 0.008] --expected-a <px>
      [--expected-b <px>] --tol <px> [--ratio <b/a> --ratio-tol <t>]
      [--exclude-angle <deg> --exclude-width <deg>]
      (angles counter-clockwise from +x with y up; a width below 180 also excludes the opposite
      wedge, a width of 180 or more excludes only the wedge itself, e.g. a terminator side)
  uni-verify points --png <file> --json <capture.json> --names <a,b,...> --tol <px>
  uni-verify bench --json <bench.json> --max-gpu-ms <ms> --max-cpu-ms <ms> --max-interval-ms <ms>
      [--window last10s|first10s|all] [--budgets]
  uni-verify ratio --hdr-json <capture.json> --num <region> --den <region>
      [--channels r/b|b/r|g|luma] [--stat mean|median] [--min x] [--max x]
      [--expect x --rel-tol t]
  uni-verify stats --hdr-json <capture.json> --region <name> [--rms-min x] [--rms-max x]
      [--mean-pre-min x]
  uni-verify samples --hdr-json <capture.json> --tol <rel>
  uni-verify profile --hdr-json <capture.json> --feature <name> --tol <slope>
  uni-verify angle --hdr-json <capture.json> --feature <name> --tol-deg <deg>
  uni-verify flux --hdr-json <capture.json> --name <star> --tol <rel>
  uni-verify limb --hdr-json <capture.json> --name <limb> --tol <rel>
  (the HDR checks may also be written `uni-verify hdr <check> ...`)";

/// Exit code when a check fails.
const EXIT_FAIL: u8 = 1;
/// Exit code for a bad command line or an I/O or decode error.
const EXIT_ERROR: u8 = 2;

/// The outcome of one check: the lines to print (numbers first, the last one ending in PASS or
/// FAIL) and whether it passed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Verdict {
    pub lines: Vec<String>,
    pub pass: bool,
}

/// "PASS" or "FAIL".
pub(crate) fn pass_word(pass: bool) -> &'static str {
    if pass { "PASS" } else { "FAIL" }
}

/// Where `shadow` reads the image from.
#[derive(Debug, PartialEq)]
enum ShadowSource {
    /// A tonemapped capture PNG: black is luma below 2/255.
    Png(PathBuf),
    /// A capture JSON with an `hdr` header: dark is below half the median background.
    Hdr(PathBuf),
}

/// A parsed command line.
#[derive(Debug, PartialEq)]
enum Command {
    Sheet {
        captures: PathBuf,
        out: PathBuf,
        bench: Option<PathBuf>,
    },
    Contact {
        captures: PathBuf,
        out: PathBuf,
        width: u32,
    },
    Shadow {
        source: ShadowSource,
        expected_px: f64,
        tol: f64,
    },
    Disk(disk::DiskArgs),
    Points {
        png: PathBuf,
        json: PathBuf,
        names: Vec<String>,
        tol: f64,
    },
    Bench(bench::BenchArgs),
    Ratio {
        hdr_json: PathBuf,
        args: hdr::RatioArgs,
    },
    Stats {
        hdr_json: PathBuf,
        args: hdr::StatsArgs,
    },
    Samples {
        hdr_json: PathBuf,
        tol: f64,
    },
    Profile {
        hdr_json: PathBuf,
        feature: String,
        tol: f64,
    },
    Angle {
        hdr_json: PathBuf,
        feature: String,
        tol_deg: f64,
    },
    Flux {
        hdr_json: PathBuf,
        name: String,
        tol: f64,
    },
    Limb {
        hdr_json: PathBuf,
        name: String,
        tol: f64,
    },
}

/// The options given to one subcommand: `--flag value` pairs and bare `--switch`es.
struct Options {
    values: BTreeMap<&'static str, OsString>,
    switches: BTreeSet<&'static str>,
}

/// Parses `options` against the flags a subcommand accepts. Every flag may be given at most
/// once; an unknown flag or a value flag without its value is an error.
fn parse_options(
    options: &[OsString],
    value_flags: &[&'static str],
    switch_flags: &[&'static str],
) -> anyhow::Result<Options> {
    let mut values = BTreeMap::new();
    let mut switches = BTreeSet::new();
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        if let Some(&name) = switch_flags.iter().find(|name| flag == **name) {
            if !switches.insert(name) {
                bail!("{flag:?} given twice\n{USAGE}");
            }
        } else if let Some(&name) = value_flags.iter().find(|name| flag == **name) {
            let value = options
                .next()
                .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
            if values.insert(name, value.clone()).is_some() {
                bail!("{flag:?} given twice\n{USAGE}");
            }
        } else {
            bail!("unknown option {flag:?}\n{USAGE}");
        }
    }
    Ok(Options { values, switches })
}

/// Which numbers a flag accepts.
#[derive(Clone, Copy)]
enum Range {
    /// Any finite number.
    Finite,
    /// A finite number ≥ 0.
    NonNegative,
    /// A finite number > 0.
    Positive,
}

impl Options {
    fn required(&self, name: &str) -> anyhow::Result<&OsString> {
        self.values
            .get(name)
            .with_context(|| format!("missing {name}\n{USAGE}"))
    }

    fn path(&self, name: &str) -> anyhow::Result<PathBuf> {
        self.required(name).map(PathBuf::from)
    }

    fn opt_path(&self, name: &str) -> Option<PathBuf> {
        self.values.get(name).map(PathBuf::from)
    }

    fn text(&self, name: &str) -> anyhow::Result<String> {
        let value = self.required(name)?;
        value
            .to_str()
            .map(str::to_owned)
            .with_context(|| format!("{name} needs UTF-8 text, not {value:?}\n{USAGE}"))
    }

    fn opt_text(&self, name: &str) -> anyhow::Result<Option<String>> {
        match self.values.contains_key(name) {
            true => self.text(name).map(Some),
            false => Ok(None),
        }
    }

    fn number(&self, name: &str, range: Range) -> anyhow::Result<f64> {
        parse_number(name, self.required(name)?, range)
    }

    fn opt_number(&self, name: &str, range: Range) -> anyhow::Result<Option<f64>> {
        self.values
            .get(name)
            .map(|value| parse_number(name, value, range))
            .transpose()
    }

    /// Two flags that must be given together, or neither.
    fn opt_pair(
        &self,
        first: (&str, Range),
        second: (&str, Range),
    ) -> anyhow::Result<Option<(f64, f64)>> {
        match (
            self.opt_number(first.0, first.1)?,
            self.opt_number(second.0, second.1)?,
        ) {
            (Some(a), Some(b)) => Ok(Some((a, b))),
            (None, None) => Ok(None),
            _ => bail!("{} and {} go together\n{USAGE}", first.0, second.0),
        }
    }

    fn switch(&self, name: &str) -> bool {
        self.switches.contains(name)
    }
}

/// Parses a number for `flag` and checks it is in `range`.
fn parse_number(flag: &str, value: &OsString, range: Range) -> anyhow::Result<f64> {
    let parsed = value.to_str().and_then(|s| s.trim().parse::<f64>().ok());
    let ok = |x: f64| match range {
        Range::Finite => x.is_finite(),
        Range::NonNegative => x.is_finite() && x >= 0.0,
        Range::Positive => x.is_finite() && x > 0.0,
    };
    match parsed {
        Some(x) if ok(x) => Ok(x),
        _ => {
            let what = match range {
                Range::Finite => "a finite number",
                Range::NonNegative => "a finite, non-negative number",
                Range::Positive => "a finite, positive number",
            };
            bail!("{flag} needs {what}, not {value:?}\n{USAGE}")
        }
    }
}

/// Parses `<x>,<y>` (two finite numbers).
fn parse_point(flag: &str, value: &OsString) -> anyhow::Result<(f64, f64)> {
    let parsed = value.to_str().and_then(|s| {
        let (x, y) = s.split_once(',')?;
        let x = x.trim().parse::<f64>().ok()?;
        let y = y.trim().parse::<f64>().ok()?;
        (x.is_finite() && y.is_finite()).then_some((x, y))
    });
    parsed.with_context(|| format!("{flag} needs <x>,<y>, not {value:?}\n{USAGE}"))
}

/// Parses the arguments after the program name.
fn parse_args(args: &[OsString]) -> anyhow::Result<Command> {
    let Some((command, options)) = args.split_first() else {
        bail!("no command given\n{USAGE}");
    };
    // `uni-verify hdr <check> …` is the same as `uni-verify <check> …`.
    if command == "hdr" {
        let Some((check, options)) = options.split_first() else {
            bail!("hdr needs a check name\n{USAGE}");
        };
        let is_hdr_check = [
            "ratio", "stats", "samples", "profile", "angle", "flux", "limb",
        ]
        .iter()
        .any(|name| check == *name);
        if !is_hdr_check {
            bail!("unknown HDR check {check:?}\n{USAGE}");
        }
        return parse_command(check, options);
    }
    parse_command(command, options)
}

/// Parses one subcommand and its options.
fn parse_command(command: &OsString, options: &[OsString]) -> anyhow::Result<Command> {
    let Some(name) = command.to_str() else {
        bail!("unknown command {command:?}\n{USAGE}");
    };
    match name {
        "sheet" => {
            let o = parse_options(options, &["--captures", "--out", "--bench"], &[])?;
            Ok(Command::Sheet {
                captures: o.path("--captures")?,
                out: o.path("--out")?,
                bench: o.opt_path("--bench"),
            })
        }
        "contact" => {
            let o = parse_options(options, &["--captures", "--out", "--width"], &[])?;
            let width = match o.values.get("--width") {
                None => contact::DEFAULT_WIDTH,
                Some(w) => w
                    .to_str()
                    .and_then(|s| s.parse::<u32>().ok())
                    .filter(|&w| (64..=16384).contains(&w))
                    .with_context(|| {
                        format!("--width needs 64..=16384 pixels, not {w:?}\n{USAGE}")
                    })?,
            };
            Ok(Command::Contact {
                captures: o.path("--captures")?,
                out: o.path("--out")?,
                width,
            })
        }
        "shadow" => {
            let o = parse_options(
                options,
                &["--png", "--hdr-json", "--expected-px", "--tol"],
                &[],
            )?;
            let source = match (o.opt_path("--png"), o.opt_path("--hdr-json")) {
                (Some(png), None) => ShadowSource::Png(png),
                (None, Some(json)) => ShadowSource::Hdr(json),
                _ => bail!("shadow needs exactly one of --png and --hdr-json\n{USAGE}"),
            };
            Ok(Command::Shadow {
                source,
                expected_px: o.number("--expected-px", Range::NonNegative)?,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        "disk" => parse_disk(options),
        "points" => {
            let o = parse_options(options, &["--png", "--json", "--names", "--tol"], &[])?;
            let names: Vec<String> = o
                .text("--names")?
                .split(',')
                .map(|n| n.trim().to_owned())
                .filter(|n| !n.is_empty())
                .collect();
            if names.is_empty() {
                bail!("--names needs at least one name\n{USAGE}");
            }
            Ok(Command::Points {
                png: o.path("--png")?,
                json: o.path("--json")?,
                names,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        "bench" => {
            let o = parse_options(
                options,
                &[
                    "--json",
                    "--max-gpu-ms",
                    "--max-cpu-ms",
                    "--max-interval-ms",
                    "--window",
                ],
                &["--budgets"],
            )?;
            let window = match o.opt_text("--window")? {
                None => bench::Window::Last10s,
                Some(w) => bench::Window::parse(&w).with_context(|| {
                    format!("--window needs last10s, first10s or all, not {w:?}\n{USAGE}")
                })?,
            };
            Ok(Command::Bench(bench::BenchArgs {
                json: o.path("--json")?,
                limits: bench::Limits {
                    max_gpu_ms: o.number("--max-gpu-ms", Range::Positive)?,
                    max_cpu_ms: o.number("--max-cpu-ms", Range::Positive)?,
                    max_interval_ms: o.number("--max-interval-ms", Range::Positive)?,
                },
                window,
                budgets: o.switch("--budgets"),
            }))
        }
        "ratio" => {
            let o = parse_options(
                options,
                &[
                    "--hdr-json",
                    "--num",
                    "--den",
                    "--channels",
                    "--stat",
                    "--min",
                    "--max",
                    "--expect",
                    "--rel-tol",
                ],
                &[],
            )?;
            let channels = match o.opt_text("--channels")? {
                None => hdr::Channels::Luma,
                Some(c) => hdr::Channels::parse(&c).with_context(|| {
                    format!("--channels needs r/b, b/r, g or luma, not {c:?}\n{USAGE}")
                })?,
            };
            let stat = match o.opt_text("--stat")? {
                None => hdr::Stat::Median,
                Some(s) => hdr::Stat::parse(&s)
                    .with_context(|| format!("--stat needs mean or median, not {s:?}\n{USAGE}"))?,
            };
            let bounds = hdr::Bounds {
                min: o.opt_number("--min", Range::Finite)?,
                max: o.opt_number("--max", Range::Finite)?,
                expect: o.opt_pair(
                    ("--expect", Range::Finite),
                    ("--rel-tol", Range::NonNegative),
                )?,
            };
            if bounds.is_empty() {
                bail!("ratio needs --min, --max or --expect with --rel-tol\n{USAGE}");
            }
            Ok(Command::Ratio {
                hdr_json: o.path("--hdr-json")?,
                args: hdr::RatioArgs {
                    num: o.text("--num")?,
                    den: o.text("--den")?,
                    channels,
                    stat,
                    bounds,
                },
            })
        }
        "stats" => {
            let o = parse_options(
                options,
                &[
                    "--hdr-json",
                    "--region",
                    "--rms-min",
                    "--rms-max",
                    "--mean-pre-min",
                ],
                &[],
            )?;
            let args = hdr::StatsArgs {
                region: o.text("--region")?,
                rms_min: o.opt_number("--rms-min", Range::NonNegative)?,
                rms_max: o.opt_number("--rms-max", Range::NonNegative)?,
                mean_pre_min: o.opt_number("--mean-pre-min", Range::Finite)?,
            };
            if args.rms_min.is_none() && args.rms_max.is_none() && args.mean_pre_min.is_none() {
                bail!("stats needs --rms-min, --rms-max or --mean-pre-min\n{USAGE}");
            }
            Ok(Command::Stats {
                hdr_json: o.path("--hdr-json")?,
                args,
            })
        }
        "samples" => {
            let o = parse_options(options, &["--hdr-json", "--tol"], &[])?;
            Ok(Command::Samples {
                hdr_json: o.path("--hdr-json")?,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        "profile" => {
            let o = parse_options(options, &["--hdr-json", "--feature", "--tol"], &[])?;
            Ok(Command::Profile {
                hdr_json: o.path("--hdr-json")?,
                feature: o.text("--feature")?,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        "angle" => {
            let o = parse_options(options, &["--hdr-json", "--feature", "--tol-deg"], &[])?;
            Ok(Command::Angle {
                hdr_json: o.path("--hdr-json")?,
                feature: o.text("--feature")?,
                tol_deg: o.number("--tol-deg", Range::NonNegative)?,
            })
        }
        "flux" => {
            let o = parse_options(options, &["--hdr-json", "--name", "--tol"], &[])?;
            Ok(Command::Flux {
                hdr_json: o.path("--hdr-json")?,
                name: o.text("--name")?,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        "limb" => {
            let o = parse_options(options, &["--hdr-json", "--name", "--tol"], &[])?;
            Ok(Command::Limb {
                hdr_json: o.path("--hdr-json")?,
                name: o.text("--name")?,
                tol: o.number("--tol", Range::NonNegative)?,
            })
        }
        _ => bail!("unknown command {command:?}\n{USAGE}"),
    }
}

/// Parses the `disk` subcommand.
fn parse_disk(options: &[OsString]) -> anyhow::Result<Command> {
    let o = parse_options(
        options,
        &[
            "--png",
            "--center",
            "--threshold",
            "--expected-a",
            "--expected-b",
            "--tol",
            "--ratio",
            "--ratio-tol",
            "--exclude-angle",
            "--exclude-width",
        ],
        &[],
    )?;
    let threshold = o
        .opt_number("--threshold", Range::Positive)?
        .unwrap_or(disk::DEFAULT_THRESHOLD);
    if threshold > 1.0 {
        bail!("--threshold is a luma in (0, 1], not {threshold}\n{USAGE}");
    }
    let exclusion = o
        .opt_pair(
            ("--exclude-angle", Range::Finite),
            ("--exclude-width", Range::NonNegative),
        )?
        .map(|(angle_deg, width_deg)| disk::Exclusion {
            angle_deg,
            width_deg,
        });
    if exclusion.is_some_and(|e| e.width_deg >= 360.0) {
        bail!("--exclude-width must be below 360 degrees\n{USAGE}");
    }
    Ok(Command::Disk(disk::DiskArgs {
        png: o.path("--png")?,
        center: parse_point("--center", o.required("--center")?)?,
        threshold,
        expected_a: o.number("--expected-a", Range::Positive)?,
        expected_b: o.opt_number("--expected-b", Range::Positive)?,
        tol: o.number("--tol", Range::NonNegative)?,
        ratio: o.opt_pair(
            ("--ratio", Range::Positive),
            ("--ratio-tol", Range::NonNegative),
        )?,
        exclusion,
    }))
}

/// Prints a check's lines and turns its verdict into an exit code.
fn finish(verdict: Verdict) -> ExitCode {
    for line in &verdict.lines {
        println!("{line}");
    }
    if verdict.pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(EXIT_FAIL)
    }
}

/// Runs a parsed command; `Ok` carries the exit code for a completed run.
fn run(command: Command) -> anyhow::Result<ExitCode> {
    match command {
        Command::Sheet {
            captures,
            out,
            bench,
        } => {
            let count = sheet::write_sheet(&captures, &out, bench.as_deref())?;
            println!("sheet: {count} captures -> {}", out.display());
            Ok(ExitCode::SUCCESS)
        }
        Command::Contact {
            captures,
            out,
            width,
        } => {
            let legend = contact::write_contact(&captures, &out, width)?;
            println!("contact: {} captures -> {}", legend.len(), out.display());
            for (n, name) in legend {
                println!("  {n} {name}");
            }
            Ok(ExitCode::SUCCESS)
        }
        Command::Shadow {
            source: ShadowSource::Png(png),
            expected_px,
            tol,
        } => {
            let measured = shadow::shadow_radius(&shadow::load_rgb8(&png)?);
            let pass = shadow::passes(measured, expected_px, tol);
            println!("{}", shadow::report_line(measured, expected_px, pass));
            Ok(finish(Verdict {
                lines: Vec::new(),
                pass,
            }))
        }
        Command::Shadow {
            source: ShadowSource::Hdr(json),
            expected_px,
            tol,
        } => {
            let capture = hdr::HdrCapture::load(&json)?;
            Ok(finish(shadow::check_hdr(&capture, expected_px, tol)?))
        }
        Command::Disk(args) => Ok(finish(disk::run(&args)?)),
        Command::Points {
            png,
            json,
            names,
            tol,
        } => Ok(finish(points::run(&png, &json, &names, tol)?)),
        Command::Bench(args) => Ok(finish(bench::run(&args)?)),
        Command::Ratio { hdr_json, args } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_ratio(&capture, &args)?))
        }
        Command::Stats { hdr_json, args } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_stats(&capture, &args)?))
        }
        Command::Samples { hdr_json, tol } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_samples(&capture, tol)?))
        }
        Command::Profile {
            hdr_json,
            feature,
            tol,
        } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_profile(&capture, &feature, tol)?))
        }
        Command::Angle {
            hdr_json,
            feature,
            tol_deg,
        } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_angle(&capture, &feature, tol_deg)?))
        }
        Command::Flux {
            hdr_json,
            name,
            tol,
        } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_flux(&capture, &name, tol)?))
        }
        Command::Limb {
            hdr_json,
            name,
            tol,
        } => {
            let capture = hdr::HdrCapture::load(&hdr_json)?;
            Ok(finish(hdr::check_limb(&capture, &name, tol)?))
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match parse_args(&args).and_then(run) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("uni-verify: {error:#}");
            ExitCode::from(EXIT_ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    fn parse(list: &[&str]) -> Command {
        parse_args(&args(list)).unwrap_or_else(|e| panic!("{list:?}: {e:#}"))
    }

    #[test]
    fn parses_sheet_and_shadow_commands() {
        assert_eq!(
            parse(&[
                "sheet",
                "--captures",
                "captures/m0",
                "--out",
                "captures/m0/index.html",
            ]),
            Command::Sheet {
                captures: PathBuf::from("captures/m0"),
                out: PathBuf::from("captures/m0/index.html"),
                bench: None,
            }
        );
        assert_eq!(
            parse(&[
                "sheet",
                "--captures",
                "c",
                "--out",
                "o.html",
                "--bench",
                "b.json"
            ]),
            Command::Sheet {
                captures: PathBuf::from("c"),
                out: PathBuf::from("o.html"),
                bench: Some(PathBuf::from("b.json")),
            }
        );
        // Flags in any order.
        assert_eq!(
            parse(&[
                "shadow",
                "--tol",
                "1",
                "--png",
                "bh.png",
                "--expected-px",
                "158.745",
            ]),
            Command::Shadow {
                source: ShadowSource::Png(PathBuf::from("bh.png")),
                expected_px: 158.745,
                tol: 1.0,
            }
        );
        assert_eq!(
            parse(&[
                "shadow",
                "--hdr-json",
                "bh.json",
                "--expected-px",
                "158.745",
                "--tol",
                "1.0",
            ]),
            Command::Shadow {
                source: ShadowSource::Hdr(PathBuf::from("bh.json")),
                expected_px: 158.745,
                tol: 1.0,
            }
        );
    }

    #[test]
    fn parses_the_contact_command() {
        let c = parse(&[
            "contact",
            "--captures",
            "captures/m0",
            "--out",
            "/tmp/m0.png",
        ]);
        assert_eq!(
            c,
            Command::Contact {
                captures: PathBuf::from("captures/m0"),
                out: PathBuf::from("/tmp/m0.png"),
                width: contact::DEFAULT_WIDTH,
            }
        );
        let c = parse(&[
            "contact",
            "--width",
            "1200",
            "--out",
            "o.png",
            "--captures",
            "c",
        ]);
        assert_eq!(
            c,
            Command::Contact {
                captures: PathBuf::from("c"),
                out: PathBuf::from("o.png"),
                width: 1200,
            }
        );
    }

    #[test]
    fn parses_disk_points_and_bench() {
        assert_eq!(
            parse(&[
                "disk",
                "--png",
                "saturn.png",
                "--center",
                "960.5,540",
                "--expected-a",
                "433.6",
                "--tol",
                "1",
                "--ratio",
                "0.9020",
                "--ratio-tol",
                "0.004",
                "--exclude-angle",
                "0",
                "--exclude-width",
                "10",
            ]),
            Command::Disk(disk::DiskArgs {
                png: PathBuf::from("saturn.png"),
                center: (960.5, 540.0),
                threshold: disk::DEFAULT_THRESHOLD,
                expected_a: 433.6,
                expected_b: None,
                tol: 1.0,
                ratio: Some((0.902, 0.004)),
                exclusion: Some(disk::Exclusion {
                    angle_deg: 0.0,
                    width_deg: 10.0,
                }),
            })
        );
        assert_eq!(
            parse(&[
                "points",
                "--png",
                "p.png",
                "--json",
                "p.json",
                "--names",
                "Betelgeuse, Rigel,Bellatrix",
                "--tol",
                "1",
            ]),
            Command::Points {
                png: PathBuf::from("p.png"),
                json: PathBuf::from("p.json"),
                names: vec!["Betelgeuse".into(), "Rigel".into(), "Bellatrix".into()],
                tol: 1.0,
            }
        );
        assert_eq!(
            parse(&[
                "bench",
                "--json",
                "b.json",
                "--window",
                "last10s",
                "--max-gpu-ms",
                "16.0",
                "--max-cpu-ms",
                "4.0",
                "--max-interval-ms",
                "16.7",
                "--budgets",
            ]),
            Command::Bench(bench::BenchArgs {
                json: PathBuf::from("b.json"),
                limits: bench::Limits {
                    max_gpu_ms: 16.0,
                    max_cpu_ms: 4.0,
                    max_interval_ms: 16.7,
                },
                window: bench::Window::Last10s,
                budgets: true,
            })
        );
    }

    #[test]
    fn parses_hdr_checks() {
        let ratio = parse(&[
            "ratio",
            "--hdr-json",
            "c.json",
            "--num",
            "limb_arc_high",
            "--den",
            "limb_arc_high",
            "--channels",
            "b/r",
            "--min",
            "1.5",
        ]);
        assert_eq!(
            ratio,
            Command::Ratio {
                hdr_json: PathBuf::from("c.json"),
                args: hdr::RatioArgs {
                    num: "limb_arc_high".into(),
                    den: "limb_arc_high".into(),
                    channels: hdr::Channels::BlueOverRed,
                    stat: hdr::Stat::Median,
                    bounds: hdr::Bounds {
                        min: Some(1.5),
                        max: None,
                        expect: None,
                    },
                },
            }
        );
        // The `hdr` prefix is optional.
        let flux = parse(&[
            "hdr",
            "flux",
            "--hdr-json",
            "c.json",
            "--name",
            "Rigel",
            "--tol",
            "0.05",
        ]);
        assert_eq!(
            flux,
            Command::Flux {
                hdr_json: PathBuf::from("c.json"),
                name: "Rigel".into(),
                tol: 0.05,
            }
        );
        assert_eq!(
            parse(&[
                "limb",
                "--hdr-json",
                "c.json",
                "--name",
                "venus_night_limb",
                "--tol",
                "0.1",
            ]),
            Command::Limb {
                hdr_json: PathBuf::from("c.json"),
                name: "venus_night_limb".into(),
                tol: 0.1,
            }
        );
        assert_eq!(
            parse(&[
                "stats",
                "--hdr-json",
                "c.json",
                "--region",
                "granulation_patch",
                "--rms-min",
                "0.08",
                "--rms-max",
                "0.20",
            ]),
            Command::Stats {
                hdr_json: PathBuf::from("c.json"),
                args: hdr::StatsArgs {
                    region: "granulation_patch".into(),
                    rms_min: Some(0.08),
                    rms_max: Some(0.2),
                    mean_pre_min: None,
                },
            }
        );
        assert_eq!(
            parse(&[
                "angle",
                "--hdr-json",
                "c.json",
                "--feature",
                "ion_tail",
                "--tol-deg",
                "1"
            ]),
            Command::Angle {
                hdr_json: PathBuf::from("c.json"),
                feature: "ion_tail".into(),
                tol_deg: 1.0,
            }
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        let bad: &[&[&str]] = &[
            &[],
            &["measure"],
            &["hdr"],
            &["hdr", "disk"],
            &["sheet", "--captures", "c"],
            &["sheet", "--captures", "c", "--out"],
            &["sheet", "--captures", "c", "--captures", "d", "--out", "o"],
            &["sheet", "--captures", "c", "--out", "o", "--png", "p"],
            &["shadow", "--png", "p", "--expected-px", "abc", "--tol", "1"],
            &[
                "shadow",
                "--png",
                "p",
                "--expected-px",
                "158",
                "--tol",
                "-1",
            ],
            &["shadow", "--png", "p", "--expected-px", "NaN", "--tol", "1"],
            &[
                "shadow",
                "--png",
                "p",
                "--expected-px",
                "158",
                "--tol",
                "inf",
            ],
            &["shadow", "--png", "p", "--expected-px", "158"],
            &["shadow", "--expected-px", "158", "--tol", "1"],
            &[
                "shadow",
                "--png",
                "p",
                "--hdr-json",
                "j",
                "--expected-px",
                "158",
                "--tol",
                "1",
            ],
            &["contact", "--captures", "c"],
            &["contact", "--captures", "c", "--out", "o", "--width", "0"],
            &[
                "contact",
                "--captures",
                "c",
                "--out",
                "o",
                "--width",
                "wide",
            ],
            &[
                "contact",
                "--captures",
                "c",
                "--out",
                "o",
                "--width",
                "1",
                "--width",
                "2",
            ],
            &["sheet", "--captures", "c", "--out", "o", "--width", "100"],
            // disk: bad centre, lone half of a pair, threshold out of range.
            &[
                "disk",
                "--png",
                "p",
                "--center",
                "960",
                "--expected-a",
                "1",
                "--tol",
                "1",
            ],
            &[
                "disk",
                "--png",
                "p",
                "--center",
                "1,2",
                "--expected-a",
                "1",
                "--tol",
                "1",
                "--ratio",
                "0.9",
            ],
            &[
                "disk",
                "--png",
                "p",
                "--center",
                "1,2",
                "--expected-a",
                "1",
                "--tol",
                "1",
                "--exclude-width",
                "10",
            ],
            &[
                "disk",
                "--png",
                "p",
                "--center",
                "1,2",
                "--expected-a",
                "1",
                "--tol",
                "1",
                "--threshold",
                "2",
            ],
            &[
                "points", "--png", "p", "--json", "j", "--names", ",", "--tol", "1",
            ],
            // bench: unknown window, a value given to the switch.
            &[
                "bench",
                "--json",
                "b",
                "--max-gpu-ms",
                "16",
                "--max-cpu-ms",
                "4",
                "--max-interval-ms",
                "16.7",
                "--window",
                "last5s",
            ],
            &[
                "bench",
                "--json",
                "b",
                "--max-gpu-ms",
                "16",
                "--max-cpu-ms",
                "4",
                "--max-interval-ms",
                "16.7",
                "--budgets",
                "--budgets",
            ],
            // ratio: no criterion, a lone --expect, unknown channels.
            &["ratio", "--hdr-json", "j", "--num", "a", "--den", "b"],
            &[
                "ratio",
                "--hdr-json",
                "j",
                "--num",
                "a",
                "--den",
                "b",
                "--expect",
                "1",
            ],
            &[
                "ratio",
                "--hdr-json",
                "j",
                "--num",
                "a",
                "--den",
                "b",
                "--min",
                "1",
                "--channels",
                "r/g",
            ],
            &["stats", "--hdr-json", "j", "--region", "a"],
        ];
        for list in bad {
            assert!(parse_args(&args(list)).is_err(), "accepted {list:?}");
        }
    }
}
