//! universe: the explorer binary. Windowed by default; `--capture`, `--bench` and `--selftest`
//! run headless.
//!
//! Exit codes: 0 on success; 2 when a baked data file the run needs is missing or unreadable
//! (`--capture`, `--bench`); 1 for every other failure, including bad arguments, GPU errors and
//! failed `--selftest` checks.

mod app;
mod bench;
#[cfg(test)]
mod bookmarks;
mod capture;
mod measure;
mod paper;
mod present;
mod scene;
mod selftest;
mod tap;
mod world;

use std::path::PathBuf;
use std::process::ExitCode;

use glam::UVec2;

use crate::bench::BenchArgs;
use crate::capture::{CaptureArgs, CaptureError, DEFAULT_FRAMES, DEFAULT_SIZE};

const USAGE: &str = "\
usage:
  universe [--start <name>]        open the window at bookmark <name> (default \"start\") of
                                   bookmarks/m1.ron
  universe --capture <bookmarks.ron> --out <dir> [--size 1920x1080] [--frames 4] [--hud] [--hdr]
                                   render each bookmark headless to <dir>/<name>.png + .json
                                   (--hdr: also <name>.hdr.bin, the pre-bloom HDR target)
  universe --bench <bookmarks.ron> [--size 1920x1080] [--frames 120] [--sustained <s>]
           [--json <out.json>]     time each bookmark headless at that internal resolution
                                   (--sustained: s seconds each, two frames in flight, with
                                   all / first10s / last10s windows)
  universe --selftest              compare the GPU shader twins with their CPU twins

  --sky v1 is still accepted anywhere and does nothing: the Wave B sky is the default.";

/// Exit code for a missing or unreadable baked data file (brief).
const EXIT_MISSING_DATA: u8 = 2;
/// Exit code for any other failure.
const EXIT_FAILURE: u8 = 1;

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    /// The window, at the named bookmark of `bookmarks/m1.ron` (`None`: "start").
    Window(Option<String>),
    Capture(CaptureArgs),
    Bench(BenchArgs),
    Selftest,
    Help,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let command = match parse_args(std::env::args().skip(1)) {
        Ok(c) => c,
        Err(msg) => {
            eprintln!("universe: {msg}\n\n{USAGE}");
            return ExitCode::from(EXIT_FAILURE);
        }
    };
    match command {
        Command::Help => {
            println!("{USAGE}");
            ExitCode::SUCCESS
        }
        Command::Selftest => {
            if selftest::run() {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(EXIT_FAILURE)
            }
        }
        Command::Capture(args) => headless_exit("--capture", capture::run(&args)),
        Command::Bench(args) => headless_exit("--bench", bench::run(&args)),
        Command::Window(start) => match app::run(start.as_deref()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("universe: {e:#}");
                ExitCode::from(EXIT_FAILURE)
            }
        },
    }
}

/// The exit code of a headless run, with its error on stderr.
fn headless_exit(mode: &str, result: Result<(), CaptureError>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(CaptureError::MissingData(msg)) => {
            eprintln!("universe {mode}: missing data: {msg}");
            ExitCode::from(EXIT_MISSING_DATA)
        }
        Err(CaptureError::Failed(e)) => {
            eprintln!("universe {mode}: {e:#}");
            ExitCode::from(EXIT_FAILURE)
        }
    }
}

/// Parses the arguments after the program name. `--sky v1` may stand anywhere and is a no-op
/// alias (B9: the Wave B sky became the default).
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let (sky, rest) = take_sky(args)?;
    let command = parse_command(rest)?;
    if sky && matches!(command, Command::Selftest | Command::Help) {
        return Err("--sky applies to the window, --capture and --bench".to_owned());
    }
    Ok(command)
}

/// Removes `--sky v1` from `args`: whether it was given, and the other arguments in order. A
/// second `--sky`, a missing value or a value other than `v1` is an error.
fn take_sky(args: impl IntoIterator<Item = String>) -> Result<(bool, Vec<String>), String> {
    let mut sky = false;
    let mut rest = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg != "--sky" {
            rest.push(arg);
            continue;
        }
        match args.next().as_deref() {
            Some("v1") => {}
            Some(other) => {
                return Err(format!(
                    "--sky takes v1 (the default sky; the M0 stack was retired at B9, its \
                     regression is bookmark bh-shadow-30M-grid), not {other:?}"
                ));
            }
            None => return Err("--sky needs a value: v1".to_owned()),
        }
        if std::mem::replace(&mut sky, true) {
            return Err("--sky is given twice".to_owned());
        }
    }
    Ok((sky, rest))
}

/// Parses the arguments other than `--sky`.
fn parse_command(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Ok(Command::Window(None));
    };
    match first.as_str() {
        "--start" => {
            let name = args.next().ok_or("--start needs a bookmark name")?;
            match args.next() {
                None => Ok(Command::Window(Some(name))),
                Some(extra) => Err(format!("--start takes one bookmark name (got {extra:?})")),
            }
        }
        "-h" | "--help" => Ok(Command::Help),
        "--selftest" => match args.next() {
            None => Ok(Command::Selftest),
            Some(extra) => Err(format!("--selftest takes no arguments (got {extra:?})")),
        },
        "--capture" => {
            let bookmarks = args.next().ok_or("--capture needs a bookmarks file")?;
            parse_capture(PathBuf::from(bookmarks), args).map(Command::Capture)
        }
        "--bench" => {
            let bookmarks = args.next().ok_or("--bench needs a bookmarks file")?;
            parse_bench(PathBuf::from(bookmarks), args).map(Command::Bench)
        }
        other => Err(format!("unknown argument {other:?}")),
    }
}

/// The options after `--capture <bookmarks.ron>`.
fn parse_capture(
    bookmarks: PathBuf,
    mut args: impl Iterator<Item = String>,
) -> Result<CaptureArgs, String> {
    let mut out = None;
    let mut size = DEFAULT_SIZE;
    let mut frames = DEFAULT_FRAMES;
    let mut hud = false;
    let mut hdr = false;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--size" => size = parse_size(&value()?)?,
            "--frames" => frames = parse_frames(&value()?)?,
            "--hud" => hud = true,
            "--hdr" => hdr = true,
            other => return Err(format!("unknown capture option {other:?}")),
        }
    }
    Ok(CaptureArgs {
        bookmarks,
        out: out.ok_or("--capture needs --out <dir>")?,
        size,
        frames,
        hud,
        hdr,
    })
}

/// The options after `--bench <bookmarks.ron>`.
fn parse_bench(
    bookmarks: PathBuf,
    mut args: impl Iterator<Item = String>,
) -> Result<BenchArgs, String> {
    let mut size = bench::DEFAULT_SIZE;
    let mut frames = bench::DEFAULT_FRAMES;
    let mut sustained_s = None;
    let mut json = None;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--size" => size = parse_size(&value()?)?,
            "--frames" => frames = parse_frames(&value()?)?,
            "--sustained" => {
                let v = value()?;
                let s: f64 = v
                    .parse()
                    .ok()
                    .filter(|s: &f64| s.is_finite() && *s > 0.0)
                    .ok_or(format!("--sustained needs seconds > 0, not {v:?}"))?;
                sustained_s = Some(s);
            }
            "--json" => json = Some(PathBuf::from(value()?)),
            other => return Err(format!("unknown bench option {other:?}")),
        }
    }
    Ok(BenchArgs {
        bookmarks,
        size,
        frames,
        sustained_s,
        json,
    })
}

/// A whole number of frames, at least 1.
fn parse_frames(v: &str) -> Result<u32, String> {
    v.parse()
        .ok()
        .filter(|&n: &u32| n >= 1)
        .ok_or(format!("--frames needs a whole number ≥ 1, not {v:?}"))
}

/// `WIDTHxHEIGHT` in pixels, each at least 1.
fn parse_size(s: &str) -> Result<UVec2, String> {
    let bad = || format!("--size needs WIDTHxHEIGHT in pixels, such as 1920x1080, not {s:?}");
    let (w, h) = s.split_once('x').ok_or_else(bad)?;
    let w: u32 = w.parse().map_err(|_| bad())?;
    let h: u32 = h.parse().map_err(|_| bad())?;
    if w == 0 || h == 0 {
        return Err(bad());
    }
    Ok(UVec2::new(w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Command, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    fn capture(out: &str, size: UVec2, frames: u32, hud: bool, hdr: bool) -> Command {
        Command::Capture(CaptureArgs {
            bookmarks: "b.ron".into(),
            out: out.into(),
            size,
            frames,
            hud,
            hdr,
        })
    }

    #[test]
    fn no_arguments_opens_the_window() {
        assert_eq!(parse(&[]), Ok(Command::Window(None)));
        assert_eq!(
            parse(&["--start", "saturn-ring-plane"]),
            Ok(Command::Window(Some("saturn-ring-plane".to_owned())))
        );
        assert!(parse(&["--start"]).is_err());
        assert!(parse(&["--start", "a", "b"]).is_err());
        assert_eq!(parse(&["--selftest"]), Ok(Command::Selftest));
        assert_eq!(parse(&["--help"]), Ok(Command::Help));
    }

    #[test]
    fn capture_takes_defaults_and_options() {
        assert_eq!(
            parse(&["--capture", "b.ron", "--out", "o"]),
            Ok(capture("o", UVec2::new(1920, 1080), 4, false, false))
        );
        assert_eq!(
            parse(&[
                "--capture",
                "b.ron",
                "--hud",
                "--size",
                "640x360",
                "--out",
                "o",
                "--frames",
                "1",
                "--hdr"
            ]),
            Ok(capture("o", UVec2::new(640, 360), 1, true, true))
        );
    }

    #[test]
    fn bench_takes_defaults_and_options() {
        assert_eq!(
            parse(&["--bench", "b.ron"]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "b.ron".into(),
                size: UVec2::new(1920, 1080),
                frames: 120,
                sustained_s: None,
                json: None,
            }))
        );
        assert_eq!(
            parse(&[
                "--bench",
                "b.ron",
                "--frames",
                "30",
                "--size",
                "1728x1080",
                "--sustained",
                "60",
                "--json",
                "captures/wave-b/bench.json"
            ]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "b.ron".into(),
                size: UVec2::new(1728, 1080),
                frames: 30,
                sustained_s: Some(60.0),
                json: Some("captures/wave-b/bench.json".into()),
            }))
        );
    }

    /// B9: `--sky v1` stays accepted as a no-op alias anywhere on the line.
    #[test]
    fn sky_v1_is_a_no_op_alias() {
        assert_eq!(parse(&["--sky", "v1"]), Ok(Command::Window(None)));
        let plain = parse(&["--capture", "b.ron", "--out", "o"]);
        for line in [
            &["--capture", "b.ron", "--out", "o", "--sky", "v1"][..],
            &["--sky", "v1", "--capture", "b.ron", "--out", "o"],
            &["--capture", "b.ron", "--sky", "v1", "--out", "o"],
        ] {
            assert_eq!(parse(line), plain, "{line:?}");
        }
        assert_eq!(
            parse(&["--bench", "b.ron", "--sky", "v1"]),
            parse(&["--bench", "b.ron"])
        );
    }

    #[test]
    fn bad_arguments_are_errors() {
        for bad in [
            &["--capture"][..],
            &["--capture", "b.ron"],
            &["--capture", "b.ron", "--out"],
            &["--capture", "b.ron", "--out", "o", "--size", "1920"],
            &["--capture", "b.ron", "--out", "o", "--size", "0x10"],
            &["--capture", "b.ron", "--out", "o", "--frames", "0"],
            &["--capture", "b.ron", "--out", "o", "--zoom"],
            &["--selftest", "now"],
            &["--fly"],
            &["--bench"],
            &["--bench", "b.ron", "--frames", "0"],
            &["--bench", "b.ron", "--hud"],
            &["--bench", "b.ron", "--sustained", "0"],
            &["--bench", "b.ron", "--sustained", "x"],
            &["--bench", "b.ron", "--json"],
            &["--sky"],
            &["--sky", "m0"],
            &["--sky", "V1"],
            &["--sky", "v1", "--sky", "v1"],
            &["--selftest", "--sky", "v1"],
            &["--help", "--sky", "v1"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
