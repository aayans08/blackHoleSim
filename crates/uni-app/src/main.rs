//! universe: the explorer binary. Windowed by default; `--capture`, `--bench` and `--selftest`
//! run headless.
//!
//! Exit codes: 0 on success; 2 when a baked data file is missing or unreadable (`--capture`,
//! `--bench`); 1 for every other failure, including bad arguments, GPU errors and failed
//! `--selftest` checks.

mod app;
mod bench;
mod capture;
mod scene;
mod selftest;
mod world;

use std::path::PathBuf;
use std::process::ExitCode;

use glam::UVec2;

use crate::bench::BenchArgs;
use crate::capture::{CaptureArgs, CaptureError, DEFAULT_FRAMES, DEFAULT_SIZE};

const USAGE: &str = "\
usage:
  universe                         open the window at bookmark \"start\" of bookmarks/m0.ron
  universe --capture <bookmarks.ron> --out <dir> [--size 1920x1080] [--frames 4] [--hud]
                                   render each bookmark headless to <dir>/<name>.png + .json
  universe --bench <bookmarks.ron> [--size 1920x1080] [--frames 120]
                                   time each bookmark headless at that internal resolution:
                                   mean and p95 frame ms, GPU frame ms, GPU ms per pass
  universe --selftest              compare the GPU shader twins with their CPU twins";

/// Exit code for a missing or unreadable baked data file (brief).
const EXIT_MISSING_DATA: u8 = 2;
/// Exit code for any other failure.
const EXIT_FAILURE: u8 = 1;

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    Window,
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
        Command::Window => match app::run() {
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

/// Parses the arguments after the program name.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Ok(Command::Window);
    };
    match first.as_str() {
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
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => out = Some(PathBuf::from(value()?)),
            "--size" => size = parse_size(&value()?)?,
            "--frames" => frames = parse_frames(&value()?)?,
            "--hud" => hud = true,
            other => return Err(format!("unknown capture option {other:?}")),
        }
    }
    Ok(CaptureArgs {
        bookmarks,
        out: out.ok_or("--capture needs --out <dir>")?,
        size,
        frames,
        hud,
    })
}

/// The options after `--bench <bookmarks.ron>`.
fn parse_bench(
    bookmarks: PathBuf,
    mut args: impl Iterator<Item = String>,
) -> Result<BenchArgs, String> {
    let mut size = bench::DEFAULT_SIZE;
    let mut frames = bench::DEFAULT_FRAMES;
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--size" => size = parse_size(&value()?)?,
            "--frames" => frames = parse_frames(&value()?)?,
            other => return Err(format!("unknown bench option {other:?}")),
        }
    }
    Ok(BenchArgs {
        bookmarks,
        size,
        frames,
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

    #[test]
    fn no_arguments_opens_the_window() {
        assert_eq!(parse(&[]), Ok(Command::Window));
        assert_eq!(parse(&["--selftest"]), Ok(Command::Selftest));
        assert_eq!(parse(&["--help"]), Ok(Command::Help));
    }

    #[test]
    fn capture_takes_defaults_and_options() {
        assert_eq!(
            parse(&["--capture", "bookmarks/m0.ron", "--out", "captures/m0"]),
            Ok(Command::Capture(CaptureArgs {
                bookmarks: "bookmarks/m0.ron".into(),
                out: "captures/m0".into(),
                size: UVec2::new(1920, 1080),
                frames: 4,
                hud: false,
            }))
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
                "1"
            ]),
            Ok(Command::Capture(CaptureArgs {
                bookmarks: "b.ron".into(),
                out: "o".into(),
                size: UVec2::new(640, 360),
                frames: 1,
                hud: true,
            }))
        );
    }

    #[test]
    fn bench_takes_defaults_and_options() {
        assert_eq!(
            parse(&["--bench", "bookmarks/m0.ron"]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "bookmarks/m0.ron".into(),
                size: UVec2::new(1920, 1080),
                frames: 120,
            }))
        );
        assert_eq!(
            parse(&["--bench", "b.ron", "--frames", "30", "--size", "1728x1080"]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "b.ron".into(),
                size: UVec2::new(1728, 1080),
                frames: 30,
            }))
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
            &["--capture", "b.ron", "--out", "o", "--size", "axb"],
            &["--capture", "b.ron", "--out", "o", "--frames", "0"],
            &["--capture", "b.ron", "--out", "o", "--frames", "-1"],
            &["--capture", "b.ron", "--out", "o", "--zoom"],
            &["--selftest", "now"],
            &["--fly"],
            &["--bench"],
            &["--bench", "b.ron", "--frames", "0"],
            &["--bench", "b.ron", "--hud"],
            &["--bench", "b.ron", "--size"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
