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
use crate::scene::SkyMode;

const USAGE: &str = "\
usage:
  universe [--sky v1]              open the window at bookmark \"start\" of bookmarks/m0.ron
                                   (with --sky v1: \"sky-wide-galactic-center\" of
                                   bookmarks/sky-v1.ron)
  universe --capture <bookmarks.ron> --out <dir> [--size 1920x1080] [--frames 4] [--hud]
           [--sky v1]              render each bookmark headless to <dir>/<name>.png + .json
  universe --bench <bookmarks.ron> [--size 1920x1080] [--frames 120] [--sky v1]
                                   time each bookmark headless at that internal resolution:
                                   mean and p95 frame ms, GPU frame ms, GPU ms per pass
  universe --selftest              compare the GPU shader twins with their CPU twins

  --sky v1   the Wave B sky: Milky Way band, AT-HYG stars in 3-D, auto-exposure with
             eye-adapted stars, bloom, and the lens over the band (default: the M0 sky,
             --sky m0); it can go anywhere on the command line";

/// Exit code for a missing or unreadable baked data file (brief).
const EXIT_MISSING_DATA: u8 = 2;
/// Exit code for any other failure.
const EXIT_FAILURE: u8 = 1;

/// What the command line asks for.
#[derive(Debug, PartialEq)]
enum Command {
    Window(SkyMode),
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
        Command::Window(sky) => match app::run(sky) {
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

/// Parses the arguments after the program name. `--sky <mode>` may stand anywhere; the rest
/// is read as before it existed.
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let (sky, rest) = take_sky(args)?;
    let command = parse_command(rest)?;
    let Some(sky) = sky else {
        return Ok(command);
    };
    match command {
        Command::Window(_) => Ok(Command::Window(sky)),
        Command::Capture(args) => Ok(Command::Capture(CaptureArgs { sky, ..args })),
        Command::Bench(args) => Ok(Command::Bench(BenchArgs { sky, ..args })),
        Command::Selftest | Command::Help => {
            Err("--sky applies to the window, --capture and --bench".to_owned())
        }
    }
}

/// Removes `--sky <mode>` from `args`: the mode (`None` when absent) and the other arguments,
/// in order. A second `--sky`, a missing value or an unknown mode is an error.
fn take_sky(
    args: impl IntoIterator<Item = String>,
) -> Result<(Option<SkyMode>, Vec<String>), String> {
    let mut sky = None;
    let mut rest = Vec::new();
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        if arg != "--sky" {
            rest.push(arg);
            continue;
        }
        let value = args.next().ok_or("--sky needs a value: v1 or m0")?;
        if sky.replace(SkyMode::parse(&value)?).is_some() {
            return Err("--sky is given twice".to_owned());
        }
    }
    Ok((sky, rest))
}

/// Parses the arguments other than `--sky`.
fn parse_command(args: impl IntoIterator<Item = String>) -> Result<Command, String> {
    let mut args = args.into_iter();
    let Some(first) = args.next() else {
        return Ok(Command::Window(SkyMode::M0));
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
        sky: SkyMode::M0,
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
        sky: SkyMode::M0,
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
        assert_eq!(parse(&[]), Ok(Command::Window(SkyMode::M0)));
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
                sky: SkyMode::M0,
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
                sky: SkyMode::M0,
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
                sky: SkyMode::M0,
            }))
        );
        assert_eq!(
            parse(&["--bench", "b.ron", "--frames", "30", "--size", "1728x1080"]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "b.ron".into(),
                size: UVec2::new(1728, 1080),
                frames: 30,
                sky: SkyMode::M0,
            }))
        );
    }

    /// `--sky v1` works with the window, --capture and --bench, anywhere on the line; without
    /// it every command keeps the M0 sky.
    #[test]
    fn sky_flag_selects_the_sky_anywhere() {
        assert_eq!(parse(&["--sky", "v1"]), Ok(Command::Window(SkyMode::V1)));
        assert_eq!(parse(&["--sky", "m0"]), Ok(Command::Window(SkyMode::M0)));
        let capture = |sky| {
            Ok(Command::Capture(CaptureArgs {
                bookmarks: "bookmarks/sky-v1.ron".into(),
                out: "captures/sky-v1".into(),
                size: UVec2::new(1920, 1080),
                frames: 4,
                hud: false,
                sky,
            }))
        };
        for line in [
            &[
                "--capture",
                "bookmarks/sky-v1.ron",
                "--out",
                "captures/sky-v1",
                "--sky",
                "v1",
            ][..],
            &[
                "--sky",
                "v1",
                "--capture",
                "bookmarks/sky-v1.ron",
                "--out",
                "captures/sky-v1",
            ],
            &[
                "--capture",
                "bookmarks/sky-v1.ron",
                "--sky",
                "v1",
                "--out",
                "captures/sky-v1",
            ],
        ] {
            assert_eq!(parse(line), capture(SkyMode::V1), "{line:?}");
        }
        assert_eq!(
            parse(&[
                "--capture",
                "bookmarks/sky-v1.ron",
                "--out",
                "captures/sky-v1"
            ]),
            capture(SkyMode::M0)
        );
        assert_eq!(
            parse(&["--bench", "b.ron", "--sky", "v1", "--frames", "30"]),
            Ok(Command::Bench(BenchArgs {
                bookmarks: "b.ron".into(),
                size: UVec2::new(1920, 1080),
                frames: 30,
                sky: SkyMode::V1,
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
            &["--sky"],
            &["--sky", "v2"],
            &["--sky", "V1"],
            &["--sky", "v1", "--sky", "v1"],
            &["--selftest", "--sky", "v1"],
            &["--help", "--sky", "v1"],
            &["--capture", "b.ron", "--out", "o", "--sky"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
