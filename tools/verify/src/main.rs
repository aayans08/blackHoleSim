//! uni-verify: checks headless captures (contact sheet, shadow-radius measurement).
//!
//! - `uni-verify sheet --captures <dir> --out <dir>/index.html` writes an HTML contact sheet
//!   of every PNG in `<dir>` with its JSON metrics (see `sheet`).
//! - `uni-verify shadow --png <file> --expected-px <f64> --tol <f64>` measures the
//!   black-hole shadow radius in a capture and prints `measured=<r> expected=<e> PASS|FAIL`
//!   (see `shadow`).
//!
//! Exit codes: 0 on success or PASS, 1 on FAIL, 2 on a bad command line or an I/O or
//! decode error (the error goes to stderr).

mod shadow;
mod sheet;

use anyhow::{Context, bail};
use std::ffi::OsString;
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "usage:
  uni-verify sheet --captures <dir> --out <dir>/index.html
  uni-verify shadow --png <file> --expected-px <f64> --tol <f64>";

/// Exit code when the shadow radius is out of tolerance.
const EXIT_FAIL: u8 = 1;
/// Exit code for a bad command line or an I/O or decode error.
const EXIT_ERROR: u8 = 2;

/// A parsed command line.
#[derive(Debug, PartialEq)]
enum Command {
    Sheet {
        captures: PathBuf,
        out: PathBuf,
    },
    Shadow {
        png: PathBuf,
        expected_px: f64,
        tol: f64,
    },
}

/// Parses the arguments after the program name.
fn parse_args(args: &[OsString]) -> anyhow::Result<Command> {
    let Some((command, options)) = args.split_first() else {
        bail!("no command given\n{USAGE}");
    };
    if command == "sheet" {
        let [captures, out] = flag_values(options, ["--captures", "--out"])?;
        Ok(Command::Sheet {
            captures: PathBuf::from(captures),
            out: PathBuf::from(out),
        })
    } else if command == "shadow" {
        let [png, expected_px, tol] = flag_values(options, ["--png", "--expected-px", "--tol"])?;
        Ok(Command::Shadow {
            png: PathBuf::from(png),
            expected_px: non_negative("--expected-px", &expected_px)?,
            tol: non_negative("--tol", &tol)?,
        })
    } else {
        bail!("unknown command {command:?}\n{USAGE}");
    }
}

/// The values of `--flag value` pairs, in the order of `names`. Every flag in `names` must be
/// given exactly once, and no other flag may be.
fn flag_values<const N: usize>(
    options: &[OsString],
    names: [&str; N],
) -> anyhow::Result<[OsString; N]> {
    let mut values: [Option<OsString>; N] = std::array::from_fn(|_| None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let Some(slot) = names.iter().position(|name| flag == *name) else {
            bail!("unknown option {flag:?}\n{USAGE}");
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        if values[slot].replace(value.clone()).is_some() {
            bail!("{flag:?} given twice\n{USAGE}");
        }
    }
    if let Some(missing) = values.iter().position(Option::is_none) {
        bail!("missing {}\n{USAGE}", names[missing]);
    }
    Ok(values.map(Option::unwrap_or_default))
}

/// A finite, non-negative number [px].
fn non_negative(flag: &str, value: &OsString) -> anyhow::Result<f64> {
    let parsed = value.to_str().and_then(|s| s.parse::<f64>().ok());
    match parsed {
        Some(x) if x.is_finite() && x >= 0.0 => Ok(x),
        _ => bail!("{flag} needs a finite, non-negative number, not {value:?}\n{USAGE}"),
    }
}

/// Runs a parsed command; `Ok` carries the exit code for a completed run.
fn run(command: Command) -> anyhow::Result<ExitCode> {
    match command {
        Command::Sheet { captures, out } => {
            let count = sheet::write_sheet(&captures, &out)?;
            println!("sheet: {count} captures -> {}", out.display());
            Ok(ExitCode::SUCCESS)
        }
        Command::Shadow {
            png,
            expected_px,
            tol,
        } => {
            let measured = shadow::shadow_radius(&shadow::load_rgb8(&png)?);
            let pass = shadow::passes(measured, expected_px, tol);
            println!("{}", shadow::report_line(measured, expected_px, pass));
            Ok(if pass {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(EXIT_FAIL)
            })
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

    #[test]
    fn parses_sheet_and_shadow_commands() {
        let sheet = parse_args(&args(&[
            "sheet",
            "--captures",
            "captures/m0",
            "--out",
            "captures/m0/index.html",
        ]));
        assert_eq!(
            sheet.expect("valid sheet command"),
            Command::Sheet {
                captures: PathBuf::from("captures/m0"),
                out: PathBuf::from("captures/m0/index.html"),
            }
        );
        // Flags in any order.
        let shadow = parse_args(&args(&[
            "shadow",
            "--tol",
            "1",
            "--png",
            "bh.png",
            "--expected-px",
            "158.745",
        ]));
        assert_eq!(
            shadow.expect("valid shadow command"),
            Command::Shadow {
                png: PathBuf::from("bh.png"),
                expected_px: 158.745,
                tol: 1.0,
            }
        );
    }

    #[test]
    fn rejects_bad_arguments() {
        let bad: &[&[&str]] = &[
            &[],
            &["measure"],
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
        ];
        for list in bad {
            assert!(parse_args(&args(list)).is_err(), "accepted {list:?}");
        }
    }
}
