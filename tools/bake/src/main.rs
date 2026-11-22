//! uni-bake: offline baker that turns raw downloaded catalogues into compact files in data/baked/.
//!
//! Usage: `uni-bake <subcommand> [options]`. `bsc5` (M0 Task 7) is parsed here:
//! `uni-bake bsc5 --input data/cache/bsc5/catalog.gz --out data/baked/bsc5.bin`. Every M1
//! subcommand hands its options to its own module's `run`, which parses them, so each data task
//! owns its command line without editing this file (review I8):
//! - A7: `athyg`, `exo`
//! - A8: `map`, `rings` (BC encoders in `bc`)
//! - A9: `sbdb`, `sats`, `facts`, `horizons`
//!
//! Log level follows `RUST_LOG` (default `info`).

// intel_tex_2 0.5.0's prebuilt ISPC objects (kernel_ispc.o, kernel_astc_ispc.o) carry no platform
// load command, so Apple ld prints "no platform load command found in …, assuming: macOS" on every
// link, which is the right platform here. Measured on the M3 in A0b; rustc reports linker stderr
// as `linker_messages` warnings. Link failures are still errors.
#![allow(linker_messages)]

mod athyg;
mod bc;
mod bsc5;
mod exo;
mod facts;
mod horizons;
mod map;
mod rings;
mod sats;
mod sbdb;

use anyhow::{Context, bail};
use std::ffi::OsString;
use std::path::PathBuf;

// A0b pre-adds every Wave A manifest entry (review I8), so a crate can be declared before the task
// that uses it lands. These lines keep `cargo::unused_dependencies` quiet until then; the lead
// deletes each one once its crate is used.
use bytemuck as _;
use glam as _;
use image as _;
use intel_tex_2 as _;
use rayon as _;
use ron as _;
use serde as _;
use serde_json as _;
use uni_physics as _;

const USAGE: &str = "usage: uni-bake <subcommand> [options]
  bsc5 --input <catalog.gz> --out <bsc5.bin>
  athyg | exo | map | rings | sbdb | sats | facts | horizons [options]: see each module's doc";

/// A parsed command line. The M1 variants carry their options unparsed, in order.
#[derive(Debug, PartialEq)]
enum Command {
    Bsc5 { input: PathBuf, out: PathBuf },
    Athyg(Vec<OsString>),
    Exo(Vec<OsString>),
    Map(Vec<OsString>),
    Rings(Vec<OsString>),
    Sbdb(Vec<OsString>),
    Sats(Vec<OsString>),
    Facts(Vec<OsString>),
    Horizons(Vec<OsString>),
}

/// Parses the arguments after the program name.
fn parse_args(args: &[OsString]) -> anyhow::Result<Command> {
    let Some((subcommand, options)) = args.split_first() else {
        bail!("no subcommand named\n{USAGE}");
    };
    let options = options.to_vec();
    Ok(match subcommand.to_str() {
        Some("bsc5") => parse_bsc5(&options)?,
        Some("athyg") => Command::Athyg(options),
        Some("exo") => Command::Exo(options),
        Some("map") => Command::Map(options),
        Some("rings") => Command::Rings(options),
        Some("sbdb") => Command::Sbdb(options),
        Some("sats") => Command::Sats(options),
        Some("facts") => Command::Facts(options),
        Some("horizons") => Command::Horizons(options),
        _ => bail!("unknown subcommand {subcommand:?}\n{USAGE}"),
    })
}

/// Parses `bsc5`'s options.
fn parse_bsc5(options: &[OsString]) -> anyhow::Result<Command> {
    let (mut input, mut out) = (None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = if flag == "--input" {
            &mut input
        } else if flag == "--out" {
            &mut out
        } else {
            bail!("unknown option {flag:?}\n{USAGE}");
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    match (input, out) {
        (Some(input), Some(out)) => Ok(Command::Bsc5 { input, out }),
        _ => bail!("bsc5 needs both --input and --out\n{USAGE}"),
    }
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args: Vec<OsString> = std::env::args_os().skip(1).collect();
    match parse_args(&args)? {
        Command::Bsc5 { input, out } => {
            let summary = bsc5::bake(&input, &out)?;
            println!("bsc5: {summary} -> {}", out.display());
        }
        Command::Athyg(options) => athyg::run(&options)?,
        Command::Exo(options) => exo::run(&options)?,
        Command::Map(options) => map::run(&options)?,
        Command::Rings(options) => rings::run(&options)?,
        Command::Sbdb(options) => sbdb::run(&options)?,
        Command::Sats(options) => sats::run(&options)?,
        Command::Facts(options) => facts::run(&options)?,
        Command::Horizons(options) => horizons::run(&options)?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<OsString> {
        list.iter().map(OsString::from).collect()
    }

    #[test]
    fn parses_bsc5_command() {
        let command = parse_args(&args(&["bsc5", "--input", "in.gz", "--out", "out.bin"]));
        assert_eq!(
            command.expect("valid"),
            Command::Bsc5 {
                input: PathBuf::from("in.gz"),
                out: PathBuf::from("out.bin"),
            }
        );
    }

    /// Every Wave A subcommand hands its options, unparsed and in order, to its own module.
    #[test]
    fn parses_module_subcommands() {
        let options = ["--input", "in", "--out", "out"];
        let cases: [(&str, fn(Vec<OsString>) -> Command); 8] = [
            ("athyg", Command::Athyg),
            ("exo", Command::Exo),
            ("map", Command::Map),
            ("rings", Command::Rings),
            ("sbdb", Command::Sbdb),
            ("sats", Command::Sats),
            ("facts", Command::Facts),
            ("horizons", Command::Horizons),
        ];
        for (name, variant) in cases {
            let mut line = vec![name];
            line.extend(options);
            assert_eq!(
                parse_args(&args(&line)).expect("valid"),
                variant(args(&options)),
                "{name}"
            );
            assert_eq!(
                parse_args(&args(&[name])).expect("no options is valid here"),
                variant(Vec::new()),
                "{name}"
            );
        }
    }

    #[test]
    fn rejects_incomplete_or_unknown_commands() {
        assert!(parse_args(&args(&[])).is_err());
        assert!(parse_args(&args(&["hyg"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--input", "in.gz"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--input", "in.gz", "--out"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--in", "in.gz", "--out", "o.bin"])).is_err());
    }
}
