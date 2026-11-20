//! uni-bake: offline baker that turns raw downloaded catalogues into compact files in data/baked/.
//!
//! Usage: `uni-bake bsc5 --input data/cache/bsc5/catalog.gz --out data/baked/bsc5.bin`.
//! Log level follows `RUST_LOG` (default `info`).

mod bsc5;

use anyhow::{Context, bail};
use std::ffi::OsString;
use std::path::PathBuf;

const USAGE: &str = "usage: uni-bake bsc5 --input <catalog.gz> --out <bsc5.bin>";

/// A parsed command line.
#[derive(Debug, PartialEq)]
enum Command {
    Bsc5 { input: PathBuf, out: PathBuf },
}

/// Parses the arguments after the program name.
fn parse_args(args: &[OsString]) -> anyhow::Result<Command> {
    let Some((catalogue, options)) = args.split_first() else {
        bail!("no catalogue named\n{USAGE}");
    };
    if catalogue != "bsc5" {
        bail!("unknown catalogue {catalogue:?}\n{USAGE}");
    }
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

    #[test]
    fn rejects_incomplete_or_unknown_commands() {
        assert!(parse_args(&args(&[])).is_err());
        assert!(parse_args(&args(&["hyg"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--input", "in.gz"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--input", "in.gz", "--out"])).is_err());
        assert!(parse_args(&args(&["bsc5", "--in", "in.gz", "--out", "o.bin"])).is_err());
    }
}
