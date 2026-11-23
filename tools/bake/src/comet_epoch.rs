//! `uni-bake comets-epoch` (M1 Task A4r, lead ruling A4 P1): `comets.ron` re-osculated to the
//! start date, so the world need not integrate comets at start-up.
//!
//! Reads `data/baked/comets.ron` (SBDB's elements, each at its own epoch), re-osculates every
//! comet to 2026-11-20 00:00:00 TDB with `uni_physics::smallbody::osculate_at` (the eight
//! planets' pulls, RK4; M1 Task A3) and writes `data/baked/comets_2026.ron` in the schema
//! `uni_data::load_comets` reads. Identity and magnitude fields are copied unchanged. A comet
//! whose result is not finite keeps its catalogue elements and is listed. CPU only; no
//! download. Regenerate whenever `comets.ron` is re-baked.
//!
//! Options: `--input <comets.ron>`, `--out <comets_2026.ron>` (defaults under `data/baked/`).

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, bail};
use rayon::prelude::*;
use uni_core::{AU, Epoch};
use uni_data::Comet;
use uni_physics::bodies::GM_SUN_DE440_M3S2;
use uni_physics::smallbody::{conic_from_perihelion, osculate_at};

/// sha256 of `data/baked/comets.ron` (data/manifest.toml, entry "sbdb-comets").
const SOURCE_SHA256: &str = "503ddcff10324ece0d46536fe86dbc1bc358ae21a86d90186498efedc3f6462a";

/// 2026-11-20 00:00:00 TDB as days since J2000.0 (JD 2461364.5; J2000.0 is JD 2451545.0, IERS
/// Conventions 2010): 9820 calendar days after 2000-01-01 00:00, which is 0.5 d before J2000.0.
const TARGET_DAYS_SINCE_J2000: f64 = 9819.5;

/// Epoch of a TDB Julian Date.
fn epoch_from_jd(jd_tdb: f64) -> Epoch {
    Epoch::from_days_since_j2000(jd_tdb - 2_451_545.0)
}

/// `a` in degrees reduced to [0, 360).
fn wrap_deg(rad: f64) -> f64 {
    rad.to_degrees().rem_euclid(360.0)
}

/// `c` (J2000 ecliptic, osculating at `epoch`) re-osculated to `target`, as a catalogue record
/// with `c`'s identity and magnitude fields; `None` when the result is not a finite conic.
pub fn re_osculate(c: &Comet, target: Epoch) -> Option<Comet> {
    let catalogue =
        conic_from_perihelion(c.q_au, c.e, c.i_deg, c.node_deg, c.argp_deg, c.tp_jd_tdb);
    let n = osculate_at(&catalogue, epoch_from_jd(c.epoch_jd_tdb), target);
    let finite = [n.q_m, n.e, n.i_rad, n.node_rad, n.argp_rad, n.tp.jd_tdb()]
        .iter()
        .all(|v| v.is_finite());
    if !(finite && n.q_m > 0.0 && n.e >= 0.0 && (0.0..=std::f64::consts::PI).contains(&n.i_rad)) {
        return None;
    }
    Some(Comet {
        q_au: n.q_m / AU,
        e: n.e,
        i_deg: n.i_rad.to_degrees(),
        node_deg: wrap_deg(n.node_rad),
        argp_deg: wrap_deg(n.argp_rad),
        tp_jd_tdb: n.tp.jd_tdb(),
        epoch_jd_tdb: target.jd_tdb(),
        ..c.clone()
    })
}

/// Runs `uni-bake comets-epoch` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let (mut input, mut out) = (
        PathBuf::from("data/baked/comets.ron"),
        PathBuf::from("data/baked/comets_2026.ron"),
    );
    let mut it = options.iter();
    while let Some(flag) = it.next() {
        let value = it
            .next()
            .with_context(|| format!("{flag:?} needs a value"))?;
        if flag == "--input" {
            input = PathBuf::from(value);
        } else if flag == "--out" {
            out = PathBuf::from(value);
        } else {
            bail!("unknown option {flag:?} (comets-epoch takes --input and --out)");
        }
    }
    let started = std::time::Instant::now();
    let comets =
        uni_data::load_comets(&input).with_context(|| format!("reading {}", input.display()))?;
    let target = Epoch::from_days_since_j2000(TARGET_DAYS_SINCE_J2000);
    let results: Vec<Option<Comet>> = comets.par_iter().map(|c| re_osculate(c, target)).collect();
    let failed: Vec<&str> = comets
        .iter()
        .zip(&results)
        .filter(|(_, r)| r.is_none())
        .map(|(c, _)| c.name.as_str())
        .collect();
    let derived: Vec<Comet> = comets
        .iter()
        .zip(results)
        .map(|(c, r)| r.unwrap_or_else(|| c.clone()))
        .collect();

    let mut header = vec![
        "Derived comet elements for the default start date (M1 Task A4r, lead ruling A4 P1).".to_owned(),
        format!("Derived from data/baked/comets.ron (sha256 {SOURCE_SHA256}, data/manifest.toml id \"sbdb-comets\"):"),
        "every conic re-osculated to 2026-11-20 00:00:00 TDB (JD 2461364.5) by uni-physics' planet-perturbed".to_owned(),
        format!(
            "RK4 integrator (uni_physics::smallbody::osculate_at, M1 Task A3; GM_sun {GM_SUN_DE440_M3S2:e} m3/s2). Identity and"
        ),
        "magnitude fields are SBDB's, unchanged; the orbital elements and the epoch are ours.".to_owned(),
        "Same schema as comets.ron (uni_data::load_comets reads both). Non-gravitational forces are not modelled.".to_owned(),
        "Regenerate with `cargo run -p uni-bake --release -- comets-epoch` whenever comets.ron is re-baked.".to_owned(),
    ];
    if failed.is_empty() {
        header.push(
            "Comets keeping catalogue elements (osculate_at gave a non-finite conic): none."
                .to_owned(),
        );
    } else {
        header.push(format!(
            "Comets keeping catalogue elements (osculate_at gave a non-finite conic): {}.",
            failed.join(", ")
        ));
    }
    crate::sbdb::write_ron(&out, &header, &derived, true)?;
    let back = uni_data::load_comets(&out).context("reading back the derived file")?;
    anyhow::ensure!(
        back.len() == comets.len(),
        "read back {} comets, wrote {}",
        back.len(),
        comets.len()
    );
    println!(
        "comets-epoch: {} comets re-osculated to JD 2461364.5 TDB in {:.1} s -> {}; {} kept catalogue elements{}",
        comets.len(),
        started.elapsed().as_secs_f64(),
        out.display(),
        failed.len(),
        if failed.is_empty() {
            String::new()
        } else {
            format!(": {}", failed.join(", "))
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_is_2026_11_20_midnight() {
        // JD 2461364.5 = 2026-11-20 00:00 (Meeus, Astronomical Algorithms ch. 7).
        assert_eq!(
            epoch_from_jd(2_461_364.5),
            Epoch::from_days_since_j2000(TARGET_DAYS_SINCE_J2000)
        );
    }
}
