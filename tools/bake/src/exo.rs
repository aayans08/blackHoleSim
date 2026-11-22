//! `uni-bake exo` (M1 Task A7): NASA Exoplanet Archive `pscomppars` CSV → `data/baked/exoplanets.ron`.
//!
//! Command line:
//! `uni-bake exo --input data/cache/exoplanets/pscomppars.csv --stars data/baked/athyg_m10.bin
//!  --athyg data/cache/athyg-m10/athyg_40_reduced_m10.csv.gz --out data/baked/exoplanets.ron`.
//! The plan's `--names athyg_names.ron` is replaced by `--athyg`: matching needs the Gaia DR3,
//! HIP and HD numbers of every AT-HYG star, which only the AT-HYG CSV has (the names file
//! lists HIP numbers for named stars only).
//!
//! Each archive host becomes one [`ExoSystem`] (hosts sorted by name, planets by name; host
//! values from the first planet, in name order, that has them, since `pscomppars` picks
//! stellar parameters per planet). Hosts are matched to an AT-HYG star of `--stars` by, in
//! order: Gaia DR3 source id, HIP number, HD number, then position within
//! [`POSITION_MATCH_ARCSEC`] and distance within [`POSITION_MATCH_DIST_FRAC`]. A HIP or HD
//! name with a component letter other than A ("HD 41004 B") names the primary's number and is
//! not used. Unmatched hosts keep `star_index: None` and their archive V (uni-sim draws them as
//! archive stars, review I11). Hosts with no archive distance take the matched star's distance;
//! unmatched ones without a distance cannot be placed and are left out (listed on the console).
//! Matched stars get `FLAG_EXO_HOST`, written back into `--stars` in place.

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::{Context, bail, ensure};
use uni_data::{ExoPlanet, ExoSystem, FLAG_EXO_HOST, Star3};

use crate::athyg::{CrossIds, read_cross_ids, split_csv};

const USAGE: &str = "usage: uni-bake exo --input <pscomppars.csv> --stars <athyg_m10.bin> \
--athyg <athyg_40_reduced_m10.csv.gz> --out <exoplanets.ron>";

/// Largest host ↔ star separation for a position match [arcsec] (M1 plan, Task A7).
const POSITION_MATCH_ARCSEC: f64 = 2.0;
/// Largest |Δdist| / archive distance for a position match (M1 plan, Task A7).
const POSITION_MATCH_DIST_FRAC: f64 = 0.10;
/// Required share of hosts with V ≤ 10 within 100 pc matched to a star (M1 review I11).
const REQUIRED_MATCH_RATE: f64 = 0.95;
/// Hosts within this distance [pc] that stay unmatched are listed by name.
const LIST_UNMATCHED_WITHIN_PC: f64 = 30.0;
/// ID matches farther than this [arcsec] from the archive position are listed for review. The
/// archive's RA and Dec are mostly Gaia positions at epoch J2015.5/J2016 while AT-HYG's are at
/// J2000, so nearby high-proper-motion stars legitimately differ by up to ~10″/yr × 16 yr.
const ID_MATCH_REVIEW_ARCSEC: f64 = 30.0;

/// Runs `uni-bake exo` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    let input = File::open(&options.input)
        .with_context(|| format!("opening {}", options.input.display()))?;
    let hosts = parse_archive(BufReader::new(input))
        .with_context(|| format!("parsing {}", options.input.display()))?;
    let mut stars = uni_data::load_athyg(&options.stars)
        .with_context(|| format!("reading {}", options.stars.display()))?;
    let cross_ids = read_cross_ids(&options.athyg)?;

    let mut systems = Vec::new();
    let mut methods = Vec::new();
    let mut dropped = Vec::new();
    {
        let catalogue = Catalogue::new(&stars, &cross_ids);
        for host in hosts {
            let name = host.name.clone();
            match build_system(host, &catalogue) {
                Outcome::System(system, method) => {
                    if let (Some(index), Some(method)) = (system.star_index, method)
                        && method != Method::Position
                    {
                        let sep = catalogue.separation_arcsec(index, system.ra_deg, system.dec_deg);
                        if sep > ID_MATCH_REVIEW_ARCSEC {
                            println!(
                                "exo: {name} matched by {method:?} to a star {sep:.0}″ from the archive position \
                                 (proper motion between the archive epoch and J2000?)"
                            );
                        }
                    }
                    systems.push(system);
                    methods.push(method);
                }
                Outcome::NoDistance => dropped.push(name),
            }
        }
    }
    report(&systems, &methods, &dropped)?;

    let text = ron::ser::to_string_pretty(
        &systems,
        ron::ser::PrettyConfig::new().compact_structs(true),
    )?;
    std::fs::write(&options.out, text)
        .with_context(|| format!("writing {}", options.out.display()))?;
    let back = uni_data::load_exoplanets(&options.out)
        .with_context(|| format!("reading back {}", options.out.display()))?;
    ensure!(
        back.len() == systems.len(),
        "{} does not read back as written",
        options.out.display()
    );

    mark_hosts(&mut stars, &systems);
    uni_data::write_athyg(&options.stars, &stars)
        .with_context(|| format!("writing {}", options.stars.display()))?;
    let back = uni_data::load_athyg(&options.stars)?;
    ensure!(
        back == stars,
        "{} does not read back as written",
        options.stars.display()
    );
    let flagged = stars.iter().filter(|s| s.is_exo_host()).count();
    println!(
        "exo: {flagged} stars of {} flagged as exoplanet hosts",
        options.stars.display()
    );
    Ok(())
}

/// Prints the match statistics and enforces [`REQUIRED_MATCH_RATE`].
fn report(
    systems: &[ExoSystem],
    methods: &[Option<Method>],
    dropped: &[String],
) -> anyhow::Result<()> {
    let planets: usize = systems.iter().map(|s| s.planets.len()).sum();
    println!(
        "exo: {} systems with {planets} planets written; {} hosts without any distance left out: {dropped:?}",
        systems.len(),
        dropped.len()
    );
    let mut by_method: BTreeMap<String, usize> = BTreeMap::new();
    for method in methods {
        *by_method
            .entry(method.map_or("unmatched".to_owned(), |m| format!("{m:?}")))
            .or_default() += 1;
    }
    println!("exo: matches by method: {by_method:?}");
    let rate = |filter: &dyn Fn(&ExoSystem) -> bool| {
        let selected: Vec<&ExoSystem> = systems.iter().filter(|s| filter(s)).collect();
        let matched = selected.iter().filter(|s| s.star_index.is_some()).count();
        (matched, selected.len())
    };
    let bright = |s: &ExoSystem| s.v_mag.is_some_and(|v| v <= 10.0);
    let near = |s: &ExoSystem| s.dist_pc <= 100.0;
    let rows: [(&str, (usize, usize)); 4] = [
        ("all hosts", rate(&|_| true)),
        ("V ≤ 10", rate(&bright)),
        ("within 100 pc", rate(&near)),
        ("V ≤ 10 within 100 pc", rate(&|s| bright(s) && near(s))),
    ];
    for (what, (matched, total)) in rows {
        println!(
            "exo: matched {what}: {matched} / {total} = {:.1} %",
            100.0 * matched as f64 / total.max(1) as f64
        );
    }
    let mut stars_used: HashMap<u32, Vec<&str>> = HashMap::new();
    for s in systems {
        if let Some(index) = s.star_index {
            stars_used.entry(index).or_default().push(&s.host);
        }
    }
    let mut shared: Vec<&Vec<&str>> = stars_used
        .values()
        .filter(|hosts| hosts.len() > 1)
        .collect();
    shared.sort();
    println!(
        "exo: {} stars host more than one archive system: {shared:?}",
        shared.len()
    );
    let mut unmatched_near: Vec<&ExoSystem> = systems
        .iter()
        .filter(|s| s.star_index.is_none() && s.dist_pc <= LIST_UNMATCHED_WITHIN_PC)
        .collect();
    unmatched_near.sort_by(|a, b| a.dist_pc.total_cmp(&b.dist_pc));
    println!(
        "exo: {} unmatched hosts within {LIST_UNMATCHED_WITHIN_PC} pc:",
        unmatched_near.len()
    );
    for s in unmatched_near {
        println!(
            "  {:<28} {:>7.3} pc  V {}",
            s.host,
            s.dist_pc,
            s.v_mag.map_or("–".to_owned(), |v| format!("{v:.2}"))
        );
    }
    let (matched, total) = rows[3].1;
    ensure!(
        total > 0 && matched as f64 >= REQUIRED_MATCH_RATE * total as f64,
        "only {matched} of {total} hosts with V ≤ 10 within 100 pc matched an AT-HYG star (need {:.0} %)",
        100.0 * REQUIRED_MATCH_RATE
    );
    Ok(())
}

#[derive(Debug, PartialEq)]
struct Options {
    input: PathBuf,
    stars: PathBuf,
    athyg: PathBuf,
    out: PathBuf,
}

fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut input, mut stars, mut athyg, mut out) = (None, None, None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--input") => &mut input,
            Some("--stars") => &mut stars,
            Some("--athyg") => &mut athyg,
            Some("--out") => &mut out,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    match (input, stars, athyg, out) {
        (Some(input), Some(stars), Some(athyg), Some(out)) => Ok(Options {
            input,
            stars,
            athyg,
            out,
        }),
        _ => bail!("exo needs --input, --stars, --athyg and --out\n{USAGE}"),
    }
}

/// One host and its planets as read from the archive.
#[derive(Clone, Debug)]
struct Host {
    name: String,
    gaia_dr3: Option<u64>,
    hip: Option<u32>,
    hd: Option<u32>,
    ra_deg: f64,
    dec_deg: f64,
    dist_pc: Option<f64>,
    v_mag: Option<f64>,
    teff_k: Option<f64>,
    radius_rsun: Option<f64>,
    planets: Vec<ExoPlanet>,
}

/// The host-level columns of one archive row.
struct HostColumns {
    gaia_dr3: Option<u64>,
    hip: Option<u32>,
    hd: Option<u32>,
    ra_deg: f64,
    dec_deg: f64,
    dist_pc: Option<f64>,
    v_mag: Option<f64>,
    teff_k: Option<f64>,
    radius_rsun: Option<f64>,
}

/// Every column the bake reads, by name (archive TAP schema of `pscomppars`).
const COLUMNS: [&str; 22] = [
    "pl_name",
    "hostname",
    "hip_name",
    "hd_name",
    "gaia_dr3_id",
    "ra",
    "dec",
    "sy_dist",
    "sy_vmag",
    "st_teff",
    "st_rad",
    "pl_orbsmax",
    "pl_orbeccen",
    "pl_orbincl",
    "pl_orblper",
    "pl_orbper",
    "pl_tranmid",
    "pl_bmassj",
    "pl_radj",
    "pl_eqt",
    "discoverymethod",
    "disc_year",
];

fn parse_archive(reader: impl BufRead) -> anyhow::Result<Vec<Host>> {
    let mut lines = reader.lines();
    let header = split_csv(&lines.next().context("empty file")??);
    let mut at = HashMap::new();
    for name in COLUMNS {
        let index = header
            .iter()
            .position(|h| h == name)
            .with_context(|| format!("no {name:?} column"))?;
        at.insert(name, index);
    }
    let mut by_host: BTreeMap<String, Vec<(ExoPlanet, HostColumns)>> = BTreeMap::new();
    for (index, line) in lines.enumerate() {
        let fields = split_csv(&line?);
        ensure!(
            fields.len() == header.len(),
            "line {}: {} fields, header has {}",
            index + 2,
            fields.len(),
            header.len()
        );
        let row = ArchiveRow {
            fields: &fields,
            at: &at,
        };
        let (planet, host) = parse_planet_row(&row)
            .with_context(|| format!("line {} ({:?})", index + 2, row.get("pl_name")))?;
        by_host
            .entry(row.get("hostname").to_owned())
            .or_default()
            .push((planet, host));
    }
    let hosts = by_host
        .into_iter()
        .map(|(name, mut rows)| {
            rows.sort_by(|a, b| a.0.name.cmp(&b.0.name));
            let first =
                |pick: &dyn Fn(&HostColumns) -> Option<f64>| rows.iter().find_map(|(_, h)| pick(h));
            Host {
                gaia_dr3: rows.iter().find_map(|(_, h)| h.gaia_dr3),
                hip: rows.iter().find_map(|(_, h)| h.hip),
                hd: rows.iter().find_map(|(_, h)| h.hd),
                ra_deg: rows[0].1.ra_deg,
                dec_deg: rows[0].1.dec_deg,
                dist_pc: first(&|h| h.dist_pc),
                v_mag: first(&|h| h.v_mag),
                teff_k: first(&|h| h.teff_k),
                radius_rsun: first(&|h| h.radius_rsun),
                planets: rows.into_iter().map(|(planet, _)| planet).collect(),
                name,
            }
        })
        .collect();
    Ok(hosts)
}

/// One CSV row with its columns looked up by name.
struct ArchiveRow<'a> {
    fields: &'a [String],
    at: &'a HashMap<&'static str, usize>,
}

impl<'a> ArchiveRow<'a> {
    fn get(&self, name: &str) -> &'a str {
        self.fields[self.at[name]].as_str()
    }
}

fn parse_planet_row(row: &ArchiveRow) -> anyhow::Result<(ExoPlanet, HostColumns)> {
    let get = |name: &str| row.get(name);
    let number = |name: &str| -> anyhow::Result<Option<f64>> {
        let field = get(name);
        if field.is_empty() {
            return Ok(None);
        }
        let value: f64 = field
            .parse()
            .with_context(|| format!("{name} = {field:?}"))?;
        ensure!(value.is_finite(), "{name} = {field:?}");
        Ok(Some(value))
    };
    let name = get("pl_name");
    ensure!(!name.is_empty(), "empty pl_name");
    let method = get("discoverymethod");
    ensure!(!method.is_empty(), "empty discoverymethod");
    let planet = ExoPlanet {
        name: name.to_owned(),
        a_au: number("pl_orbsmax")?,
        e: number("pl_orbeccen")?,
        i_deg: number("pl_orbincl")?,
        argp_deg: number("pl_orblper")?,
        period_d: number("pl_orbper")?,
        t_transit_jd: number("pl_tranmid")?,
        mass_mjup: number("pl_bmassj")?,
        radius_rjup: number("pl_radj")?,
        teq_k: number("pl_eqt")?,
        method: method.to_owned(),
        year: get("disc_year")
            .parse()
            .with_context(|| format!("disc_year = {:?}", get("disc_year")))?,
    };
    let gaia_dr3 = match get("gaia_dr3_id") {
        "" => None,
        id => Some(
            id.strip_prefix("Gaia DR3 ")
                .and_then(|n| n.parse().ok())
                .with_context(|| format!("gaia_dr3_id = {id:?}"))?,
        ),
    };
    let host = HostColumns {
        gaia_dr3,
        hip: catalogue_number(get("hip_name"), "HIP")?,
        hd: catalogue_number(get("hd_name"), "HD")?,
        ra_deg: number("ra")?.context("empty ra")?,
        dec_deg: number("dec")?.context("empty dec")?,
        dist_pc: number("sy_dist")?.filter(|d| *d > 0.0),
        v_mag: number("sy_vmag")?,
        teff_k: number("st_teff")?,
        radius_rsun: number("st_rad")?,
    };
    Ok((planet, host))
}

/// The number of an archive name such as "HIP 113357" or "HD 41004 A". `None` for an empty
/// field and for a component other than A, whose catalogue number is the primary's.
fn catalogue_number(field: &str, prefix: &str) -> anyhow::Result<Option<u32>> {
    if field.is_empty() {
        return Ok(None);
    }
    let mut words = field
        .strip_prefix(prefix)
        .context("unexpected catalogue name")?
        .split_whitespace();
    let number: u32 = words
        .next()
        .and_then(|n| n.parse().ok())
        .with_context(|| format!("no number in {field:?}"))?;
    Ok(match words.next() {
        None | Some("A") => Some(number),
        Some(_) => None,
    })
}

/// How a host was matched to an AT-HYG star.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Method {
    Gaia,
    Hip,
    Hd,
    Position,
}

/// The AT-HYG stars indexed for host matching.
struct Catalogue<'a> {
    stars: &'a [Star3],
    by_gaia: HashMap<u64, u32>,
    by_hip: HashMap<u32, u32>,
    by_hd: HashMap<u32, u32>,
    /// (declination [rad], star index), sorted by declination.
    by_dec: Vec<(f64, u32)>,
}

impl<'a> Catalogue<'a> {
    fn new(stars: &'a [Star3], cross_ids: &HashMap<u32, CrossIds>) -> Self {
        let mut catalogue = Self {
            stars,
            by_gaia: HashMap::new(),
            by_hip: HashMap::new(),
            by_hd: HashMap::new(),
            by_dec: Vec::new(),
        };
        for (index, star) in stars.iter().enumerate() {
            let index = index as u32;
            let dir = position(star).normalize();
            catalogue.by_dec.push((dir.z.asin(), index));
            let Some(ids) = cross_ids.get(&star.athyg_id()) else {
                continue;
            };
            // A few ids are shared by AT-HYG components (107 Gaia ids, 6 HD numbers in v4.0):
            // the brighter star keeps the id.
            if let Some(gaia) = ids.gaia_dr3 {
                keep_brighter(stars, &mut catalogue.by_gaia, gaia, index);
            }
            if let Some(hip) = ids.hip {
                keep_brighter(stars, &mut catalogue.by_hip, hip, index);
            }
            if let Some(hd) = ids.hd {
                keep_brighter(stars, &mut catalogue.by_hd, hd, index);
            }
        }
        catalogue.by_dec.sort_by(|a, b| a.0.total_cmp(&b.0));
        catalogue
    }

    fn find(&self, host: &Host) -> Option<(u32, Method)> {
        if let Some(gaia) = host.gaia_dr3
            && let Some(&index) = self.by_gaia.get(&gaia)
        {
            return Some((index, Method::Gaia));
        }
        if let Some(hip) = host.hip
            && let Some(&index) = self.by_hip.get(&hip)
        {
            return Some((index, Method::Hip));
        }
        if let Some(hd) = host.hd
            && let Some(&index) = self.by_hd.get(&hd)
        {
            return Some((index, Method::Hd));
        }
        let dist = host.dist_pc?;
        let target = unit(host.ra_deg, host.dec_deg);
        let max_angle = (POSITION_MATCH_ARCSEC / 3600.0).to_radians();
        let dec = target.z.asin();
        let start = self.by_dec.partition_point(|&(d, _)| d < dec - max_angle);
        self.by_dec[start..]
            .iter()
            .take_while(|&&(d, _)| d <= dec + max_angle)
            .filter_map(|&(_, index)| {
                let p = position(&self.stars[index as usize]);
                let angle = angle_between(target, p.normalize());
                let dist_ok = (p.length() - dist).abs() <= POSITION_MATCH_DIST_FRAC * dist;
                (angle <= max_angle && dist_ok).then_some((angle, index))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, index)| (index, Method::Position))
    }

    fn distance_pc(&self, index: u32) -> f64 {
        position(&self.stars[index as usize]).length()
    }

    fn separation_arcsec(&self, index: u32, ra_deg: f64, dec_deg: f64) -> f64 {
        let dir = position(&self.stars[index as usize]).normalize();
        angle_between(unit(ra_deg, dec_deg), dir).to_degrees() * 3600.0
    }
}

/// Maps `key` to star `index`, unless a brighter star already holds it.
fn keep_brighter<K: std::hash::Hash + Eq>(
    stars: &[Star3],
    map: &mut HashMap<K, u32>,
    key: K,
    index: u32,
) {
    map.entry(key)
        .and_modify(|old| {
            if apparent_v(&stars[index as usize]) < apparent_v(&stars[*old as usize]) {
                *old = index;
            }
        })
        .or_insert(index);
}

fn position(star: &Star3) -> glam::DVec3 {
    glam::Vec3::from(star.pos_pc).as_dvec3()
}

fn apparent_v(star: &Star3) -> f64 {
    f64::from(star.abs_mag_v) + 5.0 * position(star).length().log10() - 5.0
}

/// ICRS unit vector toward (RA, Dec) [deg]: x toward RA 0 Dec 0, z toward the north pole.
fn unit(ra_deg: f64, dec_deg: f64) -> glam::DVec3 {
    let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
    glam::DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
}

/// Angle [rad] between unit vectors, from the chord (accurate for small angles).
fn angle_between(a: glam::DVec3, b: glam::DVec3) -> f64 {
    2.0 * (0.5 * a.distance(b)).min(1.0).asin()
}

/// A host turned into an [`ExoSystem`], or why it was left out.
#[derive(Debug)]
enum Outcome {
    System(ExoSystem, Option<Method>),
    /// No archive distance and no matched star to take one from.
    NoDistance,
}

fn build_system(host: Host, catalogue: &Catalogue) -> Outcome {
    let found = catalogue.find(&host);
    let dist_pc = match (host.dist_pc, found) {
        (Some(dist), _) => dist,
        (None, Some((index, _))) => catalogue.distance_pc(index),
        (None, None) => return Outcome::NoDistance,
    };
    let system = ExoSystem {
        host: host.name,
        star_index: found.map(|(index, _)| index),
        ra_deg: host.ra_deg,
        dec_deg: host.dec_deg,
        dist_pc,
        v_mag: host.v_mag,
        teff_k: host.teff_k,
        radius_rsun: host.radius_rsun,
        planets: host.planets,
    };
    Outcome::System(system, found.map(|(_, method)| method))
}

/// Sets `FLAG_EXO_HOST` on exactly the stars that `systems` name, clearing stale ones.
fn mark_hosts(stars: &mut [Star3], systems: &[ExoSystem]) {
    for star in stars.iter_mut() {
        star.id_flags &= !FLAG_EXO_HOST;
    }
    for index in systems.iter().filter_map(|s| s.star_index) {
        if let Some(star) = stars.get_mut(index as usize) {
            star.id_flags |= FLAG_EXO_HOST;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uni_data::{FLAG_EXO_HOST, FLAG_NAMED};

    const HEADER: &str = "pl_name,hostname,hip_name,hd_name,gaia_dr3_id,ra,dec,sy_dist,sy_vmag,st_teff,st_rad,\
pl_orbsmax,pl_orbeccen,pl_orbincl,pl_orblper,pl_orbper,pl_tranmid,pl_tranmid_systemref,pl_bmassj,pl_radj,pl_eqt,\
discoverymethod,disc_year";

    /// One arcsecond [deg].
    const ARCSEC: f64 = 1.0 / 3600.0;

    fn unit(ra_deg: f64, dec_deg: f64) -> glam::DVec3 {
        let (ra, dec) = (ra_deg.to_radians(), dec_deg.to_radians());
        glam::DVec3::new(dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin())
    }

    fn star_at(ra_deg: f64, dec_deg: f64, dist_pc: f64, id: u32) -> Star3 {
        let p = (unit(ra_deg, dec_deg) * dist_pc).as_vec3();
        Star3::new(p.to_array(), 4.0, [0.0; 3], 0.6, id, 0)
    }

    fn host(name: &str, ra_deg: f64, dec_deg: f64, dist_pc: Option<f64>) -> Host {
        Host {
            name: name.to_owned(),
            gaia_dr3: None,
            hip: None,
            hd: None,
            ra_deg,
            dec_deg,
            dist_pc,
            v_mag: Some(9.0),
            teff_k: Some(5000.0),
            radius_rsun: Some(0.9),
            planets: vec![ExoPlanet {
                name: format!("{name} b"),
                a_au: Some(0.05),
                e: None,
                i_deg: None,
                argp_deg: None,
                period_d: Some(4.0),
                t_transit_jd: None,
                mass_mjup: Some(0.5),
                radius_rjup: None,
                teq_k: None,
                method: "Radial Velocity".to_owned(),
                year: 2000,
            }],
        }
    }

    /// Four stars: AT-HYG ids 100..=103; ids per `cross_ids`.
    fn stars() -> Vec<Star3> {
        vec![
            star_at(10.0, 20.0, 10.0, 100),
            star_at(50.0, -30.0, 20.0, 101),
            star_at(120.0, 5.0, 30.0, 102),
            star_at(200.0, 45.0, 40.0, 103),
        ]
    }

    fn cross_ids() -> HashMap<u32, CrossIds> {
        HashMap::from([
            (
                100,
                CrossIds {
                    gaia_dr3: Some(31),
                    hip: Some(11),
                    hd: Some(21),
                },
            ),
            (
                101,
                CrossIds {
                    gaia_dr3: Some(32),
                    hip: Some(12),
                    hd: None,
                },
            ),
            (
                102,
                CrossIds {
                    gaia_dr3: None,
                    hip: None,
                    hd: Some(23),
                },
            ),
            (103, CrossIds::default()),
        ])
    }

    #[test]
    fn host_matching_order() {
        let stars = stars();
        let xids = cross_ids();
        let catalogue = Catalogue::new(&stars, &xids);
        let far = (300.0, -60.0);
        let find = |h: &Host| catalogue.find(h);

        // Gaia beats HIP: the Gaia id names star 1, the HIP number star 0.
        let mut a = host("A", far.0, far.1, Some(20.0));
        (a.gaia_dr3, a.hip) = (Some(32), Some(11));
        assert_eq!(find(&a), Some((1, Method::Gaia)));
        // HIP beats HD: HIP names star 0, HD star 2.
        let mut b = host("B", far.0, far.1, Some(10.0));
        (b.hip, b.hd) = (Some(11), Some(23));
        assert_eq!(find(&b), Some((0, Method::Hip)));
        // HD alone.
        let mut c = host("C", far.0, far.1, Some(30.0));
        c.hd = Some(23);
        assert_eq!(find(&c), Some((2, Method::Hd)));
        // An unknown Gaia id falls through to HIP.
        let mut g = host("G", far.0, far.1, Some(20.0));
        (g.gaia_dr3, g.hip) = (Some(99), Some(12));
        assert_eq!(find(&g), Some((1, Method::Hip)));
        // Position: 1.5″ away and 2.5 % farther → star 3.
        let d = host("D", 200.0, 45.0 + 1.5 * ARCSEC, Some(41.0));
        assert_eq!(find(&d), Some((3, Method::Position)));
        // 1.5″ away but 25 % farther → no match.
        let e = host("E", 200.0, 45.0 + 1.5 * ARCSEC, Some(50.0));
        assert_eq!(find(&e), None);
        // 3″ away → no match.
        let f = host("F", 200.0, 45.0 + 3.0 * ARCSEC, Some(40.0));
        assert_eq!(find(&f), None);
        // Position matching needs the archive distance.
        let n = host("N", 200.0, 45.0 + 1.5 * ARCSEC, None);
        assert_eq!(find(&n), None);
        // RA separations shrink with cos(dec): 2.5″ of RA at Dec 45° is 1.77″ on the sky.
        let r = host("R", 200.0 + 2.5 * ARCSEC, 45.0, Some(40.0));
        assert_eq!(find(&r), Some((3, Method::Position)));
    }

    #[test]
    fn unmatched_host_keeps_archive_values() {
        let stars = stars();
        let xids = cross_ids();
        let catalogue = Catalogue::new(&stars, &xids);
        // TRAPPIST-1: V 18.8 at 12.47 pc, not in AT-HYG m10.
        let mut trappist = host("TRAPPIST-1", 346.6223, -5.0414, Some(12.467));
        trappist.v_mag = Some(18.8);
        match build_system(trappist, &catalogue) {
            Outcome::System(sys, method) => {
                assert_eq!(method, None);
                assert_eq!(sys.star_index, None);
                assert_eq!(sys.v_mag, Some(18.8));
                assert_eq!(sys.dist_pc, 12.467);
                assert_eq!((sys.ra_deg, sys.dec_deg), (346.6223, -5.0414));
                assert_eq!(sys.planets.len(), 1);
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn missing_distance_taken_from_matched_star_or_host_dropped() {
        let stars = stars();
        let xids = cross_ids();
        let catalogue = Catalogue::new(&stars, &xids);
        let mut by_hip = host("H", 0.0, 0.0, None);
        by_hip.hip = Some(12);
        match build_system(by_hip, &catalogue) {
            Outcome::System(sys, method) => {
                assert_eq!(method, Some(Method::Hip));
                assert_eq!(sys.star_index, Some(1));
                assert!((sys.dist_pc - 20.0).abs() < 1e-4, "{}", sys.dist_pc);
            }
            other => panic!("{other:?}"),
        }
        let lost = host("OGLE-2016-BLG-1227L", 265.0, -29.0, None);
        assert!(matches!(
            build_system(lost, &catalogue),
            Outcome::NoDistance
        ));
    }

    #[test]
    fn exo_host_flag_marks_matched_stars_only() {
        let mut stars = stars();
        stars[0].id_flags |= FLAG_EXO_HOST | FLAG_NAMED; // stale flag from an earlier bake
        let xids = cross_ids();
        let systems: Vec<ExoSystem> = {
            let catalogue = Catalogue::new(&stars, &xids);
            let mut c = host("C", 0.0, 0.0, Some(30.0));
            c.hd = Some(23);
            let lone = host("Lone", 1.0, 1.0, Some(5.0));
            [c, lone]
                .into_iter()
                .filter_map(|h| match build_system(h, &catalogue) {
                    Outcome::System(sys, _) => Some(sys),
                    Outcome::NoDistance => None,
                })
                .collect()
        };
        mark_hosts(&mut stars, &systems);
        let hosts: Vec<bool> = stars.iter().map(Star3::is_exo_host).collect();
        assert_eq!(hosts, [false, false, true, false]);
        assert!(stars[0].is_named(), "other flags are kept");
        assert_eq!(stars[2].athyg_id(), 102, "ids are kept");
    }

    #[test]
    fn archive_rows_grouped_by_host() {
        let csv = format!(
            "{HEADER}\n\
\"51 Peg b\",\"51 Peg\",\"HIP 113357\",\"HD 217014\",\"Gaia DR3 2835207319109249920\",344.3665854,20.7688425,15.4614,5.46,5758.00,1.15,0.0527,0.0100,,58.0,4.2307869,,,0.4600,,1260.0,\"Radial Velocity\",1995\n\
\"Kepler-9 c\",\"Kepler-9\",,,\"Gaia DR3 2102522069447327872\",285.3812,38.4006,641.0,13.9,5774.00,0.96,0.225,0.06,88.9,,38.9,2455013.7,\"BJD-TDB\",0.137,0.738,,\"Transit\",2010\n\
\"Kepler-9 b\",\"Kepler-9\",,,\"Gaia DR3 2102522069447327872\",285.3812,38.4006,641.0,13.9,5777.00,0.96,0.143,0.06,88.6,,19.2,2455008.1,\"BJD-TDB\",0.252,0.842,,\"Transit\",2010\n\
\"HD 41004 B b\",\"HD 41004 B\",\"HIP 28393 B\",\"HD 41004 B\",,90.0,-48.0,37.0,12.3,,,0.0177,0.081,,178.5,1.328,,,18.4,,,\"Radial Velocity\",2004\n"
        );
        let hosts = parse_archive(csv.as_bytes()).expect("parse");
        let names: Vec<&str> = hosts.iter().map(|h| h.name.as_str()).collect();
        assert_eq!(
            names,
            ["51 Peg", "HD 41004 B", "Kepler-9"],
            "sorted by host name"
        );
        let peg = &hosts[0];
        assert_eq!((peg.hip, peg.hd), (Some(113_357), Some(217_014)));
        assert_eq!(peg.gaia_dr3, Some(2_835_207_319_109_249_920));
        assert_eq!((peg.dist_pc, peg.v_mag), (Some(15.4614), Some(5.46)));
        let b = &peg.planets[0];
        assert_eq!(b.name, "51 Peg b");
        assert_eq!(
            (b.a_au, b.e, b.i_deg, b.argp_deg),
            (Some(0.0527), Some(0.01), None, Some(58.0))
        );
        assert_eq!(
            (b.period_d, b.t_transit_jd, b.mass_mjup),
            (Some(4.2307869), None, Some(0.46))
        );
        assert_eq!((b.radius_rjup, b.teq_k), (None, Some(1260.0)));
        assert_eq!((b.method.as_str(), b.year), ("Radial Velocity", 1995));
        // A "B" component shares the primary's HIP and HD numbers, so they are not used.
        assert_eq!(
            (hosts[1].hip, hosts[1].hd, hosts[1].gaia_dr3),
            (None, None, None)
        );
        // Planets sorted by name; host values from the first planet that has them.
        let kepler9 = &hosts[2];
        let planets: Vec<&str> = kepler9.planets.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(planets, ["Kepler-9 b", "Kepler-9 c"]);
        assert_eq!(kepler9.teff_k, Some(5777.0), "Kepler-9 b's st_teff");
        assert_eq!(kepler9.planets[0].t_transit_jd, Some(2455008.1));
    }

    #[test]
    fn archive_header_must_have_every_column() {
        let csv = format!("{}\n", HEADER.replace(",sy_vmag,", ",sy_gaiamag,"));
        assert!(parse_archive(csv.as_bytes()).is_err());
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&[
            "--input", "ps.csv", "--stars", "s.bin", "--athyg", "a.csv.gz", "--out", "x.ron",
        ]))
        .expect("valid");
        assert_eq!(
            options,
            Options {
                input: "ps.csv".into(),
                stars: "s.bin".into(),
                athyg: "a.csv.gz".into(),
                out: "x.ron".into()
            }
        );
        assert!(parse_options(&args(&["--input", "ps.csv"])).is_err());
        assert!(parse_options(&args(&["--names", "n.ron"])).is_err());
    }
}
