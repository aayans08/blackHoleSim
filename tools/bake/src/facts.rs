//! `uni-bake facts` (M1 Task A9 Step 3): the hand-curated NSSDCA fact sheets → `body_facts.ron`.
//!
//! `uni-bake facts --out data/baked/body_facts.ron [--sheets <dir>]`
//!
//! The facts are transcribed by hand, below, from the NASA NSSDCA fact sheets
//! (<https://nssdc.gsfc.nasa.gov/planetary/factsheet/>). Each value is written as the number
//! printed on its sheet, plus a unit. The bake downloads every sheet with `curl` into `--sheets`
//! (default `data/cache/facts/`; delete it after the bake) and stops, before writing, if a
//! transcribed number does not occur on its sheet. It also stops if a mean radius differs from
//! `uni_physics::bodies` (the volumetric mean of the WGCCRE 2015 radii) by more than 0.5 %, or a
//! mass from GM / G by more than 10 % (Deimos excepted, see [`MASS_EXCEPTIONS`]), and it prints
//! every mass that differs by more than 1 %.
//!
//! Value syntax: `[sheet:]number[ unit]`. Without a `sheet:` prefix the number is on the body's
//! own sheet; `table:` is the planetary fact table (the factsheet index page), `small:` the
//! "Solar System Small Worlds" table (`galileanfact_table.html`). Units: `e<N>` (× 10^N kg),
//! `au`, `km`, `e3km`, `e6km`, `mb`, `bar`, `microbar`, `C`, `K`, `d` (days, for a rotation
//! period). A radius may be three axes, `a x b x c`, whose volumetric mean (a·b·c)^⅓ is kept. A
//! rotation of `sync` is the orbital period × 24 h, for a moon the sheet marks "S" (synchronous).
//! Derived values: K = °C + 273.15; 1 bar = 100 kPa; 1 au = 149 597 870.7 km (IAU 2012
//! Resolution B2).

use std::collections::{BTreeMap, HashSet};
use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, bail, ensure};
use uni_data::BodyFacts;

use crate::sats::clean;
use crate::sbdb::{curl_get, utc_now, write_ron};

const USAGE: &str = "usage: uni-bake facts --out <body_facts.ron> [--sheets <dir>]";

/// The NSSDCA fact-sheet directory.
const BASE_URL: &str = "https://nssdc.gsfc.nasa.gov/planetary/factsheet/";

/// Bodies whose sheet mass is known to differ from GM / G by more than 10 %: Deimos's 2.4e15 kg
/// against 1.44e15 kg from the MAR097 GM (JPL SSD `/sats/phys_par/`).
const MASS_EXCEPTIONS: [i32; 1] = [402];

/// One hand-curated row: values as `[sheet:]number[ unit]` (module doc).
#[derive(Clone, Debug)]
struct Curated {
    naif: i32,
    name: &'static str,
    kind: &'static str,
    /// The body's own sheet (file name in the factsheet directory).
    sheet: &'static str,
    radius: &'static str,
    mass: &'static str,
    gravity: Option<&'static str>,
    escape: Option<&'static str>,
    rotation: Option<&'static str>,
    period: Option<&'static str>,
    semi_major: Option<&'static str>,
    tilt: Option<&'static str>,
    pressure: Option<&'static str>,
    atmosphere: Option<&'static str>,
    temp: Option<&'static str>,
    bond: Option<&'static str>,
    geometric: Option<&'static str>,
    moons: &'static [&'static str],
}

/// Runs `uni-bake facts` with the options after the subcommand name.
pub fn run(options: &[OsString]) -> anyhow::Result<()> {
    let options = parse_options(options)?;
    std::fs::create_dir_all(&options.sheets)
        .with_context(|| format!("creating {}", options.sheets.display()))?;
    let rows = curated();
    let fetched = utc_now();
    let mut files: Vec<&'static str> = vec![TABLE_SHEET, SMALL_SHEET];
    for row in &rows {
        if !files.contains(&row.sheet) {
            files.push(row.sheet);
        }
    }
    let mut sheets = BTreeMap::new();
    for file in files {
        let raw = options.sheets.join(file);
        let url = sheet_url(file);
        log::info!("facts: {url}");
        curl_get(&url, &[], &raw)?;
        let bytes = std::fs::read(&raw).with_context(|| format!("reading {}", raw.display()))?;
        sheets.insert(file, sheet_numbers(&String::from_utf8_lossy(&bytes)));
    }

    let mut facts = Vec::with_capacity(rows.len());
    let mut mass_gaps = Vec::new();
    for row in &rows {
        let fact =
            body_facts(row, &sheets).with_context(|| format!("{} ({})", row.name, row.naif))?;
        if let Some(body) = uni_physics::bodies::by_naif(row.naif) {
            let r = body.radii_m * 1e-3;
            let mean = (r.x * r.y * r.z).cbrt();
            let radius_gap = (fact.mean_radius_km - mean).abs() / mean;
            ensure!(
                radius_gap < 0.005,
                "{}: mean radius {} km is {:.2} % from uni_physics::bodies' {mean:.2} km",
                row.name,
                fact.mean_radius_km,
                100.0 * radius_gap
            );
            // CODATA 2018 G = 6.674 30e-11 m³ kg⁻¹ s⁻².
            let mass = body.gm_m3s2 / 6.674_30e-11;
            let mass_gap = (fact.mass_kg - mass) / mass;
            ensure!(
                mass_gap.abs() < 0.1 || MASS_EXCEPTIONS.contains(&row.naif),
                "{}: mass {:e} kg is {:.1} % from GM / G = {mass:e} kg",
                row.name,
                fact.mass_kg,
                100.0 * mass_gap
            );
            if mass_gap.abs() > 0.01 {
                mass_gaps.push(format!("{} {:+.1} %", row.name, 100.0 * mass_gap));
            }
        }
        facts.push(fact);
    }
    for gap in &mass_gaps {
        println!("facts: sheet mass differs from uni_physics GM / G by more than 1 %: {gap}");
    }

    let header = vec![
        "Body facts for the info cards, hand-curated from the NASA NSSDCA fact sheets (M1 Task A9 Step 3).".to_owned(),
        format!("Written by `uni-bake facts`, which checked every number against its sheet, fetched {fetched} UTC"),
        format!("from {BASE_URL} (`source` names each body's sheets). Public NASA data;"),
        "see ATTRIBUTION.md and data/manifest.toml (id \"nssdca-facts\").".to_owned(),
        String::new(),
        "Units: km, kg, m/s^2, km/s, hours (negative: retrograde), days, au, deg, kPa, K. Conversions:".to_owned(),
        "K = C + 273.15; 1 bar = 100 kPa; 1 au = 149 597 870.7 km (IAU 2012 B2). mean_radius_km: the".to_owned(),
        "sheet's volumetric mean, or (a b c)^(1/3) of the axes it gives. sidereal_rotation_h of a moon the".to_owned(),
        "sheet marks synchronous without a number: its orbital period x 24. mean_temp_k: the fact tables'".to_owned(),
        "\"Mean Temperature\" (giant planets at 1 bar; the Sun: effective temperature). None: not on the sheets".to_owned(),
        "(no surface pressure for giants or exospheres; Iapetus's two-faced albedo 0.05 / 0.5; Hyperion's".to_owned(),
        "chaotic rotation). moons: this body's moons that have an entry here; all moons: sat_elements.ron.".to_owned(),
        format!("Masses more than 1 % from the JPL GM / G (stale sheet values): {}.", if mass_gaps.is_empty() { "none".to_owned() } else { mass_gaps.join(", ") }),
    ];
    write_ron(&options.out, &header, &facts, false)?;
    let back = uni_data::load_body_facts(&options.out).context("reading back the written file")?;
    ensure!(
        back.len() == facts.len(),
        "read back {} of {} bodies",
        back.len(),
        facts.len()
    );
    println!(
        "facts: {} bodies from {} sheets -> {}",
        facts.len(),
        sheets.len(),
        options.out.display()
    );
    Ok(())
}

/// The planetary fact table (the factsheet directory's index page).
const TABLE_SHEET: &str = "index.html";
/// The "Solar System Small Worlds" fact table.
const SMALL_SHEET: &str = "galileanfact_table.html";

/// The URL of a sheet file.
fn sheet_url(file: &str) -> String {
    if file == TABLE_SHEET {
        BASE_URL.to_owned()
    } else {
        format!("{BASE_URL}{file}")
    }
}

/// Every number printed on a sheet: runs of digits, ',', '.', '+' and '-' in its text, each also
/// without a trailing '.' or ',' and a leading '+'. Inline tags (`<sup>`, `<a>`, …) are dropped,
/// so "10<sup>24</sup>" reads "1024"; every other tag separates.
fn sheet_numbers(html: &str) -> HashSet<String> {
    let mut text = String::with_capacity(html.len());
    let mut rest = html;
    while let Some(lt) = rest.find('<') {
        text.push_str(&rest[..lt]);
        let gt = rest[lt..].find('>').map_or(rest.len(), |g| lt + g + 1);
        let name: String = rest[lt + 1..gt]
            .trim_start_matches('/')
            .chars()
            .take_while(char::is_ascii_alphanumeric)
            .collect::<String>()
            .to_ascii_lowercase();
        if !matches!(
            name.as_str(),
            "sup" | "sub" | "a" | "b" | "i" | "em" | "strong" | "span" | "font" | "u"
        ) {
            text.push(' ');
        }
        rest = &rest[gt..];
    }
    text.push_str(rest);
    let text = clean(&text);
    let mut numbers = HashSet::new();
    for run in text.split(|c: char| !(c.is_ascii_digit() || matches!(c, ',' | '.' | '+' | '-'))) {
        let trimmed = run.trim_end_matches(['.', ',']);
        let unsigned = trimmed.strip_prefix('+').unwrap_or(trimmed);
        for variant in [run, trimmed, unsigned] {
            if variant.bytes().any(|b| b.is_ascii_digit()) {
                numbers.insert(variant.to_owned());
            }
        }
    }
    numbers
}

/// The file name of a value's sheet: its `sheet:` prefix, or the row's own sheet.
fn sheet_of(spec: &str, own: &'static str) -> anyhow::Result<&'static str> {
    match spec.split_once(':') {
        None => Ok(own),
        Some(("table", _)) => Ok(TABLE_SHEET),
        Some(("small", _)) => Ok(SMALL_SHEET),
        Some((other, _)) => bail!("unknown sheet {other:?} in {spec:?}"),
    }
}

/// Reads values of one row, checking each number against its sheet and noting the sheets used.
struct Reader<'a> {
    row: &'a Curated,
    sheets: &'a BTreeMap<&'static str, HashSet<String>>,
    used: Vec<&'static str>,
}

impl Reader<'_> {
    /// The numbers (`a x b x c` gives three) and the unit of `spec`, after checking that every
    /// number is printed on its sheet. Numbers come back as decimal text, without ',' or a
    /// trailing '.', so a power of ten can be applied exactly ([`number`]).
    fn read(&mut self, spec: &'static str) -> anyhow::Result<(Vec<String>, &'static str)> {
        let sheet = sheet_of(spec, self.row.sheet)?;
        let body = spec.split_once(':').map_or(spec, |(_, body)| body);
        let (literals, unit) = match body.rsplit_once(' ') {
            Some((head, unit)) if unit.starts_with(|c: char| c.is_ascii_alphabetic()) => {
                (head, unit)
            }
            _ => (body, ""),
        };
        let numbers = self.check(sheet, literals.split(" x "), spec)?;
        Ok((numbers, unit))
    }

    /// Checks that each literal is printed on `sheet`; returns them without ',' or a trailing '.'.
    fn check<'s>(
        &mut self,
        sheet: &'static str,
        literals: impl Iterator<Item = &'s str>,
        spec: &str,
    ) -> anyhow::Result<Vec<String>> {
        let printed = self
            .sheets
            .get(sheet)
            .with_context(|| format!("sheet {sheet} was not downloaded"))?;
        if !self.used.contains(&sheet) {
            self.used.push(sheet);
        }
        literals
            .map(|literal| {
                ensure!(
                    printed.contains(literal),
                    "{literal:?} (from {spec:?}) is not printed on {sheet}"
                );
                let decimal = literal.replace(',', "");
                number(&decimal, 0)?;
                Ok(decimal.trim_end_matches('.').to_owned())
            })
            .collect()
    }

    /// One number with one of `units` ("" for none).
    fn scalar(
        &mut self,
        spec: &'static str,
        units: &[&str],
    ) -> anyhow::Result<(String, &'static str)> {
        let (mut numbers, unit) = self.read(spec)?;
        ensure!(numbers.len() == 1, "{spec:?}: one number expected");
        ensure!(
            units.contains(&unit),
            "{spec:?}: unit {unit:?} is not one of {units:?}"
        );
        Ok((numbers.remove(0), unit))
    }

    fn plain(&mut self, spec: Option<&'static str>) -> anyhow::Result<Option<f64>> {
        spec.map(|s| self.scalar(s, &[""]).and_then(|(v, _)| number(&v, 0)))
            .transpose()
    }
}

/// The decimal `text` × 10^`exponent`, parsed as one literal: the nearest f64, with no second
/// rounding.
fn number(text: &str, exponent: i32) -> anyhow::Result<f64> {
    format!("{}e{exponent}", text.trim_end_matches('.'))
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite())
        .with_context(|| format!("{text:?} is not a number"))
}

/// Rounds a value derived by decimal arithmetic (°C + 273.15, days × 24) to 12 significant
/// digits, more than any sheet prints, so binary rounding noise stays out of the file.
fn tidy(v: f64) -> f64 {
    if v == 0.0 {
        return v;
    }
    let scale = 10_f64.powi(11 - v.abs().log10().floor() as i32);
    (v * scale).round() / scale
}

/// 1 au [km] (IAU 2012 Resolution B2).
const AU_KM: f64 = uni_core::units::AU / 1000.0;

/// Builds one body's facts, checking every number against `sheets` (file name → numbers).
fn body_facts(
    row: &Curated,
    sheets: &BTreeMap<&'static str, HashSet<String>>,
) -> anyhow::Result<BodyFacts> {
    ensure!(
        uni_data::facts::KINDS.contains(&row.kind),
        "unknown kind {:?}",
        row.kind
    );
    let mut r = Reader {
        row,
        sheets,
        used: Vec::new(),
    };
    let (axes, unit) = r.read(row.radius)?;
    ensure!(
        unit.is_empty() && matches!(axes.len(), 1 | 3),
        "radius {:?}",
        row.radius
    );
    let mean_radius_km = match axes.as_slice() {
        [radius] => number(radius, 0)?,
        _ => {
            let product: f64 = axes
                .iter()
                .map(|a| number(a, 0))
                .product::<anyhow::Result<f64>>()?;
            product.cbrt()
        }
    };
    let (mass, unit) = r.read(row.mass)?;
    let exponent: i32 = unit
        .strip_prefix('e')
        .and_then(|e| e.parse().ok())
        .filter(|_| mass.len() == 1)
        .with_context(|| {
            format!(
                "mass {:?} needs one number and a unit e<N> (× 10^N kg)",
                row.mass
            )
        })?;
    let mass_kg = number(&mass[0], exponent)?;
    let orbital_period_d = r.plain(row.period)?;
    let sidereal_rotation_h = match row.rotation {
        None => None,
        Some("sync") => Some(tidy(
            orbital_period_d.context("a synchronous rotation needs the orbital period")? * 24.0,
        )),
        Some(spec) => match r.scalar(spec, &["", "d"])? {
            (hours, "") => Some(number(&hours, 0)?),
            (days, _) => Some(tidy(number(&days, 0)? * 24.0)),
        },
    };
    let semi_major_au = match row.semi_major {
        None => None,
        Some(spec) => Some(match r.scalar(spec, &["au", "km", "e3km", "e6km"])? {
            (au, "au") => number(&au, 0)?,
            (km, "km") => number(&km, 0)? / AU_KM,
            (km, "e3km") => number(&km, 3)? / AU_KM,
            (km, _) => number(&km, 6)? / AU_KM,
        }),
    };
    // 1 bar = 100 kPa; 1 mb = 0.1 kPa; 1 microbar = 1e-4 kPa.
    let surface_pressure_kpa = match row.pressure {
        None => None,
        Some(spec) => Some(match r.scalar(spec, &["mb", "bar", "microbar"])? {
            (p, "bar") => number(&p, 2)?,
            (p, "mb") => number(&p, -1)?,
            (p, _) => number(&p, -4)?,
        }),
    };
    let mean_temp_k = match row.temp {
        None => None,
        Some(spec) => Some(match r.scalar(spec, &["C", "K"])? {
            (t, "C") => tidy(number(&t, 0)? + 273.15),
            (t, _) => number(&t, 0)?,
        }),
    };
    if let Some(atmosphere) = row.atmosphere {
        let runs = atmosphere
            .split(|c: char| !(c.is_ascii_digit() || c == '.'))
            .map(|run| run.trim_end_matches('.'))
            .filter(|run| !run.is_empty());
        r.check(row.sheet, runs, atmosphere)?;
    }
    let surface_gravity_ms2 = r.plain(row.gravity)?;
    let escape_velocity_kms = r.plain(row.escape)?;
    let axial_tilt_deg = r.plain(row.tilt)?;
    let bond_albedo = r.plain(row.bond)?;
    let geometric_albedo = r.plain(row.geometric)?;
    // The body's own sheet first, then the tables in the order first used.
    let mut used = r.used;
    used.sort_by_key(|sheet| *sheet != row.sheet);
    Ok(BodyFacts {
        naif: row.naif,
        name: row.name.to_owned(),
        kind: row.kind.to_owned(),
        mean_radius_km,
        mass_kg,
        surface_gravity_ms2,
        escape_velocity_kms,
        sidereal_rotation_h,
        orbital_period_d,
        semi_major_au,
        axial_tilt_deg,
        surface_pressure_kpa,
        atmosphere: row.atmosphere.map(str::to_owned),
        mean_temp_k,
        bond_albedo,
        geometric_albedo,
        moons: row.moons.iter().map(|m| (*m).to_owned()).collect(),
        source: used
            .iter()
            .map(|s| sheet_url(s))
            .collect::<Vec<_>>()
            .join("; "),
    })
}

#[derive(Debug)]
struct Options {
    out: PathBuf,
    sheets: PathBuf,
}

/// `--out` (required) and `--sheets` (default `<data>/cache/facts`).
fn parse_options(options: &[OsString]) -> anyhow::Result<Options> {
    let (mut out, mut sheets) = (None, None);
    let mut options = options.iter();
    while let Some(flag) = options.next() {
        let slot = match flag.to_str() {
            Some("--out") => &mut out,
            Some("--sheets") => &mut sheets,
            _ => bail!("unknown option {flag:?}\n{USAGE}"),
        };
        let value = options
            .next()
            .with_context(|| format!("{flag:?} needs a value\n{USAGE}"))?;
        *slot = Some(PathBuf::from(value));
    }
    let out = out.with_context(|| format!("facts needs --out\n{USAGE}"))?;
    let sheets = sheets.unwrap_or_else(|| uni_data::data_dir().join("cache").join("facts"));
    Ok(Options { out, sheets })
}

/// A row with every optional value absent, to fill with struct-update syntax.
const NONE: Curated = Curated {
    naif: 0,
    name: "",
    kind: "moon",
    sheet: "",
    radius: "",
    mass: "",
    gravity: None,
    escape: None,
    rotation: None,
    period: None,
    semi_major: None,
    tilt: None,
    pressure: None,
    atmosphere: None,
    temp: None,
    bond: None,
    geometric: None,
    moons: &[],
};

/// The hand-curated table, transcribed 2026-11-22 from the sheets as served that day (each sheet
/// names its own "Last Updated" date). Every number is as printed; see the module doc for units.
fn curated() -> Vec<Curated> {
    let planet = |naif, name, sheet| Curated {
        naif,
        name,
        kind: "planet",
        sheet,
        ..NONE
    };
    vec![
        Curated {
            kind: "star",
            radius: "695,700.",
            mass: "1,988,400. e24",
            gravity: Some("274.0"),
            escape: Some("617.6"),
            rotation: Some("609.12"),
            tilt: Some("7.25"),
            atmosphere: Some("H 90.965%, He 8.889% (photosphere)"),
            temp: Some("5772 K"),
            ..Curated {
                naif: 10,
                name: "Sun",
                sheet: "sunfact.html",
                ..NONE
            }
        },
        Curated {
            radius: "2439.7",
            mass: "0.33010 e24",
            gravity: Some("3.70"),
            escape: Some("4.3"),
            rotation: Some("1407.6"),
            period: Some("87.969"),
            semi_major: Some("0.38709893 au"),
            tilt: Some("0.034"),
            atmosphere: Some("Exosphere: Na, Mg, O, H, K, Ca"),
            temp: Some("table:167 C"),
            bond: Some("0.068"),
            geometric: Some("0.142"),
            ..planet(199, "Mercury", "mercuryfact.html")
        },
        Curated {
            radius: "6051.8",
            mass: "4.8673 e24",
            gravity: Some("8.87"),
            escape: Some("10.36"),
            rotation: Some("-5832.6"),
            period: Some("224.701"),
            semi_major: Some("0.72333199 au"),
            tilt: Some("177.36"),
            pressure: Some("92 bar"),
            atmosphere: Some("CO2 96.5%, N2 3.5%"),
            temp: Some("table:464 C"),
            bond: Some("0.77"),
            geometric: Some("0.689"),
            ..planet(299, "Venus", "venusfact.html")
        },
        Curated {
            radius: "6371.000",
            mass: "5.9722 e24",
            gravity: Some("9.820"),
            escape: Some("11.186"),
            rotation: Some("23.9345"),
            period: Some("365.256"),
            semi_major: Some("1.00000011 au"),
            tilt: Some("23.44"),
            pressure: Some("1014 mb"),
            atmosphere: Some("N2 78.08%, O2 20.95%, Ar 9340 ppm, CO2 420 ppm"),
            temp: Some("table:15 C"),
            bond: Some("0.294"),
            geometric: Some("0.434"),
            moons: &["Moon"],
            ..planet(399, "Earth", "earthfact.html")
        },
        Curated {
            radius: "3389.5",
            mass: "0.64169 e24",
            gravity: Some("3.73"),
            escape: Some("5.03"),
            rotation: Some("24.6229"),
            period: Some("686.980"),
            semi_major: Some("1.52366231 au"),
            tilt: Some("25.19"),
            pressure: Some("6.36 mb"),
            atmosphere: Some("CO2 95.1%, N2 2.59%, Ar 1.94%, O2 0.16%, CO 0.06%"),
            temp: Some("table:-65 C"),
            bond: Some("0.250"),
            geometric: Some("0.170"),
            moons: &["Phobos", "Deimos"],
            ..planet(499, "Mars", "marsfact.html")
        },
        Curated {
            radius: "69,911",
            mass: "1,898.13 e24",
            gravity: Some("25.92"),
            escape: Some("59.5"),
            rotation: Some("9.9250"),
            period: Some("4,332.589"),
            semi_major: Some("5.20336301 au"),
            tilt: Some("3.13"),
            atmosphere: Some("H2 89.8%, He 10.2%"),
            temp: Some("table:-110 C"),
            bond: Some("0.343"),
            geometric: Some("0.538"),
            moons: &["Io", "Europa", "Ganymede", "Callisto"],
            ..planet(599, "Jupiter", "jupiterfact.html")
        },
        Curated {
            radius: "58,232",
            mass: "568.32 e24",
            gravity: Some("11.19"),
            escape: Some("35.5"),
            rotation: Some("10.656"),
            period: Some("10,755.699"),
            semi_major: Some("9.53707032 au"),
            tilt: Some("26.73"),
            atmosphere: Some("H2 96.3%, He 3.25%"),
            temp: Some("table:-140 C"),
            bond: Some("0.342"),
            geometric: Some("0.499"),
            moons: &[
                "Mimas",
                "Enceladus",
                "Tethys",
                "Dione",
                "Rhea",
                "Titan",
                "Hyperion",
                "Iapetus",
            ],
            ..planet(699, "Saturn", "saturnfact.html")
        },
        Curated {
            radius: "25,362",
            mass: "86.811 e24",
            gravity: Some("9.01"),
            escape: Some("21.3"),
            rotation: Some("-17.24"),
            period: Some("30,685.400"),
            semi_major: Some("19.19126393 au"),
            tilt: Some("97.77"),
            atmosphere: Some("H2 82.5%, He 15.2%, CH4 2.3%"),
            temp: Some("table:-195 C"),
            bond: Some("0.300"),
            geometric: Some("0.488"),
            moons: &["Miranda", "Ariel", "Umbriel", "Titania", "Oberon"],
            ..planet(799, "Uranus", "uranusfact.html")
        },
        Curated {
            radius: "24,622",
            mass: "102.409 e24",
            gravity: Some("11.27"),
            escape: Some("23.5"),
            rotation: Some("16.11"),
            period: Some("60,189.018"),
            semi_major: Some("30.06896348 au"),
            tilt: Some("28.32"),
            atmosphere: Some("H2 80.0%, He 19.0%, CH4 1.5%"),
            temp: Some("table:-200 C"),
            bond: Some("0.290"),
            geometric: Some("0.442"),
            moons: &["Triton"],
            ..planet(899, "Neptune", "neptunefact.html")
        },
        Curated {
            kind: "dwarf planet",
            radius: "1188",
            mass: "0.01303 e24",
            gravity: Some("0.62"),
            escape: Some("1.21"),
            rotation: Some("-153.2928"),
            period: Some("90,560"),
            semi_major: Some("39.48168677 au"),
            tilt: Some("119.51"),
            pressure: Some("13 microbar"),
            atmosphere: Some("N2 99%, CH4 0.5%, CO 0.05%"),
            temp: Some("table:-225 C"),
            bond: Some("0.72"),
            geometric: Some("0.52"),
            moons: &["Charon"],
            ..planet(999, "Pluto", "plutofact.html")
        },
        Curated {
            radius: "1737.4",
            mass: "0.07346 e24",
            gravity: Some("1.62"),
            escape: Some("2.38"),
            rotation: Some("655.720"),
            period: Some("27.3217"),
            semi_major: Some("0.3844 e6km"),
            tilt: Some("6.68"),
            atmosphere: Some("Exosphere: He, Ne, H2, Ar"),
            temp: Some("table:-20 C"),
            bond: Some("0.11"),
            geometric: Some("0.12"),
            ..Curated {
                naif: 301,
                name: "Moon",
                sheet: "moonfact.html",
                ..NONE
            }
        },
        // Mars's moons: the Mars sheet gives three axes, rotation in days, mass in 10^15 kg.
        Curated {
            radius: "13.0 x 11.4 x 9.1",
            mass: "10.6 e15",
            rotation: Some("0.31891 d"),
            period: Some("0.31891"),
            semi_major: Some("9378 km"),
            geometric: Some("0.07"),
            ..Curated {
                naif: 401,
                name: "Phobos",
                sheet: "marsfact.html",
                ..NONE
            }
        },
        Curated {
            radius: "7.8 x 6.0 x 5.1",
            mass: "2.4 e15",
            rotation: Some("1.26244 d"),
            period: Some("1.26244"),
            semi_major: Some("23459 km"),
            geometric: Some("0.08"),
            ..Curated {
                naif: 402,
                name: "Deimos",
                sheet: "marsfact.html",
                ..NONE
            }
        },
        // Galilean moons: the Jovian sheet (10^20 kg, 10^3 km); gravity, escape velocity, rotation
        // (hours) and mean temperature from the Small Worlds table.
        Curated {
            gravity: Some("small:1.80"),
            escape: Some("small:2.6"),
            rotation: Some("small:42.5"),
            temp: Some("small:-155 C"),
            ..galilean(
                501,
                "Io",
                "893.2 e20",
                "1821.5",
                "1.769138",
                "421.8 e3km",
                "0.62",
            )
        },
        Curated {
            gravity: Some("small:1.31"),
            escape: Some("small:2.0"),
            rotation: Some("small:85.2"),
            temp: Some("small:-170 C"),
            ..galilean(
                502,
                "Europa",
                "480.0 e20",
                "1560.8",
                "3.551181",
                "671.1 e3km",
                "0.68",
            )
        },
        Curated {
            gravity: Some("small:1.43"),
            escape: Some("small:2.7"),
            rotation: Some("small:171.7"),
            temp: Some("small:-160 C"),
            ..galilean(
                503,
                "Ganymede",
                "1481.9 e20",
                "2631.2",
                "7.154553",
                "1070.4 e3km",
                "0.44",
            )
        },
        Curated {
            gravity: Some("small:1.24"),
            escape: Some("small:2.4"),
            rotation: Some("small:400.5"),
            temp: Some("small:-155 C"),
            ..galilean(
                504,
                "Callisto",
                "1075.9 e20",
                "2410.3",
                "16.689017",
                "1882.7 e3km",
                "0.19",
            )
        },
        // Saturn's moons: the Saturnian sheet (10^20 kg, axes in km, 10^3 km; "S" synchronous).
        saturnian(
            601,
            "Mimas",
            "0.379 e20",
            "208 x 197 x 191",
            "0.9424218",
            "185.52 e3km",
            Some("0.6"),
        ),
        saturnian(
            602,
            "Enceladus",
            "1.08 e20",
            "257 x 251 x 248",
            "1.370218",
            "238.02 e3km",
            Some("1.0"),
        ),
        saturnian(
            603,
            "Tethys",
            "6.18 e20",
            "538 x 528 x 526",
            "1.887802",
            "294.66 e3km",
            Some("0.8"),
        ),
        saturnian(
            604,
            "Dione",
            "11.0 e20",
            "563 x 561 x 560",
            "2.736915",
            "377.40 e3km",
            Some("0.7"),
        ),
        saturnian(
            605,
            "Rhea",
            "23.1 e20",
            "765 x 763 x 762",
            "4.517500",
            "527.04 e3km",
            Some("0.7"),
        ),
        Curated {
            gravity: Some("small:1.35"),
            escape: Some("small:2.6"),
            rotation: Some("small:382.7"),
            temp: Some("small:-180 C"),
            ..saturnian(
                606,
                "Titan",
                "1,345.5 e20",
                "2,575",
                "15.945421",
                "1,221.87 e3km",
                Some("0.22"),
            )
        },
        // Hyperion rotates chaotically ("C" on the sheet): no rotation period.
        Curated {
            rotation: None,
            ..saturnian(
                607,
                "Hyperion",
                "0.056 e20",
                "180 x 133 x 103",
                "21.276609",
                "1,500.93 e3km",
                Some("0.3"),
            )
        },
        // Iapetus: the sheet's geometric albedo is "0.05 / 0.5" (dark and bright faces): none kept.
        saturnian(
            608,
            "Iapetus",
            "18.1 e20",
            "746 x 746 x 712",
            "79.330183",
            "3,560.85 e3km",
            None,
        ),
        // Uranus's major moons: the Uranian sheet (10^20 kg, 10^3 km, "S" synchronous).
        uranian(
            705,
            "Miranda",
            "0.66 e20",
            "240 x 234.2 x 232.9",
            "1.413479",
            "129.90 e3km",
            "0.32",
        ),
        uranian(
            701,
            "Ariel",
            "12.9 e20",
            "581.1 x 577.9 x 577.7",
            "2.520379",
            "190.90 e3km",
            "0.39",
        ),
        uranian(
            702,
            "Umbriel",
            "12.2 e20",
            "584.7",
            "4.144176",
            "266.00 e3km",
            "0.21",
        ),
        uranian(
            703,
            "Titania",
            "34.2 e20",
            "788.9",
            "8.705867",
            "436.30 e3km",
            "0.27",
        ),
        uranian(
            704,
            "Oberon",
            "28.8 e20",
            "761.4",
            "13.463234",
            "583.50 e3km",
            "0.23",
        ),
        // Triton: the Neptunian sheet; the rest from the Small Worlds table (rotation −141.0 h:
        // synchronous with its retrograde orbit).
        Curated {
            radius: "1,353.4",
            mass: "214. e20",
            gravity: Some("small:0.78"),
            escape: Some("small:1.5"),
            rotation: Some("small:-141.0"),
            period: Some("5.876854"),
            semi_major: Some("354.76 e3km"),
            temp: Some("small:-215 C"),
            geometric: Some("0.72"),
            ..Curated {
                naif: 801,
                name: "Triton",
                sheet: "neptuniansatfact.html",
                ..NONE
            }
        },
        // Charon: the Pluto sheet (10^21 kg; rotation in days).
        Curated {
            radius: "606",
            mass: "1.586 e21",
            gravity: Some("0.29"),
            escape: Some("0.59"),
            rotation: Some("6.3872 d"),
            period: Some("6.3872"),
            semi_major: Some("19,596 km"),
            bond: Some("0.25"),
            geometric: Some("0.42"),
            ..Curated {
                naif: 901,
                name: "Charon",
                sheet: "plutofact.html",
                ..NONE
            }
        },
    ]
}

/// A Galilean moon's values from the Jovian sheet; the row adds the Small Worlds table's.
fn galilean(
    naif: i32,
    name: &'static str,
    mass: &'static str,
    radius: &'static str,
    period: &'static str,
    semi_major: &'static str,
    geometric: &'static str,
) -> Curated {
    Curated {
        naif,
        name,
        sheet: "joviansatfact.html",
        radius,
        mass,
        period: Some(period),
        semi_major: Some(semi_major),
        geometric: Some(geometric),
        ..NONE
    }
}

/// A Saturnian moon from the Saturnian sheet, rotating synchronously.
fn saturnian(
    naif: i32,
    name: &'static str,
    mass: &'static str,
    radius: &'static str,
    period: &'static str,
    semi_major: &'static str,
    geometric: Option<&'static str>,
) -> Curated {
    Curated {
        naif,
        name,
        sheet: "saturniansatfact.html",
        radius,
        mass,
        rotation: Some("sync"),
        period: Some(period),
        semi_major: Some(semi_major),
        geometric,
        ..NONE
    }
}

/// A Uranian moon from the Uranian sheet, rotating synchronously.
fn uranian(
    naif: i32,
    name: &'static str,
    mass: &'static str,
    radius: &'static str,
    period: &'static str,
    semi_major: &'static str,
    geometric: &'static str,
) -> Curated {
    Curated {
        naif,
        name,
        sheet: "uraniansatfact.html",
        radius,
        mass,
        rotation: Some("sync"),
        period: Some(period),
        semi_major: Some(semi_major),
        geometric: Some(geometric),
        ..NONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbers(html: &str) -> HashSet<String> {
        sheet_numbers(html)
    }

    #[test]
    fn sheet_numbers_are_the_printed_numbers() {
        let html = "<tr><td>Mass (10<sup>24</sup> kg)<td>1,898.13</td></tr> 9.9250* 5.876854R | -110 |\n\
                    1,988,400. (2.0%) Surface pressure: 6.36 mb &nbsp;+11.8 end 1.6.";
        let n = numbers(html);
        for present in [
            "1024",
            "1,898.13",
            "9.9250",
            "5.876854",
            "-110",
            "1,988,400.",
            "1,988,400",
            "2.0",
            "6.36",
            "11.8",
            "1.6",
        ] {
            assert!(n.contains(present), "{present} missing from {n:?}");
        }
        for absent in ["1898.13", "24", "110", "988"] {
            assert!(
                !n.contains(absent),
                "{absent} should not be a number of the sheet"
            );
        }
    }

    /// A row whose numbers are all on its sheets, to test the conversions.
    fn sample() -> Curated {
        Curated {
            naif: 399,
            name: "Earth",
            kind: "planet",
            sheet: "earthfact.html",
            radius: "6371.000",
            mass: "5.9722 e24",
            gravity: Some("9.820"),
            escape: Some("11.186"),
            rotation: Some("23.9345"),
            period: Some("365.256"),
            semi_major: Some("1.00000011 au"),
            tilt: Some("23.44"),
            pressure: Some("1014 mb"),
            atmosphere: Some("N2 78.08%, O2 20.95%"),
            temp: Some("table:15 C"),
            bond: Some("0.294"),
            geometric: Some("0.434"),
            moons: &["Moon"],
        }
    }

    fn sheets(own: &str, table: &str) -> BTreeMap<&'static str, HashSet<String>> {
        BTreeMap::from([
            ("earthfact.html", numbers(own)),
            ("index.html", numbers(table)),
            ("galileanfact_table.html", numbers("")),
        ])
    }

    const EARTH_SHEET: &str = "6371.000 5.9722 9.820 11.186 23.9345 365.256 1.00000011 23.44 1014 \
                               78.08% 20.95% 0.294 0.434 N2 O2 0.3844 9378 421.8 92 13 5772 \
                               208 x 197 x 191 6.3872 6.36 | 1,988,400. |";

    #[test]
    fn converts_each_unit() {
        let all = sheets(EARTH_SHEET, "| 15 | -155 |");
        let earth = body_facts(&sample(), &all).expect("earth");
        assert_eq!(earth.mass_kg, 5.9722e24);
        assert_eq!(earth.mean_radius_km, 6371.0);
        assert_eq!(earth.surface_pressure_kpa, Some(101.4));
        assert_eq!(earth.mean_temp_k, Some(288.15));
        assert_eq!(earth.semi_major_au, Some(1.000_000_11));
        assert_eq!(earth.sidereal_rotation_h, Some(23.9345));
        assert_eq!(earth.atmosphere.as_deref(), Some("N2 78.08%, O2 20.95%"));
        assert_eq!(earth.moons, ["Moon"]);
        assert_eq!(
            earth.source,
            "https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html; \
             https://nssdc.gsfc.nasa.gov/planetary/factsheet/"
        );
        let au_km = 149_597_870.7;
        let with = |edit: fn(&mut Curated)| {
            let mut row = sample();
            edit(&mut row);
            body_facts(&row, &all).expect("converts")
        };
        assert_eq!(
            with(|r| r.semi_major = Some("0.3844 e6km")).semi_major_au,
            Some(0.3844e6 / au_km)
        );
        assert_eq!(
            with(|r| r.semi_major = Some("421.8 e3km")).semi_major_au,
            Some(421.8e3 / au_km)
        );
        assert_eq!(
            with(|r| r.semi_major = Some("9378 km")).semi_major_au,
            Some(9378.0 / au_km)
        );
        assert_eq!(
            with(|r| r.pressure = Some("92 bar")).surface_pressure_kpa,
            Some(9200.0)
        );
        assert_eq!(
            with(|r| r.pressure = Some("13 microbar")).surface_pressure_kpa,
            Some(0.0013)
        );
        assert_eq!(
            with(|r| r.pressure = Some("6.36 mb")).surface_pressure_kpa,
            Some(0.636)
        );
        assert_eq!(with(|r| r.temp = Some("5772 K")).mean_temp_k, Some(5772.0));
        assert_eq!(
            with(|r| r.temp = Some("table:-155 C")).mean_temp_k,
            Some(118.15)
        );
        let axes = with(|r| r.radius = "208 x 197 x 191").mean_radius_km;
        assert!(
            (axes - (208.0_f64 * 197.0 * 191.0).cbrt()).abs() < 1e-12,
            "{axes}"
        );
        // Decimal results without binary noise: 365.256 × 24 = 8766.144, 6.3872 × 24 = 153.2928.
        assert_eq!(
            with(|r| r.rotation = Some("sync")).sidereal_rotation_h,
            Some(8766.144)
        );
        assert_eq!(
            with(|r| r.rotation = Some("6.3872 d")).sidereal_rotation_h,
            Some(153.2928)
        );
        assert_eq!(with(|r| r.mass = "1,988,400. e24").mass_kg, 1.9884e30);
    }

    #[test]
    fn refuses_numbers_that_are_not_on_their_sheet() {
        let all = sheets(EARTH_SHEET, "| 15 |");
        let fails = |edit: fn(&mut Curated)| {
            let mut row = sample();
            edit(&mut row);
            body_facts(&row, &all).is_err()
        };
        assert!(fails(|r| r.mass = "5.9724 e24"), "a digit slip");
        assert!(
            fails(|r| r.temp = Some("table:16 C")),
            "not on the fact table"
        );
        assert!(
            fails(|r| r.temp = Some("small:15 C")),
            "on the fact table, not the small one"
        );
        assert!(fails(|r| r.radius = "208 x 197 x 192"), "one axis wrong");
        assert!(
            fails(|r| r.atmosphere = Some("N2 78.09%")),
            "a number in the atmosphere"
        );
        assert!(
            fails(|r| r.mass = "5.9722"),
            "a mass needs its power of ten"
        );
        assert!(fails(|r| r.pressure = Some("1014 psi")), "unknown unit");
        assert!(
            fails(|r| r.gravity = Some("9.820 mb")),
            "a unit where none belongs"
        );
        assert!(fails(|r| r.temp = Some("nowhere:15 C")), "unknown sheet");
        assert!(fails(|r| r.sheet = "marsfact.html"), "sheet not downloaded");
        assert!(
            !fails(|r| r.rotation = Some("sync")),
            "sync needs only the period"
        );
        assert!(fails(|r| {
            r.rotation = Some("sync");
            r.period = None;
        }));
    }

    #[test]
    fn curated_table_is_complete() {
        let rows = curated();
        assert_eq!(rows.len(), 32, "Sun, 8 planets, Pluto, 22 moons");
        let ids: HashSet<i32> = rows.iter().map(|r| r.naif).collect();
        let names: HashSet<&str> = rows.iter().map(|r| r.name).collect();
        assert_eq!(
            (ids.len(), names.len()),
            (rows.len(), rows.len()),
            "duplicates"
        );
        for row in &rows {
            assert!(uni_data::facts::KINDS.contains(&row.kind), "{}", row.name);
            for moon in row.moons {
                assert!(
                    rows.iter().any(|r| r.name == *moon && r.kind == "moon"),
                    "{}: {moon}",
                    row.name
                );
            }
            if row.kind == "planet" {
                assert!(
                    row.geometric.is_some(),
                    "{}: geometric albedo (A3, A8)",
                    row.name
                );
            }
        }
        for body in uni_physics::bodies::major_bodies() {
            assert!(ids.contains(&body.naif), "{} has no facts", body.name);
        }
    }

    #[test]
    fn parses_options() {
        let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
        let options = parse_options(&args(&["--out", "f.ron", "--sheets", "s"])).expect("valid");
        assert_eq!(
            (options.out, options.sheets),
            (PathBuf::from("f.ron"), PathBuf::from("s"))
        );
        let options = parse_options(&args(&["--out", "f.ron"])).expect("sheets optional");
        assert!(
            options.sheets.ends_with("facts"),
            "{}",
            options.sheets.display()
        );
        assert!(parse_options(&args(&[])).is_err());
        assert!(parse_options(&args(&["--sheets", "s"])).is_err());
    }
}
