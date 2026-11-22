//! Body fact sheets (`body_facts.ron`), hand-curated from the NASA NSSDCA planetary and
//! satellite fact sheets (M1 Task A9) and written by `uni-bake facts`, which checks every
//! transcribed number against the sheet it came from.
//!
//! Format: RON, a `//` provenance header, then one sequence of [`BodyFacts`].
//!
//! Sequencing (A0b fix rounds 1–2, review I3): A9's `facts_cover_major_bodies` compares radii with
//! `uni_physics::bodies::major_bodies` (dev-dependency). The lead's A0b implemented that table
//! ahead of A1 (WGCCRE 2015 radii, DE440 GM), so the test runs for real while A1 works in
//! parallel. Its `radii_m` are triaxial (a, b, c); an NSSDCA mean radius compares with the
//! volumetric mean (a·b·c)^⅓ (Saturn: 58 232 km, not the 60 268 km equatorial radius).

use std::collections::HashSet;
use std::path::Path;

use crate::DataError;
use crate::smallbody::read_ron;

/// The info-card facts of one body. `None` where the sheets give no value (or none that applies,
/// such as a surface pressure for a giant planet).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct BodyFacts {
    /// NAIF id (10 Sun, 199–899 planets, 999 Pluto, 301 Moon, 401… moons).
    pub naif: i32,
    pub name: String,
    /// One of [`KINDS`]: "star", "planet", "dwarf planet", "moon".
    pub kind: String,
    /// Volumetric mean radius [km]: the sheet's, or (a·b·c)^⅓ where the sheet gives three axes.
    pub mean_radius_km: f64,
    pub mass_kg: f64,
    /// Mean surface gravity [m/s²] (giant planets: at the 1-bar level; the Sun: equatorial).
    pub surface_gravity_ms2: Option<f64>,
    pub escape_velocity_kms: Option<f64>,
    /// Sidereal rotation period [h]; negative for retrograde rotation. For a synchronous moon
    /// whose sheet says so without a number, its orbital period × 24.
    pub sidereal_rotation_h: Option<f64>,
    /// Sidereal orbital period [days], about the Sun or the planet.
    pub orbital_period_d: Option<f64>,
    /// Semi-major axis [au], about the Sun or the planet (a moon's from the sheet's km).
    pub semi_major_au: Option<f64>,
    /// Obliquity of the equator to the orbit [deg] (the Sun: to the ecliptic).
    pub axial_tilt_deg: Option<f64>,
    pub surface_pressure_kpa: Option<f64>,
    /// Main constituents as the sheet lists them: by volume, except the Sun's photosphere (by
    /// number of atoms); "Exosphere: …" for a surface-bounded exosphere.
    pub atmosphere: Option<String>,
    /// Mean temperature [K] from the NSSDCA fact tables (giant planets: at 1 bar; the Sun: its
    /// effective temperature).
    pub mean_temp_k: Option<f64>,
    pub bond_albedo: Option<f64>,
    /// V-band geometric albedo.
    pub geometric_albedo: Option<f64>,
    /// The moons of this body that have their own entry in the file.
    pub moons: Vec<String>,
    /// The fact-sheet URL of every value, separated by "; ".
    pub source: String,
}

/// The values of [`BodyFacts::kind`].
pub const KINDS: [&str; 4] = ["star", "planet", "dwarf planet", "moon"];

/// Reads `body_facts.ron`.
///
/// A missing file is [`DataError::NotFound`]. A file that is not a RON sequence of
/// [`BodyFacts`], or that repeats a NAIF id or name or holds an impossible entry (radius or mass
/// ≤ 0, a non-finite number, a negative albedo, an unknown kind, an empty name or source), is
/// [`DataError::BadFormat`].
pub fn load_body_facts(path: &Path) -> Result<Vec<BodyFacts>, DataError> {
    let facts: Vec<BodyFacts> = read_ron(path)?;
    let (mut ids, mut names) = (HashSet::new(), HashSet::new());
    for (index, fact) in facts.iter().enumerate() {
        let why = if !ids.insert(fact.naif) {
            Some("NAIF id repeats an earlier entry")
        } else if !names.insert(fact.name.as_str()) {
            Some("name repeats an earlier entry")
        } else {
            facts_problem(fact)
        };
        if let Some(why) = why {
            return Err(DataError::BadFormat(format!(
                "{}: entry {index} ({} {:?}): {why}",
                path.display(),
                fact.naif,
                fact.name
            )));
        }
    }
    Ok(facts)
}

fn facts_problem(f: &BodyFacts) -> Option<&'static str> {
    let optional = [
        f.surface_gravity_ms2,
        f.escape_velocity_kms,
        f.sidereal_rotation_h,
        f.orbital_period_d,
        f.semi_major_au,
        f.axial_tilt_deg,
        f.surface_pressure_kpa,
        f.mean_temp_k,
        f.bond_albedo,
        f.geometric_albedo,
    ];
    let albedos = [f.bond_albedo, f.geometric_albedo];
    if f.name.trim().is_empty() || f.source.trim().is_empty() {
        Some("empty name or source")
    } else if !KINDS.contains(&f.kind.as_str()) {
        Some("unknown kind")
    } else if !(f.mean_radius_km.is_finite() && f.mean_radius_km > 0.0) {
        Some("mean radius is not > 0")
    } else if !(f.mass_kg.is_finite() && f.mass_kg > 0.0) {
        Some("mass is not > 0")
    } else if !optional.into_iter().flatten().all(f64::is_finite) {
        Some("non-finite value")
    } else if albedos.into_iter().flatten().any(|a| a < 0.0) {
        Some("negative albedo")
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("uni-data-a9-{}-{name}", std::process::id()))
    }

    fn earth() -> BodyFacts {
        BodyFacts {
            naif: 399,
            name: "Earth".to_owned(),
            kind: "planet".to_owned(),
            mean_radius_km: 6371.0,
            mass_kg: 5.9722e24,
            surface_gravity_ms2: Some(9.82),
            escape_velocity_kms: Some(11.186),
            sidereal_rotation_h: Some(23.9345),
            orbital_period_d: Some(365.256),
            semi_major_au: Some(1.000_000_11),
            axial_tilt_deg: Some(23.44),
            surface_pressure_kpa: Some(101.4),
            atmosphere: Some("N2 78.08%, O2 20.95%".to_owned()),
            mean_temp_k: Some(288.15),
            bond_albedo: Some(0.294),
            geometric_albedo: Some(0.434),
            moons: vec!["Moon".to_owned()],
            source: "https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html".to_owned(),
        }
    }

    fn sun() -> BodyFacts {
        BodyFacts {
            naif: 10,
            name: "Sun".to_owned(),
            kind: "star".to_owned(),
            mean_radius_km: 695_700.0,
            mass_kg: 1.9884e30,
            surface_gravity_ms2: Some(274.0),
            escape_velocity_kms: Some(617.6),
            sidereal_rotation_h: Some(609.12),
            orbital_period_d: None,
            semi_major_au: None,
            axial_tilt_deg: Some(7.25),
            surface_pressure_kpa: None,
            atmosphere: None,
            mean_temp_k: Some(5772.0),
            bond_albedo: None,
            geometric_albedo: None,
            moons: Vec::new(),
            source: "https://nssdc.gsfc.nasa.gov/planetary/factsheet/sunfact.html".to_owned(),
        }
    }

    #[test]
    fn facts_ron_roundtrip() {
        let facts = vec![sun(), earth()];
        let path = temp_path("facts.ron");
        let body =
            ron::ser::to_string_pretty(&facts, ron::ser::PrettyConfig::new()).expect("serialise");
        std::fs::write(&path, format!("// header\n{body}\n")).expect("write");
        let loaded = load_body_facts(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        // RON prints the shortest representation that reads back to the same f64.
        assert_eq!(format!("{loaded:?}"), format!("{facts:?}"));
    }

    #[test]
    fn facts_loader_rejects_bad_values() {
        let path = temp_path("bad-facts.ron");
        let check = |what: &str, edit: fn(&mut BodyFacts)| {
            let mut facts = vec![sun(), earth()];
            edit(&mut facts[1]);
            std::fs::write(&path, ron::ser::to_string(&facts).expect("serialise")).expect("write");
            let result = load_body_facts(&path);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{what}: {result:?}"
            );
        };
        check("radius <= 0", |f| f.mean_radius_km = 0.0);
        check("NaN mass", |f| f.mass_kg = f64::NAN);
        check("NaN option", |f| f.mean_temp_k = Some(f64::NAN));
        check("negative albedo", |f| f.geometric_albedo = Some(-0.1));
        check("unknown kind", |f| f.kind = "comet".to_owned());
        check("no source", |f| f.source.clear());
        check("duplicate id", |f| f.naif = 10);
        check("duplicate name", |f| f.name = "Sun".to_owned());
        let _ = std::fs::remove_file(&path);
        let missing = load_body_facts(Path::new("/nonexistent/facts.ron"));
        assert!(
            matches!(missing, Err(DataError::NotFound(_))),
            "{missing:?}"
        );
    }

    fn baked() -> Vec<BodyFacts> {
        let path = crate::data_dir().join("baked").join("body_facts.ron");
        load_body_facts(&path)
            .unwrap_or_else(|err| panic!("baked facts missing or unreadable: {err} (A9 Step 3)"))
    }

    const PLANETS: [i32; 8] = [199, 299, 399, 499, 599, 699, 799, 899];

    /// Sun, 8 planets, Pluto, Moon, Galileans and Titan are present; every radius agrees with
    /// `uni_physics::bodies` (the volumetric mean of its WGCCRE 2015 radii) within 0.5 %; planets
    /// have a geometric albedo; every entry names its NSSDCA sheets.
    #[test]
    fn facts_cover_major_bodies() {
        let facts = baked();
        let required = [10, 999, 301, 501, 502, 503, 504, 606]
            .into_iter()
            .chain(PLANETS);
        for naif in required {
            assert_eq!(
                facts.iter().filter(|f| f.naif == naif).count(),
                1,
                "NAIF {naif} must appear exactly once"
            );
        }
        for body in uni_physics::bodies::major_bodies() {
            let Some(fact) = facts.iter().find(|f| f.naif == body.naif) else {
                continue;
            };
            assert_eq!(fact.name, body.name, "NAIF {}", body.naif);
            let r = body.radii_m * 1e-3;
            let mean = (r.x * r.y * r.z).cbrt();
            let error = (fact.mean_radius_km - mean).abs() / mean;
            assert!(
                error < 0.005,
                "{}: {} km vs {mean:.2} km",
                fact.name,
                fact.mean_radius_km
            );
            // Mass against GM / G (CODATA 2018 G = 6.674 30e-11 m³ kg⁻¹ s⁻²) within 10 %: this
            // catches a wrong power of ten, not the sheets' age. Some NSSDCA moon masses predate
            // the JPL solutions (`uni-bake facts` prints every gap above 1 %); Deimos's
            // 2.4e15 kg is 67 % above MAR097's GM / G = 1.44e15 kg, so it is exempt.
            let mass = body.gm_m3s2 / 6.674_30e-11;
            assert!(
                body.naif == 402 || (fact.mass_kg - mass).abs() / mass < 0.1,
                "{}: mass {:e} kg vs GM/G {mass:e} kg",
                fact.name,
                fact.mass_kg
            );
        }
        for naif in PLANETS {
            let planet = facts
                .iter()
                .find(|f| f.naif == naif)
                .expect("checked above");
            assert!(
                planet.geometric_albedo.is_some_and(|p| p > 0.0),
                "{}: geometric albedo",
                planet.name
            );
            assert_eq!(planet.kind, "planet");
        }
        for fact in &facts {
            assert!(
                fact.source
                    .split("; ")
                    .all(|url| url.starts_with("https://nssdc.gsfc.nasa.gov/planetary/factsheet/")),
                "{}: source {:?}",
                fact.name,
                fact.source
            );
            for moon in &fact.moons {
                assert!(
                    facts.iter().any(|f| &f.name == moon && f.kind == "moon"),
                    "{}: moon {moon} has no entry",
                    fact.name
                );
            }
        }
    }

    /// The UI fonts (A9 Step 4) are TrueType files and both SIL OFL 1.1 texts are beside them.
    /// It lives here because uni-data has no font module; the fonts serve the info cards.
    #[test]
    fn fonts_present_and_licensed() {
        let dir = crate::data_dir().join("fonts");
        let fonts = [
            "Inter-Light",
            "Inter-Regular",
            "Inter-Medium",
            "Inter-SemiBold",
            "IBMPlexMono-Light",
            "IBMPlexMono-Regular",
            "IBMPlexMono-Medium",
        ];
        for font in fonts {
            let path = dir.join(format!("{font}.ttf"));
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|err| panic!("{}: {err} (A9 Step 4)", path.display()));
            // sfnt version: 0x00010000 (TrueType outlines) or "true" (Apple TrueType).
            let magic = bytes.get(..4);
            assert!(
                magic == Some(&[0, 1, 0, 0][..]) || magic == Some(&b"true"[..]),
                "{font}: not a TrueType file ({magic:?})"
            );
            assert!(bytes.len() > 50_000, "{font}: only {} B", bytes.len());
        }
        for licence in ["OFL-Inter.txt", "OFL-IBMPlexMono.txt"] {
            let path = dir.join(licence);
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{}: {err} (A9 Step 4)", path.display()));
            assert!(text.contains("SIL OPEN FONT LICENSE"), "{licence}");
        }
    }
}
