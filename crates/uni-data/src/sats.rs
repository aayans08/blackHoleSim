//! Planetary-satellite mean elements and physical parameters from JPL SSD
//! (https://ssd.jpl.nasa.gov/sats/elem/, /sats/phys_par/) as `sat_elements.ron` (M1 Task A9),
//! baked by `uni-bake sats`. JPL warns that these mean elements are "not intended for ephemeris
//! computation": they describe a precessing ellipse fitted to each numerically integrated orbit.
//!
//! Format: RON, a `//` provenance header, then one sequence of [`SatRecord`], one moon per line.
//!
//! JPL refers each moon's elements to one plane, named in its "Frame" column: the moon's local
//! Laplace plane, the J2000 ecliptic, or the planet's equator. `laplace_ra_deg`/`laplace_dec_deg`
//! hold that plane's pole for every frame, and JPL measures each node from that plane's ascending
//! node on the ICRF equator, so every record reads the same way: the orbit is inclined `i_deg` to
//! the plane with that pole. Precession periods JPL does not give are `f64::INFINITY` (no
//! precession; RON writes `inf`).

use std::path::Path;

use crate::DataError;
use crate::smallbody::read_ron;

/// One moon's mean elements on its reference plane, in the source's units.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SatRecord {
    /// NAIF id (JPL's "Code").
    pub naif: i32,
    /// IAU name or designation, as JPL spells it ("Moon", "S2009_S_2").
    pub name: String,
    /// NAIF id of the planet (body centre) it orbits: 399, 499, …, 899, or 999 for Pluto.
    pub planet_naif: i32,
    /// Semi-major axis [km].
    pub a_km: f64,
    pub e: f64,
    /// Argument of periapsis [deg].
    pub argp_deg: f64,
    /// Mean anomaly at `epoch_jd_tdb` [deg].
    pub m_deg: f64,
    /// Inclination to the reference plane [deg].
    pub i_deg: f64,
    /// Ascending node, from the reference plane's node on the ICRF equator [deg].
    pub node_deg: f64,
    /// Mean motion [deg/day]: 360° over JPL's sidereal period P.
    pub n_deg_per_day: f64,
    /// Period of the argument of periapsis's precession [yr]; `f64::INFINITY` when not given.
    pub p_apsides_yr: f64,
    /// Period of the node's precession [yr]; `f64::INFINITY` when not given.
    pub p_node_yr: f64,
    /// Epoch of the mean elements, Julian Date (TDB).
    pub epoch_jd_tdb: f64,
    /// Reference-plane pole, ICRF right ascension [deg] (see the module doc for each frame).
    pub laplace_ra_deg: f64,
    /// Reference-plane pole, ICRF declination [deg].
    pub laplace_dec_deg: f64,
    /// Mean radius [km] (radius of the sphere of equal volume), from `/sats/phys_par/`.
    pub radius_km: Option<f64>,
    /// GM [km³/s²], from `/sats/phys_par/`; `None` where JPL gives only an upper limit.
    pub gm_km3_s2: Option<f64>,
}

/// NAIF ids of the bodies that JPL's table lists moons for (Earth … Neptune, then Pluto).
const PLANETS: [i32; 7] = [399, 499, 599, 699, 799, 899, 999];

/// Reads `sat_elements.ron`.
///
/// A missing file is [`DataError::NotFound`]. A file that is not a RON sequence of [`SatRecord`],
/// or that repeats a NAIF id or holds an impossible record (unknown planet, a ≤ 0, e outside
/// [0, 1), i outside [0°, 180°], n ≤ 0, a period ≤ 0 or NaN, a pole outside RA [0°, 360°) or
/// Dec [−90°, 90°], a non-finite number, a radius or GM ≤ 0), is [`DataError::BadFormat`].
pub fn load_sat_elements(path: &Path) -> Result<Vec<SatRecord>, DataError> {
    let sats: Vec<SatRecord> = read_ron(path)?;
    let mut seen = std::collections::HashSet::new();
    for (index, sat) in sats.iter().enumerate() {
        let why = if !seen.insert(sat.naif) {
            Some("NAIF id repeats an earlier moon")
        } else {
            sat_problem(sat)
        };
        if let Some(why) = why {
            return Err(DataError::BadFormat(format!(
                "{}: moon {index} ({} {:?}): {why}",
                path.display(),
                sat.naif,
                sat.name
            )));
        }
    }
    Ok(sats)
}

fn sat_problem(s: &SatRecord) -> Option<&'static str> {
    let finite = [
        s.argp_deg,
        s.m_deg,
        s.node_deg,
        s.epoch_jd_tdb,
        s.laplace_ra_deg,
    ];
    let period_ok = |p: f64| p > 0.0; // false for NaN; +inf means "no precession"
    let positive = |v: Option<f64>| v.is_none_or(|v| v.is_finite() && v > 0.0);
    if s.name.trim().is_empty() {
        Some("empty name")
    } else if !PLANETS.contains(&s.planet_naif) {
        Some("planet_naif is not a planet JPL lists moons for")
    } else if !(s.a_km.is_finite() && s.a_km > 0.0) {
        Some("semi-major axis is not > 0")
    } else if !(s.e.is_finite() && (0.0..1.0).contains(&s.e)) {
        Some("eccentricity outside [0, 1)")
    } else if !finite.iter().all(|v| v.is_finite()) {
        Some("non-finite angle, epoch or pole")
    } else if !(0.0..=180.0).contains(&s.i_deg) {
        Some("inclination outside [0, 180] deg")
    } else if !(s.n_deg_per_day.is_finite() && s.n_deg_per_day > 0.0) {
        Some("mean motion is not > 0")
    } else if !(period_ok(s.p_apsides_yr) && period_ok(s.p_node_yr)) {
        Some("precession period is not > 0 (infinity for none)")
    } else if !(0.0..360.0).contains(&s.laplace_ra_deg) {
        Some("pole RA outside [0, 360) deg")
    } else if !(-90.0..=90.0).contains(&s.laplace_dec_deg) {
        Some("pole Dec outside [-90, 90] deg")
    } else if !(positive(s.radius_km) && positive(s.gm_km3_s2)) {
        Some("radius or GM is not > 0")
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

    /// The Moon (ecliptic frame) and Charon (precession periods not given: infinite).
    fn sample() -> Vec<SatRecord> {
        vec![
            SatRecord {
                naif: 301,
                name: "Moon".to_owned(),
                planet_naif: 399,
                a_km: 384_400.0,
                e: 0.0554,
                argp_deg: 318.15,
                m_deg: 135.27,
                i_deg: 5.16,
                node_deg: 125.08,
                n_deg_per_day: 360.0 / 27.322,
                p_apsides_yr: 5.997,
                p_node_yr: 18.6,
                epoch_jd_tdb: 2_451_545.0,
                laplace_ra_deg: 270.0,
                laplace_dec_deg: 66.560_708_888_888_89,
                radius_km: Some(1737.4),
                gm_km3_s2: Some(4902.8),
            },
            SatRecord {
                naif: 901,
                name: "Charon".to_owned(),
                planet_naif: 999,
                a_km: 19_600.0,
                e: 0.0,
                argp_deg: 0.0,
                m_deg: 304.1,
                i_deg: 0.0,
                node_deg: 0.0,
                n_deg_per_day: 360.0 / 6.387_222,
                p_apsides_yr: f64::INFINITY,
                p_node_yr: f64::INFINITY,
                epoch_jd_tdb: 2_451_545.0,
                laplace_ra_deg: 132.993,
                laplace_dec_deg: -6.163,
                radius_km: None,
                gm_km3_s2: None,
            },
        ]
    }

    fn bits(s: &SatRecord) -> (i32, String, i32, Vec<u64>, [Option<u64>; 2]) {
        let numbers = [
            s.a_km,
            s.e,
            s.argp_deg,
            s.m_deg,
            s.i_deg,
            s.node_deg,
            s.n_deg_per_day,
            s.p_apsides_yr,
            s.p_node_yr,
            s.epoch_jd_tdb,
            s.laplace_ra_deg,
            s.laplace_dec_deg,
        ];
        (
            s.naif,
            s.name.clone(),
            s.planet_naif,
            numbers.iter().map(|v| v.to_bits()).collect(),
            [s.radius_km.map(f64::to_bits), s.gm_km3_s2.map(f64::to_bits)],
        )
    }

    #[test]
    fn sat_ron_roundtrip() {
        let sats = sample();
        let path = temp_path("sats.ron");
        let config = ron::ser::PrettyConfig::new().compact_structs(true);
        let body = ron::ser::to_string_pretty(&sats, config).expect("serialise");
        std::fs::write(&path, format!("// header\n{body}\n")).expect("write");
        let loaded = load_sat_elements(&path);
        let _ = std::fs::remove_file(&path);
        let loaded = loaded.expect("load");
        let all = |list: &[SatRecord]| list.iter().map(bits).collect::<Vec<_>>();
        assert_eq!(all(&loaded), all(&sats));
    }

    #[test]
    fn sat_loader_rejects_bad_values() {
        let path = temp_path("bad-sats.ron");
        let check = |what: &str, edit: fn(&mut SatRecord)| {
            let mut sats = sample();
            edit(&mut sats[0]);
            std::fs::write(&path, ron::ser::to_string(&sats).expect("serialise")).expect("write");
            let result = load_sat_elements(&path);
            assert!(
                matches!(result, Err(DataError::BadFormat(_))),
                "{what}: {result:?}"
            );
        };
        check("a <= 0", |s| s.a_km = 0.0);
        check("e >= 1", |s| s.e = 1.0);
        check("NaN M", |s| s.m_deg = f64::NAN);
        check("n <= 0", |s| s.n_deg_per_day = 0.0);
        check("period <= 0", |s| s.p_node_yr = 0.0);
        check("NaN period", |s| s.p_apsides_yr = f64::NAN);
        check("pole Dec > 90", |s| s.laplace_dec_deg = 90.5);
        check("pole RA >= 360", |s| s.laplace_ra_deg = 360.0);
        check("radius <= 0", |s| s.radius_km = Some(-1.0));
        check("unknown planet", |s| s.planet_naif = 10);
        check("duplicate moon", |s| s.naif = 901);
        let _ = std::fs::remove_file(&path);
        let missing = load_sat_elements(Path::new("/nonexistent/sats.ron"));
        assert!(
            matches!(missing, Err(DataError::NotFound(_))),
            "{missing:?}"
        );
    }

    fn baked() -> Vec<SatRecord> {
        let path = crate::data_dir().join("baked").join("sat_elements.ron");
        load_sat_elements(&path).unwrap_or_else(|err| {
            panic!("baked satellites missing or unreadable: {err} (A9 Step 3)")
        })
    }

    /// Every moon of `uni_physics::bodies` is in the baked file once, around the right planet,
    /// and its JPL GM (within 1 %) and mean radius (within 1 %) agree with the body table (JPL GMs;
    /// WGCCRE 2015 triaxial radii, compared as the volumetric mean (a·b·c)^⅓).
    #[test]
    fn sat_elements_cover_moons() {
        let sats = baked();
        assert!(sats.len() >= 400, "{} moons", sats.len());
        let mut ids: Vec<i32> = sats.iter().map(|s| s.naif).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), sats.len(), "duplicate moon ids");
        let moons = uni_physics::bodies::major_bodies()
            .iter()
            .filter(|b| b.parent.is_some_and(|p| p != 10));
        for body in moons {
            let found: Vec<_> = sats.iter().filter(|s| s.naif == body.naif).collect();
            assert_eq!(found.len(), 1, "{} ({})", body.name, body.naif);
            let sat = found[0];
            assert_eq!(Some(sat.planet_naif), body.parent, "{}", body.name);
            let gm = sat
                .gm_km3_s2
                .unwrap_or_else(|| panic!("{}: no GM", body.name));
            // 1 % catches a column mix-up (sigma or density read as GM) and admits the solutions'
            // spread: Charon's PLU060 GM, 106.1 ± 0.3 km³/s², is 0.2 % above the body table's.
            let gm_table = body.gm_m3s2 * 1e-9;
            assert!(
                ((gm - gm_table) / gm_table).abs() < 0.01,
                "{}: GM {gm} vs {gm_table} km³/s²",
                body.name
            );
            let r = body.radii_m * 1e-3;
            let mean = (r.x * r.y * r.z).cbrt();
            let radius = sat
                .radius_km
                .unwrap_or_else(|| panic!("{}: no radius", body.name));
            assert!(
                ((radius - mean) / mean).abs() < 0.01,
                "{}: radius {radius} vs {mean} km",
                body.name
            );
        }
    }

    /// The reference-plane pole of each frame: JPL's Laplace column (Titan), the J2000 ecliptic
    /// (the Moon), and the planet's equator (Ariel, whose pole must agree with JPL's Laplace pole
    /// for Uranus's inner moons, 77.3°/15.2°; Charon, Pluto's WGCCRE 2015 pole).
    #[test]
    fn sat_reference_poles() {
        let sats = baked();
        let pole = |naif: i32| {
            let s = sats
                .iter()
                .find(|s| s.naif == naif)
                .unwrap_or_else(|| panic!("{naif} missing"));
            (s.laplace_ra_deg, s.laplace_dec_deg)
        };
        let near = |(ra, dec): (f64, f64), (ra0, dec0): (f64, f64), tol: f64| {
            (ra - ra0).abs() <= tol && (dec - dec0).abs() <= tol
        };
        assert!(near(pole(606), (36.4, 84.0), 1e-9), "Titan {:?}", pole(606));
        assert!(
            near(pole(301), (270.0, 66.560_708_9), 1e-6),
            "Moon {:?}",
            pole(301)
        );
        assert!(near(pole(701), (77.3, 15.2), 0.1), "Ariel {:?}", pole(701));
        assert!(
            near(pole(901), (132.993, -6.163), 1e-9),
            "Charon {:?}",
            pole(901)
        );
        assert!(
            sats.iter().all(|s| s.laplace_dec_deg.abs() <= 90.0),
            "every moon has a pole"
        );
    }
}
